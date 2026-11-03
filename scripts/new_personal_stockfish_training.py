"""Explicitly snapshot personal PGNs and prepare Stockfish-only training targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from chess_ai.arena.gameplay_selection import GameplaySelectionConfig
from chess_ai.config import load_config
from chess_ai.data.personal_pgn import snapshot_personal_pgns
from chess_ai.data.stockfish_rebuild import RebuildConfig
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig


def prepare_personal_training(
    source_dir: Path, username: str, run_id: str, *, confirm_training_import: bool = False
) -> Path:
    if not confirm_training_import:
        raise PermissionError("Explicit --confirm-training-import is required for personal PGNs")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id) is None:
        raise ValueError("Use a safe 1-81 character run ID")
    root = Path("data/experiments") / run_id
    checkpoint_dir = Path("checkpoints") / run_id
    metrics = Path("data/metrics") / f"{run_id}.jsonl"
    evaluation = Path("data/games/evaluation") / f"{run_id}_puct_selection"
    if any(path.exists() for path in (root, checkpoint_dir, metrics, evaluation)):
        raise ValueError("Run ID exists; use its saved configs or choose another ID")
    champion = load_gameplay_champion()
    if champion is None:
        raise ValueError("Personal fine-tuning requires a registered champion")
    champion_record = json.loads(Path("checkpoints/champion.json").read_text())
    previous_root = (
        Path("data/experiments") / Path(champion_record["source_checkpoint"]).parent.name
    )
    previous = json.loads((previous_root / "experiment.json").read_text())
    teacher = load_config(previous["relabel_config"])["stockfish_rebuild"]["stockfish"]
    replay = previous_root / "relabel/dataset.json"
    replay_metadata = json.loads(replay.read_text())
    loaded = load_checkpoint(champion.checkpoint, map_location="cpu")
    if replay_metadata["value_target_spec"] != loaded.metadata["value_target_spec"]:
        raise ValueError("Historical replay semantics differ from the champion")
    # Evaluation reports contribute exclusion starts only, never training positions.
    historical_paths = {Path(item["path"]) for item in champion_record["evidence"]}
    historical_paths.update(
        Path("data/games/evaluation").glob("stockfish_refinement_*_puct_selection/report.json")
    )
    historical_paths.update(
        Path("data/games/evaluation").glob("stockfish_refinement_*_puct_confirmation_*/report.json")
    )
    expected = {Path(item["path"]): item["sha256"] for item in champion_record["evidence"]}
    starts: set[str] = set()
    evidence = []
    for path in sorted(historical_paths):
        digest = sha256_file(path)
        if path in expected and digest != expected[path]:
            raise ValueError(f"Champion evidence changed: {path}")
        raw = json.loads(path.read_text())
        starts.update(item["fen"] for item in raw["openings"])
        evidence.append({"path": str(path), "sha256": digest})
    snapshots = root / "source_games"
    imported = snapshot_personal_pgns(source_dir, username, snapshots)
    exclusions = root / "historical_evaluation_openings.json"
    atomic_json(exclusions, {"pairs": [{"opening_fen": fen} for fen in sorted(starts)]})
    rebuild = {
        "run_dir": str(root / "relabel"),
        "datasets": [],
        "pgn_roots": [str(snapshots)],
        "stockfish": teacher,
        "seed": 18102036,
        "validation_fraction": 0.1,
        "shard_size": 2048,
        "pilot_positions": 128,
    }
    config = RebuildConfig.from_mapping(rebuild)
    if config.value_spec != loaded.metadata["value_target_spec"]:
        raise ValueError("Stockfish target settings differ from the champion")
    train: dict[str, Any] = {
        "seed": 18102036,
        "device": "cuda",
        "model": loaded.model.config.to_dict(),
        "training": {
            "dataset_path": str(root / "replay.pt"),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_path": str(metrics),
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.00001,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "scheduler": True,
            "legal_policy_mask": True,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 1.0,
            "selection_metric": "validation_loss",
            "checkpoint_every": 1,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0001,
            "baseline_eligible": True,
            "game_balanced_sampling": False,
            "use_dataset_split": True,
            "value_target_spec": config.value_spec,
            "validation_source_key": "composition_source",
            "log_every": 100,
        },
    }
    TrainingConfig.from_mapping(train["training"])
    selection = {
        "champion_checkpoint": str(champion.checkpoint),
        "candidate_checkpoint_dir": str(checkpoint_dir),
        "training_manifest": str(config.run_dir / "training_manifest.json"),
        "additional_training_manifests": [
            str(previous_root / "relabel/training_manifest.json"),
            str(exclusions),
        ],
        "pgn_dir": str(evaluation),
        "report_path": str(evaluation / "report.json"),
        "openings": 16,
        "opponent_depths": [1, 2],
        "seed": 19102036,
        "opening_min_full_moves": 2,
        "opening_max_full_moves": 5,
        "max_plies": 200,
        "minimum_aggregate_improvement_points": 1.0,
        "search_simulations": 32,
        "c_puct": champion.c_puct,
        "opening_book": None,
        "device": "cuda",
    }
    GameplaySelectionConfig.from_mapping(selection)
    configs = root / "configs"
    configs.mkdir()
    payloads = {
        "relabel": {"stockfish_rebuild": rebuild},
        "train": train,
        "compose": {
            "refinement_replay": {
                "fresh_path": str(config.run_dir / "dataset.json"),
                "replay_manifest": str(replay),
                "output_path": str(root / "replay.pt"),
                "seed": 18102036,
            }
        },
        "select_puct": {"gameplay_selection": selection},
    }
    for stage, payload in payloads.items():
        (configs / f"{stage}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    atomic_json(
        root / "experiment.json",
        {
            "format": "self-improving-chess-ai.personal-stockfish-experiment",
            "version": 1,
            "username": username,
            "champion_checkpoint": str(champion.checkpoint),
            "champion_sha256": sha256_file(champion.checkpoint),
            "personal_import": str(root / "personal_import.json"),
            "personal_import_sha256": sha256_file(root / "personal_import.json"),
            "replay_manifest": str(replay),
            "replay_manifest_sha256": sha256_file(replay),
            "counts": imported["counts"],
            "historical_evidence": evidence,
            "training_updates": 0,
            "promotion_performed": False,
            "policy_target": "Stockfish MultiPV preferences, never the human played move",
            "value_target_spec": config.value_spec,
        },
    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirm-training-import", action="store_true")
    args = parser.parse_args()
    print(
        prepare_personal_training(
            args.source_dir,
            args.username,
            args.run_id,
            confirm_training_import=args.confirm_training_import,
        )
    )


if __name__ == "__main__":
    main()
