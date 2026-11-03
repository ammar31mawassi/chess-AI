"""Reserve 200 PUCT collection games plus four standard-start book games."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from chess_ai.arena.gameplay_selection import GameplaySelectionConfig
from chess_ai.config import load_config
from chess_ai.curriculum.stockfish_batch import StockfishBatchConfig, _game_specs
from chess_ai.data.opening_book import load_opening_book
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig


def prepare_refinement(run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id) is None:
        raise ValueError("Use a safe 1-81 character run ID")
    name = f"stockfish_refinement_{run_id}"
    root = Path("data/experiments") / name
    checkpoints = Path("checkpoints") / name
    metrics = Path("data/metrics") / f"{name}.jsonl"
    evaluation = Path("data/games/evaluation") / f"{name}_puct_selection"
    if any(path.exists() for path in (root, checkpoints, metrics, evaluation)):
        raise ValueError("Run ID exists; reuse its saved configs or choose a new ID")
    champion = load_gameplay_champion()
    if champion is None or champion.search_simulations != 32:
        raise ValueError("Expected a promoted PUCT-32 champion")
    record = json.loads(Path("checkpoints/champion.json").read_text())
    previous_root = Path("data/experiments") / Path(record["source_checkpoint"]).parent.name
    previous_experiment = json.loads((previous_root / "experiment.json").read_text())
    teacher = load_config(previous_experiment["relabel_config"])["stockfish_rebuild"]["stockfish"]
    replay = previous_root / "relabel/dataset.json"
    replay_record = json.loads(replay.read_text())
    checkpoint = load_checkpoint(champion.checkpoint)
    if replay_record["value_target_spec"] != checkpoint.metadata["value_target_spec"]:
        raise ValueError("Replay semantics differ from the champion")
    book = Path(previous_experiment["opening_book"])
    load_opening_book(book)
    excluded: set[str] = set()
    evidence = []
    for entry in record["evidence"]:
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Champion evaluation evidence changed: {path}")
        report = json.loads(path.read_text())
        excluded.update(item["fen"] for item in report["openings"])
        evidence.append(entry)
    exclusion_path = root / "historical_evaluation_openings.json"
    # Write the immutable exclusion list before schedule validation.
    root.mkdir(parents=True)
    atomic_json(exclusion_path, {"pairs": [{"opening_fen": fen} for fen in sorted(excluded)]})
    collection = {
        "collection_version": 2,
        "run_dir": str(root / "collection"),
        "neural_checkpoint": str(champion.checkpoint),
        "stockfish": teacher,
        "opening_positions": 50,
        "opening_min_full_moves": 2,
        "opening_max_full_moves": 5,
        "normal_start_every_positions": 0,
        "max_plies": 200,
        "search_simulations": 32,
        "c_puct": champion.c_puct,
        "extra_standard_book_games": True,
        "opening_book": str(book),
        "exclude_opening_manifests": [str(exclusion_path)],
        "validation_fraction": 0.1,
        "win_weight": 1,
        "draw_weight": 1,
        "loss_weight": 1,
        "seed": 27102036,
        "device": "cuda",
    }
    config = StockfishBatchConfig.from_mapping(collection)
    schedule = _game_specs(config)
    if len(schedule) != 204 or sum(item["book_enabled"] for item in schedule) != 4:
        raise ValueError("Expected exactly 200 varied games and four standard book games")
    train: dict[str, Any] = {
        "seed": 28102036,
        "device": "cuda",
        "model": checkpoint.model.config.to_dict(),
        "training": {
            "dataset_path": str(root / "replay.pt"),
            "checkpoint_dir": str(checkpoints),
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
        "candidate_checkpoint_dir": str(checkpoints),
        "training_manifest": str(config.run_dir / "manifest.json"),
        "additional_training_manifests": [
            str(previous_root / "relabel/training_manifest.json"),
            str(exclusion_path),
        ],
        "pgn_dir": str(evaluation),
        "report_path": str(evaluation / "report.json"),
        "openings": 16,
        "opponent_depths": [1, 2],
        "seed": 29102036,
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
    for stage, payload in {
        "collect": {"stockfish_batch": collection},
        "train": train,
        "compose": {
            "refinement_replay": {
                "fresh_path": str(config.run_dir / "dataset.pt"),
                "replay_manifest": str(replay),
                "output_path": str(root / "replay.pt"),
                "seed": 28102036,
            }
        },
        "select_puct": {"gameplay_selection": selection},
    }.items():
        (configs / f"{stage}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    atomic_json(
        root / "experiment.json",
        {
            "format": "self-improving-chess-ai.stockfish-refinement-experiment",
            "version": 1,
            "champion_checkpoint": str(champion.checkpoint),
            "champion_sha256": sha256_file(champion.checkpoint),
            "opening_book": str(book),
            "opening_book_sha256": sha256_file(book),
            "replay_manifest": str(replay),
            "replay_manifest_sha256": sha256_file(replay),
            "historical_evidence": evidence,
            "games": 204,
            "standard_book_games": 4,
            "labeling": "every_recorded_move_canonical_fen_and_forced_root_when_needed",
            "training_updates": 0,
            "promotion_performed": False,
        },
    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(prepare_refinement(args.run_id))


if __name__ == "__main__":
    main()
