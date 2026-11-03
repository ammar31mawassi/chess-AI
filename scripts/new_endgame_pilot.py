"""Reserve an isolated endgame pilot; no labeling, training, or promotion."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from chess_ai.config import load_config
from chess_ai.data.stockfish_rebuild import _blocked
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig


def reserve(run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id) is None:
        raise ValueError("Use a safe run ID")
    root = Path("data/experiments") / run_id
    checkpoints = Path("checkpoints") / run_id
    metrics = Path("data/metrics") / f"{run_id}.jsonl"
    if any(p.exists() for p in (root, checkpoints, metrics)):
        raise ValueError("Run ID already exists; use saved configs or another ID")
    champion = load_gameplay_champion()
    if champion is None:
        raise ValueError("A registered champion is required")
    original = Path("data/experiments/stockfish_rebuild_eval_v2_20261011_01")
    personal = Path("data/experiments/chesscom_xxkiller31_20261018_01")
    teacher = dict(load_config(original / "configs/relabel.yaml")["stockfish_rebuild"]["stockfish"])
    teacher.update(
        nodes=200000,
        multipv=5,
        cache_dir=str(root / "teacher_cache"),
        mate_distance_temperature=2.0,
    )
    catalogs = [original / "relabel/catalog.sqlite", personal / "relabel/catalog.sqlite"]
    protection = [
        original / "relabel/dataset.json",
        personal / "relabel/dataset.json",
        Path("data/experiments/stockfish_refinement_v2_20261015_01/collection/dataset.pt"),
    ]
    if any(not p.is_file() for p in [*catalogs, *protection]):
        raise ValueError("Expected completed source/protection artifacts are missing")
    # Every path marked evaluation/gate/holdout/audit is an exclusion source.
    dirs = sorted(
        {p.parent for p in Path("data/games").rglob("*.pgn") if _blocked(p)},
        key=lambda p: (len(p.parts), str(p)),
    )
    roots: list[Path] = []
    for path in dirs:
        if not any(path.is_relative_to(parent) for parent in roots):
            roots.append(path)
    config = {
        "run_dir": str(root),
        "seed": 23102036,
        "champion": str(champion.checkpoint),
        "champion_registry": "checkpoints/champion.json",
        "champion_sha256": sha256_file(champion.checkpoint),
        "stockfish": teacher,
        "source_catalogs": [str(p) for p in catalogs],
        "protected_datasets": [str(p) for p in protection],
        "evaluation_roots": [str(p) for p in roots],
        "max_pieces": 10,
        "roots_per_material": 40,
        "natural_per_bucket": 48,
        "final_per_bucket": 24,
    }
    loaded = load_checkpoint(champion.checkpoint, map_location="cpu")
    spec = {"kind": "stockfish_cp_tanh_v1", "perspective": "side_to_move", "scale_cp": 600.0}
    if loaded.metadata.get("value_target_spec") != spec:
        raise ValueError("Pilot requires the registered cp/600 evaluation champion")
    train: dict[str, Any] = {
        "seed": config["seed"],
        "device": "cuda",
        "model": loaded.model.config.to_dict(),
        "training": {
            "dataset_path": str(root / "replay.pt"),
            "checkpoint_dir": str(checkpoints),
            "metrics_path": str(metrics),
            "batch_size": 128,
            "epochs": 8,
            "learning_rate": 0.00001,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "scheduler": True,
            "legal_policy_mask": True,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 1.0,
            "selection_metric": "validation_loss",
            "checkpoint_every": 4,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0001,
            "baseline_eligible": True,
            "game_balanced_sampling": False,
            "use_dataset_split": True,
            "value_target_spec": spec,
            "validation_source_key": "composition_source",
            "log_every": 20,
        },
    }
    TrainingConfig.from_mapping(train["training"])
    configs = root / "configs"
    configs.mkdir(parents=True)
    payloads = {
        "pilot": {"endgame_pilot": config},
        "train": train,
        "compose": {
            "refinement_replay": {
                "fresh_path": str(root / "fresh.pt"),
                "replay_manifest": str(original / "relabel/dataset.json"),
                "output_path": str(root / "replay.pt"),
                "seed": config["seed"],
            }
        },
    }
    for name, payload in payloads.items():
        (configs / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    atomic_json(
        root / "experiment.json",
        {
            "format": "self-improving-chess-ai.endgame-pilot-experiment",
            "version": 1,
            "champion": config["champion"],
            "champion_sha256": config["champion_sha256"],
            "registry_sha256": sha256_file(Path("checkpoints/champion.json")),
            "training_updates": 0,
            "promotion_performed": False,
            "config_sha256": {p.name: sha256_file(p) for p in configs.glob("*.yaml")},
            "selection": "Compare baseline and best.pt only on development; freeze one candidate before confirmation.",
            "confirmation": "One candidate, same unassisted PUCT32 budget, fresh endgame roots and fresh whole-game seed 27102036.",
        },
    )
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    print(reserve(parser.parse_args().run_id))
