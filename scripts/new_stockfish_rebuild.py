"""Reserve a frozen-source Stockfish relabeling experiment; never train here."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from chess_ai.arena.gameplay_selection import GameplaySelectionConfig
from chess_ai.data.opening_book import load_opening_book
from chess_ai.data.stockfish_rebuild import RebuildConfig
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.training.trainer import TrainingConfig


def prepare_rebuild(run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id) is None:
        raise ValueError("Use a safe 1-81 character run ID")
    name = f"stockfish_rebuild_{run_id}"
    root = Path("data/experiments") / name
    checkpoint_dir = Path("checkpoints") / name
    metrics_path = Path("data/metrics") / f"{name}.jsonl"
    if any(p.exists() for p in (root, checkpoint_dir, metrics_path)):
        raise ValueError("Rebuild run ID already exists; use its saved configs or choose a new ID")
    champion = Path("checkpoints/joint_policy_value_v3/epoch_0120.pt")
    architecture = load_checkpoint(champion).model.config.to_dict()
    book = Path("data/opening_books/all_chess_openings_v1.json")
    load_opening_book(book)
    sources = sorted(Path("data/datasets").glob("*.pt"))
    for curriculum in sorted(Path("data/curricula").iterdir()):
        if curriculum.is_dir() and "smoke" not in curriculum.name:
            sources.extend(sorted(p for p in curriculum.rglob("*.pt") if "smoke" not in p.parts))
    sources = sorted(set(sources))
    pgn_roots = [
        Path("data/games") / n
        for n in (
            "gen1_vs_d1_d2_teacher_500",
            "gen2_mixed_d1_d2_400",
            "minimax_d1_d2_d3_3000",
            "minimax_d2_diverse_500",
            "minimax_d2_vs_random_teacher_500",
            "d1_opponent_d2_teacher_100",
            "d1_gated_from_epoch399/training",
            "human_gui",
        )
        if (Path("data/games") / n).is_dir()
    ]
    # The signed Stockfish collector stores games inside its collection directory.
    for curriculum in sorted(Path("data/curricula").iterdir()):
        collection = curriculum / "collection" / "games"
        if "smoke" not in curriculum.name and collection.is_dir():
            pgn_roots.append(collection)
    rebuild: dict[str, Any] = {
        "run_dir": str(root / "relabel"),
        "datasets": [str(p) for p in sources],
        "pgn_roots": [str(p) for p in pgn_roots],
        "seed": 21102036,
        "validation_fraction": 0.1,
        "shard_size": 2048,
        "pilot_positions": 128,
        "stockfish": {
            "executable": "references/github/stockfish-build/49ea5ded38315cff8e67f4a677a9e7811612fbf6/stockfish.exe",
            "source_commit": "49ea5ded38315cff8e67f4a677a9e7811612fbf6",
            "nodes": 50000,
            "depth": None,
            "multipv": 5,
            "threads": 1,
            "hash_mb": 128,
            "timeout_seconds": 120.0,
            "temperature_cp": 100.0,
            "evaluation_scale_cp": 600.0,
            "store_evaluation": True,
            "cache_dir": str(root / "teacher_cache"),
        },
    }
    parsed = RebuildConfig.from_mapping(rebuild)
    train: dict[str, Any] = {
        "seed": 21102036,
        "device": "cuda",
        "model": architecture,
        "training": {
            "dataset_path": str(parsed.run_dir / "dataset.json"),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_path": str(metrics_path),
            "batch_size": 128,
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "scheduler": True,
            "legal_policy_mask": True,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 1.0,
            "selection_metric": "validation_loss",
            "checkpoint_every": 50,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0001,
            "baseline_eligible": False,
            "game_balanced_sampling": False,
            "use_dataset_split": True,
            "value_target_spec": parsed.value_spec,
            "validation_source_key": "composition_source",
            "log_every": 200,
        },
    }
    TrainingConfig.from_mapping(train["training"])
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=False)
    payloads = {"relabel": {"stockfish_rebuild": rebuild}, "train": train}
    selection_paths = {}
    for mode, simulations, book_path in (
        ("greedy", 0, None),
        ("puct", 32, None),
        ("book", 32, str(book)),
    ):
        evaluation_root = Path("data/games/evaluation") / f"{name}_{mode}_selection"
        selection: dict[str, Any] = {
            "champion_checkpoint": str(champion),
            "candidate_checkpoint_dir": str(checkpoint_dir),
            "training_manifest": str(parsed.run_dir / "training_manifest.json"),
            "report_path": str(evaluation_root / "report.json"),
            "pgn_dir": str(evaluation_root),
            "openings": 8,
            "opponent_depths": [1, 2],
            "seed": 22102036,
            "opening_min_full_moves": 2,
            "opening_max_full_moves": 5,
            "max_plies": 200,
            "minimum_aggregate_improvement_points": 1.0,
            "search_simulations": simulations,
            "opening_book": book_path,
            "device": "cuda",
        }
        GameplaySelectionConfig.from_mapping(selection)
        if evaluation_root.exists():
            raise ValueError(f"Evaluation output already exists: {evaluation_root}")
        payloads[f"select_{mode}"] = {"gameplay_selection": selection}
        selection_paths[mode] = str(configs / f"select_{mode}.yaml")
    for stage, payload in payloads.items():
        (configs / f"{stage}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    atomic_json(
        root / "experiment.json",
        {
            "format": "self-improving-chess-ai.stockfish-rebuild-experiment",
            "version": 1,
            "run_id": run_id,
            "champion_checkpoint": str(champion),
            "champion_sha256": sha256_file(champion),
            "opening_book": str(book),
            "opening_book_sha256": sha256_file(book),
            "relabel_config": str(configs / "relabel.yaml"),
            "train_config": str(configs / "train.yaml"),
            "selection_configs": selection_paths,
            "training_manifest": str(parsed.run_dir / "training_manifest.json"),
            "training_updates": 0,
            "initialization": "random_weights_no_init_checkpoint",
        },
    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(prepare_rebuild(args.run_id).as_posix())


if __name__ == "__main__":
    main()
