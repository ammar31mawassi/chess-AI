"""Reserve a matched short Stockfish-sampling ablation using an existing frozen replay."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from chess_ai.config import config_section, load_config
from chess_ai.storage.artifacts import atomic_json
from chess_ai.training.trainer import TrainingConfig


def prepare_ablation(run_id: str, source_train: Path, source_select: Path) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id) is None:
        raise ValueError("run-id must be a safe 1-81 character directory name")
    original_train = load_config(source_train)
    original_select = load_config(source_select)
    settings = TrainingConfig.from_mapping(config_section(original_train, "training"))
    if settings.dataset_path is None or not settings.dataset_path.is_file():
        raise ValueError("The original composed replay must already exist")
    selection = config_section(original_select, "gameplay_selection")
    if not Path(selection["champion_checkpoint"]).is_file():
        raise ValueError("The frozen champion checkpoint must already exist")
    name = f"stockfish_sampling_ablation_{run_id}"
    root = Path("data/experiments") / name
    for variant in ("control", "teacher_focus"):
        for output in (
            Path("checkpoints") / f"{name}_{variant}",
            Path("data/metrics") / f"{name}_{variant}.jsonl",
            Path("data/games/evaluation") / f"{name}_{variant}_selection",
        ):
            if output.exists():
                raise ValueError(f"Ablation output already exists: {output}")
    root.mkdir(parents=True, exist_ok=False)
    configs = root / "configs"
    configs.mkdir()
    variants = {}
    for variant, multipliers in (
        ("control", {}),
        ("teacher_focus", {"stockfish_continuations": 8.0}),
    ):
        candidate_name = f"{name}_{variant}"
        train = {
            **original_train,
            "seed": original_train.get("seed", 2037),
            "training": {**config_section(original_train, "training")},
        }
        train["training"].update(
            {
                "checkpoint_dir": f"checkpoints/{candidate_name}",
                "metrics_path": f"data/metrics/{candidate_name}.jsonl",
                "epochs": 10,
                "checkpoint_every": 5,
                "early_stopping_patience": 0,
                "source_sampling_multipliers": multipliers,
                "sampling_source_key": "composition_source",
                "selection_metric": "validation_policy_loss",
                "validation_source_key": "composition_source",
            }
        )
        TrainingConfig.from_mapping(train["training"])
        select = {**original_select, "seed": 10112035, "gameplay_selection": {**selection}}
        select["gameplay_selection"].update(
            {
                "candidate_checkpoint_dir": f"checkpoints/{candidate_name}",
                "openings": 8,
                "report_path": f"data/games/evaluation/{candidate_name}_selection/report.json",
                "pgn_dir": f"data/games/evaluation/{candidate_name}_selection",
            }
        )
        for stage, payload in (("train", train), ("select", select)):
            (configs / f"{variant}_{stage}.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
        variants[variant] = {
            "source_sampling_multipliers": multipliers,
            "train_config": str(configs / f"{variant}_train.yaml"),
            "select_config": str(configs / f"{variant}_select.yaml"),
        }
    atomic_json(
        root / "experiment.json",
        {
            "format": "self-improving-chess-ai.stockfish-sampling-ablation",
            "version": 1,
            "run_id": run_id,
            "training_updates": 0,
            "source_train_config": str(source_train),
            "source_select_config": str(source_select),
            "initial_checkpoint": selection["champion_checkpoint"],
            "single_changed_training_factor": "source_sampling_multipliers",
            "variants": variants,
            "note": "Both variants start independently from the same frozen champion. Same replay, split seed, "
            "LR, loss weights, 10-epoch scheduler, policy selection metric and fresh gameplay seed. "
            "Only the teacher-focus variant multiplies Stockfish sampling mass by eight. "
            "No new games, datasets, or optimizer steps are produced by preparation.",
        },
    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-select", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_ablation(args.run_id, args.source_train, args.source_select).as_posix())


if __name__ == "__main__":
    main()
