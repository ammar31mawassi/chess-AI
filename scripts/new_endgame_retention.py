"""Reserve a matched BatchNorm retention experiment on the frozen endgame replay."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml

from chess_ai.config import load_config
from chess_ai.data.endgame_pilot import load_prepared
from chess_ai.data.examples import load_dataset
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig


def reserve(run_id: str, pilot: Path) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id) is None:
        raise ValueError("Use a safe run ID")
    root = Path("data/experiments") / run_id
    arms = ("control", "frozen_bn")
    outputs = [
        root,
        *[Path("checkpoints") / f"{run_id}_{arm}" for arm in arms],
        *[Path("data/metrics") / f"{run_id}_{arm}.jsonl" for arm in arms],
    ]
    if any(p.exists() for p in outputs):
        raise ValueError("Experiment output already exists; use its saved configs")
    champion = load_gameplay_champion()
    if champion is None:
        raise ValueError("A registered champion is required")
    prepared = load_prepared(pilot)
    if sha256_file(champion.checkpoint) != prepared["config"]["champion_sha256"]:
        raise ValueError("The source pilot uses a different champion")
    train = load_config(pilot / "configs/train.yaml")
    dataset = Path(train["training"]["dataset_path"])
    loaded = load_dataset(dataset)
    if loaded.metadata.get("explicit_split") is not True:
        raise ValueError("Retention experiment requires the original explicit splits")
    confirmation = {r["key"] for r in prepared["rows"] if r["split"] == "confirmation"}
    if confirmation & {str(row.metadata["position_key"]) for row in loaded}:
        raise ValueError("Reserved confirmation positions leaked into the replay")
    configs = root / "configs"
    configs.mkdir(parents=True)
    records = []
    for arm in arms:
        raw = copy.deepcopy(train)
        raw["training"].update(
            checkpoint_dir=str(Path("checkpoints") / f"{run_id}_{arm}"),
            metrics_path=str(Path("data/metrics") / f"{run_id}_{arm}.jsonl"),
            epochs=8,
            checkpoint_every=4,
            freeze_batch_norm_statistics=arm == "frozen_bn",
        )
        TrainingConfig.from_mapping(raw["training"], seed=raw["seed"], device=raw["device"])
        path = configs / f"train_{arm}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        records.append(
            {
                "arm": arm,
                "config": str(path),
                "config_sha256": sha256_file(path),
                "checkpoint_dir": raw["training"]["checkpoint_dir"],
                "metrics_path": raw["training"]["metrics_path"],
            }
        )
    manifest = {
        "format": "self-improving-chess-ai.endgame-retention-experiment",
        "version": 1,
        "pilot": str(pilot),
        "champion": str(champion.checkpoint),
        "champion_sha256": sha256_file(champion.checkpoint),
        "champion_registry_sha256": sha256_file(Path("checkpoints/champion.json")),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "positions_sha256": sha256_file(pilot / "positions.json"),
        "seed": train["seed"],
        "examples": len(loaded),
        "arms": records,
        "only_treatment_difference": "freeze_batch_norm_statistics",
        "selection": "Keep baseline eligible; compare best.pt from each arm on existing development only. Nominate at most one trained model. Baseline weights are not a candidate.",
        "gameplay": "Use the source pilot's validation roots, PUCT32, c_puct=1.5, no assistance, and the same whole-game development seed 25102037.",
        "confirmation": "Reuse the still-unused source pilot confirmation pool only after development passes; never train on it or try multiple candidates.",
        "promotion_performed": False,
    }
    atomic_json(root / "experiment.json", manifest)
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--pilot", type=Path, default=Path("data/experiments/endgame_pilot_v1_20261023_01")
    )
    args = parser.parse_args()
    print(reserve(args.run_id, args.pilot))
