"""Reserve and run the isolated 3,000-GM-game full-corpus rebuild stages."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from chess_ai.config import load_config
from chess_ai.data.endgame_pilot import _checked_json
from chess_ai.data.full_corpus import FORMAT, SPEC, _freeze, finalize, inventory, labels
from chess_ai.data.stockfish_rebuild import _blocked
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig

ORIGINAL = Path("data/experiments/stockfish_rebuild_eval_v2_20261011_01")
PILOT = Path("data/experiments/endgame_pilot_v1_20261023_01")
GM_PILOT = Path("data/experiments/gm_endgame_training_v1_20261024_01")


def reserve(run_id: str, collection: Path) -> Path:
    root = Path("data/experiments") / run_id
    for path in (
        root,
        *[Path("checkpoints") / f"{run_id}_{arm}" for arm in ("scratch", "finetune")],
        *[Path("data/metrics") / f"{run_id}_{arm}.jsonl" for arm in ("scratch", "finetune")],
    ):
        if path.exists():
            raise ValueError("Run ID exists; use its saved stages or choose another ID")
    champion = load_gameplay_champion()
    if champion is None:
        raise ValueError("A verified registered champion is required")
    loaded = load_checkpoint(champion.checkpoint, map_location="cpu")
    if loaded.metadata.get("value_target_spec") != SPEC:
        raise ValueError("Champion targets differ from the full-corpus specification")
    protected = [PILOT, GM_PILOT]
    pilot_hashes = {sha256_file(p / "positions.json"): str(p) for p in protected}
    rollouts = []
    for folder in sorted({p.parent for p in Path("data/experiments").rglob("case_*.json")}):
        binding = _checked_json(folder / "binding.json")
        digest = binding["positions_sha256"]
        if digest not in pilot_hashes:
            raise ValueError(f"Unmapped evaluation rollout: {folder}; register its source pilot")
        rollouts.append({"folder": str(folder), "pilot": pilot_hashes[digest]})
    directories = sorted(
        {p.parent for p in Path("data/games").rglob("*.pgn") if _blocked(p)},
        key=lambda p: (len(p.parts), str(p)),
    )
    roots: list[Path] = []
    for path in directories:
        if not any(path.is_relative_to(other) for other in roots):
            roots.append(path)
    teacher = copy.deepcopy(load_config(PILOT / "configs/pilot.yaml")["endgame_pilot"]["stockfish"])
    teacher["cache_dir"] = str(root / "teacher_cache")
    config = {
        "format": FORMAT,
        "version": 1,
        "run_dir": str(root),
        "seed": 31102039,
        "champion": str(champion.checkpoint),
        "champion_sha256": sha256_file(champion.checkpoint),
        "champion_registry": "checkpoints/champion.json",
        "coverage": str(collection / "coverage.json"),
        "collection_record": str(collection / "collection.json"),
        "games_per_player": 1000,
        "expected_ownership": {p: 1000 for p in ("hikaru", "magnuscarlsen", "fabianocaruana")},
        "expected_additions": {p: 500 for p in ("hikaru", "magnuscarlsen", "fabianocaruana")},
        "stockfish": teacher,
        "shard_size": 2048,
        "sources": [
            {"name": "original", "path": str(ORIGINAL / "relabel/dataset.json")},
            {
                "name": "personal",
                "path": "data/experiments/chesscom_xxkiller31_20261018_01/relabel/dataset.json",
            },
            {
                "name": "refinement",
                "path": "data/experiments/stockfish_refinement_v2_20261015_01/collection/dataset.pt",
            },
            {"name": "targeted", "path": str(PILOT / "fresh.pt")},
            {"name": "gm_previous", "path": str(GM_PILOT / "fresh.pt")},
        ],
        "protected_pilots": [str(p) for p in protected],
        "rollouts": rollouts,
        "evaluation_roots": [str(p) for p in roots],
        "sampling_shares": {"general": 0.65, "endgame": 0.30, "basic_mate": 0.05},
    }
    for source in config["sources"]:
        if not Path(source["path"]).is_file():
            raise ValueError(f"Required label source missing: {source['path']}")
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "corpus.yaml").write_text(
        yaml.safe_dump({"full_corpus": config}, sort_keys=False), encoding="utf-8"
    )
    for arm in ("scratch", "finetune"):
        train = load_config(ORIGINAL / "configs/train.yaml")
        train.update(seed=config["seed"], device="cuda", model=loaded.model.config.to_dict())
        train["training"].update(
            dataset_path=str(root / "dataset.json"),
            checkpoint_dir=str(Path("checkpoints") / f"{run_id}_{arm}"),
            metrics_path=str(Path("data/metrics") / f"{run_id}_{arm}.jsonl"),
            epochs=100 if arm == "scratch" else 8,
            learning_rate=0.001 if arm == "scratch" else 0.00001,
            checkpoint_every=10 if arm == "scratch" else 4,
            early_stopping_patience=15 if arm == "scratch" else 3,
            baseline_eligible=arm == "finetune",
            freeze_batch_norm_statistics=arm == "finetune",
            validation_source_key="origin",
            source_sampling_multipliers={},
            game_balanced_sampling=False,
        )
        TrainingConfig.from_mapping(train["training"], seed=train["seed"], device=train["device"])
        (configs / f"train_{arm}.yaml").write_text(
            yaml.safe_dump(train, sort_keys=False), encoding="utf-8"
        )
    atomic_json(
        root / "experiment.json",
        {
            "format": FORMAT + ".experiment",
            "version": 1,
            "seed": config["seed"],
            "config_sha256": {p.name: sha256_file(p) for p in configs.glob("*.yaml")},
            "champion_sha256": sha256_file(champion.checkpoint),
            "registry_sha256": sha256_file(Path("checkpoints/champion.json")),
            "scratch": "Random weights, fresh optimizer, updating BatchNorm statistics; no init checkpoint.",
            "finetune": "Optional practical comparison from the frozen champion with its own appropriate schedule; not an initialization-only causal ablation.",
            "selection": "best.pt only per arm; original synthetic and GM development suites and whole-game D1/D2 non-regression before nominating at most one confirmation candidate.",
            "training_updates": 0,
            "promotion_performed": False,
        },
    )
    return configs


def settings(root: Path) -> dict[str, Any]:
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if experiment.get("format") != FORMAT + ".experiment" or experiment.get("version") != 1:
        raise ValueError("Incompatible full-corpus experiment")
    for name, digest in experiment["config_sha256"].items():
        if sha256_file(root / "configs" / name) != digest:
            raise ValueError("Reserved corpus/training config changed")
    if sha256_file(Path("checkpoints/champion.json")) != experiment["registry_sha256"]:
        raise ValueError("Registered champion changed since reservation")
    return load_config(root / "configs/corpus.yaml")["full_corpus"]


def train(root: Path, config: dict[str, Any], arm: str, *, resume: bool, device: str) -> None:
    _freeze(config)
    receipt = json.loads((root / "training_ready.json").read_text(encoding="utf-8"))
    if sha256_file(root / "dataset.json") != receipt["dataset_sha256"]:
        raise ValueError("Ready corpus manifest changed")
    entry = receipt["configs"][arm]
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("Ready training config changed")
    command = [sys.executable, "-m", "chess_ai", "train", "--config", str(path), "--device", device]
    if resume:
        checkpoint = Path(load_config(path)["training"]["checkpoint_dir"]) / "last.pt"
        command += ["--resume", str(checkpoint)]
    elif arm == "finetune":
        command += ["--init-checkpoint", config["champion"]]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", required=True, choices=("reserve", "inventory", "labels", "finalize", "train")
    )
    parser.add_argument(
        "--collection", type=Path, default=Path("data/experiments/gm_3000_v1_20261031_01")
    )
    parser.add_argument("--confirm-training-import", action="store_true")
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--arm", choices=("scratch", "finetune"), default="scratch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.run_id) is None:
        raise ValueError("Use a safe run ID")
    if args.stage == "reserve":
        print(reserve(args.run_id, args.collection))
        return
    root = Path("data/experiments") / args.run_id
    config = settings(root)
    if args.stage == "inventory":
        report = inventory(config, confirm_training_import=args.confirm_training_import)
    elif args.stage == "labels":
        report = labels(config, args.max_positions)
    elif args.stage == "finalize":
        report = finalize(config)
        report = {key: report[key] for key in ("examples", "splits", "confirmation_exported")}
    else:
        train(root, config, args.arm, resume=args.resume, device=args.device)
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
