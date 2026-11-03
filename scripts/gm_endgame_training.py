"""Reserve/import/label GM endgames explicitly; expensive jobs are separate stages."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from chess_ai.config import load_config
from chess_ai.data.endgame_pilot import finalize, label
from chess_ai.data.examples import load_dataset
from chess_ai.data.gm_endgames import IMPORT_FORMAT, prepare_gm, validate_inputs
from chess_ai.data.refinement_replay import compose_refinement_replay
from chess_ai.data.stockfish_rebuild import _blocked
from chess_ai.data.stockfish_teacher import StockfishConfig
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig

COLLECTION = Path("data/experiments/gm_endgame_1500_v2_20261024_01")
PILOT = Path("data/experiments/endgame_pilot_v1_20261023_01")
ORIGINAL = Path("data/experiments/stockfish_rebuild_eval_v2_20261011_01")
PERSONAL = Path("data/experiments/chesscom_xxkiller31_20261018_01")


def reserve(run_id: str, collection: Path) -> Path:
    root = Path("data/experiments") / run_id
    if any(
        p.exists()
        for p in (root, Path("checkpoints") / run_id, Path("data/metrics") / f"{run_id}.jsonl")
    ):
        raise ValueError("Run ID exists; use its saved stages or choose another ID")
    champion = load_gameplay_champion()
    if champion is None:
        raise ValueError("A checksum-verified registered champion is required")
    coverage_path = collection / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    collection_record = json.loads((collection / "collection.json").read_text(encoding="utf-8"))
    if collection_record["coverage_sha256"] != sha256_file(coverage_path):
        raise ValueError("Collection coverage changed")
    if coverage["selection_ownership"] != {
        "hikaru": 500,
        "magnuscarlsen": 500,
        "fabianocaruana": 500,
    }:
        raise ValueError("This pilot requires the requested 500 games per GM")
    if sha256_file(Path(coverage["snapshot"]["path"])) != coverage["snapshot"]["sha256"]:
        raise ValueError("GM snapshot changed")
    prior = load_config(PILOT / "configs/pilot.yaml")["endgame_pilot"]
    if sha256_file(champion.checkpoint) != prior["champion_sha256"]:
        raise ValueError("Source pilot uses a different champion")
    # Discover all PGN evaluation directories, including experiments after the old pilot.
    directories = sorted(
        {p.parent for p in Path("data/games").rglob("*.pgn") if _blocked(p)},
        key=lambda p: (len(p.parts), str(p)),
    )
    roots: list[Path] = []
    for path in directories:
        if not any(path.is_relative_to(other) for other in roots):
            roots.append(path)
    teacher = copy.deepcopy(prior["stockfish"])
    teacher["cache_dir"] = str(root / "teacher_cache")
    StockfishConfig.from_mapping(teacher)
    config = {
        "run_dir": str(root),
        "seed": 24102038,
        "champion": str(champion.checkpoint),
        "champion_registry": "checkpoints/champion.json",
        "champion_sha256": sha256_file(champion.checkpoint),
        "stockfish": teacher,
        "source_catalogs": [str(p / "relabel/catalog.sqlite") for p in (ORIGINAL, PERSONAL)],
        "protected_datasets": [
            str(ORIGINAL / "relabel/dataset.json"),
            str(PERSONAL / "relabel/dataset.json"),
            "data/experiments/stockfish_refinement_v2_20261015_01/collection/dataset.pt",
        ],
        "protected_pilots": [str(PILOT)],
        "protected_rollout_roots": [
            str(PILOT / "development_champion"),
            str(PILOT / "development_candidate"),
            "data/experiments/endgame_retention_v1_20261024_01/development_candidate",
        ],
        "evaluation_roots": [str(p) for p in roots],
        "max_pieces": 10,
        "max_positions_per_game": 16,
        "final_per_bucket": 256,
        "gm_import": {
            "format": IMPORT_FORMAT,
            "version": 1,
            "coverage": str(coverage_path),
            "coverage_sha256": sha256_file(coverage_path),
            "snapshot": coverage["snapshot"]["path"],
            "games_per_player": 500,
            "players": ["fabianocaruana", "hikaru", "magnuscarlsen"],
        },
    }
    train = load_config(PILOT / "configs/train.yaml")
    train["seed"] = config["seed"]
    train["training"].update(
        dataset_path=str(root / "replay.pt"),
        checkpoint_dir=str(Path("checkpoints") / run_id),
        metrics_path=str(Path("data/metrics") / f"{run_id}.jsonl"),
        freeze_batch_norm_statistics=True,
        checkpoint_every=4,
    )
    TrainingConfig.from_mapping(train["training"], seed=train["seed"], device=train["device"])
    configs = root / "configs"
    configs.mkdir(parents=True)
    for name, payload in {"pilot": {"endgame_pilot": config}, "train": train}.items():
        (configs / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    atomic_json(
        root / "experiment.json",
        {
            "format": IMPORT_FORMAT + ".experiment",
            "version": 1,
            "champion": config["champion"],
            "champion_sha256": config["champion_sha256"],
            "registry_sha256": sha256_file(Path(config["champion_registry"])),
            "config_sha256": {p.name: sha256_file(p) for p in configs.glob("*.yaml")},
            "collection_sha256": sha256_file(collection / "collection.json"),
            "training_imported": False,
            "promotion_performed": False,
            "selection": "Baseline or best.pt only. Development on both GM validation and the original synthetic stress suite; whole-game non-regression before one confirmation candidate.",
            "confirmation": "GM whole-game components reserved before labels; never export them to training. Original confirmation remains protected and unused.",
            "intervention": "GM-derived endgame positions; fixed BatchNorm statistics based on retention diagnostic, not a demonstrated gameplay improvement.",
        },
    )
    return configs


def run_stage(
    root: Path, stage: str, *, confirm_training_import: bool = False, maximum: int | None = None
) -> dict[str, Any]:
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    for name, digest in experiment["config_sha256"].items():
        if sha256_file(root / "configs" / name) != digest:
            raise ValueError("Reserved GM experiment config changed")
    config = load_config(root / "configs/pilot.yaml")["endgame_pilot"]
    if sha256_file(Path(config["champion_registry"])) != experiment["registry_sha256"]:
        raise ValueError("Registered champion changed since reservation")
    if stage == "prepare":
        return prepare_gm(config, confirm_training_import=confirm_training_import)
    prepared = validate_inputs(root)
    if prepared["config"] != config:
        raise ValueError("Prepared GM configuration changed")
    if stage == "labels":
        return label(root, maximum)
    if stage == "finalize":
        return finalize(root)
    if stage != "compose":
        raise ValueError(f"Unknown GM stage: {stage}")
    # A successful finalize is mandatory; repeat composition verifies its binding.
    finalize(root)
    with run_lock(root):
        return compose(root, ORIGINAL / "relabel/dataset.json", seed=int(config["seed"]))


def compose(root: Path, replay_manifest: Path, *, seed: int) -> dict[str, Any]:
    """Recover an interrupted receipt write without replacing the replay tensors."""
    output, receipt = root / "replay.pt", root / "replay_receipt.json"
    expected = {
        "fresh_sha256": sha256_file(root / "fresh.pt"),
        "replay_manifest_sha256": sha256_file(replay_manifest),
        "seed": seed,
        "fresh_fraction": 0.5,
    }
    if output.exists():
        dataset = load_dataset(output, mmap=True)
        if dataset.metadata.get("composition_binding") != expected or dataset.metadata.get(
            "dataset_signature"
        ) != signature(expected):
            raise ValueError("GM replay input binding changed")
        report = {"output": str(output), "examples": len(dataset), **dataset.metadata}
        report["sha256"] = sha256_file(output)
        if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != report:
            raise ValueError("GM replay changed")
    else:
        if receipt.exists():
            raise ValueError("Committed GM replay is missing")
        report = compose_refinement_replay(root / "fresh.pt", replay_manifest, output, seed=seed)
        report["sha256"] = sha256_file(output)
    atomic_json(receipt, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", required=True, choices=("reserve", "prepare", "labels", "finalize", "compose")
    )
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    parser.add_argument("--confirm-training-import", action="store_true")
    parser.add_argument("--max-positions", type=int)
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.run_id) is None:
        raise ValueError("Use a safe run ID")
    if args.stage == "reserve":
        print(reserve(args.run_id, args.collection))
    else:
        print(
            json.dumps(
                run_stage(
                    Path("data/experiments") / args.run_id,
                    args.stage,
                    confirm_training_import=args.confirm_training_import,
                    maximum=args.max_positions,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
