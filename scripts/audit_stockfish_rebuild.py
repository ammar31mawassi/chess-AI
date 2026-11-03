"""Run fresh audits only for a candidate that passed its development selector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from chess_ai.arena.paired_audit import run_paired_audit
from chess_ai.config import config_section, load_config
from chess_ai.storage.artifacts import atomic_json, sha256_file


def audit_experiment(experiment_path: Path, mode: str, device: str, audit_id: str = "") -> None:
    if audit_id and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", audit_id) is None:
        raise ValueError("audit-id must be a safe name up to 40 characters")
    suffix = f"_{audit_id}" if audit_id else ""
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if (
        experiment.get("format") != "self-improving-chess-ai.stockfish-rebuild-experiment"
        or experiment.get("version") != 1
    ):
        raise ValueError("Unsupported Stockfish rebuild experiment")
    champion = Path(experiment["champion_checkpoint"])
    book = Path(experiment["opening_book"])
    if (
        sha256_file(champion) != experiment["champion_sha256"]
        or sha256_file(book) != experiment["opening_book_sha256"]
    ):
        raise ValueError("Frozen champion or opening book changed")
    selection = config_section(
        load_config(experiment["selection_configs"][mode]), "gameplay_selection"
    )
    report = json.loads(Path(selection["report_path"]).read_text(encoding="utf-8"))
    if not report["selected_is_candidate"]:
        print(f"{mode}: no candidate passed selection; epoch 120 remains champion.")
        return
    candidate = Path(report["selected_checkpoint"])
    selected = next(
        item
        for item in report["candidates"]
        if Path(item["checkpoint"]).resolve() == candidate.resolve()
    )
    if sha256_file(candidate) != selected["sha256"]:
        raise ValueError("Selected candidate changed since selection")
    if (
        report["champion"]["sha256"] != experiment["champion_sha256"]
        or report.get("search_simulations", 0) != selection["search_simulations"]
        or report.get("opening_book") != selection["opening_book"]
    ):
        raise ValueError("Selection report is inconsistent with the prepared comparison")
    roots = []
    excluded_starts = [item["fen"] for item in report["openings"]]
    for seed in (23102036, 24102036):
        exclusions_path = experiment_path.parent / f"{mode}_audit{suffix}_exclusions_{seed}.json"
        atomic_json(
            exclusions_path,
            {"pairs": [{"opening_fen": fen} for fen in sorted(set(excluded_starts))]},
        )
        seed_starts: list[str] = []
        for depth in (1, 2):
            root = (
                Path("data/games/evaluation")
                / f"{experiment_path.parent.name}_{mode}_audit{suffix}_{seed}_d{depth}"
            )
            if (root / "report.json").exists():
                prior = json.loads((root / "report.json").read_text(encoding="utf-8"))
                if (
                    prior["candidate_sha256"] != selected["sha256"]
                    or prior["champion_sha256"] != experiment["champion_sha256"]
                    or prior["audit_seed"] != seed
                    or prior["opponent_depth"] != depth
                    or prior["search_simulations"] != selection["search_simulations"]
                    or prior.get("opening_book") != selection["opening_book"]
                ):
                    raise ValueError("Existing audit belongs to different checkpoints")
                print(f"Completed audit retained: {root}")
            else:
                run_paired_audit(
                    candidate_checkpoint=candidate,
                    champion_checkpoint=champion,
                    openings=16,
                    audit_seed=seed,
                    opponent_depth=depth,
                    opening_min_full_moves=2,
                    opening_max_full_moves=5,
                    max_plies=200,
                    minimum_improvement_points=1.0,
                    device=device,
                    pgn_dir=root,
                    exclude_manifest=experiment["training_manifest"],
                    additional_exclude_manifests=(exclusions_path,),
                    search_simulations=selection["search_simulations"],
                    opening_book=selection["opening_book"],
                )
            roots.append(str(root / "report.json"))
            seed_starts.extend(
                item["fen"]
                for item in json.loads((root / "report.json").read_text(encoding="utf-8"))[
                    "openings"
                ]
            )
        excluded_starts.extend(seed_starts)
    print(json.dumps({"mode": mode, "reports": roots, "promotion_performed": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--mode", choices=("greedy", "puct", "book"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--audit-id", default="", help="fresh suffix for retry after partial PGN output"
    )
    args = parser.parse_args()
    audit_experiment(args.experiment, args.mode, args.device, args.audit_id)


if __name__ == "__main__":
    main()
