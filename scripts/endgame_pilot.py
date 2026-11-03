"""Run a bounded endgame stage, keeping CPU teaching separate from CUDA training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chess_ai.arena.endgame_diagnostics import diagnose_archives, evaluate_endgames
from chess_ai.config import load_config
from chess_ai.data.endgame_pilot import finalize, label, load_prepared, prepare, teacher_smoke
from chess_ai.data.stockfish_teacher import StockfishConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("prepare")
    commands.add_parser("teacher-smoke")
    labels = commands.add_parser("labels")
    labels.add_argument("--max-positions", type=int)
    commands.add_parser("finalize")
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--pgn-root", type=Path, action="append", required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--games-per-stratum", type=int, default=2)
    diagnose.add_argument("--positions-per-game", type=int, default=8)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "confirmation"), default="validation")
    evaluate.add_argument("--simulations", type=int, default=32)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--cases-per-bucket", type=int, default=2)
    evaluate.add_argument("--max-plies", type=int, default=160)
    evaluate.add_argument("--opponent-depth", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)["endgame_pilot"]
    root = Path(config["run_dir"])
    if args.stage in {"labels", "finalize", "evaluate"} and load_prepared(root)["config"] != config:
        raise ValueError("Pilot config changed after preparation; reserve another run")
    if args.stage == "prepare":
        report = prepare(config)
    elif args.stage == "teacher-smoke":
        report = teacher_smoke(config)
        report = {
            key: report[key] for key in ("positions", "best_move_changes", "score_band_changes")
        }
    elif args.stage == "labels":
        report = label(root, args.max_positions)
    elif args.stage == "finalize":
        report = finalize(root)
    elif args.stage == "diagnose":
        teacher = replace(
            StockfishConfig.from_mapping(config["stockfish"]),
            nodes=50000,
            multipv=1,
            cache_dir=args.output / "teacher_cache",
        )
        report = diagnose_archives(
            sorted({p for source in args.pgn_root for p in source.rglob("*.pgn")}),
            Path(config["champion"]),
            teacher,
            args.output,
            games_per_stratum=args.games_per_stratum,
            positions_per_game=args.positions_per_game,
            seed=config["seed"],
            max_pieces=config["max_pieces"],
        )
        report = report["summary"]
    else:
        report = evaluate_endgames(
            root,
            args.checkpoint,
            args.output,
            split=args.split,
            simulations=args.simulations,
            device=args.device,
            cases_per_bucket=args.cases_per_bucket,
            max_plies=args.max_plies,
            opponent_depth=args.opponent_depth,
        )
        report = report["summary"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
