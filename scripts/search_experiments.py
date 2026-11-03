"""Run independent search ablations and fresh failure-training stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chess_ai.agents.search_experiments import ARMS
from chess_ai.arena.search_experiments import benchmark, paired_matches
from chess_ai.config import load_config
from chess_ai.data.failure_training import collect, finalize_failures, label_failures
from chess_ai.data.stockfish_teacher import StockfishConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("benchmark", "matches", "collect"):
        command = commands.add_parser(stage)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=31102041)
        command.add_argument("--simulations", type=int, default=32)
        if stage != "collect":
            command.add_argument("--seconds", type=float)
        if stage == "benchmark":
            command.add_argument(
                "--suite", type=Path, default=Path("configs/search_development_v1.json")
            )
            command.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
            command.add_argument("--repeats", type=int, default=3)
        elif stage == "matches":
            command.add_argument("--arm", choices=ARMS[1:], required=True)
            command.add_argument("--openings", type=int, default=100)
            command.add_argument("--max-plies", type=int, default=240)
        else:
            command.add_argument("--pairs", type=int, default=50)
            command.add_argument("--max-plies", type=int, default=240)
            command.add_argument("--opponent", choices=("minimax", "self"), default="minimax")
    labels = commands.add_parser("labels")
    labels.add_argument("--collection", type=Path, required=True)
    labels.add_argument("--corpus", type=Path, required=True)
    labels.add_argument("--teacher-config", type=Path, required=True)
    labels.add_argument("--suite", type=Path, default=Path("configs/search_development_v1.json"))
    labels.add_argument("--output", type=Path, required=True)
    labels.add_argument("--seed", type=int, default=31102043)
    labels.add_argument("--positions-per-game", type=int, default=24)
    labels.add_argument("--minimum-loss-cp", type=int, default=150)
    labels.add_argument("--max-positions", type=int)
    labels.add_argument("--confirm-training-import", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--replay", type=Path, required=True)
    finalize.add_argument("--train-template", type=Path, required=True)
    finalize.add_argument("--seed", type=int, default=31102043)
    args = parser.parse_args()
    if args.stage == "benchmark":
        report = benchmark(
            args.checkpoint,
            args.suite,
            args.output,
            arms=args.arms,
            device=args.device,
            simulations=args.simulations,
            seed=args.seed,
            seconds=args.seconds,
            repeats=args.repeats,
        )
    elif args.stage == "matches":
        report = paired_matches(
            args.checkpoint,
            args.output,
            arm=args.arm,
            device=args.device,
            simulations=args.simulations,
            seed=args.seed,
            seconds=args.seconds,
            openings=args.openings,
            max_plies=args.max_plies,
        )
    elif args.stage == "collect":
        report = collect(
            args.checkpoint,
            args.output,
            device=args.device,
            simulations=args.simulations,
            seed=args.seed,
            pairs=args.pairs,
            max_plies=args.max_plies,
            opponent=args.opponent,
        )
    elif args.stage == "labels":
        config = load_config(args.teacher_config)
        teacher = replace(
            StockfishConfig.from_mapping(config["endgame_pilot"]["stockfish"]),
            cache_dir=args.output / "teacher_cache",
        )
        report = label_failures(
            args.collection,
            args.corpus,
            args.suite,
            args.output,
            teacher,
            confirm_training_import=args.confirm_training_import,
            seed=args.seed,
            positions_per_game=args.positions_per_game,
            minimum_loss_cp=args.minimum_loss_cp,
            maximum=args.max_positions,
        )
    else:
        report = finalize_failures(
            args.output,
            args.replay,
            args.train_template,
            seed=args.seed,
        )
    # Full per-case/game details remain in report.json, not thousands of terminal lines.
    printable = {k: v for k, v in report.items() if k != "results"}
    if "summary_by_arm" in printable:
        printable["summary_by_arm"] = {
            arm: {
                key: values[key]
                for key in (
                    "median_move_seconds",
                    "p95_move_seconds",
                    "network_positions",
                    "mean_inference_batch_size",
                    "cache_hits",
                    "first_repeat_solved",
                    "first_repeat_scored_cases",
                )
            }
            for arm, values in printable["summary_by_arm"].items()
        }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
