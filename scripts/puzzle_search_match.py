"""Compare five-second scratch search with the frozen champion at PUCT32."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from chess_ai.arena.audit_statistics import result_counts
from chess_ai.arena.paired_audit import _checkpoint_results, _excluded_opening_fens, _opening_suite
from chess_ai.data.puzzle_training import checked
from chess_ai.storage.artifacts import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12112027)
    args = parser.parse_args()

    run = "puzzles_250k_v1_20261105_01"
    root = Path("data/experiments") / run
    plan = checked(root / "plan.json")
    scratch = Path("checkpoints") / f"{run}_scratch" / "best.pt"
    champion = Path(plan["champion"])
    selection = root / "selection.json"
    if sha256_file(selection) != plan["selection_sha256"]:
        raise ValueError("Puzzle selection differs from the training plan")
    excluded = _excluded_opening_fens(Path(plan["corpus"]) / "training_manifest.json")
    excluded.update(row["fen"] for row in checked(selection)["rows"])
    opening_count = 10
    # The paired audit generator puts the standard start first.
    suite = _opening_suite(
        openings=opening_count + 1,
        seed=args.seed,
        minimum_full_moves=2,
        maximum_full_moves=3,
        excluded_fens=excluded,
    )[1:]
    settings = (
        ("Scratch", scratch, 1_000_000, 5.0),
        ("Champion", champion, 32, None),
    )
    report: dict[str, Any] = {
        "seed": args.seed,
        "opponent_depth": 2,
        "max_plies": 200,
        "openings": [
            {"fen": board.fen(), "moves": moves, "seed": seed} for board, moves, seed in suite
        ],
        "models": {
            label: {
                "checkpoint": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "search_simulations": simulations,
                "search_seconds": seconds,
            }
            for label, checkpoint, simulations, seconds in settings
        },
        "games": [],
        "complete": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    games: list[dict[str, Any]] = report["games"]
    atomic_json(report_path, report)

    def record_game(label: str, game: dict[str, Any]) -> None:
        game_number = len(games) + 1
        games.append({**game, "model": label, "game_number": game_number})
        atomic_json(report_path, report)
        outcome = {"win": "Win", "draw": "Draw", "loss": "Lose"}[game["neural_outcome"]]
        print(
            f"Game {game_number} | {label} | {game['neural_color'].title()} | "
            f"{outcome} | Opening {game['opening_index']}/{opening_count}",
            flush=True,
        )

    for label, checkpoint, simulations, seconds in settings:

        def on_game(game: dict[str, Any], model: str = label) -> None:
            record_game(model, game)

        _checkpoint_results(
            checkpoint,
            role=label.lower(),
            opening_suite=suite,
            opponent_depth=2,
            max_plies=200,
            seed=args.seed,
            device=args.device,
            pgn_dir=args.output,
            search_simulations=simulations,
            search_seconds=seconds,
            opening_index_start=1,
            on_game=on_game,
        )
    report["scores"] = {
        label: result_counts([game for game in games if game["model"] == label])
        for label, *_ in settings
    }
    report["complete"] = True
    atomic_json(report_path, report)
    print(f"Report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
