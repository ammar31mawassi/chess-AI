"""Read-only archived-game diagnosis and resumable unassisted endgame rollouts."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.arena.blunder_diagnostics import move_tags
from chess_ai.data.endgame_pilot import (
    _checked_json,
    _write_checked,
    family,
    labeled_rows,
    score_band,
)
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file


def review_move(teacher: StockfishTeacher, board: chess.Board, move: chess.Move) -> dict[str, Any]:
    """Teacher estimates plus exact python-chess rule outcomes from full history."""
    _, analysis = teacher.analyze(board)
    best = analysis["top_moves"][0]
    played = teacher.analyze_move(board, move)
    child = board.copy(stack=True)
    child.push(move)
    outcome = child.outcome(claim_draw=True)
    effective = played["score_cp"]
    if outcome is not None:
        effective = (
            0 if outcome.winner is None else (100000 if outcome.winner == board.turn else -100000)
        )
    tags = move_tags(board, move, chess.Move.from_uci(best["move_uci"]))
    if move.promotion:
        tags.append("promotion")
    if any(m.promotion for m in board.legal_moves):
        tags.append("promotion_available")
    if outcome is not None:
        tags.append(outcome.termination.name.lower())
    return {
        "fen": board.fen(),
        "actual_move": move.uci(),
        "teacher_move": best["move_uci"],
        "best_cp": best["score_cp"],
        "played_cp": played["score_cp"],
        "effective_played_cp": effective,
        "estimated_cp_loss": max(0, best["score_cp"] - effective),
        "win_to_nonwin_proxy": best["score_cp"] >= 300 and effective < 100,
        "balanced_to_loss_proxy": abs(best["score_cp"]) < 300 and effective <= -300,
        "tags": tags,
        "best": best,
        "played": played,
        "teacher_repetition_history": False,
    }


def summarize_games(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    moves = [move for game in games for move in game["moves"]]
    for game in games:
        groups[game["entry_band"]].append(game)
    rates = {}
    for band, subset in sorted(groups.items()):
        natural = [g for g in subset if g["result"] != "*"]
        rates[band] = {
            "games": len(subset),
            "natural_results": len(natural),
            "wins": sum(g["neural_result"] == "win" for g in natural),
            "draws": sum(g["neural_result"] == "draw" for g in natural),
            "losses": sum(g["neural_result"] == "loss" for g in natural),
            "unresolved": len(subset) - len(natural),
        }
    return {
        "games": len(games),
        "entry_bands": rates,
        "reviewed_moves": len(moves),
        "large_score_losses": sum(m["estimated_cp_loss"] >= 200 for m in moves),
        "win_to_nonwin_proxy": sum(m["win_to_nonwin_proxy"] for m in moves),
        "balanced_to_loss_proxy": sum(m["balanced_to_loss_proxy"] for m in moves),
        "tags_at_large_losses": dict(
            Counter(t for m in moves if m["estimated_cp_loss"] >= 200 for t in m["tags"])
        ),
        "terminations": dict(Counter(g["termination"] for g in games)),
    }


def _result(result: str, color: chess.Color) -> str:
    if result == "*":
        return "unresolved"
    if result == "1/2-1/2":
        return "draw"
    return "win" if (result == "1-0") == color else "loss"


def diagnose_archives(
    paths: Sequence[Path],
    checkpoint: Path,
    teacher_config: StockfishConfig,
    output: Path,
    *,
    games_per_stratum: int = 2,
    positions_per_game: int = 8,
    seed: int = 23102036,
    max_pieces: int = 10,
) -> dict[str, Any]:
    if games_per_stratum <= 0 or positions_per_game <= 0:
        raise ValueError("Diagnostic sample sizes must be positive")
    candidates: dict[str, list[tuple[Path, chess.pgn.Game, list[Any]]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    records = []
    for path in sorted(paths):
        with path.open(encoding="utf-8-sig") as handle:
            game = chess.pgn.read_game(handle)
            if game is None or chess.pgn.read_game(handle) is not None or game.errors:
                raise ValueError("Diagnosis requires one valid game per PGN")
        if (
            game.headers.get("TrainingData") != "false"
            or Path(game.headers.get("Checkpoint", "")).resolve() != checkpoint.resolve()
        ):
            raise ValueError(f"Expected evaluation-only PGNs for the frozen checkpoint: {path}")
        color_name = game.headers.get("NeuralColor")
        if color_name not in {"white", "black"}:
            raise ValueError("Diagnosis requires NeuralColor")
        board = game.board()
        choices = []
        for ply, move in enumerate(game.mainline_moves()):
            if move not in board.legal_moves:
                raise ValueError("Illegal archived move")
            if (
                len(board.piece_map()) <= max_pieces
                and board.turn == (color_name == "white")
                and not board.is_game_over(claim_draw=True)
            ):
                choices.append((ply, board.copy(stack=True), move))
            board.push(move)
        counts["games"] += 1
        counts["reached_endgame_on_neural_turn"] += bool(choices)
        counts[game.headers.get("Termination", "unknown")] += 1
        records.append({"path": str(path), "sha256": sha256_file(path)})
        if choices:
            candidates[f"{game.headers.get('OpponentDepth')}:{color_name}"].append(
                (path, game, choices)
            )
    binding = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "inputs": records,
        "teacher": teacher_config.analysis_settings(),
        "binary": sha256_file(teacher_config.executable),
        "seed": seed,
        "games_per_stratum": games_per_stratum,
        "positions_per_game": positions_per_game,
        "max_pieces": max_pieces,
    }
    with run_lock(output):
        output.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        selected = [
            item
            for key in sorted(candidates)
            for item in rng.sample(candidates[key], min(games_per_stratum, len(candidates[key])))
        ]
        reviewed = []
        with StockfishTeacher(teacher_config) as teacher:
            binding["teacher_provenance"] = teacher.provenance
            state_path = output / "binding.json"
            if state_path.exists() and _checked_json(state_path) != binding:
                raise ValueError("Diagnostic input/settings changed")
            _write_checked(state_path, binding)
            for index, (path, game, choices) in enumerate(selected):
                destination = output / f"game_{index:03d}.json"
                if destination.exists():
                    reviewed.append(_checked_json(destination))
                    continue
                # Entry is always present; spread remaining positions across the game.
                selected_indices = sorted(
                    {
                        round(
                            i
                            * (len(choices) - 1)
                            / max(1, min(positions_per_game, len(choices)) - 1)
                        )
                        for i in range(min(positions_per_game, len(choices)))
                    }
                )
                moves = [
                    {"ply": choices[i][0], **review_move(teacher, choices[i][1], choices[i][2])}
                    for i in selected_indices
                ]
                entry = moves[0]
                result = game.headers.get("Result", "*")
                if game.headers.get("Termination") == "move_limit":
                    result = "*"
                row = {
                    "pgn": str(path),
                    "pgn_sha256": sha256_file(path),
                    "entry_ply": entry["ply"],
                    "entry_band": score_band(entry["best_cp"]),
                    "entry_cp": entry["best_cp"],
                    "family_at_entry": family(choices[0][1]),
                    "result": result,
                    "neural_result": _result(result, game.headers["NeuralColor"] == "white"),
                    "neural_color": game.headers["NeuralColor"],
                    "opponent_depth": game.headers.get("OpponentDepth"),
                    "termination": game.headers.get("Termination", "unknown"),
                    "moves": moves,
                }
                _write_checked(destination, row)
                reviewed.append(row)
                print(f"Reviewed endgame {index + 1}/{len(selected)}", flush=True)
        report = {
            "format": "self-improving-chess-ai.endgame-diagnostics",
            "version": 1,
            "binding": binding,
            "coverage": dict(counts),
            "summary": summarize_games(reviewed),
            "games": reviewed,
            "training_data": False,
            "training_updates": 0,
            "limitations": "Descriptive sample of old selection games, not fresh confirmation. Entry and move quality use bounded Stockfish cp estimates, not exact WDL. Teacher FENs retain clocks but omit repetition history; exact terminal tags retain history. King-move tags do not measure good king activity. Unsampled mistakes are not counted.",
        }
        atomic_json(output / "report.json", report)
        return report


def evaluate_endgames(
    root: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "validation",
    simulations: int = 32,
    device: str = "cpu",
    cases_per_bucket: int = 2,
    max_plies: int = 160,
    opponent_depth: int = 2,
) -> dict[str, Any]:
    if (
        split not in {"validation", "confirmation"}
        or min(cases_per_bucket, max_plies, opponent_depth, simulations) <= 0
    ):
        raise ValueError("Invalid endgame evaluation limits/split")
    prepared, rows = labeled_rows(root)
    config = prepared["config"]
    buckets: Counter[str] = Counter()
    chosen = []
    seen_groups: set[str] = set()
    for row in sorted(rows, key=lambda r: r["key"]):
        board = chess.Board(row["fen"])
        bucket = f"{row['family']}:{board.turn}:{row['band']}"
        # Independent roots, not three correlated descendants counted as games.
        if (
            row["split"] == split
            and row["group_id"] not in seen_groups
            and buckets[bucket] < cases_per_bucket
        ):
            chosen.append(row)
            seen_groups.add(row["group_id"])
            buckets[bucket] += 1
    if not chosen:
        raise ValueError("No held-out endgames available")
    binding = {
        "positions_sha256": sha256_file(root / "positions.json"),
        "labels_sha256": sha256_file(root / "labels.sqlite"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": split,
        "simulations": simulations,
        "c_puct": 1.5,
        "device": device,
        "max_plies": max_plies,
        "opponent_depth": opponent_depth,
        "cases_per_bucket": cases_per_bucket,
        "cases": [r["key"] for r in chosen],
        "seed": config["seed"],
        "assistance": "none",
    }
    if split == "confirmation":
        freeze_confirmation(root, binding, config["champion_sha256"])
    with run_lock(output):
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "binding.json"
        if state_path.exists() and _checked_json(state_path) != binding:
            raise ValueError("Evaluation resume changed checkpoint, inputs, device, or budget")
        _write_checked(state_path, binding)
        agent = NeuralMCTSAgent(
            checkpoint, device=device, simulations=simulations, c_puct=1.5, seed=config["seed"]
        )
        results = []
        start = time.perf_counter()
        with StockfishTeacher(StockfishConfig.from_mapping(config["stockfish"])) as teacher:
            for index, row in enumerate(chosen):
                path = output / f"case_{index:03d}.json"
                state = (
                    _checked_json(path)
                    if path.exists()
                    else {"case_key": row["key"], "uci": [], "moves": [], "neural_seconds": 0.0}
                )
                if state["case_key"] != row["key"]:
                    raise ValueError("Evaluation case mismatch")
                board = chess.Board(row["fen"])
                color = board.turn
                for uci in state["uci"]:
                    board.push_uci(uci)
                while not board.is_game_over(claim_draw=True) and len(state["uci"]) < max_plies:
                    ply = len(state["uci"])
                    if board.turn == color:
                        before = time.perf_counter()
                        move = agent.choose_move(board.copy(stack=True))
                        elapsed = time.perf_counter() - before
                        state["moves"].append({"ply": ply, **review_move(teacher, board, move)})
                        state["neural_seconds"] += elapsed
                    else:
                        # Fresh per-ply RNG makes resume independent of process history.
                        opponent = MinimaxAgent(
                            depth=opponent_depth, seed=config["seed"] + index * 10000 + ply
                        )
                        move = opponent.choose_move(board.copy(stack=True))
                    if move not in board.legal_moves:
                        raise ValueError("Agent returned an illegal move")
                    board.push(move)
                    state["uci"].append(move.uci())
                    _write_checked(path, state)
                outcome = board.outcome(claim_draw=True)
                result = outcome.result() if outcome is not None else "*"
                state.update(
                    {
                        "entry_band": row["band"],
                        "entry_cp": row["label"]["audit"]["top_moves"][0]["score_cp"],
                        "result": result,
                        "neural_result": _result(result, color),
                        "termination": outcome.termination.name.lower()
                        if outcome
                        else "move_limit_unresolved",
                        "family": row["family"],
                        "neural_color": "white" if color else "black",
                        "complete": True,
                    }
                )
                _write_checked(path, state)
                results.append(state)
                elapsed = time.perf_counter() - start
                print(
                    f"Endgames {index + 1}/{len(chosen)}; ETA {elapsed / (index + 1) * (len(chosen) - index - 1) / 60:.1f} min (completed cases included)",
                    flush=True,
                )
        report = {
            "format": "self-improving-chess-ai.endgame-evaluation",
            "version": 1,
            "binding": binding,
            "summary": summarize_games(results),
            "cases": results,
            "training_data": False,
            "promotion_performed": False,
            "limitations": "Unassisted PUCT versus local minimax, not Stockfish gameplay. Approximate cp entry buckets; move caps are unresolved. Synthetic positions are not a population estimate. Paired independent roots, not per-move samples, are the statistical units.",
        }
        atomic_json(output / "report.json", report)
        return report


def freeze_confirmation(root: Path, binding: dict[str, Any], champion_sha256: str) -> None:
    """Allow exactly one candidate and one common budget on the reserved holdout."""
    if binding["simulations"] != 32:
        raise ValueError("Confirmation uses the preregistered PUCT32 budget")
    with run_lock(root):
        path = root / "confirmation_pair.json"
        suite = {
            key: value
            for key, value in binding.items()
            if key not in {"checkpoint", "checkpoint_sha256"}
        }
        saved = (
            _checked_json(path)
            if path.exists()
            else {"suite": suite, "champion": champion_sha256, "candidate": None}
        )
        if saved["suite"] != suite or saved["champion"] != champion_sha256:
            raise ValueError("Confirmation suite/budget changed")
        digest = binding["checkpoint_sha256"]
        if digest != champion_sha256:
            if saved["candidate"] not in (None, digest):
                raise ValueError(
                    "Confirmation already bound to another candidate; do not retest this holdout"
                )
            saved["candidate"] = digest
        _write_checked(path, saved)
