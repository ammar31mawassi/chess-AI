"""Bounded local UCI review of archived evaluation moves; never training data."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess

from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher, policy_from_rows
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.games import parse_pgn_text


def move_tags(board: chess.Board, actual: chess.Move, best: chess.Move) -> list[str]:
    """Describe concrete legal tactics; engine score loss decides their severity."""
    tags: list[str] = []
    after = board.copy(stack=False)
    after.push(actual)
    if after.is_checkmate():
        return tags
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            tags.append("missed_mate_in_one")
            break
    for reply in after.legal_moves:
        child = after.copy(stack=False)
        child.push(reply)
        if child.is_checkmate():
            tags.append("allowed_mate_in_one")
            break
    if board.is_capture(best) and actual != best:
        tags.append("teacher_preferred_capture")
    if board.piece_type_at(actual.from_square) == chess.KING and not board.is_castling(actual):
        tags.append("king_move")
    return tags


def _score(
    teacher: StockfishTeacher, board: chess.Board, move: chess.Move | None
) -> dict[str, Any]:
    assert teacher.engine is not None
    independent = chess.Board(board.fen())
    if not independent.is_valid() or (move is not None and move not in independent.legal_moves):
        raise ValueError("Cannot review an invalid position or illegal archived move")
    teacher.engine.configure({"Clear Hash": None})
    rows = teacher._analyse(independent, root_moves=[move] if move is not None else None)
    _, trace = policy_from_rows(independent, rows, replace(teacher.config, multipv=1))
    if move is not None and trace[0]["move_uci"] != move.uci():
        raise ValueError("Forced-root analysis did not return the requested move")
    return trace[0]


def review_game(
    path: Path, teacher: StockfishTeacher, *, positions: int, seed: int
) -> dict[str, Any]:
    if positions <= 0:
        raise ValueError("positions must be positive")
    games = parse_pgn_text(path.read_text(encoding="utf-8"))
    if len(games) != 1 or games[0].headers.get("TrainingData") != "false":
        raise ValueError("Review requires exactly one explicitly evaluation-only PGN")
    game = games[0]
    color_name = game.headers.get("NeuralColor")
    if color_name not in {"white", "black"}:
        raise ValueError("Evaluation PGN must identify NeuralColor")
    color = color_name == "white"
    board = game.board()
    choices = []
    for ply, move in enumerate(game.mainline_moves()):
        if move not in board.legal_moves:
            raise ValueError("Archived PGN contains an illegal move")
        if board.turn == color:
            choices.append((ply, board.copy(stack=False), move))
        board.push(move)
    # Keep early decisions visible, then cover later turns reproducibly.
    first = choices[: min(8, positions)]
    later = random.Random(seed).sample(
        choices[len(first) :], min(positions - len(first), len(choices) - len(first))
    )
    selected = sorted(first + later, key=lambda item: item[0])
    rows = []
    for ply, position, actual in selected:
        best = _score(teacher, position, None)
        best_move = chess.Move.from_uci(best["move_uci"])
        played = best if actual == best_move else _score(teacher, position, actual)
        # Independent fixed-node searches are estimates, not exact minimax proof.
        loss = max(0, best["score_cp"] - played["score_cp"])
        tags = move_tags(position, actual, best_move)
        if len(played["pv"]) >= 2:
            reply_board = position.copy(stack=False)
            reply_board.push(actual)
            reply = chess.Move.from_uci(played["pv"][1])
            victim = reply_board.piece_at(reply.to_square)
            if (
                reply_board.is_capture(reply)
                and victim is not None
                and victim.piece_type != chess.PAWN
            ):
                reply_board.push(reply)
                if not any(
                    reply_board.is_capture(answer) and answer.to_square == reply.to_square
                    for answer in reply_board.legal_moves
                ):
                    tags.append("unrecapturable_piece_capture_in_teacher_pv")
        rows.append(
            {
                "ply": ply,
                "fen": position.fen(),
                "actual_move": actual.uci(),
                "teacher_move": best_move.uci(),
                "estimated_cp_loss": loss,
                "large_score_loss": loss >= 200,
                "tags": tags,
                "best": best,
                "played": played,
            }
        )
    return {
        "pgn": str(path),
        "pgn_sha256": sha256_file(path),
        "checkpoint": game.headers.get("Checkpoint"),
        "neural_color": color_name,
        "result": game.headers.get("Result"),
        "termination": game.headers.get("Termination"),
        "available_neural_positions": len(choices),
        "reviewed_positions": len(rows),
        "moves": rows,
    }


def diagnose_selection_games(
    selection_path: Path,
    candidate: Path,
    teacher_config: StockfishConfig,
    output: Path,
    *,
    openings: int = 2,
    positions: int = 16,
    seed: int = 2040,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Game diagnostic report already exists: {output}")
    if openings <= 0 or positions <= 0 or teacher_config.multipv != 1:
        raise ValueError("Positive review limits and MultiPV 1 are required")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("format") != "self-improving-chess-ai.gameplay-epoch-selection"
        or selection.get("version") != 1
    ):
        raise ValueError("Incompatible gameplay selection report")
    matches = [
        item
        for item in selection["candidates"]
        if Path(item["checkpoint"]).resolve() == candidate.resolve()
    ]
    if len(matches) != 1:
        raise ValueError("Candidate must identify exactly one checkpoint in the selection report")
    models = {"champion": selection["champion"], "candidate": matches[0]}
    summaries = {}
    with StockfishTeacher(replace(teacher_config, cache_dir=None)) as teacher:
        for role, model in models.items():
            if sha256_file(Path(model["checkpoint"])) != model["sha256"]:
                raise ValueError("Checkpoint digest no longer matches the selection report")
            reviewed = []
            for depth, records in sorted(model["games"].items()):
                indices = sorted(
                    {row["opening_index"] for row in records if row["opening_index"] != 0}
                )[:openings]
                for record in records:
                    if record["opening_index"] not in [0, *indices]:
                        continue
                    path = Path(record["pgn_path"])
                    review = review_game(
                        path,
                        teacher,
                        positions=positions,
                        seed=seed
                        + record["opening_index"] * 101
                        + (record["neural_color"] == "black"),
                    )
                    if (
                        not isinstance(review["checkpoint"], str)
                        or Path(review["checkpoint"]).resolve()
                        != Path(model["checkpoint"]).resolve()
                        or review["result"] != record["result"]
                    ):
                        raise ValueError(
                            "PGN checkpoint/result does not match the selection report"
                        )
                    reviewed.append(
                        {
                            "opponent_depth": depth,
                            "opening_index": record["opening_index"],
                            **review,
                        }
                    )
            moves = [move for game in reviewed for move in game["moves"]]
            summaries[role] = {
                "checkpoint": model["checkpoint"],
                "reviewed_games": len(reviewed),
                "reviewed_positions": len(moves),
                "unique_position_fens": len({move["fen"] for move in moves}),
                "large_score_losses": sum(move["large_score_loss"] for move in moves),
                "tags_at_large_losses": dict(
                    Counter(
                        tag for move in moves if move["large_score_loss"] for tag in move["tags"]
                    )
                ),
                "games": reviewed,
            }
        provenance = teacher.provenance
    report = {
        "format": "self-improving-chess-ai.offline-game-diagnostics",
        "version": 1,
        "selection_report": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "training_data": False,
        "training_updates": 0,
        "seed": seed,
        "varied_openings_per_depth": openings,
        "positions_per_game": positions,
        "teacher": provenance,
        "models": summaries,
        "limitations": "A deterministic subset of existing selection games, not a fresh skill benchmark. "
        "Independent best/forced-root fixed-node searches estimate score loss (threshold 200 cp). "
        "PV capture tags describe tactics, not proof of a hanging piece. Already-lost positions can hide "
        "further errors; mate scores use the documented bounded mapping. These PGNs and reports never enter training.",
    }
    atomic_json(output, report)
    return report
