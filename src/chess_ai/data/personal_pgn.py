"""Explicit offline validation and frozen snapshots of personal training PGNs."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chess_ai.storage.artifacts import atomic_json, signature


def snapshot_personal_pgns(source_dir: Path, username: str, destination: Path) -> dict[str, Any]:
    """Validate every game before writing an explicitly authorized training snapshot.

    Historical moves identify positions only. This function creates no move/value
    targets: those must come from the separate Stockfish rebuild stages.
    """
    if destination.exists():
        raise ValueError(f"Personal PGN snapshot exists: {destination}")
    paths = sorted(source_dir.glob("*.pgn"))
    if not paths or not username.strip():
        raise ValueError("Personal PGN import requires a username and at least one local PGN")
    snapshots: list[tuple[Path, bytes]] = []
    records = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for path in paths:
        content = path.read_bytes()
        handle = io.StringIO(content.decode("utf-8-sig"))
        file_games = 0
        while (game := chess.pgn.read_game(handle)) is not None:
            file_games += 1
            label = f"{path.name} game {file_games}"
            if game.errors:
                raise ValueError(f"{label}: PGN parsing or move legality errors")
            if (
                game.headers.get("EvaluationKind")
                or game.headers.get("TrainingData", "true").lower() == "false"
            ):
                raise ValueError(f"{label}: evaluation-only game cannot enter this personal import")
            players = {game.headers.get(color, "").casefold() for color in ("White", "Black")}
            if username.casefold() not in players:
                raise ValueError(f"{label}: neither player matches the requested username")
            if game.headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
                raise ValueError(f"{label}: a complete game result is required")
            board = game.board()
            if type(board) is not chess.Board or board.chess960 or not board.is_valid():
                raise ValueError(f"{label}: only valid standard-chess games are supported")
            start_fen = board.fen()
            moves = []
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError(f"{label}: illegal move {move.uci()}")
                counts["white_positions" if board.turn else "black_positions"] += 1
                moves.append(move.uci())
                board.push(move)
            counts["games"] += 1
            if not moves:
                counts["empty_games"] += 1
                continue
            counts["games_with_positions"] += 1
            digest = signature({"fen": start_fen, "moves": moves})
            counts["duplicate_game_movetext"] += digest in seen
            seen.add(digest)
            counts["recorded_positions"] += len(moves)
        if not file_games:
            raise ValueError(f"{path}: no PGN games found")
        records.append(
            {
                "source": str(path.resolve()),
                "snapshot": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "games": file_games,
            }
        )
        snapshots.append((path, content))
    if not counts["recorded_positions"]:
        raise ValueError("Personal archives contain no recorded positions")
    # No output is created until every input has passed validation.
    destination.mkdir(parents=True)
    for path, content in snapshots:
        with (destination / path.name).open("xb") as snapshot_handle:
            snapshot_handle.write(content)
    report = {
        "format": "self-improving-chess-ai.personal-pgn-training-import",
        "version": 1,
        "username": username,
        "source_dir": str(source_dir.resolve()),
        "inputs": records,
        "counts": dict(counts),
        "unique_game_movetexts": len(seen),
        "training_import_explicit": True,
        "targets": "separate_stockfish_analysis_only",
        "training_updates": 0,
    }
    atomic_json(destination.parent / "personal_import.json", report)
    return report
