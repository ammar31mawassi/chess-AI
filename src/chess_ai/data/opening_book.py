"""Validated opening-line import for training data and a runtime opening book."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np

from chess_ai.data.examples import TrainingExample, save_dataset
from chess_ai.environment.board_encoder import encode_board
from chess_ai.environment.move_encoder import MoveEncoder

OPENING_BOOK_FORMAT = "self-improving-chess-ai.opening-book"
OPENING_BOOK_VERSION = 1


class OpeningBookError(RuntimeError):
    """Raised when opening source data or a compiled book is invalid."""


def position_key(board: chess.Board) -> str:
    """Return a clock-independent standard-chess position key."""

    fields = board.fen(en_passant="fen").split()
    return " ".join(fields[:4])


@dataclass(frozen=True, slots=True)
class OpeningImportConfig:
    source_path: Path
    dataset_path: Path
    book_path: Path
    minimum_games: int = 1
    maximum_plies: int = 24

    def __post_init__(self) -> None:
        for name in ("source_path", "dataset_path", "book_path"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.minimum_games <= 0:
            raise ValueError("minimum_games must be positive")
        if self.maximum_plies <= 0:
            raise ValueError("maximum_plies must be positive")
        if self.dataset_path.resolve() == self.book_path.resolve():
            raise ValueError("dataset_path and book_path must be different")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> OpeningImportConfig:
        known = {
            "source_path",
            "dataset_path",
            "book_path",
            "minimum_games",
            "maximum_plies",
        }
        unknown = sorted(set(raw).difference(known))
        if unknown:
            raise ValueError(f"Unknown opening import settings: {', '.join(unknown)}")
        return cls(
            source_path=Path(str(raw["source_path"])),
            dataset_path=Path(str(raw["dataset_path"])),
            book_path=Path(str(raw["book_path"])),
            minimum_games=int(raw.get("minimum_games", 1)),
            maximum_plies=int(raw.get("maximum_plies", 24)),
        )


@dataclass(frozen=True, slots=True)
class OpeningImportSummary:
    source_path: Path
    dataset_path: Path
    book_path: Path
    rows_read: int
    accepted_lines: int
    invalid_lines: int
    filtered_lines: int
    positions: int
    book_moves: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_path", "dataset_path", "book_path"):
            payload[key] = str(payload[key])
        return payload


@dataclass(slots=True)
class _PositionAccumulator:
    board: chess.Board
    move_weights: dict[int, float]
    weighted_value_sum: float
    total_weight: float
    opening_names: set[str]
    eco_codes: set[str]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _percentage(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be a percentage")
    return value


def _line_moves(raw_moves: str) -> list[chess.Move]:
    game = chess.pgn.read_game(io.StringIO(f"{raw_moves.strip()} *"))
    if game is None or game.errors:
        raise ValueError("invalid opening movetext")
    return list(game.mainline_moves())


def import_opening_book(config: OpeningImportConfig) -> OpeningImportSummary:
    """Compile reviewed opening lines into soft targets and a lookup book."""

    if not config.source_path.is_file():
        raise OpeningBookError(f"Opening source CSV does not exist: {config.source_path}")
    conflicts = [path for path in (config.dataset_path, config.book_path) if path.exists()]
    if conflicts:
        raise OpeningBookError(
            "Opening import outputs already exist: " + ", ".join(str(path) for path in conflicts)
        )

    encoder = MoveEncoder()
    positions: dict[str, _PositionAccumulator] = {}
    rows_read = accepted = invalid = filtered = 0
    with config.source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Opening", "Num Games", "ECO", "Moves", "White_win%", "Black_win%"}
        if not required.issubset(reader.fieldnames or []):
            raise OpeningBookError(f"Expected opening CSV columns {sorted(required)}")
        for row in reader:
            rows_read += 1
            try:
                games = int(float(row["Num Games"]))
                if games < config.minimum_games:
                    filtered += 1
                    continue
                moves = _line_moves(row["Moves"])
                if not moves:
                    raise ValueError("empty opening line")
                white_expectation = (
                    _percentage(row, "White_win%") - _percentage(row, "Black_win%")
                ) / 100.0
                board = chess.Board()
                for ply, move in enumerate(moves[: config.maximum_plies], start=1):
                    if move not in board.legal_moves:
                        raise ValueError(f"illegal move at ply {ply}: {move.uci()}")
                    key = position_key(board)
                    action = encoder.encode(move)
                    side_value = (
                        white_expectation if board.turn == chess.WHITE else -white_expectation
                    )
                    accumulator = positions.get(key)
                    if accumulator is None:
                        accumulator = _PositionAccumulator(
                            board=board.copy(stack=False),
                            move_weights={},
                            weighted_value_sum=0.0,
                            total_weight=0.0,
                            opening_names=set(),
                            eco_codes=set(),
                        )
                        positions[key] = accumulator
                    weight = float(games)
                    accumulator.move_weights[action] = (
                        accumulator.move_weights.get(action, 0.0) + weight
                    )
                    accumulator.weighted_value_sum += side_value * weight
                    accumulator.total_weight += weight
                    accumulator.opening_names.add(row["Opening"].strip())
                    accumulator.eco_codes.add(row["ECO"].strip())
                    board.push(move)
                accepted += 1
            except (KeyError, TypeError, ValueError, chess.IllegalMoveError):
                invalid += 1
                continue

    if not positions:
        raise OpeningBookError("No valid opening positions passed the configured filters")

    examples: list[TrainingExample] = []
    book_positions: dict[str, list[dict[str, float | str]]] = {}
    for key in sorted(positions):
        item = positions[key]
        policy = np.zeros(encoder.action_size, dtype=np.float32)
        policy_total = sum(item.move_weights.values())
        for action, weight in item.move_weights.items():
            policy[action] = np.float32(weight / policy_total)
        policy[max(item.move_weights, key=lambda action: item.move_weights[action])] += np.float32(
            1.0 - float(policy.sum(dtype=np.float64))
        )
        identifier = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        examples.append(
            TrainingExample(
                board_tensor=encode_board(item.board),
                target_policy=policy,
                target_value=item.weighted_value_sum / item.total_weight,
                metadata={
                    "game_id": f"opening-position-{identifier}",
                    "source": "alexandrelemercier/all-chess-openings",
                    "fen": item.board.fen(),
                    "position_key": key,
                    "opening_names": sorted(item.opening_names),
                    "eco_codes": sorted(item.eco_codes),
                    "continuations": len(item.move_weights),
                    "opening_frequency_weight": min(6.0, 1.0 + math.log10(item.total_weight + 1.0)),
                    "target_kind": "opening_policy_value",
                },
            )
        )
        book_positions[key] = [
            {"move_uci": encoder.decode(action).uci(), "weight": weight}
            for action, weight in sorted(item.move_weights.items())
        ]

    save_dataset(
        config.dataset_path,
        examples,
        metadata={
            "kind": "explicit_external_opening_import",
            "opening_book_format": OPENING_BOOK_FORMAT,
            "opening_book_version": OPENING_BOOK_VERSION,
            "source_path": str(config.source_path),
            "source_dataset": "alexandrelemercier/all-chess-openings",
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "rows_read": rows_read,
            "accepted_lines": accepted,
            "invalid_lines": invalid,
            "filtered_lines": filtered,
            "positions": len(examples),
        },
    )
    _atomic_json(
        config.book_path,
        {
            "format": OPENING_BOOK_FORMAT,
            "version": OPENING_BOOK_VERSION,
            "source_path": str(config.source_path),
            "source_dataset": "alexandrelemercier/all-chess-openings",
            "minimum_games": config.minimum_games,
            "maximum_plies": config.maximum_plies,
            "positions": book_positions,
        },
    )
    return OpeningImportSummary(
        source_path=config.source_path,
        dataset_path=config.dataset_path,
        book_path=config.book_path,
        rows_read=rows_read,
        accepted_lines=accepted,
        invalid_lines=invalid,
        filtered_lines=filtered,
        positions=len(examples),
        book_moves=sum(len(moves) for moves in book_positions.values()),
    )


def load_opening_book(path: str | Path) -> dict[str, tuple[tuple[chess.Move, float], ...]]:
    """Load and strictly validate a compiled opening book."""

    source = Path(path)
    if not source.is_file():
        raise OpeningBookError(f"Opening book does not exist: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpeningBookError(f"Could not read opening book {source}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != OPENING_BOOK_FORMAT:
        raise OpeningBookError("Unsupported opening-book format")
    if raw.get("version") != OPENING_BOOK_VERSION:
        raise OpeningBookError(f"Unsupported opening-book version: {raw.get('version')!r}")
    raw_positions = raw.get("positions")
    if not isinstance(raw_positions, dict):
        raise OpeningBookError("Opening book positions must be a mapping")
    loaded: dict[str, tuple[tuple[chess.Move, float], ...]] = {}
    for key, raw_moves in raw_positions.items():
        if not isinstance(key, str) or not isinstance(raw_moves, list) or not raw_moves:
            raise OpeningBookError("Opening book contains an invalid position entry")
        moves: list[tuple[chess.Move, float]] = []
        for raw_move in raw_moves:
            if not isinstance(raw_move, dict):
                raise OpeningBookError("Opening book contains an invalid move entry")
            try:
                move = chess.Move.from_uci(str(raw_move["move_uci"]))
                weight = float(raw_move["weight"])
            except (KeyError, TypeError, ValueError) as exc:
                raise OpeningBookError("Opening book contains an invalid move entry") from exc
            if not math.isfinite(weight) or weight <= 0.0:
                raise OpeningBookError("Opening-book move weights must be finite and positive")
            moves.append((move, weight))
        loaded[key] = tuple(moves)
    return loaded
