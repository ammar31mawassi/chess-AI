"""Explicit, deterministic adapters for the reviewed local Kaggle chess files."""

from __future__ import annotations

import csv
import io
import math
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import chess
import chess.pgn
import numpy as np

from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import TrainingExample, save_dataset
from chess_ai.environment.board_encoder import encode_board
from chess_ai.environment.move_encoder import MoveEncoder

KAGGLE_IMPORT_FORMAT = "self-improving-chess-ai.kaggle-import"
KAGGLE_IMPORT_VERSION = 1
ImportKind = Literal["evaluations", "tactics", "games"]


class KaggleImportError(RuntimeError):
    """Raised when reviewed external data cannot be safely converted."""


@dataclass(frozen=True, slots=True)
class KaggleImportConfig:
    """Bounded import settings for one external source."""

    kind: ImportKind
    source_path: Path
    output_path: Path
    target_examples: int
    seed: int = 2026
    max_source_rows: int | None = None
    centipawn_scale: float = 400.0
    evaluation_perspective: Literal["side_to_move"] = "side_to_move"
    minimum_elo: int = 2000
    minimum_base_seconds: int = 300
    positions_per_game: int = 4
    maximum_ply: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.kind not in {"evaluations", "tactics", "games"}:
            raise ValueError(f"Unsupported Kaggle import kind: {self.kind!r}")
        if self.target_examples <= 0:
            raise ValueError("target_examples must be positive")
        if self.max_source_rows is not None and self.max_source_rows <= 0:
            raise ValueError("max_source_rows must be positive when configured")
        if self.centipawn_scale <= 0.0:
            raise ValueError("centipawn_scale must be positive")
        if self.minimum_elo < 0 or self.minimum_base_seconds < 0:
            raise ValueError("rating and time filters cannot be negative")
        if self.positions_per_game <= 0:
            raise ValueError("positions_per_game must be positive")
        if self.maximum_ply is not None:
            if self.kind != "games":
                raise ValueError("maximum_ply is supported only for game imports")
            if self.maximum_ply <= 0:
                raise ValueError("maximum_ply must be positive when configured")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, seed: int = 2026) -> KaggleImportConfig:
        values = dict(raw)
        values.setdefault("seed", seed)
        for name in ("source_path", "output_path"):
            if name in values:
                values[name] = Path(str(values[name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown Kaggle import settings: {', '.join(unknown)}")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class KaggleImportSummary:
    kind: ImportKind
    source_path: Path
    output_path: Path
    source_rows_read: int
    accepted_rows: int
    examples: int
    invalid_rows: int
    filtered_rows: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["output_path"] = str(self.output_path)
        return result


def _evaluation_value(raw: str, scale: float) -> float:
    value = raw.strip()
    if not value:
        raise ValueError("empty evaluation")
    if value.startswith("#"):
        mate = value[1:].strip()
        if mate.startswith("+"):
            return 1.0
        if mate.startswith("-"):
            return -1.0
        number = int(mate)
        return 1.0 if number > 0 else -1.0
    centipawns = float(value)
    if not math.isfinite(centipawns):
        raise ValueError("non-finite evaluation")
    return math.tanh(centipawns / scale)


def _one_hot_policy(move: chess.Move) -> np.ndarray[Any, np.dtype[np.float32]]:
    policy = np.zeros(MoveEncoder.ACTION_SIZE, dtype=np.float32)
    policy[MoveEncoder().encode(move)] = 1.0
    return policy


def _neutral_legal_policy(board: chess.Board) -> np.ndarray[Any, np.dtype[np.float32]]:
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("terminal positions do not have a neutral legal policy")
    policy = np.zeros(MoveEncoder.ACTION_SIZE, dtype=np.float32)
    probability = np.float32(1.0 / len(legal))
    encoder = MoveEncoder()
    for move in legal:
        policy[encoder.encode(move)] = probability
    # Make the float32 sum exact enough for the dataset invariant.
    policy[encoder.encode(legal[-1])] += np.float32(1.0 - float(policy.sum()))
    return policy


def _phase(board: chess.Board) -> str:
    if board.fullmove_number <= 12:
        return "opening"
    non_pawn_material = sum(
        len(board.pieces(piece_type, color))
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for color in (chess.WHITE, chess.BLACK)
    )
    return "endgame" if non_pawn_material <= 6 else "middlegame"


def _value_band(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.2:
        return "equal"
    if magnitude < 0.65:
        return "edge_positive" if value > 0 else "edge_negative"
    return "winning_positive" if value > 0 else "winning_negative"


def _reservoir_add(
    reservoir: list[TrainingExample],
    item: TrainingExample,
    *,
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < capacity:
        reservoir[replacement] = item


def _import_positions(config: KaggleImportConfig) -> tuple[list[TrainingExample], dict[str, int]]:
    rng = random.Random(config.seed)
    strata: dict[str, list[TrainingExample]] = defaultdict(list)
    seen_by_stratum: dict[str, int] = defaultdict(int)
    # Fifteen phase/value strata keep the sample from collapsing into quiet middlegames.
    capacity = max(1, math.ceil(config.target_examples / 15))
    rows_read = accepted = invalid = 0
    with config.source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"FEN", "Evaluation"}
        if config.kind == "tactics":
            required.add("Move")
        if not required.issubset(reader.fieldnames or []):
            raise KaggleImportError(f"Expected CSV columns {sorted(required)}")
        for row_index, row in enumerate(reader, start=1):
            if config.max_source_rows is not None and rows_read >= config.max_source_rows:
                break
            rows_read += 1
            try:
                board = chess.Board(row["FEN"])
                if board.is_game_over(claim_draw=True):
                    raise ValueError("terminal position")
                value = _evaluation_value(row["Evaluation"], config.centipawn_scale)
                move: chess.Move | None = None
                if config.kind == "tactics":
                    move = chess.Move.from_uci(row["Move"].strip())
                    if move not in board.legal_moves:
                        raise ValueError("illegal tactic move")
                    policy = _one_hot_policy(move)
                else:
                    policy = _neutral_legal_policy(board)
                phase = _phase(board)
                band = _value_band(value)
                stratum = f"{phase}:{band}"
                example = TrainingExample(
                    board_tensor=encode_board(board),
                    target_policy=policy,
                    target_value=value,
                    metadata={
                        "game_id": f"kaggle-{config.kind}-{row_index}",
                        "source": "ronakbadhe/chess-evaluations",
                        "source_row": row_index,
                        "fen": board.fen(),
                        "phase": phase,
                        "evaluation_band": band,
                        "evaluation": row["Evaluation"].strip(),
                        "evaluation_perspective": config.evaluation_perspective,
                        "target_kind": "policy_value" if move is not None else "value_only",
                        **({"move_uci": move.uci()} if move is not None else {}),
                    },
                )
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            accepted += 1
            seen_by_stratum[stratum] += 1
            _reservoir_add(
                strata[stratum],
                example,
                seen=seen_by_stratum[stratum],
                capacity=capacity,
                rng=rng,
            )
    examples = [item for name in sorted(strata) for item in strata[name]]
    rng.shuffle(examples)
    return examples[: config.target_examples], {
        "rows_read": rows_read,
        "accepted": accepted,
        "invalid": invalid,
        "filtered": 0,
    }


def _base_seconds(raw: str) -> int:
    return int(raw.strip().split("+", maxsplit=1)[0])


def _import_games(config: KaggleImportConfig) -> tuple[list[TrainingExample], dict[str, int]]:
    rng = random.Random(config.seed)
    reservoir: list[TrainingExample] = []
    seen_examples = rows_read = accepted = invalid = filtered = 0
    csv.field_size_limit(16 * 1024 * 1024)
    with config.source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Result", "WhiteElo", "BlackElo", "TimeControl", "Termination", "AN"}
        if not required.issubset(reader.fieldnames or []):
            raise KaggleImportError(f"Expected CSV columns {sorted(required)}")
        for row_index, row in enumerate(reader, start=1):
            if config.max_source_rows is not None and rows_read >= config.max_source_rows:
                break
            rows_read += 1
            try:
                result = row["Result"].strip()
                if (
                    result not in {"1-0", "0-1", "1/2-1/2"}
                    or int(row["WhiteElo"]) < config.minimum_elo
                    or int(row["BlackElo"]) < config.minimum_elo
                    or _base_seconds(row["TimeControl"]) < config.minimum_base_seconds
                    or row["Termination"].strip().lower() != "normal"
                ):
                    filtered += 1
                    continue
                game = chess.pgn.read_game(io.StringIO(f'[Result "{result}"]\n\n{row["AN"]}'))
                if game is None or game.errors:
                    raise ValueError("invalid PGN movetext")
                board = game.board()
                positions: list[tuple[chess.Board, chess.Move, int]] = []
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    if move not in board.legal_moves:
                        raise ValueError("illegal move")
                    if config.maximum_ply is None or ply <= config.maximum_ply:
                        positions.append((board.copy(stack=False), move, ply))
                    board.push(move)
                if not positions:
                    raise ValueError("empty game")
                selected = sorted(
                    rng.sample(positions, k=min(config.positions_per_game, len(positions))),
                    key=lambda item: item[2],
                )
            except (KeyError, TypeError, ValueError, chess.IllegalMoveError):
                invalid += 1
                continue
            accepted += 1
            game_id = f"kaggle-lichess-{row_index}"
            for board_at_move, move, ply in selected:
                example = TrainingExample(
                    board_tensor=encode_board(board_at_move),
                    target_policy=_one_hot_policy(move),
                    target_value=result_value_for_turn(result, board_at_move.turn),
                    metadata={
                        "game_id": game_id,
                        "source": "arevel/chess-games",
                        "source_row": row_index,
                        "fen": board_at_move.fen(),
                        "move_uci": move.uci(),
                        "ply": ply,
                        "result": result,
                        "white_elo": int(row["WhiteElo"]),
                        "black_elo": int(row["BlackElo"]),
                        "target_kind": "policy_value",
                    },
                )
                seen_examples += 1
                _reservoir_add(
                    reservoir,
                    example,
                    seen=seen_examples,
                    capacity=config.target_examples,
                    rng=rng,
                )
    rng.shuffle(reservoir)
    return reservoir, {
        "rows_read": rows_read,
        "accepted": accepted,
        "invalid": invalid,
        "filtered": filtered,
    }


def import_kaggle_dataset(config: KaggleImportConfig) -> KaggleImportSummary:
    """Validate, sample, and convert one explicitly selected local CSV."""

    if not config.source_path.is_file():
        raise KaggleImportError(f"Source CSV does not exist: {config.source_path}")
    if config.output_path.exists():
        raise KaggleImportError(f"Output dataset already exists: {config.output_path}")
    if config.kind == "games":
        examples, counts = _import_games(config)
    else:
        examples, counts = _import_positions(config)
    if not examples:
        raise KaggleImportError("No valid examples passed the configured filters")
    save_dataset(
        config.output_path,
        examples,
        metadata={
            "kind": "explicit_external_kaggle_import",
            "import_format": KAGGLE_IMPORT_FORMAT,
            "import_version": KAGGLE_IMPORT_VERSION,
            "source_path": str(config.source_path),
            "source_dataset": (
                "arevel/chess-games" if config.kind == "games" else "ronakbadhe/chess-evaluations"
            ),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "examples": len(examples),
            **counts,
        },
    )
    return KaggleImportSummary(
        kind=config.kind,
        source_path=config.source_path,
        output_path=config.output_path,
        source_rows_read=counts["rows_read"],
        accepted_rows=counts["accepted"],
        examples=len(examples),
        invalid_rows=counts["invalid"],
        filtered_rows=counts["filtered"],
    )
