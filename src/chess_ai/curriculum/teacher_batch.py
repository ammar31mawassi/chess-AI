"""Offline paired-game collection with a stronger minimax teacher on every ply."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt
import torch

from chess_ai.agents.minimax_agent import MinimaxAgent, MinimaxMoveScore
from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.agents.protocol import ChessAgent
from chess_ai.arena.match import MatchResult, run_match
from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.games import save_pgn

LOGGER = logging.getLogger(__name__)

TEACHER_BATCH_FORMAT = "self-improving-chess-ai.offline-teacher-trace-batch"
TEACHER_BATCH_VERSION = 1


class TeacherBatchError(RuntimeError):
    """Raised when an offline teacher batch cannot be collected safely."""


@dataclass(frozen=True, slots=True)
class TeacherBatchConfig:
    """Settings for generation-only champion/D1 games with D2 annotations."""

    champion_checkpoint: Path
    state_path: Path
    manifest_path: Path
    dataset_path: Path
    pair_dataset_dir: Path
    pgn_dir: Path
    opening_pairs: int = 250
    opponent_depth: int = 1
    teacher_depth: int = 2
    max_plies: int = 200
    opening_min_full_moves: int = 2
    opening_max_full_moves: int = 3
    teacher_policy_temperature: float = 100.0
    teacher_policy_top_k: int = 5
    champion_win_weight: float = 2.0
    champion_draw_weight: float = 1.0
    champion_loss_weight: float = 1.0
    seed: int = 2026
    device: str = "auto"
    resume: bool = True

    def __post_init__(self) -> None:
        for name in (
            "champion_checkpoint",
            "state_path",
            "manifest_path",
            "dataset_path",
            "pair_dataset_dir",
            "pgn_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "opening_pairs",
            "opponent_depth",
            "teacher_depth",
            "max_plies",
            "teacher_policy_top_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("opening_min_full_moves", "opening_max_full_moves"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.opening_max_full_moves < self.opening_min_full_moves:
            raise ValueError("opening full-move bounds must satisfy minimum <= maximum")
        if self.opening_max_full_moves * 2 >= self.max_plies:
            raise ValueError("the longest opening must be shorter than max_plies")
        if self.teacher_depth <= self.opponent_depth:
            raise ValueError("teacher_depth must be greater than opponent_depth")
        if not math.isfinite(self.teacher_policy_temperature) or (
            self.teacher_policy_temperature <= 0.0
        ):
            raise ValueError("teacher_policy_temperature must be finite and positive")
        for name in (
            "champion_win_weight",
            "champion_draw_weight",
            "champion_loss_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.resume, bool):
            raise ValueError("resume must be a boolean")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        seed: int = 2026,
        device: str = "auto",
        resume: bool | None = None,
    ) -> TeacherBatchConfig:
        values = dict(raw)
        values.setdefault("seed", seed)
        values.setdefault("device", device)
        if resume is not None:
            values["resume"] = resume
        path_fields = {
            "champion_checkpoint",
            "state_path",
            "manifest_path",
            "dataset_path",
            "pair_dataset_dir",
            "pgn_dir",
        }
        for name in path_fields:
            if name in values:
                values[name] = Path(str(values[name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown teacher_batch settings: {', '.join(unknown)}")
        missing = sorted(path_fields.difference(values))
        if missing:
            raise ValueError(f"Missing teacher_batch settings: {', '.join(missing)}")
        return cls(**values)

    def signature_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("resume", None)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True, slots=True)
class TeacherBatchSummary:
    """Completed collection counts and frozen artifact locations."""

    requested_opening_pairs: int
    completed_opening_pairs: int
    games: int
    examples: int
    champion_wins: int
    champion_draws: int
    champion_losses: int
    teacher_agreements: int
    teacher_disagreements: int
    champion_checkpoint: Path
    dataset_path: Path
    manifest_path: Path
    state_path: Path
    pgn_dir: Path
    resumed: bool

    @property
    def teacher_agreement_rate(self) -> float:
        total = self.teacher_agreements + self.teacher_disagreements
        return self.teacher_agreements / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "champion_checkpoint",
            "dataset_path",
            "manifest_path",
            "state_path",
            "pgn_dir",
        ):
            payload[key] = str(payload[key])
        payload["teacher_agreement_rate"] = self.teacher_agreement_rate
        payload["training_updates"] = 0
        return payload


@dataclass(slots=True)
class _PendingTrace:
    board_tensor: npt.NDArray[np.float32]
    target_policy: npt.NDArray[np.float32]
    turn: chess.Color
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TeacherBatchError(f"Could not hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


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


def _signature(config: TeacherBatchConfig) -> str:
    encoded = json.dumps(config.signature_payload(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _pair_id(config: TeacherBatchConfig, pair_index: int) -> str:
    return f"teacher-batch-{_signature(config)[:10]}-pair-{pair_index:04d}"


def _game_id(config: TeacherBatchConfig, pair_index: int, champion_color: chess.Color) -> str:
    color_name = "white" if champion_color == chess.WHITE else "black"
    return f"{_pair_id(config, pair_index)}-champion-{color_name}"


def _pair_dataset_path(config: TeacherBatchConfig, pair_index: int) -> Path:
    return config.pair_dataset_dir / f"pair_{pair_index:04d}.pt"


def _game_number(pair_index: int, champion_color: chess.Color) -> int:
    return (pair_index - 1) * 2 + (1 if champion_color == chess.WHITE else 2)


def _game_pgn_path(
    config: TeacherBatchConfig,
    pair_index: int,
    champion_color: chess.Color,
) -> Path:
    color_name = "white" if champion_color == chess.WHITE else "black"
    return config.pgn_dir / (
        f"game_{_game_number(pair_index, champion_color):04d}_champion_{color_name}.pgn"
    )


def _opening_board(
    config: TeacherBatchConfig,
    pair_index: int,
) -> tuple[chess.Board, list[str], int]:
    base_seed = config.seed + pair_index * 1_000_003
    choices = list(range(config.opening_min_full_moves, config.opening_max_full_moves + 1))
    for attempt in range(100):
        opening_seed = base_seed + attempt
        rng = random.Random(opening_seed)
        requested_plies = rng.choice(choices) * 2
        board = chess.Board()
        moves: list[str] = []
        for _ in range(requested_plies):
            legal_moves = sorted(board.legal_moves, key=lambda move: move.uci())
            move = rng.choice(legal_moves)
            board.push(move)
            moves.append(move.uci())
            if board.is_game_over(claim_draw=True):
                break
        if len(moves) == requested_plies and not board.is_game_over(claim_draw=True):
            if board.turn != chess.WHITE:  # every configured opening length is a full move
                raise TeacherBatchError("An even-ply opening unexpectedly left Black to move")
            return board, moves, opening_seed
    raise TeacherBatchError("Could not create a non-terminal random opening after 100 attempts")


def _soft_teacher_policy(
    analysis: Sequence[MinimaxMoveScore],
    *,
    top_k: int,
    temperature: float,
    move_encoder: MoveEncoder,
) -> tuple[npt.NDArray[np.float32], MinimaxMoveScore, list[dict[str, Any]]]:
    if not analysis:
        raise TeacherBatchError("The D2 teacher returned no legal move scores")
    indexed = list(enumerate(analysis))
    ranked = [
        item
        for _index, item in sorted(
            indexed,
            key=lambda entry: (-entry[1].score, entry[0]),
        )
    ]
    selected = ranked[: min(top_k, len(ranked))]
    scores = np.asarray([item.score for item in selected], dtype=np.float64)
    scaled = (scores - float(scores.max())) / temperature
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    policy = np.zeros(move_encoder.action_size, dtype=np.float32)
    trace: list[dict[str, Any]] = []
    for item, probability in zip(selected, probabilities, strict=True):
        action = move_encoder.encode(item.move)
        policy[action] = float(probability)
        trace.append(
            {
                "move_uci": item.move.uci(),
                "score": item.score,
                "probability": float(probability),
            }
        )
    # Float32 rounding can leave the dense policy microscopically off one.
    policy /= policy.sum(dtype=np.float64)
    return policy, selected[0], trace


class _TeacherTraceAgent:
    """Play through a delegate while D2 independently labels the position."""

    def __init__(
        self,
        delegate: ChessAgent,
        teacher: MinimaxAgent,
        pending: list[_PendingTrace],
        *,
        config: TeacherBatchConfig,
        game_id: str,
        opening_pair_id: str,
        pair_index: int,
        game_number: int,
        actor_role: str,
        champion_color: chess.Color,
        opening_seed: int,
        opening_plies: int,
        champion_sha256: str,
    ) -> None:
        self.delegate = delegate
        self.teacher = teacher
        self.pending = pending
        self.config = config
        self.game_id = game_id
        self.opening_pair_id = opening_pair_id
        self.pair_index = pair_index
        self.game_number = game_number
        self.actor_role = actor_role
        self.champion_color = champion_color
        self.opening_seed = opening_seed
        self.opening_plies = opening_plies
        self.champion_sha256 = champion_sha256
        self.name = delegate.name
        self._boards = BoardEncoder()
        self._moves = MoveEncoder()

    def choose_move(self, board: chess.Board) -> chess.Move:
        analysis = self.teacher.analyze_moves(board.copy(stack=True))
        target_policy, teacher_best, policy_trace = _soft_teacher_policy(
            analysis,
            top_k=self.config.teacher_policy_top_k,
            temperature=self.config.teacher_policy_temperature,
            move_encoder=self._moves,
        )
        actual_move = self.delegate.choose_move(board.copy(stack=True))
        continuation_ply = len(self.pending) + 1
        best_action = self._moves.encode(teacher_best.move)
        self.pending.append(
            _PendingTrace(
                board_tensor=self._boards.encode(board),
                target_policy=target_policy,
                turn=board.turn,
                metadata={
                    "game_id": self.game_id,
                    "opening_pair_id": self.opening_pair_id,
                    "pair_index": self.pair_index,
                    "game_number": self.game_number,
                    "continuation_ply": continuation_ply,
                    "absolute_ply": self.opening_plies + continuation_ply,
                    "fen": board.fen(),
                    "player_to_move": "white" if board.turn == chess.WHITE else "black",
                    "actor_role": self.actor_role,
                    "actor_color": "white" if board.turn == chess.WHITE else "black",
                    "champion_color": ("white" if self.champion_color == chess.WHITE else "black"),
                    "champion_checkpoint": str(self.config.champion_checkpoint),
                    "champion_sha256": self.champion_sha256,
                    "actual_move_uci": actual_move.uci(),
                    "teacher_move_uci": teacher_best.move.uci(),
                    "move_uci": teacher_best.move.uci(),
                    "policy_action": best_action,
                    "actual_matched_teacher": actual_move == teacher_best.move,
                    "policy_source_agent": "minimax_teacher_soft_top_k",
                    "target_policy_kind": "softmax_exact_root_scores",
                    "teacher_depth": self.config.teacher_depth,
                    "teacher_score": teacher_best.score,
                    "teacher_nodes": self.teacher.nodes_searched,
                    "teacher_policy_temperature": self.config.teacher_policy_temperature,
                    "teacher_policy_top_k": self.config.teacher_policy_top_k,
                    "teacher_policy": policy_trace,
                    "opponent_depth": self.config.opponent_depth,
                    "opening_seed": self.opening_seed,
                    "opening_plies": self.opening_plies,
                },
            )
        )
        return actual_move


def _champion_outcome(result: str, champion_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result not in {"1-0", "0-1"}:
        raise TeacherBatchError(f"Cannot finalize training traces from result {result!r}")
    champion_won = (result == "1-0") == (champion_color == chess.WHITE)
    return "win" if champion_won else "loss"


def _outcome_weight(config: TeacherBatchConfig, outcome: str) -> float:
    return {
        "win": config.champion_win_weight,
        "draw": config.champion_draw_weight,
        "loss": config.champion_loss_weight,
    }[outcome]


def _finalize_pending(
    pending: Sequence[_PendingTrace],
    *,
    config: TeacherBatchConfig,
    result: str,
    termination: str,
    champion_color: chess.Color,
) -> list[TrainingExample]:
    champion_outcome = _champion_outcome(result, champion_color)
    training_weight = _outcome_weight(config, champion_outcome)
    return [
        TrainingExample(
            board_tensor=item.board_tensor,
            target_policy=item.target_policy,
            target_value=result_value_for_turn(result, item.turn),
            metadata={
                **item.metadata,
                "result": result,
                "termination": termination,
                "champion_outcome": champion_outcome,
                "champion_outcome_weight": training_weight,
            },
        )
        for item in pending
    ]


def _save_complete_game_pgn(
    config: TeacherBatchConfig,
    result: MatchResult,
    *,
    opening_board: chess.Board,
    opening_moves: Sequence[str],
    opening_seed: int,
    pair_index: int,
    champion_color: chess.Color,
    game_id: str,
    champion_sha256: str,
    teacher_labels: int,
) -> Path:
    complete_board = opening_board.copy(stack=True)
    for record in result.moves:
        move = chess.Move.from_uci(record.uci)
        if move not in complete_board.legal_moves:
            raise TeacherBatchError(
                f"Cannot archive {game_id}: continuation move {move.uci()} is illegal"
            )
        complete_board.push(move)
    if complete_board.fen() != result.final_fen:
        raise TeacherBatchError(f"Reconstructed complete PGN does not match {game_id} final FEN")
    game = chess.pgn.Game.from_board(complete_board)
    color_name = "white" if champion_color == chess.WHITE else "black"
    game.headers["Event"] = "Offline Champion D1 Game with D2 Teacher Trace"
    game.headers["Site"] = "Local"
    game.headers["White"] = result.white_name
    game.headers["Black"] = result.black_name
    game.headers["Result"] = result.result
    game.headers["Termination"] = result.termination
    game.headers["GameID"] = game_id
    game.headers["TeacherBatchFormat"] = TEACHER_BATCH_FORMAT
    game.headers["TeacherBatchVersion"] = str(TEACHER_BATCH_VERSION)
    game.headers["OpeningPair"] = str(pair_index)
    game.headers["ChampionColor"] = color_name
    game.headers["ChampionCheckpoint"] = str(config.champion_checkpoint)
    game.headers["ChampionSHA256"] = champion_sha256
    game.headers["OpponentDepth"] = str(config.opponent_depth)
    game.headers["TeacherDepth"] = str(config.teacher_depth)
    game.headers["OpeningSeed"] = str(opening_seed)
    game.headers["RandomOpeningPlies"] = str(len(opening_moves))
    game.headers["RandomOpeningMoves"] = " ".join(opening_moves)
    game.headers["TeacherLabels"] = str(teacher_labels)
    game.headers["TrainingDuringCollection"] = "false"
    return save_pgn(game, _game_pgn_path(config, pair_index, champion_color))


def _play_game(
    config: TeacherBatchConfig,
    champion: NeuralAgent,
    champion_sha256: str,
    *,
    pair_index: int,
    champion_color: chess.Color,
    opening_board: chess.Board,
    opening_moves: Sequence[str],
    opening_seed: int,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    pair_id = _pair_id(config, pair_index)
    game_id = _game_id(config, pair_index, champion_color)
    number = _game_number(pair_index, champion_color)
    pending: list[_PendingTrace] = []
    teacher = MinimaxAgent(
        depth=config.teacher_depth,
        deterministic=True,
        seed=config.seed + number * 17,
        name=f"Minimax D{config.teacher_depth} teacher",
    )
    opponent = MinimaxAgent(
        depth=config.opponent_depth,
        deterministic=True,
        seed=config.seed + number * 31,
        name=f"Minimax D{config.opponent_depth} opponent",
    )
    traced_champion = _TeacherTraceAgent(
        champion,
        teacher,
        pending,
        config=config,
        game_id=game_id,
        opening_pair_id=pair_id,
        pair_index=pair_index,
        game_number=number,
        actor_role="champion",
        champion_color=champion_color,
        opening_seed=opening_seed,
        opening_plies=len(opening_moves),
        champion_sha256=champion_sha256,
    )
    traced_opponent = _TeacherTraceAgent(
        opponent,
        teacher,
        pending,
        config=config,
        game_id=game_id,
        opening_pair_id=pair_id,
        pair_index=pair_index,
        game_number=number,
        actor_role="minimax_d1_opponent",
        champion_color=champion_color,
        opening_seed=opening_seed,
        opening_plies=len(opening_moves),
        champion_sha256=champion_sha256,
    )
    white = traced_champion if champion_color == chess.WHITE else traced_opponent
    black = traced_opponent if champion_color == chess.WHITE else traced_champion
    result = run_match(
        white,
        black,
        max_plies=config.max_plies - len(opening_moves),
        starting_fen=opening_board.fen(),
        seed=config.seed + number,
    )
    if result.illegal_agent is not None:
        raise TeacherBatchError(
            f"Generation agent {result.illegal_agent!r} produced illegal move "
            f"{result.illegal_move!r} in {game_id}"
        )
    examples = _finalize_pending(
        pending,
        config=config,
        result=result.result,
        termination=result.termination,
        champion_color=champion_color,
    )
    pgn_path = _save_complete_game_pgn(
        config,
        result,
        opening_board=opening_board,
        opening_moves=opening_moves,
        opening_seed=opening_seed,
        pair_index=pair_index,
        champion_color=champion_color,
        game_id=game_id,
        champion_sha256=champion_sha256,
        teacher_labels=len(examples),
    )
    outcome = _champion_outcome(result.result, champion_color)
    agreements = sum(bool(item.metadata["actual_matched_teacher"]) for item in examples)
    actor_labels = {
        role: sum(item.metadata["actor_role"] == role for item in examples)
        for role in ("champion", "minimax_d1_opponent")
    }
    return examples, {
        "game_id": game_id,
        "game_number": number,
        "champion_color": "white" if champion_color == chess.WHITE else "black",
        "result": result.result,
        "champion_outcome": outcome,
        "termination": result.termination,
        "total_plies": len(opening_moves) + result.plies,
        "teacher_labels": len(examples),
        "teacher_agreements": agreements,
        "teacher_disagreements": len(examples) - agreements,
        "actor_labels": actor_labels,
        "pgn_path": str(pgn_path),
        "pgn_sha256": _sha256_file(pgn_path),
    }


def _play_pair(
    config: TeacherBatchConfig,
    champion: NeuralAgent,
    champion_sha256: str,
    pair_index: int,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    opening_board, opening_moves, opening_seed = _opening_board(config, pair_index)
    examples: list[TrainingExample] = []
    games: list[dict[str, Any]] = []
    for champion_color in (chess.WHITE, chess.BLACK):
        game_examples, game = _play_game(
            config,
            champion,
            champion_sha256,
            pair_index=pair_index,
            champion_color=champion_color,
            opening_board=opening_board,
            opening_moves=opening_moves,
            opening_seed=opening_seed,
        )
        examples.extend(game_examples)
        games.append(game)

    pair_path = _pair_dataset_path(config, pair_index)
    save_dataset(
        pair_path,
        examples,
        metadata={
            "kind": "offline_teacher_trace_pair",
            "teacher_batch_format": TEACHER_BATCH_FORMAT,
            "teacher_batch_version": TEACHER_BATCH_VERSION,
            "config_signature": _signature(config),
            "pair_index": pair_index,
            "opening_pair_id": _pair_id(config, pair_index),
            "opening_seed": opening_seed,
            "opening_moves": list(opening_moves),
            "opening_fen": opening_board.fen(),
            "games": 2,
            "examples": len(examples),
        },
    )
    wins = sum(game["champion_outcome"] == "win" for game in games)
    draws = sum(game["champion_outcome"] == "draw" for game in games)
    losses = sum(game["champion_outcome"] == "loss" for game in games)
    return examples, {
        "pair_index": pair_index,
        "opening_pair_id": _pair_id(config, pair_index),
        "opening_seed": opening_seed,
        "opening_moves": list(opening_moves),
        "opening_fen": opening_board.fen(),
        "examples": len(examples),
        "champion_wins": wins,
        "champion_draws": draws,
        "champion_losses": losses,
        "teacher_agreements": sum(game["teacher_agreements"] for game in games),
        "teacher_disagreements": sum(game["teacher_disagreements"] for game in games),
        "games": games,
        "pair_dataset_path": str(pair_path),
        "pair_dataset_sha256": _sha256_file(pair_path),
    }


def _initial_state(config: TeacherBatchConfig, champion_sha256: str) -> dict[str, Any]:
    return {
        "format": TEACHER_BATCH_FORMAT,
        "version": TEACHER_BATCH_VERSION,
        "config_signature": _signature(config),
        "champion_sha256": champion_sha256,
        "status": "collecting",
        "completed_opening_pairs": 0,
        "history": [],
        "examples": 0,
        "champion_wins": 0,
        "champion_draws": 0,
        "champion_losses": 0,
        "teacher_agreements": 0,
        "teacher_disagreements": 0,
        "training_updates": 0,
    }


def _existing_output_conflicts(config: TeacherBatchConfig) -> list[Path]:
    conflicts = [path for path in (config.dataset_path, config.manifest_path) if path.exists()]
    if config.pair_dataset_dir.exists():
        conflicts.extend(sorted(config.pair_dataset_dir.glob("pair_*.pt")))
    if config.pgn_dir.exists():
        conflicts.extend(sorted(config.pgn_dir.glob("game_*.pgn")))
    return conflicts


def _load_or_create_state(
    config: TeacherBatchConfig,
    champion_sha256: str,
) -> tuple[dict[str, Any], bool]:
    if not config.state_path.exists():
        conflicts = _existing_output_conflicts(config)
        if conflicts:
            preview = ", ".join(str(path) for path in conflicts[:3])
            raise TeacherBatchError(
                "Teacher-batch outputs exist without their state file: "
                f"{preview}. Restore the matching state or choose fresh output paths."
            )
        state = _initial_state(config, champion_sha256)
        _atomic_json(config.state_path, state)
        return state, False
    if not config.resume:
        raise TeacherBatchError(
            f"Teacher-batch state exists at {config.state_path}; enable resume or use fresh paths."
        )
    try:
        raw = json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherBatchError(f"Could not read teacher-batch state: {exc}") from exc
    if not isinstance(raw, dict):
        raise TeacherBatchError("Teacher-batch state root must be a mapping")
    if raw.get("format") != TEACHER_BATCH_FORMAT or (raw.get("version") != TEACHER_BATCH_VERSION):
        raise TeacherBatchError("Teacher-batch state format/version is incompatible")
    if raw.get("config_signature") != _signature(config):
        raise TeacherBatchError(
            "Teacher-batch configuration differs from saved state; restore it or use fresh paths."
        )
    if raw.get("champion_sha256") != champion_sha256:
        raise TeacherBatchError("The frozen champion checkpoint changed after collection began")
    return raw, True


def _validate_committed_state(config: TeacherBatchConfig, state: Mapping[str, Any]) -> None:
    completed = state.get("completed_opening_pairs")
    history = state.get("history")
    if not isinstance(completed, int) or completed < 0 or completed > config.opening_pairs:
        raise TeacherBatchError("Saved completed_opening_pairs is invalid")
    if not isinstance(history, list) or len(history) != completed:
        raise TeacherBatchError("Saved teacher-batch history does not match its pair count")
    if state.get("training_updates") != 0:
        raise TeacherBatchError("Generation-only state unexpectedly records a training update")
    for expected_pair, item in enumerate(history, start=1):
        if not isinstance(item, dict) or item.get("pair_index") != expected_pair:
            raise TeacherBatchError("Saved teacher-batch history has an invalid pair entry")
        pair_path = Path(str(item.get("pair_dataset_path", "")))
        if not pair_path.is_file() or _sha256_file(pair_path) != item.get("pair_dataset_sha256"):
            raise TeacherBatchError(f"Committed pair dataset is missing or changed: {pair_path}")
        games = item.get("games")
        if not isinstance(games, list) or len(games) != 2:
            raise TeacherBatchError(f"Committed pair {expected_pair} does not contain two games")
        for game in games:
            if not isinstance(game, dict):
                raise TeacherBatchError("Saved teacher-batch game entry is invalid")
            pgn_path = Path(str(game.get("pgn_path", "")))
            if not pgn_path.is_file() or _sha256_file(pgn_path) != game.get("pgn_sha256"):
                raise TeacherBatchError(f"Committed PGN is missing or changed: {pgn_path}")
    status = state.get("status")
    if status not in {"collecting", "complete"}:
        raise TeacherBatchError(f"Saved teacher-batch status is invalid: {status!r}")
    if status == "complete":
        if completed != config.opening_pairs:
            raise TeacherBatchError("A complete teacher batch has the wrong pair count")
        for path, digest_key in (
            (config.dataset_path, "dataset_sha256"),
            (config.manifest_path, "manifest_sha256"),
        ):
            if not path.is_file() or _sha256_file(path) != state.get(digest_key):
                raise TeacherBatchError(f"Completed teacher-batch artifact is missing: {path}")


def _aggregate_history(history: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "examples",
        "champion_wins",
        "champion_draws",
        "champion_losses",
        "teacher_agreements",
        "teacher_disagreements",
    )
    return {key: sum(int(item[key]) for item in history) for key in keys}


def _finalize_batch(
    config: TeacherBatchConfig,
    state: dict[str, Any],
    champion_sha256: str,
) -> None:
    all_examples: list[TrainingExample] = []
    for pair_index in range(1, config.opening_pairs + 1):
        loaded = load_dataset(_pair_dataset_path(config, pair_index))
        if loaded.metadata.get("config_signature") != _signature(config) or (
            loaded.metadata.get("pair_index") != pair_index
        ):
            raise TeacherBatchError(f"Pair dataset {pair_index} has incompatible provenance")
        all_examples.extend(loaded.examples)
    aggregates = _aggregate_history(state["history"])
    if len(all_examples) != aggregates["examples"]:
        raise TeacherBatchError("Pair shards do not match the saved total example count")
    save_dataset(
        config.dataset_path,
        all_examples,
        metadata={
            "kind": "offline_teacher_trace_batch",
            "teacher_batch_format": TEACHER_BATCH_FORMAT,
            "teacher_batch_version": TEACHER_BATCH_VERSION,
            "config_signature": _signature(config),
            "champion_checkpoint": str(config.champion_checkpoint),
            "champion_sha256": champion_sha256,
            "opening_pairs": config.opening_pairs,
            "games": config.opening_pairs * 2,
            "examples": len(all_examples),
            "teacher_scope": "every_played_ply_after_random_opening",
            "teacher_target": "softmax_exact_root_scores",
            "teacher_policy_top_k": config.teacher_policy_top_k,
            "teacher_policy_temperature": config.teacher_policy_temperature,
            "outcome_weight_key": "champion_outcome_weight",
            "champion_win_weight": config.champion_win_weight,
            "champion_draw_weight": config.champion_draw_weight,
            "champion_loss_weight": config.champion_loss_weight,
            "training_during_collection": False,
            **aggregates,
        },
    )
    dataset_sha256 = _sha256_file(config.dataset_path)
    manifest = {
        "format": TEACHER_BATCH_FORMAT,
        "version": TEACHER_BATCH_VERSION,
        "config_signature": _signature(config),
        "config": config.signature_payload(),
        "champion_sha256": champion_sha256,
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": dataset_sha256,
        "training_updates": 0,
        "aggregates": aggregates,
        "pairs": state["history"],
    }
    _atomic_json(config.manifest_path, manifest)
    state.update(aggregates)
    state["dataset_sha256"] = dataset_sha256
    state["manifest_sha256"] = _sha256_file(config.manifest_path)
    state["status"] = "complete"
    _atomic_json(config.state_path, state)


def _summary(
    config: TeacherBatchConfig,
    state: Mapping[str, Any],
    *,
    resumed: bool,
) -> TeacherBatchSummary:
    return TeacherBatchSummary(
        requested_opening_pairs=config.opening_pairs,
        completed_opening_pairs=int(state["completed_opening_pairs"]),
        games=int(state["completed_opening_pairs"]) * 2,
        examples=int(state["examples"]),
        champion_wins=int(state["champion_wins"]),
        champion_draws=int(state["champion_draws"]),
        champion_losses=int(state["champion_losses"]),
        teacher_agreements=int(state["teacher_agreements"]),
        teacher_disagreements=int(state["teacher_disagreements"]),
        champion_checkpoint=config.champion_checkpoint,
        dataset_path=config.dataset_path,
        manifest_path=config.manifest_path,
        state_path=config.state_path,
        pgn_dir=config.pgn_dir,
        resumed=resumed,
    )


def run_teacher_batch(config: TeacherBatchConfig) -> TeacherBatchSummary:
    """Generate all games and D2 targets without constructing an optimizer."""

    if not config.champion_checkpoint.is_file():
        raise TeacherBatchError(f"Champion checkpoint does not exist: {config.champion_checkpoint}")
    distinct_files = {
        config.champion_checkpoint.resolve(),
        config.state_path.resolve(),
        config.manifest_path.resolve(),
        config.dataset_path.resolve(),
    }
    if len(distinct_files) != 4:
        raise TeacherBatchError("Champion, state, manifest, and dataset paths must be distinct")
    if config.pair_dataset_dir.resolve() == config.pgn_dir.resolve():
        raise TeacherBatchError("pair_dataset_dir and pgn_dir must be different directories")
    load_checkpoint(config.champion_checkpoint, map_location="cpu")
    champion_sha256 = _sha256_file(config.champion_checkpoint)
    state, resumed = _load_or_create_state(config, champion_sha256)
    _validate_committed_state(config, state)
    if state["status"] == "complete":
        return _summary(config, state, resumed=resumed)

    champion = NeuralAgent(
        config.champion_checkpoint,
        device=config.device,
        deterministic=True,
        temperature=0.0,
        seed=config.seed,
        name=f"Gameplay champion ({config.champion_checkpoint.name})",
    )
    try:
        while int(state["completed_opening_pairs"]) < config.opening_pairs:
            pair_index = int(state["completed_opening_pairs"]) + 1
            _examples, pair_summary = _play_pair(
                config,
                champion,
                champion_sha256,
                pair_index,
            )
            history = list(state["history"])
            history.append(pair_summary)
            state["history"] = history
            state["completed_opening_pairs"] = pair_index
            state.update(_aggregate_history(history))
            _atomic_json(config.state_path, state)
            LOGGER.info(
                "teacher-batch pair=%d/%d games=%d examples=%d champion_W/D/L=%d/%d/%d",
                pair_index,
                config.opening_pairs,
                pair_index * 2,
                state["examples"],
                state["champion_wins"],
                state["champion_draws"],
                state["champion_losses"],
            )
    finally:
        del champion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _finalize_batch(config, state, champion_sha256)
    return _summary(config, state, resumed=resumed)
