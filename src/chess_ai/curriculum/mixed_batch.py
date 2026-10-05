"""Offline multi-cohort game collection with stronger minimax annotations."""

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
from typing import Any, Literal

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt
import torch

from chess_ai.agents.minimax_agent import MinimaxAgent, MinimaxMoveScore
from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.agents.protocol import ChessAgent
from chess_ai.arena.match import MatchResult, run_match
from chess_ai.curriculum.teacher_batch import _soft_teacher_policy
from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.games import save_pgn

LOGGER = logging.getLogger(__name__)

MIXED_BATCH_FORMAT = "self-improving-chess-ai.mixed-offline-teacher-batch"
MIXED_BATCH_VERSION = 1


class MixedBatchError(RuntimeError):
    """Raised when the mixed offline collection cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class MixedBatchConfig:
    """Settings for a fixed neural mix or three-depth minimax league."""

    neural_checkpoint: Path | None
    state_path: Path
    manifest_path: Path
    dataset_path: Path
    opening_dataset_dir: Path
    pgn_dir: Path
    opening_positions: int = 50
    experiment: Literal["neural_mix", "minimax_league"] = "neural_mix"
    minimax_d1_depth: int = 1
    minimax_d2_depth: int = 2
    minimax_d3_depth: int = 3
    teacher_depth: int = 2
    max_plies: int = 200
    opening_min_full_moves: int = 2
    opening_max_full_moves: int = 3
    normal_start_every_positions: int = 0
    teacher_policy_temperature: float = 100.0
    teacher_policy_top_k: int = 5
    win_weight: float = 2.0
    draw_weight: float = 1.5
    loss_weight: float = 1.0
    minimax_random_tie_breaks: bool = True
    seed: int = 2026
    device: str = "auto"
    resume: bool = True

    def __post_init__(self) -> None:
        for name in (
            "state_path",
            "manifest_path",
            "dataset_path",
            "opening_dataset_dir",
            "pgn_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.neural_checkpoint is not None:
            object.__setattr__(self, "neural_checkpoint", Path(self.neural_checkpoint))
        for name in (
            "opening_positions",
            "minimax_d1_depth",
            "minimax_d2_depth",
            "minimax_d3_depth",
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
        if self.experiment not in {"neural_mix", "minimax_league"}:
            raise ValueError("experiment must be 'neural_mix' or 'minimax_league'")
        if self.experiment == "neural_mix" and self.neural_checkpoint is None:
            raise ValueError("neural_mix requires neural_checkpoint")
        if (
            isinstance(self.normal_start_every_positions, bool)
            or not isinstance(self.normal_start_every_positions, int)
            or self.normal_start_every_positions < 0
        ):
            raise ValueError("normal_start_every_positions must be a non-negative integer")
        if self.minimax_d2_depth < self.minimax_d1_depth:
            raise ValueError("minimax_d2_depth must be at least minimax_d1_depth")
        if self.minimax_d3_depth < self.minimax_d2_depth:
            raise ValueError("minimax_d3_depth must be at least minimax_d2_depth")
        deepest_actor = (
            self.minimax_d3_depth if self.experiment == "minimax_league" else self.minimax_d2_depth
        )
        if self.teacher_depth < deepest_actor:
            raise ValueError("teacher_depth must be at least the deepest configured actor")
        if not math.isfinite(self.teacher_policy_temperature) or (
            self.teacher_policy_temperature <= 0.0
        ):
            raise ValueError("teacher_policy_temperature must be finite and positive")
        for name in ("win_weight", "draw_weight", "loss_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.minimax_random_tie_breaks, bool):
            raise ValueError("minimax_random_tie_breaks must be a boolean")
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
    ) -> MixedBatchConfig:
        values = dict(raw)
        values.setdefault("seed", seed)
        values.setdefault("device", device)
        values.setdefault("neural_checkpoint", None)
        if resume is not None:
            values["resume"] = resume
        path_fields = {
            "state_path",
            "manifest_path",
            "dataset_path",
            "opening_dataset_dir",
            "pgn_dir",
        }
        for name in {*path_fields, "neural_checkpoint"}:
            if values.get(name) is not None:
                values[name] = Path(str(values[name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown mixed_batch settings: {', '.join(unknown)}")
        missing = sorted(path_fields.difference(values))
        if missing:
            raise ValueError(f"Missing mixed_batch settings: {', '.join(missing)}")
        return cls(**values)

    def signature_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("resume", None)
        # Preserve signatures created before the optional minimax-league mode
        # existed so completed neural-mix collections remain resumable.
        if self.experiment == "neural_mix":
            payload.pop("experiment", None)
            payload.pop("minimax_d3_depth", None)
            payload.pop("normal_start_every_positions", None)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True, slots=True)
class MixedBatchSummary:
    """Counts and artifacts produced by a mixed offline collection."""

    requested_opening_positions: int
    completed_opening_positions: int
    games: int
    examples: int
    teacher_agreements: int
    teacher_disagreements: int
    actor_win_examples: int
    actor_draw_examples: int
    actor_loss_examples: int
    cohort_results: dict[str, dict[str, int | float]]
    neural_checkpoint: Path | None
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
            "dataset_path",
            "manifest_path",
            "state_path",
            "pgn_dir",
        ):
            payload[key] = str(payload[key])
        payload["neural_checkpoint"] = (
            str(self.neural_checkpoint) if self.neural_checkpoint is not None else None
        )
        payload["teacher_agreement_rate"] = self.teacher_agreement_rate
        payload["training_updates"] = 0
        return payload


@dataclass(frozen=True, slots=True)
class _ActorSpec:
    role: str
    kind: Literal["neural", "minimax"]
    depth: int | None = None


@dataclass(frozen=True, slots=True)
class _Matchup:
    key: str
    agent_a: _ActorSpec
    agent_b: _ActorSpec


MATCHUPS: tuple[_Matchup, ...] = (
    _Matchup(
        "minimax_d1_vs_d2",
        _ActorSpec("minimax_d1", "minimax", 1),
        _ActorSpec("minimax_d2", "minimax", 2),
    ),
    _Matchup(
        "minimax_d2_vs_d2",
        _ActorSpec("minimax_d2_a", "minimax", 2),
        _ActorSpec("minimax_d2_b", "minimax", 2),
    ),
    _Matchup(
        "neural_vs_minimax_d1",
        _ActorSpec("neural", "neural"),
        _ActorSpec("minimax_d1", "minimax", 1),
    ),
    _Matchup(
        "neural_vs_minimax_d2",
        _ActorSpec("neural", "neural"),
        _ActorSpec("minimax_d2", "minimax", 2),
    ),
)

LEAGUE_MATCHUPS: tuple[_Matchup, ...] = (
    _Matchup(
        "minimax_d2_vs_d1",
        _ActorSpec("minimax_d2", "minimax", 2),
        _ActorSpec("minimax_d1", "minimax", 1),
    ),
    _Matchup(
        "minimax_d2_vs_d3",
        _ActorSpec("minimax_d2", "minimax", 2),
        _ActorSpec("minimax_d3", "minimax", 3),
    ),
    _Matchup(
        "minimax_d1_vs_d3",
        _ActorSpec("minimax_d1", "minimax", 1),
        _ActorSpec("minimax_d3", "minimax", 3),
    ),
)


def _matchups(config: MixedBatchConfig) -> tuple[_Matchup, ...]:
    return LEAGUE_MATCHUPS if config.experiment == "minimax_league" else MATCHUPS


@dataclass(slots=True)
class _PendingTrace:
    board_tensor: npt.NDArray[np.float32]
    target_policy: npt.NDArray[np.float32]
    turn: chess.Color
    metadata: dict[str, Any]


_TeacherAnalysis = tuple[
    npt.NDArray[np.float32],
    MinimaxMoveScore,
    list[dict[str, Any]],
    int,
    tuple[chess.Move, ...],
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MixedBatchError(f"Could not hash artifact {path}: {exc}") from exc
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


def _signature(config: MixedBatchConfig) -> str:
    encoded = json.dumps(config.signature_payload(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _opening_id(config: MixedBatchConfig, opening_index: int) -> str:
    return f"mixed-batch-{_signature(config)[:10]}-opening-{opening_index:04d}"


def _opening_dataset_path(config: MixedBatchConfig, opening_index: int) -> Path:
    return config.opening_dataset_dir / f"opening_{opening_index:04d}.pt"


def _opening_board(
    config: MixedBatchConfig,
    opening_index: int,
) -> tuple[chess.Board, list[str], int]:
    base_seed = config.seed + opening_index * 1_000_003
    if config.normal_start_every_positions and (
        opening_index % config.normal_start_every_positions == 0
    ):
        return chess.Board(), [], base_seed
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
            if board.turn != chess.WHITE:
                raise MixedBatchError("An even-ply opening unexpectedly left Black to move")
            return board, moves, opening_seed
    raise MixedBatchError("Could not create a non-terminal random opening after 100 attempts")


def _resolved_depth(config: MixedBatchConfig, spec: _ActorSpec) -> int | None:
    if spec.kind == "neural":
        return None
    if spec.depth == 1:
        return config.minimax_d1_depth
    if spec.depth == 2:
        return config.minimax_d2_depth
    if spec.depth == 3:
        return config.minimax_d3_depth
    raise MixedBatchError(f"Unsupported minimax actor depth marker: {spec.depth}")


def _make_actor(
    config: MixedBatchConfig,
    spec: _ActorSpec,
    *,
    seed: int,
    neural: NeuralAgent | None,
) -> ChessAgent:
    if spec.kind == "neural":
        if neural is None:
            raise MixedBatchError("A neural matchup requires a loaded neural checkpoint")
        return neural
    depth = _resolved_depth(config, spec)
    if depth is None:  # pragma: no cover - guarded by actor kind
        raise MixedBatchError("A minimax actor has no resolved depth")
    return MinimaxAgent(
        depth=depth,
        deterministic=not config.minimax_random_tie_breaks,
        seed=seed,
        name=f"{spec.role} (D{depth})",
    )


class _TeacherTraceAgent:
    """Play through a delegate while an independent D2 teacher labels the state."""

    def __init__(
        self,
        delegate: ChessAgent,
        teacher: MinimaxAgent,
        pending: list[_PendingTrace],
        *,
        config: MixedBatchConfig,
        game_id: str,
        opening_set_id: str,
        opening_index: int,
        matchup: _Matchup,
        actor: _ActorSpec,
        actor_slot: str,
        opening_seed: int,
        opening_plies: int,
        neural_sha256: str | None,
        teacher_cache: dict[str, _TeacherAnalysis],
        actor_seed: int,
    ) -> None:
        self.delegate = delegate
        self.teacher = teacher
        self.pending = pending
        self.config = config
        self.game_id = game_id
        self.opening_set_id = opening_set_id
        self.opening_index = opening_index
        self.matchup = matchup
        self.actor = actor
        self.actor_slot = actor_slot
        self.opening_seed = opening_seed
        self.opening_plies = opening_plies
        self.neural_sha256 = neural_sha256
        self.teacher_cache = teacher_cache
        self.actor_rng = random.Random(actor_seed)
        self.name = delegate.name
        self._boards = BoardEncoder()
        self._moves = MoveEncoder()

    def choose_move(self, board: chess.Board) -> chess.Move:
        position_key = board.fen()
        cached = self.teacher_cache.get(position_key)
        if cached is None:
            analysis = self.teacher.analyze_moves(board.copy(stack=True))
            target_policy, teacher_best, policy_trace = _soft_teacher_policy(
                analysis,
                top_k=self.config.teacher_policy_top_k,
                temperature=self.config.teacher_policy_temperature,
                move_encoder=self._moves,
            )
            teacher_nodes = self.teacher.nodes_searched
            best_moves = tuple(item.move for item in analysis if item.score == teacher_best.score)
            cached = (target_policy, teacher_best, policy_trace, teacher_nodes, best_moves)
            self.teacher_cache[position_key] = cached
        target_policy, teacher_best, policy_trace, teacher_nodes, best_moves = cached
        if self.actor.kind == "minimax" and (
            _resolved_depth(self.config, self.actor) == self.config.teacher_depth
        ):
            actual_move = (
                self.actor_rng.choice(best_moves)
                if self.config.minimax_random_tie_breaks
                else best_moves[0]
            )
        else:
            actual_move = self.delegate.choose_move(board.copy(stack=True))
        continuation_ply = len(self.pending) + 1
        self.pending.append(
            _PendingTrace(
                board_tensor=self._boards.encode(board),
                target_policy=target_policy,
                turn=board.turn,
                metadata={
                    "game_id": self.game_id,
                    "opening_set_id": self.opening_set_id,
                    "opening_index": self.opening_index,
                    "continuation_ply": continuation_ply,
                    "absolute_ply": self.opening_plies + continuation_ply,
                    "fen": board.fen(),
                    "player_to_move": "white" if board.turn == chess.WHITE else "black",
                    "matchup": self.matchup.key,
                    "actor_role": self.actor.role,
                    "actor_slot": self.actor_slot,
                    "actor_kind": self.actor.kind,
                    "actor_depth": _resolved_depth(self.config, self.actor),
                    "actor_color": "white" if board.turn == chess.WHITE else "black",
                    "neural_checkpoint": (
                        str(self.config.neural_checkpoint)
                        if self.config.neural_checkpoint is not None
                        else None
                    ),
                    "neural_sha256": self.neural_sha256,
                    "actual_move_uci": actual_move.uci(),
                    "teacher_move_uci": teacher_best.move.uci(),
                    "move_uci": teacher_best.move.uci(),
                    "policy_action": self._moves.encode(teacher_best.move),
                    "actual_matched_teacher": actual_move == teacher_best.move,
                    "policy_source_agent": "minimax_teacher_soft_top_k",
                    "target_policy_kind": "softmax_exact_root_scores",
                    "teacher_depth": self.config.teacher_depth,
                    "teacher_score": teacher_best.score,
                    "teacher_nodes": teacher_nodes,
                    "teacher_policy_temperature": self.config.teacher_policy_temperature,
                    "teacher_policy_top_k": self.config.teacher_policy_top_k,
                    "teacher_policy": policy_trace,
                    "opening_seed": self.opening_seed,
                    "opening_plies": self.opening_plies,
                },
            )
        )
        return actual_move


def _outcome_for_color(result: str, color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result not in {"1-0", "0-1"}:
        raise MixedBatchError(f"Cannot finalize training traces from result {result!r}")
    won = (result == "1-0") == (color == chess.WHITE)
    return "win" if won else "loss"


def _outcome_weight(config: MixedBatchConfig, outcome: str) -> float:
    return {
        "win": config.win_weight,
        "draw": config.draw_weight,
        "loss": config.loss_weight,
    }[outcome]


def _finalize_pending(
    pending: Sequence[_PendingTrace],
    *,
    config: MixedBatchConfig,
    result: str,
    termination: str,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for item in pending:
        actor_outcome = _outcome_for_color(result, item.turn)
        examples.append(
            TrainingExample(
                board_tensor=item.board_tensor,
                target_policy=item.target_policy,
                target_value=result_value_for_turn(result, item.turn),
                metadata={
                    **item.metadata,
                    "result": result,
                    "termination": termination,
                    "actor_outcome": actor_outcome,
                    "actor_outcome_weight": _outcome_weight(config, actor_outcome),
                },
            )
        )
    return examples


def _game_pgn_path(
    config: MixedBatchConfig,
    matchup: _Matchup,
    opening_index: int,
    agent_a_color: chess.Color,
) -> Path:
    color_name = "white" if agent_a_color == chess.WHITE else "black"
    return config.pgn_dir / matchup.key / (f"opening_{opening_index:04d}_agent_a_{color_name}.pgn")


def _save_complete_game_pgn(
    config: MixedBatchConfig,
    result: MatchResult,
    *,
    matchup: _Matchup,
    opening_board: chess.Board,
    opening_moves: Sequence[str],
    opening_seed: int,
    opening_index: int,
    agent_a_color: chess.Color,
    game_id: str,
    neural_sha256: str | None,
    teacher_labels: int,
) -> Path:
    complete_board = opening_board.copy(stack=True)
    for record in result.moves:
        move = chess.Move.from_uci(record.uci)
        if move not in complete_board.legal_moves:
            raise MixedBatchError(
                f"Cannot archive {game_id}: continuation move {move.uci()} is illegal"
            )
        complete_board.push(move)
    if complete_board.fen() != result.final_fen:
        raise MixedBatchError(f"Reconstructed complete PGN does not match {game_id} final FEN")
    game = chess.pgn.Game.from_board(complete_board)
    color_name = "white" if agent_a_color == chess.WHITE else "black"
    game.headers["Event"] = "Mixed Minimax Game with Stronger Teacher Trace"
    game.headers["Site"] = "Local"
    game.headers["White"] = result.white_name
    game.headers["Black"] = result.black_name
    game.headers["Result"] = result.result
    game.headers["Termination"] = result.termination
    game.headers["GameID"] = game_id
    game.headers["MixedBatchFormat"] = MIXED_BATCH_FORMAT
    game.headers["MixedBatchVersion"] = str(MIXED_BATCH_VERSION)
    game.headers["OpeningIndex"] = str(opening_index)
    game.headers["Matchup"] = matchup.key
    game.headers["AgentAColor"] = color_name
    if config.neural_checkpoint is not None and neural_sha256 is not None:
        game.headers["NeuralCheckpoint"] = str(config.neural_checkpoint)
        game.headers["NeuralSHA256"] = neural_sha256
    game.headers["TeacherDepth"] = str(config.teacher_depth)
    game.headers["OpeningSeed"] = str(opening_seed)
    game.headers["RandomOpeningPlies"] = str(len(opening_moves))
    game.headers["RandomOpeningMoves"] = " ".join(opening_moves)
    game.headers["TeacherLabels"] = str(teacher_labels)
    game.headers["TrainingDuringCollection"] = "false"
    return save_pgn(
        game,
        _game_pgn_path(config, matchup, opening_index, agent_a_color),
    )


def _play_game(
    config: MixedBatchConfig,
    neural: NeuralAgent | None,
    neural_sha256: str | None,
    *,
    matchup: _Matchup,
    matchup_index: int,
    opening_index: int,
    opening_board: chess.Board,
    opening_moves: Sequence[str],
    opening_seed: int,
    agent_a_color: chess.Color,
    teacher_cache: dict[str, _TeacherAnalysis],
) -> tuple[list[TrainingExample], dict[str, Any]]:
    color_name = "white" if agent_a_color == chess.WHITE else "black"
    opening_set_id = _opening_id(config, opening_index)
    game_id = f"{opening_set_id}-{matchup.key}-agent-a-{color_name}"
    matchups = _matchups(config)
    game_number = (
        (opening_index - 1) * len(matchups) * 2
        + matchup_index * 2
        + (1 if agent_a_color == chess.WHITE else 2)
    )
    pending: list[_PendingTrace] = []
    teacher = MinimaxAgent(
        depth=config.teacher_depth,
        deterministic=True,
        seed=config.seed + game_number * 13,
        name=f"Minimax D{config.teacher_depth} teacher",
    )
    actor_a_seed = config.seed + game_number * 31
    actor_b_seed = config.seed + game_number * 37
    actor_a = _make_actor(
        config,
        matchup.agent_a,
        seed=actor_a_seed,
        neural=neural,
    )
    actor_b = _make_actor(
        config,
        matchup.agent_b,
        seed=actor_b_seed,
        neural=neural,
    )
    traced_a = _TeacherTraceAgent(
        actor_a,
        teacher,
        pending,
        config=config,
        game_id=game_id,
        opening_set_id=opening_set_id,
        opening_index=opening_index,
        matchup=matchup,
        actor=matchup.agent_a,
        actor_slot="agent_a",
        opening_seed=opening_seed,
        opening_plies=len(opening_moves),
        neural_sha256=neural_sha256,
        teacher_cache=teacher_cache,
        actor_seed=actor_a_seed,
    )
    traced_b = _TeacherTraceAgent(
        actor_b,
        teacher,
        pending,
        config=config,
        game_id=game_id,
        opening_set_id=opening_set_id,
        opening_index=opening_index,
        matchup=matchup,
        actor=matchup.agent_b,
        actor_slot="agent_b",
        opening_seed=opening_seed,
        opening_plies=len(opening_moves),
        neural_sha256=neural_sha256,
        teacher_cache=teacher_cache,
        actor_seed=actor_b_seed,
    )
    white = traced_a if agent_a_color == chess.WHITE else traced_b
    black = traced_b if agent_a_color == chess.WHITE else traced_a
    result = run_match(
        white,
        black,
        max_plies=config.max_plies - len(opening_moves),
        starting_fen=opening_board.fen(),
        seed=None,
    )
    if result.illegal_agent is not None:
        raise MixedBatchError(
            f"Generation agent {result.illegal_agent!r} produced illegal move "
            f"{result.illegal_move!r} in {game_id}"
        )
    examples = _finalize_pending(
        pending,
        config=config,
        result=result.result,
        termination=result.termination,
    )
    pgn_path = _save_complete_game_pgn(
        config,
        result,
        matchup=matchup,
        opening_board=opening_board,
        opening_moves=opening_moves,
        opening_seed=opening_seed,
        opening_index=opening_index,
        agent_a_color=agent_a_color,
        game_id=game_id,
        neural_sha256=neural_sha256,
        teacher_labels=len(examples),
    )
    agent_a_outcome = _outcome_for_color(result.result, agent_a_color)
    agreements = sum(bool(item.metadata["actual_matched_teacher"]) for item in examples)
    return examples, {
        "game_id": game_id,
        "game_number": game_number,
        "matchup": matchup.key,
        "agent_a_role": matchup.agent_a.role,
        "agent_b_role": matchup.agent_b.role,
        "agent_a_color": color_name,
        "result": result.result,
        "agent_a_outcome": agent_a_outcome,
        "termination": result.termination,
        "total_plies": len(opening_moves) + result.plies,
        "teacher_labels": len(examples),
        "teacher_agreements": agreements,
        "teacher_disagreements": len(examples) - agreements,
        "actor_win_examples": sum(item.metadata["actor_outcome"] == "win" for item in examples),
        "actor_draw_examples": sum(item.metadata["actor_outcome"] == "draw" for item in examples),
        "actor_loss_examples": sum(item.metadata["actor_outcome"] == "loss" for item in examples),
        "pgn_path": str(pgn_path),
        "pgn_sha256": _sha256_file(pgn_path),
    }


def _play_opening(
    config: MixedBatchConfig,
    neural: NeuralAgent | None,
    neural_sha256: str | None,
    opening_index: int,
) -> dict[str, Any]:
    opening_board, opening_moves, opening_seed = _opening_board(config, opening_index)
    examples: list[TrainingExample] = []
    games: list[dict[str, Any]] = []
    teacher_cache: dict[str, _TeacherAnalysis] = {}
    for matchup_index, matchup in enumerate(_matchups(config)):
        for agent_a_color in (chess.WHITE, chess.BLACK):
            game_examples, game_summary = _play_game(
                config,
                neural,
                neural_sha256,
                matchup=matchup,
                matchup_index=matchup_index,
                opening_index=opening_index,
                opening_board=opening_board,
                opening_moves=opening_moves,
                opening_seed=opening_seed,
                agent_a_color=agent_a_color,
                teacher_cache=teacher_cache,
            )
            examples.extend(game_examples)
            games.append(game_summary)

    opening_path = _opening_dataset_path(config, opening_index)
    save_dataset(
        opening_path,
        examples,
        metadata={
            "kind": "mixed_offline_teacher_opening",
            "mixed_batch_format": MIXED_BATCH_FORMAT,
            "mixed_batch_version": MIXED_BATCH_VERSION,
            "config_signature": _signature(config),
            "opening_index": opening_index,
            "opening_set_id": _opening_id(config, opening_index),
            "opening_seed": opening_seed,
            "opening_moves": list(opening_moves),
            "opening_fen": opening_board.fen(),
            "games": len(games),
            "examples": len(examples),
        },
    )
    return {
        "opening_index": opening_index,
        "opening_set_id": _opening_id(config, opening_index),
        "opening_seed": opening_seed,
        "opening_moves": list(opening_moves),
        "opening_fen": opening_board.fen(),
        "games_count": len(games),
        "examples": len(examples),
        "teacher_agreements": sum(int(game["teacher_agreements"]) for game in games),
        "teacher_disagreements": sum(int(game["teacher_disagreements"]) for game in games),
        "actor_win_examples": sum(int(game["actor_win_examples"]) for game in games),
        "actor_draw_examples": sum(int(game["actor_draw_examples"]) for game in games),
        "actor_loss_examples": sum(int(game["actor_loss_examples"]) for game in games),
        "games": games,
        "opening_dataset_path": str(opening_path),
        "opening_dataset_sha256": _sha256_file(opening_path),
    }


def _initial_state(config: MixedBatchConfig, neural_sha256: str | None) -> dict[str, Any]:
    return {
        "format": MIXED_BATCH_FORMAT,
        "version": MIXED_BATCH_VERSION,
        "config_signature": _signature(config),
        "neural_sha256": neural_sha256,
        "status": "collecting",
        "completed_opening_positions": 0,
        "history": [],
        "training_updates": 0,
    }


def _existing_output_conflicts(config: MixedBatchConfig) -> list[Path]:
    conflicts = [path for path in (config.dataset_path, config.manifest_path) if path.exists()]
    if config.opening_dataset_dir.exists():
        conflicts.extend(sorted(config.opening_dataset_dir.glob("opening_*.pt")))
    if config.pgn_dir.exists():
        conflicts.extend(sorted(config.pgn_dir.glob("**/*.pgn")))
    return conflicts


def _load_or_create_state(
    config: MixedBatchConfig,
    neural_sha256: str | None,
) -> tuple[dict[str, Any], bool]:
    if not config.state_path.exists():
        conflicts = _existing_output_conflicts(config)
        if conflicts:
            preview = ", ".join(str(path) for path in conflicts[:3])
            raise MixedBatchError(
                "Mixed-batch outputs exist without their state file: "
                f"{preview}. Restore the matching state or choose fresh output paths."
            )
        state = _initial_state(config, neural_sha256)
        _atomic_json(config.state_path, state)
        return state, False
    if not config.resume:
        raise MixedBatchError(
            f"Mixed-batch state exists at {config.state_path}; enable resume or use fresh paths."
        )
    try:
        raw = json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MixedBatchError(f"Could not read mixed-batch state: {exc}") from exc
    if not isinstance(raw, dict):
        raise MixedBatchError("Mixed-batch state root must be a mapping")
    if raw.get("format") != MIXED_BATCH_FORMAT or raw.get("version") != MIXED_BATCH_VERSION:
        raise MixedBatchError("Mixed-batch state format/version is incompatible")
    if raw.get("config_signature") != _signature(config):
        raise MixedBatchError(
            "Mixed-batch configuration differs from saved state; restore it or use fresh paths."
        )
    if raw.get("neural_sha256") != neural_sha256:
        raise MixedBatchError("The frozen neural checkpoint changed after collection began")
    return raw, True


def _validate_committed_state(config: MixedBatchConfig, state: Mapping[str, Any]) -> None:
    completed = state.get("completed_opening_positions")
    history = state.get("history")
    if not isinstance(completed, int) or completed < 0 or completed > config.opening_positions:
        raise MixedBatchError("Saved completed_opening_positions is invalid")
    if not isinstance(history, list) or len(history) != completed:
        raise MixedBatchError("Saved mixed-batch history does not match its opening count")
    if state.get("training_updates") != 0:
        raise MixedBatchError("Generation-only state unexpectedly records a training update")
    for expected_opening, item in enumerate(history, start=1):
        if not isinstance(item, dict) or item.get("opening_index") != expected_opening:
            raise MixedBatchError("Saved mixed-batch history has an invalid opening entry")
        opening_path = Path(str(item.get("opening_dataset_path", "")))
        if not opening_path.is_file() or (
            _sha256_file(opening_path) != item.get("opening_dataset_sha256")
        ):
            raise MixedBatchError(
                f"Committed opening dataset is missing or changed: {opening_path}"
            )
        games = item.get("games")
        games_per_opening = len(_matchups(config)) * 2
        if not isinstance(games, list) or len(games) != games_per_opening:
            raise MixedBatchError(
                f"Committed opening {expected_opening} does not contain {games_per_opening} games"
            )
        for game in games:
            if not isinstance(game, dict):
                raise MixedBatchError("Saved mixed-batch game entry is invalid")
            pgn_path = Path(str(game.get("pgn_path", "")))
            if not pgn_path.is_file() or _sha256_file(pgn_path) != game.get("pgn_sha256"):
                raise MixedBatchError(f"Committed PGN is missing or changed: {pgn_path}")
    status = state.get("status")
    if status not in {"collecting", "complete"}:
        raise MixedBatchError(f"Saved mixed-batch status is invalid: {status!r}")
    if status == "complete":
        finalized_early = state.get("finalized_early", False)
        if not isinstance(finalized_early, bool):
            raise MixedBatchError("Saved finalized_early marker is invalid")
        if not finalized_early and completed != config.opening_positions:
            raise MixedBatchError("A complete mixed batch has the wrong opening count")
        if finalized_early and completed == config.opening_positions:
            raise MixedBatchError("An early-finalized batch unexpectedly reached its full target")
        for path, digest_key in (
            (config.dataset_path, "dataset_sha256"),
            (config.manifest_path, "manifest_sha256"),
        ):
            if not path.is_file() or _sha256_file(path) != state.get(digest_key):
                raise MixedBatchError(f"Completed mixed-batch artifact is missing: {path}")


def _aggregate_history(
    history: Sequence[Mapping[str, Any]],
    matchups: Sequence[_Matchup] = MATCHUPS,
) -> dict[str, Any]:
    totals: dict[str, Any] = {
        key: sum(int(item[key]) for item in history)
        for key in (
            "games_count",
            "examples",
            "teacher_agreements",
            "teacher_disagreements",
            "actor_win_examples",
            "actor_draw_examples",
            "actor_loss_examples",
        )
    }
    cohorts: dict[str, dict[str, int | float]] = {}
    for matchup in matchups:
        games = [
            game for item in history for game in item["games"] if game["matchup"] == matchup.key
        ]
        wins = sum(game["agent_a_outcome"] == "win" for game in games)
        draws = sum(game["agent_a_outcome"] == "draw" for game in games)
        losses = sum(game["agent_a_outcome"] == "loss" for game in games)
        cohorts[matchup.key] = {
            "games": len(games),
            "agent_a_wins": wins,
            "agent_a_draws": draws,
            "agent_a_losses": losses,
            "agent_a_points": wins + 0.5 * draws,
        }
    totals["cohort_results"] = cohorts
    return totals


def _finalize_batch(
    config: MixedBatchConfig,
    state: dict[str, Any],
    neural_sha256: str | None,
    *,
    opening_positions: int | None = None,
) -> None:
    finalized_openings = (
        int(state["completed_opening_positions"])
        if opening_positions is None
        else opening_positions
    )
    if finalized_openings <= 0 or finalized_openings > config.opening_positions:
        raise MixedBatchError("Finalized opening count is outside the configured range")
    all_examples: list[TrainingExample] = []
    for opening_index in range(1, finalized_openings + 1):
        loaded = load_dataset(_opening_dataset_path(config, opening_index))
        if loaded.metadata.get("config_signature") != _signature(config) or (
            loaded.metadata.get("opening_index") != opening_index
        ):
            raise MixedBatchError(f"Opening dataset {opening_index} has incompatible provenance")
        all_examples.extend(loaded.examples)
    matchups = _matchups(config)
    aggregates = _aggregate_history(state["history"], matchups)
    if len(all_examples) != aggregates["examples"]:
        raise MixedBatchError("Opening shards do not match the saved total example count")
    save_dataset(
        config.dataset_path,
        all_examples,
        metadata={
            "kind": "mixed_offline_teacher_batch",
            "mixed_batch_format": MIXED_BATCH_FORMAT,
            "mixed_batch_version": MIXED_BATCH_VERSION,
            "config_signature": _signature(config),
            "neural_checkpoint": (
                str(config.neural_checkpoint) if config.neural_checkpoint is not None else None
            ),
            "neural_sha256": neural_sha256,
            "opening_positions": finalized_openings,
            "requested_opening_positions": config.opening_positions,
            "finalized_early": finalized_openings < config.opening_positions,
            "matchups": [matchup.key for matchup in matchups],
            "games_per_matchup": finalized_openings * 2,
            "games": finalized_openings * len(matchups) * 2,
            "examples": len(all_examples),
            "teacher_scope": "every_played_ply_after_random_opening",
            "teacher_target": "softmax_exact_root_scores",
            "teacher_policy_top_k": config.teacher_policy_top_k,
            "teacher_policy_temperature": config.teacher_policy_temperature,
            "outcome_weight_key": "actor_outcome_weight",
            "win_weight": config.win_weight,
            "draw_weight": config.draw_weight,
            "loss_weight": config.loss_weight,
            "training_during_collection": False,
            **aggregates,
        },
    )
    dataset_sha256 = _sha256_file(config.dataset_path)
    manifest = {
        "format": MIXED_BATCH_FORMAT,
        "version": MIXED_BATCH_VERSION,
        "config_signature": _signature(config),
        "config": config.signature_payload(),
        "requested_opening_positions": config.opening_positions,
        "finalized_opening_positions": finalized_openings,
        "finalized_early": finalized_openings < config.opening_positions,
        "neural_sha256": neural_sha256,
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": dataset_sha256,
        "training_updates": 0,
        "aggregates": aggregates,
        # Keep the generic paired-opening key so paired-audit can exclude these starts.
        "pairs": state["history"],
        "openings": state["history"],
    }
    _atomic_json(config.manifest_path, manifest)
    state.update(aggregates)
    state["dataset_sha256"] = dataset_sha256
    state["manifest_sha256"] = _sha256_file(config.manifest_path)
    state["requested_opening_positions"] = config.opening_positions
    state["finalized_opening_positions"] = finalized_openings
    state["finalized_early"] = finalized_openings < config.opening_positions
    state["status"] = "complete"
    _atomic_json(config.state_path, state)


def finalize_mixed_batch(config: MixedBatchConfig) -> MixedBatchSummary:
    """Permanently assemble the currently committed opening shards.

    This is intentionally explicit: once finalized, the same collection state
    cannot resume toward its former larger target.
    """

    neural_sha256: str | None = None
    if config.neural_checkpoint is not None:
        if not config.neural_checkpoint.is_file():
            raise MixedBatchError(f"Neural checkpoint does not exist: {config.neural_checkpoint}")
        neural_sha256 = _sha256_file(config.neural_checkpoint)
    state, resumed = _load_or_create_state(config, neural_sha256)
    _validate_committed_state(config, state)
    if state["status"] == "complete":
        return _summary(config, state, resumed=resumed)
    completed = int(state["completed_opening_positions"])
    if completed == 0:
        raise MixedBatchError("Cannot finalize a mixed batch before one opening is committed")
    conflicts = [path for path in (config.dataset_path, config.manifest_path) if path.exists()]
    if conflicts:
        raise MixedBatchError(
            "Final mixed-batch outputs already exist: " + ", ".join(str(path) for path in conflicts)
        )
    _finalize_batch(config, state, neural_sha256, opening_positions=completed)
    return _summary(config, state, resumed=resumed)


def _summary(
    config: MixedBatchConfig,
    state: Mapping[str, Any],
    *,
    resumed: bool,
) -> MixedBatchSummary:
    aggregates = _aggregate_history(state["history"], _matchups(config))
    return MixedBatchSummary(
        requested_opening_positions=config.opening_positions,
        completed_opening_positions=int(state["completed_opening_positions"]),
        games=int(aggregates["games_count"]),
        examples=int(aggregates["examples"]),
        teacher_agreements=int(aggregates["teacher_agreements"]),
        teacher_disagreements=int(aggregates["teacher_disagreements"]),
        actor_win_examples=int(aggregates["actor_win_examples"]),
        actor_draw_examples=int(aggregates["actor_draw_examples"]),
        actor_loss_examples=int(aggregates["actor_loss_examples"]),
        cohort_results=aggregates["cohort_results"],
        neural_checkpoint=config.neural_checkpoint,
        dataset_path=config.dataset_path,
        manifest_path=config.manifest_path,
        state_path=config.state_path,
        pgn_dir=config.pgn_dir,
        resumed=resumed,
    )


def run_mixed_batch(config: MixedBatchConfig) -> MixedBatchSummary:
    """Generate the configured games and teacher targets without training."""

    if config.neural_checkpoint is not None and not config.neural_checkpoint.is_file():
        raise MixedBatchError(f"Neural checkpoint does not exist: {config.neural_checkpoint}")
    distinct_files = {
        config.state_path.resolve(),
        config.manifest_path.resolve(),
        config.dataset_path.resolve(),
    }
    if config.neural_checkpoint is not None:
        distinct_files.add(config.neural_checkpoint.resolve())
    expected_distinct_files = 4 if config.neural_checkpoint is not None else 3
    if len(distinct_files) != expected_distinct_files:
        raise MixedBatchError("Neural, state, manifest, and dataset paths must be distinct")
    if config.opening_dataset_dir.resolve() == config.pgn_dir.resolve():
        raise MixedBatchError("opening_dataset_dir and pgn_dir must be different directories")
    neural_sha256: str | None = None
    neural: NeuralAgent | None = None
    if config.neural_checkpoint is not None:
        load_checkpoint(config.neural_checkpoint, map_location="cpu")
        neural_sha256 = _sha256_file(config.neural_checkpoint)
    state, resumed = _load_or_create_state(config, neural_sha256)
    _validate_committed_state(config, state)
    if state["status"] == "complete":
        return _summary(config, state, resumed=resumed)

    if config.neural_checkpoint is not None:
        neural = NeuralAgent(
            config.neural_checkpoint,
            device=config.device,
            deterministic=True,
            temperature=0.0,
            seed=config.seed,
            name=f"Frozen neural ({config.neural_checkpoint.name})",
        )
    try:
        while int(state["completed_opening_positions"]) < config.opening_positions:
            opening_index = int(state["completed_opening_positions"]) + 1
            opening_summary = _play_opening(
                config,
                neural,
                neural_sha256,
                opening_index,
            )
            history = list(state["history"])
            history.append(opening_summary)
            state["history"] = history
            state["completed_opening_positions"] = opening_index
            state.update(_aggregate_history(history, _matchups(config)))
            _atomic_json(config.state_path, state)
            LOGGER.info(
                "mixed-batch opening=%d/%d games=%d examples=%d",
                opening_index,
                config.opening_positions,
                state["games_count"],
                state["examples"],
            )
    finally:
        if neural is not None:
            del neural
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _finalize_batch(config, state, neural_sha256)
    return _summary(config, state, resumed=resumed)
