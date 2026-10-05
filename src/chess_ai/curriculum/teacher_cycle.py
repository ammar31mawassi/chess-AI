"""Iterative neural-student games corrected by a stronger minimax teacher."""

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
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import chess
import numpy as np
import numpy.typing as npt
import torch

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.arena.match import MatchResult, run_match
from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import load_checkpoint, load_model
from chess_ai.training import TrainingConfig, train_model

LOGGER = logging.getLogger(__name__)

CURRICULUM_FORMAT = "self-improving-chess-ai.teacher-correction-cycle"
CURRICULUM_VERSION = 1


class TeacherCycleError(RuntimeError):
    """Raised when a teacher-correction run cannot continue safely."""


@dataclass(frozen=True, slots=True)
class TeacherCycleConfig:
    """Reproducible settings for a fixed opponent/teacher curriculum stage."""

    source_checkpoint: Path
    replay_dataset_path: Path
    state_path: Path
    corrections_dir: Path
    checkpoint_dir: Path
    metrics_path: Path
    pgn_dir: Path
    cycles: int = 100
    opponent_depth: int = 1
    teacher_depth: int = 2
    max_plies: int = 200
    opening_min_plies: int = 4
    opening_max_plies: int = 8
    normal_start_every_cycles: int = 0
    anchor_correction_share: float = 0.25
    evaluate_after_normal_start: bool = True
    training_examples_per_cycle: int = 4096
    correction_fraction: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.1
    legal_policy_mask: bool = True
    mastery_window_cycles: int = 5
    seed: int = 2026
    device: str = "auto"
    resume: bool = True

    def __post_init__(self) -> None:
        for name in (
            "source_checkpoint",
            "replay_dataset_path",
            "state_path",
            "corrections_dir",
            "checkpoint_dir",
            "metrics_path",
            "pgn_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("cycles", "opponent_depth", "teacher_depth", "max_plies", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cycles < 2:
            raise ValueError("cycles must be at least 2 so a trained candidate is played again")
        if self.teacher_depth <= self.opponent_depth:
            raise ValueError("teacher_depth must be greater than opponent_depth")
        if self.training_examples_per_cycle <= 1:
            raise ValueError("training_examples_per_cycle must be greater than one")
        if not 0.0 < self.correction_fraction < 1.0:
            raise ValueError("correction_fraction must be between 0 and 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.policy_loss_weight < 0.0 or self.value_loss_weight < 0.0:
            raise ValueError("policy/value loss weights cannot be negative")
        if self.policy_loss_weight == 0.0 and self.value_loss_weight == 0.0:
            raise ValueError("at least one policy/value loss weight must be positive")
        if self.mastery_window_cycles <= 0:
            raise ValueError("mastery_window_cycles must be positive")
        if self.mastery_window_cycles > self.cycles:
            raise ValueError("mastery_window_cycles cannot exceed cycles")
        if self.opening_min_plies < 0 or self.opening_max_plies < self.opening_min_plies:
            raise ValueError("opening ply bounds must satisfy 0 <= minimum <= maximum")
        if not any(
            plies % 2 == 0 for plies in range(self.opening_min_plies, self.opening_max_plies + 1)
        ):
            raise ValueError("opening ply bounds must contain at least one even length")
        if self.opening_max_plies >= self.max_plies:
            raise ValueError("opening_max_plies must be smaller than max_plies")
        if (
            isinstance(self.normal_start_every_cycles, bool)
            or not isinstance(self.normal_start_every_cycles, int)
            or self.normal_start_every_cycles < 0
        ):
            raise ValueError("normal_start_every_cycles must be a non-negative integer")
        if isinstance(self.anchor_correction_share, bool) or not (
            0.0 <= self.anchor_correction_share <= 1.0
        ):
            raise ValueError("anchor_correction_share must be between 0 and 1")
        if not isinstance(self.evaluate_after_normal_start, bool):
            raise ValueError("evaluate_after_normal_start must be a boolean")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        seed: int = 2026,
        device: str = "auto",
        resume: bool | None = None,
    ) -> TeacherCycleConfig:
        values = dict(raw)
        values.setdefault("seed", seed)
        values.setdefault("device", device)
        if resume is not None:
            values["resume"] = resume
        path_fields = {
            "source_checkpoint",
            "replay_dataset_path",
            "state_path",
            "corrections_dir",
            "checkpoint_dir",
            "metrics_path",
            "pgn_dir",
        }
        for name in path_fields:
            if name in values:
                values[name] = Path(str(values[name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown teacher_cycle settings: {', '.join(unknown)}")
        missing = sorted(path_fields.difference(values))
        if missing:
            raise ValueError(f"Missing teacher_cycle settings: {', '.join(missing)}")
        return cls(**values)

    def signature_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("resume", None)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True, slots=True)
class TeacherCycleSummary:
    """Compact final/progress report for one curriculum invocation."""

    completed_cycles: int
    requested_cycles: int
    games_played: int
    training_updates: int
    current_checkpoint: Path
    corrections_saved: int
    student_wins: int
    student_draws: int
    student_losses: int
    first_win_cycle: int | None
    ready_for_next_stage: bool
    mastery_requirement: str
    state_path: Path
    pgn_dir: Path
    interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("current_checkpoint", "state_path", "pgn_dir"):
            payload[key] = str(payload[key])
        return payload


@dataclass(slots=True)
class _PendingCorrection:
    board_tensor: npt.NDArray[np.float32]
    target_policy: npt.NDArray[np.float32]
    turn: chess.Color
    metadata: dict[str, Any]


class _CorrectionRecordingAgent:
    """Delegate play to the student while recording the teacher's answer."""

    def __init__(
        self,
        student: NeuralAgent,
        teacher: MinimaxAgent,
        *,
        game_id: str,
        cycle: int,
        student_color: chess.Color,
        checkpoint: Path,
        opponent_depth: int,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.game_id = game_id
        self.cycle = cycle
        self.student_color = student_color
        self.checkpoint = checkpoint
        self.opponent_depth = opponent_depth
        self.name = f"Neural student ({checkpoint.name})"
        self.pending: list[_PendingCorrection] = []
        self._boards = BoardEncoder()
        self._moves = MoveEncoder()

    def choose_move(self, board: chess.Board) -> chess.Move:
        teacher_move = self.teacher.choose_move(board.copy(stack=True))
        teacher_score = self.teacher.last_score
        student_move = self.student.choose_move(board.copy(stack=True))
        policy = np.zeros(self._moves.action_size, dtype=np.float32)
        action = self._moves.encode(teacher_move)
        policy[action] = 1.0
        self.pending.append(
            _PendingCorrection(
                board_tensor=self._boards.encode(board),
                target_policy=policy,
                turn=board.turn,
                metadata={
                    "game_id": self.game_id,
                    "cycle": self.cycle,
                    "fen": board.fen(),
                    "player_to_move": "white" if board.turn == chess.WHITE else "black",
                    "student_color": "white" if self.student_color == chess.WHITE else "black",
                    "student_checkpoint": str(self.checkpoint),
                    "student_move_uci": student_move.uci(),
                    "teacher_move_uci": teacher_move.uci(),
                    "move_uci": teacher_move.uci(),
                    "policy_action": action,
                    "student_matched_teacher": student_move == teacher_move,
                    "policy_source_agent": "minimax_teacher",
                    "teacher_depth": self.teacher.depth,
                    "teacher_score": teacher_score,
                    "opponent_depth": self.opponent_depth,
                    "correction_kind": "teacher_relabelled_student_turn",
                },
            )
        )
        return student_move

    def finalize(self, result: str, termination: str) -> list[TrainingExample]:
        return [
            TrainingExample(
                board_tensor=item.board_tensor,
                target_policy=item.target_policy,
                target_value=result_value_for_turn(result, item.turn),
                metadata={
                    **item.metadata,
                    "result": result,
                    "termination": termination,
                },
            )
            for item in self.pending
        ]


def _signature(config: TeacherCycleConfig) -> str:
    encoded = json.dumps(config.signature_payload(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _legacy_signature_without_normal_start(
    config: TeacherCycleConfig,
    *,
    cycles: int,
) -> str:
    """Recreate state signatures written before periodic anchors existed."""

    payload = config.signature_payload()
    payload["cycles"] = cycles
    payload.pop("normal_start_every_cycles", None)
    payload.pop("anchor_correction_share", None)
    payload.pop("evaluate_after_normal_start", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_signature_without_anchor_replay(
    config: TeacherCycleConfig,
    *,
    cycles: int,
) -> str:
    """Recreate signatures written after anchors but before anchor replay."""

    payload = config.signature_payload()
    payload["cycles"] = cycles
    payload.pop("anchor_correction_share", None)
    payload.pop("evaluate_after_normal_start", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _initial_state(config: TeacherCycleConfig) -> dict[str, Any]:
    return {
        "format": CURRICULUM_FORMAT,
        "version": CURRICULUM_VERSION,
        "config_signature": _signature(config),
        "status": "ready",
        "completed_cycles": 0,
        "training_updates": 0,
        "current_checkpoint": str(config.source_checkpoint),
        "history": [],
        "pending": None,
        "first_win_cycle": None,
        "ready_for_next_stage": False,
    }


def _load_state(config: TeacherCycleConfig) -> dict[str, Any]:
    if not config.state_path.exists():
        conflicts = [
            *config.checkpoint_dir.glob("*.pt"),
            *config.corrections_dir.glob("cycle_*.pt"),
            *config.pgn_dir.glob("cycle_*.pgn"),
        ]
        if config.metrics_path.exists():
            conflicts.append(config.metrics_path)
        if conflicts:
            preview = ", ".join(str(path) for path in conflicts[:3])
            raise TeacherCycleError(
                "Curriculum outputs already exist without their state file: "
                f"{preview}. Restore the matching state or choose fresh output paths."
            )
        state = _initial_state(config)
        _atomic_json(config.state_path, state)
        return state
    if not config.resume:
        raise TeacherCycleError(
            f"Curriculum state already exists at {config.state_path}; enable resume or use "
            "fresh output paths."
        )
    try:
        raw = json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherCycleError(
            f"Could not read curriculum state {config.state_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TeacherCycleError("Curriculum state root must be a mapping")
    if raw.get("format") != CURRICULUM_FORMAT or raw.get("version") != CURRICULUM_VERSION:
        raise TeacherCycleError("Curriculum state format/version is incompatible with this code")
    if raw.get("config_signature") != _signature(config):
        completed = raw.get("completed_cycles")
        saved_signature = raw.get("config_signature")
        extending_complete_run = (
            isinstance(completed, int)
            and raw.get("status") == "complete"
            and config.cycles > completed
            and saved_signature == _signature(replace(config, cycles=completed))
        )
        upgrading_completed_legacy_run = (
            isinstance(completed, int)
            and raw.get("status") == "complete"
            and config.cycles >= completed
            and saved_signature == _legacy_signature_without_normal_start(config, cycles=completed)
        )
        upgrading_completed_anchor_run = (
            isinstance(completed, int)
            and raw.get("status") == "complete"
            and config.cycles >= completed
            and saved_signature == _legacy_signature_without_anchor_replay(config, cycles=completed)
        )
        if (
            not extending_complete_run
            and not upgrading_completed_legacy_run
            and not upgrading_completed_anchor_run
        ):
            raise TeacherCycleError(
                "Curriculum configuration differs from the saved state. Only increasing the "
                "total cycle target of a completed run is allowed. A legacy completed run may "
                "also add its first periodic normal-start or anchor-replay refinement; otherwise "
                "restore the original config or choose fresh output paths."
            )
    return raw


def _validate_state_artifacts(config: TeacherCycleConfig, state: Mapping[str, Any]) -> None:
    status = state.get("status")
    if status not in {"ready", "collected", "complete"}:
        raise TeacherCycleError(f"Saved curriculum status is invalid: {status!r}")
    history = state.get("history")
    completed = state.get("completed_cycles")
    updates = state.get("training_updates")
    if (
        not isinstance(history, list)
        or not isinstance(completed, int)
        or not isinstance(updates, int)
    ):
        raise TeacherCycleError("Saved curriculum counters/history are invalid")
    if len(history) != completed:
        raise TeacherCycleError("Saved curriculum history does not match completed_cycles")
    expected_updates = completed if completed < config.cycles else config.cycles - 1
    extending_from_holdout = (
        status == "complete" and completed < config.cycles and updates == completed - 1
    )
    if updates != expected_updates and not extending_from_holdout:
        raise TeacherCycleError(
            f"Saved curriculum has {updates} training updates; expected {expected_updates}"
        )
    current = Path(str(state.get("current_checkpoint", "")))
    if not current.is_file():
        raise TeacherCycleError(f"Saved current checkpoint is missing: {current}")
    for item in history:
        if not isinstance(item, dict) or not isinstance(item.get("cycle"), int):
            raise TeacherCycleError("Saved curriculum history contains an invalid cycle")
        cycle = int(item["cycle"])
        if not _correction_path(config, cycle).is_file():
            raise TeacherCycleError(f"Saved correction file is missing for cycle {cycle}")
        games = item.get("games")
        if not isinstance(games, list) or len(games) != 2:
            raise TeacherCycleError(f"Saved cycle {cycle} does not contain two game records")
        for game in games:
            if not isinstance(game, dict) or not Path(str(game.get("pgn_path", ""))).is_file():
                raise TeacherCycleError(f"Saved PGN is missing for cycle {cycle}")
    if status == "collected":
        pending = state.get("pending")
        if not isinstance(pending, dict) or pending.get("cycle") != completed + 1:
            raise TeacherCycleError("Saved pending cycle does not follow completed history")
        if not _correction_path(config, completed + 1).is_file():
            raise TeacherCycleError("Saved pending correction file is missing")


def _validate_paths(config: TeacherCycleConfig) -> None:
    if not config.source_checkpoint.is_file():
        raise TeacherCycleError(f"Source checkpoint does not exist: {config.source_checkpoint}")
    if not config.replay_dataset_path.is_file():
        raise TeacherCycleError(f"Replay dataset does not exist: {config.replay_dataset_path}")
    if config.checkpoint_dir.resolve() == config.source_checkpoint.resolve().parent:
        raise TeacherCycleError("checkpoint_dir must not overwrite the source checkpoint directory")
    if config.state_path.resolve() == config.metrics_path.resolve():
        raise TeacherCycleError("state_path and metrics_path must be different files")
    if config.corrections_dir.resolve() == config.pgn_dir.resolve():
        raise TeacherCycleError("corrections_dir and pgn_dir must be different directories")
    load_checkpoint(config.source_checkpoint, map_location="cpu")


def _opening_board(config: TeacherCycleConfig, cycle: int) -> tuple[chess.Board, list[str], int]:
    opening_seed = config.seed + cycle * 1_000_003
    if config.normal_start_every_cycles > 0 and cycle % config.normal_start_every_cycles == 0:
        return chess.Board(), [], opening_seed
    lengths = [
        plies
        for plies in range(config.opening_min_plies, config.opening_max_plies + 1)
        if plies % 2 == 0
    ]
    for attempt in range(100):
        rng = random.Random(opening_seed + attempt)
        requested = rng.choice(lengths)
        board = chess.Board()
        moves: list[str] = []
        valid = True
        for _ in range(requested):
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            move = rng.choice(legal)
            board.push(move)
            moves.append(move.uci())
            if board.is_game_over(claim_draw=True):
                valid = False
                break
        if valid and board.turn == chess.WHITE:
            return board, moves, opening_seed + attempt
    raise TeacherCycleError("Could not create a non-terminal even-ply random opening")


def _student_outcome(result: str, student_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    student_won = (result == "1-0") == (student_color == chess.WHITE)
    return "win" if student_won else "loss"


def _play_one(
    config: TeacherCycleConfig,
    student: NeuralAgent,
    checkpoint: Path,
    *,
    cycle: int,
    student_color: chess.Color,
    starting_board: chess.Board,
    opening_moves: Sequence[str],
    opening_seed: int,
) -> tuple[MatchResult, list[TrainingExample], dict[str, Any]]:
    color_name = "white" if student_color == chess.WHITE else "black"
    game_id = f"teacher-cycle-{_signature(config)[:10]}-{cycle:04d}-{color_name}"
    teacher = MinimaxAgent(
        depth=config.teacher_depth,
        deterministic=True,
        seed=config.seed + cycle,
        name=f"Minimax D{config.teacher_depth} teacher",
    )
    recording_student = _CorrectionRecordingAgent(
        student,
        teacher,
        game_id=game_id,
        cycle=cycle,
        student_color=student_color,
        checkpoint=checkpoint,
        opponent_depth=config.opponent_depth,
    )
    opponent = MinimaxAgent(
        depth=config.opponent_depth,
        deterministic=True,
        seed=config.seed + cycle * 2 + int(student_color),
        name=f"Minimax D{config.opponent_depth} opponent",
    )
    pgn_path = config.pgn_dir / f"cycle_{cycle:04d}_student_{color_name}.pgn"
    white = recording_student if student_color == chess.WHITE else opponent
    black = opponent if student_color == chess.WHITE else recording_student
    result = run_match(
        white,
        black,
        max_plies=config.max_plies - len(opening_moves),
        starting_fen=starting_board.fen(),
        seed=config.seed + cycle * 2 + (0 if student_color == chess.WHITE else 1),
        pgn_path=pgn_path,
        extra_headers={
            "CurriculumFormat": CURRICULUM_FORMAT,
            "CurriculumVersion": str(CURRICULUM_VERSION),
            "Cycle": str(cycle),
            "StudentColor": color_name,
            "StudentCheckpoint": str(checkpoint),
            "OpponentDepth": str(config.opponent_depth),
            "TeacherDepth": str(config.teacher_depth),
            "OpeningSeed": str(opening_seed),
            "NormalStart": str(len(opening_moves) == 0).lower(),
            "UnrecordedOpeningPlies": str(len(opening_moves)),
            "UnrecordedOpeningMoves": " ".join(opening_moves),
        },
    )
    examples = recording_student.finalize(result.result, result.termination)
    game_summary = {
        "game_id": game_id,
        "student_color": color_name,
        "result": result.result,
        "student_outcome": _student_outcome(result.result, student_color),
        "termination": result.termination,
        "plies": result.plies,
        "pgn_path": str(pgn_path),
        "teacher_labels": len(examples),
        "teacher_disagreements": sum(
            not bool(example.metadata["student_matched_teacher"]) for example in examples
        ),
    }
    return result, examples, game_summary


def _position_digest(example: TrainingExample) -> bytes:
    tensor = example.board_tensor
    encoded = (
        tensor.tobytes()
        if isinstance(tensor, np.ndarray)
        else tensor.detach().cpu().numpy().tobytes()
    )
    return hashlib.sha256(encoded).digest()


def _play_cycle(
    config: TeacherCycleConfig,
    checkpoint: Path,
    cycle: int,
    seen_corrections: set[bytes],
) -> tuple[list[TrainingExample], dict[str, Any]]:
    starting_board, opening_moves, opening_seed = _opening_board(config, cycle)
    normal_start = len(opening_moves) == 0
    student = NeuralAgent(
        checkpoint,
        device=config.device,
        deterministic=True,
        temperature=0.0,
        seed=config.seed + cycle,
    )
    all_examples: list[TrainingExample] = []
    games: list[dict[str, Any]] = []
    try:
        for color in (chess.WHITE, chess.BLACK):
            _result, examples, game_summary = _play_one(
                config,
                student,
                checkpoint,
                cycle=cycle,
                student_color=color,
                starting_board=starting_board,
                opening_moves=opening_moves,
                opening_seed=opening_seed,
            )
            all_examples.extend(
                TrainingExample(
                    board_tensor=example.board_tensor,
                    target_policy=example.target_policy,
                    target_value=example.target_value,
                    metadata={**example.metadata, "normal_start": normal_start},
                )
                for example in examples
            )
            games.append(game_summary)
    finally:
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    retained: list[TrainingExample] = []
    duplicates = 0
    seen_this_cycle: set[bytes] = set()
    for example in all_examples:
        digest = _position_digest(example)
        # Repeated normal-start positions are intentional rehearsal across
        # anchor cycles. Deduplicate only within the current anchor pair so a
        # repetition loop cannot dominate one cycle. Varied cycles keep the
        # original global diversity filter.
        already_seen = digest in seen_this_cycle if normal_start else digest in seen_corrections
        if already_seen:
            duplicates += 1
            continue
        seen_this_cycle.add(digest)
        seen_corrections.add(digest)
        retained.append(example)
    wins = sum(game["student_outcome"] == "win" for game in games)
    draws = sum(game["student_outcome"] == "draw" for game in games)
    losses = sum(game["student_outcome"] == "loss" for game in games)
    summary = {
        "cycle": cycle,
        "played_checkpoint": str(checkpoint),
        "opening_fen": starting_board.fen(),
        "opening_seed": opening_seed,
        "opening_moves": list(opening_moves),
        "normal_start": normal_start,
        "games": games,
        "student_wins": wins,
        "student_draws": draws,
        "student_losses": losses,
        "corrections_collected": len(all_examples),
        "corrections_retained": len(retained),
        "correction_duplicates_discarded": duplicates,
        "teacher_disagreements": sum(game["teacher_disagreements"] for game in games),
    }
    return retained, summary


def _correction_path(config: TeacherCycleConfig, cycle: int) -> Path:
    return config.corrections_dir / f"cycle_{cycle:04d}.pt"


def _load_corrections(config: TeacherCycleConfig, through_cycle: int) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for cycle in range(1, through_cycle + 1):
        path = _correction_path(config, cycle)
        if not path.is_file():
            raise TeacherCycleError(f"Saved curriculum correction file is missing: {path}")
        examples.extend(load_dataset(path).examples)
    return examples


def _is_anchor_correction(example: TrainingExample, anchor_cycles: set[int]) -> bool:
    if example.metadata.get("normal_start") is True:
        return True
    example_cycle = example.metadata.get("cycle")
    return isinstance(example_cycle, int) and example_cycle in anchor_cycles


def _sample_training_examples(
    config: TeacherCycleConfig,
    replay: Sequence[TrainingExample],
    corrections: Sequence[TrainingExample],
    newest: Sequence[TrainingExample],
    *,
    cycle: int,
    anchor_cycles: set[int],
    sample_stats: dict[str, int | float] | None = None,
) -> list[TrainingExample]:
    if not replay:
        raise TeacherCycleError("Replay dataset contains no examples")
    if not corrections:
        raise TeacherCycleError("No teacher corrections are available for training")
    rng = random.Random(config.seed + cycle * 97_409)
    correction_limit = max(
        1,
        math.floor(config.training_examples_per_cycle * config.correction_fraction),
    )

    newest_keys = {(example.game_id, str(example.metadata.get("fen", ""))) for example in newest}
    older = [
        example
        for example in corrections
        if (example.game_id, str(example.metadata.get("fen", ""))) not in newest_keys
    ]
    newest_anchors = [
        example for example in newest if _is_anchor_correction(example, anchor_cycles)
    ]
    newest_varied = [
        example for example in newest if not _is_anchor_correction(example, anchor_cycles)
    ]
    older_anchors = [example for example in older if _is_anchor_correction(example, anchor_cycles)]
    older_varied = [
        example for example in older if not _is_anchor_correction(example, anchor_cycles)
    ]

    requested_anchor_target = math.floor(correction_limit * config.anchor_correction_share)
    if config.anchor_correction_share > 0.0:
        requested_anchor_target = max(1, requested_anchor_target)
    anchor_target = min(
        requested_anchor_target,
        len(newest_anchors) + len(older_anchors),
    )

    # Keep the latest correction pair whenever possible, but never let a large
    # varied-opening batch displace the standard-position rehearsal quota.
    if len(newest_anchors) >= correction_limit:
        selected_corrections = rng.sample(newest_anchors, correction_limit)
    else:
        selected_corrections = list(newest_anchors)
        older_anchor_needed = min(
            max(0, anchor_target - len(selected_corrections)),
            len(older_anchors),
        )
        varied_capacity = correction_limit - len(selected_corrections) - older_anchor_needed
        if len(newest_varied) <= varied_capacity:
            selected_corrections.extend(newest_varied)
        else:
            selected_corrections.extend(rng.sample(newest_varied, varied_capacity))
        selected_older_anchors = (
            rng.sample(older_anchors, older_anchor_needed) if older_anchor_needed else []
        )
        selected_corrections.extend(selected_older_anchors)

        selected_keys = {
            (example.game_id, str(example.metadata.get("fen", "")))
            for example in selected_corrections
        }
        leftovers = [
            example
            for example in (*older_varied, *older_anchors)
            if (example.game_id, str(example.metadata.get("fen", ""))) not in selected_keys
        ]
        remaining = min(correction_limit - len(selected_corrections), len(leftovers))
        if remaining:
            selected_corrections.extend(rng.sample(leftovers, remaining))

    replay_needed_for_ratio = math.floor(
        len(selected_corrections) * (1.0 - config.correction_fraction) / config.correction_fraction
    )
    replay_count = min(
        len(replay),
        config.training_examples_per_cycle - len(selected_corrections),
        replay_needed_for_ratio,
    )
    if sample_stats is not None:
        sample_stats.update(
            {
                "total_examples": replay_count + len(selected_corrections),
                "replay_examples": replay_count,
                "correction_examples": len(selected_corrections),
                "anchor_correction_requested": requested_anchor_target,
                "anchor_correction_target": anchor_target,
                "anchor_correction_examples": sum(
                    _is_anchor_correction(example, anchor_cycles)
                    for example in selected_corrections
                ),
                "configured_anchor_correction_share": config.anchor_correction_share,
            }
        )
    selected = rng.sample(list(replay), replay_count) + selected_corrections
    rng.shuffle(selected)
    return selected


def _repair_metrics(config: TeacherCycleConfig, checkpoint: Path, epoch: int) -> None:
    existing_epochs: set[int] = set()
    if config.metrics_path.exists():
        for line in config.metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeacherCycleError(f"Metrics file contains invalid JSON: {exc}") from exc
            if isinstance(row, dict) and isinstance(row.get("epoch"), int):
                existing_epochs.add(row["epoch"])
    if epoch in existing_epochs:
        return
    loaded = load_checkpoint(checkpoint, map_location="cpu")
    if loaded.epoch != epoch:
        raise TeacherCycleError("Cannot repair metrics from a checkpoint with the wrong epoch")
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with config.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(loaded.metrics, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _train_update(
    config: TeacherCycleConfig,
    state: Mapping[str, Any],
    replay: Sequence[TrainingExample],
    corrections: Sequence[TrainingExample],
    newest: Sequence[TrainingExample],
    *,
    cycle: int,
    anchor_cycles: set[int],
    cycle_summary: dict[str, Any],
) -> Path:
    expected_epoch = int(state["training_updates"]) + 1
    sample_stats: dict[str, int | float] = {}
    training_examples = _sample_training_examples(
        config,
        replay,
        corrections,
        newest,
        cycle=cycle,
        anchor_cycles=anchor_cycles,
        sample_stats=sample_stats,
    )
    cycle_summary["training_batch"] = sample_stats
    last_checkpoint = config.checkpoint_dir / "last.pt"
    if last_checkpoint.exists():
        saved_epoch = load_checkpoint(last_checkpoint, map_location="cpu").epoch
        if saved_epoch == expected_epoch:
            _repair_metrics(config, last_checkpoint, expected_epoch)
            return last_checkpoint
        if saved_epoch != expected_epoch - 1:
            raise TeacherCycleError(
                f"Curriculum checkpoint epoch {saved_epoch} does not match expected "
                f"epoch {expected_epoch - 1}."
            )

    source = Path(str(state["current_checkpoint"]))
    model = load_model(source, device="cpu", eval_mode=False)
    training_config = TrainingConfig(
        checkpoint_dir=config.checkpoint_dir,
        metrics_path=config.metrics_path,
        batch_size=min(config.batch_size, len(training_examples)),
        epochs=expected_epoch,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        legal_policy_mask=config.legal_policy_mask,
        gradient_clip=config.gradient_clip,
        validation_fraction=0.0,
        scheduler=False,
        seed=config.seed,
        device=config.device,
        log_every=20,
        resume_from=last_checkpoint if expected_epoch > 1 else None,
    )
    try:
        history = train_model(model, training_config, training_examples)
        if not history or history[-1].epoch != expected_epoch:
            raise TeacherCycleError("Curriculum training did not complete the expected update")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return last_checkpoint


def _evaluate_standard_pair(
    config: TeacherCycleConfig,
    checkpoint: Path,
    *,
    cycle: int,
) -> dict[str, Any]:
    """Save a two-color normal-start regression pair after an anchor update."""

    student = NeuralAgent(
        checkpoint,
        device=config.device,
        deterministic=True,
        temperature=0.0,
        seed=config.seed + cycle,
    )
    games: list[dict[str, Any]] = []
    try:
        for student_color in (chess.WHITE, chess.BLACK):
            color_name = "white" if student_color == chess.WHITE else "black"
            opponent = MinimaxAgent(
                depth=config.opponent_depth,
                deterministic=True,
                seed=config.seed + cycle * 2 + int(student_color),
                name=f"Minimax D{config.opponent_depth} opponent",
            )
            pgn_path = (
                config.pgn_dir
                / "standard_evaluation"
                / f"after_cycle_{cycle:04d}_student_{color_name}.pgn"
            )
            white = student if student_color == chess.WHITE else opponent
            black = opponent if student_color == chess.WHITE else student
            result = run_match(
                white,
                black,
                max_plies=config.max_plies,
                seed=config.seed + cycle * 2 + (0 if student_color == chess.WHITE else 1),
                pgn_path=pgn_path,
                extra_headers={
                    "CurriculumFormat": CURRICULUM_FORMAT,
                    "CurriculumVersion": str(CURRICULUM_VERSION),
                    "EvaluationKind": "post-training-standard-regression",
                    "SourceCycle": str(cycle),
                    "StudentColor": color_name,
                    "StudentCheckpoint": str(checkpoint),
                    "OpponentDepth": str(config.opponent_depth),
                },
            )
            games.append(
                {
                    "student_color": color_name,
                    "result": result.result,
                    "student_outcome": _student_outcome(result.result, student_color),
                    "termination": result.termination,
                    "plies": result.plies,
                    "pgn_path": str(pgn_path),
                }
            )
    finally:
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wins = sum(game["student_outcome"] == "win" for game in games)
    draws = sum(game["student_outcome"] == "draw" for game in games)
    losses = sum(game["student_outcome"] == "loss" for game in games)
    return {
        "kind": "post_training_standard_regression",
        "source_cycle": cycle,
        "checkpoint": str(checkpoint),
        "student_wins": wins,
        "student_draws": draws,
        "student_losses": losses,
        "score_points": wins + 0.5 * draws,
        "games": games,
    }


def _attach_standard_evaluation(
    config: TeacherCycleConfig,
    cycle_summary: dict[str, Any],
    checkpoint: Path,
) -> None:
    if not config.evaluate_after_normal_start or not bool(cycle_summary.get("normal_start")):
        return
    evaluation = _evaluate_standard_pair(config, checkpoint, cycle=int(cycle_summary["cycle"]))
    pre_training_score = int(cycle_summary["student_wins"]) + 0.5 * int(
        cycle_summary["student_draws"]
    )
    evaluation["pre_training_score_points"] = pre_training_score
    evaluation["regressed_from_pre_training_pair"] = (
        float(evaluation["score_points"]) < pre_training_score
    )
    cycle_summary["post_training_standard_evaluation"] = evaluation
    log = LOGGER.warning if evaluation["regressed_from_pre_training_pair"] else LOGGER.info
    log(
        "post-training standard evaluation after cycle=%d W/D/L=%d/%d/%d pre_score=%.1f "
        "post_score=%.1f regressed=%s",
        cycle_summary["cycle"],
        evaluation["student_wins"],
        evaluation["student_draws"],
        evaluation["student_losses"],
        pre_training_score,
        evaluation["score_points"],
        evaluation["regressed_from_pre_training_pair"],
    )


def _mastered(history: Sequence[Mapping[str, Any]], window: int) -> bool:
    if len(history) < window:
        return False
    return all(int(item["student_wins"]) == 2 for item in history[-window:])


def _summary(config: TeacherCycleConfig, state: Mapping[str, Any]) -> TeacherCycleSummary:
    history = list(state.get("history", []))
    return TeacherCycleSummary(
        completed_cycles=int(state["completed_cycles"]),
        requested_cycles=config.cycles,
        games_played=int(state["completed_cycles"]) * 2,
        training_updates=int(state["training_updates"]),
        current_checkpoint=Path(str(state["current_checkpoint"])),
        corrections_saved=sum(int(item["corrections_retained"]) for item in history),
        student_wins=sum(int(item["student_wins"]) for item in history),
        student_draws=sum(int(item["student_draws"]) for item in history),
        student_losses=sum(int(item["student_losses"]) for item in history),
        first_win_cycle=(
            int(state["first_win_cycle"]) if state.get("first_win_cycle") is not None else None
        ),
        ready_for_next_stage=bool(state.get("ready_for_next_stage", False)),
        mastery_requirement=(
            f"win both colors for {config.mastery_window_cycles} consecutive seeded openings"
        ),
        state_path=config.state_path,
        pgn_dir=config.pgn_dir,
    )


def run_teacher_cycle(config: TeacherCycleConfig) -> TeacherCycleSummary:
    """Run or resume the configured 200-game teacher-correction experiment.

    The final pair is deliberately held out from training, so the returned
    checkpoint is one that actually played the final two reported games.
    """

    _validate_paths(config)
    replay = list(load_dataset(config.replay_dataset_path).examples)
    state = _load_state(config)
    _validate_state_artifacts(config, state)
    completed = int(state.get("completed_cycles", 0))
    if completed < 0 or completed > config.cycles:
        raise TeacherCycleError("Saved completed_cycles is outside the configured range")
    corrections_through = completed
    if state.get("status") == "collected":
        pending = state.get("pending")
        if not isinstance(pending, dict) or not isinstance(pending.get("cycle"), int):
            raise TeacherCycleError("Saved collected state has invalid pending-cycle metadata")
        corrections_through = int(pending["cycle"])
    corrections = _load_corrections(config, corrections_through) if corrections_through else []
    seen_corrections = {_position_digest(example) for example in corrections}
    anchor_cycles = {
        int(item["cycle"])
        for item in state.get("history", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("cycle"), int)
        and bool(item.get("normal_start"))
    }
    pending_state = state.get("pending")
    if (
        isinstance(pending_state, Mapping)
        and isinstance(pending_state.get("cycle"), int)
        and bool(pending_state.get("normal_start"))
    ):
        anchor_cycles.add(int(pending_state["cycle"]))

    # A completed target ends with a held-out pair. When the user raises the
    # target, consume that pair's already-saved corrections exactly once before
    # playing the next pair. This preserves both the holdout claim of the old
    # endpoint and the play -> correct -> train sequence of the extension.
    if state.get("status") == "complete" and completed < config.cycles:
        if completed <= 0 or int(state["training_updates"]) != completed - 1:
            raise TeacherCycleError("Completed curriculum is not in an extendable holdout state")
        newest = load_dataset(_correction_path(config, completed)).examples
        history = list(state.get("history", []))
        if not history or int(history[-1]["cycle"]) != completed:
            raise TeacherCycleError("Completed curriculum history is missing its holdout cycle")
        resulting_checkpoint = _train_update(
            config,
            state,
            replay,
            corrections,
            newest,
            cycle=completed,
            anchor_cycles=anchor_cycles,
            cycle_summary=history[-1],
        )
        state["training_updates"] = completed
        state["current_checkpoint"] = str(resulting_checkpoint)
        history[-1]["trained_for_later_extension"] = True
        history[-1]["extension_resulting_checkpoint"] = str(resulting_checkpoint)
        _attach_standard_evaluation(config, history[-1], resulting_checkpoint)
        state["history"] = history
        state["status"] = "ready"
        state["config_signature"] = _signature(config)
        state.setdefault("extension_points", []).append(completed)
        _atomic_json(config.state_path, state)
        LOGGER.info(
            "extended completed run at cycle=%d; trained its saved holdout corrections before "
            "starting cycle=%d",
            completed,
            completed + 1,
        )

    while int(state["completed_cycles"]) < config.cycles:
        if state.get("status") == "collected":
            pending = state["pending"]
            if not isinstance(pending, dict):
                raise TeacherCycleError("Pending curriculum state must be a mapping")
            cycle = int(pending["cycle"])
            newest = load_dataset(_correction_path(config, cycle)).examples
            if bool(pending.get("normal_start")):
                anchor_cycles.add(cycle)
        else:
            cycle = int(state["completed_cycles"]) + 1
            checkpoint = Path(str(state["current_checkpoint"]))
            newest, pending = _play_cycle(
                config,
                checkpoint,
                cycle,
                seen_corrections,
            )
            correction_path = _correction_path(config, cycle)
            save_dataset(
                correction_path,
                newest,
                metadata={
                    "kind": "teacher_corrections",
                    "curriculum_format": CURRICULUM_FORMAT,
                    "curriculum_version": CURRICULUM_VERSION,
                    "config_signature": _signature(config),
                    "cycle": cycle,
                    "opponent_depth": config.opponent_depth,
                    "teacher_depth": config.teacher_depth,
                    "normal_start": pending["normal_start"],
                    "examples_before_deduplication": pending["corrections_collected"],
                    "duplicates_discarded": pending["correction_duplicates_discarded"],
                },
            )
            pending["correction_path"] = str(correction_path)
            state["status"] = "collected"
            state["pending"] = pending
            _atomic_json(config.state_path, state)
            corrections.extend(newest)
            if bool(pending.get("normal_start")):
                anchor_cycles.add(cycle)
            LOGGER.info(
                "cycle=%d/%d played checkpoint=%s W/D/L=%d/%d/%d corrections=%d",
                cycle,
                config.cycles,
                checkpoint,
                pending["student_wins"],
                pending["student_draws"],
                pending["student_losses"],
                len(newest),
            )

        # The 100th pair is a genuine holdout: its checkpoint is the one whose
        # final performance is reported, while its corrections remain saved.
        if cycle < config.cycles:
            resulting_checkpoint = _train_update(
                config,
                state,
                replay,
                corrections,
                newest,
                cycle=cycle,
                anchor_cycles=anchor_cycles,
                cycle_summary=pending,
            )
            state["training_updates"] = int(state["training_updates"]) + 1
            state["current_checkpoint"] = str(resulting_checkpoint)
            pending["trained_after_cycle"] = True
            pending["resulting_checkpoint"] = str(resulting_checkpoint)
            _attach_standard_evaluation(config, pending, resulting_checkpoint)
        else:
            pending["trained_after_cycle"] = False
            pending["resulting_checkpoint"] = str(state["current_checkpoint"])
            pending["holdout_pair"] = True

        history = list(state.get("history", []))
        history.append(pending)
        state["history"] = history
        state["completed_cycles"] = cycle
        if state.get("first_win_cycle") is None and int(pending["student_wins"]) > 0:
            state["first_win_cycle"] = cycle
        state["ready_for_next_stage"] = _mastered(history, config.mastery_window_cycles)
        state["pending"] = None
        state["status"] = "complete" if cycle == config.cycles else "ready"
        _atomic_json(config.state_path, state)
        if state["ready_for_next_stage"]:
            LOGGER.info(
                "mastery gate currently passed: %d consecutive two-color sweeps",
                config.mastery_window_cycles,
            )

    return _summary(config, state)
