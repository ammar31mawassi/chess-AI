"""Blockwise teacher correction with gameplay-gated champion promotion."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import chess
import torch

from chess_ai.agents import MinimaxAgent, NeuralAgent
from chess_ai.arena import run_match
from chess_ai.curriculum.teacher_cycle import (
    TeacherCycleConfig,
    _atomic_json,
    _play_cycle,
    _position_digest,
    _sample_training_examples,
)
from chess_ai.data import TrainingExample, load_dataset, save_dataset
from chess_ai.model import load_checkpoint, load_model
from chess_ai.training import TrainingConfig, train_model

GATED_FORMAT = "self-improving-chess-ai.gameplay-gated-teacher-refinement"
GATED_VERSION = 1


class GatedRefinementError(RuntimeError):
    """Raised when a gated refinement cannot continue safely."""


@dataclass(frozen=True, slots=True)
class GatedRefinementConfig:
    """Reproducible block training and D1 gameplay-gate settings."""

    initial_champion_checkpoint: Path
    replay_dataset_path: Path
    state_path: Path
    champion_dir: Path
    candidate_dir: Path
    corrections_dir: Path
    metrics_dir: Path
    training_pgn_dir: Path
    gate_pgn_dir: Path
    rounds: int = 10
    cycles_per_round: int = 10
    opponent_depth: int = 1
    teacher_depth: int = 2
    max_plies: int = 200
    opening_min_plies: int = 4
    opening_max_plies: int = 8
    normal_start_every_cycles: int = 5
    gate_openings: int = 8
    gate_opening_min_plies: int = 4
    gate_opening_max_plies: int = 8
    minimum_improvement_points: float = 0.5
    require_standard_non_regression: bool = True
    training_examples_per_round: int = 4096
    correction_fraction: float = 0.2
    anchor_correction_share: float = 0.25
    batch_size: int = 64
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.0
    legal_policy_mask: bool = True
    seed: int = 2026
    device: str = "auto"
    resume: bool = True

    def __post_init__(self) -> None:
        for name in (
            "initial_champion_checkpoint",
            "replay_dataset_path",
            "state_path",
            "champion_dir",
            "candidate_dir",
            "corrections_dir",
            "metrics_dir",
            "training_pgn_dir",
            "gate_pgn_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "rounds",
            "cycles_per_round",
            "opponent_depth",
            "teacher_depth",
            "max_plies",
            "gate_openings",
            "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.teacher_depth <= self.opponent_depth:
            raise ValueError("teacher_depth must be greater than opponent_depth")
        if self.training_examples_per_round <= 1:
            raise ValueError("training_examples_per_round must be greater than one")
        if not 0.0 < self.correction_fraction < 1.0:
            raise ValueError("correction_fraction must be between 0 and 1")
        if isinstance(self.anchor_correction_share, bool) or not (
            0.0 <= self.anchor_correction_share <= 1.0
        ):
            raise ValueError("anchor_correction_share must be between 0 and 1")
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
        if self.minimum_improvement_points <= 0.0:
            raise ValueError("minimum_improvement_points must be positive")
        if not isinstance(self.require_standard_non_regression, bool):
            raise ValueError("require_standard_non_regression must be a boolean")
        if (
            isinstance(self.normal_start_every_cycles, bool)
            or not isinstance(self.normal_start_every_cycles, int)
            or self.normal_start_every_cycles < 0
        ):
            raise ValueError("normal_start_every_cycles must be a non-negative integer")
        self._validate_opening_range(
            self.opening_min_plies,
            self.opening_max_plies,
            "training opening",
        )
        self._validate_opening_range(
            self.gate_opening_min_plies,
            self.gate_opening_max_plies,
            "gate opening",
        )

    def _validate_opening_range(self, minimum: int, maximum: int, label: str) -> None:
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"{label} bounds must satisfy 0 <= minimum <= maximum")
        if not any(plies % 2 == 0 for plies in range(minimum, maximum + 1)):
            raise ValueError(f"{label} bounds must contain at least one even length")
        if maximum >= self.max_plies:
            raise ValueError(f"{label} maximum must be smaller than max_plies")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        seed: int = 2026,
        device: str = "auto",
        resume: bool | None = None,
    ) -> GatedRefinementConfig:
        values = dict(raw)
        values.setdefault("seed", seed)
        values.setdefault("device", device)
        if resume is not None:
            values["resume"] = resume
        path_fields = {
            "initial_champion_checkpoint",
            "replay_dataset_path",
            "state_path",
            "champion_dir",
            "candidate_dir",
            "corrections_dir",
            "metrics_dir",
            "training_pgn_dir",
            "gate_pgn_dir",
        }
        for name in path_fields:
            if name in values:
                values[name] = Path(str(values[name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown gated_refinement settings: {', '.join(unknown)}")
        missing = sorted(path_fields.difference(values))
        if missing:
            raise ValueError(f"Missing gated_refinement settings: {', '.join(missing)}")
        return cls(**values)

    def signature_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("resume", None)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True, slots=True)
class GatedRefinementSummary:
    completed_rounds: int
    requested_rounds: int
    training_cycles: int
    training_games: int
    gate_games: int
    promotions: int
    champion_generation: int
    champion_checkpoint: Path
    state_path: Path
    interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("champion_checkpoint", "state_path"):
            payload[name] = str(payload[name])
        return payload


@dataclass(frozen=True, slots=True)
class GatedHoldoutAuditSummary:
    baseline_checkpoint: Path
    candidate_checkpoint: Path
    audit_seed: int
    openings: int
    games_per_checkpoint: int
    baseline_wins: int
    baseline_draws: int
    baseline_losses: int
    baseline_points: float
    candidate_wins: int
    candidate_draws: int
    candidate_losses: int
    candidate_points: float
    point_delta: float
    baseline_standard_points: float
    candidate_standard_points: float
    verdict: str
    supports_continuation: bool
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("baseline_checkpoint", "candidate_checkpoint", "report_path"):
            payload[name] = str(payload[name])
        return payload


def _signature(config: GatedRefinementConfig) -> str:
    encoded = json.dumps(config.signature_payload(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_checkpoint(source: Path, destination: Path) -> Path:
    """Create or verify an immutable, byte-identical champion snapshot."""

    load_checkpoint(source, map_location="cpu")
    if destination.exists():
        if _checkpoint_hash(source) != _checkpoint_hash(destination):
            raise GatedRefinementError(
                f"Champion snapshot already exists with different content: {destination}"
            )
        load_checkpoint(destination, map_location="cpu")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        raise GatedRefinementError(
            f"Could not create champion snapshot {destination}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    load_checkpoint(destination, map_location="cpu")
    return destination


def _champion_manifest(config: GatedRefinementConfig, state: Mapping[str, Any]) -> None:
    checkpoint = Path(str(state["champion_checkpoint"]))
    _atomic_json(
        config.champion_dir / "champion.json",
        {
            "format": GATED_FORMAT,
            "version": GATED_VERSION,
            "generation": int(state["champion_generation"]),
            "checkpoint": str(checkpoint),
            "sha256": _checkpoint_hash(checkpoint),
            "initial_source": str(config.initial_champion_checkpoint),
            "selection": "fixed D1 gameplay gate; not training loss",
        },
    )


def _initial_state(config: GatedRefinementConfig) -> dict[str, Any]:
    champion = _snapshot_checkpoint(
        config.initial_champion_checkpoint,
        config.champion_dir / "generation_0000.pt",
    )
    state: dict[str, Any] = {
        "format": GATED_FORMAT,
        "version": GATED_VERSION,
        "config_signature": _signature(config),
        "status": "ready",
        "completed_rounds": 0,
        "completed_training_cycles": 0,
        "champion_generation": 0,
        "champion_checkpoint": str(champion),
        "initial_champion_checkpoint": str(config.initial_champion_checkpoint),
        "history": [],
        "active_round": None,
    }
    _atomic_json(config.state_path, state)
    _champion_manifest(config, state)
    return state


def _output_conflicts(config: GatedRefinementConfig) -> list[Path]:
    roots = (
        config.candidate_dir,
        config.corrections_dir,
        config.metrics_dir,
        config.training_pgn_dir,
        config.gate_pgn_dir,
    )
    conflicts = [
        path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()
    ]
    allowed_snapshot = config.champion_dir / "generation_0000.pt"
    allowed_manifest = config.champion_dir / "champion.json"
    if config.champion_dir.exists():
        conflicts.extend(
            path
            for path in config.champion_dir.rglob("*")
            if path.is_file() and path not in {allowed_snapshot, allowed_manifest}
        )
    return conflicts


def _load_state(config: GatedRefinementConfig) -> dict[str, Any]:
    if not config.state_path.exists():
        conflicts = _output_conflicts(config)
        if conflicts:
            preview = ", ".join(str(path) for path in conflicts[:3])
            raise GatedRefinementError(
                "Gated outputs already exist without their state file: "
                f"{preview}. Restore the state or choose fresh output paths."
            )
        return _initial_state(config)
    if not config.resume:
        raise GatedRefinementError(
            f"Gated state already exists at {config.state_path}; enable resume or use fresh paths."
        )
    try:
        raw = json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatedRefinementError(
            f"Could not read gated state {config.state_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise GatedRefinementError("Gated state root must be a mapping")
    if raw.get("format") != GATED_FORMAT or raw.get("version") != GATED_VERSION:
        raise GatedRefinementError("Gated state format/version is incompatible with this code")
    if raw.get("config_signature") != _signature(config):
        completed = raw.get("completed_rounds")
        extendable = (
            isinstance(completed, int)
            and raw.get("status") == "complete"
            and config.rounds > completed
            and raw.get("config_signature") == _signature(replace(config, rounds=completed))
        )
        if not extendable:
            raise GatedRefinementError(
                "Gated configuration differs from saved state. Only increasing rounds on a "
                "completed run is allowed."
            )
        raw["config_signature"] = _signature(config)
        raw["status"] = "ready"
        _atomic_json(config.state_path, raw)
    return raw


def _validate_paths(config: GatedRefinementConfig) -> None:
    if not config.initial_champion_checkpoint.is_file():
        raise GatedRefinementError(
            f"Initial champion checkpoint does not exist: {config.initial_champion_checkpoint}"
        )
    if not config.replay_dataset_path.is_file():
        raise GatedRefinementError(f"Replay dataset does not exist: {config.replay_dataset_path}")
    load_checkpoint(config.initial_champion_checkpoint, map_location="cpu")
    load_dataset(config.replay_dataset_path)
    roots = (
        config.champion_dir,
        config.candidate_dir,
        config.corrections_dir,
        config.metrics_dir,
        config.training_pgn_dir,
        config.gate_pgn_dir,
    )
    resolved = [path.resolve() for path in roots]
    if len(set(resolved)) != len(resolved):
        raise GatedRefinementError("Gated artifact directories must all be different")
    if config.initial_champion_checkpoint.resolve().is_relative_to(config.champion_dir.resolve()):
        raise GatedRefinementError("champion_dir must not contain the initial champion checkpoint")


def _validate_state(config: GatedRefinementConfig, state: Mapping[str, Any]) -> None:
    history = state.get("history")
    completed = state.get("completed_rounds")
    cycles = state.get("completed_training_cycles")
    if (
        not isinstance(history, list)
        or not isinstance(completed, int)
        or not isinstance(cycles, int)
    ):
        raise GatedRefinementError("Gated state counters/history are invalid")
    if len(history) != completed:
        raise GatedRefinementError("Gated history does not match completed_rounds")
    if cycles < completed * config.cycles_per_round:
        raise GatedRefinementError("Gated training-cycle counter is incomplete")
    if completed < 0 or completed > config.rounds:
        raise GatedRefinementError("completed_rounds is outside the configured target")
    if state.get("status") not in {"ready", "collecting", "evaluating", "complete"}:
        raise GatedRefinementError(f"Gated state status is invalid: {state.get('status')!r}")
    champion = Path(str(state.get("champion_checkpoint", "")))
    if not champion.is_file():
        raise GatedRefinementError(f"Saved champion checkpoint is missing: {champion}")
    load_checkpoint(champion, map_location="cpu")
    active = state.get("active_round")
    if state.get("status") in {"collecting", "evaluating"}:
        if not isinstance(active, dict) or active.get("round") != completed + 1:
            raise GatedRefinementError("Active gated round does not follow completed history")
    elif active is not None:
        raise GatedRefinementError("Inactive gated state unexpectedly contains active_round")


def _teacher_config(
    config: GatedRefinementConfig,
    champion: Path,
    *,
    round_number: int,
) -> TeacherCycleConfig:
    total_cycles = max(2, config.rounds * config.cycles_per_round)
    return TeacherCycleConfig(
        source_checkpoint=champion,
        replay_dataset_path=config.replay_dataset_path,
        state_path=config.state_path,
        corrections_dir=config.corrections_dir,
        checkpoint_dir=config.candidate_dir,
        metrics_path=config.metrics_dir / "unused.jsonl",
        pgn_dir=config.training_pgn_dir / f"round_{round_number:04d}",
        cycles=total_cycles,
        opponent_depth=config.opponent_depth,
        teacher_depth=config.teacher_depth,
        max_plies=config.max_plies,
        opening_min_plies=config.opening_min_plies,
        opening_max_plies=config.opening_max_plies,
        normal_start_every_cycles=config.normal_start_every_cycles,
        anchor_correction_share=config.anchor_correction_share,
        evaluate_after_normal_start=False,
        training_examples_per_cycle=config.training_examples_per_round,
        correction_fraction=config.correction_fraction,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        legal_policy_mask=config.legal_policy_mask,
        mastery_window_cycles=1,
        seed=config.seed,
        device=config.device,
        resume=True,
    )


def _round_correction_path(
    config: GatedRefinementConfig,
    round_number: int,
    global_cycle: int,
) -> Path:
    return config.corrections_dir / f"round_{round_number:04d}" / f"cycle_{global_cycle:04d}.pt"


def _load_saved_corrections(state: Mapping[str, Any]) -> tuple[list[TrainingExample], set[int]]:
    corrections: list[TrainingExample] = []
    anchor_cycles: set[int] = set()
    rounds: list[Mapping[str, Any]] = list(state.get("history", []))
    active = state.get("active_round")
    if isinstance(active, Mapping):
        rounds.append(active)
    for round_item in rounds:
        cycles = round_item.get("cycles", [])
        if not isinstance(cycles, list):
            raise GatedRefinementError("Saved gated round has invalid cycle history")
        for cycle in cycles:
            if not isinstance(cycle, Mapping):
                raise GatedRefinementError("Saved gated cycle is invalid")
            path = Path(str(cycle.get("correction_path", "")))
            if not path.is_file():
                raise GatedRefinementError(f"Saved gated correction file is missing: {path}")
            corrections.extend(load_dataset(path).examples)
            if bool(cycle.get("normal_start")):
                anchor_cycles.add(int(cycle["cycle"]))
    return corrections, anchor_cycles


def _candidate_checkpoint(config: GatedRefinementConfig, round_number: int) -> Path | None:
    directory = config.candidate_dir / f"round_{round_number:04d}"
    for name in ("best.pt", "last.pt", "epoch_0001.pt"):
        path = directory / name
        if path.is_file():
            loaded = load_checkpoint(path, map_location="cpu")
            if loaded.epoch != 1:
                raise GatedRefinementError(f"Candidate checkpoint has wrong epoch: {path}")
            return path
    if any(directory.glob("*.pt")):
        raise GatedRefinementError(
            f"Candidate directory contains unusable checkpoints: {directory}"
        )
    return None


def _train_candidate(
    config: GatedRefinementConfig,
    active: dict[str, Any],
    champion: Path,
    replay: Sequence[TrainingExample],
    corrections: Sequence[TrainingExample],
    anchor_cycles: set[int],
) -> Path:
    round_number = int(active["round"])
    recovered = _candidate_checkpoint(config, round_number)
    if recovered is not None:
        active["candidate_checkpoint"] = str(recovered)
        active.setdefault("candidate_recovered_after_interruption", True)
        return recovered
    newest_paths = [Path(str(item["correction_path"])) for item in active["cycles"]]
    newest = [example for path in newest_paths for example in load_dataset(path).examples]
    teacher_config = _teacher_config(
        config,
        champion,
        round_number=round_number,
    )
    sample_stats: dict[str, int | float] = {}
    training_examples = _sample_training_examples(
        teacher_config,
        replay,
        corrections,
        newest,
        cycle=round_number,
        anchor_cycles=anchor_cycles,
        sample_stats=sample_stats,
    )
    active["training_batch"] = sample_stats
    candidate_directory = config.candidate_dir / f"round_{round_number:04d}"
    metrics_path = config.metrics_dir / f"round_{round_number:04d}.jsonl"
    if metrics_path.exists() and metrics_path.read_text(encoding="utf-8").strip():
        raise GatedRefinementError(
            f"Candidate metrics exist without a recoverable checkpoint: {metrics_path}"
        )
    model = load_model(champion, device="cpu", eval_mode=False)
    training_config = TrainingConfig(
        checkpoint_dir=candidate_directory,
        metrics_path=metrics_path,
        batch_size=min(config.batch_size, len(training_examples)),
        epochs=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        legal_policy_mask=config.legal_policy_mask,
        gradient_clip=config.gradient_clip,
        validation_fraction=0.0,
        scheduler=False,
        seed=config.seed + round_number,
        device=config.device,
        log_every=20,
    )
    try:
        history = train_model(model, training_config, training_examples)
        if len(history) != 1 or history[0].epoch != 1:
            raise GatedRefinementError("Candidate training did not complete its one expected epoch")
        active["training_metrics"] = history[0].to_dict()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    candidate = candidate_directory / "best.pt"
    if not candidate.is_file():
        raise GatedRefinementError(f"Candidate training did not write {candidate}")
    active["candidate_checkpoint"] = str(candidate)
    return candidate


def _seeded_suite_opening(
    config: GatedRefinementConfig,
    opening_index: int,
    *,
    suite_seed: int,
) -> dict[str, Any]:
    if opening_index == 0:
        return {
            "index": 0,
            "seed": suite_seed,
            "fen": chess.Board().fen(),
            "moves": [],
            "standard_start": True,
        }
    lengths = [
        plies
        for plies in range(config.gate_opening_min_plies, config.gate_opening_max_plies + 1)
        if plies % 2 == 0
    ]
    base_seed = suite_seed + opening_index * 1_000_003
    for attempt in range(100):
        rng = random.Random(base_seed + attempt)
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
            return {
                "index": opening_index,
                "seed": base_seed + attempt,
                "fen": board.fen(),
                "moves": moves,
                "standard_start": False,
            }
    raise GatedRefinementError(f"Could not create suite opening {opening_index}")


def _seeded_gate_opening(
    config: GatedRefinementConfig,
    opening_index: int,
) -> dict[str, Any]:
    return _seeded_suite_opening(
        config,
        opening_index,
        suite_seed=config.seed + 50_000_003,
    )


def _seeded_holdout_openings(
    config: GatedRefinementConfig,
    *,
    openings: int,
    audit_seed: int,
) -> list[dict[str, Any]]:
    if isinstance(openings, bool) or not isinstance(openings, int) or openings <= 0:
        raise ValueError("audit openings must be a positive integer")
    if isinstance(audit_seed, bool) or not isinstance(audit_seed, int):
        raise TypeError("audit_seed must be an integer")
    gate_fens = {
        str(_seeded_gate_opening(config, index)["fen"]) for index in range(1, config.gate_openings)
    }
    selected: list[dict[str, Any]] = []
    selected_fens: set[str] = set()
    for opening_index in range(openings):
        for collision_attempt in range(100):
            suite_seed = audit_seed + collision_attempt * 10_000_019
            opening = _seeded_suite_opening(
                config,
                opening_index,
                suite_seed=suite_seed,
            )
            fen = str(opening["fen"])
            allowed_standard = opening_index == 0 and fen == chess.Board().fen()
            if allowed_standard or (fen not in gate_fens and fen not in selected_fens):
                selected.append(opening)
                selected_fens.add(fen)
                break
        else:  # pragma: no cover - random opening space is enormous
            raise GatedRefinementError(
                f"Could not create unseen, unique holdout opening {opening_index}"
            )
    return selected


def _student_outcome(result: str, student_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    return "win" if (result == "1-0") == (student_color == chess.WHITE) else "loss"


def _evaluate_checkpoint(
    config: GatedRefinementConfig,
    checkpoint: Path,
    *,
    role: str,
    suite_name: str,
    suite_number: int,
    suite_seed: int,
    output_dir: Path,
    openings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    student = NeuralAgent(
        checkpoint,
        device=config.device,
        deterministic=True,
        temperature=0.0,
        seed=config.seed,
    )
    games: list[dict[str, Any]] = []
    try:
        for opening in openings:
            opening_index = int(opening["index"])
            opening_moves = list(opening["moves"])
            for student_color in (chess.WHITE, chess.BLACK):
                color_name = "white" if student_color == chess.WHITE else "black"
                game_seed = suite_seed + opening_index * 10_007 + int(student_color)
                opponent = MinimaxAgent(
                    depth=config.opponent_depth,
                    deterministic=True,
                    seed=game_seed,
                    name=f"Minimax D{config.opponent_depth} gate opponent",
                )
                white = student if student_color == chess.WHITE else opponent
                black = opponent if student_color == chess.WHITE else student
                pgn_path = (
                    output_dir / f"opening_{opening_index:02d}_{role}_student_{color_name}.pgn"
                )
                result = run_match(
                    white,
                    black,
                    max_plies=config.max_plies - len(opening_moves),
                    starting_fen=str(opening["fen"]),
                    seed=game_seed,
                    pgn_path=pgn_path,
                    extra_headers={
                        "GatedFormat": GATED_FORMAT,
                        "GatedVersion": str(GATED_VERSION),
                        "EvaluationSuite": suite_name,
                        "SuiteNumber": str(suite_number),
                        "SuiteRole": role,
                        "SuiteOpening": str(opening_index),
                        "SuiteOpeningSeed": str(opening["seed"]),
                        "StandardStart": str(bool(opening["standard_start"])).lower(),
                        "UnrecordedOpeningMoves": " ".join(opening_moves),
                        "StudentColor": color_name,
                        "StudentCheckpoint": str(checkpoint),
                        "OpponentDepth": str(config.opponent_depth),
                    },
                )
                games.append(
                    {
                        "opening": opening_index,
                        "standard_start": bool(opening["standard_start"]),
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
    standard = [game for game in games if game["standard_start"]]
    standard_wins = sum(game["student_outcome"] == "win" for game in standard)
    standard_draws = sum(game["student_outcome"] == "draw" for game in standard)
    return {
        "role": role,
        "checkpoint": str(checkpoint),
        "games": games,
        "student_wins": wins,
        "student_draws": draws,
        "student_losses": losses,
        "score_points": wins + 0.5 * draws,
        "standard_score_points": standard_wins + 0.5 * standard_draws,
    }


def _promotion_decision(
    config: GatedRefinementConfig,
    champion: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[bool, str]:
    improvement = float(candidate["score_points"]) - float(champion["score_points"])
    standard_regressed = float(candidate["standard_score_points"]) < float(
        champion["standard_score_points"]
    )
    if config.require_standard_non_regression and standard_regressed:
        return False, "rejected: candidate regressed on the two-color standard-start gate"
    if improvement < config.minimum_improvement_points:
        return (
            False,
            "rejected: candidate did not reach the required fixed-suite score improvement",
        )
    return True, "promoted: candidate improved the fixed suite without standard regression"


def _round_summary(active: Mapping[str, Any]) -> dict[str, int]:
    cycles = list(active.get("cycles", []))
    return {
        "student_wins": sum(int(item["student_wins"]) for item in cycles),
        "student_draws": sum(int(item["student_draws"]) for item in cycles),
        "student_losses": sum(int(item["student_losses"]) for item in cycles),
        "corrections_retained": sum(int(item["corrections_retained"]) for item in cycles),
    }


def _summary(config: GatedRefinementConfig, state: Mapping[str, Any]) -> GatedRefinementSummary:
    history = list(state.get("history", []))
    return GatedRefinementSummary(
        completed_rounds=int(state["completed_rounds"]),
        requested_rounds=config.rounds,
        training_cycles=int(state["completed_training_cycles"]),
        training_games=int(state["completed_training_cycles"]) * 2,
        gate_games=sum(
            len(item.get("gate", {}).get("champion", {}).get("games", []))
            + len(item.get("gate", {}).get("candidate", {}).get("games", []))
            for item in history
        ),
        promotions=sum(bool(item.get("promoted")) for item in history),
        champion_generation=int(state["champion_generation"]),
        champion_checkpoint=Path(str(state["champion_checkpoint"])),
        state_path=config.state_path,
    )


def run_gated_refinement(config: GatedRefinementConfig) -> GatedRefinementSummary:
    """Train candidates from a frozen champion and promote only through gameplay."""

    _validate_paths(config)
    replay = list(load_dataset(config.replay_dataset_path).examples)
    state = _load_state(config)
    _validate_state(config, state)

    while int(state["completed_rounds"]) < config.rounds:
        round_number = int(state["completed_rounds"]) + 1
        champion = Path(str(state["champion_checkpoint"]))
        if state.get("status") == "ready":
            state["active_round"] = {
                "round": round_number,
                "starting_champion_generation": int(state["champion_generation"]),
                "starting_champion_checkpoint": str(champion),
                "cycles": [],
            }
            state["status"] = "collecting"
            _atomic_json(config.state_path, state)
        active = state.get("active_round")
        if not isinstance(active, dict):
            raise GatedRefinementError("Active gated round must be a mapping")

        corrections, anchor_cycles = _load_saved_corrections(state)
        seen_corrections = {_position_digest(example) for example in corrections}
        teacher_config = _teacher_config(config, champion, round_number=round_number)
        while len(active["cycles"]) < config.cycles_per_round:
            global_cycle = int(state["completed_training_cycles"]) + 1
            newest, cycle_summary = _play_cycle(
                teacher_config,
                champion,
                global_cycle,
                seen_corrections,
            )
            correction_path = _round_correction_path(config, round_number, global_cycle)
            save_dataset(
                correction_path,
                newest,
                metadata={
                    "kind": "gated_teacher_corrections",
                    "gated_format": GATED_FORMAT,
                    "gated_version": GATED_VERSION,
                    "round": round_number,
                    "cycle": global_cycle,
                    "champion_checkpoint": str(champion),
                    "opponent_depth": config.opponent_depth,
                    "teacher_depth": config.teacher_depth,
                    "normal_start": cycle_summary["normal_start"],
                    "examples_before_deduplication": cycle_summary["corrections_collected"],
                    "duplicates_discarded": cycle_summary["correction_duplicates_discarded"],
                },
            )
            cycle_summary["correction_path"] = str(correction_path)
            active["cycles"].append(cycle_summary)
            state["completed_training_cycles"] = global_cycle
            corrections.extend(newest)
            if bool(cycle_summary["normal_start"]):
                anchor_cycles.add(global_cycle)
            _atomic_json(config.state_path, state)

        if state["status"] == "collecting":
            candidate = _train_candidate(
                config,
                active,
                champion,
                replay,
                corrections,
                anchor_cycles,
            )
            active["candidate_checkpoint"] = str(candidate)
            state["status"] = "evaluating"
            _atomic_json(config.state_path, state)
        else:
            candidate = Path(str(active["candidate_checkpoint"]))
            load_checkpoint(candidate, map_location="cpu")

        openings = [_seeded_gate_opening(config, index) for index in range(config.gate_openings)]
        champion_gate = _evaluate_checkpoint(
            config,
            champion,
            role="champion",
            suite_name="promotion_gate",
            suite_number=round_number,
            suite_seed=config.seed,
            output_dir=config.gate_pgn_dir / f"round_{round_number:04d}",
            openings=openings,
        )
        candidate_gate = _evaluate_checkpoint(
            config,
            candidate,
            role="candidate",
            suite_name="promotion_gate",
            suite_number=round_number,
            suite_seed=config.seed,
            output_dir=config.gate_pgn_dir / f"round_{round_number:04d}",
            openings=openings,
        )
        promoted, reason = _promotion_decision(config, champion_gate, candidate_gate)
        active["gate"] = {
            "openings": openings,
            "champion": champion_gate,
            "candidate": candidate_gate,
            "minimum_improvement_points": config.minimum_improvement_points,
            "require_standard_non_regression": config.require_standard_non_regression,
        }
        active.update(_round_summary(active))
        active["promoted"] = promoted
        active["decision"] = reason
        if promoted:
            generation = int(state["champion_generation"]) + 1
            promoted_checkpoint = _snapshot_checkpoint(
                candidate,
                config.champion_dir / f"generation_{generation:04d}.pt",
            )
            state["champion_generation"] = generation
            state["champion_checkpoint"] = str(promoted_checkpoint)
            active["resulting_champion_checkpoint"] = str(promoted_checkpoint)
        else:
            active["resulting_champion_checkpoint"] = str(champion)

        history = list(state.get("history", []))
        history.append(active)
        state["history"] = history
        state["completed_rounds"] = round_number
        state["active_round"] = None
        state["status"] = "complete" if round_number == config.rounds else "ready"
        _champion_manifest(config, state)
        _atomic_json(config.state_path, state)

    return _summary(config, state)


def initialize_gated_refinement(config: GatedRefinementConfig) -> GatedRefinementSummary:
    """Create/validate generation zero without collecting games or training."""

    _validate_paths(config)
    state = _load_state(config)
    _validate_state(config, state)
    _champion_manifest(config, state)
    return _summary(config, state)


def _read_state_for_audit(config: GatedRefinementConfig) -> dict[str, Any]:
    """Load completed state without extending or otherwise mutating it."""

    if not config.state_path.is_file():
        raise GatedRefinementError(
            f"Gated state does not exist yet: {config.state_path}. Run the refinement first."
        )
    try:
        raw = json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatedRefinementError(
            f"Could not read gated state {config.state_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise GatedRefinementError("Gated state root must be a mapping")
    if raw.get("format") != GATED_FORMAT or raw.get("version") != GATED_VERSION:
        raise GatedRefinementError("Gated state format/version is incompatible with this code")
    completed = raw.get("completed_rounds")
    signature_matches = raw.get("config_signature") == _signature(config)
    extension_config_matches = (
        isinstance(completed, int)
        and config.rounds >= completed
        and raw.get("config_signature") == _signature(replace(config, rounds=completed))
    )
    if not signature_matches and not extension_config_matches:
        raise GatedRefinementError("Audit configuration differs from the saved gated state")
    _validate_state(config, raw)
    if raw.get("status") != "complete" or not isinstance(completed, int) or completed <= 0:
        raise GatedRefinementError("Holdout audit requires at least one completed gated run")
    return raw


def run_gated_holdout_audit(
    config: GatedRefinementConfig,
    *,
    openings: int = 20,
    audit_seed: int | None = None,
) -> GatedHoldoutAuditSummary:
    """Compare generation zero and the champion on unseen D1 starts without promotion."""

    _validate_paths(config)
    state = _read_state_for_audit(config)
    selected_seed = config.seed + 90_000_019 if audit_seed is None else audit_seed
    holdout_openings = _seeded_holdout_openings(
        config,
        openings=openings,
        audit_seed=selected_seed,
    )
    baseline = config.champion_dir / "generation_0000.pt"
    candidate = Path(str(state["champion_checkpoint"]))
    load_checkpoint(baseline, map_location="cpu")
    load_checkpoint(candidate, map_location="cpu")
    audit_name = f"{baseline.stem}_vs_{candidate.stem}_seed_{selected_seed}_openings_{openings}"
    output_dir = config.gate_pgn_dir.parent / "holdout" / audit_name
    baseline_result = _evaluate_checkpoint(
        config,
        baseline,
        role="baseline",
        suite_name="unseen_holdout",
        suite_number=int(state["champion_generation"]),
        suite_seed=selected_seed,
        output_dir=output_dir,
        openings=holdout_openings,
    )
    candidate_result = _evaluate_checkpoint(
        config,
        candidate,
        role="candidate",
        suite_name="unseen_holdout",
        suite_number=int(state["champion_generation"]),
        suite_seed=selected_seed,
        output_dir=output_dir,
        openings=holdout_openings,
    )
    point_delta = float(candidate_result["score_points"]) - float(baseline_result["score_points"])
    standard_regressed = float(candidate_result["standard_score_points"]) < float(
        baseline_result["standard_score_points"]
    )
    if standard_regressed or point_delta < 0.0:
        verdict = "regressed"
    elif point_delta > 0.0:
        verdict = "improved"
    else:
        verdict = "tied"
    supports_continuation = not standard_regressed and point_delta >= 0.0
    report_path = output_dir / "report.json"
    summary = GatedHoldoutAuditSummary(
        baseline_checkpoint=baseline,
        candidate_checkpoint=candidate,
        audit_seed=selected_seed,
        openings=openings,
        games_per_checkpoint=len(baseline_result["games"]),
        baseline_wins=int(baseline_result["student_wins"]),
        baseline_draws=int(baseline_result["student_draws"]),
        baseline_losses=int(baseline_result["student_losses"]),
        baseline_points=float(baseline_result["score_points"]),
        candidate_wins=int(candidate_result["student_wins"]),
        candidate_draws=int(candidate_result["student_draws"]),
        candidate_losses=int(candidate_result["student_losses"]),
        candidate_points=float(candidate_result["score_points"]),
        point_delta=point_delta,
        baseline_standard_points=float(baseline_result["standard_score_points"]),
        candidate_standard_points=float(candidate_result["standard_score_points"]),
        verdict=verdict,
        supports_continuation=supports_continuation,
        report_path=report_path,
    )
    _atomic_json(
        report_path,
        {
            "format": GATED_FORMAT,
            "version": GATED_VERSION,
            "kind": "unseen_holdout_audit",
            "promotion_performed": False,
            "gate_random_opening_overlap": False,
            "summary": summary.to_dict(),
            "openings": holdout_openings,
            "baseline": baseline_result,
            "candidate": candidate_result,
        },
    )
    return summary
