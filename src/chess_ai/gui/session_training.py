"""Per-launch, collision-safe fine-tuning orchestration for the local GUI.

This module deliberately has no Tkinter dependency.  It creates one fresh
training configuration for each GUI launch, validates that only explicitly
confirmed human-GUI data will be consumed, and invokes the existing training
CLI without a shell.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import yaml

from chess_ai.config import ConfigurationError, config_section, load_config
from chess_ai.data.examples import DatasetFormatError
from chess_ai.gui.training_data import (
    DEFAULT_HUMAN_GUI_DATASET_PATH,
    load_human_gui_dataset,
)
from chess_ai.model.checkpoint import CheckpointError, load_checkpoint
from chess_ai.model.policy_value_net import ModelConfig

SESSION_PLAN_FORMAT: Final = "self-improving-chess-ai.gui-training-session"
SESSION_PLAN_VERSION: Final = 1
DEFAULT_SESSION_CONFIG_ROOT: Final = Path("data/gui_sessions")
DEFAULT_SESSION_CHECKPOINT_ROOT: Final = Path("checkpoints/human_sessions")
DEFAULT_SESSION_METRICS_ROOT: Final = Path("data/metrics/human_sessions")
DEFAULT_SESSION_TEMPLATE: Final = Path("configs/human_gui.yaml")

_SESSION_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_DEVICE_PATTERN: Final = re.compile(r"^(?:auto|cpu|cuda(?::[0-9]+)?)$")
_REQUIRED_TRAINING_SETTINGS: Final = (
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "validation_fraction",
    "scheduler",
    "log_every",
)


class SessionTrainingError(RuntimeError):
    """Raised when a GUI training session cannot be created or trained safely."""


class SessionTrainingProcessError(SessionTrainingError):
    """Raised when the existing training CLI exits unsuccessfully."""

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True, slots=True)
class SessionTrainingPlan:
    """Versioned artifact plan allocated once for a GUI launch."""

    version: int
    session_id: str
    created_utc: str
    config_path: Path
    checkpoint_dir: Path
    metrics_path: Path
    dataset_path: Path
    template_path: Path

    @property
    def best_checkpoint(self) -> Path:
        """Expected best-candidate path after successful training."""

        return self.checkpoint_dir / "best.pt"


@dataclass(frozen=True, slots=True)
class SessionTrainingResult:
    """Captured output and best candidate from one successful fine-tuning run."""

    plan: SessionTrainingPlan
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    best_checkpoint: Path
    source_checkpoint: Path
    dataset_path: Path
    dataset_examples: int
    dataset_games: int
    dataset_created_utc: str


@dataclass(frozen=True, slots=True)
class _ValidatedTrainingInputs:
    source_model_config: ModelConfig
    dataset_examples: int
    dataset_games: int
    dataset_created_utc: str


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise SessionTrainingError(
            "Session ID must be 1-64 characters, begin with a letter or number, and contain "
            "only letters, numbers, underscores, or hyphens."
        )
    return session_id


def _normalize_created_at(created_at: datetime | None) -> datetime:
    selected = datetime.now(UTC) if created_at is None else created_at
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise SessionTrainingError("Session creation time must include a timezone.")
    return selected.astimezone(UTC)


def _generated_session_id(created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _path_text(path: Path) -> str:
    return path.as_posix()


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _validated_conservative_config(raw: dict[str, Any], source: Path) -> dict[str, Any]:
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SessionTrainingError(
            f"GUI training template {source} must contain an explicit integer seed."
        )
    try:
        training = config_section(raw, "training")
    except ConfigurationError as exc:
        raise SessionTrainingError(f"Invalid GUI training template {source}: {exc}") from exc
    missing = [name for name in _REQUIRED_TRAINING_SETTINGS if name not in training]
    if missing:
        raise SessionTrainingError(
            f"GUI training template {source} is missing conservative setting(s): "
            f"{', '.join(missing)}."
        )
    return training


def _session_metadata(plan: SessionTrainingPlan) -> dict[str, Any]:
    return {
        "format": SESSION_PLAN_FORMAT,
        "version": plan.version,
        "session_id": plan.session_id,
        "created_utc": plan.created_utc,
    }


def _config_for_plan(
    raw: dict[str, Any],
    plan: SessionTrainingPlan,
) -> dict[str, Any]:
    training = _validated_conservative_config(raw, plan.template_path)
    # A GUI button always starts a fresh run from the explicitly selected
    # source checkpoint. Never inherit a stale resume target from a template.
    training.pop("resume_from", None)
    training["dataset_path"] = _path_text(plan.dataset_path)
    training["checkpoint_dir"] = _path_text(plan.checkpoint_dir)
    training["metrics_path"] = _path_text(plan.metrics_path)
    raw["gui_session"] = _session_metadata(plan)
    return raw


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise SessionTrainingError(f"Could not write session config {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_yaml(path: Path, *, purpose: str) -> dict[str, Any]:
    try:
        return load_config(path)
    except (OSError, ConfigurationError) as exc:
        raise SessionTrainingError(f"Could not load {purpose} {path}: {exc}") from exc


def _artifact_collisions(plan: SessionTrainingPlan) -> list[Path]:
    collisions: list[Path] = []
    if plan.metrics_path.exists():
        collisions.append(plan.metrics_path)
    if plan.checkpoint_dir.exists():
        try:
            collisions.extend(sorted(plan.checkpoint_dir.iterdir()))
        except OSError as exc:
            raise SessionTrainingError(
                f"Could not inspect session checkpoint directory {plan.checkpoint_dir}: {exc}"
            ) from exc
    return collisions


def _require_no_training_artifacts(plan: SessionTrainingPlan) -> None:
    collisions = _artifact_collisions(plan)
    if collisions:
        rendered = ", ".join(str(path) for path in collisions[:3])
        raise SessionTrainingError(
            "This GUI session already has training artifacts and will not overwrite them "
            f"({rendered}). Start a new GUI session to create another candidate."
        )


def create_session_training_plan(
    dataset_path: str | Path = DEFAULT_HUMAN_GUI_DATASET_PATH,
    *,
    session_id: str | None = None,
    created_at: datetime | None = None,
    sessions_root: str | Path = DEFAULT_SESSION_CONFIG_ROOT,
    checkpoints_root: str | Path = DEFAULT_SESSION_CHECKPOINT_ROOT,
    metrics_root: str | Path = DEFAULT_SESSION_METRICS_ROOT,
    template_path: str | Path = DEFAULT_SESSION_TEMPLATE,
) -> SessionTrainingPlan:
    """Allocate and atomically configure a fresh per-launch training session.

    Supplying ``session_id`` and ``created_at`` makes artifact naming fully
    deterministic in tests.  Production IDs add a random suffix so two
    launches in the same microsecond still cannot share output paths.
    """

    created = _normalize_created_at(created_at)
    selected_id = _validate_session_id(
        _generated_session_id(created) if session_id is None else session_id
    )
    session_directory = Path(sessions_root) / selected_id
    plan = SessionTrainingPlan(
        version=SESSION_PLAN_VERSION,
        session_id=selected_id,
        created_utc=created.isoformat(),
        config_path=session_directory / "training.yaml",
        checkpoint_dir=Path(checkpoints_root) / selected_id,
        metrics_path=Path(metrics_root) / f"{selected_id}.jsonl",
        dataset_path=Path(dataset_path),
        template_path=Path(template_path),
    )

    existing = [
        path
        for path in (session_directory, plan.checkpoint_dir, plan.metrics_path)
        if path.exists()
    ]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise SessionTrainingError(
            f"Session {selected_id!r} collides with existing artifact path(s): {rendered}."
        )

    template = _load_yaml(plan.template_path, purpose="GUI training template")
    configured = _config_for_plan(template, plan)
    try:
        session_directory.parent.mkdir(parents=True, exist_ok=True)
        session_directory.mkdir(exist_ok=False)
    except OSError as exc:
        raise SessionTrainingError(
            f"Could not reserve GUI training session directory {session_directory}: {exc}"
        ) from exc
    try:
        _atomic_write_yaml(plan.config_path, configured)
    except SessionTrainingError:
        with suppress(OSError):
            session_directory.rmdir()
        raise
    return plan


def _load_and_validate_plan_config(plan: SessionTrainingPlan) -> dict[str, Any]:
    if plan.version != SESSION_PLAN_VERSION:
        raise SessionTrainingError(
            f"Session plan version {plan.version} is incompatible with version "
            f"{SESSION_PLAN_VERSION}."
        )
    _validate_session_id(plan.session_id)
    raw = _load_yaml(plan.config_path, purpose="GUI session config")
    training = _validated_conservative_config(raw, plan.config_path)
    if training.get("resume_from") is not None:
        raise SessionTrainingError(
            f"Session config {plan.config_path} cannot contain training.resume_from; "
            "automatic training always starts from the session source checkpoint."
        )
    metadata = raw.get("gui_session")
    if metadata != _session_metadata(plan):
        raise SessionTrainingError(
            f"Session config {plan.config_path} metadata does not match session "
            f"{plan.session_id!r}."
        )

    expected_paths = {
        "dataset_path": plan.dataset_path,
        "checkpoint_dir": plan.checkpoint_dir,
        "metrics_path": plan.metrics_path,
    }
    for name, expected in expected_paths.items():
        raw_path = training.get(name)
        if not isinstance(raw_path, str) or not _same_path(Path(raw_path), expected):
            raise SessionTrainingError(
                f"Session config {plan.config_path} has an unexpected training.{name}; "
                "create a new GUI session rather than editing protected artifact paths."
            )
    return raw


def update_session_dataset_path(
    plan: SessionTrainingPlan,
    dataset_path: str | Path,
) -> SessionTrainingPlan:
    """Atomically point an untrained session at the GUI's current dataset."""

    _require_no_training_artifacts(plan)
    raw = _load_and_validate_plan_config(plan)
    updated_plan = replace(plan, dataset_path=Path(dataset_path))
    training = config_section(raw, "training")
    training["dataset_path"] = _path_text(updated_plan.dataset_path)
    _atomic_write_yaml(updated_plan.config_path, raw)
    return updated_plan


def run_captured_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one argv sequence while capturing output; never invoke a shell."""

    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _validate_training_inputs(
    plan: SessionTrainingPlan,
    source_checkpoint: Path,
) -> _ValidatedTrainingInputs:
    if not source_checkpoint.is_file():
        raise SessionTrainingError(
            f"Source neural checkpoint does not exist: {source_checkpoint}. "
            "Choose the checkpoint used for this game session."
        )
    if _same_path(source_checkpoint.parent, plan.checkpoint_dir):
        raise SessionTrainingError(
            "The source checkpoint and session output directory must be different so the "
            "model you played against remains unchanged."
        )
    try:
        source = load_checkpoint(source_checkpoint, map_location="cpu")
    except CheckpointError as exc:
        raise SessionTrainingError(
            f"Source neural checkpoint is not compatible: {source_checkpoint}: {exc}"
        ) from exc

    try:
        dataset = load_human_gui_dataset(plan.dataset_path)
    except DatasetFormatError as exc:
        raise SessionTrainingError(
            f"Human-game dataset is not ready: {plan.dataset_path}: {exc} "
            "Finish and explicitly confirm at least one game before training."
        ) from exc
    if len(dataset) == 0:
        raise SessionTrainingError(
            f"Human-game dataset has no examples: {plan.dataset_path}. "
            "Finish and explicitly confirm at least one game before training."
        )
    game_count = dataset.metadata.get("completed_games")
    if isinstance(game_count, bool) or not isinstance(game_count, int) or game_count < 1:
        raise SessionTrainingError(
            f"Human-game dataset has invalid completed-game metadata: {plan.dataset_path}."
        )
    return _ValidatedTrainingInputs(
        source_model_config=source.model.config,
        dataset_examples=len(dataset),
        dataset_games=game_count,
        dataset_created_utc=dataset.created_utc,
    )


def _bounded_process_detail(stdout: str, stderr: str, *, limit: int = 4000) -> str:
    detail = stderr.strip() or stdout.strip() or "the training process produced no details"
    if len(detail) <= limit:
        return detail
    return f"...{detail[-limit:]}"


def run_session_training(
    plan: SessionTrainingPlan,
    *,
    source_checkpoint: str | Path,
    device: str = "auto",
    command_runner: CommandRunner = run_captured_command,
) -> SessionTrainingResult:
    """Validate and fine-tune through the repository's existing ``train`` CLI."""

    if not _DEVICE_PATTERN.fullmatch(device):
        raise SessionTrainingError(
            f"Unsupported training device {device!r}; use auto, cpu, cuda, or cuda:N."
        )
    _load_and_validate_plan_config(plan)
    _require_no_training_artifacts(plan)
    source = Path(source_checkpoint)
    validated = _validate_training_inputs(plan, source)

    command = (
        sys.executable,
        "-m",
        "chess_ai",
        "train",
        "--config",
        str(plan.config_path),
        "--init-checkpoint",
        str(source),
        "--device",
        device,
    )
    try:
        completed = command_runner(command)
    except OSError as exc:
        raise SessionTrainingError(
            f"Could not start the training process with {sys.executable}: {exc}"
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        detail = _bounded_process_detail(stdout, stderr)
        raise SessionTrainingProcessError(
            f"Training failed with exit code {completed.returncode}: {detail}",
            command=command,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    best_checkpoint = plan.best_checkpoint
    if not best_checkpoint.is_file():
        raise SessionTrainingError(
            "Training reported success but did not create the expected candidate checkpoint: "
            f"{best_checkpoint}. Review the captured training output before retrying in a new "
            "GUI session."
        )
    try:
        candidate = load_checkpoint(best_checkpoint, map_location="cpu")
    except CheckpointError as exc:
        raise SessionTrainingError(
            f"Training created an unreadable candidate checkpoint {best_checkpoint}: {exc}"
        ) from exc
    if candidate.model.config != validated.source_model_config:
        raise SessionTrainingError(
            "Training changed the model architecture unexpectedly; the candidate will not be "
            f"used: {best_checkpoint}."
        )
    return SessionTrainingResult(
        plan=plan,
        command=command,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        best_checkpoint=best_checkpoint,
        source_checkpoint=source,
        dataset_path=plan.dataset_path,
        dataset_examples=validated.dataset_examples,
        dataset_games=validated.dataset_games,
        dataset_created_utc=validated.dataset_created_utc,
    )
