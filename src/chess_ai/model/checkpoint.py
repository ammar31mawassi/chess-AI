"""Atomic and version-checked neural-network checkpoints."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from chess_ai.model.policy_value_net import ModelConfig, PolicyValueNet

CHECKPOINT_FORMAT = "self-improving-chess-ai.policy-value-checkpoint"
CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when checkpoint I/O or structure is invalid."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when a checkpoint cannot safely be used by this code/model."""


Scheduler = LRScheduler | ReduceLROnPlateau


@dataclass(slots=True)
class LoadedCheckpoint:
    """The reconstructed model and resumable training state."""

    model: PolicyValueNet
    epoch: int
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        # os.replace is atomic when source and destination share a filesystem.
        os.replace(temporary, destination)
    except (OSError, RuntimeError, TypeError, pickle.PickleError) as exc:
        raise CheckpointError(f"Could not save checkpoint to {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _model_config(model: nn.Module) -> ModelConfig:
    config = getattr(model, "config", None)
    if not isinstance(config, ModelConfig):
        raise CheckpointError(
            "The model does not expose a ModelConfig as 'model.config'; "
            "only PolicyValueNet-compatible models can be checkpointed."
        )
    return config


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    scheduler: Scheduler | None = None,
    epoch: int = 0,
    metrics: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
    dataset_version: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save model/training state using a replace-on-success write.

    Shape-defining model settings and a format version are mandatory parts of
    the payload, so an incompatible file fails before weights are applied.
    """

    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    destination = Path(path)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "model_config": _model_config(model).to_dict(),
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "training_config": dict(training_config or {}),
        "dataset_version": dataset_version,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "extra": dict(extra or {}),
    }
    _atomic_torch_save(payload, destination)
    return destination


def _read_payload(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    try:
        try:
            raw = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch
            raw = torch.load(path, map_location=map_location)
    except (EOFError, OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise IncompatibleCheckpointError("Checkpoint root must be a mapping")
    if raw.get("format") != CHECKPOINT_FORMAT:
        raise IncompatibleCheckpointError(
            f"Unsupported checkpoint format {raw.get('format')!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    version = raw.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise IncompatibleCheckpointError(
            f"Unsupported checkpoint version {version!r}; this code supports "
            f"version {CHECKPOINT_VERSION}"
        )
    for key in ("model_config", "model_state_dict", "epoch"):
        if key not in raw:
            raise IncompatibleCheckpointError(f"Checkpoint is missing required field {key!r}")
    return raw


def load_checkpoint(
    path: str | Path,
    model: PolicyValueNet | None = None,
    *,
    optimizer: Optimizer | None = None,
    scheduler: Scheduler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> LoadedCheckpoint:
    """Load a checkpoint, optionally restoring supplied training objects."""

    source = Path(path)
    payload = _read_payload(source, map_location)
    raw_config = payload["model_config"]
    if not isinstance(raw_config, dict):
        raise IncompatibleCheckpointError("Checkpoint model_config must be a mapping")
    try:
        saved_config = ModelConfig.from_dict(raw_config)
    except ValueError as exc:
        raise IncompatibleCheckpointError(str(exc)) from exc

    restored_model = model or PolicyValueNet(config=saved_config)
    current_config = _model_config(restored_model)
    if current_config != saved_config:
        raise IncompatibleCheckpointError(
            "Checkpoint architecture does not match the supplied model: "
            f"saved={saved_config.to_dict()}, supplied={current_config.to_dict()}"
        )
    try:
        restored_model.load_state_dict(payload["model_state_dict"], strict=strict)
    except (RuntimeError, TypeError) as exc:
        raise IncompatibleCheckpointError(f"Checkpoint weights are incompatible: {exc}") from exc

    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None:
        if optimizer_state is None:
            raise IncompatibleCheckpointError("Checkpoint has no optimizer state to resume")
        try:
            optimizer.load_state_dict(optimizer_state)
        except (ValueError, RuntimeError) as exc:
            raise IncompatibleCheckpointError(
                f"Checkpoint optimizer state is incompatible: {exc}"
            ) from exc

    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None:
        if scheduler_state is None:
            raise IncompatibleCheckpointError("Checkpoint has no scheduler state to resume")
        try:
            scheduler.load_state_dict(scheduler_state)
        except (ValueError, RuntimeError) as exc:
            raise IncompatibleCheckpointError(
                f"Checkpoint scheduler state is incompatible: {exc}"
            ) from exc

    metrics = payload.get("metrics", {})
    metadata = {
        key: payload.get(key)
        for key in (
            "format",
            "format_version",
            "created_utc",
            "torch_version",
            "model_config",
            "training_config",
            "dataset_version",
        )
    }
    return LoadedCheckpoint(
        model=restored_model,
        epoch=int(payload["epoch"]),
        metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
        metadata=metadata,
        optimizer_state_dict=optimizer_state if isinstance(optimizer_state, dict) else None,
        scheduler_state_dict=scheduler_state if isinstance(scheduler_state, dict) else None,
        extra=dict(payload.get("extra", {}))
        if isinstance(payload.get("extra", {}), Mapping)
        else {},
    )


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> PolicyValueNet:
    """Convenience loader used by inference agents."""

    target = torch.device(device)
    loaded = load_checkpoint(path, map_location=target)
    loaded.model.to(target)
    if eval_mode:
        loaded.model.eval()
    return loaded.model
