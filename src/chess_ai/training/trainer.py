"""A compact, reproducible, resume-capable supervised trainer."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import chess
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from chess_ai.data.examples import (
    DATASET_VERSION,
    LoadedDataset,
    TrainingExample,
    load_dataset,
    split_examples_by_group,
)
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import load_checkpoint, save_checkpoint
from chess_ai.model.policy_value_net import PolicyValueNet
from chess_ai.training.losses import policy_accuracy, policy_value_loss

LOGGER = logging.getLogger(__name__)
SelectionMetric = Literal[
    "validation_loss",
    "validation_policy_loss",
    "validation_value_loss",
    "validation_policy_top1_accuracy",
]


class TrainingError(RuntimeError):
    """Raised for a configuration, dataset, or training-loop failure."""


class TrainingOutOfMemoryError(TrainingError):
    """Raised with practical advice when PyTorch reports device OOM."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Settings for supervised optimization and local artifacts."""

    dataset_path: Path | None = None
    checkpoint_dir: Path = Path("checkpoints")
    metrics_path: Path = Path("data/metrics/training.jsonl")
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    legal_policy_mask: bool = False
    sample_weight_key: str | None = None
    game_balanced_sampling: bool = False
    validation_group_key: str = "game_id"
    l2_regularization: float | None = None
    gradient_clip: float = 1.0
    gradient_clip_norm: float | None = None
    validation_fraction: float = 0.2
    scheduler: bool = False
    selection_metric: SelectionMetric = "validation_loss"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    checkpoint_every: int = 1
    baseline_eligible: bool = False
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 20
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_path", "resume_from"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "metrics_path", Path(self.metrics_path))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.policy_loss_weight < 0.0:
            raise ValueError("policy_loss_weight cannot be negative")
        if self.value_loss_weight < 0.0:
            raise ValueError("value_loss_weight cannot be negative")
        if self.policy_loss_weight == 0.0 and self.value_loss_weight == 0.0:
            raise ValueError("at least one policy/value loss weight must be positive")
        if self.l2_regularization is not None and self.l2_regularization < 0.0:
            raise ValueError("l2_regularization cannot be negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0)")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.selection_metric not in {
            "validation_loss",
            "validation_policy_loss",
            "validation_value_loss",
            "validation_policy_top1_accuracy",
        }:
            raise ValueError(f"Unsupported selection_metric: {self.selection_metric!r}")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if not isinstance(self.baseline_eligible, bool):
            raise ValueError("baseline_eligible must be a boolean")
        if self.sample_weight_key is not None and not self.sample_weight_key.strip():
            raise ValueError("sample_weight_key must be a non-empty string when configured")
        if not isinstance(self.game_balanced_sampling, bool):
            raise ValueError("game_balanced_sampling must be a boolean")
        if not isinstance(self.validation_group_key, str) or not self.validation_group_key.strip():
            raise ValueError("validation_group_key must be a non-empty string")

    @property
    def resolved_l2(self) -> float:
        """Support the YAML name ``weight_decay`` as an educational L2 alias."""

        return self.weight_decay if self.l2_regularization is None else self.l2_regularization

    @property
    def resolved_gradient_clip(self) -> float:
        """Return the explicit alias when supplied."""

        return self.gradient_clip if self.gradient_clip_norm is None else self.gradient_clip_norm

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        seed: int = 0,
        device: str = "auto",
        resume_from: str | Path | None = None,
    ) -> TrainingConfig:
        """Build settings from the YAML ``training`` section."""

        values = dict(raw)
        values.setdefault("seed", seed)
        values.setdefault("device", device)
        if resume_from is not None:
            values["resume_from"] = Path(resume_from)
        for name in ("dataset_path", "checkpoint_dir", "metrics_path", "resume_from"):
            if values.get(name) is not None:
                values[name] = Path(str(values[name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown training settings: {', '.join(unknown)}")
        return cls(**values)

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return only portable primitive values."""

        raw = asdict(self)
        for key, value in tuple(raw.items()):
            if isinstance(value, Path):
                raw[key] = str(value)
        return raw


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics required to interpret and reproduce one completed epoch."""

    epoch: int
    training_loss: float
    validation_loss: float | None
    policy_loss: float
    value_loss: float
    policy_top1_accuracy: float
    policy_top5_accuracy: float
    value_mean_absolute_error: float
    validation_policy_loss: float | None
    validation_value_loss: float | None
    validation_policy_top1_accuracy: float | None
    validation_policy_top5_accuracy: float | None
    validation_value_mean_absolute_error: float | None
    unweighted_training_loss: float
    unweighted_policy_loss: float
    unweighted_value_loss: float
    unweighted_policy_top1_accuracy: float
    unweighted_policy_top5_accuracy: float
    unweighted_value_mean_absolute_error: float
    validation_unweighted_loss: float | None
    validation_unweighted_policy_loss: float | None
    validation_unweighted_value_loss: float | None
    validation_unweighted_policy_top1_accuracy: float | None
    validation_unweighted_policy_top5_accuracy: float | None
    validation_unweighted_value_mean_absolute_error: float | None
    epoch_duration_seconds: float
    learning_rate: float
    training_examples: int
    validation_examples: int
    device: str
    selection_metric: str = "validation_loss"
    selection_value: float = math.inf
    early_stopped: bool = False

    @property
    def number_of_examples(self) -> int:
        return self.training_examples + self.validation_examples

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON names plus familiar short aliases."""

        result = asdict(self)
        result["number_of_examples"] = self.number_of_examples
        result["train_loss"] = self.training_loss
        result["val_loss"] = self.validation_loss
        result["policy_top_1_accuracy"] = self.policy_top1_accuracy
        result["policy_top_5_accuracy"] = self.policy_top5_accuracy
        return result


@dataclass(slots=True)
class _Aggregate:
    total: float = 0.0
    policy: float = 0.0
    value: float = 0.0
    top1: float = 0.0
    top5: float = 0.0
    value_mae: float = 0.0
    unweighted_total: float = 0.0
    unweighted_policy: float = 0.0
    unweighted_value: float = 0.0
    unweighted_top1: float = 0.0
    unweighted_top5: float = 0.0
    unweighted_value_mae: float = 0.0
    weighted_mass: float = 0.0
    examples: int = 0

    def add(
        self,
        *,
        batch_size: int,
        total: Tensor,
        policy: Tensor,
        value: Tensor,
        top1: Tensor,
        top5: Tensor,
        value_mae: Tensor,
        unweighted_total: Tensor,
        unweighted_policy: Tensor,
        unweighted_value: Tensor,
        unweighted_top1: Tensor,
        unweighted_top5: Tensor,
        unweighted_value_mae: Tensor,
        metric_weight: float,
    ) -> None:
        self.total += float(total.detach().item()) * metric_weight
        self.policy += float(policy.detach().item()) * metric_weight
        self.value += float(value.detach().item()) * metric_weight
        self.top1 += float(top1.detach().item()) * metric_weight
        self.top5 += float(top5.detach().item()) * metric_weight
        self.value_mae += float(value_mae.detach().item()) * metric_weight
        self.unweighted_total += float(unweighted_total.detach().item()) * batch_size
        self.unweighted_policy += float(unweighted_policy.detach().item()) * batch_size
        self.unweighted_value += float(unweighted_value.detach().item()) * batch_size
        self.unweighted_top1 += float(unweighted_top1.detach().item()) * batch_size
        self.unweighted_top5 += float(unweighted_top5.detach().item()) * batch_size
        self.unweighted_value_mae += float(unweighted_value_mae.detach().item()) * batch_size
        self.weighted_mass += metric_weight
        self.examples += batch_size

    def averages(self) -> dict[str, float]:
        if self.examples == 0 or self.weighted_mass <= 0.0:
            raise TrainingError("Cannot calculate metrics for an empty data loader")
        return {
            "total": self.total / self.weighted_mass,
            "policy": self.policy / self.weighted_mass,
            "value": self.value / self.weighted_mass,
            "top1": self.top1 / self.weighted_mass,
            "top5": self.top5 / self.weighted_mass,
            "value_mae": self.value_mae / self.weighted_mass,
            "unweighted_total": self.unweighted_total / self.examples,
            "unweighted_policy": self.unweighted_policy / self.examples,
            "unweighted_value": self.unweighted_value / self.examples,
            "unweighted_top1": self.unweighted_top1 / self.examples,
            "unweighted_top5": self.unweighted_top5 / self.examples,
            "unweighted_value_mae": self.unweighted_value_mae / self.examples,
        }


def _training_weight(example: TrainingExample, key: str | None) -> float:
    if key is None:
        return 1.0
    raw = example.metadata.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise TrainingError(f"Example {example.game_id!r} is missing a numeric {key!r} weight")
    try:
        weight = float(raw)
    except (TypeError, ValueError) as exc:
        raise TrainingError(
            f"Example {example.game_id!r} is missing a numeric {key!r} weight"
        ) from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise TrainingError(
            f"Example {example.game_id!r} has non-positive or non-finite {key!r} weight"
        )
    return weight


class _TrainerDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    """Attach optional sample weights and legal masks to ordinary examples."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        legal_masks: npt.NDArray[np.bool_] | None,
        sample_weight_key: str | None,
    ) -> None:
        self.examples = list(examples)
        self.legal_masks = legal_masks
        self.weights = [_training_weight(example, sample_weight_key) for example in self.examples]
        if self.legal_masks is not None and len(self.examples) != len(self.legal_masks):
            raise ValueError("examples and legal_masks must contain the same number of rows")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        example = self.examples[index]
        legal_mask = (
            torch.from_numpy(self.legal_masks[index])
            if self.legal_masks is not None
            else torch.empty(0, dtype=torch.bool)
        )
        return (
            torch.from_numpy(example.board_tensor),
            torch.from_numpy(example.target_policy),
            torch.tensor([example.target_value], dtype=torch.float32),
            torch.tensor(self.weights[index], dtype=torch.float32),
            legal_mask,
        )


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and reject unavailable explicit CUDA devices."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch device {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is False"
        )
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Optimize a :class:`PolicyValueNet` and persist every completed epoch."""

    def __init__(
        self,
        model: PolicyValueNet,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = resolve_device(config.device)
        _seed_everything(config.seed)
        self.model.to(self.device)
        # Explicit L2 is part of the reported loss. Setting AdamW's decoupled
        # decay to zero avoids applying the regularizer twice.
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.0)
        self.scheduler = (
            CosineAnnealingLR(self.optimizer, T_max=max(1, config.epochs))
            if config.scheduler
            else None
        )
        self.start_epoch = 1
        self.best_validation_loss = math.inf
        self.best_selection_score = (
            -math.inf if config.selection_metric == "validation_policy_top1_accuracy" else math.inf
        )
        self.epochs_without_improvement = 0
        self.baseline_eligible = config.baseline_eligible
        self._resume_loaded = False

    def _resume_if_requested(self) -> None:
        if self._resume_loaded or self.config.resume_from is None:
            return
        loaded = load_checkpoint(
            self.config.resume_from,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        saved_dataset_version = loaded.metadata.get("dataset_version")
        if saved_dataset_version not in (None, DATASET_VERSION):
            raise TrainingError(
                "Resume checkpoint was trained from incompatible dataset format version "
                f"{saved_dataset_version!r}; this code supports version {DATASET_VERSION}."
            )
        self.start_epoch = loaded.epoch + 1
        raw_best = (loaded.extra or {}).get("best_validation_loss")
        if isinstance(raw_best, (int, float)):
            self.best_validation_loss = float(raw_best)
        elif loaded.metrics.get("validation_loss") is not None:
            self.best_validation_loss = float(loaded.metrics["validation_loss"])
        saved_selection_metric = (loaded.extra or {}).get("selection_metric")
        if saved_selection_metric not in (None, self.config.selection_metric):
            raise TrainingError(
                "Resume checkpoint used a different selection metric: "
                f"{saved_selection_metric!r} vs {self.config.selection_metric!r}"
            )
        raw_selection = (loaded.extra or {}).get("best_selection_score")
        if isinstance(raw_selection, (int, float)):
            self.best_selection_score = float(raw_selection)
        elif self.config.selection_metric == "validation_loss" and math.isfinite(
            self.best_validation_loss
        ):
            self.best_selection_score = self.best_validation_loss
        elif self.config.selection_metric != "validation_loss":
            raise TrainingError(
                "Resume checkpoint predates configurable selection metrics; start a fresh run "
                "with --init-checkpoint instead"
            )
        raw_wait = (loaded.extra or {}).get("epochs_without_improvement", 0)
        if isinstance(raw_wait, int) and raw_wait >= 0:
            self.epochs_without_improvement = raw_wait
        self._resume_loaded = True

    def _loader(
        self,
        examples: Sequence[TrainingExample],
        *,
        training: bool,
        epoch: int,
        legal_masks: npt.NDArray[np.bool_] | None = None,
    ) -> DataLoader[Any]:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + epoch)
        dataset: Dataset[Any] = _TrainerDataset(
            examples,
            legal_masks,
            self.config.sample_weight_key,
        )
        sampler: WeightedRandomSampler | None = None
        if training and self.config.game_balanced_sampling:
            game_counts = Counter(example.game_id for example in examples)
            sampling_weights = [1.0 / game_counts[example.game_id] for example in examples]
            sampler = WeightedRandomSampler(
                sampling_weights,
                num_samples=len(examples),
                replacement=True,
                generator=generator,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def _run_epoch(
        self,
        examples: Sequence[TrainingExample],
        *,
        training: bool,
        epoch: int,
        legal_masks: npt.NDArray[np.bool_] | None = None,
    ) -> dict[str, float]:
        loader = self._loader(
            examples,
            training=training,
            epoch=epoch,
            legal_masks=legal_masks,
        )
        self.model.train(training)
        aggregate = _Aggregate()
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(loader, start=1):
                boards, target_policy, target_value, sample_weight, legal_action_mask = batch
                boards = boards.to(self.device, non_blocking=True)
                target_policy = target_policy.to(self.device, non_blocking=True)
                target_value = target_value.to(self.device, non_blocking=True)
                sample_weight = sample_weight.to(self.device, non_blocking=True)
                if legal_masks is None:
                    legal_action_mask = None
                else:
                    legal_action_mask = legal_action_mask.to(self.device, non_blocking=True)
                weighted_samples = (
                    sample_weight if self.config.sample_weight_key is not None else None
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                policy_logits, predicted_value = self.model(boards)
                components = policy_value_loss(
                    policy_logits,
                    predicted_value,
                    target_policy,
                    target_value,
                    model=self.model,
                    l2_regularization=self.config.resolved_l2,
                    policy_loss_weight=self.config.policy_loss_weight,
                    value_loss_weight=self.config.value_loss_weight,
                    legal_action_mask=legal_action_mask,
                    sample_weight=weighted_samples,
                )
                unweighted_components = (
                    components
                    if weighted_samples is None
                    else policy_value_loss(
                        policy_logits,
                        predicted_value,
                        target_policy,
                        target_value,
                        model=self.model,
                        l2_regularization=self.config.resolved_l2,
                        policy_loss_weight=self.config.policy_loss_weight,
                        value_loss_weight=self.config.value_loss_weight,
                        legal_action_mask=legal_action_mask,
                    )
                )
                if not torch.isfinite(components.total):
                    raise TrainingError(
                        "Training produced a non-finite loss. Check the dataset for invalid "
                        "values and try a smaller learning rate."
                    )
                if training:
                    components.total.backward()
                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.resolved_gradient_clip,
                    )
                    self.optimizer.step()

                top1 = policy_accuracy(
                    policy_logits,
                    target_policy,
                    top_k=1,
                    legal_action_mask=legal_action_mask,
                    sample_weight=weighted_samples,
                )
                top5 = policy_accuracy(
                    policy_logits,
                    target_policy,
                    top_k=5,
                    legal_action_mask=legal_action_mask,
                    sample_weight=weighted_samples,
                )
                unweighted_top1 = policy_accuracy(
                    policy_logits,
                    target_policy,
                    top_k=1,
                    legal_action_mask=legal_action_mask,
                )
                unweighted_top5 = policy_accuracy(
                    policy_logits,
                    target_policy,
                    top_k=5,
                    legal_action_mask=legal_action_mask,
                )
                value_errors = (predicted_value - target_value).abs().mean(dim=1)
                unweighted_value_mae = value_errors.mean()
                value_mae = (
                    unweighted_value_mae
                    if weighted_samples is None
                    else (value_errors * sample_weight).sum() / sample_weight.sum()
                )
                aggregate.add(
                    batch_size=boards.shape[0],
                    total=components.total,
                    policy=components.policy,
                    value=components.value,
                    top1=top1,
                    top5=top5,
                    value_mae=value_mae,
                    unweighted_total=unweighted_components.total,
                    unweighted_policy=unweighted_components.policy,
                    unweighted_value=unweighted_components.value,
                    unweighted_top1=unweighted_top1,
                    unweighted_top5=unweighted_top5,
                    unweighted_value_mae=unweighted_value_mae,
                    metric_weight=(
                        float(sample_weight.sum().detach().item())
                        if weighted_samples is not None
                        else float(boards.shape[0])
                    ),
                )
                if training and batch_index % self.config.log_every == 0:
                    LOGGER.info(
                        "epoch=%d batch=%d loss=%.4f",
                        epoch,
                        batch_index,
                        float(components.total.detach().item()),
                    )
        return aggregate.averages()

    @staticmethod
    def _build_legal_masks(
        examples: Sequence[TrainingExample],
    ) -> npt.NDArray[np.bool_]:
        move_encoder = MoveEncoder()
        masks = np.zeros((len(examples), move_encoder.action_size), dtype=np.bool_)
        for index, example in enumerate(examples):
            raw_fen = example.metadata.get("fen")
            if not isinstance(raw_fen, str) or not raw_fen.strip():
                raise TrainingError(
                    "legal_policy_mask requires every training example to contain FEN metadata"
                )
            try:
                board = chess.Board(raw_fen)
            except ValueError as exc:
                raise TrainingError(
                    f"Example {example.game_id!r} has invalid FEN metadata: {exc}"
                ) from exc
            masks[index] = move_encoder.legal_action_mask(board).astype(np.bool_)
            if np.any(example.target_policy[~masks[index]] > 0.0):
                raise TrainingError(
                    f"Example {example.game_id!r} assigns policy probability to an illegal "
                    f"move in {raw_fen}"
                )
        return masks

    def _prepare_examples(
        self,
        examples: Sequence[TrainingExample] | LoadedDataset | str | Path | None,
        validation_examples: Sequence[TrainingExample] | None,
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        source = examples
        if source is None:
            if self.config.dataset_path is None:
                raise TrainingError("No examples or dataset_path were provided")
            source = self.config.dataset_path
        if isinstance(source, (str, Path)):
            source = load_dataset(source)
        all_examples = list(source.examples if isinstance(source, LoadedDataset) else source)
        if not all_examples:
            raise TrainingError("The training dataset contains no examples")

        if validation_examples is None:
            training, validation = split_examples_by_group(
                all_examples,
                self.config.validation_fraction,
                seed=self.config.seed,
                group_key=self.config.validation_group_key,
            )
        else:
            training = all_examples
            validation = list(validation_examples)
            group_key = self.config.validation_group_key

            def group_id(item: TrainingExample) -> str:
                if group_key == "game_id":
                    return item.game_id
                value = item.metadata.get(group_key)
                if value is None or not str(value).strip():
                    raise TrainingError(
                        f"Example {item.game_id!r} is missing validation group {group_key!r}"
                    )
                return str(value)

            train_ids = {group_id(item) for item in training}
            validation_ids = {group_id(item) for item in validation}
            overlap = train_ids.intersection(validation_ids)
            if overlap:
                preview = ", ".join(sorted(overlap)[:3])
                raise TrainingError(
                    "Training and validation sets share game IDs (dataset leakage): " + preview
                )
        if not training:
            raise TrainingError("The game-group split produced no training examples")
        return training, validation

    def _append_metrics(self, metrics: EpochMetrics) -> None:
        path = self.config.metrics_path
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(metrics.to_dict(), sort_keys=True, allow_nan=False)
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise TrainingError(f"Could not append training metrics to {path}: {exc}") from exc

    def _selection_value(self, values: Mapping[str, float]) -> float:
        key = {
            "validation_loss": "total",
            "validation_policy_loss": "policy",
            "validation_value_loss": "value",
            "validation_policy_top1_accuracy": "top1",
        }[self.config.selection_metric]
        return float(values[key])

    def _is_selection_improvement(self, value: float) -> bool:
        if self.config.selection_metric == "validation_policy_top1_accuracy":
            return value > self.best_selection_score + self.config.early_stopping_min_delta
        return value < self.best_selection_score - self.config.early_stopping_min_delta

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {
            "best_validation_loss": self.best_validation_loss,
            "selection_metric": self.config.selection_metric,
            "best_selection_score": self.best_selection_score,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def _save_baseline(self, values: Mapping[str, float]) -> None:
        metrics = {
            "epoch": 0,
            "validation_loss": values["total"],
            "validation_policy_loss": values["policy"],
            "validation_value_loss": values["value"],
            "validation_policy_top1_accuracy": values["top1"],
            "validation_policy_top5_accuracy": values["top5"],
            "selection_metric": self.config.selection_metric,
            "selection_value": self.best_selection_score,
            "baseline": True,
        }

        def save_to(path: Path) -> None:
            save_checkpoint(
                path,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=0,
                metrics=metrics,
                training_config=self.config.checkpoint_metadata(),
                dataset_version=DATASET_VERSION,
                extra=self._checkpoint_extra(),
            )

        save_to(self.config.checkpoint_dir / "baseline.pt")
        save_to(self.config.checkpoint_dir / "best.pt")

    def _save_epoch(
        self,
        metrics: EpochMetrics,
        *,
        is_best: bool,
        retain_epoch: bool,
    ) -> None:
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def save_to(path: Path) -> None:
            save_checkpoint(
                path,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=metrics.epoch,
                metrics=metrics.to_dict(),
                training_config=self.config.checkpoint_metadata(),
                dataset_version=DATASET_VERSION,
                extra=self._checkpoint_extra(),
            )

        if retain_epoch:
            save_to(checkpoint_dir / f"epoch_{metrics.epoch:04d}.pt")
        save_to(checkpoint_dir / "last.pt")
        if is_best:
            save_to(checkpoint_dir / "best.pt")

    @staticmethod
    def _raise_helpful_oom(exc: RuntimeError, device: torch.device) -> None:
        if "out of memory" not in str(exc).lower():
            raise exc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise TrainingOutOfMemoryError(
            f"PyTorch ran out of memory on {device}. Reduce training.batch_size or "
            "model.channels, then resume from the latest epoch checkpoint."
        ) from exc

    def fit(
        self,
        examples: Sequence[TrainingExample] | LoadedDataset | str | Path | None = None,
        *,
        validation_examples: Sequence[TrainingExample] | None = None,
    ) -> list[EpochMetrics]:
        """Train through ``config.epochs`` and return newly completed epochs."""

        training, validation = self._prepare_examples(examples, validation_examples)
        training_legal_masks = (
            self._build_legal_masks(training) if self.config.legal_policy_mask else None
        )
        validation_legal_masks = (
            self._build_legal_masks(validation)
            if self.config.legal_policy_mask and validation
            else None
        )
        self._resume_if_requested()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.start_epoch == 1:
            try:
                self.config.metrics_path.write_text("", encoding="utf-8")
            except OSError as exc:
                raise TrainingError(
                    f"Could not initialize metrics file {self.config.metrics_path}: {exc}"
                ) from exc

        history: list[EpochMetrics] = []
        if self.start_epoch == 1 and self.baseline_eligible:
            baseline_values = (
                self._run_epoch(
                    validation,
                    training=False,
                    epoch=0,
                    legal_masks=validation_legal_masks,
                )
                if validation
                else self._run_epoch(
                    training,
                    training=False,
                    epoch=0,
                    legal_masks=training_legal_masks,
                )
            )
            self.best_validation_loss = baseline_values["total"]
            self.best_selection_score = self._selection_value(baseline_values)
            self._save_baseline(baseline_values)
        if self.start_epoch > self.config.epochs:
            LOGGER.warning(
                "Checkpoint already completed epoch %d; configured total epochs is %d, "
                "so there is nothing left to train",
                self.start_epoch - 1,
                self.config.epochs,
            )
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            started = time.perf_counter()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            try:
                train_values = self._run_epoch(
                    training,
                    training=True,
                    epoch=epoch,
                    legal_masks=training_legal_masks,
                )
                validation_values = (
                    self._run_epoch(
                        validation,
                        training=False,
                        epoch=epoch,
                        legal_masks=validation_legal_masks,
                    )
                    if validation
                    else None
                )
            except RuntimeError as exc:
                self._raise_helpful_oom(exc, self.device)
                raise  # pragma: no cover - helper always raises

            selection_values = validation_values or train_values
            selection_value = self._selection_value(selection_values)
            is_best = self._is_selection_improvement(selection_value)
            if is_best:
                self.best_selection_score = selection_value
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            self.best_validation_loss = min(
                self.best_validation_loss,
                selection_values["total"],
            )
            early_stopped = (
                self.config.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            )
            if self.scheduler is not None:
                self.scheduler.step()
            metrics = EpochMetrics(
                epoch=epoch,
                training_loss=train_values["total"],
                validation_loss=(
                    validation_values["total"] if validation_values is not None else None
                ),
                policy_loss=train_values["policy"],
                value_loss=train_values["value"],
                policy_top1_accuracy=train_values["top1"],
                policy_top5_accuracy=train_values["top5"],
                value_mean_absolute_error=train_values["value_mae"],
                validation_policy_loss=(
                    validation_values["policy"] if validation_values is not None else None
                ),
                validation_value_loss=(
                    validation_values["value"] if validation_values is not None else None
                ),
                validation_policy_top1_accuracy=(
                    validation_values["top1"] if validation_values is not None else None
                ),
                validation_policy_top5_accuracy=(
                    validation_values["top5"] if validation_values is not None else None
                ),
                validation_value_mean_absolute_error=(
                    validation_values["value_mae"] if validation_values is not None else None
                ),
                unweighted_training_loss=train_values["unweighted_total"],
                unweighted_policy_loss=train_values["unweighted_policy"],
                unweighted_value_loss=train_values["unweighted_value"],
                unweighted_policy_top1_accuracy=train_values["unweighted_top1"],
                unweighted_policy_top5_accuracy=train_values["unweighted_top5"],
                unweighted_value_mean_absolute_error=train_values["unweighted_value_mae"],
                validation_unweighted_loss=(
                    validation_values["unweighted_total"] if validation_values is not None else None
                ),
                validation_unweighted_policy_loss=(
                    validation_values["unweighted_policy"]
                    if validation_values is not None
                    else None
                ),
                validation_unweighted_value_loss=(
                    validation_values["unweighted_value"] if validation_values is not None else None
                ),
                validation_unweighted_policy_top1_accuracy=(
                    validation_values["unweighted_top1"] if validation_values is not None else None
                ),
                validation_unweighted_policy_top5_accuracy=(
                    validation_values["unweighted_top5"] if validation_values is not None else None
                ),
                validation_unweighted_value_mean_absolute_error=(
                    validation_values["unweighted_value_mae"]
                    if validation_values is not None
                    else None
                ),
                epoch_duration_seconds=time.perf_counter() - started,
                learning_rate=learning_rate,
                training_examples=len(training),
                validation_examples=len(validation),
                device=str(self.device),
                selection_metric=self.config.selection_metric,
                selection_value=selection_value,
                early_stopped=early_stopped,
            )
            retain_epoch = (
                epoch % self.config.checkpoint_every == 0
                or epoch == self.config.epochs
                or early_stopped
            )
            self._save_epoch(metrics, is_best=is_best, retain_epoch=retain_epoch)
            self._append_metrics(metrics)
            history.append(metrics)
            LOGGER.info(
                "epoch=%d train_loss=%.4f validation_loss=%s selection=%s:%.6f wait=%d/%d",
                epoch,
                metrics.training_loss,
                f"{metrics.validation_loss:.4f}"
                if metrics.validation_loss is not None
                else "n/a (one game only)",
                self.config.selection_metric,
                selection_value,
                self.epochs_without_improvement,
                self.config.early_stopping_patience,
            )
            if early_stopped:
                LOGGER.info(
                    "Early stopping at epoch %d after %d epochs without improvement",
                    epoch,
                    self.epochs_without_improvement,
                )
                break
        return history


def train_model(
    model: PolicyValueNet,
    config: TrainingConfig,
    examples: Sequence[TrainingExample] | LoadedDataset | str | Path | None = None,
    *,
    validation_examples: Sequence[TrainingExample] | None = None,
) -> list[EpochMetrics]:
    """One-call convenience wrapper around :class:`Trainer`."""

    return Trainer(model, config).fit(
        examples,
        validation_examples=validation_examples,
    )
