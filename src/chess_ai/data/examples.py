"""Training-example schema and a compact, versioned local file format."""

from __future__ import annotations

import json
import os
import pickle
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, overload
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

DATASET_FORMAT = "self-improving-chess-ai.supervised-examples"
DATASET_VERSION = 1
BOARD_SHAPE = (18, 8, 8)
ACTION_SIZE = 4208

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class DatasetFormatError(RuntimeError):
    """Raised when a dataset cannot be read or validated."""


class IncompatibleDatasetError(DatasetFormatError):
    """Raised when a dataset belongs to another schema/version."""


def _float_array(value: NDArray[np.floating[Any]] | Tensor, name: str) -> NDArray[np.float32]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    return np.ascontiguousarray(array)


def _json_mapping(metadata: Mapping[str, Any]) -> dict[str, JsonValue]:
    copied = dict(metadata)
    try:
        # Round-tripping also prevents values such as Path, chess.Color, and
        # arbitrary objects from becoming unsafe pickle dependencies.
        normalized = json.loads(json.dumps(copied, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain only JSON-compatible values") from exc
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
        raise ValueError("metadata must be a mapping")
    return normalized


@dataclass(slots=True)
class TrainingExample:
    """One supervised position and its policy/value teaching targets."""

    board_tensor: NDArray[np.float32] | Tensor
    target_policy: NDArray[np.float32] | Tensor
    target_value: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        board = _float_array(self.board_tensor, "board_tensor")
        policy = _float_array(self.target_policy, "target_policy")
        if board.shape != BOARD_SHAPE:
            raise ValueError(f"board_tensor must have shape {BOARD_SHAPE}; got {board.shape}")
        if policy.shape != (ACTION_SIZE,):
            raise ValueError(f"target_policy must have shape ({ACTION_SIZE},); got {policy.shape}")
        if not np.isfinite(board).all():
            raise ValueError("board_tensor contains a non-finite value")
        if not np.isfinite(policy).all():
            raise ValueError("target_policy contains a non-finite value")
        if np.any(policy < 0.0):
            raise ValueError("target_policy cannot contain negative probabilities")
        policy_sum = float(policy.sum(dtype=np.float64))
        if not np.isclose(policy_sum, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(f"target_policy probabilities must sum to 1.0; got {policy_sum}")
        value = float(self.target_value)
        if not np.isfinite(value):
            raise ValueError("target_value must be finite")
        if not -1.0 <= value <= 1.0:
            raise ValueError("target_value must be between -1.0 and 1.0")

        self.board_tensor = board
        self.target_policy = policy
        self.target_value = value
        self.metadata = _json_mapping(self.metadata)

    @property
    def game_id(self) -> str:
        """Stable group identifier used for leak-free validation splits."""

        value = self.metadata.get("game_id")
        if value is None or str(value).strip() == "":
            raise ValueError("Training-example metadata is missing a non-empty 'game_id'")
        return str(value)


@dataclass(slots=True)
class LoadedDataset(Sequence[TrainingExample]):
    """Examples plus file-level provenance metadata."""

    examples: list[TrainingExample]
    metadata: dict[str, JsonValue]
    format_version: int
    created_utc: str

    def __len__(self) -> int:
        return len(self.examples)

    @overload
    def __getitem__(self, index: int) -> TrainingExample: ...

    @overload
    def __getitem__(self, index: slice) -> list[TrainingExample]: ...

    def __getitem__(self, index: int | slice) -> TrainingExample | list[TrainingExample]:
        return self.examples[index]

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)


class TrainingExamplesDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """PyTorch Dataset adapter that keeps metadata outside model batches."""

    def __init__(self, examples: Sequence[TrainingExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        example = self.examples[index]
        return (
            torch.from_numpy(example.board_tensor),
            torch.from_numpy(example.target_policy),
            torch.tensor([example.target_value], dtype=torch.float32),
        )

    @property
    def game_ids(self) -> list[str]:
        """Return group labels in the same order as the dataset."""

        return [example.game_id for example in self.examples]


def _atomic_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except (OSError, RuntimeError, TypeError, pickle.PickleError) as exc:
        raise DatasetFormatError(f"Could not save dataset to {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def save_dataset(
    path: str | Path,
    examples: Sequence[TrainingExample],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save examples in dataset format version 1."""

    normalized = list(examples)
    if normalized:
        boards = torch.from_numpy(np.stack([item.board_tensor for item in normalized]))
        policies = torch.from_numpy(np.stack([item.target_policy for item in normalized]))
        values = torch.tensor([[item.target_value] for item in normalized], dtype=torch.float32)
    else:
        boards = torch.empty((0, *BOARD_SHAPE), dtype=torch.float32)
        policies = torch.empty((0, ACTION_SIZE), dtype=torch.float32)
        values = torch.empty((0, 1), dtype=torch.float32)

    encoded_metadata = [
        json.dumps(dict(item.metadata), sort_keys=True, separators=(",", ":"))
        for item in normalized
    ]
    file_metadata = _json_mapping(metadata or {})
    payload = {
        "format": DATASET_FORMAT,
        "format_version": DATASET_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "board_shape": BOARD_SHAPE,
        "action_size": ACTION_SIZE,
        "example_count": len(normalized),
        "metadata": file_metadata,
        "boards": boards,
        "target_policies": policies,
        "target_values": values,
        "example_metadata_json": encoded_metadata,
    }
    destination = Path(path)
    _atomic_save(payload, destination)
    return destination


def _torch_load(path: Path) -> Any:
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch
            return torch.load(path, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
        raise DatasetFormatError(f"Could not read dataset {path}: {exc}") from exc


def load_dataset(path: str | Path) -> LoadedDataset:
    """Load a dataset after checking its schema and tensor dimensions."""

    source = Path(path)
    if not source.is_file():
        raise DatasetFormatError(f"Dataset does not exist: {source}")
    raw = _torch_load(source)
    if not isinstance(raw, dict):
        raise IncompatibleDatasetError("Dataset root must be a mapping")
    if raw.get("format") != DATASET_FORMAT:
        raise IncompatibleDatasetError(
            f"Unsupported dataset format {raw.get('format')!r}; expected {DATASET_FORMAT!r}"
        )
    version = raw.get("format_version")
    if version != DATASET_VERSION:
        raise IncompatibleDatasetError(
            f"Unsupported dataset version {version!r}; this code supports {DATASET_VERSION}"
        )
    if tuple(raw.get("board_shape", ())) != BOARD_SHAPE:
        raise IncompatibleDatasetError(
            f"Dataset board shape {raw.get('board_shape')!r} is incompatible with {BOARD_SHAPE}"
        )
    if raw.get("action_size") != ACTION_SIZE:
        raise IncompatibleDatasetError(
            f"Dataset action size {raw.get('action_size')!r} is incompatible with {ACTION_SIZE}"
        )
    required = ("boards", "target_policies", "target_values", "example_metadata_json")
    missing = [name for name in required if name not in raw]
    if missing:
        raise IncompatibleDatasetError(f"Dataset is missing fields: {', '.join(missing)}")

    boards = raw["boards"]
    policies = raw["target_policies"]
    values = raw["target_values"]
    metadata_json = raw["example_metadata_json"]
    if not isinstance(boards, Tensor) or boards.ndim != 4 or tuple(boards.shape[1:]) != BOARD_SHAPE:
        raise IncompatibleDatasetError("Dataset boards tensor has an invalid shape")
    count = boards.shape[0]
    if not isinstance(policies, Tensor) or tuple(policies.shape) != (count, ACTION_SIZE):
        raise IncompatibleDatasetError("Dataset target_policies tensor has an invalid shape")
    if not isinstance(values, Tensor) or tuple(values.shape) != (count, 1):
        raise IncompatibleDatasetError("Dataset target_values tensor has an invalid shape")
    if not isinstance(metadata_json, list) or len(metadata_json) != count:
        raise IncompatibleDatasetError("Dataset example metadata count is inconsistent")
    if raw.get("example_count") != count:
        raise IncompatibleDatasetError("Dataset example_count does not match its tensors")

    examples: list[TrainingExample] = []
    for index in range(count):
        encoded = metadata_json[index]
        if not isinstance(encoded, str):
            raise IncompatibleDatasetError(f"Example {index} metadata is not JSON text")
        try:
            item_metadata = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise IncompatibleDatasetError(f"Example {index} metadata is invalid JSON") from exc
        if not isinstance(item_metadata, dict):
            raise IncompatibleDatasetError(f"Example {index} metadata must be a mapping")
        try:
            examples.append(
                TrainingExample(
                    board_tensor=boards[index].numpy(),
                    target_policy=policies[index].numpy(),
                    target_value=float(values[index].item()),
                    metadata=item_metadata,
                )
            )
        except ValueError as exc:
            raise IncompatibleDatasetError(f"Example {index} is invalid: {exc}") from exc

    raw_metadata = raw.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise IncompatibleDatasetError("Dataset file metadata must be a mapping")
    created = raw.get("created_utc", "")
    return LoadedDataset(
        examples=examples,
        metadata=_json_mapping(raw_metadata),
        format_version=DATASET_VERSION,
        created_utc=str(created),
    )


def save_examples(
    path: str | Path,
    examples: Sequence[TrainingExample],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Friendly alias for :func:`save_dataset`."""

    return save_dataset(path, examples, metadata=metadata)


def load_examples(path: str | Path) -> list[TrainingExample]:
    """Load only the example list when file-level metadata is not needed."""

    return load_dataset(path).examples


def split_examples_by_game(
    examples: Sequence[TrainingExample],
    validation_fraction: float = 0.2,
    *,
    seed: int = 0,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split complete games, never individual positions, into train/validation.

    With only one game there is no leak-free way to create both sets, so all
    positions remain in training and the validation set is empty.
    """

    return split_examples_by_group(
        examples,
        validation_fraction,
        seed=seed,
        group_key="game_id",
    )


def split_examples_by_group(
    examples: Sequence[TrainingExample],
    validation_fraction: float = 0.2,
    *,
    seed: int = 0,
    group_key: str = "game_id",
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split complete metadata groups without leaking related positions.

    ``game_id`` remains the ordinary default. Paired-opening experiments can
    instead use an ``opening_pair_id`` so both color-swapped games always land
    on the same side of the split.
    """

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0)")
    if not isinstance(group_key, str) or not group_key.strip():
        raise ValueError("group_key must be a non-empty string")

    def group_id(item: TrainingExample) -> str:
        if group_key == "game_id":
            return item.game_id
        value = item.metadata.get(group_key)
        if value is None or str(value).strip() == "":
            raise ValueError(f"Training-example metadata is missing a non-empty {group_key!r}")
        return str(value)

    normalized = list(examples)
    group_ids_by_example = [group_id(item) for item in normalized]
    if not normalized or validation_fraction == 0.0:
        return normalized, []

    group_ids = sorted(set(group_ids_by_example))
    if len(group_ids) == 1:
        return normalized, []
    randomizer = random.Random(seed)
    randomizer.shuffle(group_ids)
    validation_groups = round(len(group_ids) * validation_fraction)
    validation_groups = min(max(validation_groups, 1), len(group_ids) - 1)
    validation_ids = set(group_ids[:validation_groups])

    training = [
        item
        for item, item_group_id in zip(normalized, group_ids_by_example, strict=True)
        if item_group_id not in validation_ids
    ]
    validation = [
        item
        for item, item_group_id in zip(normalized, group_ids_by_example, strict=True)
        if item_group_id in validation_ids
    ]
    return training, validation
