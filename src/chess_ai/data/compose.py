"""Explicitly compose reviewed supervised datasets with source-aware weights."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset

COMPOSITION_FORMAT = "self-improving-chess-ai.dataset-composition"
COMPOSITION_VERSION = 1


class DatasetCompositionError(RuntimeError):
    """Raised when source datasets cannot be safely combined."""


@dataclass(frozen=True, slots=True)
class CompositionSource:
    path: Path
    name: str
    constant_weight: float | None = None
    metadata_weight_key: str | None = None
    group_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.name.strip():
            raise ValueError("composition source name must not be empty")
        if (self.constant_weight is None) == (self.metadata_weight_key is None):
            raise ValueError(
                "each composition source needs exactly one of constant_weight or "
                "metadata_weight_key"
            )
        if self.constant_weight is not None and (
            not math.isfinite(self.constant_weight) or self.constant_weight <= 0.0
        ):
            raise ValueError("constant_weight must be finite and positive")
        if self.metadata_weight_key is not None and not self.metadata_weight_key.strip():
            raise ValueError("metadata_weight_key must not be empty")
        if self.group_key is not None and not self.group_key.strip():
            raise ValueError("group_key must not be empty")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CompositionSource:
        known = {"path", "name", "constant_weight", "metadata_weight_key", "group_key"}
        unknown = sorted(set(raw).difference(known))
        if unknown:
            raise ValueError(f"Unknown composition source settings: {', '.join(unknown)}")
        return cls(
            path=Path(str(raw["path"])),
            name=str(raw["name"]),
            constant_weight=(
                float(raw["constant_weight"]) if raw.get("constant_weight") is not None else None
            ),
            metadata_weight_key=(
                str(raw["metadata_weight_key"])
                if raw.get("metadata_weight_key") is not None
                else None
            ),
            group_key=str(raw["group_key"]) if raw.get("group_key") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class CompositionConfig:
    output_path: Path
    sources: tuple[CompositionSource, ...]
    output_weight_key: str = "combined_training_weight"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "sources", tuple(self.sources))
        if len(self.sources) < 2:
            raise ValueError("dataset composition requires at least two sources")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("composition source names must be unique")
        if not self.output_weight_key.strip():
            raise ValueError("output_weight_key must not be empty")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CompositionConfig:
        unknown = sorted(set(raw).difference({"output_path", "sources", "output_weight_key"}))
        if unknown:
            raise ValueError(f"Unknown composition settings: {', '.join(unknown)}")
        raw_sources = raw.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("composition sources must be a sequence")
        sources = tuple(
            CompositionSource.from_mapping(item)
            for item in raw_sources
            if isinstance(item, Mapping)
        )
        if len(sources) != len(raw_sources):
            raise ValueError("every composition source must be a mapping")
        return cls(
            output_path=Path(str(raw["output_path"])),
            sources=sources,
            output_weight_key=str(raw.get("output_weight_key", "combined_training_weight")),
        )


@dataclass(frozen=True, slots=True)
class CompositionSummary:
    output_path: Path
    examples: int
    source_examples: dict[str, int]
    output_weight_key: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path)
        return payload


def _source_weight(example: TrainingExample, source: CompositionSource) -> float:
    if source.constant_weight is not None:
        return source.constant_weight
    assert source.metadata_weight_key is not None
    raw = example.metadata.get(source.metadata_weight_key)
    try:
        weight = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DatasetCompositionError(
            f"Source {source.name!r} example {example.game_id!r} lacks numeric weight "
            f"{source.metadata_weight_key!r}"
        ) from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise DatasetCompositionError(
            f"Source {source.name!r} contains a non-positive or non-finite weight"
        )
    return weight


def compose_datasets(config: CompositionConfig) -> CompositionSummary:
    """Combine sources without mutating them and attach one normalized weight key."""

    if config.output_path.exists():
        raise DatasetCompositionError(f"Composed dataset already exists: {config.output_path}")
    examples: list[TrainingExample] = []
    counts: dict[str, int] = {}
    source_metadata: list[dict[str, Any]] = []
    for source in config.sources:
        if not source.path.is_file():
            raise DatasetCompositionError(f"Composition source does not exist: {source.path}")
        loaded = load_dataset(source.path)
        counts[source.name] = len(loaded)
        source_metadata.append(
            {
                "name": source.name,
                "path": str(source.path),
                "examples": len(loaded),
                "constant_weight": source.constant_weight,
                "metadata_weight_key": source.metadata_weight_key,
                "group_key": source.group_key,
            }
        )
        for example in loaded:
            weight = _source_weight(example, source)
            if source.group_key is None:
                source_group = example.metadata.get("opening_set_id", example.game_id)
            else:
                source_group = example.metadata.get(source.group_key)
                if source_group is None or not str(source_group).strip():
                    raise DatasetCompositionError(
                        f"Source {source.name!r} example {example.game_id!r} lacks non-empty "
                        f"group key {source.group_key!r}"
                    )
            examples.append(
                TrainingExample(
                    board_tensor=example.board_tensor,
                    target_policy=example.target_policy,
                    target_value=example.target_value,
                    metadata={
                        **example.metadata,
                        "composition_source": source.name,
                        "composition_group_id": f"{source.name}:{source_group}",
                        config.output_weight_key: weight,
                    },
                )
            )
    save_dataset(
        config.output_path,
        examples,
        metadata={
            "kind": "explicit_weighted_dataset_composition",
            "composition_format": COMPOSITION_FORMAT,
            "composition_version": COMPOSITION_VERSION,
            "output_weight_key": config.output_weight_key,
            "sources": source_metadata,
            "examples": len(examples),
        },
    )
    return CompositionSummary(
        output_path=config.output_path,
        examples=len(examples),
        source_examples=counts,
        output_weight_key=config.output_weight_key,
    )
