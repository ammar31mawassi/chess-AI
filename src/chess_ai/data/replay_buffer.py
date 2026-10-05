"""A small bounded replay store backed by the versioned dataset format."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset


class ReplayBuffer:
    """Keep the newest examples and sample them reproducibly.

    Phase 1 mainly uses supervised data, but this simple abstraction gives the
    later self-play work a well-tested storage boundary without implementing an
    AlphaZero replay system prematurely.
    """

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self._random = random.Random(self.seed)
        self._examples: list[TrainingExample] = []

    def __len__(self) -> int:
        return len(self._examples)

    def add(self, example: TrainingExample) -> None:
        """Append one example, evicting the oldest if full."""

        self._examples.append(example)
        overflow = len(self._examples) - self.capacity
        if overflow > 0:
            del self._examples[:overflow]

    def extend(self, examples: Iterable[TrainingExample]) -> None:
        """Append multiple examples while preserving capacity."""

        self._examples.extend(examples)
        overflow = len(self._examples) - self.capacity
        if overflow > 0:
            del self._examples[:overflow]

    def sample(self, count: int) -> list[TrainingExample]:
        """Sample without replacement using this buffer's private RNG."""

        if count <= 0:
            raise ValueError("sample count must be positive")
        if count > len(self._examples):
            raise ValueError(f"Cannot sample {count} examples from a buffer containing {len(self)}")
        return self._random.sample(self._examples, count)

    def snapshot(self) -> list[TrainingExample]:
        """Return a shallow copy in oldest-to-newest order."""

        return list(self._examples)

    def save(self, path: str | Path) -> Path:
        """Persist the buffer through the shared versioned file format."""

        return save_dataset(
            path,
            self._examples,
            metadata={
                "kind": "replay_buffer",
                "capacity": self.capacity,
                "seed": self.seed,
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        capacity: int | None = None,
        seed: int | None = None,
    ) -> ReplayBuffer:
        """Restore examples and use saved settings unless explicitly replaced."""

        loaded = load_dataset(path)
        saved_capacity = loaded.metadata.get("capacity")
        saved_seed = loaded.metadata.get("seed", 0)
        resolved_capacity = capacity if capacity is not None else saved_capacity
        if not isinstance(resolved_capacity, int):
            raise ValueError("Saved replay buffer has no valid capacity metadata")
        try:
            resolved_seed = seed if seed is not None else int(saved_seed)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("Saved replay buffer has invalid seed metadata") from exc
        buffer = cls(resolved_capacity, seed=resolved_seed)
        buffer.extend(loaded.examples)
        return buffer
