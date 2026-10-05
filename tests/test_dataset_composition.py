"""Tests for explicit source-weighted dataset composition."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chess_ai.data import (
    CompositionConfig,
    CompositionSource,
    TrainingExample,
    compose_datasets,
    load_dataset,
    save_dataset,
)
from chess_ai.environment.board_encoder import encode_board


def _example(
    game_id: str, weight: float | None = None, opening_pair_id: str | None = None
) -> TrainingExample:
    policy = np.zeros(4208, dtype=np.float32)
    policy[0] = 1.0
    metadata: dict[str, str | float] = {"game_id": game_id, "fen": chess.Board().fen()}
    if weight is not None:
        metadata["actor_weight"] = weight
    if opening_pair_id is not None:
        metadata["opening_pair_id"] = opening_pair_id
    return TrainingExample(encode_board(chess.Board()), policy, 0.0, metadata)


def test_composition_preserves_sources_and_assigns_one_weight_key(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    save_dataset(first, [_example("a")])
    save_dataset(second, [_example("b", 2.0, "pair-7")])
    config = CompositionConfig(
        output_path=tmp_path / "combined.pt",
        sources=(
            CompositionSource(first, "human", constant_weight=0.5),
            CompositionSource(
                second,
                "teacher",
                metadata_weight_key="actor_weight",
                group_key="opening_pair_id",
            ),
        ),
    )

    summary = compose_datasets(config)
    loaded = load_dataset(config.output_path)

    assert summary.source_examples == {"human": 1, "teacher": 1}
    assert [item.metadata["combined_training_weight"] for item in loaded] == [0.5, 2.0]
    assert [item.metadata["composition_source"] for item in loaded] == ["human", "teacher"]
    assert all(item.metadata["composition_group_id"] for item in loaded)
    assert loaded[1].metadata["composition_group_id"] == "teacher:pair-7"
