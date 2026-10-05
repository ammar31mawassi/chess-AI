"""Tests for versioned datasets, replay storage, and bounded generation."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_ai.agents.random_agent import RandomAgent
from chess_ai.data import (
    DataGenerationError,
    GenerationConfig,
    IncompatibleDatasetError,
    ReplayBuffer,
    TrainingExample,
    generate_game_examples,
    generate_supervised_data,
    load_dataset,
    save_dataset,
)
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.games import load_pgn


def _example(game_id: str, action: int = 0, value: float = 0.0) -> TrainingExample:
    policy = np.zeros(4208, dtype=np.float32)
    policy[action] = 1.0
    return TrainingExample(
        board_tensor=BoardEncoder().encode(chess.Board()),
        target_policy=policy,
        target_value=value,
        metadata={"game_id": game_id, "note": "test"},
    )


def test_dataset_round_trip_preserves_examples_and_metadata(tmp_path: Path) -> None:
    path = save_dataset(
        tmp_path / "examples.pt",
        [_example("g1", value=1.0), _example("g2", action=10, value=-1.0)],
        metadata={"experiment": "tiny"},
    )

    loaded = load_dataset(path)

    assert loaded.format_version == 1
    assert loaded.metadata["experiment"] == "tiny"
    assert [example.game_id for example in loaded] == ["g1", "g2"]
    np.testing.assert_array_equal(loaded[1].target_policy, _example("g2", 10).target_policy)
    assert loaded[0].target_value == 1.0


def test_dataset_rejects_unknown_version(tmp_path: Path) -> None:
    path = save_dataset(tmp_path / "examples.pt", [_example("g1")])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = 2
    torch.save(payload, path)

    with pytest.raises(IncompatibleDatasetError, match="version"):
        load_dataset(path)


def test_replay_buffer_evicts_oldest_and_restores(tmp_path: Path) -> None:
    buffer = ReplayBuffer(2, seed=5)
    buffer.extend([_example("g1"), _example("g2"), _example("g3")])
    assert [item.game_id for item in buffer.snapshot()] == ["g2", "g3"]

    path = buffer.save(tmp_path / "replay.pt")
    restored = ReplayBuffer.load(path)
    assert restored.capacity == 2
    assert [item.game_id for item in restored.snapshot()] == ["g2", "g3"]


def test_random_generation_is_bounded_saved_and_resumable(tmp_path: Path) -> None:
    path = tmp_path / "generated.pt"
    first = GenerationConfig(
        dataset_path=path,
        games=1,
        white_agent="random",
        black_agent="random",
        max_moves=4,
        seed=11,
    )
    first_summary = generate_supervised_data(first)
    second_summary = generate_supervised_data(
        GenerationConfig(
            dataset_path=path,
            games=2,
            white_agent="random",
            black_agent="random",
            max_moves=4,
            seed=11,
        )
    )

    assert first_summary.completed_games == 1
    assert first_summary.examples == 4
    assert second_summary.resumed
    assert second_summary.completed_games == 2
    assert second_summary.examples == 8

    loaded = load_dataset(path)
    assert len({example.game_id for example in loaded}) == 2
    encoder = MoveEncoder()
    for example in loaded:
        board = chess.Board(str(example.metadata["fen"]))
        selected = encoder.decode(int(example.metadata["policy_action"]))
        assert selected in board.legal_moves
        assert np.count_nonzero(example.target_policy) == 1
        assert example.target_value == 0.0  # move-limit games are conservatively labelled draws


def test_game_generation_starts_after_an_unrecorded_prefix() -> None:
    starting_board = chess.Board()
    starting_board.push_uci("e2e4")
    starting_board.push_uci("e7e5")
    original_fen = starting_board.fen()

    examples = generate_game_examples(
        RandomAgent(seed=1, name="White teacher"),
        RandomAgent(seed=2, name="Black teacher"),
        game_id="prefixed",
        max_moves=2,
        starting_board=starting_board,
        example_metadata={"opening_plies": 2},
    )

    assert len(examples) == 2
    assert examples[0].metadata["fen"] == original_fen
    assert examples[0].metadata["ply"] == 2
    assert examples[0].metadata["opening_plies"] == 2
    assert starting_board.fen() == original_fen


def test_diverse_random_openings_are_not_policy_targets(tmp_path: Path) -> None:
    path = tmp_path / "diverse.pt"
    pgn_dir = tmp_path / "games"
    summary = generate_supervised_data(
        GenerationConfig(
            dataset_path=path,
            pgn_dir=pgn_dir,
            games=2,
            white_agent="minimax",
            black_agent="minimax",
            minimax_depth=1,
            max_moves=6,
            random_opening_min_plies=4,
            random_opening_max_plies=4,
            seed=17,
        )
    )

    loaded = load_dataset(path)
    starts_by_game = {
        example.game_id: str(example.metadata["recorded_starting_fen"]) for example in loaded
    }
    assert summary.examples == 4
    assert len(starts_by_game) == 2
    assert len(set(starts_by_game.values())) == 2
    assert all(example.metadata["opening_plies"] == 4 for example in loaded)
    assert all(int(example.metadata["ply"]) >= 4 for example in loaded)
    assert summary.pgn_dir == pgn_dir
    saved_games = sorted(pgn_dir.glob("*.pgn"))
    assert [path.name for path in saved_games] == ["game_0001.pgn", "game_0002.pgn"]
    archived = load_pgn(saved_games[0])[0]
    assert len(list(archived.mainline_moves())) == 6
    assert archived.headers["UnrecordedOpeningPlies"] == "4"
    assert archived.headers["TeacherPlies"] == "2"
    assert archived.headers["RetainedTrainingExamples"] == "2"


def test_position_deduplication_survives_resume(tmp_path: Path) -> None:
    path = tmp_path / "deduplicated.pt"
    common = {
        "dataset_path": path,
        "white_agent": "minimax",
        "black_agent": "minimax",
        "minimax_depth": 1,
        "max_moves": 4,
        "deduplicate_positions": True,
        "seed": 23,
    }
    first = generate_supervised_data(GenerationConfig(games=1, **common))
    second = generate_supervised_data(GenerationConfig(games=2, **common))

    assert first.examples == 4
    assert first.duplicates_discarded == 0
    assert second.resumed
    assert second.completed_games == 2
    assert second.examples == 4
    assert second.duplicates_discarded == 4
    loaded = load_dataset(path)
    assert loaded.metadata["duplicates_discarded"] == 4


def test_random_opening_must_fit_inside_move_limit() -> None:
    with pytest.raises(ValueError, match="smaller than max_moves"):
        GenerationConfig(max_moves=4, random_opening_max_plies=4)


def test_resume_rejects_a_missing_training_game_pgn(tmp_path: Path) -> None:
    dataset_path = tmp_path / "archive.pt"
    pgn_dir = tmp_path / "games"
    common = {
        "dataset_path": dataset_path,
        "pgn_dir": pgn_dir,
        "white_agent": "random",
        "black_agent": "random",
        "max_moves": 2,
        "seed": 31,
    }
    generate_supervised_data(GenerationConfig(games=1, **common))
    (pgn_dir / "game_0001.pgn").unlink()

    with pytest.raises(DataGenerationError, match="archived training-game PGNs are missing"):
        generate_supervised_data(GenerationConfig(games=2, **common))


def test_alternating_teacher_only_generation_keeps_anchor_and_minimax_moves(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "teacher.pt"
    pgn_dir = tmp_path / "teacher_games"
    summary = generate_supervised_data(
        GenerationConfig(
            dataset_path=dataset_path,
            pgn_dir=pgn_dir,
            games=3,
            white_agent="minimax",
            black_agent="random",
            minimax_depth=1,
            alternate_agents=True,
            record_agent="minimax",
            anchor_minimax_games=1,
            max_moves=6,
            seed=43,
        )
    )

    loaded = load_dataset(dataset_path)
    examples_by_game: dict[str, list[TrainingExample]] = {}
    for example in loaded:
        examples_by_game.setdefault(example.game_id, []).append(example)

    assert summary.completed_games == 3
    assert summary.examples == 12
    assert summary.unrecorded_agent_moves == 6
    assert [len(examples_by_game[key]) for key in sorted(examples_by_game)] == [6, 3, 3]
    assert all(example.metadata["policy_source_agent"] == "minimax" for example in loaded)
    assert all(
        example.metadata["anchor_game"] for example in examples_by_game[sorted(examples_by_game)[0]]
    )
    assert not any(
        example.metadata["anchor_game"]
        for key in sorted(examples_by_game)[1:]
        for example in examples_by_game[key]
    )

    second_game = load_pgn(pgn_dir / "game_0002.pgn")[0]
    third_game = load_pgn(pgn_dir / "game_0003.pgn")[0]
    assert second_game.headers["White"].startswith("Minimax")
    assert second_game.headers["Black"].startswith("Random")
    assert third_game.headers["White"].startswith("Random")
    assert third_game.headers["Black"].startswith("Minimax")
    assert third_game.headers["PolicySourceAgent"] == "minimax"
    assert third_game.headers["UnrecordedAgentPlies"] == "3"
    assert third_game.headers["RetainedTrainingExamples"] == "3"


def test_teacher_filter_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="record_agent"):
        GenerationConfig(record_agent="neural")
    with pytest.raises(ValueError, match="anchor_minimax_games"):
        GenerationConfig(games=2, anchor_minimax_games=3)
