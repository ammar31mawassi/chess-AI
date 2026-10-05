"""Tests for the fixed four-cohort offline collection."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn

from chess_ai.config import config_section, load_config
from chess_ai.curriculum import MixedBatchConfig, finalize_mixed_batch, run_mixed_batch
from chess_ai.curriculum.mixed_batch import (
    LEAGUE_MATCHUPS,
    MATCHUPS,
    _opening_board,
    _outcome_for_color,
)
from chess_ai.data import load_dataset
from chess_ai.model import PolicyValueNet, save_checkpoint


def _config(tmp_path: Path, *, opening_full_moves: int = 0) -> MixedBatchConfig:
    return MixedBatchConfig(
        neural_checkpoint=tmp_path / "champion" / "generation_0002.pt",
        state_path=tmp_path / "curriculum" / "state.json",
        manifest_path=tmp_path / "curriculum" / "manifest.json",
        dataset_path=tmp_path / "datasets" / "mixed.pt",
        opening_dataset_dir=tmp_path / "curriculum" / "openings",
        pgn_dir=tmp_path / "games",
        opening_positions=1,
        minimax_d1_depth=1,
        minimax_d2_depth=1,
        teacher_depth=1,
        max_plies=4,
        opening_min_full_moves=opening_full_moves,
        opening_max_full_moves=opening_full_moves,
        seed=41,
        device="cpu",
    )


def _league_config(tmp_path: Path) -> MixedBatchConfig:
    return MixedBatchConfig(
        neural_checkpoint=None,
        state_path=tmp_path / "league" / "state.json",
        manifest_path=tmp_path / "league" / "manifest.json",
        dataset_path=tmp_path / "league" / "dataset.pt",
        opening_dataset_dir=tmp_path / "league" / "openings",
        pgn_dir=tmp_path / "league" / "games",
        opening_positions=1,
        experiment="minimax_league",
        minimax_d1_depth=1,
        minimax_d2_depth=1,
        minimax_d3_depth=1,
        teacher_depth=1,
        max_plies=4,
        opening_min_full_moves=0,
        opening_max_full_moves=0,
        normal_start_every_positions=1,
        seed=43,
        device="cpu",
    )


def test_opening_is_reused_and_reproducible_in_full_moves(tmp_path: Path) -> None:
    config = _config(tmp_path, opening_full_moves=1)

    board, moves, seed = _opening_board(config, 1)
    repeated, repeated_moves, repeated_seed = _opening_board(config, 1)

    assert len(moves) == 2
    assert board.turn == chess.WHITE
    assert board.fen() == repeated.fen()
    assert moves == repeated_moves
    assert seed == repeated_seed


def test_outcome_is_assigned_to_the_actor_color() -> None:
    assert _outcome_for_color("1-0", chess.WHITE) == "win"
    assert _outcome_for_color("1-0", chess.BLACK) == "loss"
    assert _outcome_for_color("0-1", chess.WHITE) == "loss"
    assert _outcome_for_color("0-1", chess.BLACK) == "win"
    assert _outcome_for_color("1/2-1/2", chess.WHITE) == "draw"


def test_one_opening_saves_all_eight_games_and_never_trains(tmp_path: Path) -> None:
    config = _config(tmp_path)
    save_checkpoint(
        config.neural_checkpoint,
        PolicyValueNet(channels=4, residual_blocks=0),
        epoch=2,
    )

    summary = run_mixed_batch(config)

    assert summary.completed_opening_positions == 1
    assert summary.games == 8
    assert summary.examples == 32
    assert set(summary.cohort_results) == {matchup.key for matchup in MATCHUPS}
    assert all(result["games"] == 2 for result in summary.cohort_results.values())
    assert summary.teacher_agreements + summary.teacher_disagreements == summary.examples
    assert len(list(config.pgn_dir.glob("**/*.pgn"))) == 8

    dataset = load_dataset(config.dataset_path)
    assert len(dataset) == summary.examples
    assert dataset.metadata["games"] == 8
    assert dataset.metadata["training_during_collection"] is False
    assert dataset.metadata["outcome_weight_key"] == "actor_outcome_weight"
    assert {item.metadata["matchup"] for item in dataset} == {matchup.key for matchup in MATCHUPS}
    assert {float(item.metadata["actor_outcome_weight"]) for item in dataset}.issubset(
        {1.0, 1.5, 2.0}
    )
    assert len({item.metadata["opening_set_id"] for item in dataset}) == 1

    for pgn_path in config.pgn_dir.glob("**/*.pgn"):
        with pgn_path.open(encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        assert game is not None
        assert game.headers["TrainingDuringCollection"] == "false"

    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["training_updates"] == 0
    assert manifest["training_updates"] == 0
    assert len(manifest["pairs"]) == 1

    resumed = run_mixed_batch(config)
    assert resumed.resumed is True
    assert resumed.games == 8


def test_one_league_opening_saves_three_color_switched_pairs(tmp_path: Path) -> None:
    config = _league_config(tmp_path)

    summary = run_mixed_batch(config)

    assert summary.games == 6
    assert summary.examples == 24
    assert summary.neural_checkpoint is None
    assert set(summary.cohort_results) == {matchup.key for matchup in LEAGUE_MATCHUPS}
    assert all(result["games"] == 2 for result in summary.cohort_results.values())
    assert len(list(config.pgn_dir.glob("**/*.pgn"))) == 6


def test_large_league_config_is_exactly_three_thousand_games() -> None:
    raw = load_config("configs/minimax_d1_d2_d3_3000.yaml")
    config = MixedBatchConfig.from_mapping(
        config_section(raw, "mixed_batch"),
        seed=int(raw["seed"]),
        device=str(raw["device"]),
    )

    assert config.experiment == "minimax_league"
    assert config.opening_positions * len(LEAGUE_MATCHUPS) * 2 == 3_000
    assert config.opening_min_full_moves == 2
    assert config.opening_max_full_moves == 5
    assert config.normal_start_every_positions == 10

    random_board, random_moves, _random_seed = _opening_board(config, 9)
    anchor_board, anchor_moves, _anchor_seed = _opening_board(config, 10)
    assert 4 <= len(random_moves) <= 10
    assert random_board != chess.Board()
    assert anchor_moves == []
    assert anchor_board == chess.Board()


def test_partial_collection_can_be_permanently_finalized(tmp_path: Path) -> None:
    config = _league_config(tmp_path)
    config = MixedBatchConfig(
        **{
            **config.signature_payload(),
            "opening_positions": 2,
            "state_path": config.state_path,
            "manifest_path": config.manifest_path,
            "dataset_path": config.dataset_path,
            "opening_dataset_dir": config.opening_dataset_dir,
            "pgn_dir": config.pgn_dir,
        }
    )
    # Produce one complete shard using the same signed two-opening configuration.
    from chess_ai.curriculum import mixed_batch as module

    state = module._initial_state(config, None)
    first = module._play_opening(config, None, None, 1)
    state["history"] = [first]
    state["completed_opening_positions"] = 1
    state.update(module._aggregate_history(state["history"], module._matchups(config)))
    module._atomic_json(config.state_path, state)

    summary = finalize_mixed_batch(config)

    assert summary.completed_opening_positions == 1
    assert summary.requested_opening_positions == 2
    assert summary.games == 6
    dataset = load_dataset(config.dataset_path)
    assert dataset.metadata["finalized_early"] is True
    assert dataset.metadata["opening_positions"] == 1
    assert dataset.metadata["requested_opening_positions"] == 2
    saved_state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "complete"
    assert saved_state["finalized_early"] is True

    resumed = run_mixed_batch(config)
    assert resumed.completed_opening_positions == 1
