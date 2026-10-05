"""Tests for offline paired collection with all-ply D2 teaching traces."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.curriculum import TeacherBatchConfig, run_teacher_batch
from chess_ai.curriculum.teacher_batch import _opening_board, _soft_teacher_policy
from chess_ai.data import load_dataset
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model import PolicyValueNet, save_checkpoint


def _config(tmp_path: Path, *, opening_full_moves: int = 0) -> TeacherBatchConfig:
    return TeacherBatchConfig(
        champion_checkpoint=tmp_path / "champion" / "generation_0001.pt",
        state_path=tmp_path / "curriculum" / "state.json",
        manifest_path=tmp_path / "curriculum" / "manifest.json",
        dataset_path=tmp_path / "datasets" / "teacher_batch.pt",
        pair_dataset_dir=tmp_path / "curriculum" / "pairs",
        pgn_dir=tmp_path / "games",
        opening_pairs=1,
        opponent_depth=1,
        teacher_depth=2,
        max_plies=4,
        opening_min_full_moves=opening_full_moves,
        opening_max_full_moves=opening_full_moves,
        teacher_policy_temperature=100.0,
        teacher_policy_top_k=5,
        champion_win_weight=2.0,
        champion_draw_weight=1.0,
        champion_loss_weight=1.0,
        seed=29,
        device="cpu",
        resume=True,
    )


def test_opening_is_reproducible_and_measured_in_full_moves(tmp_path: Path) -> None:
    config = _config(tmp_path, opening_full_moves=1)

    board, moves, seed = _opening_board(config, 1)
    repeated, repeated_moves, repeated_seed = _opening_board(config, 1)

    assert len(moves) == 2
    assert board.turn == chess.WHITE
    assert board.fen() == repeated.fen()
    assert moves == repeated_moves
    assert seed == repeated_seed


def test_soft_teacher_policy_uses_only_legal_top_scored_moves() -> None:
    board = chess.Board()
    analysis = MinimaxAgent(depth=1).analyze_moves(board)
    encoder = MoveEncoder()

    policy, best, trace = _soft_teacher_policy(
        analysis,
        top_k=5,
        temperature=100.0,
        move_encoder=encoder,
    )

    assert np.isclose(policy.sum(), 1.0)
    assert 1 <= np.count_nonzero(policy) <= 5
    assert best.score == max(item.score for item in analysis)
    assert trace[0]["move_uci"] == best.move.uci()
    assert all(encoder.decode(int(index)) in board.legal_moves for index in np.flatnonzero(policy))


def test_one_pair_collection_saves_two_complete_games_and_never_trains(tmp_path: Path) -> None:
    config = _config(tmp_path, opening_full_moves=1)
    save_checkpoint(
        config.champion_checkpoint,
        PolicyValueNet(channels=4, residual_blocks=0),
        epoch=1,
    )

    summary = run_teacher_batch(config)

    assert summary.completed_opening_pairs == 1
    assert summary.games == 2
    assert summary.examples == 4
    assert summary.champion_wins + summary.champion_draws + summary.champion_losses == 2
    assert summary.teacher_agreements + summary.teacher_disagreements == summary.examples
    assert len(list(config.pgn_dir.glob("game_*.pgn"))) == 2
    dataset = load_dataset(config.dataset_path)
    assert len(dataset) == summary.examples
    assert dataset.metadata["training_during_collection"] is False
    assert {item.metadata["actor_role"] for item in dataset} == {
        "champion",
        "minimax_d1_opponent",
    }
    assert {float(item.metadata["champion_outcome_weight"]) for item in dataset}.issubset(
        {1.0, 2.0}
    )
    assert all(
        item.metadata["opening_pair_id"] == dataset[0].metadata["opening_pair_id"]
        for item in dataset
    )

    first_pgn = sorted(config.pgn_dir.glob("game_*.pgn"))[0]
    with first_pgn.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    assert game is not None
    assert len(list(game.mainline_moves())) == 4
    assert game.headers["TrainingDuringCollection"] == "false"

    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["training_updates"] == 0
    assert manifest["training_updates"] == 0
    resumed = run_teacher_batch(config)
    assert resumed.resumed is True
    assert resumed.dataset_path == summary.dataset_path
    assert resumed.examples == summary.examples
