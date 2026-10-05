"""Tests for gameplay-gated continuation from a frozen champion."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np

from chess_ai.curriculum import (
    GatedRefinementConfig,
    initialize_gated_refinement,
    run_gated_holdout_audit,
    run_gated_refinement,
)
from chess_ai.curriculum.gated_refinement import (
    _promotion_decision,
    _seeded_gate_opening,
    _seeded_holdout_openings,
)
from chess_ai.data import TrainingExample, save_dataset
from chess_ai.environment import BoardEncoder, MoveEncoder
from chess_ai.model import PolicyValueNet, save_checkpoint


def _config(tmp_path: Path, *, rounds: int = 1) -> GatedRefinementConfig:
    return GatedRefinementConfig(
        initial_champion_checkpoint=tmp_path / "source" / "epoch_0399.pt",
        replay_dataset_path=tmp_path / "replay.pt",
        state_path=tmp_path / "curriculum" / "state.json",
        champion_dir=tmp_path / "checkpoints" / "champions",
        candidate_dir=tmp_path / "checkpoints" / "candidates",
        corrections_dir=tmp_path / "curriculum" / "corrections",
        metrics_dir=tmp_path / "metrics",
        training_pgn_dir=tmp_path / "games" / "training",
        gate_pgn_dir=tmp_path / "games" / "gate",
        rounds=rounds,
        cycles_per_round=1,
        opponent_depth=1,
        teacher_depth=2,
        max_plies=2,
        opening_min_plies=0,
        opening_max_plies=0,
        normal_start_every_cycles=1,
        gate_openings=1,
        gate_opening_min_plies=0,
        gate_opening_max_plies=0,
        minimum_improvement_points=0.5,
        training_examples_per_round=16,
        correction_fraction=0.5,
        anchor_correction_share=0.5,
        batch_size=2,
        learning_rate=1e-4,
        value_loss_weight=0.0,
        seed=19,
        device="cpu",
    )


def _example(game_id: str) -> TrainingExample:
    board = chess.Board()
    move = chess.Move.from_uci("b1c3")
    action = MoveEncoder().encode(move)
    policy = np.zeros(4208, dtype=np.float32)
    policy[action] = 1.0
    return TrainingExample(
        board_tensor=BoardEncoder().encode(board),
        target_policy=policy,
        target_value=0.0,
        metadata={
            "game_id": game_id,
            "fen": board.fen(),
            "move_uci": move.uci(),
            "policy_action": action,
        },
    )


def test_gate_openings_are_fixed_and_promotion_requires_no_standard_regression(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        max_plies=20,
        gate_openings=3,
        gate_opening_min_plies=4,
        gate_opening_max_plies=8,
    )

    first = [_seeded_gate_opening(config, index) for index in range(3)]
    repeated = [_seeded_gate_opening(config, index) for index in range(3)]

    assert first == repeated
    assert first[0]["fen"] == chess.Board().fen()
    assert first[0]["standard_start"] is True
    assert all(len(item["moves"]) % 2 == 0 for item in first)
    holdout = _seeded_holdout_openings(config, openings=10, audit_seed=91_337)
    gate_random_fens = {item["fen"] for item in first[1:]}
    holdout_random_fens = {item["fen"] for item in holdout[1:]}
    assert gate_random_fens.isdisjoint(holdout_random_fens)
    assert len(holdout_random_fens) == 9

    champion = {"score_points": 5.0, "standard_score_points": 1.0}
    stronger = {"score_points": 5.5, "standard_score_points": 1.0}
    regressed = {"score_points": 6.0, "standard_score_points": 0.5}
    tied = {"score_points": 5.0, "standard_score_points": 1.0}

    assert _promotion_decision(config, champion, stronger)[0] is True
    assert _promotion_decision(config, champion, regressed)[0] is False
    assert _promotion_decision(config, champion, tied)[0] is False


def test_gated_smoke_run_freezes_source_rejects_tie_and_extends(tmp_path: Path) -> None:
    config = _config(tmp_path)
    save_checkpoint(
        config.initial_champion_checkpoint,
        PolicyValueNet(channels=4, residual_blocks=0),
        epoch=399,
    )
    original_bytes = config.initial_champion_checkpoint.read_bytes()
    save_dataset(
        config.replay_dataset_path,
        [_example(f"replay-{index}") for index in range(4)],
    )

    initialized = initialize_gated_refinement(config)

    assert initialized.completed_rounds == 0
    assert initialized.training_games == 0
    assert initialized.gate_games == 0
    assert initialized.champion_checkpoint.read_bytes() == original_bytes

    summary = run_gated_refinement(config)

    assert summary.completed_rounds == 1
    assert summary.training_cycles == 1
    assert summary.training_games == 2
    assert summary.gate_games == 4
    assert summary.promotions == 0
    assert summary.champion_generation == 0
    assert summary.champion_checkpoint == config.champion_dir / "generation_0000.pt"
    assert summary.champion_checkpoint.read_bytes() == original_bytes
    assert config.initial_champion_checkpoint.read_bytes() == original_bytes
    assert (config.champion_dir / "champion.json").is_file()
    assert (config.candidate_dir / "round_0001" / "best.pt").is_file()
    assert len(list(config.training_pgn_dir.rglob("*.pgn"))) == 2
    assert len(list(config.gate_pgn_dir.rglob("*.pgn"))) == 4
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["history"][0]["promoted"] is False
    assert "required fixed-suite score improvement" in state["history"][0]["decision"]
    assert run_gated_refinement(config).to_dict() == summary.to_dict()

    audit = run_gated_holdout_audit(config, openings=1, audit_seed=123_456)

    assert audit.games_per_checkpoint == 2
    assert audit.point_delta == 0.0
    assert audit.verdict == "tied"
    assert audit.supports_continuation is True
    assert audit.report_path.is_file()

    extended = run_gated_refinement(replace(config, rounds=2))

    assert extended.completed_rounds == 2
    assert extended.training_cycles == 2
    assert extended.training_games == 4
    assert extended.gate_games == 8
    assert len(list(config.training_pgn_dir.rglob("*.pgn"))) == 4
    assert len(list(config.gate_pgn_dir.rglob("*.pgn"))) == 8
