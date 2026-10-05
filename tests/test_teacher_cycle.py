"""Focused tests for the D1-opponent/D2-teacher correction curriculum."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np

from chess_ai.curriculum import TeacherCycleConfig, run_teacher_cycle
from chess_ai.curriculum.teacher_cycle import (
    _CorrectionRecordingAgent,
    _legacy_signature_without_anchor_replay,
    _legacy_signature_without_normal_start,
    _opening_board,
    _sample_training_examples,
)
from chess_ai.data import TrainingExample, load_dataset, save_dataset
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model import PolicyValueNet, save_checkpoint


class _FirstLegalStudent:
    name = "test student"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return sorted(board.legal_moves, key=lambda move: move.uci())[0]


def _config(tmp_path: Path, *, cycles: int = 2) -> TeacherCycleConfig:
    return TeacherCycleConfig(
        source_checkpoint=tmp_path / "source" / "best.pt",
        replay_dataset_path=tmp_path / "replay.pt",
        state_path=tmp_path / "curriculum" / "state.json",
        corrections_dir=tmp_path / "curriculum" / "corrections",
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_path=tmp_path / "metrics.jsonl",
        pgn_dir=tmp_path / "games",
        cycles=cycles,
        opponent_depth=1,
        teacher_depth=2,
        max_plies=2,
        opening_min_plies=0,
        opening_max_plies=0,
        training_examples_per_cycle=16,
        correction_fraction=0.5,
        batch_size=2,
        learning_rate=1e-4,
        mastery_window_cycles=2,
        seed=17,
        device="cpu",
    )


def _replay_example(
    game_id: str,
    *,
    cycle: int | None = None,
    normal_start: bool | None = None,
) -> TrainingExample:
    board = chess.Board()
    action = MoveEncoder().encode(chess.Move.from_uci("b1c3"))
    policy = np.zeros(4208, dtype=np.float32)
    policy[action] = 1.0
    metadata: dict[str, str | int | bool] = {
        "game_id": game_id,
        "fen": board.fen(),
        "move_uci": "b1c3",
        "policy_action": action,
    }
    if cycle is not None:
        metadata["cycle"] = cycle
    if normal_start is not None:
        metadata["normal_start"] = normal_start
    return TrainingExample(
        board_tensor=BoardEncoder().encode(board),
        target_policy=policy,
        target_value=0.0,
        metadata=metadata,
    )


def test_recording_agent_labels_the_students_position_with_teacher_move(tmp_path: Path) -> None:
    from chess_ai.agents.minimax_agent import MinimaxAgent

    board = chess.Board()
    recorder = _CorrectionRecordingAgent(
        _FirstLegalStudent(),  # type: ignore[arg-type]
        MinimaxAgent(depth=2),
        game_id="correction-game",
        cycle=1,
        student_color=chess.WHITE,
        checkpoint=tmp_path / "student.pt",
        opponent_depth=1,
    )

    student_move = recorder.choose_move(board)
    examples = recorder.finalize("0-1", "checkmate")

    assert len(examples) == 1
    example = examples[0]
    teacher_move = chess.Move.from_uci(str(example.metadata["teacher_move_uci"]))
    assert teacher_move in board.legal_moves
    assert example.metadata["student_move_uci"] == student_move.uci()
    assert MoveEncoder().decode(int(example.metadata["policy_action"])) == teacher_move
    assert example.target_value == -1.0


def test_seeded_opening_is_reproducible_even_and_changes_by_cycle(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path, cycles=6),
        max_plies=20,
        opening_min_plies=4,
        opening_max_plies=8,
        normal_start_every_cycles=5,
    )

    first_board, first_moves, first_seed = _opening_board(config, 1)
    repeated_board, repeated_moves, repeated_seed = _opening_board(config, 1)
    next_board, next_moves, _next_seed = _opening_board(config, 2)
    anchor_board, anchor_moves, _anchor_seed = _opening_board(config, 5)

    assert first_board.fen() == repeated_board.fen()
    assert first_moves == repeated_moves
    assert first_seed == repeated_seed
    assert len(first_moves) % 2 == 0
    assert first_board.turn == chess.WHITE
    assert (next_board.fen(), next_moves) != (first_board.fen(), first_moves)
    assert anchor_board.fen() == chess.Board().fen()
    assert anchor_moves == []


def test_two_cycle_smoke_run_saves_four_games_and_holds_out_final_pair(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = PolicyValueNet(channels=4, residual_blocks=0)
    save_checkpoint(config.source_checkpoint, source, epoch=0)
    save_dataset(
        config.replay_dataset_path,
        [_replay_example(f"replay-{index}") for index in range(4)],
    )

    summary = run_teacher_cycle(config)

    assert summary.completed_cycles == 2
    assert summary.games_played == 4
    assert summary.training_updates == 1
    assert summary.current_checkpoint == config.checkpoint_dir / "last.pt"
    assert len(list(config.pgn_dir.glob("cycle_*.pgn"))) == 4
    assert len(load_dataset(config.corrections_dir / "cycle_0001.pt")) > 0
    assert len(load_dataset(config.corrections_dir / "cycle_0002.pt")) > 0
    assert (config.pgn_dir / "standard_evaluation" / "after_cycle_0001_student_white.pgn").is_file()
    first_state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert "post_training_standard_evaluation" in first_state["history"][0]
    assert first_state["history"][0]["training_batch"]["correction_examples"] > 0
    assert run_teacher_cycle(config).to_dict() == summary.to_dict()

    # Simulate the user's completed state, whose signature predates periodic
    # normal-start anchors, then add the refinement during an upward extension.
    raw_state = json.loads(config.state_path.read_text(encoding="utf-8"))
    raw_state["config_signature"] = _legacy_signature_without_normal_start(config, cycles=2)
    config.state_path.write_text(json.dumps(raw_state), encoding="utf-8")
    extended = run_teacher_cycle(replace(config, cycles=3, normal_start_every_cycles=1))

    assert extended.completed_cycles == 3
    assert extended.games_played == 6
    assert extended.training_updates == 2
    assert len(list(config.pgn_dir.glob("cycle_*.pgn"))) == 6
    extended_state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert extended_state["history"][-1]["normal_start"] is True

    # The user's 500-cycle state already had normal-start anchors, but predates
    # reserved anchor replay. It can likewise adopt the refinement on extension.
    extended_config = replace(config, cycles=3, normal_start_every_cycles=1)
    extended_state["config_signature"] = _legacy_signature_without_anchor_replay(
        extended_config,
        cycles=3,
    )
    config.state_path.write_text(json.dumps(extended_state), encoding="utf-8")
    twice_extended = run_teacher_cycle(replace(extended_config, cycles=4))

    assert twice_extended.completed_cycles == 4
    assert twice_extended.training_updates == 3


def test_training_sample_reserves_anchor_correction_share(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        training_examples_per_cycle=20,
        correction_fraction=0.5,
        anchor_correction_share=0.4,
    )
    replay = [_replay_example(f"replay-{index}") for index in range(20)]
    anchors = [
        # Older saved correction files identify anchors through state history;
        # they predate the per-example normal_start metadata flag.
        _replay_example(f"anchor-{index}", cycle=index + 1)
        for index in range(8)
    ]
    varied = [
        _replay_example(f"varied-{index}", cycle=index + 20, normal_start=False)
        for index in range(20)
    ]
    newest = varied[-2:]
    sample_stats: dict[str, int | float] = {}

    selected = _sample_training_examples(
        config,
        replay,
        [*anchors, *varied],
        newest,
        cycle=99,
        anchor_cycles=set(range(1, 9)),
        sample_stats=sample_stats,
    )

    selected_corrections = [
        example for example in selected if not example.game_id.startswith("replay-")
    ]
    selected_anchors = [
        example for example in selected_corrections if example.metadata.get("cycle") in range(1, 9)
    ]
    assert len(selected_corrections) == 10
    assert len(selected_anchors) >= 4
    assert sample_stats["anchor_correction_requested"] == 4
    assert sample_stats["anchor_correction_examples"] >= 4
    assert {example.game_id for example in newest}.issubset(
        {example.game_id for example in selected_corrections}
    )
