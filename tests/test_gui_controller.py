"""Tests for the UI-independent human-vs-neural game controller."""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Event, Thread

import chess
import chess.pgn
import numpy as np
import pytest

from chess_ai.data import TrainingExample, load_dataset, save_dataset
from chess_ai.environment.game import IllegalMoveError, NoMoveToUndoError
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.gui.controller import (
    AgentMoveError,
    GameNotCompleteError,
    HumanNeuralGame,
    TrainingDisabledError,
    WrongTurnError,
)
from chess_ai.gui.training_data import (
    ExplicitTrainingOptInRequired,
    HumanGuiDatasetError,
    append_human_gui_game,
    load_human_gui_dataset,
)


class ScriptedAgent:
    name = "Scripted neural test agent"

    def __init__(self, moves: list[str]) -> None:
        self._moves: Iterator[str] = iter(moves)

    def choose_move(self, _board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(next(self._moves))


class BlockingAgent:
    name = "Blocking neural test agent"

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def choose_move(self, _board: chess.Board) -> chess.Move:
        self.started.set()
        assert self.release.wait(timeout=5)
        return chess.Move.from_uci("e7e5")


def _game(
    tmp_path: Path,
    moves: list[str],
    *,
    session_id: str,
    human_color: chess.Color = chess.WHITE,
    training_enabled: bool = True,
    dataset_path: Path | None = None,
) -> HumanNeuralGame:
    return HumanNeuralGame(
        ScriptedAgent(moves),
        human_color=human_color,
        checkpoint_label="checkpoints/test.pt",
        training_enabled=training_enabled,
        session_seed=17,
        session_id=session_id,
        pgn_dir=tmp_path / "pgn",
        dataset_path=dataset_path or tmp_path / "human_gui.pt",
        clock=lambda: datetime(2026, 8, 5, 12, 0, tzinfo=UTC),
    )


def _finish_human_win(game: HumanNeuralGame) -> None:
    for human_move in ("e2e4", "f1c4", "d1h5"):
        game.play_human_move(human_move[:2], human_move[2:])
        game.play_ai_turn()
    game.play_human_move("h5", "f7")


def _finish_human_loss(game: HumanNeuralGame) -> None:
    game.play_human_move("f2", "f3")
    game.play_ai_turn()
    game.play_human_move("g2", "g4")
    game.play_ai_turn()


def _finish_draw(game: HumanNeuralGame) -> None:
    for _ in range(2):
        game.play_human_move("g1", "f3")
        game.play_ai_turn()
        game.play_human_move("f3", "g1")
        if not game.game_over:
            game.play_ai_turn()


def test_legal_queries_and_examples_contain_human_policies_only(tmp_path: Path) -> None:
    game = _game(tmp_path, ["e7e5", "b8c6", "g8f6"], session_id="human-win")

    assert chess.E2 in game.legal_sources()
    assert chess.E4 in game.legal_destinations("e2")
    assert game.promotion_choices("e2", "e4") == ()
    _finish_human_win(game)

    assert game.game_over
    assert game.result == "1-0"
    examples = game.complete()
    assert len(examples) == 4
    assert [example.metadata["move_uci"] for example in examples] == [
        "e2e4",
        "f1c4",
        "d1h5",
        "h5f7",
    ]
    encoder = MoveEncoder()
    for example in examples:
        action = int(np.argmax(example.target_policy))
        assert encoder.decode(action).uci() == example.metadata["move_uci"]
        assert np.count_nonzero(example.target_policy) == 1
        assert example.metadata["policy_source"] == "human"
        assert example.metadata["player_to_move"] == "white"


def test_final_values_use_the_stored_human_position_perspective(tmp_path: Path) -> None:
    win = _game(tmp_path, ["e7e5", "b8c6", "g8f6"], session_id="win")
    loss = _game(tmp_path, ["e7e5", "d8h4"], session_id="loss")
    draw = _game(
        tmp_path,
        ["g8f6", "f6g8", "g8f6", "f6g8"],
        session_id="draw",
    )

    _finish_human_win(win)
    _finish_human_loss(loss)
    _finish_draw(draw)

    assert {example.target_value for example in win.complete()} == {1.0}
    assert {example.target_value for example in loss.complete()} == {-1.0}
    assert {example.target_value for example in draw.complete()} == {0.0}
    assert draw.result == "1/2-1/2"


def test_undo_removes_the_last_human_decision_and_neural_reply(tmp_path: Path) -> None:
    game = _game(tmp_path, ["e7e5"], session_id="undo")
    game.play_human_move("e2", "e4")
    game.play_ai_turn()

    undone = game.undo_turn()

    assert tuple(move.uci() for move in undone) == ("e7e5", "e2e4")
    assert game.board.fen() == chess.STARTING_FEN
    assert game.pending_human_examples == ()
    assert game.is_human_turn

    game.play_human_move("d2", "d4")
    assert tuple(move.uci() for move in game.undo_turn()) == ("d2d4",)
    assert game.board.fen() == chess.STARTING_FEN


def test_illegal_moves_wrong_turns_and_bad_agent_output_are_rejected(tmp_path: Path) -> None:
    game = _game(tmp_path, ["e2e4"], session_id="illegal")

    with pytest.raises(IllegalMoveError):
        game.play_human_move("e2", "e5")
    assert game.move_history == ()
    assert game.pending_human_examples == ()

    game.play_human_move("e2", "e4")
    with pytest.raises(WrongTurnError):
        game.play_human_move("d2", "d4")
    with pytest.raises(AgentMoveError):
        game.play_ai_turn()

    black_human = _game(
        tmp_path,
        ["e2e4"],
        session_id="black-opening",
        human_color=chess.BLACK,
    )
    with pytest.raises(WrongTurnError):
        black_human.play_human_move("e7", "e5")
    black_human.play_ai_turn()
    with pytest.raises(NoMoveToUndoError):
        black_human.undo_turn()


def test_completed_games_append_atomically_and_resume_without_mixing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "human_gui.pt"
    first = _game(
        tmp_path,
        ["e7e5"],
        session_id="append-one",
        dataset_path=dataset_path,
    )
    first.play_human_move("e2", "e4")
    first.play_ai_turn()
    first.resign_human()

    with pytest.raises(ExplicitTrainingOptInRequired):
        first.append_training_examples(confirm_training=False)
    first_result = first.append_training_examples(confirm_training=True)
    assert first_result.added_examples == 1
    assert first_result.completed_games == 1

    second = _game(
        tmp_path,
        ["e2e4", "g1f3"],
        session_id="append-two",
        human_color=chess.BLACK,
        dataset_path=dataset_path,
    )
    second.play_ai_turn()
    second.play_human_move("e7", "e5")
    second.play_ai_turn()
    second.resign_human()
    second_result = second.append_training_examples(confirm_training=True)

    loaded = load_human_gui_dataset(dataset_path)
    assert second_result.total_examples == 2
    assert second_result.completed_games == 2
    assert len(loaded) == 2
    assert len({example.game_id for example in loaded}) == 2
    assert all(example.metadata["training_opt_in_confirmed"] is True for example in loaded)
    assert loaded.metadata["policy_source"] == "human_moves_only"


def test_non_human_gui_dataset_is_never_reused_as_append_target(tmp_path: Path) -> None:
    dataset_path = tmp_path / "not_human_gui.pt"
    source = _game(tmp_path, ["e7e5"], session_id="wrong-origin")
    source.play_human_move("e2", "e4")
    source.play_ai_turn()
    source.resign_human()
    save_dataset(dataset_path, source.complete(), metadata={"kind": "external_benchmark"})

    destination = _game(
        tmp_path,
        ["e7e5"],
        session_id="refuse-mix",
        dataset_path=dataset_path,
    )
    destination.play_human_move("e2", "e4")
    destination.play_ai_turn()
    destination.resign_human()

    with pytest.raises(HumanGuiDatasetError, match="Refusing to mix"):
        destination.append_training_examples(confirm_training=True)
    assert load_dataset(dataset_path).metadata["kind"] == "external_benchmark"


def test_pgn_round_trip_preserves_result_headers_and_position(tmp_path: Path) -> None:
    game = _game(tmp_path, ["e7e5"], session_id="pgn")
    game.play_human_move("e2", "e4")
    game.play_ai_turn()
    game.resign_human()

    saved = game.save_pgn()
    parsed = chess.pgn.read_game(io.StringIO(saved.read_text(encoding="utf-8")))

    assert parsed is not None
    assert parsed.errors == []
    assert parsed.headers["Result"] == "0-1"
    assert parsed.headers["Termination"] == "human_resignation"
    assert parsed.headers["GameId"] == game.game_id
    assert parsed.headers["Checkpoint"] == "checkpoints/test.pt"
    assert parsed.end().board().fen() == game.board.fen()


def test_unfinished_or_disabled_games_produce_no_saved_examples(tmp_path: Path) -> None:
    unfinished = _game(tmp_path, ["e7e5"], session_id="unfinished")
    unfinished.play_human_move("e2", "e4")
    assert len(unfinished.pending_human_examples) == 1
    with pytest.raises(GameNotCompleteError):
        unfinished.complete()
    with pytest.raises(GameNotCompleteError):
        unfinished.append_training_examples(confirm_training=True)
    assert not unfinished.dataset_path.exists()

    disabled = _game(
        tmp_path,
        ["e7e5"],
        session_id="disabled",
        training_enabled=False,
    )
    disabled.play_human_move("e2", "e4")
    disabled.play_ai_turn()
    disabled.resign_human()
    assert disabled.pending_human_examples == ()
    assert disabled.complete() == ()
    with pytest.raises(TrainingDisabledError):
        disabled.append_training_examples(confirm_training=True)


def test_game_ids_are_deterministic_within_an_injected_session(tmp_path: Path) -> None:
    first = _game(tmp_path, [], session_id="deterministic")
    second = _game(tmp_path, [], session_id="deterministic")

    assert first.game_id == second.game_id
    original = first.game_id
    assert first.new_game() == second.new_game()
    assert first.game_id != original


def test_ai_inference_does_not_lock_rendering_and_stale_move_is_discarded(
    tmp_path: Path,
) -> None:
    agent = BlockingAgent()
    game = HumanNeuralGame(
        agent,
        human_color=chess.WHITE,
        checkpoint_label="checkpoints/test.pt",
        training_enabled=True,
        session_id="background-ai",
        pgn_dir=tmp_path,
        dataset_path=tmp_path / "human_gui.pt",
    )
    game.play_human_move("e2", "e4")
    failures: list[Exception] = []

    def run_ai() -> None:
        try:
            game.play_ai_turn()
        except Exception as exc:
            failures.append(exc)

    worker = Thread(target=run_ai)
    worker.start()
    assert agent.started.wait(timeout=5)

    # Both operations acquire the controller lock.  They complete while the
    # agent remains blocked only if inference released that lock.
    assert game.board.peek().uci() == "e2e4"
    game.new_game()
    agent.release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], AgentMoveError)
    assert "stale move" in str(failures[0])
    assert game.board.fen() == chess.STARTING_FEN


def test_human_dataset_rejects_a_board_tensor_that_disagrees_with_its_fen(
    tmp_path: Path,
) -> None:
    game = _game(tmp_path, ["e7e5"], session_id="corrupt-board")
    game.play_human_move("e2", "e4")
    game.play_ai_turn()
    game.resign_human()
    original = game.complete()[0]
    corrupted_board = original.board_tensor.copy()
    corrupted_board[0, 0, 0] = 1.0 - corrupted_board[0, 0, 0]
    corrupted = TrainingExample(
        board_tensor=corrupted_board,
        target_policy=original.target_policy,
        target_value=original.target_value,
        metadata=original.metadata,
    )

    with pytest.raises(HumanGuiDatasetError, match="board tensor"):
        append_human_gui_game(
            tmp_path / "corrupt.pt",
            [corrupted],
            confirm_training=True,
        )


def test_concurrent_human_game_appends_do_not_lose_either_game(tmp_path: Path) -> None:
    dataset_path = tmp_path / "concurrent.pt"
    games = [
        _game(
            tmp_path,
            ["e7e5"],
            session_id=f"concurrent-{index}",
            dataset_path=dataset_path,
        )
        for index in range(2)
    ]
    for game in games:
        game.play_human_move("e2", "e4")
        game.play_ai_turn()
        game.resign_human()

    start = Barrier(2)
    failures: list[Exception] = []

    def append(game: HumanNeuralGame) -> None:
        try:
            start.wait(timeout=5)
            game.append_training_examples(confirm_training=True)
        except Exception as exc:
            failures.append(exc)

    workers = [Thread(target=append, args=(game,)) for game in games]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert not any(worker.is_alive() for worker in workers)
    assert failures == []
    loaded = load_human_gui_dataset(dataset_path)
    assert len(loaded) == 2
    assert {example.game_id for example in loaded} == {game.game_id for game in games}
