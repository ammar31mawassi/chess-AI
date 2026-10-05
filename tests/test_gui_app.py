from __future__ import annotations

import os
from pathlib import Path
from queue import Queue
from typing import Any

import chess
import pytest

from chess_ai.gui.app import (
    PIECE_SYMBOLS,
    NeuralChessApp,
    discover_default_checkpoint,
    grid_to_square,
    square_to_grid,
)
from chess_ai.gui.session_training import SessionTrainingPlan, SessionTrainingResult


class _StringVarStub:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def _training_plan(tmp_path: Path, session_id: str = "gui-session") -> SessionTrainingPlan:
    return SessionTrainingPlan(
        version=1,
        session_id=session_id,
        created_utc="2026-08-05T12:00:00+00:00",
        config_path=tmp_path / session_id / "training.yaml",
        checkpoint_dir=tmp_path / "checkpoints" / session_id,
        metrics_path=tmp_path / "metrics" / f"{session_id}.jsonl",
        dataset_path=tmp_path / "human_gui.pt",
        template_path=tmp_path / "human_gui.yaml",
    )


@pytest.mark.parametrize("flipped", [False, True])
def test_board_grid_mapping_round_trips_every_square(flipped: bool) -> None:
    for square in chess.SQUARES:
        row, column = square_to_grid(square, flipped=flipped)
        assert grid_to_square(row, column, flipped=flipped) == square


def test_board_orientation_places_the_expected_corner_squares() -> None:
    assert grid_to_square(7, 0, flipped=False) == chess.A1
    assert grid_to_square(0, 7, flipped=False) == chess.H8
    assert grid_to_square(7, 0, flipped=True) == chess.H8
    assert grid_to_square(0, 7, flipped=True) == chess.A1


def test_default_checkpoint_prefers_gpu_first_then_dev(tmp_path: Path) -> None:
    dev = tmp_path / "dev" / "best.pt"
    gpu = tmp_path / "gpu_first" / "best.pt"
    dev.parent.mkdir(parents=True)
    gpu.parent.mkdir(parents=True)
    dev.touch()
    assert discover_default_checkpoint(tmp_path) == dev
    gpu.touch()
    assert discover_default_checkpoint(tmp_path) == gpu


def test_default_checkpoint_prefers_the_newest_automatic_human_session(
    tmp_path: Path,
) -> None:
    gpu = tmp_path / "gpu_first" / "best.pt"
    older = tmp_path / "human_sessions" / "session-one" / "best.pt"
    newer = tmp_path / "human_sessions" / "session-two" / "best.pt"
    for candidate in (gpu, older, newer):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert discover_default_checkpoint(tmp_path) == newer


def test_default_checkpoint_uses_newest_other_best_then_last(tmp_path: Path) -> None:
    older = tmp_path / "one" / "best.pt"
    newer = tmp_path / "two" / "best.pt"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.touch()
    newer.touch()
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    assert discover_default_checkpoint(tmp_path) == newer

    older.unlink()
    newer.unlink()
    latest = tmp_path / "candidate" / "last.pt"
    latest.parent.mkdir(parents=True)
    latest.touch()
    assert discover_default_checkpoint(tmp_path) == latest


def test_piece_symbols_cover_standard_chess() -> None:
    assert len(PIECE_SYMBOLS) == 12
    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            assert PIECE_SYMBOLS[(color, piece_type)]


def test_finished_game_can_retry_after_a_pgn_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinishedController:
        game_over = True
        game_id = "retry-game"
        status = "Neural AI wins - human resignation."
        training_enabled = False

        def __init__(self) -> None:
            self.attempts = 0

        def save_pgn(self) -> Path:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("temporary save failure")
            return tmp_path / "retry.pgn"

    controller = FinishedController()
    app = object.__new__(NeuralChessApp)
    app.controller = controller  # type: ignore[assignment]
    app._finalized_game_id = None
    app.root = object()  # type: ignore[assignment]
    app.status_var = _StringVarStub()  # type: ignore[assignment]
    updates: list[str] = []
    app._update_controls = lambda: updates.append("controls")  # type: ignore[method-assign]
    app.render_board = lambda: updates.append("board")  # type: ignore[method-assign]
    app._render_history = lambda: updates.append("history")  # type: ignore[method-assign]
    errors: list[str] = []
    monkeypatch.setattr(
        "chess_ai.gui.app.messagebox.showerror",
        lambda _title, detail, **_kwargs: errors.append(str(detail)),
    )

    app._finalize_game()

    assert app._finalized_game_id is None
    assert "Retry Save" in app.status_var.value
    assert errors == ["temporary save failure"]

    app._finalize_game()

    assert controller.attempts == 2
    assert app._finalized_game_id == controller.game_id
    assert "PGN saved" in app.status_var.value
    assert "board" in updates
    assert "history" in updates


def test_unexpected_ai_worker_error_is_returned_to_the_ui_queue() -> None:
    class FailingController:
        game_over = False
        is_human_turn = False

        def play_ai_turn(self) -> chess.Move:
            raise KeyError("unexpected inference failure")

    app = object.__new__(NeuralChessApp)
    app.controller = FailingController()  # type: ignore[assignment]
    app._worker_token = 0
    app._ai_busy = False
    app._training_busy = False
    app.status_var = _StringVarStub()  # type: ignore[assignment]
    app._update_controls = lambda: None  # type: ignore[method-assign]
    app._ai_results = Queue()  # type: ignore[assignment]

    app._start_ai_turn()
    result: Any = app._ai_results.get(timeout=2)

    assert result.error is not None
    assert "unexpected inference failure" in result.error


def test_confirmed_game_locks_training_to_the_controller_settings(tmp_path: Path) -> None:
    class ControllerStub:
        checkpoint_label = str(tmp_path / "played-checkpoint.pt")
        dataset_path = tmp_path / "confirmed-human.pt"
        agent = object()

    app = object.__new__(NeuralChessApp)
    app._training_session = _training_plan(tmp_path)
    app._session_source_checkpoint = None
    app._session_dataset_path = None
    app._session_device = None
    app._session_setup_error = None
    app._session_confirmed_games = 0
    app._session_confirmed_examples = 0
    app.checkpoint_var = _StringVarStub("edited-after-game.pt")  # type: ignore[assignment]
    app.dataset_var = _StringVarStub("edited-after-game.pt")  # type: ignore[assignment]
    app.device_var = _StringVarStub("cuda")  # type: ignore[assignment]
    app.session_var = _StringVarStub()  # type: ignore[assignment]
    synced: list[Path] = []
    app._sync_session_dataset = lambda path: synced.append(path) or True  # type: ignore[method-assign]

    app._record_confirmed_game(ControllerStub(), 17)  # type: ignore[arg-type]

    assert app._session_source_checkpoint == (tmp_path / "played-checkpoint.pt").resolve()
    assert app._session_dataset_path == (tmp_path / "confirmed-human.pt").resolve()
    assert app._session_device == "cuda"
    assert app._session_confirmed_games == 1
    assert app._session_confirmed_examples == 17
    assert Path(app.checkpoint_var.get()) == app._session_source_checkpoint
    assert Path(app.dataset_var.get()) == app._session_dataset_path
    assert synced == [app._session_dataset_path]


def test_done_action_refuses_to_discard_an_unfinished_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ActiveController:
        game_over = False
        game_id = "active-game"

    app = object.__new__(NeuralChessApp)
    app._ai_busy = False
    app._training_busy = False
    app.controller = ActiveController()  # type: ignore[assignment]
    app._finalized_game_id = None
    app.root = object()  # type: ignore[assignment]
    messages: list[str] = []
    monkeypatch.setattr(
        "chess_ai.gui.app.messagebox.showinfo",
        lambda _title, detail, **_kwargs: messages.append(str(detail)),
    )

    app.finish_session_and_train()

    assert messages and "Finish or resign" in messages[0]


def test_completed_training_selects_candidate_and_reserves_the_next_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _training_plan(tmp_path, "completed")
    next_plan = _training_plan(tmp_path, "next")
    best = completed.best_checkpoint
    result = SessionTrainingResult(
        plan=completed,
        command=("python", "-m", "chess_ai"),
        returncode=0,
        stdout="done",
        stderr="",
        best_checkpoint=best,
        source_checkpoint=tmp_path / "source.pt",
        dataset_path=completed.dataset_path,
        dataset_examples=20,
        dataset_games=2,
        dataset_created_utc="2026-08-05T12:00:00+00:00",
    )
    app = object.__new__(NeuralChessApp)
    app._training_session = completed
    app._session_dataset_path = completed.dataset_path
    app._session_confirmed_games = 2
    app._session_confirmed_examples = 20
    app._session_game_number = 2
    app._session_source_checkpoint = tmp_path / "source.pt"
    app._session_device = "cuda"
    app._session_setup_error = None
    app.controller = object()  # type: ignore[assignment]
    app._agent_cache = object()  # type: ignore[assignment]
    app.checkpoint_var = _StringVarStub()  # type: ignore[assignment]
    app.dataset_var = _StringVarStub(str(completed.dataset_path))  # type: ignore[assignment]
    app.status_var = _StringVarStub()  # type: ignore[assignment]
    app.session_var = _StringVarStub()  # type: ignore[assignment]
    app.render_board = lambda: None  # type: ignore[method-assign]
    app._render_history = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chess_ai.gui.app.create_session_training_plan",
        lambda **_kwargs: next_plan,
    )

    app._complete_training_cycle(result)

    assert Path(app.checkpoint_var.get()) == best.resolve()
    assert app._training_session == next_plan
    assert app._session_confirmed_games == 0
    assert app._session_confirmed_examples == 0
    assert app._session_source_checkpoint is None
    assert "Training complete" in app.status_var.get()
