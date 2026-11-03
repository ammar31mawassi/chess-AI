"""Transient Windows locks must not lose the last committed artifact."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
import torch

from chess_ai.data.examples import load_dataset, save_dataset
from chess_ai.model import PolicyValueNet, load_checkpoint, save_checkpoint
from chess_ai.storage import artifacts
from chess_ai.storage.games import parse_pgn_text, save_pgn


def windows_error(code: int) -> OSError:
    error = PermissionError(13, "Simulated Windows sharing/access lock")
    error.winerror = code
    return error


@pytest.mark.parametrize("kind", ["json", "pgn", "dataset", "checkpoint"])
def test_atomic_writers_retry_same_temporary_without_touching_committed_file(
    tmp_path, monkeypatch, kind
):
    path = (
        tmp_path
        / {"json": "state.json", "pgn": "game.pgn", "dataset": "data.pt", "checkpoint": "model.pt"}[
            kind
        ]
    )
    old = b"last committed artifact"
    path.write_bytes(old)
    original = artifacts.os.replace
    attempts = []
    sleeps = []

    def temporarily_locked(source, destination):
        attempts.append(Path(source))
        assert Path(destination) == path
        assert path.read_bytes() == old
        assert Path(source).is_file()
        if len(attempts) <= 3:
            raise windows_error((5, 32, 33)[len(attempts) - 1])
        original(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", temporarily_locked)
    monkeypatch.setattr(artifacts.time, "sleep", sleeps.append)
    if kind == "json":
        artifacts.atomic_json(path, {"games": 14})
        assert json.loads(path.read_text()) == {"games": 14}
    elif kind == "pgn":
        save_pgn(chess.pgn.Game(), path)
        assert len(parse_pgn_text(path.read_text())) == 1
    elif kind == "dataset":
        save_dataset(path, [])
        assert len(load_dataset(path)) == 0
    else:
        torch.manual_seed(59)
        save_checkpoint(path, PolicyValueNet(channels=2, residual_blocks=0), epoch=60)
        assert load_checkpoint(path).epoch == 60
    assert len(attempts) == 4 and len(set(attempts)) == 1
    assert sleeps == [0.05, 0.1, 0.2]
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_windows_lock_stops_with_old_state_intact(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    old = '{"games": 13}\n'
    path.write_text(old)
    attempts, sleeps = [], []

    def locked(source, destination):
        attempts.append(Path(source))
        raise windows_error(5)

    monkeypatch.setattr(artifacts.os, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        artifacts.atomic_json(path, {"games": 14})
    assert path.read_text() == old
    assert len(attempts) == 8 and len(sleeps) == 7 and sum(sleeps) == pytest.approx(3.15)
    assert not list(tmp_path.glob("*.tmp"))


def test_other_io_failures_are_not_retried(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"games": 13}')
    attempts = []

    def disk_full(source, destination):
        attempts.append(source)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(artifacts.os, "replace", disk_full)
    monkeypatch.setattr(artifacts.time, "sleep", lambda _: pytest.fail("Not a sharing lock"))
    with pytest.raises(OSError, match="No space"):
        artifacts.atomic_json(path, {"games": 14})
    assert len(attempts) == 1 and json.loads(path.read_text())["games"] == 13
