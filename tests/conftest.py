"""Shared local fake-engine fixture; no Stockfish installation required."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import chess
import chess.engine
import pytest

from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher


@pytest.fixture
def teacher_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binary = tmp_path / "stockfish.exe"
    binary.write_bytes(b"fake engine version 1")
    config = StockfishConfig(binary, "a" * 40, multipv=3, cache_dir=tmp_path / "cache")
    engine = Mock()
    engine.id = {"name": "Stockfish Mock", "author": "test"}
    engine.options = {
        name: SimpleNamespace(default=value)
        for name, value in (
            ("Threads", 1),
            ("Hash", 16),
            ("MultiPV", 1),
            ("Clear Hash", None),
            ("Skill Level", 20),
            ("UCI_LimitStrength", False),
            ("SyzygyPath", ""),
        )
    }
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: engine)

    def analyse(self, board, *, root_moves=None):
        moves = (
            list(root_moves)
            if root_moves is not None
            else sorted(board.legal_moves, key=lambda move: move.uci())[: self.config.multipv]
        )
        return [
            {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(100 - i * 100), board.turn),
                "depth": 4,
                "nodes": 1000,
            }
            for i, move in enumerate(moves)
        ]

    monkeypatch.setattr(StockfishTeacher, "_analyse", analyse)
    return config, engine
