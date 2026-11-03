"""Offline move-review boundaries and python-chess tactical classification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import chess
import pytest

from chess_ai.arena import blunder_diagnostics as review
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.storage.artifacts import sha256_file


def test_mate_tags_use_legal_moves():
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4"]:
        board.push_uci(uci)
    mate = chess.Move.from_uci("d8h4")
    assert "missed_mate_in_one" in review.move_tags(board, chess.Move.from_uci("b8c6"), mate)
    assert review.move_tags(board, mate, mate) == []
    board.pop()
    assert "allowed_mate_in_one" in review.move_tags(
        board, chess.Move.from_uci("g2g4"), chess.Move.from_uci("g2g3")
    )


def test_review_preserves_archived_games_and_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "evaluation.pgn"
    text = (
        '[Result "0-1"]\n[TrainingData "false"]\n[NeuralColor "white"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'
    )
    path.write_text(text)

    def score(teacher, board, move):
        chosen = move or sorted(board.legal_moves, key=lambda m: m.uci())[0]
        return {
            "move_uci": chosen.uci(),
            "score_cp": 100 if move is None else -200,
            "pv": [chosen.uci()],
        }

    monkeypatch.setattr(review, "_score", score)
    report = review.review_game(path, None, positions=1, seed=83)
    assert report["reviewed_positions"] == 1 and report["available_neural_positions"] == 2
    assert report["moves"][0]["large_score_loss"]
    assert report == review.review_game(path, None, positions=1, seed=83)
    assert path.read_text() == text
    path.write_text(text.replace('"false"', '"true"'))
    with pytest.raises(ValueError, match="evaluation-only"):
        review.review_game(path, None, positions=1, seed=83)


def test_selection_review_validates_provenance_and_never_overwrites(
    tmp_path, teacher_setup, monkeypatch
):
    teacher_config, _ = teacher_setup
    models = {}
    for role in ("champion", "candidate"):
        checkpoint = tmp_path / f"{role}.pt"
        checkpoint.write_bytes(role.encode())
        pgn = tmp_path / f"{role}.pgn"
        pgn.write_text(
            f'[Result "0-1"]\n[TrainingData "false"]\n[NeuralColor "white"]\n'
            f'[Checkpoint "{checkpoint.as_posix()}"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'
        )
        models[role] = {
            "checkpoint": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "games": {
                "d1": [
                    {
                        "opening_index": 0,
                        "neural_color": "white",
                        "pgn_path": str(pgn),
                        "result": "0-1",
                    }
                ]
            },
        }
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "format": "self-improving-chess-ai.gameplay-epoch-selection",
                "version": 1,
                "champion": models["champion"],
                "candidates": [models["candidate"]],
            }
        )
    )

    def score(teacher, board, move):
        chosen = move or sorted(board.legal_moves, key=lambda m: m.uci())[0]
        return {"move_uci": chosen.uci(), "score_cp": 0, "pv": [chosen.uci()]}

    monkeypatch.setattr(review, "_score", score)
    from dataclasses import replace

    config = replace(teacher_config, multipv=1)
    output = tmp_path / "review.json"
    result = review.diagnose_selection_games(
        selection, Path(models["candidate"]["checkpoint"]), config, output, positions=1
    )
    assert result["training_data"] is False and result["training_updates"] == 0
    assert result["models"]["candidate"]["reviewed_positions"] == 1
    with pytest.raises(ValueError, match="already exists"):
        review.diagnose_selection_games(
            selection, Path(models["candidate"]["checkpoint"]), config, output
        )
    Path(models["champion"]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        review.diagnose_selection_games(
            selection, Path(models["candidate"]["checkpoint"]), config, tmp_path / "new.json"
        )
    assert not (tmp_path / "new.json").exists()


def test_optional_real_stockfish_forced_root():
    executable = os.environ.get("CHESS_AI_STOCKFISH")
    commit = os.environ.get("CHESS_AI_STOCKFISH_COMMIT")
    if not executable or not commit:
        pytest.skip("Set local Stockfish environment variables for offline UCI checks")
    with StockfishTeacher(
        StockfishConfig(Path(executable), commit, nodes=1000, multipv=1)
    ) as teacher:
        actual = chess.Move.from_uci("e2e4")
        trace = review._score(teacher, chess.Board(), actual)
        assert trace["move_uci"] == actual.uci()
        assert trace["nodes"] >= 1
