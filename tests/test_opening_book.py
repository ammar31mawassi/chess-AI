from __future__ import annotations

import csv
from pathlib import Path

import chess

from chess_ai.agents.opening_book_agent import OpeningBookAgent
from chess_ai.data import (
    OpeningImportConfig,
    import_opening_book,
    load_dataset,
    load_opening_book,
    position_key,
)
from chess_ai.environment.move_encoder import MoveEncoder


def _source(path: Path) -> Path:
    fields = ["Opening", "Num Games", "ECO", "Moves", "White_win%", "Black_win%"]
    rows = [
        {
            "Opening": "King Pawn Main",
            "Num Games": "100",
            "ECO": "C20",
            "Moves": "1.e4 e5 2.Nf3",
            "White_win%": "45",
            "Black_win%": "35",
        },
        {
            "Opening": "Sicilian",
            "Num Games": "50",
            "ECO": "B20",
            "Moves": "1.e4 c5 2.Nf3",
            "White_win%": "40",
            "Black_win%": "40",
        },
        {
            "Opening": "Queen Pawn",
            "Num Games": "50",
            "ECO": "D00",
            "Moves": "1.d4 d5 2.c4",
            "White_win%": "42",
            "Black_win%": "32",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _compile(tmp_path: Path) -> OpeningImportConfig:
    config = OpeningImportConfig(
        source_path=_source(tmp_path / "openings.csv"),
        dataset_path=tmp_path / "openings.pt",
        book_path=tmp_path / "book.json",
    )
    import_opening_book(config)
    return config


def test_opening_import_aggregates_weighted_legal_continuations(tmp_path: Path) -> None:
    config = _compile(tmp_path)
    dataset = load_dataset(config.dataset_path)
    book = load_opening_book(config.book_path)
    root = chess.Board()
    root_example = next(
        item for item in dataset if item.metadata["position_key"] == position_key(root)
    )
    encoder = MoveEncoder()

    assert root_example.target_policy[encoder.encode(chess.Move.from_uci("e2e4"))] == 0.75
    assert root_example.target_policy[encoder.encode(chess.Move.from_uci("d2d4"))] == 0.25
    assert {move.uci() for move, _weight in book[position_key(root)]} == {"e2e4", "d2d4"}
    assert all(move in root.legal_moves for move, _weight in book[position_key(root)])


class _FallbackAgent:
    name = "fallback"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return sorted(board.legal_moves, key=lambda move: move.uci())[0]


def test_opening_agent_conditions_black_reply_and_falls_back(tmp_path: Path) -> None:
    config = _compile(tmp_path)
    agent = OpeningBookAgent(_FallbackAgent(), config.book_path, seed=7)
    board = chess.Board()
    board.push_uci("e2e4")

    assert agent.choose_move(board).uci() in {"e7e5", "c7c5"}

    outside_book = chess.Board()
    outside_book.push_uci("a2a3")
    expected = _FallbackAgent().choose_move(outside_book)
    assert agent.choose_move(outside_book) == expected
    assert agent.book_active is False

    # Book exit is one-way within the game, even if a later supplied position
    # happens to match the starting book again.
    assert agent.choose_move(chess.Board()) == _FallbackAgent().choose_move(chess.Board())
    assert agent.book_active is False


def test_opening_agent_is_seed_reproducible(tmp_path: Path) -> None:
    config = _compile(tmp_path)
    first = OpeningBookAgent(_FallbackAgent(), config.book_path, seed=19)
    second = OpeningBookAgent(_FallbackAgent(), config.book_path, seed=19)

    assert first.choose_move(chess.Board()) == second.choose_move(chess.Board())
