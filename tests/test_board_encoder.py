"""Exact tests for the 18-plane position representation."""

from __future__ import annotations

import chess
import numpy as np

from chess_ai.environment.board_encoder import BoardEncoder, encode_board


def test_initial_board_shape_dtype_and_binary_values() -> None:
    tensor = BoardEncoder().encode(chess.Board())

    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == np.float32
    assert set(np.unique(tensor)).issubset({0.0, 1.0})


def test_initial_piece_planes_use_white_diagram_orientation() -> None:
    tensor = BoardEncoder().encode(chess.Board())

    # Row 0 is rank 8; row 7 is rank 1. Column 0 is always file a.
    assert tensor[3, 7, 0] == 1.0  # White rook on a1.
    assert tensor[5, 7, 4] == 1.0  # White king on e1.
    assert tensor[9, 0, 7] == 1.0  # Black rook on h8.
    assert tensor[11, 0, 4] == 1.0  # Black king on e8.
    np.testing.assert_array_equal(tensor[0, 6], np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(tensor[6, 1], np.ones(8, dtype=np.float32))
    assert tensor[:12].sum() == 32.0


def test_initial_metadata_planes_are_exact() -> None:
    tensor = BoardEncoder().encode(chess.Board())

    np.testing.assert_array_equal(tensor[12], np.ones((8, 8), dtype=np.float32))
    for plane in range(13, 17):
        np.testing.assert_array_equal(tensor[plane], np.ones((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(tensor[17], np.zeros((8, 8), dtype=np.float32))


def test_black_turn_castling_and_en_passant_planes() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    tensor = encode_board(board)

    np.testing.assert_array_equal(tensor[12], np.zeros((8, 8), dtype=np.float32))
    assert tensor[17].sum() == 1.0
    assert tensor[17, 5, 4] == 1.0  # e3 is row 5, column 4.

    no_rights = BoardEncoder().encode(chess.Board("7k/8/8/8/8/8/8/K7 b - - 0 1"))
    for plane in range(13, 17):
        assert no_rights[plane].sum() == 0.0


def test_encoding_returns_an_independent_array_each_time() -> None:
    encoder = BoardEncoder()
    first = encoder.encode(chess.Board())
    second = encoder.encode(chess.Board())

    first.fill(0.0)

    assert second.sum() > 0.0
