"""Tests for the stable 4,208-action move vocabulary."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_ai.environment.move_encoder import BASE_ACTIONS, MoveEncoder


def test_action_space_has_expected_size() -> None:
    assert MoveEncoder.ACTION_SIZE == 4208
    assert MoveEncoder().action_size == 4208
    assert BASE_ACTIONS == 4032


def test_all_source_destination_pairs_round_trip() -> None:
    encoder = MoveEncoder()
    indexes: set[int] = set()

    for source in chess.SQUARES:
        for destination in chess.SQUARES:
            if source == destination:
                continue
            move = chess.Move(source, destination)
            index = encoder.encode(move)
            indexes.add(index)
            assert encoder.decode(index) == move

    assert indexes == set(range(4032))


@pytest.mark.parametrize(
    "uci",
    [
        "e2e4",  # Ordinary move.
        "e1g1",  # Castling.
        "e5d6",  # En passant geometry.
        "a7a8q",
        "a7b8r",
        "h7g8b",
        "h7h8n",
        "a2a1q",
        "a2b1r",
        "h2g1b",
        "h2h1n",
    ],
)
def test_representative_moves_round_trip(uci: str) -> None:
    encoder = MoveEncoder()
    move = chess.Move.from_uci(uci)

    assert encoder.decode(encoder.encode(move)) == move


def test_all_promotion_actions_are_unique_and_round_trip() -> None:
    encoder = MoveEncoder()
    promotions = [encoder.decode(index) for index in range(4032, encoder.action_size)]

    assert len(promotions) == 176
    assert len(set(promotions)) == 176
    assert {move.promotion for move in promotions} == {
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    }
    for index, move in enumerate(promotions, start=4032):
        assert encoder.encode(move) == index


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # Castling.
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # En passant.
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",  # Promotions available.
    ],
)
def test_legal_action_mask_contains_exactly_the_legal_moves(fen: str) -> None:
    encoder = MoveEncoder()
    board = chess.Board(fen)
    legal_moves = set(board.legal_moves)

    mask = encoder.legal_action_mask(board)
    decoded_moves = {encoder.decode(int(index)) for index in np.flatnonzero(mask)}

    assert mask.shape == (4208,)
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert int(mask.sum()) == len(legal_moves)
    assert decoded_moves == legal_moves


def test_promotion_mask_distinguishes_all_four_choices() -> None:
    encoder = MoveEncoder()
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    mask = encoder.legal_action_mask(board)

    promotion_moves = {
        encoder.decode(int(index))
        for index in np.flatnonzero(mask)
        if encoder.decode(int(index)).promotion is not None
    }

    assert promotion_moves == {
        chess.Move.from_uci("a7a8q"),
        chess.Move.from_uci("a7a8r"),
        chess.Move.from_uci("a7a8b"),
        chess.Move.from_uci("a7a8n"),
    }


@pytest.mark.parametrize("index", [-1, 4208, 9999])
def test_invalid_indexes_raise_clear_errors(index: int) -> None:
    with pytest.raises(IndexError, match="between 0 and 4207"):
        MoveEncoder().decode(index)


def test_non_integer_index_and_unrepresentable_moves_are_rejected() -> None:
    encoder = MoveEncoder()

    with pytest.raises(TypeError, match="integer"):
        encoder.decode(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different"):
        encoder.encode(chess.Move.null())
    with pytest.raises(ValueError, match="promotion geometry"):
        encoder.encode(chess.Move(chess.A6, chess.A7, promotion=chess.QUEEN))


def test_indexes_are_deterministic_across_encoder_instances() -> None:
    moves = [
        chess.Move.from_uci("b1c3"),
        chess.Move.from_uci("e1c1"),
        chess.Move.from_uci("g7h8n"),
    ]

    assert [MoveEncoder().encode(move) for move in moves] == [
        MoveEncoder().encode(move) for move in moves
    ]
