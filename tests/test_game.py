"""Tests for the python-chess-backed game wrapper."""

from __future__ import annotations

import io

import chess
import chess.pgn
import pytest

from chess_ai.environment.game import (
    ChessGame,
    IllegalMoveError,
    InvalidPositionError,
    MalformedMoveError,
    NoMoveToUndoError,
)


def test_new_game_has_twenty_legal_moves_and_standard_fen() -> None:
    game = ChessGame()

    assert len(game.legal_moves()) == 20
    assert game.fen() == chess.STARTING_FEN
    assert game.move_history == ()


def test_apply_move_tracks_history_and_undo_restores_position() -> None:
    game = ChessGame()
    initial_fen = game.fen()

    played = game.apply_uci(" e2e4 ")

    assert played == chess.Move.from_uci("e2e4")
    assert game.move_history_uci == ("e2e4",)
    assert game.ply_count == 1
    assert game.fen().split()[3] == "e3"
    assert game.undo() == played
    assert game.fen() == initial_fen


def test_malformed_and_illegal_moves_have_distinct_readable_errors() -> None:
    game = ChessGame()

    with pytest.raises(MalformedMoveError, match="UCI"):
        game.apply_uci("knight to f3")
    with pytest.raises(IllegalMoveError, match="not legal"):
        game.apply_uci("e2e5")

    assert game.move_history == ()


def test_undo_without_history_is_rejected() -> None:
    with pytest.raises(NoMoveToUndoError, match="no moves"):
        ChessGame().undo()


def test_invalid_fen_is_rejected() -> None:
    with pytest.raises(InvalidPositionError, match="Invalid FEN"):
        ChessGame("not a fen")
    with pytest.raises(InvalidPositionError, match="valid position"):
        ChessGame("8/8/8/8/8/8/8/8 w - - 0 1")


def test_fools_mate_is_recognized_with_official_outcome() -> None:
    game = ChessGame()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        game.apply_uci(move)

    outcome = game.outcome()
    assert game.is_check()
    assert game.is_checkmate()
    assert game.is_game_over()
    assert outcome is not None
    assert outcome.winner == chess.BLACK
    assert outcome.termination == chess.Termination.CHECKMATE
    assert game.result() == "0-1"


def test_stalemate_and_insufficient_material_are_recognized() -> None:
    stalemate = ChessGame("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    kings_only = ChessGame("7k/8/8/8/8/8/8/K7 w - - 0 1")

    assert stalemate.is_stalemate()
    assert stalemate.is_draw()
    assert stalemate.outcome() is not None
    assert stalemate.outcome().termination == chess.Termination.STALEMATE
    assert kings_only.is_insufficient_material()


def test_castling_moves_both_king_and_rook() -> None:
    game = ChessGame("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    assert "e1g1" in game.legal_moves_uci()
    game.apply_uci("e1g1")

    assert game.board.king(chess.WHITE) == chess.G1
    assert game.board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert not game.board.has_castling_rights(chess.WHITE)


def test_en_passant_capture_removes_the_bypassed_pawn() -> None:
    game = ChessGame("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")

    assert "e5d6" in game.legal_moves_uci()
    game.apply_uci("e5d6")

    assert game.board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert game.board.piece_at(chess.D5) is None


@pytest.mark.parametrize(
    ("suffix", "piece_type"),
    [
        ("q", chess.QUEEN),
        ("r", chess.ROOK),
        ("b", chess.BISHOP),
        ("n", chess.KNIGHT),
    ],
)
def test_all_four_promotion_pieces_are_supported(suffix: str, piece_type: int) -> None:
    game = ChessGame("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    game.apply_uci(f"a7a8{suffix}")

    assert game.board.piece_at(chess.A8) == chess.Piece(piece_type, chess.WHITE)


def test_threefold_repetition_can_be_claimed_and_reported() -> None:
    game = ChessGame()
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for _ in range(2):
        for move in cycle:
            game.apply_uci(move)

    outcome = game.outcome(claim_draw=True)
    assert game.is_repetition(count=3)
    assert game.can_claim_threefold_repetition()
    assert game.is_repetition_draw()
    assert outcome is not None
    assert outcome.termination == chess.Termination.THREEFOLD_REPETITION
    # A threefold draw needs a claim; it is not automatic yet.
    assert game.outcome(claim_draw=False) is None


def test_fivefold_repetition_is_an_automatic_draw() -> None:
    game = ChessGame()
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for _ in range(4):
        for move in cycle:
            game.apply_uci(move)

    outcome = game.outcome(claim_draw=False)
    assert game.is_fivefold_repetition()
    assert outcome is not None
    assert outcome.termination == chess.Termination.FIVEFOLD_REPETITION


def test_fifty_and_seventyfive_move_draw_detection() -> None:
    fifty = ChessGame("7k/8/8/8/8/8/R7/K7 w - - 100 51")
    seventy_five = ChessGame("7k/8/8/8/8/8/R7/K7 w - - 150 76")

    assert fifty.is_fifty_moves()
    assert fifty.can_claim_fifty_moves()
    assert fifty.is_move_count_draw()
    assert fifty.outcome(claim_draw=False) is None
    assert seventy_five.is_seventyfive_moves()
    assert seventy_five.outcome(claim_draw=False) is not None
    assert seventy_five.outcome(claim_draw=False).termination == chess.Termination.SEVENTYFIVE_MOVES


def test_exported_pgn_can_be_loaded_and_reaches_the_same_position() -> None:
    game = ChessGame(headers={"Event": "Round-trip test", "White": "Student"})
    for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
        game.apply_uci(move)

    pgn_text = game.export_pgn(headers={"Black": "Computer"})
    loaded = chess.pgn.read_game(io.StringIO(pgn_text))

    assert loaded is not None
    assert loaded.errors == []
    assert loaded.headers["Event"] == "Round-trip test"
    assert loaded.headers["White"] == "Student"
    assert loaded.headers["Black"] == "Computer"
    assert loaded.end().board().fen(en_passant="fen") == game.fen()


def test_custom_fen_pgn_contains_setup_and_replays_from_that_position() -> None:
    game = ChessGame("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    game.apply_uci("a7a8q")

    loaded = chess.pgn.read_game(io.StringIO(game.to_pgn()))

    assert loaded is not None
    assert loaded.headers["SetUp"] == "1"
    assert loaded.end().board().piece_at(chess.A8) == chess.Piece(chess.QUEEN, chess.WHITE)
