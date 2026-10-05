"""Convert a :mod:`python-chess` board into neural-network input planes.

The array is viewed from White's side: row ``0`` is rank 8, row ``7`` is
rank 1, column ``0`` is file ``a``, and column ``7`` is file ``h``.  This is
the same orientation used by most printed chess diagrams.
"""

from __future__ import annotations

from typing import ClassVar

import chess
import numpy as np
import numpy.typing as npt

BoardTensor = npt.NDArray[np.float32]


class BoardEncoder:
    """Encode board state as 18 binary planes with shape ``(18, 8, 8)``.

    Piece planes contain a one only at the piece's square.  The side-to-move
    and castling planes describe global facts, so each is filled with either
    zeros or ones.  The en-passant plane contains at most one one, located at
    the en-passant target square.
    """

    NUM_PLANES = 18
    shape = (NUM_PLANES, 8, 8)

    _PIECE_PLANE: ClassVar[dict[tuple[chess.Color, chess.PieceType], int]] = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 1,
        (chess.WHITE, chess.BISHOP): 2,
        (chess.WHITE, chess.ROOK): 3,
        (chess.WHITE, chess.QUEEN): 4,
        (chess.WHITE, chess.KING): 5,
        (chess.BLACK, chess.PAWN): 6,
        (chess.BLACK, chess.KNIGHT): 7,
        (chess.BLACK, chess.BISHOP): 8,
        (chess.BLACK, chess.ROOK): 9,
        (chess.BLACK, chess.QUEEN): 10,
        (chess.BLACK, chess.KING): 11,
    }

    def encode(self, board: chess.Board) -> BoardTensor:
        """Return a new ``float32`` array containing the board state.

        Args:
            board: Position to encode.  It is read but never modified.

        Raises:
            TypeError: If *board* is not a :class:`chess.Board`.
        """

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")

        planes = np.zeros(self.shape, dtype=np.float32)

        for square, piece in board.piece_map().items():
            plane = self._PIECE_PLANE[(piece.color, piece.piece_type)]
            row, column = self.square_to_coordinates(square)
            planes[plane, row, column] = 1.0

        # A plane of ones means White is to move; zeros means Black.
        planes[12].fill(float(board.turn == chess.WHITE))
        planes[13].fill(float(board.has_kingside_castling_rights(chess.WHITE)))
        planes[14].fill(float(board.has_queenside_castling_rights(chess.WHITE)))
        planes[15].fill(float(board.has_kingside_castling_rights(chess.BLACK)))
        planes[16].fill(float(board.has_queenside_castling_rights(chess.BLACK)))

        if board.ep_square is not None:
            row, column = self.square_to_coordinates(board.ep_square)
            planes[17, row, column] = 1.0

        return planes

    @staticmethod
    def square_to_coordinates(square: chess.Square) -> tuple[int, int]:
        """Map a python-chess square to ``(row, column)`` tensor coordinates."""

        if square not in chess.SQUARES:
            raise ValueError(f"square must be between 0 and 63, got {square}")
        row = 7 - chess.square_rank(square)
        column = chess.square_file(square)
        return row, column


def encode_board(board: chess.Board) -> BoardTensor:
    """Convenience function equivalent to ``BoardEncoder().encode(board)``."""

    return BoardEncoder().encode(board)
