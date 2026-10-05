"""A deterministic 4,208-action vocabulary for standard chess moves."""

from __future__ import annotations

from typing import Final

import chess
import numpy as np
import numpy.typing as npt

ActionMask = npt.NDArray[np.float32]

BASE_ACTIONS: Final = 64 * 63
PROMOTION_PIECES: Final[tuple[chess.PieceType, ...]] = (
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)


def _promotion_vocabulary() -> tuple[chess.Move, ...]:
    """Build all 44 promotion geometries, once for each promotion piece.

    The order is deliberately explicit and independent of sets or hashing:
    White then Black, source files a through h, destination files from left
    to right, and Queen/Rook/Bishop/Knight.
    """

    moves: list[chess.Move] = []
    for color in (chess.WHITE, chess.BLACK):
        source_rank = 6 if color == chess.WHITE else 1
        destination_rank = 7 if color == chess.WHITE else 0

        for source_file in range(8):
            first_destination = max(0, source_file - 1)
            last_destination = min(7, source_file + 1)
            for destination_file in range(first_destination, last_destination + 1):
                source = chess.square(source_file, source_rank)
                destination = chess.square(destination_file, destination_rank)
                for piece_type in PROMOTION_PIECES:
                    moves.append(chess.Move(source, destination, promotion=piece_type))

    return tuple(moves)


_PROMOTION_MOVES: Final = _promotion_vocabulary()
_PROMOTION_TO_OFFSET: Final = {move: offset for offset, move in enumerate(_PROMOTION_MOVES)}


class MoveEncoder:
    """Map :class:`chess.Move` objects to stable neural-network action indexes.

    Indexes 0 through 4031 cover every distinct source/destination pair.  The
    remaining 176 indexes distinguish all legal promotion geometries and four
    promotion piece choices for both colors.
    """

    ACTION_SIZE = BASE_ACTIONS + len(_PROMOTION_MOVES)
    action_size = ACTION_SIZE

    if ACTION_SIZE != 4208:  # A development-time guard against vocabulary drift.
        raise RuntimeError(f"move vocabulary must contain 4208 actions, got {ACTION_SIZE}")

    def encode(self, move: chess.Move) -> int:
        """Return the deterministic action index for *move*.

        This method encodes move syntax, not legality in a particular position.
        Use :meth:`legal_action_mask` when board-specific legality is needed.
        """

        if not isinstance(move, chess.Move):
            raise TypeError("move must be a python-chess Move")
        if move.drop is not None:
            raise ValueError("drop moves are not part of the standard-chess action space")

        if move.promotion is not None:
            offset = _PROMOTION_TO_OFFSET.get(move)
            if offset is None:
                raise ValueError(f"{move.uci()} is not a valid standard-chess promotion geometry")
            return BASE_ACTIONS + offset

        source = move.from_square
        destination = move.to_square
        if source not in chess.SQUARES or destination not in chess.SQUARES:
            raise ValueError("move squares must both be between 0 and 63")
        if source == destination:
            raise ValueError("source and destination squares must be different")

        # Each source owns 63 consecutive entries.  Skip its identical target.
        target_offset = destination if destination < source else destination - 1
        return source * 63 + target_offset

    def decode(self, index: int) -> chess.Move:
        """Return the move represented by *index*.

        Raises:
            TypeError: If the index is not an integer.
            IndexError: If the index is outside ``[0, action_size)``.
        """

        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("action index must be an integer")
        integer_index = int(index)
        if not 0 <= integer_index < self.action_size:
            raise IndexError(
                f"action index must be between 0 and {self.action_size - 1}, got {integer_index}"
            )

        if integer_index >= BASE_ACTIONS:
            return _PROMOTION_MOVES[integer_index - BASE_ACTIONS]

        source, target_offset = divmod(integer_index, 63)
        destination = target_offset if target_offset < source else target_offset + 1
        return chess.Move(source, destination)

    def legal_action_mask(self, board: chess.Board) -> ActionMask:
        """Return a binary mask containing exactly *board*'s legal moves."""

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")

        mask = np.zeros(self.action_size, dtype=np.float32)
        for move in board.legal_moves:
            mask[self.encode(move)] = 1.0
        return mask
