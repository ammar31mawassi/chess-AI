"""The small interface shared by every chess-playing agent."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chess


class NoLegalMovesError(ValueError):
    """Raised when an agent is asked to move in a position with no legal move."""


@runtime_checkable
class ChessAgent(Protocol):
    """Anything with a name and ``choose_move`` can participate in a match."""

    name: str

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose one legal move without modifying *board*."""
        ...
