"""A uniformly random baseline agent."""

from __future__ import annotations

import random

import chess

from chess_ai.agents.protocol import NoLegalMovesError


class RandomAgent:
    """Choose uniformly from all legal moves using a private random generator."""

    def __init__(self, seed: int | None = None, name: str = "Random") -> None:
        self.name = name
        self.seed = seed
        self._random = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return a uniformly selected legal move without changing *board*."""

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise NoLegalMovesError("RandomAgent cannot move because the game is over.")
        return self._random.choice(legal_moves)

    def reseed(self, seed: int | None) -> None:
        """Restart this agent's pseudo-random sequence."""

        self.seed = seed
        self._random.seed(seed)
