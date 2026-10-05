"""Opening-book wrapper that falls back to any ordinary chess agent."""

from __future__ import annotations

import random
from pathlib import Path

import chess

from chess_ai.agents.protocol import ChessAgent
from chess_ai.data.opening_book import load_opening_book, position_key


class OpeningBookAgent:
    """Choose a weighted seeded continuation, then delegate after book exit."""

    def __init__(
        self,
        agent: ChessAgent,
        book_path: str | Path,
        *,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self.agent = agent
        self.book_path = Path(book_path)
        self.book = load_opening_book(self.book_path)
        self._rng = random.Random(seed)
        self._book_active = True
        self.name = name or f"{agent.name}+book:{self.book_path.name}"

    @property
    def book_active(self) -> bool:
        """Whether this game is still following a compatible opening line."""

        return self._book_active

    def choose_move(self, board: chess.Board) -> chess.Move:
        if self._book_active:
            entries = self.book.get(position_key(board), ())
            legal_entries = [
                (move, weight) for move, weight in entries if move in board.legal_moves
            ]
            if legal_entries:
                moves, weights = zip(*legal_entries, strict=True)
                return self._rng.choices(moves, weights=weights, k=1)[0]
            # Leaving theory is permanent for this game. Re-entering a matching
            # position later must not make the agent switch back out of search.
            self._book_active = False
        return self.agent.choose_move(board)
