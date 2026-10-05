"""A terminal-driven human chess agent."""

from __future__ import annotations

from collections.abc import Callable

import chess

from chess_ai.agents.protocol import NoLegalMovesError

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class HumanInputAborted(RuntimeError):
    """Raised when terminal input ends before the human supplies a move."""


class HumanAgent:
    """Prompt for UCI moves until the user enters a legal one.

    Custom input and output functions make the behavior easy to test without a
    real terminal.
    """

    def __init__(
        self,
        input_fn: InputFunction | None = None,
        output_fn: OutputFunction | None = None,
        name: str = "Human",
    ) -> None:
        self.name = name
        self._input = input_fn if input_fn is not None else input
        self._output = output_fn if output_fn is not None else print

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Read and validate a UCI move, printing useful retry messages."""

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")
        if not any(board.legal_moves):
            raise NoLegalMovesError("HumanAgent cannot move because the game is over.")

        while True:
            try:
                response = self._input("Enter a move in UCI notation (for example e2e4): ")
            except (EOFError, KeyboardInterrupt) as error:
                raise HumanInputAborted("Human move input was cancelled.") from error

            move_text = response.strip().lower()
            try:
                move = chess.Move.from_uci(move_text)
            except ValueError:
                self._output(
                    "That is not valid UCI notation. Try a move like e2e4 (or a7a8q for promotion)."
                )
                continue

            if move not in board.legal_moves:
                self._output(f"{move_text!r} is not legal in this position. Please try again.")
                continue

            return move
