"""A small, readable wrapper around :mod:`python-chess` rule handling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import chess
import chess.pgn


class ChessGameError(ValueError):
    """Base class for user-facing chess game errors."""


class InvalidPositionError(ChessGameError):
    """Raised when a FEN string cannot describe a valid standard position."""


class MalformedMoveError(ChessGameError):
    """Raised when text is not valid UCI move syntax."""


class IllegalMoveError(ChessGameError):
    """Raised when a syntactically valid move is illegal in the position."""


class NoMoveToUndoError(ChessGameError):
    """Raised when undo is requested before any move has been played."""


class ChessGame:
    """Manage one standard chess game while delegating rules to python-chess.

    ``ChessGame`` deliberately adds convenience and readable errors, but it
    does not duplicate chess-rule logic.  The wrapped :class:`chess.Board` is
    available through :attr:`board` for agents and educational exploration.
    """

    def __init__(
        self,
        fen: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._headers = dict(headers or {})
        self._board = self._board_from_fen(fen) if fen is not None else chess.Board()

    @classmethod
    def from_fen(
        cls,
        fen: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> ChessGame:
        """Construct a game whose initial position is *fen*."""

        return cls(fen, headers=headers)

    @property
    def board(self) -> chess.Board:
        """The mutable python-chess board owned by this game.

        Callers that only need a position snapshot should use
        :meth:`board_copy`; agents receive copies from match runners so that
        they cannot accidentally alter the official game.
        """

        return self._board

    @property
    def headers(self) -> dict[str, str]:
        """Return a copy of the PGN headers associated with this game."""

        return self._headers.copy()

    @property
    def move_history(self) -> tuple[chess.Move, ...]:
        """Moves played since this game's initial position."""

        return tuple(self._board.move_stack)

    @property
    def move_history_uci(self) -> tuple[str, ...]:
        """Move history in portable UCI notation."""

        return tuple(move.uci() for move in self._board.move_stack)

    @property
    def ply_count(self) -> int:
        """Number of half-moves played since the initial position."""

        return len(self._board.move_stack)

    def new_game(self) -> None:
        """Reset to the standard initial position and clear saved headers."""

        self._board = chess.Board()
        self._headers.clear()

    reset = new_game

    def load_fen(self, fen: str) -> None:
        """Replace the current game with a validated FEN position."""

        self._board = self._board_from_fen(fen)

    set_fen = load_fen

    def fen(
        self,
        *,
        en_passant: Literal["legal", "fen", "xfen"] = "fen",
    ) -> str:
        """Export the current position as FEN.

        By default the literal FEN en-passant target is retained after every
        two-square pawn move, even when no opposing pawn can capture it.  Pass
        ``en_passant="legal"`` to use python-chess's stricter default style.
        """

        return self._board.fen(en_passant=en_passant)

    def board_copy(self, *, stack: bool = True) -> chess.Board:
        """Return an independent copy of the current board."""

        return self._board.copy(stack=stack)

    def legal_moves(self) -> list[chess.Move]:
        """Return a concrete snapshot of the legal moves."""

        return list(self._board.legal_moves)

    def legal_moves_uci(self) -> list[str]:
        """Return all legal moves in UCI notation."""

        return [move.uci() for move in self._board.legal_moves]

    def apply_uci(self, move_text: str) -> chess.Move:
        """Parse, validate, and play one UCI move.

        Malformed input and well-formed-but-illegal input use different error
        types so a terminal interface can give an accurate retry message.
        """

        if not isinstance(move_text, str):
            raise TypeError("UCI move must be text, for example 'e2e4'")

        stripped = move_text.strip()
        try:
            move = chess.Move.from_uci(stripped)
        except ValueError as error:
            raise MalformedMoveError(
                f"Invalid UCI move {move_text!r}. Use notation such as 'e2e4' or 'a7a8q'."
            ) from error
        return self.apply_move(move)

    def apply_move(self, move: chess.Move | str) -> chess.Move:
        """Validate and play a :class:`chess.Move` (or UCI string)."""

        if isinstance(move, str):
            return self.apply_uci(move)
        if not isinstance(move, chess.Move):
            raise TypeError("move must be a python-chess Move or UCI string")
        if move not in self._board.legal_moves:
            turn = "White" if self._board.turn == chess.WHITE else "Black"
            raise IllegalMoveError(
                f"Move {move.uci()!r} is not legal for {turn} in the current position."
            )

        self._board.push(move)
        return move

    def undo(self) -> chess.Move:
        """Undo and return the most recent move."""

        if not self._board.move_stack:
            raise NoMoveToUndoError("Cannot undo because no moves have been played.")
        return self._board.pop()

    pop = undo

    def is_check(self) -> bool:
        return self._board.is_check()

    def is_checkmate(self) -> bool:
        return self._board.is_checkmate()

    def is_stalemate(self) -> bool:
        return self._board.is_stalemate()

    def is_insufficient_material(self) -> bool:
        return self._board.is_insufficient_material()

    def is_repetition(self, count: int = 3) -> bool:
        """Whether the current position has occurred at least *count* times."""

        if count < 1:
            raise ValueError("repetition count must be at least 1")
        return self._board.is_repetition(count=count)

    def can_claim_threefold_repetition(self) -> bool:
        return self._board.can_claim_threefold_repetition()

    def is_fivefold_repetition(self) -> bool:
        return self._board.is_fivefold_repetition()

    def is_repetition_draw(self) -> bool:
        """Whether a repetition draw is automatic or currently claimable."""

        return self.is_fivefold_repetition() or self.can_claim_threefold_repetition()

    def can_claim_fifty_moves(self) -> bool:
        return self._board.can_claim_fifty_moves()

    def is_fifty_moves(self) -> bool:
        return self._board.is_fifty_moves()

    def is_seventyfive_moves(self) -> bool:
        return self._board.is_seventyfive_moves()

    def is_move_count_draw(self) -> bool:
        """Whether a move-count draw is automatic or currently claimable."""

        return self.is_seventyfive_moves() or self.can_claim_fifty_moves()

    def outcome(self, *, claim_draw: bool = True) -> chess.Outcome | None:
        """Return python-chess's official outcome, or ``None`` if play continues.

        Claimable threefold and fifty-move draws are included by default,
        which is convenient for autonomous games.  Set ``claim_draw=False``
        to include only automatic endings.
        """

        return self._board.outcome(claim_draw=claim_draw)

    get_outcome = outcome

    def is_game_over(self, *, claim_draw: bool = True) -> bool:
        return self._board.is_game_over(claim_draw=claim_draw)

    def is_draw(self, *, claim_draw: bool = True) -> bool:
        outcome = self.outcome(claim_draw=claim_draw)
        return outcome is not None and outcome.winner is None

    def result(self, *, claim_draw: bool = True) -> str:
        """Return ``'1-0'``, ``'0-1'``, ``'1/2-1/2'``, or ``'*'``."""

        return self._board.result(claim_draw=claim_draw)

    def pgn_game(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        claim_draw: bool = True,
    ) -> chess.pgn.Game:
        """Build a replayable PGN game from the initial position and history."""

        game = chess.pgn.Game.from_board(self._board)
        for key, value in self._headers.items():
            game.headers[key] = value
        if headers is not None:
            for key, value in headers.items():
                game.headers[key] = value
        game.headers["Result"] = self.result(claim_draw=claim_draw)
        return game

    def export_pgn(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        claim_draw: bool = True,
    ) -> str:
        """Export the position history as a PGN string."""

        game = self.pgn_game(headers=headers, claim_draw=claim_draw)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)

    to_pgn = export_pgn

    @staticmethod
    def _board_from_fen(fen: str) -> chess.Board:
        if not isinstance(fen, str):
            raise TypeError("FEN must be text")
        try:
            board = chess.Board(fen)
        except ValueError as error:
            raise InvalidPositionError(f"Invalid FEN: {error}") from error

        status = board.status()
        if status != chess.STATUS_VALID:
            status_name = status.name if status.name is not None else str(status)
            description = status_name.lower().replace("_", " ")
            raise InvalidPositionError(f"FEN does not describe a valid position: {description}")
        return board
