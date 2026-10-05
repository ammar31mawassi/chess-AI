"""UI-independent controller for a human playing the neural chess agent.

Tkinter is deliberately absent from this module.  It owns one standard game,
validates every move through :mod:`python-chess`, and exposes small methods a
desktop interface can call from click handlers or an AI worker thread.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import chess
import chess.pgn
import numpy as np
from numpy.typing import NDArray

from chess_ai.agents.protocol import ChessAgent
from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import TrainingExample
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.game import ChessGame, IllegalMoveError, NoMoveToUndoError
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.gui.training_data import (
    DEFAULT_HUMAN_GUI_DATASET_PATH,
    ExplicitTrainingOptInRequired,
    HumanGuiAppendResult,
    append_human_gui_game,
)
from chess_ai.storage.games import save_pgn

SquareInput = chess.Square | str
PromotionInput = chess.PieceType | str | None
Clock = Callable[[], datetime]

_PROMOTION_NAMES: dict[str, chess.PieceType] = {
    "q": chess.QUEEN,
    "queen": chess.QUEEN,
    "r": chess.ROOK,
    "rook": chess.ROOK,
    "b": chess.BISHOP,
    "bishop": chess.BISHOP,
    "n": chess.KNIGHT,
    "knight": chess.KNIGHT,
}
_PROMOTION_ORDER = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


class HumanNeuralGameError(RuntimeError):
    """Base class for controller state and neural-agent errors."""


class WrongTurnError(HumanNeuralGameError):
    """Raised when the human or neural agent is asked to move out of turn."""


class GameAlreadyOverError(HumanNeuralGameError):
    """Raised when an action requires a game that is still in progress."""


class GameNotCompleteError(HumanNeuralGameError):
    """Raised when completed-game artifacts are requested too early."""


class AgentMoveError(HumanNeuralGameError):
    """Raised when the neural agent returns something other than a legal move."""


class PromotionRequiredError(IllegalMoveError):
    """Raised when a pawn promotion needs an explicit piece choice."""


class TrainingDisabledError(HumanNeuralGameError):
    """Raised when append is requested for a game that did not collect examples."""


@dataclass(frozen=True, slots=True)
class PendingHumanExample:
    """A pre-human-move policy label awaiting the game's final result."""

    board_tensor: NDArray[np.float32]
    target_policy: NDArray[np.float32]
    turn: chess.Color
    metadata: Mapping[str, Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_color(value: chess.Color | str) -> chess.Color:
    if value is chess.WHITE or (isinstance(value, str) and value.strip().lower() == "white"):
        return chess.WHITE
    if value is chess.BLACK or (isinstance(value, str) and value.strip().lower() == "black"):
        return chess.BLACK
    raise ValueError("human_color must be 'white', 'black', chess.WHITE, or chess.BLACK")


def _parse_square(value: SquareInput, *, name: str) -> chess.Square:
    if isinstance(value, str):
        try:
            return chess.parse_square(value.strip().lower())
        except ValueError as exc:
            raise ValueError(f"{name} must be a square such as 'e2'") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value not in chess.SQUARES:
        raise ValueError(f"{name} must be a python-chess square or text such as 'e2'")
    return value


def _parse_promotion(value: PromotionInput) -> chess.PieceType | None:
    if value is None:
        return None
    if isinstance(value, str):
        piece_type = _PROMOTION_NAMES.get(value.strip().lower())
        if piece_type is None:
            raise ValueError("promotion must be queen, rook, bishop, or knight")
        return piece_type
    if isinstance(value, bool) or value not in _PROMOTION_ORDER:
        raise ValueError("promotion must be queen, rook, bishop, or knight")
    return value


class HumanNeuralGame:
    """Coordinate standard human-vs-neural games for a local desktop UI.

    ``training_enabled`` controls whether pre-human-move positions are kept in
    memory.  It never writes them.  After a completed result, callers must
    separately invoke :meth:`append_training_examples` with
    ``confirm_training=True`` to mutate the dedicated dataset.
    """

    def __init__(
        self,
        agent: ChessAgent,
        *,
        human_color: chess.Color | str,
        checkpoint_label: str,
        training_enabled: bool,
        session_seed: int = 0,
        session_id: str | None = None,
        pgn_dir: str | Path = Path("data/games/human_gui"),
        dataset_path: str | Path = DEFAULT_HUMAN_GUI_DATASET_PATH,
        clock: Clock = _utc_now,
    ) -> None:
        if not isinstance(agent, ChessAgent):
            raise TypeError("agent must provide a name and choose_move(board) method")
        if not str(agent.name).strip():
            raise ValueError("agent name cannot be empty")
        if not checkpoint_label.strip():
            raise ValueError("checkpoint_label cannot be empty")
        if not isinstance(training_enabled, bool):
            raise TypeError("training_enabled must be True or False")
        if isinstance(session_seed, bool) or not isinstance(session_seed, int):
            raise TypeError("session_seed must be an integer")
        if session_id is not None and not session_id.strip():
            raise ValueError("session_id cannot be empty")

        self.agent = agent
        self.human_color = _parse_color(human_color)
        self.checkpoint_label = checkpoint_label.strip()
        self.training_enabled = training_enabled
        self.session_seed = session_seed
        self.session_id = session_id.strip() if session_id is not None else uuid4().hex
        self.pgn_dir = Path(pgn_dir)
        self.dataset_path = Path(dataset_path)
        self._clock = clock
        self._lock = RLock()
        self._game_number = 0
        self._game = ChessGame()
        self._actors: list[str] = []
        self._pending: list[PendingHumanExample] = []
        self._completed_examples: tuple[TrainingExample, ...] | None = None
        self._append_result: HumanGuiAppendResult | None = None
        self._forced_result: str | None = None
        self._forced_termination: str | None = None
        self._started_at = _utc_now()
        self._game_id = ""
        self.new_game()

    @property
    def board(self) -> chess.Board:
        """Return a safe snapshot for rendering; mutating it cannot alter the game."""

        with self._lock:
            return self._game.board_copy(stack=True)

    @property
    def game_id(self) -> str:
        with self._lock:
            return self._game_id

    @property
    def move_history(self) -> tuple[chess.Move, ...]:
        with self._lock:
            return self._game.move_history

    @property
    def pending_human_examples(self) -> tuple[PendingHumanExample, ...]:
        """Human policy labels collected so far, without speculative values."""

        with self._lock:
            return tuple(self._pending)

    @property
    def game_over(self) -> bool:
        with self._lock:
            return self._forced_result is not None or self._game.is_game_over(claim_draw=True)

    @property
    def result(self) -> str:
        with self._lock:
            if self._forced_result is not None:
                return self._forced_result
            return self._game.result(claim_draw=True)

    @property
    def termination(self) -> str:
        with self._lock:
            if self._forced_termination is not None:
                return self._forced_termination
            outcome = self._game.outcome(claim_draw=True)
            return "unterminated" if outcome is None else outcome.termination.name.lower()

    @property
    def is_human_turn(self) -> bool:
        with self._lock:
            return not self.game_over and self._game.board.turn == self.human_color

    @property
    def status(self) -> str:
        """Short human-readable state suitable for a GUI status label."""

        with self._lock:
            if self.game_over:
                result = self.result
                reason = self.termination.replace("_", " ")
                if result == "1/2-1/2":
                    return f"Draw - {reason}."
                human_won = (result == "1-0") == (self.human_color == chess.WHITE)
                winner = "You win" if human_won else "Neural AI wins"
                return f"{winner} - {reason}."
            prefix = "Check. " if self._game.is_check() else ""
            turn = "Your turn." if self._game.board.turn == self.human_color else "Neural AI turn."
            return prefix + turn

    def _next_game_id(self) -> str:
        payload = json.dumps(
            {
                "game_number": self._game_number,
                "session_id": self.session_id,
                "session_seed": self.session_seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:20]
        return f"human-gui-{self._game_number:04d}-{digest}"

    def new_game(self, *, human_color: chess.Color | str | None = None) -> str:
        """Start a fresh standard game and return its deterministic session ID."""

        with self._lock:
            if human_color is not None:
                self.human_color = _parse_color(human_color)
            self._game_number += 1
            self._game = ChessGame()
            self._actors.clear()
            self._pending.clear()
            self._completed_examples = None
            self._append_result = None
            self._forced_result = None
            self._forced_termination = None
            started_at = self._clock()
            if not isinstance(started_at, datetime):
                raise TypeError("clock must return a datetime")
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            self._started_at = started_at.astimezone(UTC)
            self._game_id = self._next_game_id()
            return self._game_id

    def legal_sources(self) -> tuple[chess.Square, ...]:
        """Return selectable source squares for the current human turn."""

        with self._lock:
            if not self.is_human_turn:
                return ()
            return tuple(sorted({move.from_square for move in self._game.board.legal_moves}))

    def legal_destinations(self, source: SquareInput) -> tuple[chess.Square, ...]:
        """Return distinct legal destinations for one human source square."""

        source_square = _parse_square(source, name="source")
        with self._lock:
            if not self.is_human_turn:
                return ()
            return tuple(
                sorted(
                    {
                        move.to_square
                        for move in self._game.board.legal_moves
                        if move.from_square == source_square
                    }
                )
            )

    def promotion_choices(
        self,
        source: SquareInput,
        destination: SquareInput,
    ) -> tuple[chess.PieceType, ...]:
        """Return legal promotion pieces for a selected source/destination pair."""

        source_square = _parse_square(source, name="source")
        destination_square = _parse_square(destination, name="destination")
        with self._lock:
            if not self.is_human_turn:
                return ()
            available = {
                move.promotion
                for move in self._game.board.legal_moves
                if move.from_square == source_square
                and move.to_square == destination_square
                and move.promotion is not None
            }
            return tuple(piece for piece in _PROMOTION_ORDER if piece in available)

    def _ensure_active_turn(self, color: chess.Color, actor: str) -> None:
        if self.game_over:
            raise GameAlreadyOverError("The game is already over. Start a new game to move again.")
        if self._game.board.turn != color:
            raise WrongTurnError(f"It is not the {actor}'s turn.")

    def _pending_human_move(self, move: chess.Move) -> PendingHumanExample:
        board = self._game.board
        action = MoveEncoder().encode(move)
        policy = np.zeros(MoveEncoder.ACTION_SIZE, dtype=np.float32)
        policy[action] = 1.0
        color_name = "white" if board.turn == chess.WHITE else "black"
        return PendingHumanExample(
            board_tensor=BoardEncoder().encode(board),
            target_policy=policy,
            turn=board.turn,
            metadata={
                "game_id": self._game_id,
                "origin": "human_gui",
                "policy_source": "human",
                "ply": len(board.move_stack),
                # Preserve the raw en-passant target because it is one of the
                # model's input planes even when no capture is currently legal.
                "fen": board.fen(en_passant="fen"),
                "move_uci": move.uci(),
                "policy_action": action,
                "player_to_move": color_name,
                "human_color": "white" if self.human_color == chess.WHITE else "black",
                "checkpoint": self.checkpoint_label,
                "session_seed": self.session_seed,
            },
        )

    def play_human_move(
        self,
        source: SquareInput,
        destination: SquareInput,
        promotion: PromotionInput = None,
    ) -> chess.Move:
        """Validate and play one clicked human move."""

        source_square = _parse_square(source, name="source")
        destination_square = _parse_square(destination, name="destination")
        promotion_piece = _parse_promotion(promotion)
        with self._lock:
            self._ensure_active_turn(self.human_color, "human")
            matching = [
                move
                for move in self._game.board.legal_moves
                if move.from_square == source_square and move.to_square == destination_square
            ]
            if matching and all(move.promotion is not None for move in matching):
                if promotion_piece is None:
                    raise PromotionRequiredError(
                        "Choose queen, rook, bishop, or knight for this promotion."
                    )
                move = chess.Move(source_square, destination_square, promotion=promotion_piece)
            else:
                move = chess.Move(source_square, destination_square, promotion=promotion_piece)
            if move not in self._game.board.legal_moves:
                raise IllegalMoveError(f"Move {move.uci()!r} is not legal in the current position.")

            pending = self._pending_human_move(move) if self.training_enabled else None
            played = self._game.apply_move(move)
            self._actors.append("human")
            if pending is not None:
                self._pending.append(pending)
            self._completed_examples = None
            return played

    def play_ai_turn(self) -> chess.Move:
        """Ask the neural agent for one move and validate it before applying it."""

        with self._lock:
            ai_color = not self.human_color
            self._ensure_active_turn(ai_color, "neural agent")
            board_snapshot = self._game.board_copy(stack=True)
            expected_fen = self._game.fen()
            expected_game_id = self._game_id

        # GPU inference can be slow enough to notice.  It runs outside the
        # controller lock so Tk's main thread can keep rendering status and the
        # board.  The official position is revalidated before applying output.
        move = self.agent.choose_move(board_snapshot)

        with self._lock:
            if self._game_id != expected_game_id or self._game.fen() != expected_fen:
                raise AgentMoveError(
                    "The game changed while the neural agent was thinking; its stale move was "
                    "discarded."
                )
            self._ensure_active_turn(ai_color, "neural agent")
            if not isinstance(move, chess.Move) or move not in self._game.board.legal_moves:
                rendered = move.uci() if isinstance(move, chess.Move) else repr(move)
                raise AgentMoveError(
                    f"Neural agent {self.agent.name!r} returned illegal move {rendered}."
                )
            played = self._game.apply_move(move)
            self._actors.append("ai")
            self._completed_examples = None
            return played

    def undo_turn(self) -> tuple[chess.Move, ...]:
        """Undo the last human move and any neural reply after it.

        Returned moves are ordered from most recently played to oldest, matching
        repeated calls to :meth:`ChessGame.undo`.  An opening neural move made
        before any human decision is not considered a full human turn.
        """

        with self._lock:
            if self._append_result is not None:
                raise HumanNeuralGameError(
                    "Cannot undo after this game was appended to the training dataset."
                )
            self._forced_result = None
            self._forced_termination = None
            try:
                human_index = len(self._actors) - 1 - self._actors[::-1].index("human")
            except ValueError as exc:
                raise NoMoveToUndoError("No human turn is available to undo.") from exc

            undone: list[chess.Move] = []
            while len(self._actors) > human_index:
                undone.append(self._game.undo())
                self._actors.pop()
            self._pending = [
                item for item in self._pending if int(item.metadata["ply"]) < human_index
            ]
            self._completed_examples = None
            return tuple(undone)

    def resign_human(self) -> str:
        """Record a human resignation and return its PGN result."""

        with self._lock:
            if self.game_over:
                raise GameAlreadyOverError("The game is already over.")
            self._forced_result = "0-1" if self.human_color == chess.WHITE else "1-0"
            self._forced_termination = "human_resignation"
            self._completed_examples = None
            return self._forced_result

    def complete(self) -> tuple[TrainingExample, ...]:
        """Finalize in-memory labels for a completed game without writing files."""

        with self._lock:
            if not self.game_over:
                raise GameNotCompleteError(
                    "Unfinished games do not produce training examples. Finish or resign first."
                )
            if self._completed_examples is not None:
                return self._completed_examples
            if not self.training_enabled:
                self._completed_examples = ()
                return self._completed_examples

            result = self.result
            termination = self.termination
            completed: list[TrainingExample] = []
            for pending in self._pending:
                metadata = dict(pending.metadata)
                metadata["result"] = result
                metadata["termination"] = termination
                completed.append(
                    TrainingExample(
                        board_tensor=pending.board_tensor,
                        target_policy=pending.target_policy,
                        target_value=result_value_for_turn(result, pending.turn),
                        metadata=metadata,
                    )
                )
            self._completed_examples = tuple(completed)
            return self._completed_examples

    def pgn_game(self) -> chess.pgn.Game:
        """Build a replayable PGN with human-GUI provenance headers."""

        with self._lock:
            game = chess.pgn.Game.from_board(self._game.board_copy(stack=True))
            ai_name = str(self.agent.name)
            human_is_white = self.human_color == chess.WHITE
            game.headers["Event"] = "Human vs Neural Training Game"
            game.headers["Site"] = "Local GUI"
            game.headers["Date"] = self._started_at.strftime("%Y.%m.%d")
            game.headers["White"] = "Human" if human_is_white else ai_name
            game.headers["Black"] = ai_name if human_is_white else "Human"
            game.headers["Result"] = self.result
            game.headers["Termination"] = self.termination
            game.headers["GameId"] = self._game_id
            game.headers["HumanColor"] = "White" if human_is_white else "Black"
            game.headers["AIModel"] = ai_name
            game.headers["Checkpoint"] = self.checkpoint_label
            game.headers["DataOrigin"] = "HumanGUI"
            game.headers["TrainingEnabled"] = "True" if self.training_enabled else "False"
            game.headers["PlyCount"] = str(len(self._game.move_history))
            return game

    def export_pgn(self) -> str:
        """Export the current game; unfinished games are marked ``*``."""

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return self.pgn_game().accept(exporter)

    def save_pgn(self, path: str | Path | None = None) -> Path:
        """Atomically save PGN independently of any training-data decision."""

        with self._lock:
            destination = (
                self.pgn_dir / f"human_gui_{self._game_id}.pgn" if path is None else Path(path)
            )
            return save_pgn(self.pgn_game(), destination)

    def append_training_examples(
        self,
        *,
        confirm_training: bool,
    ) -> HumanGuiAppendResult:
        """Explicitly append this completed game's human moves to the GUI dataset."""

        with self._lock:
            if confirm_training is not True:
                raise ExplicitTrainingOptInRequired(
                    "Confirm this completed game explicitly before adding its human moves."
                )
            if not self.training_enabled:
                raise TrainingDisabledError(
                    "Training collection was disabled when this game started."
                )
            if self._append_result is not None:
                return self._append_result
            examples = self.complete()
            self._append_result = append_human_gui_game(
                self.dataset_path,
                examples,
                confirm_training=True,
            )
            return self._append_result
