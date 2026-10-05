"""Human-in-the-loop play against an external offline chess application.

This module never inspects or controls another application.  It prints the AI's
move and waits for a human to type the external opponent's reply in UCI form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.pgn

from chess_ai.arena.match import ArenaAgent
from chess_ai.storage.games import PgnStore
from chess_ai.storage.metrics import (
    BenchmarkRecord,
    append_benchmark,
    format_benchmark_report,
    summarize_benchmarks,
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
ResultConfirmer = Callable[[str], str | bool | None]

HELP_TEXT = (
    "Commands: undo (take back the last ply), show (print the board), "
    "fen (print FEN), help, resign, quit. Enter moves in UCI form, for example e7e5."
)


class ExternalAgentMoveError(RuntimeError):
    """Raised when the AI returns something other than a legal chess move."""


@dataclass(frozen=True, slots=True)
class ExternalSessionResult:
    """Saved artifacts and final state from one external-opponent session."""

    result: str
    termination: str
    final_fen: str
    moves: tuple[str, ...]
    pgn: str
    pgn_path: Path
    benchmark_path: Path
    benchmark_record: BenchmarkRecord

    @property
    def move_count(self) -> int:
        """Number of plies in the saved game."""

        return len(self.moves)

    @property
    def unfinished(self) -> bool:
        return self.result == "*"


def parse_ai_color(value: chess.Color | str) -> chess.Color:
    if value is chess.WHITE or (isinstance(value, str) and value.strip().lower() == "white"):
        return chess.WHITE
    if value is chess.BLACK or (isinstance(value, str) and value.strip().lower() == "black"):
        return chess.BLACK
    raise ValueError("ai_color must be 'white', 'black', chess.WHITE, or chess.BLACK")


def _normalize_confirmed_result(value: str) -> str:
    aliases = {
        "white": "1-0",
        "white win": "1-0",
        "black": "0-1",
        "black win": "0-1",
        "draw": "1/2-1/2",
        "unfinished": "*",
        "aborted": "*",
    }
    normalized = value.strip().lower()
    result = aliases.get(normalized, value.strip())
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        raise ValueError("confirmed result must be 1-0, 0-1, 1/2-1/2, or *")
    return result


class ExternalOpponentSession:
    """Coordinate one manual benchmark game and save its PGN/JSONL record."""

    def __init__(
        self,
        ai_agent: ArenaAgent,
        *,
        ai_color: chess.Color | str,
        opponent_label: str,
        difficulty_level: str,
        checkpoint: str,
        pgn_dir: str | Path = Path("data/games/external"),
        benchmark_path: str | Path = Path("data/metrics/external_benchmarks.jsonl"),
        starting_fen: str | None = None,
        max_plies: int = 512,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
        result_confirmer: ResultConfirmer | None = None,
    ) -> None:
        if not opponent_label.strip():
            raise ValueError("opponent_label cannot be empty")
        if not difficulty_level.strip():
            raise ValueError("difficulty_level cannot be empty")
        if not checkpoint.strip():
            raise ValueError("checkpoint cannot be empty")
        if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies <= 0:
            raise ValueError("max_plies must be a positive integer")
        try:
            board = chess.Board(starting_fen) if starting_fen is not None else chess.Board()
        except ValueError as exc:
            raise ValueError(f"Invalid starting FEN: {starting_fen!r}") from exc
        if not board.is_valid():
            raise ValueError(f"Invalid starting position: {board.fen()!r}")

        self.ai_agent = ai_agent
        self.ai_color = parse_ai_color(ai_color)
        self.opponent_label = opponent_label.strip()
        self.difficulty_level = difficulty_level.strip()
        self.checkpoint = checkpoint.strip()
        self.pgn_store = PgnStore(pgn_dir)
        self.benchmark_path = Path(benchmark_path)
        self.max_plies = max_plies
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.result_confirmer = result_confirmer
        self._initial_board = board.copy(stack=False)

    def _display_board(self, board: chess.Board) -> None:
        self.output_fn(str(board))

    def _confirm_result(self, proposed: str, reason: str) -> str | None:
        prompt = f"{reason} Proposed PGN result: {proposed}. Confirm or enter a corrected result: "
        if self.result_confirmer is None:
            if proposed != "*":
                return proposed
            response: str | bool | None = self.input_fn(prompt)
            if isinstance(response, str) and not response.strip():
                return proposed
        else:
            response = self.result_confirmer(prompt)
        if response is False:
            return None
        if response is True or response is None:
            return proposed
        return _normalize_confirmed_result(response)

    def _ai_move(self, board: chess.Board) -> None:
        move = self.ai_agent.choose_move(board.copy(stack=True))
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            rendered = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise ExternalAgentMoveError(
                f"AI agent {getattr(self.ai_agent, 'name', 'AI')!r} returned illegal move "
                f"{rendered} for position {board.fen()}"
            )
        self.output_fn(f"AI move: {move.uci()}")
        self.output_fn("Enter that move in the external application.")
        board.push(move)
        self._display_board(board)

    def _read_opponent_turn(self, board: chess.Board) -> tuple[str, str] | None:
        """Read until a legal move, a state-changing command, or termination."""

        while True:
            raw = self.input_fn("Type the opponent move: ").strip()
            command = raw.lower()
            if command == "help":
                self.output_fn(HELP_TEXT)
                continue
            if command == "show":
                self._display_board(board)
                continue
            if command == "fen":
                self.output_fn(f"FEN: {board.fen()}")
                continue
            if command == "undo":
                if not board.move_stack:
                    self.output_fn("Nothing to undo.")
                    continue
                undone = board.pop()
                self.output_fn(f"Undid {undone.uci()}. Undo the same move externally too.")
                self._display_board(board)
                return None
            if command == "resign":
                proposed = "1-0" if self.ai_color == chess.WHITE else "0-1"
                confirmed = self._confirm_result(proposed, "The external opponent resigned.")
                if confirmed is None:
                    self.output_fn("Resignation cancelled.")
                    continue
                return confirmed, "opponent_resignation"
            if command == "quit":
                confirmed = self._confirm_result("*", "The session was quit before a known result.")
                if confirmed is None:
                    self.output_fn("Quit cancelled.")
                    continue
                return confirmed, "abandoned"
            if not raw:
                self.output_fn("Enter a UCI move or type help to list commands.")
                continue

            try:
                move = chess.Move.from_uci(raw.lower())
            except chess.InvalidMoveError:
                self.output_fn(
                    f"Malformed move {raw!r}. Use UCI notation such as e7e5 or type help."
                )
                continue
            if move not in board.legal_moves:
                self.output_fn(f"Illegal move {move.uci()!r} in this position. Try again.")
                continue
            board.push(move)
            self._display_board(board)
            return None

    def _make_pgn(
        self,
        board: chess.Board,
        *,
        result: str,
        termination: str,
        timestamp: datetime,
    ) -> chess.pgn.Game:
        game = chess.pgn.Game.from_board(board)
        ai_name = str(getattr(self.ai_agent, "name", self.ai_agent.__class__.__name__))
        opponent_name = f"{self.opponent_label} [{self.difficulty_level}]"
        game.headers["Event"] = "Manual External Opponent Benchmark"
        game.headers["Date"] = timestamp.strftime("%Y.%m.%d")
        game.headers["White"] = ai_name if self.ai_color == chess.WHITE else opponent_name
        game.headers["Black"] = opponent_name if self.ai_color == chess.WHITE else ai_name
        game.headers["Result"] = result
        game.headers["Termination"] = termination
        game.headers["AIModel"] = ai_name
        game.headers["Checkpoint"] = self.checkpoint
        game.headers["Opponent"] = self.opponent_label
        game.headers["Difficulty"] = self.difficulty_level
        game.headers["AIColor"] = "White" if self.ai_color == chess.WHITE else "Black"
        game.headers["PlyCount"] = str(len(board.move_stack))
        return game

    def run(self) -> ExternalSessionResult:
        """Run the terminal loop, then save PGN and a separate benchmark record."""

        board = self._initial_board.copy(stack=False)
        color_name = "White" if self.ai_color == chess.WHITE else "Black"
        self.output_fn(f"Opponent label: {self.opponent_label}")
        self.output_fn(f"Difficulty level: {self.difficulty_level}")
        self.output_fn(f"AI color: {color_name}")
        self.output_fn(HELP_TEXT)
        self._display_board(board)

        result = "*"
        termination = "unterminated"
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result = outcome.result()
                termination = outcome.termination.name.lower()
                break
            if len(board.move_stack) >= self.max_plies:
                confirmed = self._confirm_result(
                    "*", "The external-session move limit was reached."
                )
                if confirmed is None:
                    confirmed = "*"
                result = confirmed
                termination = "move_limit"
                break

            if board.turn == self.ai_color:
                self._ai_move(board)
                continue

            terminal = self._read_opponent_turn(board)
            if terminal is not None:
                result, termination = terminal
                break

        timestamp = datetime.now(UTC)
        game = self._make_pgn(
            board,
            result=result,
            termination=termination,
            timestamp=timestamp,
        )
        safe_stamp = timestamp.strftime("%Y%m%dT%H%M%S_%f")
        pgn_path = self.pgn_store.save(game, f"external_{safe_stamp}.pgn")
        record = BenchmarkRecord(
            external_opponent_name=self.opponent_label,
            difficulty_level=self.difficulty_level,
            checkpoint=self.checkpoint,
            color=color_name.lower(),
            result=result,
            move_count=len(board.move_stack),
            timestamp=timestamp.isoformat(),
            pgn_path=str(pgn_path),
        )
        saved_benchmark_path = append_benchmark(self.benchmark_path, record)
        self.output_fn(f"Saved PGN: {pgn_path}")
        self.output_fn(f"Saved benchmark record: {saved_benchmark_path}")

        return ExternalSessionResult(
            result=result,
            termination=termination,
            final_fen=board.fen(),
            moves=tuple(move.uci() for move in board.move_stack),
            pgn=str(game),
            pgn_path=pgn_path,
            benchmark_path=saved_benchmark_path,
            benchmark_record=record,
        )


def run_external_session(
    ai_agent: ArenaAgent,
    *,
    ai_color: chess.Color | str,
    opponent_label: str,
    difficulty_level: str,
    checkpoint: str,
    pgn_dir: str | Path = Path("data/games/external"),
    benchmark_path: str | Path = Path("data/metrics/external_benchmarks.jsonl"),
    starting_fen: str | None = None,
    max_plies: int = 512,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    result_confirmer: ResultConfirmer | None = None,
) -> ExternalSessionResult:
    """Functional wrapper around :class:`ExternalOpponentSession`."""

    return ExternalOpponentSession(
        ai_agent,
        ai_color=ai_color,
        opponent_label=opponent_label,
        difficulty_level=difficulty_level,
        checkpoint=checkpoint,
        pgn_dir=pgn_dir,
        benchmark_path=benchmark_path,
        starting_fen=starting_fen,
        max_plies=max_plies,
        input_fn=input_fn,
        output_fn=output_fn,
        result_confirmer=result_confirmer,
    ).run()


def report_external_benchmarks(path: str | Path) -> str:
    """Return cumulative results grouped by opponent, level, and checkpoint."""

    return format_benchmark_report(summarize_benchmarks(path))
