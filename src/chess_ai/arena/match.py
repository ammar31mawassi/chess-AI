"""Play one controlled game between two chess agents.

The arena deliberately works with :class:`chess.Board` directly.  Agents receive
a copy of the board, so a buggy agent cannot mutate the match runner's state.
``python-chess`` remains the sole authority for move legality and game outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, runtime_checkable

import chess
import chess.pgn


@runtime_checkable
class ArenaAgent(Protocol):
    """The small agent contract required by the arena."""

    name: str

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return one legal move for ``board`` without mutating it."""


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """One legal move and the wall-clock time the agent used to choose it."""

    ply: int
    color: chess.Color
    agent_name: str
    uci: str
    san: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Complete, serializable summary of one arena game."""

    white_name: str
    black_name: str
    result: str
    termination: str
    moves: tuple[MoveRecord, ...]
    final_fen: str
    pgn: str
    duration_seconds: float
    seed: int | None = None
    illegal_agent: str | None = None
    illegal_move: str | None = None
    pgn_path: Path | None = None

    @property
    def plies(self) -> int:
        """Number of half-moves played in this match."""

        return len(self.moves)

    @property
    def is_draw(self) -> bool:
        return self.result == "1/2-1/2"

    @property
    def winner_name(self) -> str | None:
        if self.result == "1-0":
            return self.white_name
        if self.result == "0-1":
            return self.black_name
        return None

    @property
    def white_time_seconds(self) -> float:
        return sum(move.elapsed_seconds for move in self.moves if move.color == chess.WHITE)

    @property
    def black_time_seconds(self) -> float:
        return sum(move.elapsed_seconds for move in self.moves if move.color == chess.BLACK)


def _agent_name(agent: ArenaAgent) -> str:
    name = getattr(agent, "name", agent.__class__.__name__)
    return str(name)


def reseed_agent(agent: ArenaAgent, seed: int) -> None:
    """Best-effort reseeding for reusable agents.

    Agent factories are the most reliable way to get deterministic tournaments.
    This helper also supports the common ``set_seed`` method and ``rng``/``_rng``
    attributes, which keeps simple educational agents convenient to reuse.
    """

    for method_name in ("set_seed", "reseed"):
        seed_method = getattr(agent, method_name, None)
        if callable(seed_method):
            seed_method(seed)
            return

    for attribute in ("rng", "_rng", "_random"):
        rng = getattr(agent, attribute, None)
        seed_method = getattr(rng, "seed", None)
        if callable(seed_method):
            seed_method(seed)
            return


def _outcome_result(board: chess.Board, *, claim_draw: bool) -> tuple[str, str]:
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is None:
        return "*", "unterminated"
    return outcome.result(), outcome.termination.name.lower()


def _make_pgn(
    initial_board: chess.Board,
    moves: tuple[MoveRecord, ...],
    *,
    white_name: str,
    black_name: str,
    result: str,
    termination: str,
    seed: int | None,
    extra_headers: dict[str, str] | None,
) -> str:
    game = chess.pgn.Game()
    game.setup(initial_board)
    game.headers["Event"] = "Self-Improving Chess AI Arena"
    game.headers["Date"] = datetime.now(UTC).strftime("%Y.%m.%d")
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    game.headers["PlyCount"] = str(len(moves))
    if seed is not None:
        game.headers["Seed"] = str(seed)
    if extra_headers:
        for key, value in extra_headers.items():
            game.headers[str(key)] = str(value)

    node: chess.pgn.GameNode = game
    for move_record in moves:
        node = node.add_variation(chess.Move.from_uci(move_record.uci))

    return str(game)


def run_match(
    white: ArenaAgent,
    black: ArenaAgent,
    *,
    max_plies: int = 512,
    starting_fen: str | None = None,
    seed: int | None = None,
    claim_draw: bool = True,
    pgn_path: str | Path | None = None,
    extra_headers: dict[str, str] | None = None,
) -> MatchResult:
    """Play one match and return its result.

    An illegal or non-``chess.Move`` agent response is recorded as a forfeit.  A
    move-limit game is scored as a draw; this is an arena safeguard rather than
    an official chess termination.  ``max_plies`` counts half-moves.
    """

    if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies <= 0:
        raise ValueError("max_plies must be a positive integer")

    try:
        board = chess.Board(starting_fen) if starting_fen is not None else chess.Board()
    except ValueError as exc:
        raise ValueError(f"Invalid starting FEN: {starting_fen!r}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid starting position: {board.fen()!r}")

    initial_board = board.copy(stack=False)
    white_name = _agent_name(white)
    black_name = _agent_name(black)
    if seed is not None:
        reseed_agent(white, seed)
        reseed_agent(black, seed + 1)

    records: list[MoveRecord] = []
    match_started = perf_counter()
    illegal_agent: str | None = None
    illegal_move: str | None = None
    result = "*"
    termination = "unterminated"

    while True:
        if board.is_game_over(claim_draw=claim_draw):
            result, termination = _outcome_result(board, claim_draw=claim_draw)
            break
        if len(records) >= max_plies:
            result = "1/2-1/2"
            termination = "move_limit"
            break

        color = board.turn
        agent = white if color == chess.WHITE else black
        agent_name = white_name if color == chess.WHITE else black_name
        move_started = perf_counter()
        move = agent.choose_move(board.copy(stack=True))
        elapsed = perf_counter() - move_started

        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            illegal_agent = agent_name
            illegal_move = move.uci() if isinstance(move, chess.Move) else repr(move)
            result = "0-1" if color == chess.WHITE else "1-0"
            termination = "illegal_agent_move"
            break

        san = board.san(move)
        records.append(
            MoveRecord(
                ply=len(records) + 1,
                color=color,
                agent_name=agent_name,
                uci=move.uci(),
                san=san,
                elapsed_seconds=elapsed,
            )
        )
        board.push(move)

    duration = perf_counter() - match_started
    frozen_records = tuple(records)
    pgn = _make_pgn(
        initial_board,
        frozen_records,
        white_name=white_name,
        black_name=black_name,
        result=result,
        termination=termination,
        seed=seed,
        extra_headers=extra_headers,
    )

    saved_path: Path | None = None
    if pgn_path is not None:
        from chess_ai.storage.games import save_pgn

        saved_path = save_pgn(pgn, pgn_path)

    return MatchResult(
        white_name=white_name,
        black_name=black_name,
        result=result,
        termination=termination,
        moves=frozen_records,
        final_fen=board.fen(),
        pgn=pgn,
        duration_seconds=duration,
        seed=seed,
        illegal_agent=illegal_agent,
        illegal_move=illegal_move,
        pgn_path=saved_path,
    )


# A readable alias for callers that prefer the verb "play".
play_match = run_match
