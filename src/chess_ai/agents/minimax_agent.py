"""A compact educational Negamax engine with alpha-beta pruning."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Final

import chess

from chess_ai.agents.protocol import NoLegalMovesError

# Values are in centipawn-like units.  Only relative sizes matter here.
PIECE_VALUES: Final[dict[chess.PieceType, int]] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Tables are indexed using python-chess squares: a1 through h1, then a2, ...
# They describe good squares for White.  Black uses a vertically mirrored index.
# Keeping eight entries per line makes the board geometry visible to a learner.
# fmt: off
PAWN_TABLE: Final = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
)
KNIGHT_TABLE: Final = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
BISHOP_TABLE: Final = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
ROOK_TABLE: Final = (
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)
QUEEN_TABLE: Final = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)
KING_TABLE: Final = (
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)
# fmt: on

PIECE_SQUARE_TABLES: Final[dict[chess.PieceType, tuple[int, ...]]] = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


@dataclass(frozen=True, slots=True)
class MinimaxMoveScore:
    """One legal root move and its exact minimax score for the active player."""

    move: chess.Move
    score: int


class MinimaxAgent:
    """Search legal moves with Negamax and alpha-beta pruning.

    Every score returned by :meth:`evaluate` or ``_negamax`` is from the
    *active player's* perspective.  A positive score is good for the player
    whose turn it is.  After making a move, Negamax changes perspective simply
    by negating the child's score.
    """

    MATE_SCORE = 100_000
    INFINITY = MATE_SCORE + 10_000

    def __init__(
        self,
        depth: int = 2,
        deterministic: bool = True,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("search depth must be an integer of at least 1")

        self.depth = depth
        self.deterministic = deterministic
        self.seed = seed
        self.name = name if name is not None else "Minimax"
        self.node_count = 0
        self.last_search_time_seconds = 0.0
        self.last_score: int | None = None
        self._random = random.Random(seed)

    @property
    def nodes_searched(self) -> int:
        """Alias that reads naturally in reports."""

        return self.node_count

    @property
    def search_time_seconds(self) -> float:
        """Duration of the most recently completed search."""

        return self.last_search_time_seconds

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return the best move found without changing the supplied board."""

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise NoLegalMovesError("MinimaxAgent cannot move because the game is over.")

        started = time.perf_counter()
        self.node_count = 0
        self.last_score = None

        alpha = -self.INFINITY
        beta = self.INFINITY
        best_score = -self.INFINITY
        best_moves: list[chess.Move] = []

        for move in self._ordered_moves(board, legal_moves):
            board.push(move)
            try:
                score = -self._negamax(
                    board,
                    depth=self.depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    ply_from_root=1,
                )
            finally:
                # Even an unexpected evaluation error must not corrupt the
                # board that belongs to the caller.
                board.pop()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, score)

        self.last_score = best_score
        self.last_search_time_seconds = time.perf_counter() - started
        if self.deterministic:
            return best_moves[0]
        return self._random.choice(best_moves)

    def analyze_moves(self, board: chess.Board) -> tuple[MinimaxMoveScore, ...]:
        """Score every legal root move with a full search window.

        ``choose_move`` may narrow later root searches as an alpha-beta
        optimization because it only needs the best move. Teacher-policy data
        needs comparable scores for several moves, so this method deliberately
        evaluates every root move with the same full window.
        """

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise NoLegalMovesError("MinimaxAgent cannot analyze moves because the game is over.")

        started = time.perf_counter()
        self.node_count = 0
        self.last_score = None
        scored: list[MinimaxMoveScore] = []
        for move in self._ordered_moves(board, legal_moves):
            board.push(move)
            try:
                score = -self._negamax(
                    board,
                    depth=self.depth - 1,
                    alpha=-self.INFINITY,
                    beta=self.INFINITY,
                    ply_from_root=1,
                )
            finally:
                board.pop()
            scored.append(MinimaxMoveScore(move=move, score=score))

        self.last_score = max(item.score for item in scored)
        self.last_search_time_seconds = time.perf_counter() - started
        return tuple(scored)

    def evaluate(self, board: chess.Board) -> int:
        """Evaluate *board* from the player-to-move's perspective."""

        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")
        if board.is_checkmate():
            return -self.MATE_SCORE
        if board.is_game_over(claim_draw=True):
            return 0

        white_score = 0
        black_score = 0
        for square, piece in board.piece_map().items():
            table = PIECE_SQUARE_TABLES[piece.piece_type]
            if piece.color == chess.WHITE:
                white_score += PIECE_VALUES[piece.piece_type] + table[square]
            else:
                mirrored_square = chess.square_mirror(square)
                black_score += PIECE_VALUES[piece.piece_type] + table[mirrored_square]

        score_for_white = white_score - black_score
        return score_for_white if board.turn == chess.WHITE else -score_for_white

    def _negamax(
        self,
        board: chess.Board,
        *,
        depth: int,
        alpha: int,
        beta: int,
        ply_from_root: int,
    ) -> int:
        self.node_count += 1

        if board.is_checkmate():
            # The active player is mated.  Adding ply prefers faster mates and
            # delays an unavoidable loss.
            return -self.MATE_SCORE + ply_from_root
        if board.is_game_over(claim_draw=True):
            return 0
        if depth == 0:
            return self.evaluate(board)

        best_score = -self.INFINITY
        for move in self._ordered_moves(board, list(board.legal_moves)):
            board.push(move)
            try:
                score = -self._negamax(
                    board,
                    depth=depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    ply_from_root=ply_from_root + 1,
                )
            finally:
                board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score

    def _ordered_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
    ) -> list[chess.Move]:
        """Try tactical moves first so alpha-beta can discard more branches."""

        if self.deterministic:
            return sorted(
                moves, key=lambda move: (-self._move_order_score(board, move), move.uci())
            )

        # Python's sort is stable, so shuffling first randomizes only equal
        # ordering scores.  A supplied seed makes this mode reproducible.
        self._random.shuffle(moves)
        return sorted(moves, key=lambda move: -self._move_order_score(board, move))

    @staticmethod
    def _move_order_score(board: chess.Board, move: chess.Move) -> int:
        score = 0

        if move.promotion is not None:
            score += 800 + PIECE_VALUES[move.promotion]

        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            # In en passant, the captured pawn is behind the empty target.
            victim_value = (
                PIECE_VALUES[chess.PAWN] if victim is None else PIECE_VALUES[victim.piece_type]
            )
            attacker = board.piece_at(move.from_square)
            attacker_value = 0 if attacker is None else PIECE_VALUES[attacker.piece_type]
            score += 10 * victim_value - attacker_value

        if board.gives_check(move):
            score += 50
        if board.is_castling(move):
            score += 25
        return score
