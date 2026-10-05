"""Small deterministic PUCT search using the network's policy and value heads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from chess_ai.agents.neural_agent import NeuralAgent, NeuralAgentError


@dataclass(slots=True)
class _Edge:
    """Statistics stored from the parent position's perspective."""

    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: _Node | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class _Node:
    visits: int = 0
    expanded: bool = False
    edges: dict[int, _Edge] = field(default_factory=dict)


def _terminal_value(board: chess.Board) -> float | None:
    """Return an exact result for the side to move, or ``None`` if non-terminal."""

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class NeuralMCTSAgent(NeuralAgent):
    """Select moves with deterministic policy/value PUCT search.

    The policy head supplies legal-move priors. The value head evaluates newly
    reached leaves from the leaf side-to-move perspective. Values are negated
    while backing up each ply, so every edge statistic remains in its parent
    position's perspective.

    This deliberately compact implementation rebuilds a tree for every move.
    It is intended for evaluation and modest searches, not high-throughput
    AlphaZero self-play; batched leaf inference is a later optimization.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        simulations: int = 64,
        c_puct: float = 1.5,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if not math.isfinite(c_puct) or c_puct <= 0.0:
            raise ValueError("c_puct must be finite and positive")
        super().__init__(
            checkpoint,
            device=device,
            deterministic=True,
            temperature=0.0,
            seed=seed,
            name=name or f"neural-mcts:{Path(checkpoint).name}",
        )
        self.simulations = simulations
        self.c_puct = float(c_puct)

    def _expand(self, board: chess.Board, node: _Node) -> float:
        logits, value = self.evaluate(board)
        legal_actions = sorted(self._move_encoder.encode(move) for move in board.legal_moves)
        if not legal_actions:
            raise NeuralAgentError("No legal moves are available at a non-terminal search node")
        legal_logits = logits[legal_actions]
        legal_logits -= np.max(legal_logits)
        probabilities = np.exp(legal_logits)
        total = float(probabilities.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise NeuralAgentError("Could not form finite PUCT policy priors")
        probabilities /= total
        node.edges = {
            action: _Edge(prior=float(probability))
            for action, probability in zip(legal_actions, probabilities, strict=True)
        }
        node.expanded = True
        return value

    def _select_action(self, node: _Node) -> int:
        exploration_scale = math.sqrt(node.visits + 1)

        def rank(item: tuple[int, _Edge]) -> tuple[float, float, int]:
            action, edge = item
            score = edge.mean_value + (
                self.c_puct * edge.prior * exploration_scale / (1 + edge.visits)
            )
            # Prior and then the lower stable action index resolve exact ties.
            return score, edge.prior, -action

        return max(node.edges.items(), key=rank)[0]

    def _simulate(self, board: chess.Board, node: _Node) -> float:
        terminal = _terminal_value(board)
        if terminal is not None:
            return terminal
        if not node.expanded:
            return self._expand(board, node)

        action = self._select_action(node)
        edge = node.edges[action]
        move = self._move_encoder.decode(action)
        board.push(move)
        if edge.child is None:
            edge.child = _Node()
        child_value = self._simulate(board, edge.child)
        board.pop()

        parent_value = -child_value
        edge.visits += 1
        edge.value_sum += parent_value
        node.visits += 1
        return parent_value

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Run exactly ``simulations`` root traversals and return the visit winner."""

        if board.is_game_over(claim_draw=True):
            raise NeuralAgentError("Cannot choose a move because the game is already over")
        # Preserve repetition history so python-chess can adjudicate claimable
        # draws inside the search exactly as it can at the real root.
        search_board = board.copy(stack=True)
        root = _Node()
        self._expand(search_board, root)
        for _ in range(self.simulations):
            self._simulate(search_board, root)

        selected_action, _edge = max(
            root.edges.items(),
            key=lambda item: (item[1].visits, item[1].mean_value, item[1].prior, -item[0]),
        )
        move = self._move_encoder.decode(selected_action)
        if move not in board.legal_moves:
            raise NeuralAgentError(f"Internal search error: selected illegal move {move.uci()}")
        return move
