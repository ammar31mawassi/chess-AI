"""Opt-in, separately selectable PUCT experiments; production search is unchanged."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_ai.agents.neural_agent import NeuralAgentError
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent, _Edge, _Node, _terminal_value

ARMS = ("baseline", "cache", "batch", "tactical", "reuse")


@dataclass
class SearchMetrics:
    proven_mate_in_one: bool = False
    simulations: int = 0
    network_positions: int = 0
    network_batches: int = 0
    cache_hits: int = 0
    terminal_checks: int = 0
    tactical_nodes: int = 0
    reused_visits: int = 0
    encode_seconds: float = 0.0
    network_seconds: float = 0.0
    terminal_seconds: float = 0.0
    legal_seconds: float = 0.0
    select_seconds: float = 0.0
    elapsed_seconds: float = 0.0


class ExperimentalPUCT(NeuralMCTSAgent):
    """One intervention at a time, with checkpoint-scoped cache and per-move metrics.

    Timed searches check their soft deadline between traversals/batches. Fixed-work
    runs are deterministic; timed runs are deliberately not claimed reproducible.
    Model weights must remain immutable for the lifetime of this agent.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        arm: str = "baseline",
        device: str = "cpu",
        simulations: int = 32,
        seed: int = 31102041,
        seconds: float | None = None,
        cache_size: int = 4096,
        batch_size: int = 8,
        tactical_depth: int = 2,
        tactical_budget: int = 32,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Choose exactly one arm from {ARMS}")
        for name, value in (
            ("cache_size", cache_size),
            ("batch_size", batch_size),
            ("tactical_depth", tactical_depth),
            ("tactical_budget", tactical_budget),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("seconds must be finite and positive")
        super().__init__(checkpoint, device=device, simulations=simulations, seed=seed)
        self.arm = arm
        self.name = f"experimental-puct:{arm}"
        self.seconds = seconds
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.tactical_depth = tactical_depth
        self.tactical_budget = tactical_budget
        self.metrics = SearchMetrics()
        self._cache: OrderedDict[bytes, tuple[np.ndarray, float]] = OrderedDict()
        self._previous: tuple[str, tuple[str, ...], _Node] | None = None
        self.root_visits: dict[str, int] = {}

    def report(self) -> dict[str, Any]:
        return asdict(self.metrics)

    def clear_state(self) -> None:
        """Start a cold benchmark/game without cross-case cache or tree reuse."""
        self._cache.clear()
        self._previous = None

    def _forward(self, encoded: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        started = time.perf_counter()
        batch = torch.from_numpy(np.stack(encoded)).to(self.device)
        with torch.inference_mode():
            policy, value = self.model(batch)
        logits = policy.detach().cpu().numpy().astype(np.float64)
        values = value.detach().cpu().numpy().reshape(-1)
        # CPU transfer synchronizes CUDA before stopping the timer.
        self.metrics.network_seconds += time.perf_counter() - started
        self.metrics.network_positions += len(encoded)
        self.metrics.network_batches += 1
        if not np.isfinite(logits).all() or not np.isfinite(values).all():
            raise NeuralAgentError("The checkpoint produced a non-finite policy or value")
        return [(row.copy(), float(v)) for row, v in zip(logits, values, strict=True)]

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        encoded = self._board_encoder.encode(board)
        self.metrics.encode_seconds += time.perf_counter() - started
        if self.arm != "cache":
            return self._forward([encoded])[0]
        # Exact network input, including inactive EP. No terminal or tree results
        # are cached: those depend on clocks/history absent from these planes.
        key = encoded.tobytes()
        if key in self._cache:
            self.metrics.cache_hits += 1
            self._cache.move_to_end(key)
            logits, value = self._cache[key]
            return logits.copy(), value
        result = self._forward([encoded])[0]
        self._cache[key] = (result[0].copy(), result[1])
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _terminal(self, board: chess.Board) -> float | None:
        started = time.perf_counter()
        result = _terminal_value(board)
        self.metrics.terminal_checks += 1
        self.metrics.terminal_seconds += time.perf_counter() - started
        return result

    def _mate_in_one(self, board: chess.Board) -> chess.Move | None:
        """Exact tactical safeguard; use python-chess, not a material heuristic."""
        for move in sorted(board.legal_moves, key=self._move_encoder.encode):
            board.push(move)
            try:
                started = time.perf_counter()
                mates = board.is_checkmate()
                self.metrics.terminal_checks += 1
                self.metrics.terminal_seconds += time.perf_counter() - started
                if mates:
                    return move
            finally:
                board.pop()
        return None

    def _populate(self, board: chess.Board, node: _Node, logits: np.ndarray) -> None:
        started = time.perf_counter()
        actions = sorted(self._move_encoder.encode(move) for move in board.legal_moves)
        self.metrics.legal_seconds += time.perf_counter() - started
        if not actions:
            raise NeuralAgentError("Non-terminal node has no legal moves")
        selected = logits[actions]
        probabilities = np.exp(selected - selected.max())
        probabilities /= probabilities.sum()
        node.edges = {
            action: _Edge(prior=float(p)) for action, p in zip(actions, probabilities, strict=True)
        }
        node.expanded = True

    def _quiet_value(
        self,
        board: chess.Board,
        depth: int,
        remaining: list[int],
        static: float | None = None,
    ) -> float:
        terminal = self._terminal(board)
        if terminal is not None:
            return terminal
        if static is None:
            _, static = self.evaluate(board)
        if depth == 0 or remaining[0] == 0:
            return static
        in_check = board.is_check()
        # Stand pat is forbidden while in check. At the hard work boundary the
        # network is an approximate fallback, never an asserted terminal result.
        best = -math.inf if in_check else static
        moves = sorted(
            (m for m in board.legal_moves if in_check or board.is_capture(m) or m.promotion),
            key=lambda m: (not bool(m.promotion), not board.is_capture(m), m.uci()),
        )
        for move in moves:
            if remaining[0] == 0:
                break
            remaining[0] -= 1
            self.metrics.tactical_nodes += 1
            board.push(move)
            try:
                score = -self._quiet_value(board, depth - 1, remaining)
            finally:
                board.pop()
            best = max(best, score)
        return static if best == -math.inf else best

    def _expand(self, board: chess.Board, node: _Node) -> float:
        logits, value = self.evaluate(board)
        self._populate(board, node, logits)
        if self.arm == "tactical":
            value = self._quiet_value(board, self.tactical_depth, [self.tactical_budget], value)
        return value

    def _select_action(self, node: _Node) -> int:
        started = time.perf_counter()
        action = super()._select_action(node)
        self.metrics.select_seconds += time.perf_counter() - started
        return action

    @staticmethod
    def _backup(path: list[tuple[_Node, _Edge]], value: float) -> None:
        for node, edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.value_sum += value
            node.visits += 1

    def _traverse(
        self,
        board: chess.Board,
        root: _Node,
    ) -> tuple[_Node, list[tuple[_Node, _Edge]], float | None]:
        node = root
        path: list[tuple[_Node, _Edge]] = []
        while True:
            terminal = self._terminal(board)
            if terminal is not None or not node.expanded:
                return node, path, terminal
            action = self._select_action(node)
            edge = node.edges[action]
            board.push(self._move_encoder.decode(action))
            if edge.child is None:
                edge.child = _Node()
            path.append((node, edge))
            node = edge.child

    def _wave(self, board: chess.Board, root: _Node, count: int) -> int:
        pending: list[tuple[_Node, list[tuple[_Node, _Edge]], chess.Board]] = []
        pending_ids: set[int] = set()
        completed = 0
        try:
            for _ in range(count):
                leaf_board = board.copy(stack=True)
                node, path, terminal = self._traverse(leaf_board, root)
                if terminal is not None:
                    self._backup(path, terminal)
                    completed += 1
                    continue
                if id(node) in pending_ids:
                    # Do not evaluate/expand the same pending leaf twice. Flush
                    # this smaller batch rather than pretending a visit happened.
                    break
                pending_ids.add(id(node))
                pending.append((node, path, leaf_board))
                for parent, edge in path:
                    parent.visits += 1
                    edge.visits += 1
                    edge.value_sum -= 1.0  # temporary virtual loss
            if pending:
                started = time.perf_counter()
                tensors = [self._board_encoder.encode(b) for _, _, b in pending]
                self.metrics.encode_seconds += time.perf_counter() - started
                evaluations = self._forward(tensors)
            else:
                evaluations = []
        finally:
            for _, path, _ in pending:
                for parent, edge in path:
                    parent.visits -= 1
                    edge.visits -= 1
                    edge.value_sum += 1.0
        for (node, path, leaf_board), (logits, value) in zip(pending, evaluations, strict=True):
            self._populate(leaf_board, node, logits)
            self._backup(path, value)
        return completed + len(pending)

    @staticmethod
    def _history(board: chess.Board) -> tuple[str, tuple[str, ...]]:
        return board.root().fen(en_passant="fen"), tuple(m.uci() for m in board.move_stack)

    def _root(self, board: chess.Board) -> _Node:
        if self.arm == "reuse" and self._previous is not None:
            previous_fen, previous_moves, node = self._previous
            fen, moves = self._history(board)
            if fen == previous_fen and moves[: len(previous_moves)] == previous_moves:
                for uci in moves[len(previous_moves) :]:
                    edge = node.edges.get(self._move_encoder.encode(chess.Move.from_uci(uci)))
                    if edge is None or edge.child is None:
                        break
                    node = edge.child
                else:
                    self.metrics.reused_visits = node.visits
                    return node
        return _Node()

    def choose_move(self, board: chess.Board) -> chess.Move:
        started = time.perf_counter()
        self.metrics = SearchMetrics()
        if self._terminal(board) is not None:
            raise NeuralAgentError("Cannot choose a move because the game is already over")
        root = self._root(board)
        search_board = board.copy(stack=True)
        if self.arm == "tactical":
            mate = self._mate_in_one(search_board)
            if mate is not None:
                self.metrics.proven_mate_in_one = True
                self.metrics.elapsed_seconds = time.perf_counter() - started
                self.root_visits = {}
                return mate
        if not root.expanded:
            # The root value is not backed up; tactical verification there would
            # spend work without affecting any edge statistic.
            logits, _ = self.evaluate(search_board)
            self._populate(search_board, root, logits)
        while self.metrics.simulations < self.simulations:
            # Always complete one traversal, even when initial expansion exceeds
            # the soft deadline. Report actual elapsed time, including overshoot.
            if (
                self.metrics.simulations
                and self.seconds is not None
                and time.perf_counter() - started >= self.seconds
            ):
                break
            if self.arm == "batch":
                self.metrics.simulations += self._wave(
                    search_board,
                    root,
                    min(self.batch_size, self.simulations - self.metrics.simulations),
                )
            else:
                initial_depth = len(search_board.move_stack)
                try:
                    node, path, terminal = self._traverse(search_board, root)
                    value = terminal if terminal is not None else self._expand(search_board, node)
                finally:
                    while len(search_board.move_stack) > initial_depth:
                        search_board.pop()
                self._backup(path, value)
                self.metrics.simulations += 1
        action, _ = max(
            root.edges.items(),
            key=lambda item: (item[1].visits, item[1].mean_value, item[1].prior, -item[0]),
        )
        move = self._move_encoder.decode(action)
        if move not in board.legal_moves:
            raise NeuralAgentError("Experimental search selected an illegal move")
        self.root_visits = {
            self._move_encoder.decode(a).uci(): e.visits for a, e in root.edges.items()
        }
        if self.arm == "reuse":
            fen, moves = self._history(board)
            self._previous = (fen, moves, root)
        self.metrics.elapsed_seconds = time.perf_counter() - started
        return move
