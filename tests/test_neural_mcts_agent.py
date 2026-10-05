from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from chess_ai.agents.neural_agent import NeuralAgentError
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent, _terminal_value
from chess_ai.model.checkpoint import save_checkpoint
from chess_ai.model.policy_value_net import PolicyValueNet


def _checkpoint(path: Path) -> Path:
    model = PolicyValueNet(channels=8, residual_blocks=0)
    return save_checkpoint(path, model)


def test_mcts_returns_a_legal_deterministic_move_without_mutating_board(
    tmp_path: Path,
) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    first = NeuralMCTSAgent(path, device="cpu", simulations=4)
    second = NeuralMCTSAgent(path, device="cpu", simulations=4)
    board = chess.Board()

    move = first.choose_move(board)

    assert move in board.legal_moves
    assert move == second.choose_move(board)
    assert board == chess.Board()


def test_terminal_value_uses_the_side_to_move_perspective() -> None:
    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

    assert _terminal_value(checkmate) == -1.0
    assert _terminal_value(stalemate) == 0.0
    assert _terminal_value(chess.Board()) is None


def test_mcts_validates_search_settings(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")

    with pytest.raises(ValueError, match="simulations"):
        NeuralMCTSAgent(path, device="cpu", simulations=0)
    with pytest.raises(ValueError, match="c_puct"):
        NeuralMCTSAgent(path, device="cpu", c_puct=0.0)


def test_mcts_uses_leaf_value_and_negates_the_opponent_perspective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    agent = NeuralMCTSAgent(path, device="cpu", simulations=2)
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")

    def fake_evaluate(board: chess.Board) -> tuple[np.ndarray, float]:
        logits = np.full(4208, -100.0, dtype=np.float64)
        if not board.move_stack:
            logits[agent._move_encoder.encode(e4)] = 2.0
            logits[agent._move_encoder.encode(d4)] = 1.0
            return logits, 0.0
        # Values are for Black after each candidate move. Black likes e4's
        # child and dislikes d4's child, so White should ultimately choose d4.
        return logits, 1.0 if board.peek() == e4 else -1.0

    monkeypatch.setattr(agent, "evaluate", fake_evaluate)

    assert agent.choose_move(chess.Board()) == d4


def test_mcts_rejects_terminal_board(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    agent = NeuralMCTSAgent(path, device="cpu", simulations=1)

    with pytest.raises(NeuralAgentError, match="game is already over"):
        agent.choose_move(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"))
