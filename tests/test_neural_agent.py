from __future__ import annotations

from pathlib import Path

import chess
import pytest

from chess_ai.agents.neural_agent import NeuralAgent, NeuralAgentError
from chess_ai.model.checkpoint import save_checkpoint
from chess_ai.model.policy_value_net import PolicyValueNet


def _checkpoint(path: Path, *, action_size: int = 4208) -> Path:
    model = PolicyValueNet(channels=8, residual_blocks=0, action_size=action_size)
    return save_checkpoint(path, model)


def test_neural_agent_always_returns_a_legal_move(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    agent = NeuralAgent(path, device="cpu")
    board = chess.Board()

    move = agent.choose_move(board)

    assert move in board.legal_moves
    assert board == chess.Board()  # choosing does not mutate the supplied board


def test_neural_agent_sampling_is_reproducible_with_a_seed(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    first = NeuralAgent(path, device="cpu", deterministic=False, temperature=1.0, seed=17)
    second = NeuralAgent(path, device="cpu", deterministic=False, temperature=1.0, seed=17)

    assert first.choose_move(chess.Board()) == second.choose_move(chess.Board())


def test_neural_agent_rejects_incompatible_action_space(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "wrong-actions.pt", action_size=10)

    with pytest.raises(NeuralAgentError, match="4208"):
        NeuralAgent(path, device="cpu")


def test_neural_agent_rejects_unavailable_cuda(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(NeuralAgentError, match="CUDA"):
        NeuralAgent(path, device="cuda")
