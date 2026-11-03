from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_ai.agents.neural_agent import NeuralAgentError
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent, _Node
from chess_ai.agents.search_experiments import ARMS, ExperimentalPUCT
from chess_ai.arena.search_experiments import benchmark, paired_matches, read_cases
from chess_ai.model.checkpoint import save_checkpoint
from chess_ai.model.policy_value_net import PolicyValueNet

SUITE = Path(__file__).resolve().parents[1] / "configs/search_development_v1.json"


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(31102041)
    return save_checkpoint(tmp_path / "model.pt", PolicyValueNet(channels=8, residual_blocks=0))


@pytest.mark.parametrize("case", read_cases(SUITE), ids=lambda case: case["id"])
def test_instrumented_baseline_matches_production(checkpoint: Path, case: dict) -> None:
    board = chess.Board(case["fen"])
    for uci in case.get("setup_moves", []):
        board.push_uci(uci)
    original = board.copy(stack=True)
    production = NeuralMCTSAgent(checkpoint, device="cpu", simulations=8)
    baseline = ExperimentalPUCT(checkpoint, simulations=8)
    cached = ExperimentalPUCT(checkpoint, arm="cache", simulations=8)
    assert production.choose_move(board) == baseline.choose_move(board) == cached.choose_move(board)
    assert board == original and board.move_stack == original.move_stack
    assert sum(baseline.root_visits.values()) == 8


def test_cache_is_bounded_checkpoint_local_and_defensive(checkpoint: Path) -> None:
    agent = ExperimentalPUCT(checkpoint, arm="cache", cache_size=1)
    first, value = agent.evaluate(chess.Board())
    original = first.copy()
    first[:] = 999
    # Same planes with a different clock may share ONLY the network prediction.
    board = chess.Board()
    board.halfmove_clock = 100
    second, second_value = agent.evaluate(board)
    assert np.array_equal(second, original) and second_value == value
    assert agent.metrics.cache_hits == 1
    with pytest.raises(NeuralAgentError, match="already over"):
        agent.choose_move(board)
    other = chess.Board()
    other.push_uci("e2e4")
    agent.evaluate(other)
    assert len(agent._cache) == 1
    agent.evaluate(chess.Board())
    assert agent.metrics.network_positions == 2  # metrics reset by terminal choose_move
    new_agent = ExperimentalPUCT(checkpoint, arm="cache")
    new_agent.evaluate(chess.Board())
    assert new_agent.metrics.cache_hits == 0


def test_batch_one_matches_sequential_and_large_batches_account_exact_visits(
    checkpoint: Path,
) -> None:
    baseline = ExperimentalPUCT(checkpoint, simulations=17)
    singleton = ExperimentalPUCT(checkpoint, arm="batch", batch_size=1, simulations=17)
    assert baseline.choose_move(chess.Board()) == singleton.choose_move(chess.Board())
    assert baseline.root_visits == singleton.root_visits
    batch = ExperimentalPUCT(checkpoint, arm="batch", batch_size=8, simulations=17)
    move = batch.choose_move(chess.Board())
    assert move in chess.Board().legal_moves
    assert sum(batch.root_visits.values()) == batch.metrics.simulations == 17
    assert batch.metrics.network_positions == 18
    assert batch.metrics.network_batches < batch.metrics.network_positions


def test_batch_rollback_removes_virtual_loss_on_error(checkpoint: Path, monkeypatch) -> None:
    agent = ExperimentalPUCT(checkpoint, arm="batch")
    root = _Node()
    agent._expand(chess.Board(), root)

    def fail(_):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(agent, "_forward", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        agent._wave(chess.Board(), root, 8)
    assert root.visits == 0
    assert all(e.visits == 0 and e.value_sum == 0 for e in root.edges.values())


def test_tactical_extension_recaptures_and_does_not_stand_pat_in_check(
    checkpoint: Path, monkeypatch
) -> None:
    agent = ExperimentalPUCT(checkpoint, arm="tactical", tactical_budget=3)
    board = chess.Board("7k/8/8/8/8/8/7r/7K w - - 0 1")
    before = board.fen()
    # The current checked side is assigned an artificially optimistic static value.
    monkeypatch.setattr(agent, "evaluate", lambda b: (np.zeros(4208), 0.9 if b.turn else 0.6))
    result = agent._quiet_value(board, 1, [3], static=0.9)
    assert result < 0.9
    assert agent.metrics.tactical_nodes <= 3
    assert board.fen() == before
    assert agent.metrics.tactical_nodes > 0


def test_tactical_root_takes_proven_mate_without_neural_work(checkpoint: Path) -> None:
    for case in read_cases(SUITE):
        if case.get("criterion") != "mate_in_one":
            continue
        board = chess.Board(case["fen"])
        agent = ExperimentalPUCT(checkpoint, arm="tactical", simulations=1)
        before = board.fen()
        move = agent.choose_move(board)
        assert board.fen() == before
        board.push(move)
        assert board.is_checkmate()
        assert agent.metrics.proven_mate_in_one
        assert agent.metrics.network_positions == agent.metrics.simulations == 0


def test_reuse_requires_the_same_full_history(checkpoint: Path) -> None:
    agent = ExperimentalPUCT(checkpoint, arm="reuse", simulations=64)
    board = chess.Board()
    move = agent.choose_move(board)
    board.push(move)
    previous_root = agent._previous[2]
    action = agent._move_encoder.encode(move)
    expected = previous_root.edges[action].child
    assert expected is not None
    assert agent._root(board) is expected
    # Identical FEN without its original move history must not inherit visits.
    assert agent._root(chess.Board(board.fen())) is not expected
    before = expected.visits
    agent.choose_move(board)
    assert agent.metrics.reused_visits == before
    agent.clear_state()
    assert agent._previous is None


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_preserves_board_and_visit_budget(checkpoint: Path, arm: str) -> None:
    agent = ExperimentalPUCT(checkpoint, arm=arm, simulations=5, tactical_budget=3)
    board = chess.Board()
    board.push_uci("e2e4")
    before = board.copy(stack=True)
    assert agent.choose_move(board) in board.legal_moves
    assert board == before and board.move_stack == before.move_stack
    assert sum(agent.root_visits.values()) == agent.metrics.simulations == 5


def test_deadline_is_soft_and_always_completes_one_traversal(checkpoint: Path) -> None:
    agent = ExperimentalPUCT(checkpoint, simulations=100, seconds=1e-12)
    assert agent.choose_move(chess.Board()) in chess.Board().legal_moves
    assert agent.metrics.simulations == 1
    assert agent.metrics.elapsed_seconds > agent.seconds


def test_benchmark_and_matches_resume_with_binding_and_checksums(
    checkpoint: Path, tmp_path: Path
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "format": "self-improving-chess-ai.search-experiments.suite",
                "version": 1,
                "cases": read_cases(SUITE)[:1],
            }
        )
    )
    kwargs = {
        "arms": ["baseline", "cache"],
        "device": "cpu",
        "simulations": 2,
        "seed": 31102041,
        "repeats": 1,
    }
    first = benchmark(checkpoint, suite, tmp_path / "bench", **kwargs)
    assert benchmark(checkpoint, suite, tmp_path / "bench", **kwargs) == first
    with pytest.raises(ValueError, match="binding changed"):
        benchmark(checkpoint, suite, tmp_path / "bench", **{**kwargs, "simulations": 3})
    matches = {
        "arm": "cache",
        "device": "cpu",
        "simulations": 2,
        "seed": 31102042,
        "openings": 1,
        "max_plies": 2,
    }
    report = paired_matches(checkpoint, tmp_path / "matches", **matches)
    assert report["games"] == 2
    assert report["unresolved_move_caps"] == 2
    assert paired_matches(checkpoint, tmp_path / "matches", **matches) == report


def test_invalid_search_options_fail_clearly(checkpoint: Path) -> None:
    for kwargs in (
        {"arm": "cache+batch"},
        {"cache_size": 0},
        {"seconds": float("nan")},
        {"batch_size": True},
        {"tactical_budget": -1},
    ):
        with pytest.raises(ValueError):
            ExperimentalPUCT(checkpoint, **kwargs)
