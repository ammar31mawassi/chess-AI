"""Behavioral tests shared by human, random, and Minimax agents."""

from __future__ import annotations

import chess
import pytest

from chess_ai.agents import ChessAgent, HumanAgent, MinimaxAgent, NoLegalMovesError, RandomAgent


def test_agents_satisfy_the_runtime_protocol() -> None:
    assert isinstance(RandomAgent(seed=1), ChessAgent)
    assert isinstance(HumanAgent(input_fn=lambda _prompt: "e2e4"), ChessAgent)
    assert isinstance(MinimaxAgent(depth=1), ChessAgent)


def test_random_agent_is_legal_reproducible_and_non_mutating() -> None:
    board = chess.Board()
    original_fen = board.fen()
    first = RandomAgent(seed=123)
    second = RandomAgent(seed=123)

    first_sequence = [first.choose_move(board) for _ in range(20)]
    second_sequence = [second.choose_move(board) for _ in range(20)]

    assert first_sequence == second_sequence
    assert all(move in board.legal_moves for move in first_sequence)
    assert board.fen() == original_fen


def test_human_agent_retries_malformed_and_illegal_moves() -> None:
    responses = iter(["not a move", "e2e5", "E2E4"])
    messages: list[str] = []
    agent = HumanAgent(input_fn=lambda _prompt: next(responses), output_fn=messages.append)
    board = chess.Board()

    move = agent.choose_move(board)

    assert move == chess.Move.from_uci("e2e4")
    assert move in board.legal_moves
    assert board.move_stack == []
    assert any("UCI notation" in message for message in messages)
    assert any("not legal" in message for message in messages)


@pytest.mark.parametrize(
    "agent",
    [
        RandomAgent(seed=1),
        HumanAgent(input_fn=lambda _prompt: "a1a2"),
        MinimaxAgent(depth=1),
    ],
)
def test_agents_reject_positions_without_legal_moves(agent: ChessAgent) -> None:
    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert checkmate.is_checkmate()
    with pytest.raises(NoLegalMovesError, match="game is over"):
        agent.choose_move(checkmate)


def test_minimax_returns_a_legal_move_and_search_statistics() -> None:
    board = chess.Board()
    original_fen = board.fen()
    agent = MinimaxAgent(depth=2)

    move = agent.choose_move(board)

    assert move in board.legal_moves
    assert board.fen() == original_fen
    assert agent.node_count > 0
    assert agent.nodes_searched == agent.node_count
    assert agent.last_search_time_seconds >= 0.0
    assert agent.search_time_seconds == agent.last_search_time_seconds
    assert agent.last_score is not None


def test_minimax_exact_root_analysis_is_legal_and_non_mutating() -> None:
    board = chess.Board()
    original_fen = board.fen()
    agent = MinimaxAgent(depth=2)

    analysis = agent.analyze_moves(board)

    assert len(analysis) == board.legal_moves.count()
    assert {item.move for item in analysis} == set(board.legal_moves)
    assert agent.last_score == max(item.score for item in analysis)
    assert agent.nodes_searched > 0
    assert board.fen() == original_fen


def test_minimax_finds_mate_in_one() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    move = MinimaxAgent(depth=1).choose_move(board)

    board.push(move)

    assert board.is_checkmate()


def test_minimax_captures_a_hanging_queen() -> None:
    board = chess.Board("4k3/8/8/8/3q4/8/3R4/4K3 w - - 0 1")

    move = MinimaxAgent(depth=1).choose_move(board)

    assert move == chess.Move.from_uci("d2d4")


def test_evaluation_is_from_the_active_players_perspective() -> None:
    white_to_move = chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")
    black_to_move = chess.Board("4k3/8/8/8/8/8/Q7/4K3 b - - 0 1")
    agent = MinimaxAgent(depth=1)

    assert agent.evaluate(white_to_move) > 0
    assert agent.evaluate(black_to_move) == -agent.evaluate(white_to_move)


def test_deterministic_mode_breaks_equal_scores_consistently() -> None:
    board = chess.Board()

    first = MinimaxAgent(depth=1, deterministic=True).choose_move(board)
    second = MinimaxAgent(depth=1, deterministic=True).choose_move(board)

    assert first == second


def test_seed_makes_nondeterministic_mode_reproducible() -> None:
    board = chess.Board()

    first = MinimaxAgent(depth=1, deterministic=False, seed=99).choose_move(board)
    second = MinimaxAgent(depth=1, deterministic=False, seed=99).choose_move(board)

    assert first == second


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_minimax_rejects_invalid_depth(depth: object) -> None:
    with pytest.raises(ValueError, match="depth"):
        MinimaxAgent(depth=depth)  # type: ignore[arg-type]
