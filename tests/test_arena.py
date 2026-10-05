from __future__ import annotations

from collections.abc import Iterator

import chess

from chess_ai.arena import (
    approximate_elo,
    candidate_champion_report,
    run_match,
    run_tournament,
)


class FirstLegalAgent:
    def __init__(self, seed: int = 0, name: str = "first") -> None:
        self.seed = seed
        self.name = name

    def choose_move(self, board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))


class ScriptedAgent:
    def __init__(self, moves: list[str], name: str) -> None:
        self.name = name
        self._moves: Iterator[str] = iter(moves)

    def choose_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(next(self._moves))


def test_match_records_timing_and_reloadable_pgn(tmp_path) -> None:
    output = tmp_path / "short_game.pgn"
    result = run_match(
        FirstLegalAgent(name="White"),
        FirstLegalAgent(name="Black"),
        max_plies=4,
        seed=7,
        pgn_path=output,
    )

    assert result.result == "1/2-1/2"
    assert result.termination == "move_limit"
    assert result.plies == 4
    assert all(move.elapsed_seconds >= 0.0 for move in result.moves)
    assert result.pgn_path == output.resolve()

    with output.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    assert game is not None
    assert game.headers["Result"] == "1/2-1/2"
    assert len(list(game.mainline_moves())) == 4


def test_illegal_agent_move_is_an_explicit_forfeit() -> None:
    result = run_match(
        ScriptedAgent(["e7e5"], "Broken"),
        FirstLegalAgent(name="Legal"),
        max_plies=2,
    )

    assert result.result == "0-1"
    assert result.termination == "illegal_agent_move"
    assert result.illegal_agent == "Broken"
    assert result.illegal_move == "e7e5"
    assert result.plies == 0


def test_tournament_switches_colors_and_uses_fixed_per_game_seeds() -> None:
    a_seeds: list[int] = []
    b_seeds: list[int] = []

    def make_a(seed: int) -> FirstLegalAgent:
        a_seeds.append(seed)
        return FirstLegalAgent(seed, "A")

    def make_b(seed: int) -> FirstLegalAgent:
        b_seeds.append(seed)
        return FirstLegalAgent(seed, "B")

    tournament = run_tournament(make_a, make_b, games=4, seed=20, max_plies=2)

    assert a_seeds == [20, 22, 24, 26]
    assert b_seeds == [21, 23, 25, 27]
    assert [game.white_name for game in tournament.matches] == ["A", "B", "A", "B"]
    assert tournament.agent_a.draws == 4
    assert tournament.agent_b.draws == 4
    assert tournament.approximate_elo_difference == 0.0


def test_approximate_elo_and_candidate_report_do_not_promote() -> None:
    assert approximate_elo(5, 0, 5) == 0.0
    assert approximate_elo(10, 0, 0) == 800.0
    assert approximate_elo(0, 0, 10) == -800.0

    tournament = run_tournament(
        FirstLegalAgent(name="Candidate"),
        FirstLegalAgent(name="Champion"),
        games=2,
        max_plies=2,
    )
    report = candidate_champion_report(tournament)
    assert report.appears_stronger is False
    assert report.promotion_performed is False
    assert report.to_dict()["promotion_performed"] is False
