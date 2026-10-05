"""Batch matches, color alternation, and candidate/champion reports."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

from chess_ai.arena.match import ArenaAgent, MatchResult, reseed_agent, run_match
from chess_ai.arena.ratings import approximate_elo

AgentFactory: TypeAlias = Callable[[int], ArenaAgent]
AgentSource: TypeAlias = ArenaAgent | AgentFactory


@dataclass(frozen=True, slots=True)
class ScoreStats:
    games: int
    wins: int
    draws: int
    losses: int

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        return self.points / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "points": self.points, "score_rate": self.score_rate}


@dataclass(frozen=True, slots=True)
class TournamentResult:
    agent_a_name: str
    agent_b_name: str
    agent_a: ScoreStats
    agent_b: ScoreStats
    matches: tuple[MatchResult, ...]
    seed: int
    switched_colors: bool
    approximate_elo_difference: float

    @property
    def games(self) -> int:
        return len(self.matches)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_a": self.agent_a_name,
            "agent_b": self.agent_b_name,
            "games": self.games,
            "seed": self.seed,
            "switched_colors": self.switched_colors,
            "agent_a_stats": self.agent_a.to_dict(),
            "agent_b_stats": self.agent_b.to_dict(),
            "approximate_elo_difference": self.approximate_elo_difference,
            "elo_note": "Approximate logistic conversion; not an official rating.",
        }


@dataclass(frozen=True, slots=True)
class CandidateChampionReport:
    candidate_name: str
    champion_name: str
    candidate: ScoreStats
    champion: ScoreStats
    approximate_elo_difference: float
    appears_stronger: bool
    promotion_performed: bool
    recommendation: str
    tournament: TournamentResult

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate_name,
            "champion": self.champion_name,
            "candidate_stats": self.candidate.to_dict(),
            "champion_stats": self.champion.to_dict(),
            "approximate_elo_difference": self.approximate_elo_difference,
            "appears_stronger": self.appears_stronger,
            "promotion_performed": False,
            "recommendation": self.recommendation,
            "games": self.tournament.games,
            "seed": self.tournament.seed,
        }


def _materialize(source: AgentSource, seed: int) -> ArenaAgent:
    if hasattr(source, "choose_move"):
        agent = cast(ArenaAgent, source)
        reseed_agent(agent, seed)
        return agent
    agent = source(seed)
    if not hasattr(agent, "choose_move"):
        raise TypeError("agent factory must return an object with choose_move(board)")
    return agent


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "agent"


def _stats(wins: int, draws: int, losses: int) -> ScoreStats:
    return ScoreStats(games=wins + draws + losses, wins=wins, draws=draws, losses=losses)


def run_tournament(
    agent_a: AgentSource,
    agent_b: AgentSource,
    *,
    games: int,
    seed: int = 0,
    switch_colors: bool = True,
    max_plies: int = 512,
    starting_fen: str | None = None,
    claim_draw: bool = True,
    pgn_dir: str | Path | None = None,
) -> TournamentResult:
    """Play a deterministic-seeded batch, alternating colors by default.

    Passing ``Callable[[int], ChessAgent]`` factories gives the strongest
    reproducibility because each game receives fresh agents with fixed seeds.
    Reusable instances are also accepted and are reseeded when their API allows.
    """

    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("games must be a positive integer")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies <= 0:
        raise ValueError("max_plies must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(switch_colors, bool):
        raise TypeError("switch_colors must be a boolean")

    output_dir = Path(pgn_dir) if pgn_dir is not None else None
    matches: list[MatchResult] = []
    a_wins = a_draws = a_losses = 0
    a_name = "agent_a"
    b_name = "agent_b"

    for game_index in range(games):
        game_seed = seed + game_index * 2
        a = _materialize(agent_a, game_seed)
        b = _materialize(agent_b, game_seed + 1)
        a_name = str(getattr(a, "name", a.__class__.__name__))
        b_name = str(getattr(b, "name", b.__class__.__name__))
        a_is_white = not switch_colors or game_index % 2 == 0
        white, black = (a, b) if a_is_white else (b, a)

        pgn_path: Path | None = None
        if output_dir is not None:
            filename = (
                f"game_{game_index + 1:04d}_"
                f"{_safe_filename(str(getattr(white, 'name', 'white')))}_vs_"
                f"{_safe_filename(str(getattr(black, 'name', 'black')))}.pgn"
            )
            pgn_path = output_dir / filename

        match = run_match(
            white,
            black,
            max_plies=max_plies,
            starting_fen=starting_fen,
            seed=None,
            claim_draw=claim_draw,
            pgn_path=pgn_path,
            extra_headers={
                "TournamentSeed": str(seed),
                "AgentASeed": str(game_seed),
                "AgentBSeed": str(game_seed + 1),
                "GameNumber": str(game_index + 1),
            },
        )
        matches.append(match)

        if match.result == "1/2-1/2":
            a_draws += 1
        else:
            white_won = match.result == "1-0"
            a_won = white_won == a_is_white
            if a_won:
                a_wins += 1
            else:
                a_losses += 1

    a_stats = _stats(a_wins, a_draws, a_losses)
    b_stats = _stats(a_losses, a_draws, a_wins)
    return TournamentResult(
        agent_a_name=a_name,
        agent_b_name=b_name,
        agent_a=a_stats,
        agent_b=b_stats,
        matches=tuple(matches),
        seed=seed,
        switched_colors=switch_colors,
        approximate_elo_difference=approximate_elo(a_wins, a_draws, a_losses),
    )


def candidate_champion_report(
    tournament: TournamentResult,
    *,
    stronger_threshold: float = 0.5,
) -> CandidateChampionReport:
    """Interpret agent A as candidate and agent B as champion, without promotion."""

    if not 0.0 <= stronger_threshold <= 1.0:
        raise ValueError("stronger_threshold must be between 0 and 1")
    appears_stronger = tournament.agent_a.score_rate > stronger_threshold
    if appears_stronger:
        recommendation = (
            "Candidate scored above the configured threshold. Review the sample size and "
            "promote manually only after additional evaluation."
        )
    else:
        recommendation = (
            "Candidate did not score above the configured threshold; keep the current "
            "champion and gather more evidence."
        )
    return CandidateChampionReport(
        candidate_name=tournament.agent_a_name,
        champion_name=tournament.agent_b_name,
        candidate=tournament.agent_a,
        champion=tournament.agent_b,
        approximate_elo_difference=tournament.approximate_elo_difference,
        appears_stronger=appears_stronger,
        promotion_performed=False,
        recommendation=recommendation,
        tournament=tournament,
    )


def evaluate_candidate(
    candidate: AgentSource,
    champion: AgentSource,
    *,
    games: int,
    seed: int = 0,
    switch_colors: bool = True,
    max_plies: int = 512,
    starting_fen: str | None = None,
    claim_draw: bool = True,
    pgn_dir: str | Path | None = None,
    stronger_threshold: float = 0.5,
) -> CandidateChampionReport:
    tournament = run_tournament(
        candidate,
        champion,
        games=games,
        seed=seed,
        switch_colors=switch_colors,
        max_plies=max_plies,
        starting_fen=starting_fen,
        claim_draw=claim_draw,
        pgn_dir=pgn_dir,
    )
    return candidate_champion_report(tournament, stronger_threshold=stronger_threshold)
