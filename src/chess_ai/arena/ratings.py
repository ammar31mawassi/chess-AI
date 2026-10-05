"""Small, explicitly approximate Elo helpers for arena reports."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EloEstimate:
    """An Elo difference inferred only from the supplied match score."""

    rating_difference: float
    score_rate: float
    games: int
    method: str = "logistic score conversion (approximate)"


def approximate_elo(
    wins: int,
    draws: int,
    losses: int,
    *,
    max_abs: float = 800.0,
) -> float:
    """Estimate rating difference from results using the logistic Elo formula.

    Perfect scores would mathematically be infinite, so they are capped at
    ``max_abs``.  This number is descriptive, not an official rating and not a
    statistically reliable claim for small samples.
    """

    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (wins, draws, losses)
    ):
        raise TypeError("wins, draws, and losses must be integers")
    if wins < 0 or draws < 0 or losses < 0:
        raise ValueError("wins, draws, and losses cannot be negative")
    if max_abs <= 0:
        raise ValueError("max_abs must be positive")

    games = wins + draws + losses
    if games == 0:
        raise ValueError("at least one completed game is required")
    score_rate = (wins + 0.5 * draws) / games
    if score_rate <= 0.0:
        return -float(max_abs)
    if score_rate >= 1.0:
        return float(max_abs)
    raw = 400.0 * math.log10(score_rate / (1.0 - score_rate))
    return max(-float(max_abs), min(float(max_abs), raw))


def build_elo_estimate(wins: int, draws: int, losses: int) -> EloEstimate:
    games = wins + draws + losses
    rating = approximate_elo(wins, draws, losses)
    return EloEstimate(
        rating_difference=rating,
        score_rate=(wins + 0.5 * draws) / games,
        games=games,
    )


# Common discoverable names for the same deliberately approximate calculation.
estimate_elo = approximate_elo
estimate_elo_difference = approximate_elo
