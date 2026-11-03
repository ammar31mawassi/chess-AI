"""Descriptive slices and opening-clustered uncertainty for paired evaluations."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def result_counts(games: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    wins = sum(game["neural_outcome"] == "win" for game in games)
    draws = sum(game["neural_outcome"] == "draw" for game in games)
    return {
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": len(games) - wins - draws,
        "points": wins + 0.5 * draws,
    }


def result_slices(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    standard = [game for game in games if game["opening_index"] == 0]
    varied = [game for game in games if game["opening_index"] != 0]
    return {
        **result_counts(games),
        "standard_points": result_counts(standard)["points"],
        "standard": result_counts(standard),
        "varied": result_counts(varied),
        "by_color": {
            color: result_counts([game for game in games if game["neural_color"] == color])
            for color in ("white", "black")
        },
    }


def paired_uncertainty(
    candidate: Sequence[Mapping[str, Any]],
    champion: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Bootstrap whole unseen opening pairs, preserving dependence between colors.

    Standard-start games are a regression anchor, not random independent samples.
    This descriptive interval does not account for checkpoint-selection bias.
    """

    def pairs(games: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], float]:
        result: dict[tuple[int, str], float] = {}
        for game in games:
            key = (game["opening_index"], game["neural_color"])
            if key in result:
                raise ValueError("Duplicate paired audit opening/color")
            result[key] = {"win": 1.0, "draw": 0.5, "loss": 0.0}[game["neural_outcome"]]
        return result

    left, right = pairs(candidate), pairs(champion)
    if left.keys() != right.keys():
        raise ValueError("Candidate and baseline audit pairs do not match")
    indices = sorted({index for index, _ in left if index != 0})
    deltas = []
    for index in indices:
        if any((index, color) not in left for color in ("white", "black")):
            raise ValueError("Each opening requires both colors")
        deltas.append(sum(left[index, color] - right[index, color] for color in ("white", "black")))
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    means = (
        [sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(resamples)]
        if len(deltas) >= 2
        else []
    )
    return {
        "method": "paired_varied_opening_cluster_bootstrap_percentile",
        "seed": seed,
        "resamples": resamples,
        "varied_opening_pairs": len(deltas),
        "mean_point_delta_per_opening": sum(deltas) / len(deltas) if deltas else None,
        "approximate_95_percent_interval": np.quantile(means, [0.025, 0.975]).tolist()
        if means
        else None,
        "small_sample": len(deltas) < 30,
        "caution": "Descriptive only: repeated deterministic starts are correlated, checkpoint selection introduces bias, and multiple fresh seeds are required.",
    }
