"""Opening-level uncertainty preserves pairing and does not manufacture sample size."""

from __future__ import annotations

import pytest

from chess_ai.arena.audit_statistics import paired_uncertainty, result_slices


def records(outcomes):
    return [
        {"opening_index": index, "neural_color": color, "neural_outcome": outcome}
        for index, pair in enumerate(outcomes)
        for color, outcome in zip(("white", "black"), pair, strict=True)
    ]


def test_slices_and_paired_bootstrap_are_seeded():
    candidate = records([("win", "loss"), ("win", "draw"), ("draw", "win")])
    champion = records([("draw", "draw"), ("loss", "draw"), ("draw", "loss")])
    slices = result_slices(candidate)
    assert slices["by_color"]["white"]["points"] == 2.5
    assert slices["standard"]["points"] == 1
    assert slices["varied"]["points"] == 3
    interval = paired_uncertainty(candidate, champion, seed=73)
    assert interval == paired_uncertainty(candidate, champion, seed=73)
    assert interval["varied_opening_pairs"] == 2
    assert interval["approximate_95_percent_interval"] == [1, 1]
    assert interval["small_sample"]


def test_single_varied_pair_cannot_estimate_uncertainty():
    games = records([("win", "draw"), ("win", "win")])
    assert paired_uncertainty(games, games, seed=73)["approximate_95_percent_interval"] is None
    with pytest.raises(ValueError, match="do not match"):
        paired_uncertainty(games, games[:-1], seed=73)
