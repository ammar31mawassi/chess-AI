"""Tests for leak-aware paired candidate/champion evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest

from chess_ai.arena import run_paired_audit
from chess_ai.arena.paired_audit import _direct_baseline_results, _excluded_opening_fens
from chess_ai.model import PolicyValueNet, save_checkpoint


def test_training_manifest_accepts_duplicate_but_valid_opening_fens(tmp_path: Path) -> None:
    fen = chess.Board().fen()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"pairs": [{"opening_fen": fen}, {"opening_fen": fen}]}),
        encoding="utf-8",
    )

    assert _excluded_opening_fens(manifest) == {fen}


def test_direct_baseline_color_is_the_opponents_actual_color(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "chess_ai.arena.paired_audit.NeuralAgent", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "chess_ai.arena.paired_audit.run_match",
        lambda *args, **kwargs: SimpleNamespace(
            result="1-0",
            is_draw=False,
            illegal_agent=None,
            termination="checkmate",
            pgn_path=kwargs["pgn_path"],
        ),
    )
    result = _direct_baseline_results(
        Path("candidate.pt"),
        Path("baseline.pt"),
        opening_suite=[(chess.Board(), [], 73)],
        max_plies=4,
        seed=73,
        device="cpu",
        pgn_dir=tmp_path,
        search_simulations=0,
        c_puct=1.5,
    )
    assert result["candidate_stats"]["by_color"]["white"]["points"] == 1
    assert result["candidate_stats"]["by_color"]["black"]["points"] == 0
    assert result["baseline_stats"]["by_color"]["white"]["points"] == 1
    assert result["baseline_stats"]["by_color"]["black"]["points"] == 0


def test_paired_audit_uses_identical_starts_and_never_promotes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    champion = tmp_path / "champion.pt"
    save_checkpoint(candidate, PolicyValueNet(channels=4, residual_blocks=0), epoch=1)
    save_checkpoint(champion, PolicyValueNet(channels=4, residual_blocks=0), epoch=1)

    summary = run_paired_audit(
        candidate_checkpoint=candidate,
        champion_checkpoint=champion,
        openings=2,
        audit_seed=101,
        opponent_depth=1,
        opening_min_full_moves=1,
        opening_max_full_moves=1,
        max_plies=4,
        device="cpu",
        pgn_dir=tmp_path / "audit",
        direct_baseline=True,
    )

    assert summary.games_per_checkpoint == 4
    assert summary.candidate_wins + summary.candidate_draws + summary.candidate_losses == 4
    assert summary.champion_wins + summary.champion_draws + summary.champion_losses == 4
    report = json.loads(summary.report_path.read_text(encoding="utf-8"))
    assert report["training_updates"] == 0
    assert report["direct_baseline"]["candidate_stats"]["games"] == 4
    assert len(report["direct_baseline"]["games"]) == 4
    assert report["candidate_stats"]["by_color"]["white"]["games"] == 2
    assert report["uncertainty"]["small_sample"] is True
    assert len(report["candidate_games"]) == len(report["champion_games"]) == 4
    assert [game["opening_index"] for game in report["candidate_games"]] == [
        game["opening_index"] for game in report["champion_games"]
    ]
