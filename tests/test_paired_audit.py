"""Tests for leak-aware paired candidate/champion evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import chess

from chess_ai.arena import run_paired_audit
from chess_ai.arena.paired_audit import _excluded_opening_fens
from chess_ai.model import PolicyValueNet, save_checkpoint


def test_training_manifest_accepts_duplicate_but_valid_opening_fens(tmp_path: Path) -> None:
    fen = chess.Board().fen()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"pairs": [{"opening_fen": fen}, {"opening_fen": fen}]}),
        encoding="utf-8",
    )

    assert _excluded_opening_fens(manifest) == {fen}


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
    )

    assert summary.games_per_checkpoint == 4
    assert summary.candidate_wins + summary.candidate_draws + summary.candidate_losses == 4
    assert summary.champion_wins + summary.champion_draws + summary.champion_losses == 4
    report = json.loads(summary.report_path.read_text(encoding="utf-8"))
    assert report["training_updates"] == 0
    assert len(report["candidate_games"]) == len(report["champion_games"]) == 4
    assert [game["opening_index"] for game in report["candidate_games"]] == [
        game["opening_index"] for game in report["champion_games"]
    ]
