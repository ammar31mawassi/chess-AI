"""Tests for gameplay-first epoch selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from chess_ai.arena.gameplay_selection import (
    GameplaySelectionConfig,
    _candidate_eligibility,
    _choose_best_candidate,
    run_gameplay_selection,
)
from chess_ai.model import PolicyValueNet, save_checkpoint


def _config(tmp_path: Path) -> GameplaySelectionConfig:
    return GameplaySelectionConfig(
        champion_checkpoint=tmp_path / "champion.pt",
        candidate_checkpoint_dir=tmp_path / "candidates",
        training_manifest=tmp_path / "manifest.json",
        report_path=tmp_path / "evaluation" / "report.json",
        pgn_dir=tmp_path / "evaluation",
        opponent_depths=(1, 2),
        openings=2,
        max_plies=4,
        opening_min_full_moves=0,
        opening_max_full_moves=0,
        seed=73,
        device="cpu",
    )


def _selection_record(
    *, epoch: int, d1: float, d2: float, validation_loss: float
) -> dict[str, Any]:
    return {
        "checkpoint": f"epoch_{epoch:04d}.pt",
        "epoch": epoch,
        "aggregate_points": d1 + d2,
        "validation_loss": validation_loss,
        "by_opponent_depth": {
            "d1": {"points": d1, "standard_points": 1.0},
            "d2": {"points": d2, "standard_points": 1.0},
        },
        "eligible": True,
    }


def test_candidate_must_not_trade_away_one_opponent_depth(tmp_path: Path) -> None:
    config = _config(tmp_path)
    champion = _selection_record(epoch=0, d1=4.0, d2=4.0, validation_loss=9.0)
    candidate = _selection_record(epoch=1, d1=7.0, d2=3.5, validation_loss=1.0)

    eligible, reasons = _candidate_eligibility(candidate, champion, config=config)

    assert eligible is False
    assert "total score regressed against D2" in reasons


def test_gameplay_score_outranks_validation_loss() -> None:
    stronger = _selection_record(epoch=1, d1=6.0, d2=5.0, validation_loss=3.0)
    lower_loss = _selection_record(epoch=2, d1=5.0, d2=5.0, validation_loss=1.0)

    selected = _choose_best_candidate([lower_loss, stronger])

    assert selected is stronger


def test_selection_evaluates_all_epochs_without_promoting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    model = PolicyValueNet(channels=4, residual_blocks=0)
    save_checkpoint(config.champion_checkpoint, model, epoch=0)
    save_checkpoint(
        config.candidate_checkpoint_dir / "epoch_0001.pt",
        model,
        epoch=1,
        metrics={"validation_loss": 3.0},
    )
    save_checkpoint(
        config.candidate_checkpoint_dir / "epoch_0002.pt",
        model,
        epoch=2,
        metrics={"validation_loss": 1.0},
    )
    config.training_manifest.write_text('{"pairs": []}', encoding="utf-8")

    monkeypatch.setattr(
        "chess_ai.arena.gameplay_selection._opening_suite",
        lambda **_kwargs: [(chess.Board(), [], 73), (chess.Board(), [], 74)],
    )
    monkeypatch.setattr(
        "chess_ai.arena.gameplay_selection._excluded_opening_fens",
        lambda _path: set(),
    )

    def fake_results(
        checkpoint: Path,
        *,
        opponent_depth: int,
        **_kwargs: Any,
    ) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
        if checkpoint.name == "champion.pt":
            points = 4.0
        elif checkpoint.name == "epoch_0001.pt":
            points = 5.0 if opponent_depth == 1 else 4.0
        else:
            points = 6.0 if opponent_depth == 1 else 3.0
        return {
            "wins": int(points),
            "draws": 0,
            "losses": 4 - int(min(points, 4.0)),
            "points": points,
            "standard_points": 1.0,
        }, []

    monkeypatch.setattr(
        "chess_ai.arena.gameplay_selection._checkpoint_results",
        fake_results,
    )

    summary = run_gameplay_selection(config)

    assert summary.selected_checkpoint.name == "epoch_0001.pt"
    assert summary.selected_epoch == 1
    assert summary.selected_is_candidate is True
    report = json.loads(config.report_path.read_text(encoding="utf-8"))
    assert report["promotion_performed"] is False
    assert len(report["candidates"]) == 2
    assert report["candidates"][1]["eligible"] is False
