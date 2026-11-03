"""Fair search/book settings and per-game runtime book reset."""

from __future__ import annotations

import json
from pathlib import Path

import chess

from chess_ai.agents.opening_book_agent import OpeningBookAgent
from chess_ai.arena.gameplay_selection import GameplaySelectionConfig, run_gameplay_selection
from chess_ai.arena.paired_audit import _checkpoint_results
from chess_ai.data.opening_book import OPENING_BOOK_FORMAT, position_key
from chess_ai.model import PolicyValueNet, save_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file


def _book(path: Path) -> None:
    atomic_json(
        path,
        {
            "format": OPENING_BOOK_FORMAT,
            "version": 1,
            "positions": {position_key(chess.Board()): [{"move_uci": "e2e4", "weight": 1}]},
        },
    )


def test_selection_passes_same_search_and_book_to_both_models(tmp_path, monkeypatch):
    champion = tmp_path / "champion.pt"
    candidates = tmp_path / "candidates"
    model = PolicyValueNet(channels=2, residual_blocks=0)
    save_checkpoint(champion, model)
    save_checkpoint(candidates / "best.pt", model, epoch=1, metrics={"validation_loss": 1.0})
    manifest = tmp_path / "training.json"
    atomic_json(manifest, {"pairs": []})
    book = tmp_path / "book.json"
    _book(book)
    config = GameplaySelectionConfig(
        champion,
        candidates,
        manifest,
        tmp_path / "report.json",
        tmp_path / "games",
        openings=2,
        opening_min_full_moves=1,
        opening_max_full_moves=1,
        max_plies=4,
        search_simulations=4,
        opening_book=book,
        device="cpu",
    )
    calls = []

    def fake_results(path, **kwargs):
        calls.append((path, kwargs))
        return {"points": 1.0, "standard_points": 0.5}, []

    monkeypatch.setattr("chess_ai.arena.gameplay_selection._checkpoint_results", fake_results)
    run_gameplay_selection(config)
    assert len(calls) == 4
    assert all(
        kwargs["search_simulations"] == 4 and kwargs["opening_book"] == book for _, kwargs in calls
    )
    report = json.loads(config.report_path.read_text())
    assert report["opening_book_sha256"] == sha256_file(book)
    assert report["search_simulations"] == 4
    assert report["promotion_performed"] is False


def test_runtime_book_is_reset_for_each_color_game(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, PolicyValueNet(channels=2, residual_blocks=0))
    book = tmp_path / "book.json"
    _book(book)
    instances = []

    class SpyBook(OpeningBookAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("chess_ai.arena.paired_audit.OpeningBookAgent", SpyBook)
    _, games = _checkpoint_results(
        checkpoint,
        role="candidate",
        opening_suite=[(chess.Board(), [], 5)],
        opponent_depth=1,
        max_plies=2,
        seed=5,
        device="cpu",
        pgn_dir=tmp_path / "games",
        search_simulations=2,
        opening_book=book,
    )
    assert len(instances) == 2 and instances[0] is not instances[1]
    assert len(games) == 2
