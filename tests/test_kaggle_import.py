"""Tests for explicit local Kaggle dataset conversion."""

from __future__ import annotations

import csv
from pathlib import Path

import chess

from chess_ai.data import KaggleImportConfig, import_kaggle_dataset, load_dataset


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluation_import_uses_neutral_legal_policy_and_bounded_value(tmp_path: Path) -> None:
    source = tmp_path / "eval.csv"
    _write_csv(
        source,
        ["FEN", "Evaluation"],
        [
            {"FEN": chess.Board().fen(), "Evaluation": "+100"},
            {"FEN": chess.Board().fen(), "Evaluation": "#-2"},
        ],
    )
    config = KaggleImportConfig(
        kind="evaluations",
        source_path=source,
        output_path=tmp_path / "eval.pt",
        target_examples=2,
    )

    summary = import_kaggle_dataset(config)
    loaded = load_dataset(config.output_path)

    assert summary.examples == 2
    assert loaded.metadata["kind"] == "explicit_external_kaggle_import"
    assert all(item.metadata["target_kind"] == "value_only" for item in loaded)
    assert all((item.target_policy > 0).sum() == 20 for item in loaded)
    assert all(-1.0 <= item.target_value <= 1.0 for item in loaded)


def test_tactic_import_rejects_illegal_moves(tmp_path: Path) -> None:
    source = tmp_path / "tactics.csv"
    _write_csv(
        source,
        ["FEN", "Evaluation", "Move"],
        [
            {"FEN": chess.Board().fen(), "Evaluation": "+25", "Move": "e2e4"},
            {"FEN": chess.Board().fen(), "Evaluation": "+25", "Move": "e2e5"},
        ],
    )
    config = KaggleImportConfig(
        kind="tactics",
        source_path=source,
        output_path=tmp_path / "tactics.pt",
        target_examples=2,
    )

    summary = import_kaggle_dataset(config)
    loaded = load_dataset(config.output_path)

    assert summary.examples == 1
    assert summary.invalid_rows == 1
    assert loaded[0].metadata["move_uci"] == "e2e4"


def test_game_import_filters_rating_and_preserves_game_groups(tmp_path: Path) -> None:
    source = tmp_path / "games.csv"
    fields = ["Result", "WhiteElo", "BlackElo", "TimeControl", "Termination", "AN"]
    _write_csv(
        source,
        fields,
        [
            {
                "Result": "1-0",
                "WhiteElo": "2100",
                "BlackElo": "2200",
                "TimeControl": "600+5",
                "Termination": "Normal",
                "AN": "1. e4 e5 2. Nf3 Nc6 1-0",
            },
            {
                "Result": "0-1",
                "WhiteElo": "1200",
                "BlackElo": "2200",
                "TimeControl": "600+5",
                "Termination": "Normal",
                "AN": "1. e4 e5 0-1",
            },
        ],
    )
    config = KaggleImportConfig(
        kind="games",
        source_path=source,
        output_path=tmp_path / "games.pt",
        target_examples=4,
        positions_per_game=2,
    )

    summary = import_kaggle_dataset(config)
    loaded = load_dataset(config.output_path)

    assert summary.examples == 2
    assert summary.filtered_rows == 1
    assert len({item.game_id for item in loaded}) == 1
    assert all(item.metadata["source"] == "arevel/chess-games" for item in loaded)


def test_game_import_can_limit_examples_to_opening_plies(tmp_path: Path) -> None:
    source = tmp_path / "games.csv"
    fields = ["Result", "WhiteElo", "BlackElo", "TimeControl", "Termination", "AN"]
    _write_csv(
        source,
        fields,
        [
            {
                "Result": "1-0",
                "WhiteElo": "2300",
                "BlackElo": "2250",
                "TimeControl": "900+10",
                "Termination": "Normal",
                "AN": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0",
            }
        ],
    )
    config = KaggleImportConfig(
        kind="games",
        source_path=source,
        output_path=tmp_path / "openings.pt",
        target_examples=6,
        positions_per_game=6,
        maximum_ply=4,
    )

    summary = import_kaggle_dataset(config)
    loaded = load_dataset(config.output_path)

    assert summary.examples == 4
    assert {int(item.metadata["ply"]) for item in loaded} == {1, 2, 3, 4}
    assert loaded.metadata["config"]["maximum_ply"] == 4
