from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chess_ai.storage import (
    BenchmarkRecord,
    ExplicitImportRequired,
    JsonlFormatError,
    append_benchmark,
    append_jsonl,
    format_benchmark_report,
    import_external_games,
    load_benchmarks,
    read_jsonl,
    summarize_benchmarks,
)


def _benchmark(result: str, color: str = "white") -> BenchmarkRecord:
    return BenchmarkRecord(
        external_opponent_name="The Chess Lv.100",
        difficulty_level="Level 1",
        checkpoint="checkpoints/best.pt",
        color=color,
        result=result,
        move_count=12,
        timestamp=datetime.now(UTC).isoformat(),
    )


def test_jsonl_round_trip_and_line_validation(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"loss": 1.25, "epoch": 1})
    append_jsonl(path, {"loss": 0.75, "epoch": 2})
    assert read_jsonl(path) == (
        {"epoch": 1, "loss": 1.25},
        {"epoch": 2, "loss": 0.75},
    )

    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(JsonlFormatError, match="line 2"):
        read_jsonl(path)


def test_benchmarks_are_summarized_from_ai_perspective(tmp_path) -> None:
    path = tmp_path / "benchmarks.jsonl"
    append_benchmark(path, _benchmark("1-0", "white"))
    append_benchmark(path, _benchmark("1-0", "black"))
    append_benchmark(path, _benchmark("1/2-1/2", "white"))
    append_benchmark(path, _benchmark("*", "white"))

    loaded = load_benchmarks(path)
    summary = summarize_benchmarks(loaded)[0]
    assert summary.wins == 1
    assert summary.draws == 1
    assert summary.losses == 1
    assert summary.unfinished == 1
    assert summary.score_rate == 0.5
    report = format_benchmark_report([summary])
    assert "The Chess Lv.100" in report
    assert "not proof of model improvement" in report


def test_external_import_requires_explicit_acknowledgement(tmp_path) -> None:
    with pytest.raises(ExplicitImportRequired):
        import_external_games(
            [tmp_path / "missing.pgn"],
            destination_dir=tmp_path / "imports",
            confirm_evaluation_data_import=False,
        )
