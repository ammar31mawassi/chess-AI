"""Append-only JSONL metrics and cumulative external benchmark reports."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlFormatError(ValueError):
    """Raised for a malformed line in an append-only JSONL file."""


JsonObject = dict[str, Any]


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _required_integer(data: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> Path:
    """Append one JSON object as one UTF-8 line."""

    if not isinstance(record, Mapping):
        raise TypeError("JSONL record must be a mapping")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        line = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record is not valid JSON data: {exc}") from exc
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
    return destination.resolve()


def read_jsonl(path: str | Path) -> tuple[JsonObject, ...]:
    """Read a JSONL file and report the exact bad line when validation fails."""

    source = Path(path)
    if not source.exists():
        return ()
    records: list[JsonObject] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                decoded = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JsonlFormatError(
                    f"Invalid JSON in {source} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(decoded, dict):
                raise JsonlFormatError(f"Expected a JSON object in {source} on line {line_number}")
            records.append(decoded)
    return tuple(records)


class JsonlStore:
    """A tiny append-only store suitable for metrics and experiment events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, object]) -> Path:
        return append_jsonl(self.path, record)

    def read(self) -> tuple[JsonObject, ...]:
        return read_jsonl(self.path)


class MetricsLogger(JsonlStore):
    """Adds a timestamp and record type to general training/evaluation metrics."""

    def log(self, metrics: Mapping[str, object], *, step: int | None = None) -> Path:
        record: dict[str, object] = {
            "schema_version": 1,
            "record_type": "metrics",
            "timestamp": datetime.now(UTC).isoformat(),
            **dict(metrics),
        }
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("step must be a non-negative integer")
            record["step"] = step
        return self.append(record)


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One human-mediated game against an external offline opponent."""

    external_opponent_name: str
    difficulty_level: str
    checkpoint: str
    color: str
    result: str
    move_count: int
    timestamp: str
    pgn_path: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.external_opponent_name.strip():
            raise ValueError("external_opponent_name cannot be empty")
        if not self.difficulty_level.strip():
            raise ValueError("difficulty_level cannot be empty")
        if not self.checkpoint.strip():
            raise ValueError("checkpoint cannot be empty")
        normalized_color = self.color.strip().lower()
        if normalized_color not in {"white", "black"}:
            raise ValueError("color must be 'white' or 'black'")
        object.__setattr__(self, "color", normalized_color)
        if self.result not in {"1-0", "0-1", "1/2-1/2", "*"}:
            raise ValueError("result must be 1-0, 0-1, 1/2-1/2, or *")
        if isinstance(self.move_count, bool) or not isinstance(self.move_count, int):
            raise TypeError("move_count must be an integer number of plies")
        if self.move_count < 0:
            raise ValueError("move_count cannot be negative")
        try:
            parsed = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        if self.schema_version != 1:
            raise ValueError(f"Unsupported benchmark schema version: {self.schema_version}")

    @property
    def ai_score(self) -> float | None:
        if self.result == "*":
            return None
        if self.result == "1/2-1/2":
            return 0.5
        ai_won = (self.result == "1-0") == (self.color == "white")
        return 1.0 if ai_won else 0.0

    def to_dict(self) -> dict[str, object]:
        return {"record_type": "external_benchmark", **asdict(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BenchmarkRecord:
        try:
            return cls(
                external_opponent_name=_required_string(data, "external_opponent_name"),
                difficulty_level=_required_string(data, "difficulty_level"),
                checkpoint=_required_string(data, "checkpoint"),
                color=_required_string(data, "color"),
                result=_required_string(data, "result"),
                move_count=_required_integer(data, "move_count"),
                timestamp=_required_string(data, "timestamp"),
                pgn_path=(
                    _required_string(data, "pgn_path") if data.get("pgn_path") is not None else None
                ),
                schema_version=_required_integer(data, "schema_version", default=1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JsonlFormatError(f"Invalid benchmark record: {exc}") from exc


def append_benchmark(path: str | Path, record: BenchmarkRecord) -> Path:
    return append_jsonl(path, record.to_dict())


def load_benchmarks(path: str | Path) -> tuple[BenchmarkRecord, ...]:
    records: list[BenchmarkRecord] = []
    for data in read_jsonl(path):
        if data.get("record_type", "external_benchmark") != "external_benchmark":
            continue
        records.append(BenchmarkRecord.from_dict(data))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    external_opponent_name: str
    difficulty_level: str
    checkpoint: str
    games: int
    wins: int
    draws: int
    losses: int
    unfinished: int
    points: float
    score_rate: float
    color: str | None = None


def summarize_benchmarks(
    records_or_path: Iterable[BenchmarkRecord] | str | Path,
    *,
    group_by_color: bool = False,
) -> tuple[BenchmarkSummary, ...]:
    """Aggregate results by external opponent level and checkpoint."""

    if isinstance(records_or_path, (str, Path)):
        records = load_benchmarks(records_or_path)
    else:
        records = tuple(records_or_path)

    grouped: dict[tuple[str, str, str, str | None], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        color = record.color if group_by_color else None
        key = (
            record.external_opponent_name,
            record.difficulty_level,
            record.checkpoint,
            color,
        )
        grouped[key].append(record)

    summaries: list[BenchmarkSummary] = []
    for key, group in sorted(grouped.items()):
        wins = sum(record.ai_score == 1.0 for record in group)
        draws = sum(record.ai_score == 0.5 for record in group)
        losses = sum(record.ai_score == 0.0 for record in group)
        unfinished = sum(record.ai_score is None for record in group)
        completed = wins + draws + losses
        points = wins + 0.5 * draws
        summaries.append(
            BenchmarkSummary(
                external_opponent_name=key[0],
                difficulty_level=key[1],
                checkpoint=key[2],
                color=key[3],
                games=completed,
                wins=wins,
                draws=draws,
                losses=losses,
                unfinished=unfinished,
                points=points,
                score_rate=points / completed if completed else 0.0,
            )
        )
    return tuple(summaries)


def format_benchmark_report(summaries: Iterable[BenchmarkSummary]) -> str:
    """Render a compact cumulative terminal report."""

    rows = tuple(summaries)
    if not rows:
        return "No external benchmark games have been recorded."
    header = "Opponent | Level | Checkpoint | W-D-L | Score | Unfinished"
    divider = "-" * len(header)
    lines = [header, divider]
    for row in rows:
        opponent = row.external_opponent_name
        if row.color is not None:
            opponent = f"{opponent} ({row.color})"
        lines.append(
            f"{opponent} | {row.difficulty_level} | {row.checkpoint} | "
            f"{row.wins}-{row.draws}-{row.losses} | {row.score_rate:.3f} | "
            f"{row.unfinished}"
        )
    lines.append("Scores are cumulative observations, not proof of model improvement.")
    return "\n".join(lines)
