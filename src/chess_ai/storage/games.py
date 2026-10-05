"""PGN persistence and explicit validation of external benchmark games."""

from __future__ import annotations

import io
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import chess
import chess.pgn


class PgnValidationError(ValueError):
    """Raised when a PGN cannot be trusted as a complete legal game."""


class ExplicitImportRequired(PermissionError):
    """Raised when benchmark data import was not explicitly acknowledged."""


@dataclass(frozen=True, slots=True)
class ValidatedExternalGame:
    """Metadata proven to be present in a parseable external-session PGN."""

    source_path: Path
    game_index: int
    opponent_label: str
    checkpoint: str
    ai_color: str
    result: str
    move_count: int
    game: chess.pgn.Game


@dataclass(frozen=True, slots=True)
class ExternalImportReport:
    """Files created by one user-approved external-game import."""

    imported_paths: tuple[Path, ...]
    manifest_path: Path
    game_count: int
    evaluation_origin: bool = True


def _pgn_text(game_or_text: chess.pgn.Game | str) -> str:
    if isinstance(game_or_text, chess.pgn.Game):
        text = str(game_or_text)
    elif isinstance(game_or_text, str):
        text = game_or_text
    else:
        raise TypeError("PGN must be a chess.pgn.Game or string")
    if not text.strip():
        raise ValueError("PGN text cannot be empty")
    return text.rstrip() + "\n"


def save_pgn(game_or_text: chess.pgn.Game | str, path: str | Path) -> Path:
    """Atomically write one or more PGN games and return the resolved path."""

    destination = Path(path)
    if destination.suffix.lower() != ".pgn":
        destination = destination.with_suffix(".pgn")
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = _pgn_text(game_or_text)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
    return destination.resolve()


def parse_pgn_text(text: str) -> tuple[chess.pgn.Game, ...]:
    """Parse every PGN in ``text`` and reject parser errors or empty input."""

    if not isinstance(text, str) or not text.strip():
        raise PgnValidationError("PGN text is empty")
    stream = io.StringIO(text)
    games: list[chess.pgn.Game] = []
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        if game.errors:
            messages = "; ".join(str(error) for error in game.errors)
            raise PgnValidationError(f"PGN contains invalid moves or syntax: {messages}")
        games.append(game)
    if not games:
        raise PgnValidationError("No PGN game was found")
    return tuple(games)


def load_pgn(path: str | Path) -> tuple[chess.pgn.Game, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PGN file does not exist: {source}")
    return parse_pgn_text(source.read_text(encoding="utf-8"))


class PgnStore:
    """Directory-backed convenience wrapper for uniquely named PGN files."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, game_or_text: chess.pgn.Game | str, filename: str | None = None) -> Path:
        if filename is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            filename = f"game_{stamp}_{uuid4().hex[:8]}.pgn"
        return save_pgn(game_or_text, self.directory / filename)

    def load(self, filename: str) -> tuple[chess.pgn.Game, ...]:
        return load_pgn(self.directory / filename)


def _header(game: chess.pgn.Game, *names: str) -> str:
    for name in names:
        value = game.headers.get(name)
        if value and value not in {"?", "????.??.??"}:
            return value.strip()
    return ""


def validate_external_pgn(
    path: str | Path,
    *,
    allow_unfinished: bool = False,
) -> tuple[ValidatedExternalGame, ...]:
    """Validate legality and required benchmark headers before any import.

    Accepted custom tags match those emitted by
    :class:`chess_ai.arena.external.ExternalOpponentSession`.
    """

    source = Path(path)
    games = load_pgn(source)
    validated: list[ValidatedExternalGame] = []
    valid_results = {"1-0", "0-1", "1/2-1/2"}
    if allow_unfinished:
        valid_results.add("*")

    for game_index, game in enumerate(games, start=1):
        date = _header(game, "Date")
        checkpoint = _header(game, "Checkpoint", "AIModel")
        opponent = _header(game, "Opponent", "OpponentLabel")
        ai_color = _header(game, "AIColor").lower()
        result = _header(game, "Result")

        missing: list[str] = []
        if not date:
            missing.append("Date")
        if not checkpoint:
            missing.append("Checkpoint/AIModel")
        if not opponent:
            missing.append("Opponent")
        if not ai_color:
            missing.append("AIColor")
        if not result:
            missing.append("Result")
        if missing:
            joined = ", ".join(missing)
            raise PgnValidationError(f"Game {game_index} is missing required headers: {joined}")
        if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", date):
            raise PgnValidationError(f"Game {game_index} has invalid Date header: {date!r}")
        if ai_color not in {"white", "black"}:
            raise PgnValidationError(
                f"Game {game_index} AIColor must be White or Black, got {ai_color!r}"
            )
        if result not in valid_results:
            expected = ", ".join(sorted(valid_results))
            raise PgnValidationError(
                f"Game {game_index} result {result!r} is not importable; expected {expected}"
            )

        board = game.board()
        move_count = 0
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                raise PgnValidationError(
                    f"Game {game_index} contains illegal move {move.uci()} at ply {board.ply() + 1}"
                )
            board.push(move)
            move_count += 1
        official_outcome = board.outcome(claim_draw=True)
        if official_outcome is not None and result != official_outcome.result():
            raise PgnValidationError(
                f"Game {game_index} Result header {result!r} disagrees with the final position "
                f"({official_outcome.result()!r})"
            )

        validated.append(
            ValidatedExternalGame(
                source_path=source.resolve(),
                game_index=game_index,
                opponent_label=opponent,
                checkpoint=checkpoint,
                ai_color=ai_color,
                result=result,
                move_count=move_count,
                game=game,
            )
        )
    return tuple(validated)


def _unique_import_path(destination: Path, source_stem: str, game_index: int) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_stem).strip("._") or "external_game"
    suffix = f"_{game_index}" if game_index > 1 else ""
    candidate = destination / f"{stem}{suffix}.pgn"
    counter = 2
    while candidate.exists():
        candidate = destination / f"{stem}{suffix}_{counter}.pgn"
        counter += 1
    return candidate


def import_external_games(
    pgn_paths: Iterable[str | Path],
    *,
    destination_dir: str | Path,
    confirm_evaluation_data_import: bool,
) -> ExternalImportReport:
    """Explicitly copy validated benchmark PGNs into a separate import area.

    The required boolean is intentionally noisy: evaluation data must never be
    mixed into training by an implicit directory scan.  This function preserves
    ``evaluation_origin=true`` in a manifest; conversion into training examples
    remains a separate, deliberate pipeline step.
    """

    if confirm_evaluation_data_import is not True:
        raise ExplicitImportRequired(
            "External games are evaluation data. Set confirm_evaluation_data_import=True "
            "only after reviewing the PGNs."
        )

    sources = tuple(Path(path) for path in pgn_paths)
    if not sources:
        raise ValueError("At least one external PGN path is required")

    # Validate the full batch first so one bad game cannot produce a partial import.
    validated = tuple(game for source in sources for game in validate_external_pgn(source))
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    imported: list[Path] = []
    manifest_records: list[dict[str, object]] = []
    imported_at = datetime.now(UTC).isoformat()
    for external_game in validated:
        output_path = _unique_import_path(
            destination,
            external_game.source_path.stem,
            external_game.game_index,
        )
        saved = save_pgn(external_game.game, output_path)
        imported.append(saved)
        manifest_records.append(
            {
                "schema_version": 1,
                "record_type": "external_game_import",
                "source_pgn": external_game.source_path.name,
                "imported_pgn": saved.name,
                "imported_at": imported_at,
                "evaluation_origin": True,
                "explicitly_approved": True,
                "opponent_label": external_game.opponent_label,
                "checkpoint": external_game.checkpoint,
                "ai_color": external_game.ai_color,
                "result": external_game.result,
                "move_count": external_game.move_count,
            }
        )

    from chess_ai.storage.metrics import append_jsonl

    manifest_path = destination / "import_manifest.jsonl"
    for record in manifest_records:
        append_jsonl(manifest_path, record)
    return ExternalImportReport(
        imported_paths=tuple(imported),
        manifest_path=manifest_path.resolve(),
        game_count=len(imported),
    )
