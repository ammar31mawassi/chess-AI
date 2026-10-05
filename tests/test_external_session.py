from __future__ import annotations

from collections.abc import Iterator

import chess
import pytest

from chess_ai.arena.external import ExternalOpponentSession, run_external_session
from chess_ai.storage import (
    ExplicitImportRequired,
    import_external_games,
    load_benchmarks,
    load_pgn,
    read_jsonl,
    validate_external_pgn,
)


class ScriptedAi:
    name = "Scripted AI"

    def __init__(self, moves: list[str]) -> None:
        self._moves: Iterator[str] = iter(moves)

    def choose_move(self, board: chess.Board) -> chess.Move:
        move = chess.Move.from_uci(next(self._moves))
        assert move in board.legal_moves
        return move


def _input_from(values: list[str]) -> tuple[Iterator[str], object]:
    iterator = iter(values)
    return iterator, lambda _prompt: next(iterator)


def test_external_session_rejects_bad_input_and_saves_required_records(tmp_path) -> None:
    _iterator, input_fn = _input_from(["not-a-move", "e7e5", "quit"])
    output: list[str] = []
    confirmations: list[str] = []

    def confirmer(prompt: str) -> str:
        confirmations.append(prompt)
        return "1/2-1/2"

    result = run_external_session(
        ScriptedAi(["e2e4", "g1f3"]),
        ai_color="white",
        opponent_label="The Chess Lv.100",
        difficulty_level="Level 1",
        checkpoint="checkpoints/best.pt",
        pgn_dir=tmp_path / "games",
        benchmark_path=tmp_path / "metrics" / "benchmarks.jsonl",
        input_fn=input_fn,  # type: ignore[arg-type]
        output_fn=output.append,
        result_confirmer=confirmer,
    )

    assert result.result == "1/2-1/2"
    assert result.move_count == 3
    assert confirmations
    assert any("Malformed move" in line for line in output)
    assert sum("\n" in line for line in output) >= 4  # initial board plus every move

    games = load_pgn(result.pgn_path)
    assert len(games) == 1
    game = games[0]
    assert game.headers["AIModel"] == "Scripted AI"
    assert game.headers["Checkpoint"] == "checkpoints/best.pt"
    assert game.headers["Opponent"] == "The Chess Lv.100"
    assert game.headers["AIColor"] == "White"
    assert game.headers["Result"] == "1/2-1/2"

    benchmark = load_benchmarks(result.benchmark_path)[0]
    assert benchmark.external_opponent_name == "The Chess Lv.100"
    assert benchmark.difficulty_level == "Level 1"
    assert benchmark.move_count == 3
    assert benchmark.result == "1/2-1/2"


def test_external_commands_show_fen_help_undo_resign(tmp_path) -> None:
    _iterator, input_fn = _input_from(["show", "fen", "help", "e2e4", "undo", "e2e4", "resign"])
    output: list[str] = []
    session = ExternalOpponentSession(
        ScriptedAi(["e7e5", "e7e5"]),
        ai_color=chess.BLACK,
        opponent_label="Offline engine",
        difficulty_level="Easy",
        checkpoint="model.pt",
        pgn_dir=tmp_path / "games",
        benchmark_path=tmp_path / "benchmarks.jsonl",
        input_fn=input_fn,  # type: ignore[arg-type]
        output_fn=output.append,
    )

    result = session.run()
    assert result.result == "0-1"  # external opponent (White) resigned to Black AI
    assert result.termination == "opponent_resignation"
    assert result.moves == ("e2e4", "e7e5")
    assert any(line.startswith("FEN: ") for line in output)
    assert any(line.startswith("Commands:") for line in output)
    assert any(line.startswith("Undid e7e5") for line in output)


def test_validated_import_remains_explicit_and_keeps_origin_manifest(tmp_path) -> None:
    _iterator, input_fn = _input_from(["e7e5", "quit"])
    session_result = run_external_session(
        ScriptedAi(["e2e4", "g1f3"]),
        ai_color="white",
        opponent_label="Offline engine",
        difficulty_level="1",
        checkpoint="candidate.pt",
        pgn_dir=tmp_path / "games",
        benchmark_path=tmp_path / "benchmarks.jsonl",
        input_fn=input_fn,  # type: ignore[arg-type]
        output_fn=lambda _line: None,
        result_confirmer=lambda _prompt: "1/2-1/2",
    )
    assert len(validate_external_pgn(session_result.pgn_path)) == 1

    with pytest.raises(ExplicitImportRequired):
        import_external_games(
            [session_result.pgn_path],
            destination_dir=tmp_path / "imports",
            confirm_evaluation_data_import=False,
        )

    report = import_external_games(
        [session_result.pgn_path],
        destination_dir=tmp_path / "imports",
        confirm_evaluation_data_import=True,
    )
    assert report.game_count == 1
    assert report.imported_paths[0].is_file()
    manifest = read_jsonl(report.manifest_path)
    assert manifest[0]["evaluation_origin"] is True
    assert manifest[0]["explicitly_approved"] is True
