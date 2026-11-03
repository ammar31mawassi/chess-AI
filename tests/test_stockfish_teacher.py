"""Ordinary teacher tests use a mock engine; real smoke is explicitly opt-in."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import chess
import chess.engine
import numpy as np
import pytest

from chess_ai.data.stockfish_teacher import (
    StockfishConfig,
    StockfishTeacher,
    StockfishTeacherError,
    convert_score,
    policy_from_rows,
)
from chess_ai.environment.move_encoder import MoveEncoder


@pytest.mark.parametrize(
    "score,turn,sign,mate",
    [
        (chess.engine.Cp(120), chess.WHITE, 1, None),
        (chess.engine.Cp(120), chess.BLACK, -1, None),
        (chess.engine.Mate(3), chess.WHITE, 1, 3),
        (chess.engine.Mate(3), chess.BLACK, -1, -3),
        (chess.engine.Mate(-2), chess.WHITE, -1, -2),
        (chess.engine.Mate(0), chess.WHITE, -1, 0),
        (chess.engine.MateGiven, chess.WHITE, 1, 0),
    ],
)
def test_score_conversion_and_perspective(teacher_setup, score, turn, sign, mate):
    config, _ = teacher_setup
    converted = convert_score(chess.engine.PovScore(score, chess.WHITE), turn, config)
    assert converted["score_cp"] * sign > 0
    assert converted["normalized_evaluation"] * sign > 0
    assert converted["mate"] == mate


def test_soft_policy_and_cold_hash(teacher_setup):
    config, engine = teacher_setup
    board = chess.Board()
    with StockfishTeacher(config) as teacher:
        policy, audit = teacher.analyze(board)
        assert np.isclose(policy.sum(), 1)
        legal = {MoveEncoder().encode(move) for move in board.legal_moves}
        assert all(value == 0 for index, value in enumerate(policy) if index not in legal)
        assert np.count_nonzero(policy) == 3
        assert sum(item["probability"] for item in audit["top_moves"]) == pytest.approx(1)
        assert audit["top_moves"][0]["score_cp"] == 100
        assert teacher.provenance["analysis"]["clear_hash"] is True
    assert {"Clear Hash": None} in [call.args[0] for call in engine.configure.call_args_list]
    engine.quit.assert_called_once()
    engine.close.assert_called_once()


@pytest.mark.parametrize(
    "case", ["illegal_root", "illegal_later", "missing_score", "duplicate", "bound", "missing_line"]
)
def test_invalid_multipv_is_rejected(teacher_setup, case):
    config, _ = teacher_setup
    config = replace(config, multipv=2)
    rows = [
        {
            "pv": [chess.Move.from_uci("e2e4")],
            "score": chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE),
        },
        {
            "pv": [chess.Move.from_uci("d2d4")],
            "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE),
        },
    ]
    if case == "illegal_root":
        rows[0]["pv"] = [chess.Move.from_uci("e2e5")]
    if case == "illegal_later":
        rows[0]["pv"].append(chess.Move.from_uci("d2d4"))
    if case == "missing_score":
        rows[0].pop("score")
    if case == "duplicate":
        rows[1]["pv"] = rows[0]["pv"]
    if case == "bound":
        rows[0]["lowerbound"] = True
    if case == "missing_line":
        rows.pop()
    with pytest.raises(StockfishTeacherError):
        policy_from_rows(chess.Board(), rows, config)


def test_cache_binds_exact_fen_binary_commit_and_all_settings(teacher_setup, monkeypatch):
    config, _ = teacher_setup
    with StockfishTeacher(config) as teacher:
        key = teacher.cache_signature(chess.Board().fen())
        original_policy, original_audit = teacher.analyze(chess.Board())
        monkeypatch.setattr(teacher, "_analyse", lambda board: pytest.fail("Expected cache hit"))
        policy, audit = teacher.analyze(chess.Board())
        np.testing.assert_array_equal(policy, original_policy)
        assert audit == original_audit
        changed_clock = chess.Board()
        changed_clock.halfmove_clock = 1
        assert teacher.cache_signature(changed_clock.fen()) != key
    for change in (
        {"source_commit": "b" * 40},
        {"nodes": 1000},
        {"temperature_cp": 20},
        {"threads": 2},
        {"hash_mb": 32},
        {"multipv": 2},
        {"timeout_seconds": 60},
        {"store_evaluation": False},
        {"mate_score_cp": 10000},
        {"evaluation_scale_cp": 100},
    ):
        with StockfishTeacher(replace(config, **change)) as teacher:
            assert teacher.cache_signature(chess.Board().fen()) != key
    config.executable.write_bytes(b"fake engine version 2")
    with StockfishTeacher(config) as teacher:
        assert teacher.cache_signature(chess.Board().fen()) != key


def test_corrupt_cache_fails_clearly(teacher_setup):
    config, _ = teacher_setup
    with StockfishTeacher(config) as teacher:
        teacher.analyze(chess.Board())
        path = next(config.cache_dir.glob("*.json"))
        raw = json.loads(path.read_text())
        raw["payload"]["top_moves"][0]["cp"] = 1
        path.write_text(json.dumps(raw))
        with pytest.raises(StockfishTeacherError, match="digest"):
            teacher.analyze(chess.Board())


@pytest.mark.parametrize("error", [TimeoutError(), chess.engine.EngineTerminatedError("crashed")])
def test_failure_closes_engine_and_never_caches(teacher_setup, monkeypatch, error):
    config, engine = teacher_setup

    def fail(board):
        raise error

    with StockfishTeacher(config) as teacher:
        monkeypatch.setattr(teacher, "_analyse", fail)
        with pytest.raises(StockfishTeacherError, match=r"timed out|crashed"):
            teacher.analyze(chess.Board())
    assert engine.close.called
    assert not config.cache_dir.exists()


def test_missing_binary_invalid_fen_and_startup_cleanup(teacher_setup):
    config, engine = teacher_setup
    with pytest.raises(StockfishTeacherError, match="does not exist"):
        StockfishTeacher(replace(config, executable=Path("absent-stockfish-test.exe")))
    with StockfishTeacher(config) as teacher, pytest.raises(StockfishTeacherError, match="invalid"):
        teacher.analyze(chess.Board.empty())
    engine.configure.side_effect = chess.engine.EngineError("bad options")
    with pytest.raises(StockfishTeacherError, match="startup"):
        StockfishTeacher(config)
    assert engine.close.call_count >= 2


def test_mate_targets_favor_fast_wins_and_delay_losses(teacher_setup):
    config, _ = teacher_setup

    def pov(score):
        return chess.engine.PovScore(score, chess.WHITE)

    assert (
        convert_score(pov(chess.engine.Mate(1)), chess.WHITE, config)["score_cp"]
        > convert_score(pov(chess.engine.Mate(5)), chess.WHITE, config)["score_cp"]
    )
    assert (
        convert_score(pov(chess.engine.Mate(-5)), chess.WHITE, config)["score_cp"]
        > convert_score(pov(chess.engine.Mate(-1)), chess.WHITE, config)["score_cp"]
    )


@pytest.mark.parametrize("distances", [(1, 5, None), (-1, -5, -9)])
def test_opt_in_mate_policy_is_decisive_and_preserves_values(teacher_setup, distances):
    config, _ = teacher_setup
    board = chess.Board()
    moves = sorted(board.legal_moves, key=lambda move: move.uci())[:3]
    rows = [
        {
            "pv": [move],
            "score": chess.engine.PovScore(
                chess.engine.Cp(900) if distance is None else chess.engine.Mate(distance),
                board.turn,
            ),
        }
        for move, distance in zip(moves, distances, strict=True)
    ]
    legacy, old_trace = policy_from_rows(board, rows, config)
    enabled = replace(config, mate_distance_temperature=2.0)
    policy, trace = policy_from_rows(board, rows, enabled)
    encoder = MoveEncoder()
    if distances[0] > 0:
        assert policy[encoder.encode(moves[0])] > 0.85
        assert policy[encoder.encode(moves[2])] == 0
        assert legacy[encoder.encode(moves[0])] < 0.55
    else:
        assert policy[encoder.encode(moves[2])] > 0.85
    assert [(r["score_cp"], r["normalized_evaluation"]) for r in trace] == [
        (r["score_cp"], r["normalized_evaluation"]) for r in old_trace
    ]
    assert "mate_distance_temperature" not in config.analysis_settings()
    assert enabled.analysis_settings()["mate_distance_temperature"] == 2.0
    with StockfishTeacher(config) as old, StockfishTeacher(enabled) as new:
        assert old.cache_signature(board.fen()) != new.cache_signature(board.fen())


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), True])
def test_invalid_mate_policy_temperature(teacher_setup, temperature):
    with pytest.raises(ValueError, match="mate_distance_temperature"):
        replace(teacher_setup[0], mate_distance_temperature=temperature)


@pytest.mark.parametrize(
    "fen,nodes,multipv",
    [
        (chess.STARTING_FEN, 1000, 3),
        # First uncached position of the interrupted 600-game collection:
        # Stockfish's final snapshot mixes depth 9 and depth 8 ranks.
        ("rn1qkb1r/p1ppnp1p/bp2p3/8/5pP1/1P3N2/PBPPP1BP/RN1QK2R w KQkq - 0 7", 20000, 5),
    ],
    ids=["starting-position", "interrupted-multipv-reorder"],
)
def test_optional_real_stockfish_smoke(fen, nodes, multipv):
    executable = os.environ.get("CHESS_AI_STOCKFISH")
    commit = os.environ.get("CHESS_AI_STOCKFISH_COMMIT")
    if not executable or not commit:
        pytest.skip("Set CHESS_AI_STOCKFISH and CHESS_AI_STOCKFISH_COMMIT for a local UCI smoke")
    config = StockfishConfig(Path(executable), commit, nodes=nodes, multipv=multipv)
    board = chess.Board(fen)
    with StockfishTeacher(config) as teacher:
        policy, audit = teacher.analyze(board)
        assert np.isclose(policy.sum(), 1)
        assert chess.Move.from_uci(audit["teacher_move_uci"]) in board.legal_moves
        assert len({item["move_uci"] for item in audit["top_moves"]}) == multipv
        assert len({item["depth"] for item in audit["top_moves"]}) == 1


def test_fixed_nodes_deadline_cancels_hung_protocol(tmp_path, monkeypatch):
    binary = tmp_path / "fake.exe"
    binary.write_bytes(b"fake")
    config = StockfishConfig(binary, "a" * 40, timeout_seconds=0.05)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    engine = Mock()
    engine.id = {"name": "Stockfish fake"}
    engine.options = {
        name: SimpleNamespace(default=None) for name in ("Threads", "Hash", "MultiPV", "Clear Hash")
    }
    calls = []

    async def hung(board, limit, **kwargs):
        calls.append((board, limit, kwargs))
        await asyncio.sleep(60)

    engine.protocol.loop = loop
    engine.protocol.analysis = hung
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: engine)
    try:
        start = time.monotonic()
        with (
            StockfishTeacher(config) as teacher,
            pytest.raises(StockfishTeacherError, match="timed out"),
        ):
            teacher.analyze(chess.Board())
        assert time.monotonic() - start < 2
        assert calls[0][1].nodes == config.nodes and calls[0][1].time is None
        engine.close.assert_called_once()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1)
        loop.close()


@pytest.mark.parametrize(
    "tail",
    [
        [(3, 1, "e2e4", 60, {}), (3, 2, "d2d4", 50, {"upperbound": True})],
        # Final rank 2 at depth 2 must not overwrite the previous snapshot's
        # rank 2 and duplicate its rank 1 after roots are reordered.
        [(3, 1, "d2d4", 60, {}), (2, 2, "e2e4", 10, {})],
        [(2, 1, "d2d4", 60, {})],
        [(2, 1, "d2d4", 60, {"lowerbound": True}), (2, 2, "e2e4", 50, {})],
        [(3, 1, "e2e4", 60, {}), (3, 2, "e2e4", 50, {})],
    ],
    ids=["bound", "mixed-depth-reorder", "partial-reorder", "bound-first", "duplicate-snapshot"],
)
def test_partial_node_iteration_keeps_last_complete_exact_scores(tmp_path, monkeypatch, tail):
    binary = tmp_path / "fake.exe"
    binary.write_bytes(b"fake")
    config = StockfishConfig(binary, "a" * 40, multipv=2)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    engine = Mock()
    engine.id = {"name": "Stockfish fake"}
    engine.options = {
        name: SimpleNamespace(default=None) for name in ("Threads", "Hash", "MultiPV", "Clear Hash")
    }

    def row(depth, rank, uci, cp, **extra):
        return {
            "depth": depth,
            "multipv": rank,
            "pv": [chess.Move.from_uci(uci)],
            "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
            **extra,
        }

    packets = [
        row(2, 1, "e2e4", 30),
        row(2, 2, "d2d4", 20),
        *(row(depth, rank, uci, cp, **extra) for depth, rank, uci, cp, extra in tail),
    ]

    class Stream:
        def __aiter__(self):
            async def generate():
                for packet in packets:
                    yield packet

            return generate()

        async def wait(self):
            pass

        def stop(self):
            pass

    async def analysis(*args, **kwargs):
        return Stream()

    engine.protocol.loop = loop
    engine.protocol.analysis = analysis
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: engine)
    try:
        with StockfishTeacher(config) as teacher:
            _, audit = teacher.analyze(chess.Board())
        assert [item["depth"] for item in audit["top_moves"]] == [2, 2]
        assert [item["cp"] for item in audit["top_moves"]] == [30, 20]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1)
        loop.close()
