"""Endgame leakage boundaries, resumability, and draw-rule diagnosis."""

from __future__ import annotations

import json
import runpy
import sqlite3
from dataclasses import asdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest

from chess_ai.arena.endgame_diagnostics import (
    evaluate_endgames,
    freeze_confirmation,
    review_move,
    summarize_games,
)
from chess_ai.data.endgame_pilot import (
    _checked_json,
    _write_checked,
    family,
    finalize,
    generated_positions,
    label,
    labeled_rows,
    orbit_keys,
    prepare,
    score_band,
)
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.stockfish_rebuild import _canonical, _label_row
from chess_ai.data.stockfish_teacher import StockfishTeacher
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import sha256_file

SPEC = {"kind": "stockfish_cp_tanh_v1", "perspective": "side_to_move", "scale_cp": 600.0}


def test_generated_families_are_valid_deterministic_and_disjoint():
    rows = generated_positions(123, 10)
    assert rows == generated_positions(123, 10)
    assert rows != generated_positions(124, 10)
    groups: dict[str, str] = {}
    owners: dict[str, str] = {}
    colors: dict[str, set[bool]] = {}
    for row in rows:
        board = chess.Board(row["fen"])
        assert board.is_valid() and not board.is_game_over(claim_draw=True)
        assert _canonical(board)[0] == row["key"]
        group = row["group_id"]
        assert groups.setdefault(group, row["split"]) == row["split"]
        colors.setdefault(group, set()).add(board.turn)
        for key in orbit_keys(board):
            assert owners.setdefault(key, group) == group
    assert all(turns == {True, False} for turns in colors.values())
    assert set(groups.values()) == {"train", "validation", "confirmation"}
    assert {row["family"] for row in rows} >= {"basic_mate", "rook", "minor", "pawn"}
    with pytest.raises(ValueError, match="multiple of 10"):
        generated_positions(123, 9)


@pytest.fixture
def pilot_setup(tmp_path, teacher_setup):
    teacher_config, _ = teacher_setup
    source = tmp_path / "catalog.sqlite"
    rows = generated_positions(654, 10)[:12]
    with StockfishTeacher(teacher_config) as teacher, sqlite3.connect(source) as db:
        db.execute(
            "CREATE TABLE positions(key TEXT PRIMARY KEY,fen TEXT,split TEXT,group_id TEXT,source TEXT,label TEXT)"
        )
        for index, row in enumerate({r["key"]: r for r in rows}.values()):
            db.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,?)",
                (
                    row["key"],
                    row["fen"],
                    "validation" if index == 0 else "train",
                    row["group_id"],
                    "training",
                    json.dumps(_label_row(teacher, row)),
                ),
            )
    generated = generated_positions(321, 10)
    held = next(r for r in generated if r["split"] == "validation")
    protected = next(r for r in generated if r["split"] == "train")
    excluded = next(r for r in generated if r["split"] == "confirmation")
    dataset = tmp_path / "protected.pt"
    examples = []
    for row, split in ((held, "train"), (protected, "validation")):
        board = chess.Board(row["fen"])
        policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
        policy[MoveEncoder().encode(next(iter(board.legal_moves)))] = 1
        examples.append(
            TrainingExample(
                BoardEncoder().encode(board),
                policy,
                0.0,
                {
                    "position_key": row["key"],
                    "fen": row["fen"],
                    "group_id": row["group_id"],
                    "game_id": row["group_id"],
                    "split": split,
                    "value_target_kind": SPEC["kind"],
                },
            )
        )
    save_dataset(
        dataset,
        examples,
        metadata={"value_target_spec": SPEC, "explicit_split": True},
        format_version=2,
    )
    eval_dir = tmp_path / "eval_games"
    eval_dir.mkdir()
    game = chess.pgn.Game()
    game.setup(chess.Board(excluded["fen"]))
    game.add_main_variation(next(iter(game.board().legal_moves)))
    game.headers["TrainingData"] = "false"
    (eval_dir / "protected.pgn").write_text(str(game), encoding="utf-8")
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"frozen test checkpoint")
    registry = tmp_path / "champion.json"
    registry.write_text("{}")
    teacher = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(teacher_config).items()
    }
    config = {
        "run_dir": str(tmp_path / "pilot"),
        "seed": 321,
        "champion": str(checkpoint),
        "champion_sha256": sha256_file(checkpoint),
        "champion_registry": str(registry),
        "source_catalogs": [str(source)],
        "protected_datasets": [str(dataset)],
        "evaluation_roots": [str(eval_dir)],
        "stockfish": teacher,
        "roots_per_material": 10,
        "max_pieces": 10,
        "natural_per_bucket": 4,
        "final_per_bucket": 4,
    }
    return config, {held["group_id"], protected["group_id"], excluded["group_id"]}


def test_preparation_protects_old_training_validation_and_eval(pilot_setup):
    config, forbidden = pilot_setup
    before = {
        p: sha256_file(Path(p))
        for p in [*config["source_catalogs"], *config["protected_datasets"], config["champion"]]
    }
    report = prepare(config)
    root = Path(config["run_dir"])
    prepared = _checked_json(root / "positions.json")
    assert not {r["group_id"] for r in prepared["rows"]} & forbidden
    assert set(report["counts"]) == {"train", "validation", "confirmation"}
    assert prepare(config) == report
    assert before == {p: sha256_file(Path(p)) for p in before}
    (Path(config["evaluation_roots"][0]) / "protected.pgn").write_text("changed")
    with pytest.raises(ValueError, match="PGN changed"):
        prepare(config)


def test_labels_resume_export_and_integrity(pilot_setup):
    config, _ = pilot_setup
    root = Path(config["run_dir"])
    prepare(config)
    partial = label(root, maximum=3)
    assert partial["labeled_this_invocation"] == 3 and partial["remaining"] > 0
    with pytest.raises(ValueError, match="Finish all"):
        finalize(root)
    assert label(root)["remaining"] == 0
    assert label(root)["labeled_this_invocation"] == 0
    report = finalize(root)
    assert finalize(root) == report
    (root / "finalized.json").unlink()
    assert finalize(root) == report
    data = load_dataset(root / "fresh.pt")
    assert data.metadata["value_target_spec"] == SPEC
    assert {r.metadata["split"] for r in data} == {"train", "validation"}
    prepared, _ = labeled_rows(root)
    confirm = {r["key"] for r in prepared["rows"] if r["split"] == "confirmation"}
    assert not confirm & {r.metadata["position_key"] for r in data}
    with sqlite3.connect(root / "labels.sqlite") as db:
        key, encoded = db.execute("SELECT key,payload FROM labels LIMIT 1").fetchone()
        payload = json.loads(encoded)
        payload["value"] = -0.9
        db.execute("UPDATE labels SET payload=? WHERE key=?", (json.dumps(payload), key))
    with pytest.raises(ValueError, match="integrity"):
        labeled_rows(root)


def test_resume_rejects_teacher_and_checkpoint_changes(pilot_setup):
    config, _ = pilot_setup
    root = Path(config["run_dir"])
    prepare(config)
    label(root, maximum=1)
    prepared = _checked_json(root / "positions.json")
    prepared["config"]["stockfish"]["nodes"] *= 2
    _write_checked(root / "positions.json", prepared)
    with pytest.raises(ValueError, match="binding changed"):
        label(root, maximum=1)
    Path(config["champion"]).write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="champion changed"):
        label(root, maximum=1)


def test_rule_draw_uses_full_history_and_move_caps_are_unresolved(teacher_setup):
    config, _ = teacher_setup
    # White's queen move stalemates; no separate rules engine or cp guess decides it.
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    move = chess.Move.from_uci("g6f5")
    assert move in board.legal_moves
    child = board.copy()
    child.push(move)
    assert child.is_stalemate()
    with StockfishTeacher(config) as teacher:
        review = review_move(teacher, board, move)
    assert review["effective_played_cp"] == 0
    assert "stalemate" in review["tags"]
    summary = summarize_games(
        [
            {
                "entry_band": "winning",
                "result": "*",
                "neural_result": "unresolved",
                "termination": "move_limit_unresolved",
                "moves": [],
            }
        ]
    )
    assert summary["entry_bands"]["winning"]["draws"] == 0
    assert summary["entry_bands"]["winning"]["unresolved"] == 1


def test_families_and_score_bands_are_descriptive():
    assert family(chess.Board("8/8/8/8/3k4/8/3P4/3K4 w - - 0 1")) == "pawn"
    assert [score_band(cp) for cp in (-301, -300, -299, 0, 299, 300)] == [
        "losing",
        "losing",
        "balanced",
        "balanced",
        "balanced",
        "winning",
    ]


def test_confirmation_locks_one_candidate_and_common_budget(tmp_path):
    binding = {
        "simulations": 32,
        "checkpoint": "base.pt",
        "checkpoint_sha256": "champion",
        "cases": ["one"],
    }
    freeze_confirmation(tmp_path / "pilot", binding, "champion")
    binding.update(checkpoint="candidate.pt", checkpoint_sha256="candidate1")
    freeze_confirmation(tmp_path / "pilot", binding, "champion")
    freeze_confirmation(tmp_path / "pilot", binding, "champion")
    with pytest.raises(ValueError, match="another candidate"):
        freeze_confirmation(
            tmp_path / "pilot", {**binding, "checkpoint_sha256": "candidate2"}, "champion"
        )
    with pytest.raises(ValueError, match="budget"):
        freeze_confirmation(tmp_path / "pilot", {**binding, "simulations": 64}, "champion")


def test_rollout_resume_preserves_moves_and_rejects_budget_change(pilot_setup, monkeypatch):
    config, _ = pilot_setup
    root = Path(config["run_dir"])
    prepare(config)
    label(root)

    class FirstLegal:
        calls = 0
        fail = True

        def __init__(self, *args, **kwargs):
            pass

        def choose_move(self, board):
            FirstLegal.calls += 1
            if FirstLegal.fail and FirstLegal.calls == 3:
                raise RuntimeError("simulated interruption")
            return sorted(board.legal_moves, key=lambda move: move.uci())[0]

    monkeypatch.setattr("chess_ai.arena.endgame_diagnostics.NeuralMCTSAgent", FirstLegal)
    monkeypatch.setattr("chess_ai.arena.endgame_diagnostics.MinimaxAgent", FirstLegal)
    output = root / "resumed"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluate_endgames(root, Path(config["champion"]), output, max_plies=3, cases_per_bucket=1)
    assert len(_checked_json(output / "case_000.json")["uci"]) == 2
    FirstLegal.fail = False
    resumed = evaluate_endgames(
        root, Path(config["champion"]), output, max_plies=3, cases_per_bucket=1
    )
    fresh = evaluate_endgames(
        root, Path(config["champion"]), root / "fresh_eval", max_plies=3, cases_per_bucket=1
    )
    assert resumed["summary"] == fresh["summary"]
    assert [g["uci"] for g in resumed["cases"]] == [g["uci"] for g in fresh["cases"]]
    with pytest.raises(ValueError, match="resume changed"):
        evaluate_endgames(root, Path(config["champion"]), output, max_plies=4, cases_per_bucket=1)


def test_multi_player_coverage_deduplicates_without_importing(tmp_path):
    profile = runpy.run_path(str(Path(__file__).parents[1] / "scripts/profile_endgame_pgns.py"))[
        "profile"
    ]
    game = chess.pgn.Game()
    game.setup(chess.Board("8/8/8/8/3k4/8/3P4/3K4 w - - 0 1"))
    game.add_main_variation(chess.Move.from_uci("d1c1"))
    game.headers.update(
        {"White": "player1", "Black": "player2", "Result": "1-0", "TimeControl": "300"}
    )
    sources = []
    for player in ("player1", "player2"):
        source = tmp_path / player
        source.mkdir()
        (source / "month.pgn").write_text(str(game), encoding="utf-8")
        sources.append(source)
    report = profile(sources, ["player1", "player2"], tmp_path / "coverage.json")
    assert report["counts"]["duplicate_movetext"] == 1
    assert report["selected_games"] == 1
    assert report["unique_endgame_positions"] == 1
    assert not report["training_imported"] and report["teacher_calls"] == 0


def test_gm_player_quotas_snapshot_and_shortfall(tmp_path):
    profile = runpy.run_path(str(Path(__file__).parents[1] / "scripts/profile_endgame_pgns.py"))[
        "profile"
    ]
    sources = []
    for player, moves in (
        ("player1", ["e2e4", "d2d4", "c2c4"]),
        ("player2", ["e2e3", "d2d3", "c2c3"]),
    ):
        source = tmp_path / player
        source.mkdir()
        games = []
        empty = chess.pgn.Game()
        empty.headers.update(
            {"White": player, "Black": "other", "Result": "1-0", "TimeControl": "300"}
        )
        games.append(str(empty))
        for move in moves:
            game = chess.pgn.Game()
            game.add_main_variation(chess.Move.from_uci(move))
            game.headers.update(
                {"White": player, "Black": "other", "Result": "1-0", "TimeControl": "300"}
            )
            games.append(str(game))
        (source / "month.pgn").write_text("\n\n".join(games), encoding="utf-8")
        sources.append(source)
    report = profile(
        sources,
        ["player1", "player2"],
        tmp_path / "coverage.json",
        maximum=2,
        maximum_per_player=1,
        snapshot_dir=tmp_path / "snapshot",
    )
    assert report["selection_ownership"] == {"player1": 1, "player2": 1}
    assert report["version"] == 2 and report["counts"]["excluded_empty_games"] == 2
    assert len(report["selected_game_owners"]) == report["snapshot"]["games"] == 2
    with Path(report["snapshot"]["path"]).open(encoding="utf-8") as handle:
        games = []
        while (game := chess.pgn.read_game(handle)) is not None:
            assert not game.errors
            games.append(game)
    assert len(games) == 2
    again = profile(
        sources,
        ["player1", "player2"],
        tmp_path / "same.json",
        maximum=2,
        maximum_per_player=1,
        snapshot_dir=tmp_path / "snapshot",
    )
    assert again["snapshot"] == report["snapshot"]
    with pytest.raises(ValueError, match="complete explicit per-player quotas"):
        profile(
            sources,
            ["player1", "player2"],
            tmp_path / "shortfall.json",
            maximum=8,
            maximum_per_player=4,
            snapshot_dir=tmp_path / "incomplete",
        )
    assert not (tmp_path / "incomplete").exists()
