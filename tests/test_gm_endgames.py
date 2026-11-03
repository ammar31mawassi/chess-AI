"""GM imports protect games, transpositions, old holdouts and resumable labels."""

from __future__ import annotations

import runpy
import sqlite3
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import pytest
from test_endgame_pilot import pilot_setup as pilot_setup

from chess_ai.data.endgame_pilot import (
    FORMAT,
    _checked_json,
    _write_checked,
    finalize,
    generated_positions,
    label,
    load_prepared,
    orbit_keys,
    prepare,
)
from chess_ai.data.examples import load_dataset, save_dataset
from chess_ai.data.gm_endgames import IMPORT_FORMAT, build_plan, prepare_gm, validate_inputs
from chess_ai.data.stockfish_rebuild import _canonical, _game_digest
from chess_ai.data.value_targets import SHARDED_DATASET_FORMAT
from chess_ai.storage.artifacts import atomic_json, sha256_file


@pytest.fixture
def gm_setup(pilot_setup):
    base, _ = pilot_setup
    config = dict(base)
    directory = Path(config["run_dir"]).parent
    roots = list({r["group_id"]: r for r in generated_positions(777, 10)}.values())[:58]
    boards = [chess.Board(r["fen"]) for r in roots]
    boards += [boards[0].mirror(), boards[1].transform(chess.flip_horizontal)]
    games = []
    owners = {}
    for index, board in enumerate(boards):
        game = chess.pgn.Game()
        game.setup(board)
        owner = "player1" if index % 2 else "player2"
        game.headers.update(White=owner, Black="other", Result="1-0", TimeControl="300")
        node = game
        for _ in range(4):
            if board.is_game_over(claim_draw=True):
                break
            move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
            node = node.add_main_variation(move)
            board.push(move)
        games.append(game)
        owners[_game_digest(game)] = owner
    snapshot = directory / "gm_snapshot.pgn"
    snapshot.write_text("\n\n".join(str(g) for g in games), encoding="utf-8")
    coverage = directory / "gm_coverage.json"
    atomic_json(
        coverage,
        {
            "format": "self-improving-chess-ai.endgame-archive-coverage",
            "version": 2,
            "require_recorded_moves": True,
            "snapshot": {"path": str(snapshot), "sha256": sha256_file(snapshot)},
            "selection_ownership": dict(Counter(owners.values())),
            "selected_game_digests": list(owners),
            "selected_game_owners": owners,
            "selected_games": len(games),
        },
    )
    config.update(
        run_dir=str(directory / "gm_run"),
        seed=24102038,
        max_positions_per_game=3,
        protected_pilots=[],
        protected_rollout_roots=[],
        gm_import={
            "format": IMPORT_FORMAT,
            "version": 1,
            "coverage": str(coverage),
            "coverage_sha256": sha256_file(coverage),
            "snapshot": str(snapshot),
            "games_per_player": 30,
            "players": ["player1", "player2"],
        },
    )
    return config, games


def test_gm_grouping_caps_and_prelabel_splits(gm_setup):
    config, games = gm_setup
    plan = build_plan(config)
    assert build_plan(config) == plan
    rows = plan["rows"]
    assert set(r["split"] for r in rows) == {"train", "validation", "confirmation"}
    assert max(Counter(r["source_game"] for r in rows).values()) <= 3
    assert len({r["key"] for r in rows}) == len(rows)
    split_owners = {}
    groups = {}
    for row in rows:
        assert groups.setdefault(row["group_id"], row["split"]) == row["split"]
        for key in orbit_keys(chess.Board(row["fen"])):
            assert split_owners.setdefault(key, row["split"]) == row["split"]
    related = {_game_digest(g) for g in (games[0], games[-2])}
    assert len({r["group_id"] for r in rows if r["source_game"] in related}) == 1
    # No human result, played move or evaluation becomes a target at preparation.
    assert all("label" not in row and "value" not in row and "policy" not in row for row in rows)


def test_gm_history_and_unselected_mirror_evaluation_exclude_whole_groups(gm_setup):
    config, games = gm_setup
    before = build_plan(config)
    natural = next(r for r in before["rows"] if r["source_game"] == _game_digest(games[5]))
    catalog = Path(config["source_catalogs"][0])
    with sqlite3.connect(catalog) as db:
        db.execute(
            "INSERT INTO positions VALUES (?,?,?,?,?,?)",
            (
                natural["key"],
                natural["fen"],
                "validation",
                "old",
                "old_training",
                None,
            ),
        )
    # Use a mirror of an endgame input that was not picked by the per-game cap.
    selected_keys = {r["key"] for r in before["rows"]}
    target = None
    for game in games[10:]:
        board = game.board()
        for move in game.mainline_moves():
            if _canonical(board)[0] not in selected_keys:
                target = (game, board.copy())
                break
            board.push(move)
        if target:
            break
    assert target is not None
    game, board = target
    exclusion = chess.pgn.Game()
    exclusion.setup(board.mirror())
    exclusion.add_main_variation(next(iter(exclusion.board().legal_moves)))
    protected_path = Path(config["evaluation_roots"][0]) / "mirror.pgn"
    protected_path.write_text(str(exclusion), encoding="utf-8")
    prepare_gm(config, confirm_training_import=True)
    prepared = load_prepared(Path(config["run_dir"]))
    sources = {r["source_game"] for r in prepared["rows"]}
    assert natural["source_game"] not in sources
    assert _game_digest(game) not in sources
    assert prepared["excluded_evaluation_groups"] >= 1


def test_gm_confirmation_and_rollout_paths_are_protected(gm_setup):
    config, games = gm_setup
    directory = Path(config["run_dir"]).parent
    old = directory / "old_pilot"
    initial = games[4].board()
    key, fen = _canonical(initial)
    _write_checked(
        old / "positions.json",
        {
            "format": FORMAT,
            "version": 1,
            "config": config,
            "rows": [{"key": key, "fen": fen, "split": "confirmation"}],
        },
    )
    rollout = directory / "rollout"
    moves = [m.uci() for m in games[4].mainline_moves()]
    _write_checked(rollout / "case_000.json", {"case_key": key, "uci": moves})
    config["protected_pilots"] = [str(old)]
    config["protected_rollout_roots"] = [str(rollout)]
    rows = build_plan(config)["rows"]
    assert _game_digest(games[4]) not in {r["source_game"] for r in rows}


def test_gm_labels_resume_confirmation_not_exported_and_sources_bound(gm_setup):
    config, _ = gm_setup
    root = Path(config["run_dir"])
    with pytest.raises(PermissionError, match="confirm-training-import"):
        prepare_gm(config, confirm_training_import=False)
    report = prepare_gm(config, confirm_training_import=True)
    assert prepare_gm(config, confirm_training_import=True) == report
    validate_inputs(root)
    assert label(root, maximum=2)["labeled_this_invocation"] == 2
    with pytest.raises(ValueError, match="Finish all"):
        finalize(root)
    assert label(root)["remaining"] == 0
    finalize(root)
    dataset = load_dataset(root / "fresh.pt")
    confirmation = {
        r["key"]
        for r in _checked_json(root / "positions.json")["rows"]
        if r["split"] == "confirmation"
    }
    assert not confirmation & {e.metadata["position_key"] for e in dataset}
    assert {e.metadata["split"] for e in dataset} == {"train", "validation"}
    Path(config["gm_import"]["snapshot"]).write_text("edited")
    with pytest.raises(ValueError, match="input changed"):
        validate_inputs(root)
    with pytest.raises(ValueError, match="Preparation input changed"):
        prepare_gm(config, confirm_training_import=True)


def test_gm_rejects_coverage_edits_and_unknown_versions(gm_setup):
    config, _ = gm_setup
    config["gm_import"]["version"] = 99
    with pytest.raises(ValueError, match="Incompatible GM"):
        build_plan(config)
    config["gm_import"]["version"] = 1
    Path(config["gm_import"]["coverage"]).write_text("{}")
    with pytest.raises(ValueError, match="coverage changed"):
        build_plan(config)


def test_external_plan_rejects_a_different_config(gm_setup):
    config, _ = gm_setup
    plan = build_plan(config)
    plan["config"] = {**config, "seed": 99}
    with pytest.raises(ValueError, match="different settings"):
        prepare(config, external_plan=plan)


def test_gm_replay_receipt_recovers_without_replacing_tensors(gm_setup):
    config, _ = gm_setup
    root = Path(config["run_dir"])
    prepare_gm(config, confirm_training_import=True)
    label(root)
    finalize(root)
    data = load_dataset(root / "fresh.pt")
    small = root.parent / "small_replay"
    small.mkdir()
    selected = [
        next(e for e in data if e.metadata["split"] == split) for split in ("train", "validation")
    ]
    save_dataset(small / "fresh.pt", selected, metadata=data.metadata, format_version=2)
    old = Path(config["protected_datasets"][0])
    manifest = old.parent / "replay_manifest.json"
    atomic_json(
        manifest,
        {
            "format": SHARDED_DATASET_FORMAT,
            "version": 1,
            "examples": 2,
            "value_target_spec": data.metadata["value_target_spec"],
            "shards": [{"path": old.name, "sha256": sha256_file(old), "examples": 2}],
        },
    )
    compose = runpy.run_path(str(Path(__file__).parents[1] / "scripts/gm_endgame_training.py"))[
        "compose"
    ]
    report = compose(small, manifest, seed=123)
    assert report["fresh_by_split"] == report["replay_by_split"] == {"train": 1, "validation": 1}
    assert compose(small, manifest, seed=123) == report
    (small / "replay_receipt.json").unlink()
    assert compose(small, manifest, seed=123) == report
    assert sha256_file(small / "replay.pt") == report["sha256"]
    with pytest.raises(ValueError, match="binding changed"):
        compose(small, manifest, seed=124)
