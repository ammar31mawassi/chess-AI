"""Full-corpus reuse, protected groups, source integrity and stage recovery."""

from __future__ import annotations

import json
import runpy
import sqlite3
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
from test_endgame_pilot import pilot_setup as pilot_setup
from test_gm_endgames import gm_setup as gm_setup

from chess_ai.data import full_corpus
from chess_ai.data.endgame_pilot import FORMAT as PILOT_FORMAT
from chess_ai.data.endgame_pilot import _write_checked
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.full_corpus import FORMAT, SPEC, _source_position, finalize, inventory, labels
from chess_ai.data.stockfish_rebuild import _canonical, _game_digest
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import atomic_json, sha256_file


@pytest.fixture
def corpus_setup(gm_setup):
    old_config, games = gm_setup
    directory = Path(old_config["run_dir"]).parent
    original = json.loads(Path(old_config["gm_import"]["coverage"]).read_text())
    base_games = games[:2]
    base_snapshot = directory / "base.pgn"
    base_snapshot.write_text("\n\n".join(str(g) for g in base_games), encoding="utf-8")
    base = directory / "base_coverage.json"
    owners = {
        _game_digest(g): original["selected_game_owners"][_game_digest(g)] for g in base_games
    }
    atomic_json(
        base,
        {
            "version": 2,
            "selected_game_digests": list(owners),
            "selected_game_owners": owners,
            "snapshot": {"path": str(base_snapshot), "sha256": sha256_file(base_snapshot)},
        },
    )
    expanded = {
        **original,
        "version": 3,
        "base_coverage": {"path": str(base), "sha256": sha256_file(base)},
        "added_ownership": {"player1": 29, "player2": 29},
    }
    coverage = directory / "expanded.json"
    atomic_json(coverage, expanded)
    reference = load_dataset(Path(old_config["protected_datasets"][0]))
    board = chess.Board()
    board.push_san("a3")
    board.push_san("h6")
    key, fen = _canonical(board)
    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
    policy[MoveEncoder().encode(next(iter(board.legal_moves)))] = 1
    extra = TrainingExample(
        BoardEncoder().encode(board),
        policy,
        0.2,
        {
            "fen": fen,
            "position_key": key,
            "group_id": "general_train",
            "game_id": "general_train",
            "split": "train",
            "value_target_kind": SPEC["kind"],
        },
    )
    source = directory / "original_labels.pt"
    save_dataset(
        source,
        [*reference, extra],
        metadata={"value_target_spec": SPEC, "explicit_split": True},
        format_version=2,
    )
    config = {
        "format": FORMAT,
        "version": 1,
        "run_dir": str(directory / "full_corpus"),
        "seed": 31102039,
        "champion": old_config["champion"],
        "champion_sha256": old_config["champion_sha256"],
        "champion_registry": old_config["champion_registry"],
        "coverage": str(coverage),
        "expected_ownership": {"player1": 30, "player2": 30},
        "expected_additions": {"player1": 29, "player2": 29},
        "games_per_player": 30,
        "stockfish": old_config["stockfish"],
        "sources": [{"name": "original", "path": str(source)}],
        "protected_pilots": [],
        "rollouts": [],
        "evaluation_roots": old_config["evaluation_roots"],
        "shard_size": 32,
        "sampling_shares": {"general": 0.65, "endgame": 0.30, "basic_mate": 0.05},
    }
    root = Path(config["run_dir"])
    (root / "configs").mkdir(parents=True)
    import yaml

    for arm in ("scratch", "finetune"):
        train = {
            "seed": 31102039,
            "device": "cpu",
            "model": {"channels": 8, "residual_blocks": 0},
            "training": {
                "dataset_path": str(root / "dataset.json"),
                "checkpoint_dir": str(root / arm),
                "metrics_path": str(root / f"{arm}.jsonl"),
                "epochs": 1,
                "batch_size": 8,
                "use_dataset_split": True,
                "value_target_spec": SPEC,
                "freeze_batch_norm_statistics": arm == "finetune",
            },
        }
        (root / "configs" / f"train_{arm}.yaml").write_text(yaml.safe_dump(train), encoding="utf-8")
    return config, games, key


def test_full_corpus_labels_are_reused_and_pending_labels_resume(corpus_setup, monkeypatch):
    config, _, reference_key = corpus_setup
    with pytest.raises(PermissionError, match="confirm-training-import"):
        inventory(config)
    report = inventory(config, confirm_training_import=True)
    assert report == inventory(config, confirm_training_import=True)
    assert report["pending_labels"] > 0
    root = Path(config["run_dir"])
    with sqlite3.connect(root / "catalog.sqlite") as db:
        assert db.execute(
            "SELECT ref_path,label FROM positions WHERE key=?", (reference_key,)
        ).fetchone() == (config["sources"][0]["path"], None)
        assert not db.execute(
            "SELECT group_id FROM positions GROUP BY group_id HAVING count(DISTINCT split)>1"
        ).fetchall()
    first = labels(config, maximum=2)
    assert first["labeled_this_invocation"] == 2
    with pytest.raises(ValueError, match="Finish pending"):
        finalize(config)
    assert labels(config)["remaining"] == 0
    assert labels(config)["labeled_this_invocation"] == 0
    manifest = finalize(config)
    assert finalize(config) == manifest
    data = load_dataset(root / "dataset.json")
    assert len(data) == report["examples"]
    assert len({e.metadata["position_key"] for e in data}) == len(data)
    original = next(e for e in data if e.metadata["position_key"] == reference_key)
    assert original.target_value == pytest.approx(0.2)
    assert original.metadata["label_dataset"] == config["sources"][0]["path"]
    ready = json.loads((root / "training_ready.json").read_text())
    weights = full_corpus.load_config(root / "configs/ready_scratch.yaml")["training"][
        "source_sampling_multipliers"
    ]
    mass = {key: weights[key] * count for key, count in ready["training_bucket_counts"].items()}
    assert mass == pytest.approx(config["sampling_shares"])
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/full_gm_rebuild.py"))
    commands = []
    monkeypatch.setattr(
        script["subprocess"], "run", lambda command, **kwargs: commands.append(command)
    )
    script["train"](root, config, "scratch", resume=False, device="cpu")
    assert "--init-checkpoint" not in commands[-1] and "--resume" not in commands[-1]
    script["train"](root, config, "finetune", resume=False, device="cpu")
    assert commands[-1][-2:] == ["--init-checkpoint", config["champion"]]
    script["train"](root, config, "scratch", resume=True, device="cpu")
    assert "--resume" in commands[-1] and "--init-checkpoint" not in commands[-1]
    manifest["examples"] += 1
    atomic_json(root / "dataset.json", manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        script["train"](root, config, "scratch", resume=False, device="cpu")


def test_full_corpus_preserves_whole_gm_confirmation_games(corpus_setup):
    config, games, _ = corpus_setup
    root = Path(config["run_dir"])
    held = games[4]
    board = held.board()
    key, fen = _canonical(board)
    old = root.parent / "reserved_pilot"
    _write_checked(
        old / "positions.json",
        {
            "format": PILOT_FORMAT,
            "version": 1,
            "config": config,
            "rows": [
                {"key": key, "fen": fen, "split": "confirmation", "source_game": _game_digest(held)}
            ],
        },
    )
    config["protected_pilots"] = [str(old)]
    inventory(config, confirm_training_import=True)
    split_report = json.loads((root / "gm_game_splits.json").read_text())["games"]
    assert split_report[_game_digest(held)]["split"] == "confirmation"
    keys = set()
    for move in held.mainline_moves():
        keys.add(_canonical(board)[0])
        board.push(move)
    with sqlite3.connect(root / "catalog.sqlite") as db:
        assert not keys & {row[0] for row in db.execute("SELECT key FROM positions")}
        assert all(row[0] != "confirmation" for row in db.execute("SELECT split FROM positions"))


def test_full_corpus_resume_after_uncommitted_game(corpus_setup, monkeypatch):
    config, _, _ = corpus_setup
    mark = full_corpus._mark
    tripped = False

    def interrupt(db, name):
        nonlocal tripped
        if name.startswith("gm:") and not tripped:
            tripped = True
            raise RuntimeError("interrupted unit")
        mark(db, name)

    monkeypatch.setattr(full_corpus, "_mark", interrupt)
    with pytest.raises(RuntimeError, match="interrupted unit"):
        inventory(config, confirm_training_import=True)
    monkeypatch.setattr(full_corpus, "_mark", mark)
    resumed = inventory(config, confirm_training_import=True)
    fresh_config = {
        **config,
        "run_dir": str(Path(config["run_dir"]).with_name("fresh_full_corpus")),
    }
    fresh = inventory(fresh_config, confirm_training_import=True)
    assert fresh["catalog_signature"] == resumed["catalog_signature"]
    assert fresh["examples"] == resumed["examples"]


def test_full_corpus_source_and_label_integrity(corpus_setup):
    config, _, _ = corpus_setup
    inventory(config, confirm_training_import=True)
    labels(config)
    root = Path(config["run_dir"])
    with sqlite3.connect(root / "catalog.sqlite") as db:
        key, payload = db.execute(
            "SELECT key,label FROM positions WHERE label IS NOT NULL LIMIT 1"
        ).fetchone()
        raw = json.loads(payload)
        raw["value"] = -0.9
        db.execute("UPDATE positions SET label=? WHERE key=?", (json.dumps(raw), key))
    with pytest.raises(ValueError, match="checksum mismatch"):
        finalize(config)
    Path(config["sources"][0]["path"]).write_bytes(b"edited")
    with pytest.raises(ValueError, match="input changed"):
        labels(config)


def test_full_corpus_rejects_an_orphaned_catalog(corpus_setup):
    config, _, _ = corpus_setup
    (Path(config["run_dir"]) / "catalog.sqlite").write_bytes(b"orphaned progress")
    with pytest.raises(ValueError, match="without its frozen plan"):
        inventory(config, confirm_training_import=True)


def test_inventory_never_writes_into_a_protected_rollout(corpus_setup):
    config, games, _ = corpus_setup
    root = Path(config["run_dir"])
    pilot = root.parent / "protected_pilot"
    board = games[2].board()
    key, fen = _canonical(board)
    _write_checked(
        pilot / "positions.json",
        {
            "format": PILOT_FORMAT,
            "version": 1,
            "config": config,
            "rows": [{"key": key, "fen": fen, "split": "train"}],
        },
    )
    folder = root.parent / "protected_rollout"
    case = folder / "case_000.json"
    _write_checked(case, {"case_key": key, "uci": [m.uci() for m in games[2].mainline_moves()]})
    config["protected_pilots"] = [str(pilot)]
    config["rollouts"] = [{"folder": str(folder), "pilot": str(pilot)}]
    protected = [case, pilot / "positions.json", Path(config["sources"][0]["path"])]
    before = {p: sha256_file(p) for p in protected}
    inventory(config, confirm_training_import=True)
    assert before == {p: sha256_file(p) for p in protected}
    assert (root / "plan.json").is_file()


def _lossy_ep_example(moves="c4"):
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    key, _ = _canonical(board)
    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
    policy[MoveEncoder().encode(next(iter(board.legal_moves)))] = 1
    fields = board.fen().split()
    fields[3] = "-"
    return TrainingExample(
        BoardEncoder().encode(board),
        policy,
        0.25,
        {
            "fen": " ".join(fields),
            "position_key": key,
            "split": "train",
            "group_id": "legacy_ep",
            "game_id": "legacy_ep",
            "value_target_kind": SPEC["kind"],
        },
    )


def test_legacy_inactive_ep_restores_the_tensor_identity():
    row = _lossy_ep_example()
    board, key, fen, repaired = _source_position(
        row,
        {"format": "self-improving-chess-ai.stockfish-offline-batch", "version": 2},
        Path("legacy.pt"),
        1,
    )
    assert repaired and board.ep_square == chess.C3
    assert not board.has_legal_en_passant()
    assert fen.split()[3] == "c3" and key == row.metadata["position_key"]
    assert np.array_equal(BoardEncoder().encode(board), row.board_tensor)
    assert row.metadata["fen"].split()[3] == "-"


@pytest.mark.parametrize("case", ["legal_ep", "bad_key", "piece_change", "unknown_producer"])
def test_legacy_ep_adapter_does_not_relax_other_mismatches(case):
    row = _lossy_ep_example("a4 h6 a5 b5" if case == "legal_ep" else "c4")
    producer = {"format": "self-improving-chess-ai.stockfish-offline-batch", "version": 2}
    if case == "bad_key":
        row.metadata = {**row.metadata, "position_key": "0" * 64}
    elif case == "piece_change":
        fields = str(row.metadata["fen"]).split()
        changed = chess.Board(str(row.metadata["fen"]))
        changed.remove_piece_at(chess.A7)
        fields[0] = changed.board_fen()
        row.metadata = {**row.metadata, "fen": " ".join(fields)}
    elif case == "unknown_producer":
        producer["version"] = 99
    with pytest.raises(ValueError, match=r"legacy.pt, row 7"):
        _source_position(row, producer, Path("legacy.pt"), 7)


def test_inventory_repair_resumes_and_exports_original_label(corpus_setup):
    config, _, _ = corpus_setup
    root = Path(config["run_dir"])
    row = _lossy_ep_example()
    source = root.parent / "legacy_ep.pt"
    save_dataset(
        source,
        [row],
        metadata={
            "value_target_spec": SPEC,
            "format": "self-improving-chess-ai.stockfish-offline-batch",
            "version": 2,
        },
        format_version=2,
    )
    config["sources"].append({"name": "legacy_refinement", "path": str(source)})
    before = sha256_file(source)
    report = inventory(config, confirm_training_import=True)
    assert report["source_fen_repairs"] == 1
    assert inventory(config, confirm_training_import=True) == report
    labels(config)
    finalize(config)
    data = load_dataset(root / "dataset.json")
    repaired = next(e for e in data if e.metadata["position_key"] == row.metadata["position_key"])
    assert repaired.metadata["fen"].split()[3] == "c3"
    assert repaired.metadata["source_fen"] == row.metadata["fen"]
    assert repaired.metadata["source_fen_adjustment"] == "inactive_en_passant_restored_v1"
    assert np.array_equal(repaired.board_tensor, row.board_tensor)
    assert np.array_equal(repaired.target_policy, row.target_policy)
    assert repaired.target_value == 0.25
    assert sha256_file(source) == before


def test_gm_expansion_pins_original_choices_and_ownership(tmp_path):
    profile = runpy.run_path(str(Path(__file__).parents[1] / "scripts/profile_endgame_pgns.py"))[
        "profile"
    ]
    source = tmp_path / "raw"
    source.mkdir()
    games = []
    for name, moves in (
        ("player1", ["e2e4", "d2d4", "c2c4"]),
        ("player2", ["e2e3", "d2d3", "c2c3"]),
    ):
        for move in moves:
            game = chess.pgn.Game()
            game.headers.update(White=name, Black="other", Result="1-0", TimeControl="300")
            game.add_main_variation(chess.Move.from_uci(move))
            games.append(game)
    (source / "month.pgn").write_text("\n\n".join(str(g) for g in games), encoding="utf-8")
    base_path = tmp_path / "base.json"
    base = profile(
        [source],
        ["player1", "player2"],
        base_path,
        maximum=2,
        maximum_per_player=1,
        snapshot_dir=tmp_path / "base_snapshot",
    )
    expanded = profile(
        [source],
        ["player1", "player2"],
        tmp_path / "expanded.json",
        seed=999,
        maximum=4,
        maximum_per_player=2,
        snapshot_dir=tmp_path / "expanded_snapshot",
        base_coverage=base_path,
    )
    assert expanded["version"] == 3
    assert expanded["added_ownership"] == {"player1": 1, "player2": 1}
    assert set(base["selected_game_digests"]) < set(expanded["selected_game_digests"])
    assert all(
        expanded["selected_game_owners"][k] == v for k, v in base["selected_game_owners"].items()
    )
    Path(base["snapshot"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Original GM collection changed"):
        profile(
            [source],
            ["player1", "player2"],
            tmp_path / "bad.json",
            maximum=4,
            maximum_per_player=2,
            base_coverage=base_path,
        )
