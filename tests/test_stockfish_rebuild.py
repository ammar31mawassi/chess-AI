"""End-to-end relabeling tests, including isolation, resume, and value semantics."""

from __future__ import annotations

import csv
import json
import os
import runpy
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from chess_ai.arena.paired_audit import run_paired_audit
from chess_ai.cli import build_parser
from chess_ai.data.examples import (
    IncompatibleDatasetError,
    TrainingExample,
    load_dataset,
    save_dataset,
)
from chess_ai.data.stockfish_rebuild import (
    RebuildConfig,
    _board_from_tensor,
    _split,
    finalize,
    inventory,
    label_positions,
    pilot,
)
from chess_ai.data.stockfish_teacher import StockfishConfig
from chess_ai.data.value_targets import STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model import PolicyValueNet
from chess_ai.model.checkpoint import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.training.trainer import Trainer, TrainingConfig, TrainingError


def _example(board: chess.Board, game_id: str, **metadata) -> TrainingExample:
    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
    policy[MoveEncoder().encode(next(iter(board.legal_moves)))] = 1
    return TrainingExample(
        BoardEncoder().encode(board),
        policy,
        -1,
        {"game_id": game_id, "fen": board.fen(), **metadata},
    )


@pytest.fixture
def rebuild_setup(tmp_path, teacher_setup):
    teacher, _ = teacher_setup
    config = RebuildConfig(
        tmp_path / "relabel",
        (tmp_path / "original.pt",),
        (tmp_path / "games",),
        teacher,
        validation_fraction=0.4,
        pilot_positions=4,
        shard_size=3,
    )
    ids = {
        split: next(f"game-{i}" for i in range(100) if _split(config, f"game-{i}") == split)
        for split in ("train", "validation")
    }
    examples = []
    config.pgn_roots[0].mkdir()
    for split, moves in (("train", "e4 e5 Nf3 Nc6 Bc4 Nf6"), ("validation", "d4 d5 c4 e6 Nc3 Nf6")):
        board = chess.Board()
        for san in moves.split():
            board.push_san(san)
        game = chess.pgn.Game.from_board(board)
        game.headers["GameID"] = ids[split]
        game.headers["Result"] = "0-1"
        (config.pgn_roots[0] / f"{split}.pgn").write_text(str(game), encoding="utf-8")
        first = chess.Board()
        first.push_san(moves.split()[0])
        examples.append(_example(first, ids[split]))
    save_dataset(config.datasets[0], examples)
    return config


def test_inventory_recovers_full_two_color_games_and_excludes_evaluations(rebuild_setup):
    config = rebuild_setup
    denied = chess.pgn.Game()
    denied.headers["TrainingData"] = "false"
    (config.pgn_roots[0] / "evaluation.pgn").write_text(str(denied), encoding="utf-8")
    report = inventory(config)
    assert report["unique_positions"] == 11  # six positions per game, common start once
    assert report["colors"] == {"black": 6, "white": 5}
    assert report["counts"]["excluded_evaluation_games"] == 1
    assert report["counts"]["dataset_positions"] == 2
    assert report["splits"]["train"] > 0
    assert report == inventory(config)
    with sqlite3.connect(config.run_dir / "catalog.sqlite") as db:
        assert not db.execute(
            "SELECT group_id FROM positions GROUP BY group_id HAVING count(DISTINCT split)>1"
        ).fetchall()


def test_labels_are_position_values_and_resume_without_reanalysis(rebuild_setup, monkeypatch):
    config = rebuild_setup
    inventory(config)
    first = label_positions(config, maximum=3)
    assert first["remaining"] == 8
    assert first["labeled_this_invocation"] == 3
    second = label_positions(config)
    assert second["labeled_this_invocation"] == 8
    assert second["complete"]
    assert label_positions(config)["labeled_this_invocation"] == 0
    manifest = finalize(config)
    assert manifest["examples"] == 11
    assert len(manifest["shards"]) == 4
    dataset = load_dataset(config.run_dir / "dataset.json")
    assert dataset.format_version == 2
    # The archived games were losses, but the mock teacher favors the side to move.
    assert all(item.target_value == pytest.approx(np.tanh(100 / 600)) for item in dataset)
    assert {item.metadata["player_to_move"] for item in dataset} == {"white", "black"}
    assert all(item.metadata["value_target_kind"] == STOCKFISH_VALUE_KIND for item in dataset)
    with pytest.raises(IncompatibleDatasetError, match="require dataset version 2"):
        save_dataset(config.run_dir / "downgraded.pt", dataset)
    assert finalize(config) == manifest
    manifest["shards"][0]["sha256"] = "corrupted"
    (config.run_dir / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IncompatibleDatasetError, match="hash mismatch"):
        load_dataset(config.run_dir / "dataset.json")


def test_pilot_counts_both_colors_and_reuses_labels(rebuild_setup):
    config = rebuild_setup
    inventory(config)
    result = pilot(config)
    assert result["positions"] == 4
    assert {item["color"] for item in result["samples"]} == {"white", "black"}
    assert result["estimated_serial_label_hours"] >= 0
    assert pilot(config) == result
    assert label_positions(config)["labeled_this_invocation"] == 7


def test_source_and_config_changes_fail_before_resume(rebuild_setup):
    config = rebuild_setup
    inventory(config)
    with pytest.raises(ValueError, match="differs"):
        label_positions(replace(config, seed=config.seed + 1))
    config.datasets[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        label_positions(config)


def test_finalize_refuses_incomplete_analysis(rebuild_setup):
    inventory(rebuild_setup)
    with pytest.raises(ValueError, match="Finish labeling"):
        finalize(rebuild_setup)


def test_tensor_reconstruction_preserves_en_passant_and_castling():
    board = chess.Board()
    board.push_san("e4")
    restored = _board_from_tensor(torch.from_numpy(BoardEncoder().encode(board)))
    assert restored.fen(en_passant="fen").split()[:4] == board.fen(en_passant="fen").split()[:4]
    tensor = torch.from_numpy(BoardEncoder().encode(board))
    tensor[0, 0, 0] = 1  # overlaps the black rook
    with pytest.raises(ValueError, match="overlap"):
        _board_from_tensor(tensor)


def test_csv_recovery_includes_only_previously_imported_games(tmp_path, teacher_setup):
    teacher, _ = teacher_setup
    csv_path = tmp_path / "games.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AN"])
        writer.writeheader()
        for moves in ("1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. a4 a5 2. Ra3 Ra6"):
            writer.writerow({"AN": moves})
    config = RebuildConfig(
        tmp_path / "relabel", (tmp_path / "source.pt",), (), teacher, validation_fraction=0.4
    )
    # Pick a seed which assigns the two existing imported games to different groups.
    config = next(
        replace(config, seed=seed)
        for seed in range(100)
        if _split(replace(config, seed=seed), "kaggle-lichess-1")
        != _split(replace(config, seed=seed), "kaggle-lichess-2")
    )
    examples = []
    for row, move in ((1, "e4"), (2, "d4")):
        board = chess.Board()
        board.push_san(move)
        examples.append(
            _example(board, f"kaggle-lichess-{row}", source="arevel/chess-games", source_row=row)
        )
    save_dataset(config.datasets[0], examples, metadata={"config": {"source_path": str(csv_path)}})
    result = inventory(config)
    assert result["unique_positions"] == 7
    assert result["counts"]["csv_positions"] == 8
    with sqlite3.connect(config.run_dir / "catalog.sqlite") as db:
        assert "kaggle-lichess-3" not in {
            row[0] for row in db.execute("SELECT DISTINCT game_id FROM positions")
        }


def test_audit_helper_excludes_selection_and_previous_seed_starts(tmp_path, monkeypatch):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/audit_stockfish_rebuild.py")
    )
    audit = script["audit_experiment"]
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "experiment"
    root.mkdir()
    champion, candidate, book = (
        root / name for name in ("champion.pt", "candidate.pt", "book.json")
    )
    for path in (champion, candidate, book):
        path.write_bytes(path.name.encode())
    selection_path = root / "selection.json"
    settings = {"report_path": str(selection_path), "search_simulations": 0, "opening_book": None}
    yaml_path = root / "select.yaml"
    yaml_path.write_text(json.dumps({"gameplay_selection": settings}), encoding="utf-8")
    selection_board = chess.Board()
    selection_board.push_san("Nc3")
    selection = {
        "selected_is_candidate": True,
        "selected_checkpoint": str(candidate),
        "champion": {"sha256": sha256_file(champion)},
        "candidates": [{"checkpoint": str(candidate), "sha256": sha256_file(candidate)}],
        "search_simulations": 0,
        "opening_book": None,
        "openings": [{"fen": selection_board.fen()}],
    }
    atomic_json(selection_path, selection)
    atomic_json(root / "training.json", {"pairs": []})
    experiment = {
        "format": "self-improving-chess-ai.stockfish-rebuild-experiment",
        "version": 1,
        "champion_checkpoint": str(champion),
        "champion_sha256": sha256_file(champion),
        "opening_book": str(book),
        "opening_book_sha256": sha256_file(book),
        "selection_configs": {"greedy": str(yaml_path)},
        "training_manifest": str(root / "training.json"),
    }
    atomic_json(root / "experiment.json", experiment)
    calls = []

    def fake_audit(*, candidate_checkpoint, champion_checkpoint, **kwargs):
        assert candidate_checkpoint == candidate
        assert champion_checkpoint == champion
        excluded = json.loads(kwargs["additional_exclude_manifests"][0].read_text())
        assert selection_board.fen() in {item["opening_fen"] for item in excluded["pairs"]}
        if kwargs["audit_seed"] == 24102036:
            board = chess.Board()
            board.push_san("e4")
            assert board.fen() in {item["opening_fen"] for item in excluded["pairs"]}
        board = chess.Board()
        board.push_san("e4" if kwargs["audit_seed"] == 23102036 else "d4")
        atomic_json(
            kwargs["pgn_dir"] / "report.json",
            {
                "candidate_sha256": sha256_file(candidate_checkpoint),
                "champion_sha256": sha256_file(champion_checkpoint),
                "audit_seed": kwargs["audit_seed"],
                "opponent_depth": kwargs["opponent_depth"],
                "search_simulations": kwargs["search_simulations"],
                "opening_book": None,
                "openings": [{"fen": board.fen()}],
            },
        )
        calls.append(kwargs)

    monkeypatch.setitem(
        audit.__globals__,
        "run_paired_audit",
        create_autospec(run_paired_audit, side_effect=fake_audit),
    )
    audit(root / "experiment.json", "greedy", "cpu")
    assert len(calls) == 4
    audit(root / "experiment.json", "greedy", "cpu")
    assert len(calls) == 4


def test_new_training_preserves_explicit_split_and_checkpoint_semantics(rebuild_setup):
    config = rebuild_setup
    inventory(config)
    label_positions(config)
    finalize(config)
    settings = TrainingConfig(
        dataset_path=config.run_dir / "dataset.json",
        checkpoint_dir=config.run_dir / "checkpoints",
        metrics_path=config.run_dir / "metrics.jsonl",
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=7,
        legal_policy_mask=True,
        use_dataset_split=True,
        value_target_spec=config.value_spec,
    )
    trainer = Trainer(PolicyValueNet(channels=2, residual_blocks=0), settings)
    dataset = load_dataset(settings.dataset_path)
    training, validation = trainer._prepare_examples(dataset, None)
    assert all(item.metadata["split"] == "train" for item in training)
    assert all(item.metadata["split"] == "validation" for item in validation)
    assert len(trainer.fit()) == 1
    checkpoint = load_checkpoint(settings.checkpoint_dir / "last.pt")
    assert checkpoint.metadata["format_version"] == 2
    assert checkpoint.metadata["value_target_spec"] == config.value_spec
    with pytest.raises(TrainingError, match="semantics"):
        Trainer(checkpoint.model, replace(settings, value_target_spec=None))._prepare_examples(
            dataset, None
        )
    with pytest.raises(TrainingError, match="semantics"):
        Trainer(
            checkpoint.model,
            replace(
                settings,
                resume_from=settings.checkpoint_dir / "last.pt",
                value_target_spec={**config.value_spec, "scale_cp": 400},
            ),
        )._resume_if_requested()


def test_unknown_v2_semantics_are_rejected(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, PolicyValueNet(channels=2, residual_blocks=0))
    raw = torch.load(path, weights_only=False)
    raw["format_version"] = 2
    torch.save(raw, path)
    with pytest.raises(IncompatibleCheckpointError, match="value_target_spec"):
        load_checkpoint(path)


def test_rebuild_and_position_evaluation_cli_options():
    args = build_parser().parse_args(
        [
            "stockfish-rebuild",
            "--config",
            "config.yaml",
            "--stage",
            "labels",
            "--max-positions",
            "20",
        ]
    )
    assert args.max_positions == 20
    assert args.stage == "labels"
    args = build_parser().parse_args(["evaluate-position", "--checkpoint", "new.pt"])
    assert args.fen == chess.STARTING_FEN


@pytest.mark.skipif(
    not os.environ.get("CHESS_AI_STOCKFISH"), reason="local UCI executable not configured"
)
def test_real_stockfish_rebuild_pipeline(tmp_path):
    teacher = StockfishConfig(
        Path(os.environ["CHESS_AI_STOCKFISH"]),
        os.environ["CHESS_AI_STOCKFISH_COMMIT"],
        nodes=3000,
        multipv=3,
    )
    config = RebuildConfig(
        tmp_path / "relabel",
        (tmp_path / "source.pt",),
        (),
        teacher,
        validation_fraction=0.4,
        pilot_positions=4,
    )
    ids = {
        split: next(f"group-{i}" for i in range(100) if _split(config, f"group-{i}") == split)
        for split in ("train", "validation")
    }
    examples = []
    for split, moves in (("train", ("e4", "e5")), ("validation", ("d4", "d5"))):
        board = chess.Board()
        for san in moves:
            board.push_san(san)
            examples.append(_example(board, ids[split]))
    save_dataset(config.datasets[0], examples)
    assert inventory(config)["unique_positions"] == 4
    assert pilot(config)["positions"] == 4
    assert label_positions(config)["remaining"] == 0
    finalize(config)
    dataset = load_dataset(config.run_dir / "dataset.json")
    for item in dataset:
        board = chess.Board(item.metadata["fen"])
        assert item.target_value == pytest.approx(item.metadata["stockfish_evaluation"])
        assert chess.Move.from_uci(item.metadata["teacher_move_uci"]) in board.legal_moves
