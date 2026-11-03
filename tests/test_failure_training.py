from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_ai.arena.search_experiments import opening
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.failure_training import (
    SPEC,
    collect,
    finalize_failures,
    label_failures,
    position_keys,
    protected_inputs,
)
from chess_ai.data.stockfish_rebuild import _canonical
from chess_ai.data.stockfish_teacher import StockfishTeacher
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import save_checkpoint
from chess_ai.model.policy_value_net import PolicyValueNet
from chess_ai.storage.artifacts import atomic_json, sha256_file

SUITE = Path(__file__).resolve().parents[1] / "configs/search_development_v1.json"


def _corpus(root: Path) -> Path:
    root.mkdir()
    examples = []
    with sqlite3.connect(root / "catalog.sqlite") as db:
        db.execute("CREATE TABLE blocked(key TEXT PRIMARY KEY,kind INTEGER)")
        db.execute("CREATE TABLE positions(key TEXT PRIMARY KEY,fen TEXT,split TEXT)")
        for index in range(100):
            board = opening(991122, index)
            key, fen = _canonical(board)
            split = "train" if index < 50 else "validation"
            policy = np.zeros(4208, dtype=np.float32)
            policy[MoveEncoder().encode(next(iter(board.legal_moves)))] = 1
            examples.append(
                TrainingExample(
                    BoardEncoder().encode(board),
                    policy,
                    0.0,
                    {
                        "fen": fen,
                        "position_key": key,
                        "split": split,
                        "group_id": f"old:{index}",
                        "game_id": f"old:{index}",
                        "value_target_kind": SPEC["kind"],
                    },
                )
            )
            db.execute("INSERT INTO positions VALUES(?,?,?)", (key, fen, split))
    save_dataset(
        root / "shard.pt",
        examples,
        format_version=2,
        metadata={"explicit_split": True, "value_target_spec": SPEC},
    )
    atomic_json(
        root / "dataset.json",
        {
            "format": "self-improving-chess-ai.stockfish-evaluation-shards",
            "version": 1,
            "value_target_spec": SPEC,
            "examples": len(examples),
            "shards": [
                {
                    "path": "shard.pt",
                    "sha256": sha256_file(root / "shard.pt"),
                    "examples": len(examples),
                }
            ],
        },
    )
    atomic_json(
        root / "training_ready.json", {"dataset_sha256": sha256_file(root / "dataset.json")}
    )
    return root


def test_failure_collection_labels_replay_and_ready_resume(
    tmp_path: Path, monkeypatch, teacher_setup
) -> None:
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(31102043)
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", PolicyValueNet(channels=8, residual_blocks=0)
    )
    corpus = _corpus(tmp_path / "corpus")
    before = sha256_file(checkpoint)
    collection = tmp_path / "collection"
    kwargs = {
        "seed": 31102043,
        "pairs": 3,
        "simulations": 2,
        "max_plies": 4,
        "device": "cpu",
        "opponent": "self",
    }
    report = collect(checkpoint, collection, **kwargs)
    assert collect(checkpoint, collection, **kwargs) == report
    assert report["games"] == {"train": 4, "validation": 2}
    config, _ = teacher_setup
    config = replace(config, cache_dir=tmp_path / "teacher_cache")
    output = tmp_path / "labels"
    with pytest.raises(ValueError, match="confirm-training-import"):
        label_failures(
            collection, corpus, SUITE, output, config, confirm_training_import=False, seed=31102043
        )
    monkeypatch.setattr(StockfishTeacher, "analyze_move", lambda self, b, m: {"score_cp": -400})
    label_kwargs = {"confirm_training_import": True, "seed": 31102043, "positions_per_game": 3}
    timed = label_failures(collection, corpus, SUITE, output, config, maximum=1, **label_kwargs)
    assert not timed["complete"] and timed["labeled"] == 1
    done = label_failures(collection, corpus, SUITE, output, config, **label_kwargs)
    assert done["complete"] and done["labeled"] > 1
    assert label_failures(collection, corpus, SUITE, output, config, **label_kwargs) == done
    template = tmp_path / "template.yaml"
    template.write_text("seed: 1\ndevice: cpu\ntraining:\n  batch_size: 8\n")
    ready = finalize_failures(output, corpus / "dataset.json", template, seed=31102043)
    assert finalize_failures(output, corpus / "dataset.json", template, seed=31102043) == ready
    dataset = load_dataset(output / "replay.pt")
    train = set()
    validation = set()
    for example in dataset:
        import chess

        keys = position_keys(chess.Board(str(example.metadata["fen"])))
        (train if example.metadata["split"] == "train" else validation).update(keys)
    assert not train & validation
    assert sha256_file(checkpoint) == before
    assert ready["training_updates"] == 0
    # Ready artifacts cannot silently be edited and reused.
    (output / "train.yaml").write_text("changed")
    with pytest.raises(ValueError, match="Ready failure artifacts changed"):
        finalize_failures(output, corpus / "dataset.json", template, seed=31102043)


def test_protection_covers_full_history_evaluation_and_endgame_mirrors(
    tmp_path: Path, monkeypatch
) -> None:
    import chess
    import chess.pgn

    from chess_ai.data.endgame_pilot import _write_checked

    monkeypatch.chdir(tmp_path)
    corpus = _corpus(tmp_path / "corpus")
    board = chess.Board("7k/8/8/8/8/2K5/8/Q7 w - - 0 1")
    game = chess.pgn.Game.from_board(board)
    move = next(iter(board.legal_moves))
    game.add_main_variation(move)
    path = tmp_path / "data/games/evaluation/keep.pgn"
    path.parent.mkdir(parents=True)
    path.write_text(str(game))
    pilot = tmp_path / "data/experiments/pilot"
    pilot.mkdir(parents=True)
    key = _canonical(board)[0]
    _write_checked(pilot / "positions.json", {"rows": [{"key": key, "fen": board.fen()}]})
    rollout = tmp_path / "data/experiments/rollout"
    rollout.mkdir()
    _write_checked(
        rollout / "binding.json", {"positions_sha256": sha256_file(pilot / "positions.json")}
    )
    _write_checked(rollout / "case_001.json", {"case_key": key, "uci": [move.uci()]})
    blocked, _, files = protected_inputs(corpus, SUITE)
    assert position_keys(board) <= blocked
    board.push(move)
    assert position_keys(board) <= blocked
    assert (
        str(path.relative_to(tmp_path)) in files
        or str(path.relative_to(tmp_path)).replace("\\", "/") in files
    )
    assert files


def test_evaluation_games_cannot_be_imported_as_collection(
    tmp_path: Path, monkeypatch, teacher_setup
) -> None:
    monkeypatch.chdir(tmp_path)
    config, _ = teacher_setup
    collection = tmp_path / "evaluation"
    collection.mkdir()
    atomic_json(collection / "collection.json", {"format": "evaluation", "version": 1})
    with pytest.raises(ValueError, match="Only fresh training"):
        label_failures(
            collection,
            tmp_path,
            SUITE,
            tmp_path / "out",
            config,
            confirm_training_import=True,
            seed=31102043,
        )
