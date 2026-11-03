"""All-move PUCT collection, extra book games, replay isolation, and evaluation targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_ai.agents.opening_book_agent import OpeningBookAgent
from chess_ai.curriculum import stockfish_batch as batch
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.opening_book import OPENING_BOOK_FORMAT, position_key
from chess_ai.data.refinement_replay import compose_refinement_replay
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher, StockfishTeacherError
from chess_ai.data.value_targets import SHARDED_DATASET_FORMAT, STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model import PolicyValueNet, save_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file, signature
from chess_ai.training.trainer import Trainer, TrainingConfig


def refinement_config(tmp_path, teacher):
    config = batch.StockfishBatchConfig(
        tmp_path / "collection",
        tmp_path / "champion.pt",
        teacher,
        opening_positions=2,
        opening_min_full_moves=1,
        opening_max_full_moves=1,
        normal_start_every_positions=0,
        max_plies=6,
        positions_per_game=1,
        d1_depth=1,
        d2_depth=1,
        device="cpu",
        collection_version=2,
        search_simulations=2,
        opening_book=tmp_path / "book.json",
        extra_standard_book_games=True,
        validation_fraction=0.4,
    )
    config = replace(
        config,
        seed=next(
            seed
            for seed in range(100)
            if len(
                {
                    int(signature({"group": f"sf-{seed}-opening-{i:04d}", "seed": seed})[:16], 16)
                    / 2**64
                    < 0.4
                    for i in (0, 1, 2)
                }
            )
            == 2
        ),
    )
    atomic_json(
        config.opening_book,
        {
            "format": OPENING_BOOK_FORMAT,
            "version": 1,
            "positions": {position_key(chess.Board()): [{"move_uci": "e2e4", "weight": 1}]},
        },
    )
    torch.manual_seed(192)
    save_checkpoint(
        config.neural_checkpoint,
        PolicyValueNet(channels=2, residual_blocks=0),
        epoch=60,
        training_config={"value_target_spec": config.value_spec},
    )
    return config


def test_every_move_both_colors_book_reset_and_versioned_training(
    tmp_path, teacher_setup, monkeypatch
):
    config = refinement_config(tmp_path, teacher_setup[0])
    wrappers = []

    class BookSpy(OpeningBookAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            wrappers.append(self)

    monkeypatch.setattr(batch, "OpeningBookAgent", BookSpy)
    original = batch.StockfishTeacher.analyze
    monkeypatch.setattr(
        batch.StockfishTeacher,
        "analyze",
        lambda *args: pytest.fail("Teacher must not analyze during gameplay"),
    )
    before = sha256_file(config.neural_checkpoint)
    result = batch.run_stockfish_batch(config, stage="games")
    assert result["games"] == 12 and not result["complete"]
    assert len(wrappers) == 4 and len({id(item) for item in wrappers}) == 4
    state = json.loads((config.run_dir / "state.json").read_text())
    standard = [game for game in state["games"] if game["book_enabled"]]
    assert len(standard) == 4
    assert {(game["cohort"], game["agent_a_color"]) for game in standard} == {
        (cohort, color)
        for cohort in ("neural_vs_d1", "neural_vs_d2")
        for color in ("white", "black")
    }
    assert all(
        game["opening_fen"] == chess.STARTING_FEN and not game["opening_moves"] for game in standard
    )
    assert all(not game["book_enabled"] and game["opening_moves"] for game in state["games"][:8])
    monkeypatch.setattr(batch.StockfishTeacher, "analyze", original)
    batch.run_stockfish_batch(config, stage="labels")
    for game in state["games"]:
        rows = load_dataset(game["shard"])
        assert rows.format_version == 2 and len(rows) == game["plies"]
        assert {row.metadata["player_to_move"] for row in rows} == {"white", "black"}
        assert [row.metadata["absolute_ply"] for row in rows] == list(range(game["plies"]))
        for row in rows:
            assert row.target_value == pytest.approx(math.tanh(100 / 600))
            assert row.metadata["value_target_kind"] == STOCKFISH_VALUE_KIND
            assert (
                row.metadata["played_move_evaluation"]["move_uci"]
                == row.metadata["actual_move_uci"]
            )
            assert row.metadata["completed_game_outcome"] == 0
            assert row.metadata["fen"].split()[-2:] == ["0", "1"]
    final = batch.run_stockfish_batch(config, stage="finalize")
    data = load_dataset(final["dataset_path"])
    assert len({row.metadata["position_key"] for row in data}) == len(data)
    groups = {
        split: {row.metadata["group_id"] for row in data if row.metadata["split"] == split}
        for split in ("train", "validation")
    }
    assert groups["train"] and groups["validation"] and not groups["train"] & groups["validation"]
    trainer = Trainer(
        PolicyValueNet(channels=2, residual_blocks=0),
        TrainingConfig(
            dataset_path=Path(final["dataset_path"]),
            checkpoint_dir=tmp_path / "candidate",
            metrics_path=tmp_path / "metrics.jsonl",
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=192,
            use_dataset_split=True,
            legal_policy_mask=True,
            value_target_spec=config.value_spec,
        ),
    )
    assert len(trainer.fit()) == 1
    assert sha256_file(config.neural_checkpoint) == before
    assert batch.run_stockfish_batch(config)["complete"]


def test_refinement_schedule_has_exactly_204_games_and_excludes_holdout(tmp_path, teacher_setup):
    config = replace(refinement_config(tmp_path, teacher_setup[0]), opening_positions=50)
    initial = batch._game_specs(config)
    excluded = tmp_path / "holdout.json"
    atomic_json(excluded, {"pairs": [{"opening_fen": initial[0]["opening_fen"]}]})
    config = replace(config, exclude_opening_manifests=(excluded,))
    specs = batch._game_specs(config)
    assert len(specs) == 204 and sum(item["book_enabled"] for item in specs) == 4
    assert initial[0]["opening_fen"] not in {
        item["opening_fen"] for item in specs if item["opening_index"]
    }
    assert specs == batch._game_specs(config)


@pytest.mark.parametrize("change", ["book", "search", "exclusions"])
def test_refinement_resume_binds_book_search_and_exclusions(tmp_path, teacher_setup, change):
    config = refinement_config(tmp_path, teacher_setup[0])
    excluded = tmp_path / "holdout.json"
    atomic_json(excluded, {"pairs": []})
    config = replace(config, exclude_opening_manifests=(excluded,))
    batch.run_stockfish_batch(config, stage="games")
    if change == "book":
        config.opening_book.write_text(config.opening_book.read_text() + "\n")
    elif change == "search":
        config = replace(config, search_simulations=3)
    else:
        excluded.write_text(excluded.read_text() + "\n")
    with pytest.raises(batch.StockfishBatchError, match="changed"):
        batch.run_stockfish_batch(config, stage="labels")


def test_interrupted_refinement_reuses_saved_games_and_teacher_cache(
    tmp_path, teacher_setup, monkeypatch
):
    config = refinement_config(tmp_path, teacher_setup[0])
    batch.run_stockfish_batch(config, stage="games")
    original = batch._label_game

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_label_game", interrupted)
    with pytest.raises(KeyboardInterrupt):
        batch.run_stockfish_batch(config, stage="labels")
    monkeypatch.setattr(batch, "_label_game", original)
    monkeypatch.setattr(batch, "_generate_game", lambda *args: pytest.fail("Already generated"))
    assert batch.run_stockfish_batch(config)["complete"]


def test_forced_root_scores_illegal_moves_and_cache_integrity(teacher_setup, monkeypatch):
    config, _ = teacher_setup
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    with StockfishTeacher(config) as teacher:
        row = teacher.analyze_move(board, move)
        assert row["analysis_kind"] == "forced_root" and row["score_cp"] == 100
        monkeypatch.setattr(
            teacher, "_analyse", lambda *args, **kwargs: pytest.fail("Expected cached scores")
        )
        assert teacher.analyze_move(board, move) == row
        with pytest.raises(StockfishTeacherError, match="illegal"):
            teacher.analyze_move(board, chess.Move.from_uci("e2e5"))
        cache = next(config.cache_dir.glob("forced_*.json"))
        cache.write_text('{"signature": "broken"}')
        with pytest.raises(StockfishTeacherError, match="forced-root cache"):
            teacher.analyze_move(board, move)


def _position(move, split, group):
    board = chess.Board()
    if move:
        board.push_uci(move)
    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
    policy[MoveEncoder().encode(next(iter(board.legal_moves)))] = 1
    encoded = BoardEncoder().encode(board)
    return TrainingExample(
        encoded,
        policy,
        0.2,
        {
            "fen": board.fen(),
            "game_id": group,
            "group_id": group,
            "position_key": hashlib.sha256(encoded.tobytes()).hexdigest(),
            "split": split,
            "value_target_kind": STOCKFISH_VALUE_KIND,
        },
    )


def test_replay_50_50_protects_all_old_holdout_inputs_and_keeps_splits(tmp_path):
    spec = {"kind": STOCKFISH_VALUE_KIND, "perspective": "side_to_move", "scale_cp": 600}
    old = [
        _position("e2e4", "validation", "old-valid"),
        _position("d2d4", "validation", "old-valid"),
        _position("c2c4", "train", "old-train"),
        _position("g1f3", "train", "old-train"),
    ]
    old_shard = tmp_path / "old.pt"
    save_dataset(old_shard, old, metadata={"value_target_spec": spec}, format_version=2)
    manifest = tmp_path / "old.json"
    atomic_json(
        manifest,
        {
            "format": SHARDED_DATASET_FORMAT,
            "version": 1,
            "value_target_spec": spec,
            "examples": 4,
            "shards": [{"path": "old.pt", "sha256": sha256_file(old_shard), "examples": 4}],
        },
    )
    fresh_path = tmp_path / "fresh.pt"
    fresh = [
        _position("e2e4", "train", "fresh-train"),
        _position("b1c3", "train", "fresh-train"),
        _position("b2b3", "validation", "fresh-valid"),
        _position("c2c4", "validation", "fresh-valid"),
    ]
    save_dataset(
        fresh_path,
        fresh,
        metadata={"value_target_spec": spec, "explicit_split": True},
        format_version=2,
    )
    output = tmp_path / "replay.pt"
    result = compose_refinement_replay(fresh_path, manifest, output, seed=19)
    data = load_dataset(output)
    assert (
        result["examples"] == 4 and result["fresh_training_rows_excluded_for_old_validation"] == 1
    )
    assert result["fresh_validation_rows_excluded_for_old_training"] == 1
    assert data.metadata["fresh_by_split"] == {"train": 1, "validation": 1}
    assert data.metadata["replay_by_split"] == {"train": 1, "validation": 1}
    protected = old[0].metadata["position_key"]
    assert protected not in {
        row.metadata["position_key"] for row in data if row.metadata["split"] == "train"
    }
    assert len({row.metadata["position_key"] for row in data}) == len(data)
    output2 = tmp_path / "same.pt"
    compose_refinement_replay(fresh_path, manifest, output2, seed=19)
    assert [row.metadata for row in data] == [row.metadata for row in load_dataset(output2)]
    with pytest.raises(ValueError, match="output exists"):
        compose_refinement_replay(fresh_path, manifest, output, seed=19)
    old_shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        compose_refinement_replay(fresh_path, manifest, tmp_path / "bad.pt", seed=19)


@pytest.mark.skipif(
    not os.environ.get("CHESS_AI_STOCKFISH"), reason="local UCI executable not configured"
)
def test_real_stockfish_all_move_refinement_and_forced_root(tmp_path):
    teacher = StockfishConfig(
        Path(os.environ["CHESS_AI_STOCKFISH"]),
        os.environ["CHESS_AI_STOCKFISH_COMMIT"],
        nodes=3000,
        multipv=2,
        cache_dir=tmp_path / "cache",
    )
    config = refinement_config(tmp_path, teacher)
    final = batch.run_stockfish_batch(config)
    state = json.loads((config.run_dir / "state.json").read_text())
    assert final["complete"] and final["games"] == 12
    for game in state["games"]:
        shard = load_dataset(game["shard"])
        assert len(shard) == game["plies"]
        for row in shard:
            assert row.target_value == pytest.approx(row.metadata["stockfish_evaluation"])
            assert (
                row.metadata["played_move_evaluation"]["move_uci"]
                == row.metadata["actual_move_uci"]
            )
