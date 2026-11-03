"""Collection transaction, teacher binding, outcome semantics, and format regressions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from chess_ai.curriculum import stockfish_batch as batch
from chess_ai.data.examples import IncompatibleDatasetError, load_dataset, split_examples_by_group
from chess_ai.data.stockfish_teacher import StockfishTeacherError
from chess_ai.model import PolicyValueNet, save_checkpoint
from chess_ai.model.checkpoint import load_checkpoint


def config_for(tmp_path: Path, teacher_setup):
    teacher_config, _ = teacher_setup
    checkpoint = tmp_path / "baseline" / "epoch_0120.pt"
    torch.manual_seed(83)
    save_checkpoint(checkpoint, PolicyValueNet(channels=4, residual_blocks=0), epoch=120)
    return batch.StockfishBatchConfig(
        tmp_path / "run",
        checkpoint,
        teacher_config,
        opening_positions=1,
        opening_min_full_moves=0,
        opening_max_full_moves=0,
        normal_start_every_positions=1,
        max_plies=4,
        positions_per_game=2,
        d1_depth=1,
        d2_depth=1,
        seed=83,
        device="cpu",
    )


def test_games_before_labels_and_finalization_is_immutable(tmp_path, teacher_setup, monkeypatch):
    config = config_for(tmp_path, teacher_setup)
    checkpoint_before = config.neural_checkpoint.read_bytes()
    original = batch.StockfishTeacher.analyze
    monkeypatch.setattr(
        batch.StockfishTeacher, "analyze", lambda *args: pytest.fail("No labels during games")
    )
    games = batch.run_stockfish_batch(config, stage="games")
    assert games["games"] == 6 and games["labeled_games"] == 0
    assert not Path(games["dataset_path"]).exists()
    monkeypatch.setattr(batch.StockfishTeacher, "analyze", original)
    labels = batch.run_stockfish_batch(config, stage="labels")
    assert labels["examples"] == 12 and not labels["complete"]
    summary = batch.run_stockfish_batch(config, stage="finalize")
    assert summary["complete"] and summary["training_updates"] == 0
    data = load_dataset(summary["dataset_path"])
    assert len(data) == 12
    assert data.metadata["teacher"]["source_commit"] == "a" * 40
    assert all(item.target_value == 0 for item in data)  # all move-limit draws
    assert all(item.metadata["actor_outcome_weight"] == 1.5 for item in data)
    assert all("stockfish_evaluation" in item.metadata for item in data)
    assert config.neural_checkpoint.read_bytes() == checkpoint_before
    train, validation = split_examples_by_group(data, seed=83, group_key="opening_set_id")
    assert len(train) == 12 and not validation
    before = {path: path.read_bytes() for path in config.run_dir.rglob("*") if path.is_file()}
    assert batch.run_stockfish_batch(config)["complete"]
    assert before == {path: path.read_bytes() for path in before}
    assert not list(config.run_dir.rglob("*.tmp"))


@pytest.mark.parametrize("phase", ["games", "labels", "finalize"])
def test_interruption_after_artifact_write_before_state_commit(
    tmp_path, teacher_setup, monkeypatch, phase
):
    config = config_for(tmp_path, teacher_setup)
    if phase == "labels":
        batch.run_stockfish_batch(config, stage="games")
    if phase == "finalize":
        batch.run_stockfish_batch(config, stage="games")
        batch.run_stockfish_batch(config, stage="labels")
    name = {"games": "_generate_game", "labels": "_label_game", "finalize": "atomic_json"}[phase]
    original = getattr(batch, name)
    triggered = False

    def interrupted(*args, **kwargs):
        nonlocal triggered
        result = original(*args, **kwargs)
        is_target = phase != "finalize" or Path(args[0]).name == "manifest.json"
        if not triggered and is_target:
            triggered = True
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(batch, name, interrupted)
    with pytest.raises(KeyboardInterrupt):
        batch.run_stockfish_batch(config, stage=phase)
    assert triggered
    state = json.loads((config.run_dir / "state.json").read_text())
    assert state["pending"]["stage"] == phase
    monkeypatch.setattr(batch, name, original)
    summary = batch.run_stockfish_batch(config)
    assert summary["complete"] and summary["examples"] == 12


def test_duplicate_teacher_failure_resumes_without_regenerating_games(
    tmp_path, teacher_setup, monkeypatch
):
    config = config_for(tmp_path, teacher_setup)
    batch.run_stockfish_batch(config, stage="games")
    before = json.loads((config.run_dir / "state.json").read_text())
    original = batch.StockfishTeacher._analyse

    def duplicate(self, board):
        rows = original(self, board)
        rows[-1] = rows[0].copy()
        return rows

    monkeypatch.setattr(batch.StockfishTeacher, "_analyse", duplicate)
    with pytest.raises(StockfishTeacherError, match="duplicate MultiPV root moves"):
        batch.run_stockfish_batch(config, stage="labels")
    failed = json.loads((config.run_dir / "state.json").read_text())
    assert not failed["labels"] and failed["pending"]["stage"] == "labels"
    assert not Path(failed["pending"]["paths"][0]).exists()
    monkeypatch.setattr(batch.StockfishTeacher, "_analyse", original)
    monkeypatch.setattr(batch, "_generate_game", lambda *args: pytest.fail("Games already saved"))
    result = batch.run_stockfish_batch(config)
    after = json.loads((config.run_dir / "state.json").read_text())
    assert result["complete"] and result["examples"] == 12
    assert before["games"] == after["games"]
    assert before["signature"] == after["signature"]


@pytest.mark.parametrize(
    "change", ["commit", "binary", "nodes", "checkpoint", "seed", "temperature"]
)
def test_resume_rejects_changed_teacher_or_collection(tmp_path, teacher_setup, change):
    config = config_for(tmp_path, teacher_setup)
    batch.run_stockfish_batch(config, stage="games")
    if change == "commit":
        config = replace(config, stockfish=replace(config.stockfish, source_commit="b" * 40))
    if change == "binary":
        config.stockfish.executable.write_bytes(b"different engine")
    if change == "nodes":
        config = replace(config, stockfish=replace(config.stockfish, nodes=1000))
    if change == "temperature":
        config = replace(config, stockfish=replace(config.stockfish, temperature_cp=200))
    if change == "checkpoint":
        save_checkpoint(
            config.neural_checkpoint, PolicyValueNet(channels=4, residual_blocks=0), epoch=121
        )
    if change == "seed":
        config = replace(config, seed=84)
    with pytest.raises(batch.StockfishBatchError, match="changed"):
        batch.run_stockfish_batch(config, stage="labels")


def test_corrupt_shards_pg_ns_and_incompatible_formats_fail(tmp_path, teacher_setup):
    config = config_for(tmp_path, teacher_setup)
    result = batch.run_stockfish_batch(config)
    payload = torch.load(result["dataset_path"], weights_only=True)
    payload["format_version"] = 999
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)
    with pytest.raises(IncompatibleDatasetError):
        load_dataset(incompatible)
    checkpoint = torch.load(config.neural_checkpoint, weights_only=False)
    checkpoint["format_version"] = 999
    torch.save(checkpoint, incompatible)
    with pytest.raises(RuntimeError, match="version"):
        load_checkpoint(incompatible)
    shard = next((config.run_dir / "shards").glob("*.pt"))
    shard.write_bytes(b"partial-write")
    with pytest.raises(batch.StockfishBatchError, match="digest"):
        batch.run_stockfish_batch(config)


def test_cannot_label_early_or_finalize_partial(tmp_path, teacher_setup):
    config = config_for(tmp_path, teacher_setup)
    with pytest.raises(batch.StockfishBatchError, match="games stage"):
        batch.run_stockfish_batch(config, stage="labels")
    batch.run_stockfish_batch(config, stage="games")
    with pytest.raises(batch.StockfishBatchError, match="Finalize requires"):
        batch.run_stockfish_batch(config, stage="finalize")


def test_sampling_is_seeded_bounded_and_groups_never_leak(tmp_path, teacher_setup):
    config = config_for(tmp_path, teacher_setup)
    board_a, moves_a = batch.opening_board(
        replace(
            config,
            opening_min_full_moves=2,
            opening_max_full_moves=5,
            normal_start_every_positions=0,
            max_plies=20,
        ),
        1,
    )
    board_b, moves_b = batch.opening_board(
        replace(
            config,
            opening_min_full_moves=2,
            opening_max_full_moves=5,
            normal_start_every_positions=0,
            max_plies=20,
        ),
        1,
    )
    assert board_a.fen() == board_b.fen() and moves_a == moves_b
    batch.run_stockfish_batch(config)
    data = load_dataset(config.run_dir / "dataset.pt")
    second = [
        replace(example, metadata={**example.metadata, "opening_set_id": "second"})
        for example in data
    ]
    train, valid = split_examples_by_group([*data, *second], seed=83, group_key="opening_set_id")
    assert {item.metadata["opening_set_id"] for item in train}.isdisjoint(
        {item.metadata["opening_set_id"] for item in valid}
    )


def test_collection_writer_lock_and_unknown_state_version(tmp_path, teacher_setup):
    from chess_ai.storage.artifacts import run_lock

    config = config_for(tmp_path, teacher_setup)
    with run_lock(config.run_dir), pytest.raises(RuntimeError, match="locked"):
        batch.run_stockfish_batch(config)
    batch.run_stockfish_batch(config, stage="games")
    state_path = config.run_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["version"] = 999
    state_path.write_text(json.dumps(state))
    with pytest.raises(batch.StockfishBatchError, match="version"):
        batch.run_stockfish_batch(config)


def test_finalized_digest_and_pgn_digest_are_checked(tmp_path, teacher_setup):
    config = config_for(tmp_path, teacher_setup)
    batch.run_stockfish_batch(config)
    manifest = config.run_dir / "manifest.json"
    original = manifest.read_bytes()
    manifest.write_bytes(b"partial")
    with pytest.raises(batch.StockfishBatchError, match="Finalized artifact"):
        batch.run_stockfish_batch(config)
    manifest.write_bytes(original)
    next((config.run_dir / "games").glob("*.pgn")).write_text("invalid")
    with pytest.raises(batch.StockfishBatchError, match="PGN digest"):
        batch.run_stockfish_batch(config)
