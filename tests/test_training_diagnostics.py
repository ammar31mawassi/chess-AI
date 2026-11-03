"""Source coverage, weighted holdout metrics, and read-only diagnostic regression tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_ai.data.examples import TrainingExample, save_dataset
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model import PolicyValueNet, save_checkpoint
from chess_ai.training.diagnostics import diagnose_training, sampling_summary
from chess_ai.training.trainer import (
    Trainer,
    TrainingConfig,
    TrainingError,
    training_sampling_weights,
)


def example(game: str, source: str, move: str, weight: float = 1.0, value: float = 0.0):
    policy = np.zeros(4208, dtype=np.float32)
    policy[MoveEncoder().encode(chess.Move.from_uci(move))] = 1
    return TrainingExample(
        BoardEncoder().encode(chess.Board()),
        policy,
        value,
        {
            "game_id": game,
            "composition_source": source,
            "fen": chess.STARTING_FEN,
            "weight": weight,
            "group": game,
        },
    )


def config(tmp_path: Path):
    return TrainingConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_path=tmp_path / "metrics.jsonl",
        batch_size=2,
        seed=83,
        device="cpu",
        legal_policy_mask=True,
        validation_source_key="composition_source",
        sample_weight_key="weight",
        game_balanced_sampling=True,
    )


def test_sampling_reports_games_rather_than_raw_source_size(tmp_path):
    rows = [
        example("a", "teacher", "e2e4", 1),
        example("a", "teacher", "d2d4", 3),
        example("b", "replay", "d2d4"),
    ]
    summary = sampling_summary(rows, config(tmp_path), "composition_source")
    assert summary["teacher"]["example_share"] == pytest.approx(2 / 3)
    assert summary["teacher"]["expected_draw_share"] == pytest.approx(0.5)
    assert summary["teacher"]["expected_weighted_draw_share"] == pytest.approx(2 / 3)
    ordinary = sampling_summary(
        rows, replace(config(tmp_path), game_balanced_sampling=False), "composition_source"
    )
    assert ordinary["teacher"]["expected_draw_share"] == pytest.approx(2 / 3)


def test_source_sampling_changes_exposure_without_changing_loss_weights(tmp_path):
    rows = [
        example("a", "teacher", "e2e4", 1),
        example("a", "teacher", "d2d4", 3),
        example("b", "replay", "d2d4"),
    ]
    focused = replace(config(tmp_path), source_sampling_multipliers={"teacher": 8.0})
    assert training_sampling_weights(rows, focused) == [4.0, 4.0, 1.0]
    summary = sampling_summary(rows, focused, "composition_source")
    assert summary["teacher"]["expected_draw_share"] == pytest.approx(8 / 9)
    trainer = Trainer(PolicyValueNet(channels=4, residual_blocks=0), focused)
    loader = trainer._loader(rows, training=True, epoch=1)
    assert loader.sampler.weights.tolist() == [4, 4, 1]
    assert loader.dataset.weights == [1, 3, 1]
    first = list(loader.sampler)
    second = list(trainer._loader(rows, training=True, epoch=1).sampler)
    assert first == second
    ordinary = replace(focused, game_balanced_sampling=False)
    assert training_sampling_weights(rows, ordinary) == [8, 8, 1]
    with pytest.raises(TrainingError, match="absent"):
        training_sampling_weights(rows, replace(focused, source_sampling_multipliers={"typo": 8}))


@pytest.mark.parametrize(
    "multipliers", [{"teacher": 0}, {"teacher": True}, {"teacher": float("inf")}, {"": 8}]
)
def test_invalid_source_multipliers_fail_clearly(tmp_path, multipliers):
    with pytest.raises(ValueError, match="sampling"):
        replace(config(tmp_path), source_sampling_multipliers=multipliers)


def test_source_metrics_respect_mask_weights_and_do_not_change_model(tmp_path):
    torch.manual_seed(83)
    model = PolicyValueNet(channels=4, residual_blocks=0)
    with torch.no_grad():
        linear = model.policy_head[-1]
        linear.weight.zero_()
        linear.bias.zero_()
        linear.bias[MoveEncoder().encode(chess.Move.from_uci("e2e4"))] = 5
        # An illegal logit must not win inference or source accuracy.
        linear.bias[MoveEncoder().encode(chess.Move.from_uci("e2e5"))] = 100
        for parameter in model.value_head.parameters():
            parameter.zero_()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rows = [
        example("a", "teacher", "e2e4", 1, 1),
        example("a", "teacher", "d2d4", 3, -1),
        example("b", "replay", "d2d4"),
    ]
    result = Trainer(model, config(tmp_path)).evaluate_by_source(
        rows, source_key="composition_source"
    )
    assert result["teacher"]["examples"] == 2
    assert result["teacher"]["top1"] == pytest.approx(0.25)
    assert result["teacher"]["unweighted_top1"] == pytest.approx(0.5)
    assert result["teacher"]["value_mae"] == pytest.approx(1)
    assert result["replay"]["top1"] == 0
    assert result["replay"]["value_mae"] == 0
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert not list(tmp_path.rglob("*.pt"))


def test_fit_records_sources_and_rejects_unlabelled_rows_before_training(tmp_path):
    torch.manual_seed(83)
    settings = config(tmp_path)
    train = [example("train", "teacher", "e2e4")]
    validation = [example("v1", "teacher", "e2e4"), example("v2", "replay", "d2d4")]
    history = Trainer(PolicyValueNet(channels=4, residual_blocks=0), settings).fit(
        train, validation_examples=validation
    )
    assert set(history[0].validation_by_source) == {"teacher", "replay"}
    saved = json.loads(settings.metrics_path.read_text())
    assert saved["validation_by_source"]["teacher"]["examples"] == 1
    del train[0].metadata["composition_source"]
    fresh = replace(
        settings, checkpoint_dir=tmp_path / "fresh", metrics_path=tmp_path / "fresh.jsonl"
    )
    with pytest.raises(TrainingError, match="source label"):
        Trainer(PolicyValueNet(channels=4, residual_blocks=0), fresh).fit(
            train, validation_examples=validation
        )
    assert not fresh.checkpoint_dir.exists()


def test_report_uses_same_holdout_and_preserves_inputs(tmp_path):
    torch.manual_seed(83)
    dataset = tmp_path / "dataset.pt"
    rows = [example(str(index), "teacher" if index % 2 else "replay", "e2e4") for index in range(8)]
    save_dataset(dataset, rows)
    checkpoint = tmp_path / "baseline.pt"
    save_checkpoint(checkpoint, PolicyValueNet(channels=4, residual_blocks=0), epoch=120)
    settings = replace(
        config(tmp_path),
        dataset_path=dataset,
        validation_fraction=0.5,
        validation_group_key="group",
    )
    before = {path: path.read_bytes() for path in (dataset, checkpoint)}
    first = diagnose_training(settings, [checkpoint], tmp_path / "first.json")
    second = diagnose_training(settings, [checkpoint], tmp_path / "second.json")
    assert first == second
    assert first["training_updates"] == 0 and first["validation_examples"] == 4
    assert first["checkpoints"][0]["epoch"] == 120
    assert before == {path: path.read_bytes() for path in before}
    with pytest.raises(TrainingError, match="already exists"):
        diagnose_training(settings, [checkpoint], tmp_path / "first.json")


def test_source_config_and_labels_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="validation_source_key"):
        replace(config(tmp_path), validation_source_key=" ")
    with pytest.raises(TrainingError, match="source_key"):
        Trainer.source_indices([], "")
    bad = example("bad", "teacher", "e2e4")
    bad.metadata["composition_source"] = 42
    with pytest.raises(TrainingError, match="source label"):
        sampling_summary([bad], config(tmp_path), "composition_source")
    with pytest.raises(TrainingError, match="non-empty"):
        dataset = tmp_path / "single.pt"
        save_dataset(dataset, [example("one", "teacher", "e2e4")])
        diagnose_training(
            replace(config(tmp_path), dataset_path=dataset),
            [tmp_path / "absent.pt"],
            tmp_path / "out.json",
        )
