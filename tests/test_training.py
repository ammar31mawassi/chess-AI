"""Tests for values, game-group splits, losses, and trainer resume."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_ai.data import TrainingExample, split_examples_by_game, split_examples_by_group
from chess_ai.data.dataset_generator import generate_game_examples, result_value_for_turn
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.model import PolicyValueNet, load_checkpoint
from chess_ai.training import Trainer, TrainingConfig, policy_accuracy, policy_value_loss


def _example(game_id: str, action: int, value: float) -> TrainingExample:
    policy = np.zeros(4208, dtype=np.float32)
    policy[action] = 1.0
    return TrainingExample(
        BoardEncoder().encode(chess.Board()),
        policy,
        value,
        {"game_id": game_id},
    )


@pytest.mark.parametrize(
    ("result", "turn", "expected"),
    [
        ("1-0", chess.WHITE, 1.0),
        ("1-0", chess.BLACK, -1.0),
        ("0-1", chess.WHITE, -1.0),
        ("0-1", chess.BLACK, 1.0),
        ("1/2-1/2", chess.WHITE, 0.0),
        ("1/2-1/2", chess.BLACK, 0.0),
    ],
)
def test_training_value_uses_player_to_move_perspective(
    result: str, turn: chess.Color, expected: float
) -> None:
    assert result_value_for_turn(result, turn) == expected


def test_generated_game_labels_each_position_from_active_perspective() -> None:
    class ScriptedAgent:
        def __init__(self, name: str, moves: list[str]) -> None:
            self.name = name
            self._moves = iter(moves)

        def choose_move(self, board: chess.Board) -> chess.Move:
            move = chess.Move.from_uci(next(self._moves))
            assert move in board.legal_moves
            return move

    # Fool's Mate: Black wins on the fourth ply.
    examples = generate_game_examples(
        ScriptedAgent("scripted-white", ["f2f3", "g2g4"]),
        ScriptedAgent("scripted-black", ["e7e5", "d8h4"]),
        game_id="black-wins",
        max_moves=8,
    )

    assert [item.metadata["result"] for item in examples] == ["0-1"] * 4
    assert [item.target_value for item in examples] == [-1.0, 1.0, -1.0, 1.0]


def test_game_group_split_has_no_leakage_and_is_reproducible() -> None:
    examples = [
        _example(game_id, action=index, value=0.0)
        for index, game_id in enumerate(["a", "a", "b", "b", "c", "c", "d", "d"])
    ]
    first_train, first_validation = split_examples_by_game(examples, 0.25, seed=7)
    second_train, second_validation = split_examples_by_game(examples, 0.25, seed=7)

    train_ids = {item.game_id for item in first_train}
    validation_ids = {item.game_id for item in first_validation}
    assert train_ids.isdisjoint(validation_ids)
    assert [item.game_id for item in first_train] == [item.game_id for item in second_train]
    assert [item.game_id for item in first_validation] == [
        item.game_id for item in second_validation
    ]


def test_opening_pair_group_split_keeps_color_swapped_games_together() -> None:
    examples: list[TrainingExample] = []
    for pair in range(4):
        for color in ("white", "black"):
            for position in range(2):
                example = _example(
                    f"pair-{pair}-{color}",
                    action=pair * 4 + position,
                    value=0.0,
                )
                example.metadata = {**example.metadata, "opening_pair_id": f"pair-{pair}"}
                examples.append(example)

    training, validation = split_examples_by_group(
        examples,
        0.25,
        seed=11,
        group_key="opening_pair_id",
    )

    training_pairs = {str(item.metadata["opening_pair_id"]) for item in training}
    validation_pairs = {str(item.metadata["opening_pair_id"]) for item in validation}
    assert training_pairs.isdisjoint(validation_pairs)
    assert len(validation_pairs) == 1


def test_policy_value_loss_is_finite_and_backpropagates() -> None:
    model = PolicyValueNet(channels=4, residual_blocks=0)
    boards = torch.randn(2, 18, 8, 8)
    policy_logits, values = model(boards)
    target_policy = torch.zeros_like(policy_logits)
    target_policy[0, 4] = 1.0
    target_policy[1, 9] = 1.0
    target_value = torch.tensor([[1.0], [-1.0]])

    losses = policy_value_loss(
        policy_logits,
        values,
        target_policy,
        target_value,
        model=model,
        l2_regularization=1e-8,
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_policy_value_loss_applies_positive_per_example_weights() -> None:
    logits = torch.tensor([[0.0, 0.0], [0.0, 3.0]], requires_grad=True)
    target_policy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    predicted_value = torch.zeros((2, 1), requires_grad=True)
    target_value = torch.zeros((2, 1))

    unweighted = policy_value_loss(logits, predicted_value, target_policy, target_value)
    weighted = policy_value_loss(
        logits,
        predicted_value,
        target_policy,
        target_value,
        sample_weight=torch.tensor([2.0, 1.0]),
    )

    assert weighted.policy < unweighted.policy
    weighted.total.backward()
    assert logits.grad is not None


def test_policy_loss_and_accuracy_can_match_the_inference_legality_mask() -> None:
    logits = torch.tensor([[100.0, 1.0, 2.0]])
    target_policy = torch.tensor([[0.0, 0.0, 1.0]])
    predicted_value = torch.tensor([[0.0]])
    target_value = torch.tensor([[1.0]])
    legal_mask = torch.tensor([[False, True, True]])

    losses = policy_value_loss(
        logits,
        predicted_value,
        target_policy,
        target_value,
        value_loss_weight=0.1,
        legal_action_mask=legal_mask,
    )

    assert losses.policy.item() < 1.0
    assert losses.total.item() < losses.policy.item() + losses.value.item()
    assert (
        policy_accuracy(
            logits,
            target_policy,
            legal_action_mask=legal_mask,
        ).item()
        == 1.0
    )


def test_tiny_training_writes_metrics_checkpoints_and_resumes(tmp_path: Path) -> None:
    examples = [
        _example(game_id, action=index + 1, value=(-1.0 if index % 2 else 1.0))
        for index, game_id in enumerate(["g1", "g1", "g2", "g2", "g3", "g3"])
    ]
    checkpoints = tmp_path / "checkpoints"
    metrics = tmp_path / "metrics.jsonl"
    first_config = TrainingConfig(
        checkpoint_dir=checkpoints,
        metrics_path=metrics,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        validation_fraction=1 / 3,
        seed=3,
        device="cpu",
        log_every=100,
    )
    first_history = Trainer(PolicyValueNet(channels=4, residual_blocks=0), first_config).fit(
        examples
    )

    assert len(first_history) == 1
    assert np.isfinite(first_history[0].training_loss)
    assert first_history[0].validation_loss is not None
    assert (checkpoints / "epoch_0001.pt").is_file()
    assert (checkpoints / "best.pt").is_file()
    assert (checkpoints / "last.pt").is_file()

    resumed_config = TrainingConfig(
        checkpoint_dir=checkpoints,
        metrics_path=metrics,
        batch_size=2,
        epochs=2,
        learning_rate=1e-3,
        validation_fraction=1 / 3,
        seed=3,
        device="cpu",
        log_every=100,
        resume_from=checkpoints / "last.pt",
    )
    resumed_history = Trainer(PolicyValueNet(channels=4, residual_blocks=0), resumed_config).fit(
        examples
    )

    assert [item.epoch for item in resumed_history] == [2]
    assert load_checkpoint(checkpoints / "last.pt").epoch == 2
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[-1]["device"] == "cpu"
    assert rows[-1]["number_of_examples"] == len(examples)


def test_policy_selection_early_stops_and_keeps_initial_baseline_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = [_example("g1", 1, 0.0), _example("g2", 2, 0.0)]
    config = TrainingConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_path=tmp_path / "metrics.jsonl",
        epochs=10,
        batch_size=2,
        validation_fraction=0.5,
        selection_metric="validation_policy_loss",
        early_stopping_patience=2,
        early_stopping_min_delta=0.001,
        checkpoint_every=10,
        baseline_eligible=True,
        device="cpu",
    )
    trainer = Trainer(PolicyValueNet(channels=4, residual_blocks=0), config)
    policy_by_epoch = {0: 1.0, 1: 0.9, 2: 0.9005, 3: 0.9007}

    def fake_epoch(
        *_args: object, training: bool, epoch: int, **_kwargs: object
    ) -> dict[str, float]:
        policy = 2.0 if training else policy_by_epoch[epoch]
        return {
            "total": policy,
            "policy": policy,
            "value": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "value_mae": 0.0,
            "unweighted_total": policy,
            "unweighted_policy": policy,
            "unweighted_value": 0.0,
            "unweighted_top1": 0.0,
            "unweighted_top5": 0.0,
            "unweighted_value_mae": 0.0,
        }

    monkeypatch.setattr(trainer, "_run_epoch", fake_epoch)

    history = trainer.fit(examples)

    assert [item.epoch for item in history] == [1, 2, 3]
    assert history[-1].early_stopped is True
    assert (config.checkpoint_dir / "baseline.pt").is_file()
    assert load_checkpoint(config.checkpoint_dir / "best.pt").epoch == 1
    assert load_checkpoint(config.checkpoint_dir / "last.pt").epoch == 3
    assert not (config.checkpoint_dir / "epoch_0001.pt").exists()
    assert (config.checkpoint_dir / "epoch_0003.pt").is_file()
