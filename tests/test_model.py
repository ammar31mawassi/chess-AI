"""Tests for the compact policy-value network and checkpoint contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_ai.model import (
    IncompatibleCheckpointError,
    PolicyValueNet,
    load_checkpoint,
    load_model,
    save_checkpoint,
)


def test_policy_value_network_shapes_finite_and_backward() -> None:
    model = PolicyValueNet(channels=8, residual_blocks=1)
    boards = torch.randn(3, 18, 8, 8)

    policy_logits, values = model(boards)

    assert policy_logits.shape == (3, 4208)
    assert values.shape == (3, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(values).all()
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)

    (policy_logits.square().mean() + values.square().mean()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_network_rejects_wrong_board_shape() -> None:
    model = PolicyValueNet(channels=4, residual_blocks=0)
    with pytest.raises(ValueError, match="18, 8, 8"):
        model(torch.zeros(1, 17, 8, 8))


def test_checkpoint_round_trip_preserves_outputs(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = PolicyValueNet(channels=8, residual_blocks=1).eval()
    boards = torch.randn(2, 18, 8, 8)
    with torch.inference_mode():
        expected_policy, expected_value = model(boards)

    checkpoint_path = save_checkpoint(tmp_path / "model.pt", model, epoch=4)
    loaded = load_checkpoint(checkpoint_path)
    restored = load_model(checkpoint_path)

    assert loaded.epoch == 4
    assert loaded.metadata["model_config"]["action_size"] == 4208
    with torch.inference_mode():
        actual_policy, actual_value = restored(boards)
    torch.testing.assert_close(actual_policy, expected_policy)
    torch.testing.assert_close(actual_value, expected_value)


def test_checkpoint_rejects_architecture_mismatch(tmp_path: Path) -> None:
    path = save_checkpoint(
        tmp_path / "model.pt",
        PolicyValueNet(channels=8, residual_blocks=1),
    )
    incompatible = PolicyValueNet(channels=16, residual_blocks=1)

    with pytest.raises(IncompatibleCheckpointError, match="architecture"):
        load_checkpoint(path, incompatible)


def test_checkpoint_rejects_unknown_version(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "model.pt", PolicyValueNet(channels=4, residual_blocks=0))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 999
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="version"):
        load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_network_executes_on_cuda() -> None:
    model = PolicyValueNet(channels=4, residual_blocks=0).cuda()
    policy_logits, value = model(torch.zeros(1, 18, 8, 8, device="cuda"))
    assert policy_logits.is_cuda
    assert value.is_cuda
