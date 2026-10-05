"""Readable policy/value loss and metric calculations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class LossComponents:
    """Differentiable pieces of the combined training objective."""

    total: Tensor
    policy: Tensor
    value: Tensor
    l2: Tensor

    def detached(self) -> dict[str, float]:
        """Convert scalar tensors to ordinary numbers for metrics output."""

        return {
            "total_loss": float(self.total.detach().item()),
            "policy_loss": float(self.policy.detach().item()),
            "value_loss": float(self.value.detach().item()),
            "l2_loss": float(self.l2.detach().item()),
        }


def policy_value_loss(
    policy_logits: Tensor,
    predicted_value: Tensor,
    target_policy: Tensor,
    target_value: Tensor,
    *,
    model: nn.Module | None = None,
    l2_regularization: float = 0.0,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    legal_action_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> LossComponents:
    """Compute soft-label policy CE + value MSE + explicit L2.

    Policies are probability vectors rather than class indices so this same
    function will later accept a search visit distribution, not just Phase 1's
    one-hot classical move target.
    """

    if policy_logits.ndim != 2:
        raise ValueError("policy_logits must have shape (batch, actions)")
    if target_policy.shape != policy_logits.shape:
        raise ValueError(
            "target_policy must have the same shape as policy_logits; "
            f"got {tuple(target_policy.shape)} and {tuple(policy_logits.shape)}"
        )
    if predicted_value.ndim == 1:
        predicted_value = predicted_value.unsqueeze(1)
    if target_value.ndim == 1:
        target_value = target_value.unsqueeze(1)
    if predicted_value.shape != target_value.shape or predicted_value.shape[1:] != (1,):
        raise ValueError("predicted_value and target_value must have shape (batch, 1)")
    if predicted_value.shape[0] != policy_logits.shape[0]:
        raise ValueError("policy and value batches must contain the same number of examples")
    if l2_regularization < 0.0:
        raise ValueError("l2_regularization cannot be negative")
    if policy_loss_weight < 0.0:
        raise ValueError("policy_loss_weight cannot be negative")
    if value_loss_weight < 0.0:
        raise ValueError("value_loss_weight cannot be negative")
    if policy_loss_weight == 0.0 and value_loss_weight == 0.0:
        raise ValueError("at least one policy/value loss weight must be positive")

    normalized_weight: Tensor | None = None
    if sample_weight is not None:
        normalized_weight = sample_weight.reshape(-1).to(
            device=policy_logits.device,
            dtype=policy_logits.dtype,
        )
        if normalized_weight.shape[0] != policy_logits.shape[0]:
            raise ValueError("sample_weight must contain one value per example")
        if not torch.isfinite(normalized_weight).all():
            raise ValueError("sample_weight must be finite")
        if (normalized_weight <= 0.0).any():
            raise ValueError("sample_weight values must be positive")

    selected_logits = policy_logits
    if legal_action_mask is not None:
        if legal_action_mask.shape != policy_logits.shape:
            raise ValueError("legal_action_mask must have the same shape as policy_logits")
        legal = legal_action_mask.to(dtype=torch.bool)
        if not legal.any(dim=1).all():
            raise ValueError("every example must contain at least one legal action")
        if (target_policy.masked_select(~legal) > 0.0).any():
            raise ValueError("target_policy assigns probability to an illegal action")
        selected_logits = policy_logits.masked_fill(~legal, torch.finfo(policy_logits.dtype).min)

    log_probabilities = F.log_softmax(selected_logits, dim=1)
    # Avoid ``0 * -inf`` for illegal actions under legal-move masking.
    policy_terms = torch.where(
        target_policy > 0.0,
        target_policy * log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    policy_per_example = -policy_terms.sum(dim=1)
    value_per_example = (predicted_value - target_value).square().mean(dim=1)
    if normalized_weight is None:
        policy_loss = policy_per_example.mean()
        value_loss = value_per_example.mean()
    else:
        weight_total = normalized_weight.sum()
        policy_loss = (policy_per_example * normalized_weight).sum() / weight_total
        value_loss = (value_per_example * normalized_weight).sum() / weight_total
    l2_loss = policy_logits.new_zeros(())
    if l2_regularization > 0.0:
        if model is None:
            raise ValueError("model is required when l2_regularization is non-zero")
        squared_weights = [parameter.square().sum() for parameter in model.parameters()]
        if squared_weights:
            l2_loss = torch.stack(squared_weights).sum() * l2_regularization
    return LossComponents(
        total=(policy_loss * policy_loss_weight + value_loss * value_loss_weight + l2_loss),
        policy=policy_loss,
        value=value_loss,
        l2=l2_loss,
    )


def policy_accuracy(
    policy_logits: Tensor,
    target_policy: Tensor,
    *,
    top_k: int = 1,
    legal_action_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Fraction whose target move appears among the top-k policy logits."""

    if policy_logits.ndim != 2 or target_policy.shape != policy_logits.shape:
        raise ValueError("policy tensors must share shape (batch, actions)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    selected_logits = policy_logits
    if legal_action_mask is not None:
        if legal_action_mask.shape != policy_logits.shape:
            raise ValueError("legal_action_mask must have the same shape as policy logits")
        legal = legal_action_mask.to(dtype=torch.bool)
        if not legal.any(dim=1).all():
            raise ValueError("every example must contain at least one legal action")
        selected_logits = policy_logits.masked_fill(~legal, torch.finfo(policy_logits.dtype).min)
    actual_k = min(top_k, policy_logits.shape[1])
    target_actions = target_policy.argmax(dim=1, keepdim=True)
    predicted_actions = selected_logits.topk(actual_k, dim=1).indices
    correct = predicted_actions.eq(target_actions).any(dim=1).float()
    if sample_weight is None:
        return correct.mean()
    normalized_weight = sample_weight.reshape(-1).to(
        device=policy_logits.device,
        dtype=policy_logits.dtype,
    )
    if normalized_weight.shape[0] != policy_logits.shape[0]:
        raise ValueError("sample_weight must contain one value per example")
    if not torch.isfinite(normalized_weight).all() or (normalized_weight <= 0.0).any():
        raise ValueError("sample_weight values must be finite and positive")
    return (correct * normalized_weight).sum() / normalized_weight.sum()


def value_mean_absolute_error(predicted_value: Tensor, target_value: Tensor) -> Tensor:
    """Mean absolute distance between predicted and final perspective value."""

    if predicted_value.shape != target_value.shape:
        raise ValueError("predicted_value and target_value must share a shape")
    return (predicted_value - target_value).abs().mean()


class PolicyValueLoss(nn.Module):
    """Module wrapper convenient for notebooks and custom training loops."""

    def __init__(
        self,
        *,
        l2_regularization: float = 0.0,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if l2_regularization < 0.0:
            raise ValueError("l2_regularization cannot be negative")
        if policy_loss_weight < 0.0 or value_loss_weight < 0.0:
            raise ValueError("policy/value loss weights cannot be negative")
        if policy_loss_weight == 0.0 and value_loss_weight == 0.0:
            raise ValueError("at least one policy/value loss weight must be positive")
        self.l2_regularization = l2_regularization
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight

    def forward(
        self,
        policy_logits: Tensor,
        predicted_value: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
        *,
        model: nn.Module | None = None,
        legal_action_mask: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> LossComponents:
        """Delegate to :func:`policy_value_loss`."""

        return policy_value_loss(
            policy_logits,
            predicted_value,
            target_policy,
            target_value,
            model=model,
            l2_regularization=self.l2_regularization,
            policy_loss_weight=self.policy_loss_weight,
            value_loss_weight=self.value_loss_weight,
            legal_action_mask=legal_action_mask,
            sample_weight=sample_weight,
        )
