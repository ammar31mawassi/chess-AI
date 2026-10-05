"""A compact residual policy-value network.

The network deliberately stays small enough for CPU experiments.  Its shared
convolutional trunk learns board features; two heads then answer different
questions: "which move?" (policy) and "who is favoured?" (value).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape-defining settings saved in every checkpoint."""

    channels: int = 32
    residual_blocks: int = 2
    input_planes: int = 18
    action_size: int = 4208

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be a positive integer")
        if self.residual_blocks < 0:
            raise ValueError("residual_blocks cannot be negative")
        if self.input_planes <= 0:
            raise ValueError("input_planes must be a positive integer")
        if self.action_size <= 1:
            raise ValueError("action_size must be greater than one")

    def to_dict(self) -> dict[str, int]:
        """Return a JSON/checkpoint-friendly representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        """Validate and construct a model configuration from metadata."""

        required = {"channels", "residual_blocks", "input_planes", "action_size"}
        missing = required.difference(raw)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Model configuration is missing: {joined}")
        try:
            return cls(**{name: int(raw[name]) for name in required})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model configuration: {raw!r}") from exc


class ResidualBlock(nn.Module):
    """Two convolutions with an identity skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform *inputs* while preserving its shape."""

        residual = inputs
        features = self.activation(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        return self.activation(features + residual)


class PolicyValueNet(nn.Module):
    """Small convolutional network producing policy logits and a value.

    Parameters may be passed individually for an approachable public API, or
    through :class:`ModelConfig` when reconstructing an exact checkpoint.
    """

    def __init__(
        self,
        channels: int = 32,
        residual_blocks: int = 2,
        *,
        input_planes: int = 18,
        action_size: int = 4208,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            channels=channels,
            residual_blocks=residual_blocks,
            input_planes=input_planes,
            action_size=action_size,
        )

        self.input_block = nn.Sequential(
            nn.Conv2d(
                self.config.input_planes,
                self.config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.config.channels),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(self.config.channels) for _ in range(self.config.residual_blocks))
        )

        policy_channels = 2
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.config.channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_channels * 8 * 8, self.config.action_size),
        )

        value_hidden = max(32, self.config.channels)
        self.value_features = nn.Sequential(
            nn.Conv2d(self.config.channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

        self._initialize_weights()

    @property
    def action_size(self) -> int:
        """Number of logits in the fixed move action space."""

        return self.config.action_size

    @property
    def input_planes(self) -> int:
        """Number of planes expected for each board."""

        return self.config.input_planes

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, boards: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate a batch shaped ``(batch, 18, 8, 8)``.

        A clear validation error here tends to be much more useful to a learner
        than the lower-level convolution error that would otherwise follow.
        """

        if boards.ndim != 4:
            raise ValueError(
                "boards must have four dimensions: (batch, planes, rank, file); "
                f"received shape {tuple(boards.shape)}"
            )
        expected_tail = (self.config.input_planes, 8, 8)
        if tuple(boards.shape[1:]) != expected_tail:
            raise ValueError(
                f"boards must have shape (batch, {expected_tail[0]}, 8, 8); "
                f"received {tuple(boards.shape)}"
            )
        features = self.residual_tower(self.input_block(boards))
        policy_logits = self.policy_head(features)
        value = self.value_head(self.value_features(features))
        return policy_logits, value
