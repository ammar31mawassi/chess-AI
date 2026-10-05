"""Policy-network chess agent with mandatory legal-move masking."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import CheckpointError, load_model


class NeuralAgentError(RuntimeError):
    """Raised when a neural agent cannot safely select a move."""


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or a CUDA device with a useful error."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except RuntimeError as exc:
        raise NeuralAgentError(f"Invalid PyTorch device {requested!r}: {exc}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NeuralAgentError(
            "A CUDA device was requested, but this PyTorch installation cannot access CUDA. "
            "Use --device cpu or install a CUDA-compatible PyTorch build."
        )
    return device


class NeuralAgent:
    """Choose legal moves greedily from a saved policy-value checkpoint.

    ``temperature=0`` (or ``deterministic=True``) selects the largest legal
    logit. A positive temperature samples reproducibly from the legal softmax
    distribution when a seed is supplied. The value head is not used to pick a
    move in this lightweight baseline. :class:`NeuralMCTSAgent` combines both
    heads when stronger, search-backed play is wanted.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = True,
        temperature: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        self.checkpoint = Path(checkpoint)
        self.device = resolve_device(device)
        self.deterministic = deterministic
        self.temperature = temperature
        self.name = name or f"neural:{self.checkpoint.name}"
        self._rng = np.random.default_rng(seed)
        self._board_encoder = BoardEncoder()
        self._move_encoder = MoveEncoder()
        try:
            self.model = load_model(self.checkpoint, device=self.device)
        except CheckpointError as exc:
            raise NeuralAgentError(
                f"Could not load neural checkpoint {self.checkpoint}: {exc}"
            ) from exc
        if self.model.input_planes != BoardEncoder.NUM_PLANES:
            raise NeuralAgentError(
                f"Checkpoint expects {self.model.input_planes} input planes, but this version "
                f"uses {BoardEncoder.NUM_PLANES}."
            )
        if self.model.action_size != self._move_encoder.action_size:
            raise NeuralAgentError(
                f"Checkpoint has {self.model.action_size} policy actions, but this version "
                f"uses {self._move_encoder.action_size}."
            )

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Return full policy logits and a side-to-move value without mutation."""

        encoded = self._board_encoder.encode(board)
        batch = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            policy_logits, predicted_value = self.model(batch)
        logits = policy_logits[0].detach().to("cpu").numpy().astype(np.float64, copy=False)
        value = float(predicted_value[0, 0].detach().to("cpu").item())
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise NeuralAgentError("The checkpoint produced a non-finite policy or value")
        return logits, value

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Evaluate *board*, exclude every illegal action, and choose a move."""

        if board.is_game_over(claim_draw=True):
            raise NeuralAgentError("Cannot choose a move because the game is already over")

        logits, _value = self.evaluate(board)

        legal_mask = self._move_encoder.legal_action_mask(board).astype(bool)
        legal_indexes = np.flatnonzero(legal_mask)
        if legal_indexes.size == 0:
            raise NeuralAgentError("No legal moves are available")
        legal_logits = logits[legal_indexes]

        if self.deterministic or self.temperature == 0:
            # np.argmax returns the first maximum, making ties deterministic.
            selected = int(legal_indexes[int(np.argmax(legal_logits))])
        else:
            scaled = legal_logits / self.temperature
            scaled -= np.max(scaled)  # numerical stability before exponentiation
            probabilities = np.exp(scaled)
            total = float(probabilities.sum())
            if not np.isfinite(total) or total <= 0:
                raise NeuralAgentError("Could not form a finite policy distribution")
            probabilities /= total
            selected = int(self._rng.choice(legal_indexes, p=probabilities))

        move = self._move_encoder.decode(selected)
        if move not in board.legal_moves:  # defensive assertion at the integration boundary
            raise NeuralAgentError(
                f"Internal move-encoding error: selected illegal move {move.uci()}"
            )
        return move
