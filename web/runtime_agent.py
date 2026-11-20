"""ONNX inference adapter for the shared champion search."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import chess
import numpy as np
import onnxruntime as ort

from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder


class NeuralAgentError(RuntimeError):
    """The runtime cannot evaluate the position."""


@lru_cache(maxsize=1)
def _load_model(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class NeuralAgent:
    def __init__(self, checkpoint: str | Path, **_options: Any) -> None:
        self._session = _load_model(Path(checkpoint))
        self._board_encoder = BoardEncoder()
        self._move_encoder = MoveEncoder()

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        logits, value = self._session.run(None, {"board": self._board_encoder.encode(board)[None]})
        return logits[0], float(value.item())
