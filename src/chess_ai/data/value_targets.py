"""Explicit semantics for Stockfish position-evaluation targets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

STOCKFISH_VALUE_KIND = "stockfish_cp_tanh_v1"
SHARDED_DATASET_FORMAT = "self-improving-chess-ai.stockfish-evaluation-shards"


def validate_value_spec(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Evaluation dataset/checkpoint requires a value_target_spec mapping")
    spec = dict(raw)
    if spec.get("kind") != STOCKFISH_VALUE_KIND or spec.get("perspective") != "side_to_move":
        raise ValueError("Unsupported evaluation value target kind or perspective")
    scale = spec.get("scale_cp")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Evaluation value scale_cp must be finite and positive")
    return spec
