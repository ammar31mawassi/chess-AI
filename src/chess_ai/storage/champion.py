"""Explicit, checksum-bound gameplay champion and its inference settings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from chess_ai.storage.artifacts import sha256_file

CHAMPION_FORMAT = "self-improving-chess-ai.gameplay-champion"


@dataclass(frozen=True, slots=True)
class GameplayChampion:
    checkpoint: Path
    search_simulations: int
    c_puct: float


def load_gameplay_champion(root: str | Path = "checkpoints") -> GameplayChampion | None:
    """Return the registered champion, failing clearly if its record or copy changed."""

    checkpoint_root = Path(root)
    manifest = checkpoint_root / "champion.json"
    if not manifest.exists():
        return None
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("format") != CHAMPION_FORMAT
            or type(raw.get("version")) is not int
            or raw["version"] != 1
        ):
            raise ValueError("unsupported format/version")
        relative = Path(raw["checkpoint"])
        checkpoint = checkpoint_root / relative
        if relative.is_absolute() or not checkpoint.resolve().is_relative_to(
            checkpoint_root.resolve()
        ):
            raise ValueError("checkpoint must be relative to the checkpoint directory")
        inference = raw["inference"]
        simulations = inference["search_simulations"]
        c_puct = inference["c_puct"]
        if type(simulations) is not int or simulations < 0:
            raise ValueError("search_simulations must be a non-negative integer")
        if (
            isinstance(c_puct, bool)
            or not isinstance(c_puct, int | float)
            or not math.isfinite(c_puct)
            or c_puct <= 0
        ):
            raise ValueError("c_puct must be finite and positive")
        if inference.get("opening_book") is not None:
            raise ValueError("this champion record version supports model-only play")
        if sha256_file(checkpoint) != raw["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        return GameplayChampion(checkpoint, simulations, float(c_puct))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid gameplay champion {manifest}: {exc}") from exc
