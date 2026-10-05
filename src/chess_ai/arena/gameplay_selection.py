"""Select a provisional epoch checkpoint through fixed gameplay suites."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from chess_ai.arena.paired_audit import (
    _checkpoint_results,
    _excluded_opening_fens,
    _opening_suite,
)
from chess_ai.model.checkpoint import LoadedCheckpoint, load_checkpoint

GAMEPLAY_SELECTION_FORMAT = "self-improving-chess-ai.gameplay-epoch-selection"
GAMEPLAY_SELECTION_VERSION = 1


class GameplaySelectionError(RuntimeError):
    """Raised when epoch selection cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class GameplaySelectionConfig:
    """Settings for gameplay-first selection from retained epoch checkpoints."""

    champion_checkpoint: Path
    candidate_checkpoint_dir: Path
    training_manifest: Path
    report_path: Path
    pgn_dir: Path
    opponent_depths: tuple[int, ...] = (1, 2)
    openings: int = 10
    opening_min_full_moves: int = 2
    opening_max_full_moves: int = 3
    max_plies: int = 200
    minimum_aggregate_improvement_points: float = 0.5
    require_depth_non_regression: bool = True
    require_standard_non_regression: bool = True
    seed: int = 9_700_205
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "champion_checkpoint",
            "candidate_checkpoint_dir",
            "training_manifest",
            "report_path",
            "pgn_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "opponent_depths", tuple(self.opponent_depths))
        if not self.opponent_depths or any(
            isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0
            for depth in self.opponent_depths
        ):
            raise ValueError("opponent_depths must contain positive integers")
        if len(set(self.opponent_depths)) != len(self.opponent_depths):
            raise ValueError("opponent_depths cannot contain duplicates")
        if (
            isinstance(self.openings, bool)
            or not isinstance(self.openings, int)
            or self.openings < 2
        ):
            raise ValueError("openings must be an integer of at least 2")
        if self.opening_min_full_moves < 0 or (
            self.opening_max_full_moves < self.opening_min_full_moves
        ):
            raise ValueError("opening full-move bounds are invalid")
        if self.max_plies <= 0 or self.opening_max_full_moves * 2 >= self.max_plies:
            raise ValueError("max_plies must exceed the longest opening")
        if not math.isfinite(self.minimum_aggregate_improvement_points) or (
            self.minimum_aggregate_improvement_points < 0.0
        ):
            raise ValueError("minimum_aggregate_improvement_points must be finite and non-negative")
        if not isinstance(self.require_depth_non_regression, bool):
            raise ValueError("require_depth_non_regression must be a boolean")
        if not isinstance(self.require_standard_non_regression, bool):
            raise ValueError("require_standard_non_regression must be a boolean")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        seed: int = 9_700_205,
        device: str = "auto",
    ) -> GameplaySelectionConfig:
        values = dict(raw)
        values.setdefault("seed", seed)
        values.setdefault("device", device)
        path_fields = {
            "champion_checkpoint",
            "candidate_checkpoint_dir",
            "training_manifest",
            "report_path",
            "pgn_dir",
        }
        for name in path_fields:
            if name in values:
                values[name] = Path(str(values[name]))
        if "opponent_depths" in values:
            raw_depths = values["opponent_depths"]
            if not isinstance(raw_depths, Sequence) or isinstance(raw_depths, (str, bytes)):
                raise ValueError("opponent_depths must be a sequence of integers")
            values["opponent_depths"] = tuple(raw_depths)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown gameplay_selection settings: {', '.join(unknown)}")
        missing = sorted(path_fields.difference(values))
        if missing:
            raise ValueError(f"Missing gameplay_selection settings: {', '.join(missing)}")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GameplaySelectionSummary:
    """Provisional selection result; no champion is promoted or overwritten."""

    champion_checkpoint: Path
    selected_checkpoint: Path
    selected_epoch: int | None
    selected_is_candidate: bool
    aggregate_point_delta: float
    evaluated_candidates: int
    opponent_depths: tuple[int, ...]
    openings: int
    games_per_checkpoint: int
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("champion_checkpoint", "selected_checkpoint", "report_path"):
            payload[key] = str(payload[key])
        return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _validation_loss(checkpoint: LoadedCheckpoint) -> float:
    raw = checkpoint.metrics.get("validation_loss")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    return math.inf


def _candidate_paths(config: GameplaySelectionConfig) -> list[Path]:
    paths = sorted(config.candidate_checkpoint_dir.glob("epoch_*.pt"))
    best_path = config.candidate_checkpoint_dir / "best.pt"
    if best_path.is_file():
        best = load_checkpoint(best_path, map_location="cpu")
        retained_epochs = {load_checkpoint(path, map_location="cpu").epoch for path in paths}
        if best.epoch not in retained_epochs:
            paths.append(best_path)
    if not paths:
        raise GameplaySelectionError(
            f"No retained epoch or best checkpoint exists in {config.candidate_checkpoint_dir}"
        )
    return paths


def _aggregate_points(depth_results: Mapping[str, Mapping[str, Any]]) -> float:
    return sum(float(stats["points"]) for stats in depth_results.values())


def _candidate_eligibility(
    candidate: Mapping[str, Any],
    champion: Mapping[str, Any],
    *,
    config: GameplaySelectionConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    delta = float(candidate["aggregate_points"]) - float(champion["aggregate_points"])
    if delta < config.minimum_aggregate_improvement_points:
        reasons.append(
            f"aggregate point gain is below {config.minimum_aggregate_improvement_points:g}"
        )
    for depth in config.opponent_depths:
        key = f"d{depth}"
        candidate_stats = candidate["by_opponent_depth"][key]
        champion_stats = champion["by_opponent_depth"][key]
        if config.require_depth_non_regression and (
            float(candidate_stats["points"]) < float(champion_stats["points"])
        ):
            reasons.append(f"total score regressed against D{depth}")
        if config.require_standard_non_regression and (
            float(candidate_stats["standard_points"]) < float(champion_stats["standard_points"])
        ):
            reasons.append(f"standard-start score regressed against D{depth}")
    return not reasons, reasons


def _selection_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    depth_results = candidate["by_opponent_depth"]
    ordered_depths = sorted(
        (int(key.removeprefix("d")), stats) for key, stats in depth_results.items()
    )
    # Gameplay dominates. Stronger-opponent points break aggregate ties, then
    # validation loss and the earlier epoch make the final choice reproducible.
    depth_points = tuple(float(stats["points"]) for _depth, stats in reversed(ordered_depths))
    return (
        float(candidate["aggregate_points"]),
        *depth_points,
        -float(candidate["validation_loss"]),
        -float(candidate["epoch"]),
    )


def _choose_best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    return max(eligible, key=_selection_key) if eligible else None


def run_gameplay_selection(config: GameplaySelectionConfig) -> GameplaySelectionSummary:
    """Evaluate every epoch through gameplay and select without promotion."""

    for path, label in (
        (config.champion_checkpoint, "Champion checkpoint"),
        (config.training_manifest, "Training manifest"),
    ):
        if not path.is_file():
            raise GameplaySelectionError(f"{label} does not exist: {path}")
    if config.report_path.exists() or any(config.pgn_dir.glob("**/*.pgn")):
        raise GameplaySelectionError(
            "Gameplay-selection outputs already exist; choose fresh report_path and pgn_dir values."
        )
    champion_loaded = load_checkpoint(config.champion_checkpoint, map_location="cpu")
    candidates: list[tuple[Path, LoadedCheckpoint]] = []
    for path in _candidate_paths(config):
        loaded = load_checkpoint(path, map_location="cpu")
        if loaded.model.config != champion_loaded.model.config:
            raise GameplaySelectionError(
                f"Candidate architecture differs from the champion: {path}"
            )
        candidates.append((path, loaded))

    excluded_fens = _excluded_opening_fens(config.training_manifest)
    suite = _opening_suite(
        openings=config.openings,
        seed=config.seed,
        minimum_full_moves=config.opening_min_full_moves,
        maximum_full_moves=config.opening_max_full_moves,
        excluded_fens=excluded_fens,
    )

    def evaluate(path: Path, role: str) -> dict[str, Any]:
        depth_results: dict[str, dict[str, float | int]] = {}
        depth_games: dict[str, list[dict[str, Any]]] = {}
        for depth in config.opponent_depths:
            key = f"d{depth}"
            stats, games = _checkpoint_results(
                path,
                role=f"{role}_{key}",
                opening_suite=suite,
                opponent_depth=depth,
                max_plies=config.max_plies,
                seed=config.seed + depth * 10_007,
                device=config.device,
                pgn_dir=config.pgn_dir,
            )
            depth_results[key] = stats
            depth_games[key] = games
        return {
            "checkpoint": str(path),
            "sha256": _sha256_file(path),
            "by_opponent_depth": depth_results,
            "games": depth_games,
            "aggregate_points": _aggregate_points(depth_results),
        }

    champion_result = evaluate(config.champion_checkpoint, "champion")
    candidate_results: list[dict[str, Any]] = []
    for path, loaded in candidates:
        result = evaluate(path, f"epoch_{loaded.epoch:04d}")
        result["epoch"] = loaded.epoch
        result["validation_loss"] = _validation_loss(loaded)
        result["aggregate_point_delta"] = float(result["aggregate_points"]) - float(
            champion_result["aggregate_points"]
        )
        eligible, reasons = _candidate_eligibility(result, champion_result, config=config)
        result["eligible"] = eligible
        result["rejection_reasons"] = reasons
        candidate_results.append(result)

    selected = _choose_best_candidate(candidate_results)
    selected_checkpoint = (
        Path(str(selected["checkpoint"])) if selected is not None else config.champion_checkpoint
    )
    selected_epoch = int(selected["epoch"]) if selected is not None else None
    selected_delta = float(selected["aggregate_point_delta"]) if selected is not None else 0.0
    report = {
        "format": GAMEPLAY_SELECTION_FORMAT,
        "version": GAMEPLAY_SELECTION_VERSION,
        "kind": "gameplay_epoch_selection",
        "selection_seed": config.seed,
        "training_manifest": str(config.training_manifest),
        "openings": [
            {"index": index, "fen": board.fen(), "moves": moves, "seed": opening_seed}
            for index, (board, moves, opening_seed) in enumerate(suite)
        ],
        "opponent_depths": list(config.opponent_depths),
        "games_per_checkpoint": config.openings * 2 * len(config.opponent_depths),
        "minimum_aggregate_improvement_points": config.minimum_aggregate_improvement_points,
        "require_depth_non_regression": config.require_depth_non_regression,
        "require_standard_non_regression": config.require_standard_non_regression,
        "champion": champion_result,
        "candidates": candidate_results,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_epoch": selected_epoch,
        "selected_is_candidate": selected is not None,
        "aggregate_point_delta": selected_delta,
        "selection_note": (
            "Provisional gameplay selection only; use fresh paired audits before promotion."
        ),
        "training_updates": 0,
        "promotion_performed": False,
    }
    _atomic_json(config.report_path, report)
    return GameplaySelectionSummary(
        champion_checkpoint=config.champion_checkpoint,
        selected_checkpoint=selected_checkpoint,
        selected_epoch=selected_epoch,
        selected_is_candidate=selected is not None,
        aggregate_point_delta=selected_delta,
        evaluated_candidates=len(candidate_results),
        opponent_depths=config.opponent_depths,
        openings=config.openings,
        games_per_checkpoint=config.openings * 2 * len(config.opponent_depths),
        report_path=config.report_path,
    )
