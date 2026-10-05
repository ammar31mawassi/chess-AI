"""Evaluation-only paired-opening audit for a candidate and frozen champion."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import torch

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.arena.match import run_match
from chess_ai.model.checkpoint import load_checkpoint


class PairedAuditError(RuntimeError):
    """Raised when a holdout comparison cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class PairedAuditSummary:
    """Candidate/champion results against the same D1 opening suite."""

    candidate_checkpoint: Path
    champion_checkpoint: Path
    openings: int
    games_per_checkpoint: int
    candidate_wins: int
    candidate_draws: int
    candidate_losses: int
    candidate_points: float
    candidate_standard_points: float
    champion_wins: int
    champion_draws: int
    champion_losses: int
    champion_points: float
    champion_standard_points: float
    point_delta: float
    standard_non_regression: bool
    supports_promotion: bool
    verdict: str
    audit_seed: int
    search_simulations: int
    c_puct: float
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("candidate_checkpoint", "champion_checkpoint", "report_path"):
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


def _excluded_opening_fens(manifest_path: Path | None) -> set[str]:
    if manifest_path is None:
        return set()
    if not manifest_path.is_file():
        raise PairedAuditError(f"Training manifest does not exist: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedAuditError(f"Could not read training manifest: {exc}") from exc
    pairs = raw.get("pairs") if isinstance(raw, dict) else None
    if not isinstance(pairs, list):
        raise PairedAuditError("Training manifest has no valid paired-opening list")
    fens: set[str] = set()
    for index, item in enumerate(pairs, start=1):
        raw_fen = item.get("opening_fen") if isinstance(item, dict) else None
        if not isinstance(raw_fen, str) or not raw_fen.strip():
            raise PairedAuditError(
                f"Training manifest pair {index} has a missing opening FEN entry"
            )
        try:
            board = chess.Board(raw_fen)
        except ValueError as exc:
            raise PairedAuditError(
                f"Training manifest pair {index} has an invalid opening FEN: {exc}"
            ) from exc
        if not board.is_valid():
            raise PairedAuditError(
                f"Training manifest pair {index} contains an invalid chess position"
            )
        # Two different random seeds can legitimately transpose into the same
        # position. A set is exactly what the holdout collision check needs;
        # duplicates are not malformed data.
        fens.add(board.fen())
    return fens


def _opening_suite(
    *,
    openings: int,
    seed: int,
    minimum_full_moves: int,
    maximum_full_moves: int,
    excluded_fens: set[str],
) -> list[tuple[chess.Board, list[str], int]]:
    suite: list[tuple[chess.Board, list[str], int]] = [(chess.Board(), [], seed)]
    seen = {chess.Board().fen(), *excluded_fens}
    lengths = list(range(minimum_full_moves, maximum_full_moves + 1))
    for opening_index in range(1, openings):
        base_seed = seed + opening_index * 1_000_003
        for attempt in range(1_000):
            opening_seed = base_seed + attempt
            rng = random.Random(opening_seed)
            board = chess.Board()
            moves: list[str] = []
            requested_plies = rng.choice(lengths) * 2
            for _ in range(requested_plies):
                legal = sorted(board.legal_moves, key=lambda move: move.uci())
                move = rng.choice(legal)
                board.push(move)
                moves.append(move.uci())
                if board.is_game_over(claim_draw=True):
                    break
            if (
                len(moves) == requested_plies
                and not board.is_game_over(claim_draw=True)
                and board.fen() not in seen
            ):
                seen.add(board.fen())
                suite.append((board, moves, opening_seed))
                break
        else:
            raise PairedAuditError("Could not create a collision-free holdout opening suite")
    return suite


def _checkpoint_results(
    checkpoint: Path,
    *,
    role: str,
    opening_suite: Sequence[tuple[chess.Board, list[str], int]],
    opponent_depth: int,
    max_plies: int,
    seed: int,
    device: str,
    pgn_dir: Path,
    search_simulations: int = 0,
    c_puct: float = 1.5,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    neural = (
        NeuralMCTSAgent(
            checkpoint,
            device=device,
            simulations=search_simulations,
            c_puct=c_puct,
            seed=seed,
            name=f"{role} PUCT ({checkpoint.name})",
        )
        if search_simulations > 0
        else NeuralAgent(
            checkpoint,
            device=device,
            deterministic=True,
            temperature=0.0,
            seed=seed,
            name=f"{role} ({checkpoint.name})",
        )
    )
    games: list[dict[str, Any]] = []
    try:
        for opening_index, (board, moves, opening_seed) in enumerate(opening_suite):
            for neural_color in (chess.WHITE, chess.BLACK):
                color_name = "white" if neural_color == chess.WHITE else "black"
                opponent = MinimaxAgent(
                    depth=opponent_depth,
                    deterministic=True,
                    seed=seed + opening_index * 2 + int(neural_color),
                    name=f"Minimax D{opponent_depth} opponent",
                )
                white = neural if neural_color == chess.WHITE else opponent
                black = opponent if neural_color == chess.WHITE else neural
                pgn_path = pgn_dir / role / (f"opening_{opening_index:03d}_neural_{color_name}.pgn")
                result = run_match(
                    white,
                    black,
                    max_plies=max_plies - len(moves),
                    starting_fen=board.fen(),
                    seed=seed + opening_index * 2 + (0 if neural_color == chess.WHITE else 1),
                    pgn_path=pgn_path,
                    extra_headers={
                        "EvaluationKind": "paired-candidate-champion-holdout",
                        "EvaluationRole": role,
                        "OpeningIndex": str(opening_index),
                        "OpeningSeed": str(opening_seed),
                        "OpeningMoves": " ".join(moves),
                        "NeuralColor": color_name,
                        "Checkpoint": str(checkpoint),
                        "OpponentDepth": str(opponent_depth),
                        "TrainingData": "false",
                    },
                )
                if result.result == "1/2-1/2":
                    outcome = "draw"
                else:
                    neural_won = (result.result == "1-0") == (neural_color == chess.WHITE)
                    outcome = "win" if neural_won else "loss"
                games.append(
                    {
                        "opening_index": opening_index,
                        "opening_seed": opening_seed,
                        "neural_color": color_name,
                        "result": result.result,
                        "neural_outcome": outcome,
                        "termination": result.termination,
                        "plies_after_opening": result.plies,
                        "pgn_path": str(result.pgn_path),
                    }
                )
    finally:
        del neural
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wins = sum(game["neural_outcome"] == "win" for game in games)
    draws = sum(game["neural_outcome"] == "draw" for game in games)
    losses = sum(game["neural_outcome"] == "loss" for game in games)
    standard_games = [game for game in games if game["opening_index"] == 0]
    standard_points = sum(
        1.0 if game["neural_outcome"] == "win" else 0.5 if game["neural_outcome"] == "draw" else 0.0
        for game in standard_games
    )
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": wins + 0.5 * draws,
        "standard_points": standard_points,
    }, games


def run_paired_audit(
    *,
    candidate_checkpoint: str | Path,
    champion_checkpoint: str | Path,
    openings: int = 20,
    audit_seed: int = 9_300_205,
    opponent_depth: int = 1,
    opening_min_full_moves: int = 2,
    opening_max_full_moves: int = 3,
    max_plies: int = 200,
    minimum_improvement_points: float = 0.5,
    device: str = "auto",
    pgn_dir: str | Path = Path("data/games/evaluation/paired_audit"),
    exclude_manifest: str | Path | None = None,
    search_simulations: int = 0,
    c_puct: float = 1.5,
) -> PairedAuditSummary:
    """Compare two checkpoints on identical unseen starts without promotion."""

    candidate = Path(candidate_checkpoint)
    champion = Path(champion_checkpoint)
    destination = Path(pgn_dir)
    manifest = Path(exclude_manifest) if exclude_manifest is not None else None
    for path, label in ((candidate, "Candidate"), (champion, "Champion")):
        if not path.is_file():
            raise PairedAuditError(f"{label} checkpoint does not exist: {path}")
        load_checkpoint(path, map_location="cpu")
    if candidate.resolve() == champion.resolve():
        raise PairedAuditError("Candidate and champion checkpoints must be different files")
    if isinstance(openings, bool) or not isinstance(openings, int) or openings < 2:
        raise ValueError("openings must be an integer of at least 2")
    if opponent_depth <= 0 or max_plies <= 0:
        raise ValueError("opponent_depth and max_plies must be positive")
    if opening_min_full_moves < 0 or opening_max_full_moves < opening_min_full_moves:
        raise ValueError("opening full-move bounds are invalid")
    if opening_max_full_moves * 2 >= max_plies:
        raise ValueError("the longest opening must be shorter than max_plies")
    if minimum_improvement_points < 0.0:
        raise ValueError("minimum_improvement_points cannot be negative")
    if isinstance(search_simulations, bool) or search_simulations < 0:
        raise ValueError("search_simulations must be a non-negative integer")
    if not isinstance(search_simulations, int):
        raise ValueError("search_simulations must be a non-negative integer")
    if not math.isfinite(c_puct) or c_puct <= 0.0:
        raise ValueError("c_puct must be finite and positive")
    conflicts = [*destination.glob("**/*.pgn"), destination / "report.json"]
    conflicts = [path for path in conflicts if path.exists()]
    if conflicts:
        preview = ", ".join(str(path) for path in conflicts[:3])
        raise PairedAuditError(
            f"Paired-audit output already exists ({preview}); choose a fresh pgn_dir."
        )

    excluded_fens = _excluded_opening_fens(manifest)
    suite = _opening_suite(
        openings=openings,
        seed=audit_seed,
        minimum_full_moves=opening_min_full_moves,
        maximum_full_moves=opening_max_full_moves,
        excluded_fens=excluded_fens,
    )
    candidate_stats, candidate_games = _checkpoint_results(
        candidate,
        role="candidate",
        opening_suite=suite,
        opponent_depth=opponent_depth,
        max_plies=max_plies,
        seed=audit_seed,
        device=device,
        pgn_dir=destination,
        search_simulations=search_simulations,
        c_puct=c_puct,
    )
    champion_stats, champion_games = _checkpoint_results(
        champion,
        role="champion",
        opening_suite=suite,
        opponent_depth=opponent_depth,
        max_plies=max_plies,
        seed=audit_seed,
        device=device,
        pgn_dir=destination,
        search_simulations=search_simulations,
        c_puct=c_puct,
    )
    point_delta = float(candidate_stats["points"]) - float(champion_stats["points"])
    standard_non_regression = float(candidate_stats["standard_points"]) >= float(
        champion_stats["standard_points"]
    )
    supports_promotion = point_delta >= minimum_improvement_points and standard_non_regression
    verdict = "improved" if point_delta > 0.0 else "tied" if point_delta == 0.0 else "regressed"
    report_path = destination / "report.json"
    report = {
        "kind": "paired_candidate_champion_holdout",
        "audit_seed": audit_seed,
        "opponent_depth": opponent_depth,
        "search_simulations": search_simulations,
        "c_puct": c_puct,
        "openings": [
            {
                "index": index,
                "fen": board.fen(),
                "moves": moves,
                "seed": opening_seed,
            }
            for index, (board, moves, opening_seed) in enumerate(suite)
        ],
        "excluded_training_manifest": str(manifest) if manifest is not None else None,
        "candidate_checkpoint": str(candidate),
        "candidate_sha256": _sha256_file(candidate),
        "candidate_stats": candidate_stats,
        "candidate_games": candidate_games,
        "champion_checkpoint": str(champion),
        "champion_sha256": _sha256_file(champion),
        "champion_stats": champion_stats,
        "champion_games": champion_games,
        "point_delta": point_delta,
        "minimum_improvement_points": minimum_improvement_points,
        "standard_non_regression": standard_non_regression,
        "supports_promotion": supports_promotion,
        "verdict": verdict,
        "training_updates": 0,
    }
    _atomic_json(report_path, report)
    return PairedAuditSummary(
        candidate_checkpoint=candidate,
        champion_checkpoint=champion,
        openings=openings,
        games_per_checkpoint=openings * 2,
        candidate_wins=int(candidate_stats["wins"]),
        candidate_draws=int(candidate_stats["draws"]),
        candidate_losses=int(candidate_stats["losses"]),
        candidate_points=float(candidate_stats["points"]),
        candidate_standard_points=float(candidate_stats["standard_points"]),
        champion_wins=int(champion_stats["wins"]),
        champion_draws=int(champion_stats["draws"]),
        champion_losses=int(champion_stats["losses"]),
        champion_points=float(champion_stats["points"]),
        champion_standard_points=float(champion_stats["standard_points"]),
        point_delta=point_delta,
        standard_non_regression=standard_non_regression,
        supports_promotion=supports_promotion,
        verdict=verdict,
        audit_seed=audit_seed,
        search_simulations=search_simulations,
        c_puct=c_puct,
        report_path=report_path,
    )
