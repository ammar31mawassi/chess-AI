"""Development-only puzzle, speed, and paired search-ablation reports."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_ai.agents.neural_agent import NeuralAgentError
from chess_ai.agents.search_experiments import ARMS, ExperimentalPUCT
from chess_ai.arena.match import run_match
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature

FORMAT = "self-improving-chess-ai.search-experiments"


def implementation_signature() -> dict[str, str]:
    """Bind resumed work to the actual search, rules encoding, and label code."""
    package = Path(__file__).resolve().parents[1]
    names = (
        "arena/search_experiments.py",
        "agents/search_experiments.py",
        "agents/neural_mcts_agent.py",
        "agents/neural_agent.py",
        "environment/board_encoder.py",
        "environment/move_encoder.py",
        "model/policy_value_net.py",
        "data/failure_training.py",
        "data/stockfish_teacher.py",
        "data/refinement_replay.py",
    )
    return {name: sha256_file(package / name) for name in names}


def bind_run(root: Path, binding: dict[str, Any]) -> None:
    """Resume only an identical experiment; reject orphaned output directories."""
    path = root / "binding.json"
    record = {
        "format": FORMAT,
        "version": 1,
        "implementation_sha256": implementation_signature(),
        **binding,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Experiment binding changed; choose a new output directory")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Output directory contains artifacts without a binding")
    else:
        atomic_json(path, record)


def read_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != FORMAT + ".suite" or raw.get("version") != 1:
        raise ValueError("Expected a version-1 development suite")
    cases = raw["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must have unique IDs")
    for case in cases:
        board = chess.Board(case["fen"])
        if not board.is_valid():
            raise ValueError(f"Invalid puzzle: {case['id']}")
        for uci in case.get("setup_moves", []):
            board.push_uci(uci)
        if board.is_game_over(claim_draw=True):
            raise ValueError("Benchmark roots must be non-terminal")
        for uci in case.get("expected_moves", []):
            if chess.Move.from_uci(uci) not in board.legal_moves:
                raise ValueError("Illegal expected move")
        validation_board = board.copy(stack=True)
        for uci in case.get("solution", []):
            validation_board.push_uci(uci)
        if case.get("criterion") == "mate_in_one" and not any(
            _mates(board, move) for move in board.legal_moves
        ):
            raise ValueError("Mate-in-one case has no mating move")
    return cases


def _mates(board: chess.Board, move: chess.Move) -> bool:
    child = board.copy(stack=True)
    child.push(move)
    return bool(child.is_checkmate())


def solve_case(agent: ExperimentalPUCT, case: dict[str, Any]) -> dict[str, Any]:
    board = chess.Board(case["fen"])
    for uci in case.get("setup_moves", []):
        board.push_uci(uci)
    agent.clear_state()
    steps = []
    expected = case.get("solution", [])
    solved: bool | None = True if expected else None
    for ply in range(0, max(1, len(expected)), 2):
        move = agent.choose_move(board)
        steps.append({"move": move.uci(), "metrics": agent.report()})
        if expected:
            if move.uci() != expected[ply] and not _mates(board, move):
                solved = False
                break
        elif case.get("criterion") == "mate_in_one":
            solved = _mates(board, move)
        elif "expected_moves" in case:
            solved = move.uci() in case["expected_moves"]
        board.push(move)
        if board.is_game_over(claim_draw=True) or ply + 1 >= len(expected):
            break
        board.push_uci(expected[ply + 1])
    return {"id": case["id"], "category": case["category"], "solved": solved, "steps": steps}


def benchmark(
    checkpoint: Path,
    suite: Path,
    output: Path,
    *,
    arms: list[str],
    device: str,
    simulations: int,
    seed: int,
    seconds: float | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    if repeats < 1 or not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("Choose distinct valid arms and positive repeats")
    cases = read_cases(suite)
    with run_lock(output):
        bind_run(
            output,
            {
                "stage": "benchmark",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "suite_sha256": sha256_file(suite),
                "arms": arms,
                "device": device,
                "simulations": simulations,
                "seed": seed,
                "seconds": seconds,
                "repeats": repeats,
            },
        )
        reports = []
        for arm in arms:
            agent = ExperimentalPUCT(
                checkpoint,
                arm=arm,
                device=device,
                simulations=simulations,
                seed=seed,
                seconds=seconds,
            )
            # Untimed warmup. Clear persistent state before each measured case.
            agent.evaluate(chess.Board())
            for repeat in range(repeats):
                for index, case in enumerate(cases):
                    path = output / f"{arm}_{repeat:03}_{index:04}.json"
                    if path.exists():
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        if payload["checksum"] != signature(payload["result"]):
                            raise ValueError("Benchmark case checksum mismatch")
                        report = payload["result"]
                    else:
                        report = {"arm": arm, "repeat": repeat, **solve_case(agent, case)}
                        atomic_json(path, {"result": report, "checksum": signature(report)})
                    reports.append(report)
            del agent
        aggregates = {}
        for arm in arms:
            measured = [r for r in reports if r["arm"] == arm]
            metrics = [step["metrics"] for r in measured for step in r["steps"]]
            times = [m["elapsed_seconds"] for m in metrics]
            total_seconds = sum(times)
            positions = sum(m["network_positions"] for m in metrics)
            batches = sum(m["network_batches"] for m in metrics)
            first_repeat = [r for r in measured if r["repeat"] == 0 and r["solved"] is not None]
            aggregates[arm] = {
                "unique_cases": len(cases),
                "measured_moves": len(metrics),
                "median_move_seconds": float(np.median(times)),
                "p95_move_seconds": float(np.quantile(times, 0.95)),
                "network_positions": positions,
                "network_batches": batches,
                "mean_inference_batch_size": positions / batches if batches else 0,
                "cache_hits": sum(m["cache_hits"] for m in metrics),
                "network_positions_per_second": positions / total_seconds if total_seconds else 0,
                "first_repeat_solved": sum(r["solved"] is True for r in first_repeat),
                "first_repeat_scored_cases": len(first_repeat),
                "component_seconds": {
                    key: sum(m[key] for m in metrics)
                    for key in (
                        "encode_seconds",
                        "network_seconds",
                        "terminal_seconds",
                        "legal_seconds",
                        "select_seconds",
                    )
                },
                "by_category": {
                    category: {
                        "scored": sum(r["category"] == category for r in first_repeat),
                        "solved": sum(
                            r["category"] == category and r["solved"] is True for r in first_repeat
                        ),
                    }
                    for category in sorted({r["category"] for r in measured})
                },
            }
        summary: dict[str, Any] = {
            "format": FORMAT + ".benchmark",
            "version": 1,
            "results": reports,
            "summary_by_arm": aggregates,
            "limitations": "Development only. Timing is hardware/load dependent; a soft deadline may overshoot. Repeats are timing samples, not independent puzzle evidence. Untagged cases measure speed only. No training or promotion.",
        }
        atomic_json(output / "report.json", summary)
        return summary


def opening(seed: int, index: int) -> chess.Board:
    """Fresh deterministic opening, independent of all training/evaluation RNGs."""
    rng = random.Random(f"search-pilot:{seed}:{index}")
    board = chess.Board()
    for _ in range(rng.randrange(6, 13)):
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(moves))
    return board


def paired_matches(
    checkpoint: Path,
    output: Path,
    *,
    arm: str,
    device: str,
    simulations: int,
    seed: int,
    openings: int,
    max_plies: int,
    seconds: float | None = None,
) -> dict[str, Any]:
    if arm not in ARMS[1:] or openings < 1 or max_plies < 1:
        raise ValueError("Select a non-baseline arm and positive opening/ply limits")
    with run_lock(output):
        bind_run(
            output,
            {
                "stage": "matches",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "arm": arm,
                "device": device,
                "simulations": simulations,
                "seed": seed,
                "openings": openings,
                "max_plies": max_plies,
                "seconds": seconds,
            },
        )
        results = []
        for index in range(openings):
            board = opening(seed, index)
            if board.is_game_over(claim_draw=True):
                raise ValueError("Generated terminal opening; choose another seed/run")
            for candidate_white in (True, False):
                path = output / f"game_{index:05}_{int(candidate_white)}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved["checksum"] != signature(saved["result"]):
                        raise ValueError("Match checksum mismatch")
                    result = saved["result"]
                else:
                    common: dict[str, Any] = {
                        "device": device,
                        "simulations": simulations,
                        "seed": seed,
                        "seconds": seconds,
                    }
                    candidate = ExperimentalPUCT(checkpoint, arm=arm, **common)
                    baseline = ExperimentalPUCT(checkpoint, arm="baseline", **common)
                    # Warm up both models outside game timing.
                    candidate.evaluate(board)
                    baseline.evaluate(board)
                    candidate.clear_state()
                    baseline.clear_state()
                    match = run_match(
                        candidate if candidate_white else baseline,
                        baseline if candidate_white else candidate,
                        starting_fen=board.fen(en_passant="fen"),
                        max_plies=max_plies,
                        extra_headers={"Event": "Search pilot DEVELOPMENT EVALUATION ONLY"},
                    )
                    score = {
                        "1-0": float(candidate_white),
                        "0-1": float(not candidate_white),
                        "1/2-1/2": 0.5,
                    }[match.result]
                    if match.termination == "illegal_agent_move":
                        raise NeuralAgentError("Illegal move in experimental match")
                    result = {
                        "opening": index,
                        "candidate_white": candidate_white,
                        "points": score,
                        "termination": match.termination,
                        "result": match.result,
                        "pgn": str(match.pgn),
                        "candidate_seconds": sum(
                            m.elapsed_seconds for m in match.moves if m.agent_name == candidate.name
                        ),
                        "baseline_seconds": sum(
                            m.elapsed_seconds for m in match.moves if m.agent_name == baseline.name
                        ),
                    }
                    atomic_json(path, {"result": result, "checksum": signature(result)})
                    del candidate, baseline
                results.append(result)
        # Bootstrap opening pairs, never treat swapped colors as independent.
        pair_scores = [
            sum(r["points"] for r in results if r["opening"] == i) / 2 for i in range(openings)
        ]
        rng = random.Random(seed)
        samples = [sum(rng.choices(pair_scores, k=openings)) / openings for _ in range(2000)]
        report = {
            "format": FORMAT + ".matches",
            "version": 1,
            "games": len(results),
            "candidate_points": sum(r["points"] for r in results),
            "results": results,
            "opening_cluster_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist()
            if openings >= 2
            else None,
            "unresolved_move_caps": sum(r["termination"] == "move_limit" for r in results),
            "limitations": "Development head-to-head; no automatic promotion or rating claim. Move caps score 0.5 for accounting but are unresolved. Baseline and variant share one checkpoint. Timed results depend on hardware; budgets are soft.",
        }
        atomic_json(output / "report.json", report)
        return report
