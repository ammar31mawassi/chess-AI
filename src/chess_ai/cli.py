"""Command-line composition for the Phase 1 chess-AI pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess

from chess_ai.agents.human_agent import HumanAgent
from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.agents.opening_book_agent import OpeningBookAgent
from chess_ai.agents.protocol import ChessAgent
from chess_ai.agents.random_agent import RandomAgent
from chess_ai.config import config_section, load_config

LOGGER = logging.getLogger(__name__)
AgentFactory = Callable[[int], ChessAgent]
AGENT_CHOICES = ("human", "random", "minimax", "neural", "neural-mcts")
ARENA_AGENT_CHOICES = ("random", "minimax", "neural", "neural-mcts")


def _utc_filename(prefix: str, suffix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    return f"{prefix}_{stamp}{suffix}"


def _as_int(value: Any, *, name: str, default: int) -> int:
    selected = default if value is None else value
    if isinstance(selected, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {selected!r}") from exc


def _make_agent(
    kind: str,
    *,
    seed: int,
    depth: int,
    checkpoint: str | Path | None,
    device: str,
    deterministic: bool = True,
    temperature: float = 1.0,
    simulations: int = 64,
    c_puct: float = 1.5,
    opening_book: str | Path | None = None,
) -> ChessAgent:
    def with_book(agent: ChessAgent) -> ChessAgent:
        if opening_book is None:
            return agent
        return OpeningBookAgent(agent, opening_book, seed=seed)

    if kind == "human":
        return with_book(HumanAgent())
    if kind == "random":
        return with_book(RandomAgent(seed=seed))
    if kind == "minimax":
        return with_book(MinimaxAgent(depth=depth, deterministic=deterministic, seed=seed))
    if kind == "neural":
        if checkpoint is None:
            raise ValueError(
                "A neural agent needs a checkpoint. Use --white-checkpoint or "
                "--black-checkpoint for that side."
            )
        return with_book(
            NeuralAgent(
                checkpoint,
                device=device,
                deterministic=deterministic,
                temperature=temperature,
                seed=seed,
            )
        )
    if kind == "neural-mcts":
        if checkpoint is None:
            raise ValueError(
                "A neural-mcts agent needs a checkpoint. Use --white-checkpoint or "
                "--black-checkpoint for that side."
            )
        return with_book(
            NeuralMCTSAgent(
                checkpoint,
                device=device,
                simulations=simulations,
                c_puct=c_puct,
                seed=seed,
            )
        )
    raise ValueError(f"Unknown agent kind: {kind}")


def _agent_factory(
    kind: str,
    *,
    depth: int,
    checkpoint: str | Path | None,
    device: str,
    deterministic: bool,
    temperature: float,
    simulations: int = 64,
    c_puct: float = 1.5,
    opening_book: str | Path | None = None,
) -> AgentFactory:
    def build(seed: int) -> ChessAgent:
        return _make_agent(
            kind,
            seed=seed,
            depth=depth,
            checkpoint=checkpoint,
            device=device,
            deterministic=deterministic,
            temperature=temperature,
            simulations=simulations,
            c_puct=c_puct,
            opening_book=opening_book,
        )

    return build


class _ConsoleAgent:
    """Show the board and selected move around an ordinary agent call."""

    def __init__(self, agent: ChessAgent) -> None:
        self.agent = agent
        self.name = agent.name

    def choose_move(self, board: chess.Board) -> chess.Move:
        color = "White" if board.turn == chess.WHITE else "Black"
        print(f"\n{color} to move ({self.name})")
        print(board)
        move = self.agent.choose_move(board)
        print(f"{self.name} chose {move.uci()}")
        return move


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def command_doctor(args: argparse.Namespace) -> int:
    """Print dependencies, hardware selection, paths, and a model smoke test."""

    import numpy as np
    import torch
    import yaml

    from chess_ai.agents.neural_agent import resolve_device
    from chess_ai.model.policy_value_net import PolicyValueNet

    directory_paths = (
        Path("data/games"),
        Path("data/datasets"),
        Path("data/metrics"),
        Path("checkpoints"),
    )
    device_error: str | None = None
    try:
        device = resolve_device(args.device)
        selected_device = str(device)
    except RuntimeError as exc:
        device = torch.device("cpu")
        selected_device = "unavailable"
        device_error = str(exc)

    forward_ok = False
    forward_detail = "not run"
    try:
        model = PolicyValueNet(channels=8, residual_blocks=1).to(device).eval()
        sample = torch.zeros((1, 18, 8, 8), dtype=torch.float32, device=device)
        with torch.inference_mode():
            policy, value = model(sample)
        forward_ok = (
            tuple(policy.shape) == (1, 4208)
            and tuple(value.shape) == (1, 1)
            and bool(torch.isfinite(policy).all())
            and bool(torch.isfinite(value).all())
        )
        forward_detail = (
            f"policy={tuple(policy.shape)}, value={tuple(value.shape)}, finite={forward_ok}"
        )
    except (RuntimeError, ValueError) as exc:
        forward_detail = str(exc)

    python_supported = sys.version_info >= (3, 11)
    print("Self-Improving Chess AI doctor")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Python >= 3.11: {'yes' if python_supported else 'NO'}")
    print(f"python-chess: {_package_version('python-chess')}")
    print(f"NumPy: {np.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'none (CPU build)'}")
    print(f"Requested device: {args.device}")
    print(f"Selected device: {selected_device}")
    if device_error is not None:
        print(f"Device error: {device_error}")
    print("Required directories:")
    directories_ok = True
    for path in directory_paths:
        exists = path.is_dir()
        directories_ok = directories_ok and exists
        print(f"  {path}: {'ok' if exists else 'MISSING'}")
    print(f"Sample forward pass: {'ok' if forward_ok else 'FAILED'} ({forward_detail})")
    return 0 if python_supported and directories_ok and forward_ok and device_error is None else 1


def command_play(args: argparse.Namespace) -> int:
    from chess_ai.arena import run_match

    white = _ConsoleAgent(
        _make_agent(
            args.white,
            seed=args.seed,
            depth=args.white_depth if args.white_depth is not None else args.depth,
            checkpoint=args.white_checkpoint,
            device=args.device,
            deterministic=args.deterministic,
            temperature=args.temperature,
            simulations=(
                args.white_simulations if args.white_simulations is not None else args.simulations
            ),
            c_puct=args.c_puct,
            opening_book=args.white_opening_book,
        )
    )
    black = _ConsoleAgent(
        _make_agent(
            args.black,
            seed=args.seed + 1,
            depth=args.black_depth if args.black_depth is not None else args.depth,
            checkpoint=args.black_checkpoint,
            device=args.device,
            deterministic=args.deterministic,
            temperature=args.temperature,
            simulations=(
                args.black_simulations if args.black_simulations is not None else args.simulations
            ),
            c_puct=args.c_puct,
            opening_book=args.black_opening_book,
        )
    )
    pgn_path = (
        Path(args.pgn)
        if args.pgn is not None
        else Path("data/games/play") / _utc_filename("play", ".pgn")
    )
    result = run_match(
        white,
        black,
        max_plies=args.max_plies,
        starting_fen=args.fen,
        seed=args.seed,
        pgn_path=pgn_path,
    )
    print("\nFinal board")
    print(chess.Board(result.final_fen))
    print(
        f"Result: {result.result} ({result.termination}); plies={result.plies}; "
        f"duration={result.duration_seconds:.3f}s"
    )
    print(f"Saved PGN: {result.pgn_path}")
    return 0


def command_gui(args: argparse.Namespace) -> int:
    """Launch the optional local neural-play workbench."""

    # Tkinter and the GUI package stay out of non-GUI commands and test collection.
    from chess_ai.gui import launch_gui

    launch_gui(
        checkpoint=args.checkpoint,
        human_color=args.human_color,
        device=args.device,
        opening_book=args.opening_book,
        search_simulations=args.search_simulations,
        c_puct=args.c_puct,
        dataset_path=args.dataset_path,
        pgn_dir=args.pgn_dir,
        training_enabled=args.training_enabled,
    )
    return 0


def command_generate_data(args: argparse.Namespace) -> int:
    from chess_ai.data import GenerationConfig, generate_supervised_data

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=0)
    config = GenerationConfig.from_mapping(
        config_section(raw, "generation"),
        seed=seed,
        resume=args.resume,
    )
    summary = generate_supervised_data(config)
    print(
        json.dumps(
            {
                "dataset_path": str(summary.dataset_path),
                "pgn_dir": str(summary.pgn_dir) if summary.pgn_dir is not None else None,
                "completed_games": summary.completed_games,
                "examples": summary.examples,
                "duplicates_discarded": summary.duplicates_discarded,
                "unrecorded_agent_moves": summary.unrecorded_agent_moves,
                "resumed": summary.resumed,
                "interrupted": summary.interrupted,
            },
            indent=2,
        )
    )
    return 130 if summary.interrupted else 0


def command_teacher_cycle(args: argparse.Namespace) -> int:
    from chess_ai.curriculum import TeacherCycleConfig, run_teacher_cycle

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    teacher_values = config_section(raw, "teacher_cycle")
    if args.cycles is not None:
        teacher_values["cycles"] = args.cycles
    if args.normal_start_every is not None:
        teacher_values["normal_start_every_cycles"] = args.normal_start_every
    if args.anchor_correction_share is not None:
        teacher_values["anchor_correction_share"] = args.anchor_correction_share
    if args.evaluate_after_normal_start is not None:
        teacher_values["evaluate_after_normal_start"] = args.evaluate_after_normal_start
    config = TeacherCycleConfig.from_mapping(
        teacher_values,
        seed=seed,
        device=device,
        resume=args.resume,
    )
    summary = run_teacher_cycle(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    if summary.first_win_cycle is None:
        print("No student win was observed; the script does not claim success or promotion.")
    elif summary.ready_for_next_stage:
        print("The configured D1 mastery gate passed. Review the games before starting D2.")
    else:
        print(
            f"First student win occurred in cycle {summary.first_win_cycle}; "
            "the stricter D1 mastery gate has not passed."
        )
    return 0


def command_teacher_batch(args: argparse.Namespace) -> int:
    """Collect a frozen offline D2 trace dataset without training."""

    from chess_ai.curriculum import TeacherBatchConfig, run_teacher_batch

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    config = TeacherBatchConfig.from_mapping(
        config_section(raw, "teacher_batch"),
        seed=seed,
        device=device,
        resume=args.resume,
    )
    summary = run_teacher_batch(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Collection complete. No optimizer was created and no training update was run.")
    return 0


def command_mixed_batch(args: argparse.Namespace) -> int:
    """Collect a configured multi-matchup minimax-teacher dataset."""

    from chess_ai.curriculum import MixedBatchConfig, run_mixed_batch

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    config = MixedBatchConfig.from_mapping(
        config_section(raw, "mixed_batch"),
        seed=seed,
        device=device,
        resume=args.resume,
    )
    summary = run_mixed_batch(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Collection complete. All configured cohorts were traced; no training update was run.")
    return 0


def command_finalize_mixed_batch(args: argparse.Namespace) -> int:
    """Permanently assemble the currently committed mixed-batch shards."""

    from chess_ai.curriculum import MixedBatchConfig, finalize_mixed_batch

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    config = MixedBatchConfig.from_mapping(
        config_section(raw, "mixed_batch"),
        seed=seed,
        device=device,
    )
    summary = finalize_mixed_batch(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Saved committed openings as a final immutable dataset; collection is now closed.")
    return 0


def command_gated_refinement(args: argparse.Namespace) -> int:
    from chess_ai.curriculum import (
        GatedRefinementConfig,
        initialize_gated_refinement,
        run_gated_refinement,
    )

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    gated_values = config_section(raw, "gated_refinement")
    if args.rounds is not None:
        gated_values["rounds"] = args.rounds
    config = GatedRefinementConfig.from_mapping(
        gated_values,
        seed=seed,
        device=device,
        resume=args.resume,
    )
    summary = (
        initialize_gated_refinement(config)
        if args.initialize_only
        else run_gated_refinement(config)
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    if args.initialize_only:
        print(f"Gameplay champion initialized without training: {summary.champion_checkpoint}")
        return 0
    if summary.promotions:
        print(
            f"Gameplay gate promoted {summary.promotions} candidate(s); current champion: "
            f"{summary.champion_checkpoint}"
        )
    else:
        print("No candidate passed the gameplay gate; epoch 399 remains champion.")
    return 0


def command_gated_audit(args: argparse.Namespace) -> int:
    from chess_ai.curriculum import GatedRefinementConfig, run_gated_holdout_audit

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    config = GatedRefinementConfig.from_mapping(
        config_section(raw, "gated_refinement"),
        seed=seed,
        device=device,
    )
    summary = run_gated_holdout_audit(
        config,
        openings=args.openings,
        audit_seed=args.audit_seed,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Holdout audit is evaluation-only; no checkpoint was promoted or modified.")
    return 0


def command_paired_audit(args: argparse.Namespace) -> int:
    """Compare a candidate and champion on identical unseen D1 starts."""

    from chess_ai.arena import run_paired_audit

    summary = run_paired_audit(
        candidate_checkpoint=args.candidate,
        champion_checkpoint=args.champion,
        openings=args.openings,
        audit_seed=args.audit_seed,
        opponent_depth=args.opponent_depth,
        opening_min_full_moves=args.opening_min_full_moves,
        opening_max_full_moves=args.opening_max_full_moves,
        max_plies=args.max_plies,
        minimum_improvement_points=args.minimum_improvement_points,
        device=args.device,
        pgn_dir=args.pgn_dir,
        exclude_manifest=args.exclude_manifest,
        search_simulations=args.search_simulations,
        c_puct=args.c_puct,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Paired audit is evaluation-only; no training or checkpoint promotion was performed.")
    return 0


def command_gameplay_select(args: argparse.Namespace) -> int:
    """Select a provisional epoch using D1/D2 gameplay before final audits."""

    from chess_ai.arena import GameplaySelectionConfig, run_gameplay_selection

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=9_700_205)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    config = GameplaySelectionConfig.from_mapping(
        config_section(raw, "gameplay_selection"),
        seed=seed,
        device=device,
    )
    summary = run_gameplay_selection(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print(
        "Gameplay selection is provisional and evaluation-only; run fresh paired audits before "
        "promoting the selected checkpoint."
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    from chess_ai.model import PolicyValueNet, load_model
    from chess_ai.training import TrainingConfig, train_model

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=0)
    configured_device = str(raw.get("device", "auto"))
    device = args.device if args.device is not None else configured_device
    config = TrainingConfig.from_mapping(
        config_section(raw, "training"),
        seed=seed,
        device=device,
        resume_from=args.resume,
    )
    if args.init_checkpoint is not None:
        source_directory = args.init_checkpoint.resolve().parent
        output_directory = config.checkpoint_dir.resolve()
        if output_directory == source_directory:
            raise ValueError(
                "--init-checkpoint must write to a different checkpoint directory so the "
                "source model cannot be overwritten; change training.checkpoint_dir."
            )
        existing_outputs = [
            path
            for path in (
                output_directory / "best.pt",
                output_directory / "last.pt",
                *output_directory.glob("*.pt"),
                config.metrics_path.resolve(),
            )
            if path.exists()
        ]
        if existing_outputs:
            rendered = ", ".join(str(path) for path in existing_outputs[:3])
            raise ValueError(
                "--init-checkpoint starts a fresh run and will not overwrite an earlier "
                f"candidate ({rendered}). Choose new training.checkpoint_dir and "
                "training.metrics_path values, or intentionally archive the old run first."
            )
        model = load_model(args.init_checkpoint, device="cpu", eval_mode=False)
    else:
        model_values = config_section(raw, "model")
        channels = _as_int(model_values.get("channels"), name="model.channels", default=32)
        residual_blocks = _as_int(
            model_values.get("residual_blocks"), name="model.residual_blocks", default=2
        )
        model = PolicyValueNet(channels=channels, residual_blocks=residual_blocks)
    history = train_model(model, config)
    if not history:
        print(
            "No new epochs were needed: the resume checkpoint already reached "
            f"the configured target of {config.epochs} epoch(s)."
        )
        return 0
    print(json.dumps(history[-1].to_dict(), indent=2, sort_keys=True))
    print(f"Epoch checkpoints: {config.checkpoint_dir}")
    print(f"Best checkpoint: {config.checkpoint_dir / 'best.pt'}")
    print(f"Latest checkpoint: {config.checkpoint_dir / 'last.pt'}")
    print(f"Metrics: {config.metrics_path}")
    return 0


def _tournament_payload(result: Any) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


def command_arena(args: argparse.Namespace) -> int:
    from chess_ai.arena import run_tournament

    white_source = _agent_factory(
        args.white,
        depth=args.white_depth if args.white_depth is not None else args.depth,
        checkpoint=args.white_checkpoint,
        device=args.device,
        deterministic=args.deterministic,
        temperature=args.temperature,
        simulations=(
            args.white_simulations if args.white_simulations is not None else args.simulations
        ),
        c_puct=args.c_puct,
        opening_book=args.white_opening_book,
    )
    black_source = _agent_factory(
        args.black,
        depth=args.black_depth if args.black_depth is not None else args.depth,
        checkpoint=args.black_checkpoint,
        device=args.device,
        deterministic=args.deterministic,
        temperature=args.temperature,
        simulations=(
            args.black_simulations if args.black_simulations is not None else args.simulations
        ),
        c_puct=args.c_puct,
        opening_book=args.black_opening_book,
    )
    result = run_tournament(
        white_source,
        black_source,
        games=args.games,
        seed=args.seed,
        switch_colors=args.switch_colors,
        max_plies=args.max_plies,
        starting_fen=args.fen,
        pgn_dir=args.pgn_dir,
    )
    print(_tournament_payload(result))
    print("Approximate Elo is descriptive only; it is not an official rating.")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    from chess_ai.arena import evaluate_candidate

    candidate = _agent_factory(
        "neural",
        depth=1,
        checkpoint=args.candidate,
        device=args.device,
        deterministic=True,
        temperature=0.0,
    )
    champion = _agent_factory(
        "neural",
        depth=1,
        checkpoint=args.champion,
        device=args.device,
        deterministic=True,
        temperature=0.0,
    )
    report = evaluate_candidate(
        candidate,
        champion,
        games=args.games,
        seed=args.seed,
        switch_colors=True,
        max_plies=args.max_plies,
        pgn_dir=args.pgn_dir,
        stronger_threshold=args.threshold,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print("No checkpoint was promoted or modified.")
    return 0


def _result_confirmer(prompt: str) -> str | bool | None:
    response = input(prompt).strip()
    if response.lower() in {"", "y", "yes"}:
        return True
    if response.lower() in {"n", "no"}:
        return False
    return response


def command_external(args: argparse.Namespace) -> int:
    from chess_ai.arena import run_external_session

    opponent_label = args.opponent or input("Opponent label: ").strip()
    difficulty = args.difficulty or input("Difficulty level: ").strip()
    agent = NeuralAgent(
        args.checkpoint,
        device=args.device,
        deterministic=args.deterministic,
        temperature=args.temperature,
        seed=args.seed,
    )
    result = run_external_session(
        agent,
        ai_color=args.ai_color,
        opponent_label=opponent_label,
        difficulty_level=difficulty,
        checkpoint=str(args.checkpoint),
        pgn_dir=args.pgn_dir,
        benchmark_path=args.benchmark_path,
        starting_fen=args.fen,
        max_plies=args.max_plies,
        result_confirmer=_result_confirmer,
    )
    print(
        f"External session result: {result.result} ({result.termination}), "
        f"plies={result.move_count}"
    )
    return 0


def command_report(args: argparse.Namespace) -> int:
    from chess_ai.arena import report_external_benchmarks

    print(report_external_benchmarks(args.benchmark_path))
    return 0


def command_import_external(args: argparse.Namespace) -> int:
    from chess_ai.storage import import_external_games

    report = import_external_games(
        args.pgn,
        destination_dir=args.destination_dir,
        confirm_evaluation_data_import=args.confirm_evaluation_data_import,
    )
    payload = asdict(report)
    payload["imported_paths"] = [str(path) for path in report.imported_paths]
    payload["manifest_path"] = str(report.manifest_path)
    print(json.dumps(payload, indent=2))
    print("Imported files retain evaluation_origin=true and are not a training dataset.")
    return 0


def command_import_kaggle(args: argparse.Namespace) -> int:
    """Explicitly convert one reviewed local Kaggle CSV into training data."""

    if not args.confirm_external_training_data:
        raise ValueError("External training import requires --confirm-external-training-data")
    from chess_ai.data import KaggleImportConfig, import_kaggle_dataset

    raw = load_config(args.config)
    seed = _as_int(raw.get("seed"), name="seed", default=2026)
    config = KaggleImportConfig.from_mapping(
        config_section(raw, "kaggle_import"),
        seed=seed,
    )
    summary = import_kaggle_dataset(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("External data was explicitly validated, sampled, and kept in a separate dataset.")
    return 0


def command_import_openings(args: argparse.Namespace) -> int:
    """Compile reviewed opening lines into training and runtime artifacts."""

    if not args.confirm_external_training_data:
        raise ValueError("External opening import requires --confirm-external-training-data")
    from chess_ai.data import OpeningImportConfig, import_opening_book

    raw = load_config(args.config)
    config = OpeningImportConfig.from_mapping(config_section(raw, "opening_import"))
    summary = import_opening_book(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Opening lines were validated with python-chess and kept as explicit external data.")
    return 0


def command_compose_datasets(args: argparse.Namespace) -> int:
    """Explicitly combine reviewed datasets with source-aware weights."""

    from chess_ai.data import CompositionConfig, compose_datasets

    raw = load_config(args.config)
    config = CompositionConfig.from_mapping(config_section(raw, "composition"))
    summary = compose_datasets(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Datasets were copied into one weighted artifact; source files were not modified.")
    return 0


def _add_shared_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depth", type=int, default=2, help="default minimax search depth")
    parser.add_argument("--white-depth", type=int, help="override White's minimax depth")
    parser.add_argument("--black-depth", type=int, help="override Black's minimax depth")
    parser.add_argument("--white-checkpoint", type=Path, help="checkpoint for a neural White")
    parser.add_argument("--black-checkpoint", type=Path, help="checkpoint for a neural Black")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use deterministic agent selection (default: true)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="neural sampling temperature when --no-deterministic is used",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=64,
        help="PUCT traversals per move for neural-mcts agents (default: 64)",
    )
    parser.add_argument(
        "--white-simulations",
        type=int,
        help="override White's neural-mcts traversal count",
    )
    parser.add_argument(
        "--black-simulations",
        type=int,
        help="override Black's neural-mcts traversal count",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant for neural-mcts agents (default: 1.5)",
    )
    parser.add_argument(
        "--white-opening-book",
        type=Path,
        help="compiled book used before White's configured agent fallback",
    )
    parser.add_argument(
        "--black-opening-book",
        type=Path,
        help="compiled book used before Black's configured agent fallback",
    )
    parser.add_argument("--fen", help="optional starting FEN")
    parser.add_argument(
        "--max-plies",
        "--max-moves",
        dest="max_plies",
        type=int,
        default=200,
        help="maximum half-moves before a safeguard draw",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chess_ai",
        description="Educational Phase 1 tools for a self-improving chess AI.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect dependencies and run a model smoke test")
    doctor.add_argument("--device", default="auto")
    doctor.set_defaults(handler=command_doctor)

    gui = subparsers.add_parser(
        "gui", help="play a neural checkpoint in the local graphical workbench"
    )
    gui.add_argument(
        "--checkpoint",
        type=Path,
        help="neural checkpoint (the GUI discovers a local best/last checkpoint if omitted)",
    )
    gui.add_argument("--human-color", choices=("white", "black"), default="white")
    gui.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    gui.add_argument(
        "--opening-book",
        type=Path,
        help="compiled opening book used until the human leaves known theory",
    )
    gui.add_argument(
        "--search-simulations",
        type=int,
        default=0,
        help="PUCT traversals after leaving the book; zero uses greedy neural play",
    )
    gui.add_argument("--c-puct", type=float, default=1.5)
    gui.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/datasets/human_gui.pt"),
        help="separate dataset that can receive confirmed completed games",
    )
    gui.add_argument(
        "--pgn-dir",
        type=Path,
        default=Path("data/games/human_gui"),
        help="directory for independently saved GUI game PGNs",
    )
    gui.add_argument(
        "--training-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="preselect opt-in collection of human move labels (default: false)",
    )
    gui.set_defaults(handler=command_gui)

    play = subparsers.add_parser("play", help="play one terminal-visible game")
    play.add_argument("--white", choices=AGENT_CHOICES, default="human")
    play.add_argument("--black", choices=AGENT_CHOICES, default="random")
    play.add_argument("--pgn", type=Path, help="PGN output path (an automatic path is the default)")
    _add_shared_agent_options(play)
    play.set_defaults(handler=command_play)

    generate = subparsers.add_parser(
        "generate-data", help="generate a versioned supervised dataset"
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resume a compatible existing dataset (generator default: true)",
    )
    generate.set_defaults(handler=command_generate_data)

    teacher_cycle = subparsers.add_parser(
        "teacher-cycle",
        help="run a resumable neural-vs-minimax teacher-correction curriculum",
    )
    teacher_cycle.add_argument("--config", type=Path, required=True)
    teacher_cycle.add_argument("--device", help="override the YAML device")
    teacher_cycle.add_argument(
        "--cycles",
        type=int,
        help="override the total cycle target; a completed run may only be extended upward",
    )
    teacher_cycle.add_argument(
        "--normal-start-every",
        type=int,
        help="use the normal starting position every N cycles; zero disables it",
    )
    teacher_cycle.add_argument(
        "--anchor-correction-share",
        type=float,
        help="reserve this share of the correction batch for normal-start rehearsal",
    )
    teacher_cycle.add_argument(
        "--evaluate-after-normal-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="save a two-color standard-position regression pair after each anchor update",
    )
    teacher_cycle.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resume matching saved state (configuration default: true)",
    )
    teacher_cycle.set_defaults(handler=command_teacher_cycle)

    teacher_batch = subparsers.add_parser(
        "teacher-batch",
        help="collect paired champion/D1 games with offline D2 labels and no training",
    )
    teacher_batch.add_argument("--config", type=Path, required=True)
    teacher_batch.add_argument("--device", help="override the YAML neural-inference device")
    teacher_batch.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resume matching saved collection state (configuration default: true)",
    )
    teacher_batch.set_defaults(handler=command_teacher_batch)

    mixed_batch = subparsers.add_parser(
        "mixed-batch",
        help="collect configured color-switched cohorts with stronger minimax labels",
    )
    mixed_batch.add_argument("--config", type=Path, required=True)
    mixed_batch.add_argument("--device", help="override the YAML neural-inference device")
    mixed_batch.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resume matching saved state (configuration default: true)",
    )
    mixed_batch.set_defaults(handler=command_mixed_batch)

    finalize_mixed = subparsers.add_parser(
        "finalize-mixed-batch",
        help="permanently assemble committed mixed-batch shards without collecting more games",
    )
    finalize_mixed.add_argument("--config", type=Path, required=True)
    finalize_mixed.add_argument("--device", help="override the YAML neural-inference device")
    finalize_mixed.set_defaults(handler=command_finalize_mixed_batch)

    gated_refinement = subparsers.add_parser(
        "gated-refinement",
        help="train blockwise candidates and promote only through a fixed D1 gameplay gate",
    )
    gated_refinement.add_argument("--config", type=Path, required=True)
    gated_refinement.add_argument("--device", help="override the YAML device")
    gated_refinement.add_argument(
        "--rounds",
        type=int,
        help="override the total round target; a completed run may only be extended upward",
    )
    gated_refinement.add_argument(
        "--initialize-only",
        action="store_true",
        help="mark generation zero and validate paths without playing or training",
    )
    gated_refinement.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resume matching saved state (configuration default: true)",
    )
    gated_refinement.set_defaults(handler=command_gated_refinement)

    gated_audit = subparsers.add_parser(
        "gated-audit",
        help="compare generation zero and the gameplay champion on unseen D1 openings",
    )
    gated_audit.add_argument("--config", type=Path, required=True)
    gated_audit.add_argument("--device", help="override the YAML device")
    gated_audit.add_argument(
        "--openings",
        type=int,
        default=20,
        help="number of holdout starts including the standard position (default: 20)",
    )
    gated_audit.add_argument(
        "--audit-seed",
        type=int,
        help="override the separate holdout seed",
    )
    gated_audit.set_defaults(handler=command_gated_audit)

    paired_audit = subparsers.add_parser(
        "paired-audit",
        help="compare candidate and champion against D1 on identical unseen openings",
    )
    paired_audit.add_argument("--candidate", type=Path, required=True)
    paired_audit.add_argument("--champion", type=Path, required=True)
    paired_audit.add_argument("--openings", type=int, default=20)
    paired_audit.add_argument("--audit-seed", type=int, default=9_300_205)
    paired_audit.add_argument("--opponent-depth", type=int, default=1)
    paired_audit.add_argument("--opening-min-full-moves", type=int, default=2)
    paired_audit.add_argument("--opening-max-full-moves", type=int, default=3)
    paired_audit.add_argument("--max-plies", type=int, default=200)
    paired_audit.add_argument("--minimum-improvement-points", type=float, default=0.5)
    paired_audit.add_argument("--device", default="auto")
    paired_audit.add_argument(
        "--search-simulations",
        type=int,
        default=0,
        help="PUCT traversals per neural move; zero keeps greedy policy play",
    )
    paired_audit.add_argument("--c-puct", type=float, default=1.5)
    paired_audit.add_argument(
        "--pgn-dir",
        type=Path,
        default=Path("data/games/evaluation/paired_audit"),
    )
    paired_audit.add_argument(
        "--exclude-manifest",
        type=Path,
        help="teacher-batch manifest whose opening FENs must not appear in this audit",
    )
    paired_audit.set_defaults(handler=command_paired_audit)

    gameplay_select = subparsers.add_parser(
        "gameplay-select",
        help="rank retained epoch checkpoints through fixed D1/D2 gameplay suites",
    )
    gameplay_select.add_argument("--config", type=Path, required=True)
    gameplay_select.add_argument("--device", help="override the YAML neural-inference device")
    gameplay_select.set_defaults(handler=command_gameplay_select)

    train = subparsers.add_parser("train", help="train a policy-value model")
    train.add_argument("--config", type=Path, required=True)
    train_checkpoint = train.add_mutually_exclusive_group()
    train_checkpoint.add_argument(
        "--resume", type=Path, help="resume model, optimizer, scheduler, and epoch state"
    )
    train_checkpoint.add_argument(
        "--init-checkpoint",
        type=Path,
        help="start a fresh epoch-1 run from checkpoint architecture and weights",
    )
    train.add_argument("--device", help="override the YAML device")
    train.set_defaults(handler=command_train)

    arena = subparsers.add_parser("arena", help="run a batch tournament")
    arena.add_argument("--white", choices=ARENA_AGENT_CHOICES, default="minimax")
    arena.add_argument("--black", choices=ARENA_AGENT_CHOICES, default="random")
    arena.add_argument("--games", type=int, default=2)
    arena.add_argument(
        "--switch-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="alternate which named agent receives White (default: true)",
    )
    arena.add_argument("--pgn-dir", type=Path, default=Path("data/games/arena"))
    _add_shared_agent_options(arena)
    arena.set_defaults(handler=command_arena)

    evaluate = subparsers.add_parser(
        "evaluate", help="compare candidate/champion checkpoints without promotion"
    )
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--champion", type=Path, required=True)
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--max-plies", type=int, default=200)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--pgn-dir", type=Path, default=Path("data/games/evaluation"))
    evaluate.set_defaults(handler=command_evaluate)

    external = subparsers.add_parser(
        "external", help="manually mediate a game with an offline external opponent"
    )
    external.add_argument("--checkpoint", type=Path, required=True)
    external.add_argument("--ai-color", choices=("white", "black"), required=True)
    external.add_argument("--opponent", help="opponent application label (prompted if omitted)")
    external.add_argument("--difficulty", help="user-entered difficulty (prompted if omitted)")
    external.add_argument("--device", default="auto")
    external.add_argument("--seed", type=int, default=0)
    external.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    external.add_argument("--temperature", type=float, default=1.0)
    external.add_argument("--fen", help="optional starting FEN")
    external.add_argument("--max-plies", type=int, default=512)
    external.add_argument("--pgn-dir", type=Path, default=Path("data/games/external"))
    external.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/metrics/external_benchmarks.jsonl"),
    )
    external.set_defaults(handler=command_external)

    report = subparsers.add_parser("report", help="summarize cumulative external benchmarks")
    report.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/metrics/external_benchmarks.jsonl"),
    )
    report.set_defaults(handler=command_report)

    import_external = subparsers.add_parser(
        "import-external", help="explicitly validate and segregate external benchmark PGNs"
    )
    import_external.add_argument("--pgn", type=Path, nargs="+", required=True)
    import_external.add_argument(
        "--destination-dir",
        type=Path,
        default=Path("data/games/imported_external"),
    )
    import_external.add_argument(
        "--confirm-evaluation-data-import",
        action="store_true",
        help="acknowledge that reviewed evaluation PGNs are being explicitly imported",
    )
    import_external.set_defaults(handler=command_import_external)

    import_kaggle = subparsers.add_parser(
        "import-kaggle",
        help="explicitly validate and sample a reviewed local Kaggle chess CSV",
    )
    import_kaggle.add_argument("--config", type=Path, required=True)
    import_kaggle.add_argument(
        "--confirm-external-training-data",
        action="store_true",
        help="acknowledge that this external source is intentionally entering training data",
    )
    import_kaggle.set_defaults(handler=command_import_kaggle)

    import_openings = subparsers.add_parser(
        "import-openings",
        help="validate opening lines into a training dataset and runtime book",
    )
    import_openings.add_argument("--config", type=Path, required=True)
    import_openings.add_argument(
        "--confirm-external-training-data",
        action="store_true",
        help="acknowledge that this external source enters training and runtime play",
    )
    import_openings.set_defaults(handler=command_import_openings)

    compose = subparsers.add_parser(
        "compose-datasets",
        help="explicitly combine supervised datasets with source-aware training weights",
    )
    compose.add_argument("--config", type=Path, required=True)
    compose.set_defaults(handler=command_compose_datasets)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nCancelled by user. Completed atomic saves remain valid.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.debug("Command failed", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
