from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import chess_ai.model as model_module
import chess_ai.training as training_module
from chess_ai.cli import build_parser, command_gui, command_train


def test_gui_parser_uses_safe_collection_defaults() -> None:
    args = build_parser().parse_args(["gui"])

    assert args.checkpoint is None
    assert args.human_color == "white"
    assert args.device == "auto"
    assert args.opening_book is None
    assert args.search_simulations == 0
    assert args.c_puct == 1.5
    assert args.dataset_path == Path("data/datasets/human_gui.pt")
    assert args.pgn_dir == Path("data/games/human_gui")
    assert args.training_enabled is False


def test_gui_parser_accepts_neural_play_options() -> None:
    args = build_parser().parse_args(
        [
            "gui",
            "--checkpoint",
            "checkpoints/base/best.pt",
            "--human-color",
            "black",
            "--device",
            "cuda",
            "--opening-book",
            "data/opening_books/all_chess_openings_v1.json",
            "--search-simulations",
            "32",
            "--c-puct",
            "1.25",
            "--dataset-path",
            "data/datasets/mine.pt",
            "--pgn-dir",
            "data/games/mine",
            "--training-enabled",
        ]
    )

    assert args.checkpoint == Path("checkpoints/base/best.pt")
    assert args.human_color == "black"
    assert args.device == "cuda"
    assert args.opening_book == Path("data/opening_books/all_chess_openings_v1.json")
    assert args.search_simulations == 32
    assert args.c_puct == 1.25
    assert args.dataset_path == Path("data/datasets/mine.pt")
    assert args.pgn_dir == Path("data/games/mine")
    assert args.training_enabled is True


def test_gui_command_lazily_forwards_options(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    gui_module = ModuleType("chess_ai.gui")

    def fake_launch_gui(**options: Any) -> None:
        observed.update(options)

    gui_module.launch_gui = fake_launch_gui  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chess_ai.gui", gui_module)
    args = build_parser().parse_args(
        [
            "gui",
            "--checkpoint",
            "checkpoints/base/best.pt",
            "--human-color",
            "black",
            "--device",
            "cpu",
            "--training-enabled",
        ]
    )

    assert command_gui(args) == 0
    assert observed == {
        "checkpoint": Path("checkpoints/base/best.pt"),
        "human_color": "black",
        "device": "cpu",
        "opening_book": None,
        "search_simulations": 0,
        "c_puct": 1.5,
        "dataset_path": Path("data/datasets/human_gui.pt"),
        "pgn_dir": Path("data/games/human_gui"),
        "training_enabled": True,
    }


def test_train_resume_and_init_checkpoint_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "train",
                "--config",
                "configs/dev.yaml",
                "--resume",
                "checkpoints/run/last.pt",
                "--init-checkpoint",
                "checkpoints/base/best.pt",
            ]
        )


def test_arena_parser_accepts_neural_mcts_search_options() -> None:
    args = build_parser().parse_args(
        [
            "arena",
            "--white",
            "neural-mcts",
            "--white-checkpoint",
            "checkpoints/candidate/best.pt",
            "--black",
            "minimax",
            "--simulations",
            "96",
            "--c-puct",
            "1.25",
            "--white-opening-book",
            "data/opening_books/all_chess_openings_v1.json",
        ]
    )

    assert args.white == "neural-mcts"
    assert args.simulations == 96
    assert args.c_puct == 1.25
    assert args.white_opening_book == Path("data/opening_books/all_chess_openings_v1.json")


def test_opening_import_parser_requires_an_explicit_config() -> None:
    args = build_parser().parse_args(
        [
            "import-openings",
            "--config",
            "configs/all_chess_openings_import.yaml",
            "--confirm-external-training-data",
        ]
    )

    assert args.config == Path("configs/all_chess_openings_import.yaml")
    assert args.confirm_external_training_data is True


def test_teacher_cycle_parser_accepts_config_device_and_resume_override() -> None:
    args = build_parser().parse_args(
        [
            "teacher-cycle",
            "--config",
            "configs/d1_opponent_d2_teacher_100.yaml",
            "--device",
            "cuda",
            "--cycles",
            "200",
            "--normal-start-every",
            "5",
            "--anchor-correction-share",
            "0.25",
            "--no-evaluate-after-normal-start",
            "--no-resume",
        ]
    )

    assert args.config == Path("configs/d1_opponent_d2_teacher_100.yaml")
    assert args.device == "cuda"
    assert args.cycles == 200
    assert args.normal_start_every == 5
    assert args.anchor_correction_share == 0.25
    assert args.evaluate_after_normal_start is False
    assert args.resume is False


def test_teacher_batch_parser_accepts_generation_only_options() -> None:
    args = build_parser().parse_args(
        [
            "teacher-batch",
            "--config",
            "configs/gen1_vs_d1_d2_teacher_500.yaml",
            "--device",
            "cuda",
            "--no-resume",
        ]
    )

    assert args.config == Path("configs/gen1_vs_d1_d2_teacher_500.yaml")
    assert args.device == "cuda"
    assert args.resume is False


def test_mixed_batch_parser_accepts_generation_only_options() -> None:
    args = build_parser().parse_args(
        [
            "mixed-batch",
            "--config",
            "configs/gen2_mixed_d1_d2_400.yaml",
            "--device",
            "cuda",
            "--no-resume",
        ]
    )

    assert args.config == Path("configs/gen2_mixed_d1_d2_400.yaml")
    assert args.device == "cuda"
    assert args.resume is False


def test_gated_refinement_parser_accepts_total_round_override() -> None:
    args = build_parser().parse_args(
        [
            "gated-refinement",
            "--config",
            "configs/d1_gated_refinement_from_epoch399.yaml",
            "--device",
            "cuda",
            "--rounds",
            "20",
            "--initialize-only",
            "--no-resume",
        ]
    )

    assert args.config == Path("configs/d1_gated_refinement_from_epoch399.yaml")
    assert args.device == "cuda"
    assert args.rounds == 20
    assert args.initialize_only is True
    assert args.resume is False


def test_gated_audit_parser_accepts_holdout_overrides() -> None:
    args = build_parser().parse_args(
        [
            "gated-audit",
            "--config",
            "configs/d1_gated_refinement_from_epoch399.yaml",
            "--device",
            "cuda",
            "--openings",
            "20",
            "--audit-seed",
            "9202026",
        ]
    )

    assert args.config == Path("configs/d1_gated_refinement_from_epoch399.yaml")
    assert args.device == "cuda"
    assert args.openings == 20
    assert args.audit_seed == 9_202_026


def test_paired_audit_parser_accepts_candidate_champion_and_manifest() -> None:
    args = build_parser().parse_args(
        [
            "paired-audit",
            "--candidate",
            "checkpoints/candidate/best.pt",
            "--champion",
            "checkpoints/champion/generation_0001.pt",
            "--openings",
            "40",
            "--audit-seed",
            "9300205",
            "--device",
            "cuda",
            "--exclude-manifest",
            "data/curricula/batch/manifest.json",
        ]
    )

    assert args.candidate == Path("checkpoints/candidate/best.pt")
    assert args.champion == Path("checkpoints/champion/generation_0001.pt")
    assert args.openings == 40
    assert args.audit_seed == 9_300_205
    assert args.exclude_manifest == Path("data/curricula/batch/manifest.json")


def test_gameplay_select_parser_accepts_config_and_device() -> None:
    args = build_parser().parse_args(
        [
            "gameplay-select",
            "--config",
            "configs/gen2_mixed_d1_d2_400_select.yaml",
            "--device",
            "cuda",
        ]
    )

    assert args.config == Path("configs/gen2_mixed_d1_d2_400_select.yaml")
    assert args.device == "cuda"


def test_init_checkpoint_uses_saved_architecture_and_fresh_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "fine_tune.yaml"
    output_dir = tmp_path / "fine_tune_output"
    config_path.write_text(
        "\n".join(
            (
                "seed: 17",
                "device: cpu",
                "model:",
                "  channels: deliberately-ignored",
                "  residual_blocks: deliberately-ignored",
                "training:",
                f"  checkpoint_dir: {output_dir.as_posix()}",
                f"  metrics_path: {(tmp_path / 'metrics.jsonl').as_posix()}",
                "  epochs: 1",
            )
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source" / "best.pt"
    restored_model = object()
    observed: dict[str, Any] = {}

    def fake_load_model(
        path: str | Path,
        *,
        device: str,
        eval_mode: bool,
    ) -> object:
        observed["load"] = (Path(path), device, eval_mode)
        return restored_model

    class FakeMetrics:
        def to_dict(self) -> dict[str, float]:
            return {"training_loss": 1.0}

    def fake_train_model(model: object, config: Any) -> list[FakeMetrics]:
        observed["train"] = (model, config)
        return [FakeMetrics()]

    monkeypatch.setattr(model_module, "load_model", fake_load_model)
    monkeypatch.setattr(training_module, "train_model", fake_train_model)

    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(config_path),
            "--init-checkpoint",
            str(source),
        ]
    )
    assert command_train(args) == 0

    assert observed["load"] == (source, "cpu", False)
    trained_model, training_config = observed["train"]
    assert trained_model is restored_model
    assert training_config.resume_from is None


def test_init_checkpoint_cannot_overwrite_its_source_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "champion"
    source = source_dir / "best.pt"
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        "\n".join(
            (
                "training:",
                f"  checkpoint_dir: {source_dir.as_posix()}",
                f"  metrics_path: {(tmp_path / 'metrics.jsonl').as_posix()}",
            )
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(config_path),
            "--init-checkpoint",
            str(source),
        ]
    )

    with pytest.raises(ValueError, match="different checkpoint directory"):
        command_train(args)


def test_init_checkpoint_cannot_overwrite_an_existing_candidate(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    (output_dir / "best.pt").touch()
    metrics_path = tmp_path / "candidate.jsonl"
    config_path = tmp_path / "existing.yaml"
    config_path.write_text(
        "\n".join(
            (
                "training:",
                f"  checkpoint_dir: {output_dir.as_posix()}",
                f"  metrics_path: {metrics_path.as_posix()}",
            )
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(config_path),
            "--init-checkpoint",
            str(tmp_path / "source" / "best.pt"),
        ]
    )

    with pytest.raises(ValueError, match="will not overwrite an earlier candidate"):
        command_train(args)
