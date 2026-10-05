from __future__ import annotations

import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

import chess_ai.gui.session_training as session_training
from chess_ai.gui.session_training import (
    SESSION_PLAN_FORMAT,
    SESSION_PLAN_VERSION,
    SessionTrainingError,
    SessionTrainingPlan,
    SessionTrainingProcessError,
    create_session_training_plan,
    run_captured_command,
    run_session_training,
    update_session_dataset_path,
)


def _write_template(path: Path, *, seed: int = 42) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"seed: {seed}",
                "device: auto",
                "training:",
                "  dataset_path: data/datasets/old.pt",
                "  checkpoint_dir: checkpoints/old",
                "  metrics_path: data/metrics/old.jsonl",
                "  batch_size: 16",
                "  epochs: 5",
                "  learning_rate: 0.0001",
                "  weight_decay: 0.0001",
                "  gradient_clip: 1.0",
                "  validation_fraction: 0.2",
                "  scheduler: false",
                "  log_every: 5",
                "  resume_from: checkpoints/stale/last.pt",
            )
        ),
        encoding="utf-8",
    )


def _make_plan(tmp_path: Path, *, session_id: str = "session_001") -> SessionTrainingPlan:
    template = tmp_path / "templates" / "human_gui.yaml"
    _write_template(template)
    return create_session_training_plan(
        tmp_path / "datasets" / "human.pt",
        session_id=session_id,
        created_at=datetime(2026, 8, 5, 12, 30, tzinfo=UTC),
        sessions_root=tmp_path / "sessions",
        checkpoints_root=tmp_path / "checkpoints",
        metrics_root=tmp_path / "metrics",
        template_path=template,
    )


def _loaded_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


class _FakeModel:
    config = object()


class _FakeLoadedCheckpoint:
    model = _FakeModel()


class _FakeDataset:
    metadata: ClassVar = {"completed_games": 3}
    created_utc = "2026-08-05T12:29:00+00:00"

    def __len__(self) -> int:
        return 12


def test_create_plan_preserves_settings_and_overrides_only_artifact_paths(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)

    assert plan.version == SESSION_PLAN_VERSION
    assert plan.session_id == "session_001"
    assert plan.created_utc == "2026-08-05T12:30:00+00:00"
    assert plan.config_path == tmp_path / "sessions/session_001/training.yaml"
    assert plan.checkpoint_dir == tmp_path / "checkpoints/session_001"
    assert plan.metrics_path == tmp_path / "metrics/session_001.jsonl"
    assert plan.best_checkpoint == tmp_path / "checkpoints/session_001/best.pt"

    raw = _loaded_yaml(plan.config_path)
    assert raw["seed"] == 42
    assert raw["device"] == "auto"
    assert raw["training"] == {
        "dataset_path": plan.dataset_path.as_posix(),
        "checkpoint_dir": plan.checkpoint_dir.as_posix(),
        "metrics_path": plan.metrics_path.as_posix(),
        "batch_size": 16,
        "epochs": 5,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "validation_fraction": 0.2,
        "scheduler": False,
        "log_every": 5,
    }
    assert raw["gui_session"] == {
        "format": SESSION_PLAN_FORMAT,
        "version": SESSION_PLAN_VERSION,
        "session_id": plan.session_id,
        "created_utc": plan.created_utc,
    }
    assert not list(plan.config_path.parent.glob(".*.tmp"))


def test_generated_ids_are_safe_and_unique_for_the_same_injected_time(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    _write_template(template)
    created_at = datetime(2026, 8, 5, 12, 30, 45, 123456, tzinfo=UTC)
    options = {
        "created_at": created_at,
        "sessions_root": tmp_path / "sessions",
        "checkpoints_root": tmp_path / "checkpoints",
        "metrics_root": tmp_path / "metrics",
        "template_path": template,
    }

    first = create_session_training_plan(**options)
    second = create_session_training_plan(**options)

    assert first.session_id != second.session_id
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", first.session_id)
    assert first.session_id.startswith("20260805T123045_123456Z_")


@pytest.mark.parametrize("session_id", ("", "../escape", "has space", "bad/name", ".hidden"))
def test_create_plan_rejects_unsafe_session_ids(tmp_path: Path, session_id: str) -> None:
    template = tmp_path / "template.yaml"
    _write_template(template)

    with pytest.raises(SessionTrainingError, match="Session ID"):
        create_session_training_plan(
            session_id=session_id,
            sessions_root=tmp_path / "sessions",
            checkpoints_root=tmp_path / "checkpoints",
            metrics_root=tmp_path / "metrics",
            template_path=template,
        )


@pytest.mark.parametrize("collision_kind", ("config", "checkpoint", "metrics"))
def test_create_plan_refuses_artifact_collisions(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    template = tmp_path / "template.yaml"
    _write_template(template)
    if collision_kind == "config":
        (tmp_path / "sessions/taken").mkdir(parents=True)
    elif collision_kind == "checkpoint":
        (tmp_path / "checkpoints/taken").mkdir(parents=True)
    else:
        (tmp_path / "metrics").mkdir(parents=True)
        (tmp_path / "metrics/taken.jsonl").touch()

    with pytest.raises(SessionTrainingError, match="collides"):
        create_session_training_plan(
            session_id="taken",
            sessions_root=tmp_path / "sessions",
            checkpoints_root=tmp_path / "checkpoints",
            metrics_root=tmp_path / "metrics",
            template_path=template,
        )


def test_update_dataset_path_is_atomic_and_preserves_seed_and_settings(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    replacement = tmp_path / "datasets" / "more_games.pt"

    updated = update_session_dataset_path(plan, replacement)

    assert plan.dataset_path != updated.dataset_path
    assert updated.dataset_path == replacement
    raw = _loaded_yaml(updated.config_path)
    assert raw["seed"] == 42
    assert raw["training"]["dataset_path"] == replacement.as_posix()
    assert raw["training"]["epochs"] == 5
    assert raw["training"]["learning_rate"] == 0.0001
    assert not list(updated.config_path.parent.glob(".*.tmp"))


def test_update_dataset_refuses_to_change_a_trained_session(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    plan.checkpoint_dir.mkdir(parents=True)
    plan.best_checkpoint.touch()

    with pytest.raises(SessionTrainingError, match="already has training artifacts"):
        update_session_dataset_path(plan, tmp_path / "other.pt")


def test_session_config_rejects_a_manually_added_resume_target(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    raw = _loaded_yaml(plan.config_path)
    raw["training"]["resume_from"] = "checkpoints/stale/last.pt"
    plan.config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(SessionTrainingError, match=r"cannot contain training\.resume_from"):
        update_session_dataset_path(plan, tmp_path / "other.pt")


def test_run_training_validates_inputs_and_invokes_existing_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan(tmp_path)
    plan.dataset_path.parent.mkdir(parents=True)
    plan.dataset_path.touch()
    source = tmp_path / "source" / "best.pt"
    source.parent.mkdir(parents=True)
    source.touch()
    observed: dict[str, Any] = {}

    def fake_load_checkpoint(path: Path, *, map_location: str) -> _FakeLoadedCheckpoint:
        observed.setdefault("checkpoints", []).append((path, map_location))
        return _FakeLoadedCheckpoint()

    def fake_load_dataset(path: Path) -> _FakeDataset:
        observed["dataset"] = path
        return _FakeDataset()

    def fake_runner(command: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = tuple(command)
        plan.checkpoint_dir.mkdir(parents=True)
        plan.best_checkpoint.touch()
        return subprocess.CompletedProcess(command, 0, "epoch complete\n", "")

    monkeypatch.setattr(session_training, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(session_training, "load_human_gui_dataset", fake_load_dataset)

    result = run_session_training(
        plan,
        source_checkpoint=source,
        device="cuda",
        command_runner=fake_runner,
    )

    expected_command = (
        sys.executable,
        "-m",
        "chess_ai",
        "train",
        "--config",
        str(plan.config_path),
        "--init-checkpoint",
        str(source),
        "--device",
        "cuda",
    )
    assert observed == {
        "checkpoints": [(source, "cpu"), (plan.best_checkpoint, "cpu")],
        "dataset": plan.dataset_path,
        "command": expected_command,
    }
    assert result.command == expected_command
    assert result.returncode == 0
    assert result.stdout == "epoch complete\n"
    assert result.best_checkpoint == plan.best_checkpoint
    assert result.source_checkpoint == source
    assert result.dataset_path == plan.dataset_path
    assert result.dataset_examples == 12
    assert result.dataset_games == 3
    assert result.dataset_created_utc == "2026-08-05T12:29:00+00:00"


def test_run_training_requires_source_checkpoint_and_confirmed_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan(tmp_path)

    with pytest.raises(SessionTrainingError, match="Source neural checkpoint does not exist"):
        run_session_training(plan, source_checkpoint=tmp_path / "missing.pt")

    source = tmp_path / "source.pt"
    source.touch()
    monkeypatch.setattr(
        session_training,
        "load_checkpoint",
        lambda *args, **kwargs: _FakeLoadedCheckpoint(),
    )
    with pytest.raises(SessionTrainingError, match="Human-game dataset is not ready"):
        run_session_training(plan, source_checkpoint=source)


def test_run_training_surfaces_captured_cli_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan(tmp_path)
    plan.dataset_path.parent.mkdir(parents=True)
    plan.dataset_path.touch()
    source = tmp_path / "source.pt"
    source.touch()
    monkeypatch.setattr(
        session_training,
        "load_checkpoint",
        lambda *args, **kwargs: _FakeLoadedCheckpoint(),
    )
    monkeypatch.setattr(
        session_training,
        "load_human_gui_dataset",
        lambda path: _FakeDataset(),
    )

    def failed(command: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, "", "dataset needs two games")

    with pytest.raises(SessionTrainingProcessError, match="dataset needs two games") as caught:
        run_session_training(plan, source_checkpoint=source, command_runner=failed)

    assert caught.value.returncode == 2
    assert caught.value.stderr == "dataset needs two games"


def test_run_training_requires_best_checkpoint_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan(tmp_path)
    plan.dataset_path.parent.mkdir(parents=True)
    plan.dataset_path.touch()
    source = tmp_path / "source.pt"
    source.touch()
    monkeypatch.setattr(
        session_training,
        "load_checkpoint",
        lambda *args, **kwargs: _FakeLoadedCheckpoint(),
    )
    monkeypatch.setattr(
        session_training,
        "load_human_gui_dataset",
        lambda path: _FakeDataset(),
    )

    def no_artifact(command: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "done", "")

    with pytest.raises(SessionTrainingError, match="did not create the expected"):
        run_session_training(plan, source_checkpoint=source, command_runner=no_artifact)


def test_run_training_rejects_candidate_with_changed_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan(tmp_path)
    plan.dataset_path.parent.mkdir(parents=True)
    plan.dataset_path.touch()
    source = tmp_path / "source.pt"
    source.touch()

    class FakeModel:
        def __init__(self, config: object) -> None:
            self.config = config

    class FakeCheckpoint:
        def __init__(self, config: object) -> None:
            self.model = FakeModel(config)

    source_config = object()
    candidate_config = object()

    def fake_load_checkpoint(path: Path, *, map_location: str) -> FakeCheckpoint:
        del map_location
        config = source_config if path == source else candidate_config
        return FakeCheckpoint(config)

    def changed_candidate(command: Any) -> subprocess.CompletedProcess[str]:
        plan.checkpoint_dir.mkdir(parents=True)
        plan.best_checkpoint.touch()
        return subprocess.CompletedProcess(command, 0, "done", "")

    monkeypatch.setattr(session_training, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        session_training,
        "load_human_gui_dataset",
        lambda path: _FakeDataset(),
    )

    with pytest.raises(SessionTrainingError, match="changed the model architecture"):
        run_session_training(plan, source_checkpoint=source, command_runner=changed_candidate)


def test_run_training_refuses_existing_session_output_before_starting(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    plan.checkpoint_dir.mkdir(parents=True)
    (plan.checkpoint_dir / "epoch_0001.pt").touch()

    with pytest.raises(SessionTrainingError, match="already has training artifacts"):
        run_session_training(plan, source_checkpoint=tmp_path / "source.pt")


def test_default_command_runner_captures_output_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    completed = run_captured_command(("python", "-V"))

    assert completed.stdout == "ok"
    assert observed == {
        "command": ["python", "-V"],
        "kwargs": {"capture_output": True, "text": True, "check": False},
    }
