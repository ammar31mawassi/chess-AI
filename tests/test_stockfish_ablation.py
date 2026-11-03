"""Preparation reserves one-factor configs without touching training artifacts."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import yaml


def test_ablation_reserves_matching_inputs_and_only_changes_source_exposure(tmp_path, monkeypatch):
    prepare = runpy.run_path(str(Path("scripts/new_stockfish_ablation.py").resolve()))[
        "prepare_ablation"
    ]
    replay = tmp_path / "replay.pt"
    replay.write_bytes(b"preparation only checks existence")
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"immutable")
    train = tmp_path / "train.yaml"
    train.write_text(
        yaml.safe_dump(
            {
                "seed": 2037,
                "device": "cpu",
                "training": {
                    "dataset_path": str(replay),
                    "epochs": 60,
                    "learning_rate": 5e-7,
                    "value_loss_weight": 0.1,
                    "scheduler": True,
                    "game_balanced_sampling": True,
                    "validation_group_key": "composition_group_id",
                },
            }
        )
    )
    select = tmp_path / "select.yaml"
    select.write_text(
        yaml.safe_dump(
            {
                "seed": 10060205,
                "gameplay_selection": {
                    "champion_checkpoint": str(champion),
                    "opponent_depths": [1, 2],
                    "training_manifest": "already-reviewed.json",
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    root = prepare("test", train, select)
    control = yaml.safe_load((root / "control_train.yaml").read_text())
    focused = yaml.safe_load((root / "teacher_focus_train.yaml").read_text())
    assert control["seed"] == focused["seed"] == 2037
    c, f = control["training"], focused["training"]
    assert c["source_sampling_multipliers"] == {}
    assert f["source_sampling_multipliers"] == {"stockfish_continuations": 8}
    for settings in (c, f):
        assert settings["epochs"] == 10 and settings["value_loss_weight"] == 0.1
        assert settings["dataset_path"] == str(replay)
    excluded = {"checkpoint_dir", "metrics_path", "source_sampling_multipliers"}
    assert {k: v for k, v in c.items() if k not in excluded} == {
        k: v for k, v in f.items() if k not in excluded
    }
    assert yaml.safe_load((root / "control_select.yaml").read_text())["seed"] == 10112035
    assert not Path("checkpoints").exists() and not Path("data/metrics").exists()
    assert champion.read_bytes() == b"immutable"
    with pytest.raises(FileExistsError):
        prepare("test", train, select)
    with pytest.raises(ValueError, match="safe"):
        prepare("../escape", train, select)
