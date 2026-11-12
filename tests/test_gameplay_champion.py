"""A promoted model stays selected until explicitly changed or overridden."""

from pathlib import Path

import pytest

from chess_ai.cli import build_parser
from chess_ai.gui.app import discover_default_checkpoint, resolve_gui_defaults
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import CHAMPION_FORMAT, load_gameplay_champion


@pytest.fixture
def champion_record(tmp_path):
    checkpoint = tmp_path / "champions" / "epoch_0060.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen promoted checkpoint")
    record = {
        "format": CHAMPION_FORMAT,
        "version": 1,
        "checkpoint": "champions/epoch_0060.pt",
        "sha256": sha256_file(checkpoint),
        "inference": {"search_simulations": 32, "c_puct": 1.5, "opening_book": None},
    }
    atomic_json(tmp_path / "champion.json", record)
    return checkpoint, record


def test_registered_champion_overrides_newer_unpromoted_sessions(tmp_path, champion_record):
    checkpoint, _ = champion_record
    newer = tmp_path / "human_sessions" / "newer" / "best.pt"
    newer.parent.mkdir(parents=True)
    newer.touch()
    assert discover_default_checkpoint(tmp_path) == checkpoint
    args = build_parser().parse_args(["gui"])
    assert resolve_gui_defaults(
        args.checkpoint, args.search_simulations, args.c_puct, checkpoint_root=tmp_path
    ) == (checkpoint, 1_000_000, 1.5, 5.0)


def test_explicit_checkpoint_and_search_overrides(tmp_path, champion_record):
    checkpoint, _ = champion_record
    assert resolve_gui_defaults(search_simulations=0, c_puct=1.25, checkpoint_root=tmp_path) == (
        checkpoint,
        0,
        1.25,
        None,
    )
    rollback = Path("old_epoch_0120.pt")
    assert resolve_gui_defaults(search_simulations=32, checkpoint_root=tmp_path) == (
        checkpoint,
        32,
        1.5,
        None,
    )
    assert resolve_gui_defaults(rollback, checkpoint_root=tmp_path) == (rollback, 0, 1.5, None)
    assert resolve_gui_defaults(rollback, 32, 1.5, checkpoint_root=tmp_path) == (
        rollback,
        32,
        1.5,
        None,
    )


@pytest.mark.parametrize("delete", [False, True])
def test_damaged_champion_fails_instead_of_selecting_another_model(
    tmp_path, champion_record, delete
):
    checkpoint, _ = champion_record
    if delete:
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"changed weights")
    fallback = tmp_path / "dev" / "best.pt"
    fallback.parent.mkdir()
    fallback.touch()
    with pytest.raises(ValueError, match="Invalid gameplay champion"):
        discover_default_checkpoint(tmp_path)
    # An explicit rollback remains usable even if the champion record is broken.
    assert resolve_gui_defaults(fallback, checkpoint_root=tmp_path) == (fallback, 0, 1.5, None)


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"checkpoint": "../outside.pt"},
        {"inference": {"search_simulations": -1, "c_puct": 1.5}},
        {"inference": {"search_simulations": True, "c_puct": 1.5}},
        {"inference": {"search_simulations": 32, "c_puct": 0}},
        {"inference": {"search_simulations": 32, "c_puct": 1.5, "opening_book": "book.json"}},
    ],
)
def test_unknown_record_or_invalid_inference_settings_fail(tmp_path, champion_record, change):
    _, record = champion_record
    atomic_json(tmp_path / "champion.json", {**record, **change})
    with pytest.raises(ValueError, match="Invalid gameplay champion"):
        load_gameplay_champion(tmp_path)


def test_no_champion_keeps_existing_greedy_discovery_defaults(tmp_path):
    assert load_gameplay_champion(tmp_path) is None
    assert resolve_gui_defaults(checkpoint_root=tmp_path) == (None, 0, 1.5, None)
