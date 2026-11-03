"""Personal moves supply positions; independent Stockfish supplies every target."""

from __future__ import annotations

import json
import math
import runpy
from dataclasses import replace
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

from chess_ai.config import load_config
from chess_ai.data.examples import load_dataset
from chess_ai.data.personal_pgn import snapshot_personal_pgns
from chess_ai.data.stockfish_rebuild import (
    RebuildConfig,
    _game_digest,
    _split,
    finalize,
    inventory,
    label_positions,
)
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model import PolicyValueNet, save_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import CHAMPION_FORMAT


def _game(moves: str, *, user_white: bool = True) -> chess.pgn.Game:
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    game = chess.pgn.Game.from_board(board)
    game.headers.update(
        {
            "White": "MyPlayer" if user_white else "Other",
            "Black": "Other" if user_white else "MyPlayer",
            "Result": "1-0",
        }
    )
    return game


def test_pgn_only_teacher_targets_both_colors_and_frozen_sources(tmp_path, teacher_setup):
    teacher = teacher_setup[0]
    source = tmp_path / "downloaded"
    source.mkdir()
    games = [
        _game("e4 e5 Nf3 Nc6"),
        _game("d4 d5 c4 e6", user_white=False),
        _game("c4 e5 Nc3 Nf6"),
        _game("Nf3 d5 g3 e5", user_white=False),
    ]
    original = source / "month.pgn"
    original.write_text("\n\n".join(str(game) for game in games), encoding="utf-8")
    before = original.read_bytes()
    snapshots = tmp_path / "experiment/source_games"
    report = snapshot_personal_pgns(source, "myplayer", snapshots)
    assert report["counts"]["games"] == 4 and report["counts"]["recorded_positions"] == 16
    assert report["counts"]["white_positions"] == report["counts"]["black_positions"] == 8
    assert original.read_bytes() == before == (snapshots / original.name).read_bytes()
    assert not list(snapshots.rglob("*.pt"))
    config = RebuildConfig(
        tmp_path / "experiment/relabel",
        (),
        (snapshots,),
        teacher,
        validation_fraction=0.5,
        shard_size=3,
    )
    config = next(
        replace(config, seed=seed)
        for seed in range(100)
        if len({_split(replace(config, seed=seed), f"pgn-{_game_digest(game)}") for game in games})
        == 2
    )
    result = inventory(config)
    assert result["complete_pgn_games"] == 4 and result["counts"]["pgn_positions"] == 16
    # Later archive downloads cannot change the frozen source snapshot.
    original.write_text("new archive version", encoding="utf-8")
    assert label_positions(config)["complete"]
    finalize(config)
    data = load_dataset(config.run_dir / "dataset.json")
    assert data.format_version == 2
    assert {row.metadata["player_to_move"] for row in data} == {"white", "black"}
    assert all(row.target_value == pytest.approx(math.tanh(100 / 600)) for row in data)
    start = next(row for row in data if row.metadata["fen"] == chess.STARTING_FEN)
    # All four human first moves differ from this independent mock teacher's preferences.
    assert start.target_policy[MoveEncoder().encode(chess.Move.from_uci("e2e4"))] == 0
    assert start.metadata["teacher_move_uci"] == "a2a3"
    groups = {
        split: {row.metadata["group_id"] for row in data if row.metadata["split"] == split}
        for split in ("train", "validation")
    }
    assert (
        groups["train"]
        and groups["validation"]
        and groups["train"].isdisjoint(groups["validation"])
    )
    (snapshots / original.name).write_text("changed frozen PGN", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        label_positions(config)


@pytest.mark.parametrize(
    "problem", ["username", "unfinished", "evaluation", "variant", "illegal", "empty"]
)
def test_invalid_or_unapproved_games_do_not_create_a_snapshot(tmp_path, problem):
    source = tmp_path / "source"
    source.mkdir()
    game = _game("e4 e5")
    if problem == "empty":
        game = _game("")
    if problem == "username":
        game.headers["White"] = "SomeoneElse"
    elif problem == "unfinished":
        game.headers["Result"] = "*"
    elif problem == "evaluation":
        game.headers["TrainingData"] = "false"
    elif problem == "variant":
        game.headers["Variant"] = "Chess960"
    text = str(game)
    if problem == "illegal":
        text = text.replace("e4 e5", "e4 e4")
    (source / "month.pgn").write_text(text, encoding="utf-8")
    destination = tmp_path / "experiment/source_games"
    with pytest.raises(ValueError):
        snapshot_personal_pgns(source, "myplayer", destination)
    assert not destination.exists()


def test_duplicates_are_reported_and_existing_snapshot_is_never_overwritten(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    text = str(_game("e4 e5"))
    for name in ("a.pgn", "b.pgn"):
        (source / name).write_text(text, encoding="utf-8")
    (source / "empty.pgn").write_text(str(_game("")), encoding="utf-8")
    snapshot = tmp_path / "experiment/source_games"
    report = snapshot_personal_pgns(source, "myplayer", snapshot)
    assert report["counts"]["games"] == 3
    assert report["counts"]["empty_games"] == 1
    assert report["counts"]["games_with_positions"] == 2
    assert report["counts"]["duplicate_game_movetext"] == 1
    assert report["unique_game_movetexts"] == 1
    with pytest.raises(ValueError, match="snapshot exists"):
        snapshot_personal_pgns(source, "myplayer", snapshot)


def test_preparation_requires_import_acknowledgement_and_uses_frozen_champion(
    tmp_path, teacher_setup, monkeypatch
):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/new_personal_stockfish_training.py")
    )
    prepare = module["prepare_personal_training"]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PermissionError, match="confirm-training-import"):
        prepare(tmp_path, "MyPlayer", "personal_test")
    source = tmp_path / "downloaded"
    source.mkdir()
    (source / "month.pgn").write_text(str(_game("e4 e5")), encoding="utf-8")
    spec = {"kind": "stockfish_cp_tanh_v1", "perspective": "side_to_move", "scale_cp": 600.0}
    checkpoint = Path("checkpoints/champions/epoch_0060.pt")
    torch.manual_seed(63)
    save_checkpoint(
        checkpoint,
        PolicyValueNet(channels=2, residual_blocks=0),
        epoch=60,
        training_config={"value_target_spec": spec},
    )
    champion = {
        "format": CHAMPION_FORMAT,
        "version": 1,
        "checkpoint": "champions/epoch_0060.pt",
        "sha256": sha256_file(checkpoint),
        "inference": {"search_simulations": 32, "c_puct": 1.5},
        "source_checkpoint": "checkpoints/original/best.pt",
        "evidence": [],
    }
    atomic_json(Path("checkpoints/champion.json"), champion)
    original = Path("data/experiments/original")
    atomic_json(original / "relabel/dataset.json", {"value_target_spec": spec})
    teacher = teacher_setup[0]
    teacher_mapping = {
        "executable": str(teacher.executable),
        "source_commit": teacher.source_commit,
    }
    (original / "configs").mkdir()
    config_path = original / "configs/relabel.yaml"
    config_path.write_text(json.dumps({"stockfish_rebuild": {"stockfish": teacher_mapping}}))
    atomic_json(original / "experiment.json", {"relabel_config": str(config_path)})
    pointer_before = Path("checkpoints/champion.json").read_bytes()
    configs = prepare(source, "MyPlayer", "personal_test", confirm_training_import=True)
    relabel = load_config(configs / "relabel.yaml")["stockfish_rebuild"]
    assert relabel["datasets"] == [] and len(relabel["pgn_roots"]) == 1
    compose = load_config(configs / "compose.yaml")["refinement_replay"]
    assert compose["fresh_path"].endswith("dataset.json")
    assert load_config(configs / "train.yaml")["training"]["baseline_eligible"]
    assert Path("checkpoints/champion.json").read_bytes() == pointer_before
    assert not (configs.parent / "relabel").exists()
