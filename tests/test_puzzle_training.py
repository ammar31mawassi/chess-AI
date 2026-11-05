"""Puzzle setup semantics, disjoint splits and isolated training; no real training/UCI."""

import json
import random
from contextlib import closing
from pathlib import Path
from typing import ClassVar

import chess
import numpy as np
import pytest

from chess_ai.data import puzzle_training as work
from chess_ai.data.examples import load_dataset, save_dataset
from chess_ai.data.failure_training import position_keys
from chess_ai.data.full_corpus import SPEC
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import atomic_json, sha256_file, signature


def rows(count):
    rng = random.Random(77)
    result = []
    for index in range(count):
        board = chess.Board()
        for _ in range(8 + index % 8):
            board.push(rng.choice(list(board.legal_moves)))
        moves = []
        start = board.fen()
        for _ in range(4):
            move = rng.choice(list(board.legal_moves))
            moves.append(move.uci())
            board.push(move)
        result.append(
            {
                "PuzzleId": f"p{index}",
                "GameId": f"{index:08d}#20",
                "FEN": start,
                "Moves": " ".join(moves),
                "Rating": 1200,
                "Themes": ["fork"],
            }
        )
    return result


def test_setup_move_is_not_the_solver_target_and_complete_line_is_validated():
    raw = rows(1)[0]
    root = work.puzzle_root(raw)
    board = chess.Board(raw["FEN"])
    board.push_uci(raw["Moves"].split()[0])
    assert root["fen"].split()[:4] == board.fen(en_passant="fen").split()[:4]
    assert root["targets"] == [raw["Moves"].split()[1]]
    raw["Moves"] += " a1a8"
    with pytest.raises(ValueError):
        work.puzzle_root(raw)


def test_mate_targets_include_all_legal_mates():
    raw = {
        "PuzzleId": "mate",
        "GameId": "abcd1234",
        "Rating": 800,
        "Themes": ["mateIn1"],
        "FEN": "r4rk1/pp3ppp/8/2bqp2b/3N4/P2Bn2P/1PPN1P2/R1BQ1RK1 w - - 1 15",
        "Moves": "d1h5 d5g2",
    }
    row = work.puzzle_root(raw)
    board = chess.Board(row["fen"])
    expected = []
    for move in board.legal_moves:
        after = board.copy()
        after.push(move)
        if after.is_checkmate():
            expected.append(move.uci())
    assert row["targets"] == sorted(expected)


def test_selection_is_seeded_grouped_and_excludes_history_and_duplicates():
    raw = rows(50)
    protected = position_keys(chess.Board(work.puzzle_root(raw[0])["fen"]))
    historical = position_keys(chess.Board(work.puzzle_root(raw[1])["fen"]))
    duplicate = {**raw[2], "PuzzleId": "duplicate", "GameId": "zzzz9999"}
    selected, rejected = work.choose_puzzles(
        [*raw[:3], duplicate, *raw[3:]], protected, historical, count=20, seed=123
    )
    again, _ = work.choose_puzzles(
        [*raw[:3], duplicate, *raw[3:]], protected, historical, count=20, seed=123
    )
    assert selected == again
    assert rejected["protected_or_duplicate"] >= 3
    assert len({row["game"] for row in selected}) == 20
    assert {
        split: sum(row["split"] == split for row in selected)
        for split in ("train", "validation", "confirmation")
    } == {"train": 18, "validation": 1, "confirmation": 1}
    seen = set()
    for row in selected:
        keys = position_keys(chess.Board(row["fen"]))
        assert not keys & (seen | protected | historical)
        seen.update(keys)


def test_import_acknowledgement_is_required_before_io(tmp_path):
    with pytest.raises(ValueError, match="confirm-training-import"):
        work.prepare(tmp_path, confirm_training_import=False)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("data/experiments/puzzle_test")
    root.mkdir(parents=True)
    selected, _ = work.choose_puzzles(rows(40), set(), set(), count=20, seed=3)
    work.signed(root / "selection.json", {"rows": selected})
    old_root = Path("data/experiments/old")
    old_root.mkdir()
    originals, _ = work.choose_puzzles(rows(40)[20:], set(), set(), count=20, seed=4)
    # Only two old examples are needed; preserve an existing train/validation split.
    originals = originals[:2]
    for row, split in zip(originals, ["train", "validation"], strict=True):
        row["split"] = split
        row["group_id"] = "old:" + row["group_id"]
    old_path = old_root / "old.pt"
    examples = [work.puzzle_example(row, {"value": 0.2, "audit": {}}) for row in originals]
    save_dataset(old_path, examples, format_version=2, metadata={"value_target_spec": SPEC})
    work._manifest(old_root / "dataset.json", [(old_path, 2)], {"train": 1, "validation": 1})
    plan = {
        "format": work.FORMAT,
        "version": 1,
        "seed": 3,
        "selection_sha256": sha256_file(root / "selection.json"),
        "frozen_files": {},
        "corpus": str(old_root),
        "champion": "frozen-champion.pt",
        "model": {"channels": 4, "residual_blocks": 1},
        "training_template": {"training": {}},
    }
    work.signed(root / "plan.json", plan)
    with closing(work._connect(root)) as db:
        db.execute(
            "INSERT INTO bindings VALUES ('teacher',?)",
            (json.dumps({"plan": sha256_file(root / "plan.json")}),),
        )
        for row in selected:
            if row["split"] != "confirmation":
                payload = {"value": 0.3, "audit": {}}
                db.execute(
                    "INSERT INTO labels VALUES (?,?)",
                    (row["key"], json.dumps({"payload": payload, "checksum": signature(payload)})),
                )
        db.commit()
    return root, old_path, selected


def test_finalize_composes_all_old_rows_preserves_splits_and_excludes_confirmation(prepared):
    root, old_path, selected = prepared
    old_hash = sha256_file(old_path)
    receipt = work.finalize(root)
    assert receipt["combined_splits"] == {"train": 19, "validation": 2}
    assert receipt["puzzle_splits"] == {"train": 18, "validation": 1}
    assert sha256_file(old_path) == old_hash
    combined = load_dataset(root.parent / f"{root.name}_combined.json")
    assert len(combined) == 21
    assert not {row["key"] for row in selected if row["split"] == "confirmation"} & {
        row.metadata["position_key"] for row in combined
    }
    puzzle = load_dataset(root / "puzzles.json")
    assert len(puzzle) == 19
    for example in puzzle:
        target = next(row for row in selected if row["key"] == example.metadata["position_key"])
        assert np.isclose(
            example.target_policy[MoveEncoder().encode(chess.Move.from_uci(target["targets"][0]))],
            1.0,
        )
    assert work.finalize(root) == receipt


@pytest.mark.parametrize("arm", ["scratch", "finetune"])
def test_train_uses_correct_initialization_and_refuses_overwrite(prepared, monkeypatch, arm):
    root, _, _ = prepared
    work.finalize(root)
    commands = []
    monkeypatch.setattr(work.subprocess, "run", lambda command, **kwargs: commands.append(command))
    work.train(root, arm, device="cpu")
    assert ("--init-checkpoint" in commands[0]) == (arm == "finetune")
    if arm == "finetune":
        assert commands[0][-1] == "frozen-champion.pt"
    checkpoint = Path("checkpoints") / f"{root.name}_{arm}" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="resume"):
        work.train(root, arm)
    work.train(root, arm, resume=True)
    assert "--resume" in commands[-1] and "--init-checkpoint" not in commands[-1]


def test_tampered_selection_or_ready_config_is_rejected(prepared):
    root, _, _ = prepared
    receipt = work.finalize(root)
    Path(receipt["configs"]["scratch"]).write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        work.verify_ready(root)
    atomic_json(root / "selection.json", {"payload": {}, "checksum": "wrong"})
    with pytest.raises(ValueError, match="selection changed"):
        work._verified(root)


def test_value_labeling_resumes_and_never_labels_confirmation(prepared, monkeypatch):
    root, _, selected = prepared
    (root / "labels.sqlite").unlink()
    plan = work.checked(root / "plan.json")
    plan["stockfish"] = {"executable": "unused.exe", "source_commit": "a" * 40}
    work.signed(root / "plan.json", plan)
    calls = []

    class Teacher:
        provenance: ClassVar[dict[str, str]] = {"mock": "offline-only"}

        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def analyze(self, board):
            calls.append(board.fen())
            return None, {"stockfish_evaluation": 0.4}

    monkeypatch.setattr(work, "StockfishTeacher", Teacher)
    assert work.label(root, 3)["labeled_total"] == 3
    assert work.label(root, 3)["labeled_total"] == 6
    assert work.label(root)["remaining"] == 0
    assert len(calls) == 19 and len(set(calls)) == 19
    assert not {row["fen"] for row in selected if row["split"] == "confirmation"} & set(calls)
    Teacher.provenance = {"mock": "changed"}
    with pytest.raises(ValueError, match="provenance changed"):
        work.label(root)


def test_first_move_evaluation_resumes_without_replaying_positions(prepared, monkeypatch):
    root, _, selected = prepared
    checkpoint = Path("candidate.pt")
    checkpoint.write_bytes(b"mock checkpoint")
    targets = {row["fen"]: row["targets"][0] for row in selected}
    calls = []

    class Agent:
        def __init__(self, *args, **kwargs):
            assert kwargs["simulations"] == 32

        def choose_move(self, board):
            calls.append(board.fen(en_passant="fen"))
            return chess.Move.from_uci(targets[calls[-1]])

    monkeypatch.setattr(work, "NeuralMCTSAgent", Agent)
    result = work.evaluate(root, checkpoint, device="cpu")
    assert result["positions"] == 1 and result["first_move_accuracy"] == 1
    assert result["full_puzzle_solve_rate_measured"] is False
    assert work.evaluate(root, checkpoint, device="cpu") == result
    assert len(calls) == 1
