"""Explicit puzzle sampling, CPU value labels, and two isolated training arms.

One solver root per puzzle; first UCI move is setup, second is the target.
Existing shards/splits are reused without changing their labels or files.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import chess
import numpy as np
import yaml

from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.config import load_config
from chess_ai.data.examples import TrainingExample, save_dataset
from chess_ai.data.failure_training import position_keys, protected_inputs
from chess_ai.data.full_corpus import SPEC, source_shards
from chess_ai.data.stockfish_rebuild import _canonical
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.data.value_targets import SHARDED_DATASET_FORMAT, STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.training.trainer import TrainingConfig

FORMAT = "self-improving-chess-ai.lichess-puzzle-training"
DEFAULT_SNAPSHOT = Path("downloaded_datasets/lichess_puzzles/hf_461725966ea4")
DEFAULT_CORPUS = Path("data/experiments/full_gm_rebuild_v1_20261031_01")
DEFAULT_SUITE = Path("configs/search_development_v1.json")


def checked(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("checksum") != signature(raw["payload"]):
        raise ValueError(f"Puzzle artifact checksum mismatch: {path}")
    return dict(raw["payload"])


def signed(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, {"payload": payload, "checksum": signature(payload)})


def puzzle_root(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the entire supplied line with python-chess; return solver root only."""
    board = chess.Board(row["FEN"])
    if not board.is_valid():
        raise ValueError("Invalid puzzle FEN")
    moves = str(row["Moves"]).split()
    if len(moves) < 2:
        raise ValueError("Puzzle needs setup plus a solver move")
    setup = chess.Move.from_uci(moves[0])
    if setup not in board.legal_moves:
        raise ValueError("Illegal puzzle setup move")
    board.push(setup)
    if board.is_game_over(claim_draw=True):
        raise ValueError("Terminal puzzle root")
    root = board.copy(stack=False)
    for uci in moves[1:]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("Illegal puzzle solution move")
        board.push(move)
    game = str(row["GameId"]).split("/")[0].split("#")[0]
    if re.fullmatch(r"[A-Za-z0-9]{8}", game) is None:
        raise ValueError("Invalid puzzle source game ID")
    target = chess.Move.from_uci(moves[1])
    after = root.copy(stack=False)
    after.push(target)
    targets = [moves[1]]
    if after.is_checkmate():
        targets = []
        for move in root.legal_moves:
            after = root.copy(stack=False)
            after.push(move)
            if after.is_checkmate():
                targets.append(move.uci())
    key, fen = _canonical(root)
    return {
        "id": str(row["PuzzleId"]),
        "game": game,
        "key": key,
        "fen": fen,
        "targets": sorted(targets),
        "setup_fen": str(row["FEN"]),
        "line": moves,
        "rating": int(row["Rating"]),
        "themes": list(row["Themes"]),
    }


def choose_puzzles(
    rows: Iterable[Mapping[str, Any]],
    blocked: set[str],
    historical: set[str],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One puzzle per source game and disjoint input orbits across every split."""
    if count < 20 or count % 20:
        raise ValueError("Puzzle count must be a positive multiple of 20")
    selected: list[dict[str, Any]] = []
    games: set[str] = set()
    seen: set[str] = set()
    ids: set[str] = set()
    rejected: Counter[str] = Counter()
    for raw in rows:
        try:
            row = puzzle_root(raw)
        except (ValueError, KeyError, TypeError):
            rejected["invalid"] += 1
            continue
        if row["game"] in games or row["id"] in ids:
            rejected["same_game_or_id"] += 1
            continue
        keys = position_keys(chess.Board(row["fen"]))
        if keys & blocked or keys & historical or keys & seen:
            rejected["protected_or_duplicate"] += 1
            continue
        selected.append(row)
        games.add(row["game"])
        ids.add(row["id"])
        seen.update(keys)
        if len(selected) % 10000 == 0:
            print(f"Validated selected puzzles: {len(selected)}/{count}", flush=True)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible puzzles; increase candidate pool")
    # Selection order is randomized upstream. Independently shuffle fixed accepted
    # roots so the split is independent of rejection reasons and puzzle difficulty.
    import random

    random.Random(seed).shuffle(selected)
    n_train, n_validation = count * 9 // 10, count // 20
    for index, row in enumerate(selected):
        row["split"] = (
            "train"
            if index < n_train
            else "validation"
            if index < n_train + n_validation
            else "confirmation"
        )
        row["group_id"] = f"lichess-puzzle-game:{row['game']}"
    return selected, dict(rejected)


def candidates(snapshot: Path, seed: int, count: int) -> Iterator[dict[str, Any]]:
    """Random finite pool from all shards, with bounded Parquet batch memory."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "Install the puzzles extra: python -m pip install -e '.[puzzles]'"
        ) from exc
    manifest = json.loads((snapshot / "download_manifest.json").read_text(encoding="utf-8"))
    files = []
    for item in manifest["files"]:
        if not item["path"].endswith(".parquet"):
            continue
        path = (snapshot / item["path"]).resolve()
        if not path.is_relative_to(snapshot.resolve()) or sha256_file(path) != item["sha256"]:
            raise ValueError("Downloaded puzzle shard changed")
        files.append(path)
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    ranks = {
        int(index): rank
        for rank, index in enumerate(
            np.random.default_rng(seed).choice(total, min(total, count * 4), replace=False)
        )
    }
    pool: list[tuple[int, dict[str, Any]]] = []
    offset = 0
    columns = [
        "PuzzleId",
        "GameId",
        "FEN",
        "Moves",
        "Rating",
        "RatingDeviation",
        "Popularity",
        "NbPlays",
        "Themes",
    ]
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=columns):
            indexes = [index for index in range(batch.num_rows) if offset + index in ranks]
            for index, row in zip(
                indexes, batch.take(pa.array(indexes, type=pa.int64())).to_pylist(), strict=True
            ):
                if (
                    row["NbPlays"] >= 100
                    and row["Popularity"] >= 80
                    and row["RatingDeviation"] <= 100
                ):
                    pool.append((ranks[offset + index], row))
            offset += batch.num_rows
        print(
            f"Puzzle candidate scan: {offset}/{total}; quality-approved pool {len(pool)}",
            flush=True,
        )
    pool.sort(key=lambda item: item[0])
    yield from (row for _, row in pool)


def _connect(root: Path) -> sqlite3.Connection:
    db = sqlite3.connect(root / "labels.sqlite")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS labels(key TEXT PRIMARY KEY,payload TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS bindings(name TEXT PRIMARY KEY,payload TEXT)")
    return db


def _verified(root: Path) -> dict[str, Any]:
    plan = checked(root / "plan.json")
    if plan.get("format") != FORMAT or plan.get("version") != 1:
        raise ValueError("Unsupported puzzle training experiment")
    for path, digest in plan["frozen_files"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Frozen puzzle experiment input changed: {path}")
    if sha256_file(root / "selection.json") != plan["selection_sha256"]:
        raise ValueError("Puzzle selection changed")
    return plan


def prepare(
    root: Path,
    *,
    snapshot: Path = DEFAULT_SNAPSHOT,
    corpus: Path = DEFAULT_CORPUS,
    suite: Path = DEFAULT_SUITE,
    count: int = 250000,
    seed: int = 5112041,
    confirm_training_import: bool = False,
) -> dict[str, Any]:
    if not confirm_training_import:
        raise ValueError("Explicit --confirm-training-import is required")
    if not root.resolve().is_relative_to(Path("data/experiments").resolve()):
        raise ValueError("Puzzle experiment must stay under data/experiments")
    with run_lock(root):
        if (root / "plan.json").exists():
            plan = _verified(root)
            if plan["seed"] != seed or plan["selected"] != count:
                raise ValueError("Existing run uses a different seed/count")
            return plan["summary"]
        if root.exists() and any(root.iterdir()):
            raise ValueError("Incomplete reservation exists; preserve it and use a new run ID")
        champion = load_gameplay_champion()
        if champion is None:
            raise ValueError("Registered champion required")
        loaded = load_checkpoint(champion.checkpoint, map_location="cpu")
        if loaded.metadata.get("value_target_spec") != SPEC:
            raise ValueError("Champion has incompatible value targets")
        print("Protecting historical training, validation and evaluation inputs...", flush=True)
        blocked, training, files = protected_inputs(corpus, suite)
        protected_file_snapshot = dict(files)
        # Live online evaluation journals may continue growing. The frozen
        # position-key snapshot protects what existed at reservation; bind
        # immutable training inputs below rather than mutable game logs.
        files = {
            str(corpus / "dataset.json"): sha256_file(corpus / "dataset.json"),
            str(corpus / "catalog.sqlite"): sha256_file(corpus / "catalog.sqlite"),
            str(suite): sha256_file(suite),
        }
        # Both categories are excluded from new puzzles, including new validation.
        print(f"Protected keys: {len(blocked)}; historical train keys: {len(training)}", flush=True)
        rows, rejected = choose_puzzles(
            candidates(snapshot, seed, count), blocked, training, count=count, seed=seed
        )
        corpus_manifest = json.loads((corpus / "dataset.json").read_text(encoding="utf-8"))
        for path in source_shards(corpus / "dataset.json"):
            files[str(path)] = sha256_file(path)
        teacher = load_config(corpus / "configs/corpus.yaml")["full_corpus"]["stockfish"]
        teacher.update(multipv=1, cache_dir=None, mate_distance_temperature=None)
        teacher_config = StockfishConfig.from_mapping(teacher)
        files.update(
            {
                str(champion.checkpoint): sha256_file(champion.checkpoint),
                "checkpoints/champion.json": sha256_file(Path("checkpoints/champion.json")),
                str(teacher_config.executable): sha256_file(teacher_config.executable),
                str(snapshot / "download_manifest.json"): sha256_file(
                    snapshot / "download_manifest.json"
                ),
            }
        )
        for item in json.loads((snapshot / "download_manifest.json").read_text())["files"]:
            files[str(snapshot / item["path"])] = item["sha256"]
        root.mkdir(parents=True)
        signed(
            root / "selection.json",
            {"format": FORMAT + ".selection", "version": 1, "seed": seed, "rows": rows},
        )
        summary = {
            "selected_puzzles": len(rows),
            "puzzle_splits": dict(Counter(r["split"] for r in rows)),
            "existing_corpus_splits": corpus_manifest["splits"],
            "rejected": rejected,
            "themes": dict(Counter(theme for row in rows for theme in row["themes"])),
            "rating_range": [min(r["rating"] for r in rows), max(r["rating"] for r in rows)],
            "training_updates": 0,
            "promotion_performed": False,
        }
        template = load_config(corpus / "configs/ready_scratch.yaml")
        plan = {
            "format": FORMAT,
            "version": 1,
            "seed": seed,
            "selected": count,
            "snapshot": str(snapshot),
            "corpus": str(corpus),
            "suite": str(suite),
            "champion": str(champion.checkpoint),
            "stockfish": teacher,
            "model": loaded.model.config.to_dict(),
            "training_template": template,
            "selection_sha256": sha256_file(root / "selection.json"),
            "frozen_files": files,
            "summary": summary,
            "protected_files_snapshot": protected_file_snapshot,
            "protected_keys_sha256": signature({"keys": sorted(blocked)}),
            "historical_training_keys_sha256": signature({"keys": sorted(training)}),
        }
        signed(root / "plan.json", plan)
        atomic_json(root / "preparation.json", summary)
        return summary


def label(root: Path, maximum: int | None = None) -> dict[str, Any]:
    if maximum is not None and maximum < 1:
        raise ValueError("max-positions must be positive")
    with run_lock(root):
        plan = _verified(root)
        rows = [r for r in checked(root / "selection.json")["rows"] if r["split"] != "confirmation"]
        started = time.perf_counter()
        completed = 0
        with (
            closing(_connect(root)) as db,
            StockfishTeacher(StockfishConfig.from_mapping(plan["stockfish"])) as teacher,
        ):
            binding = json.dumps(
                {"plan": sha256_file(root / "plan.json"), "teacher": teacher.provenance},
                sort_keys=True,
            )
            saved = db.execute("SELECT payload FROM bindings WHERE name='teacher'").fetchone()
            if saved and saved[0] != binding:
                raise ValueError("Puzzle teacher provenance changed")
            db.execute("INSERT OR IGNORE INTO bindings VALUES ('teacher',?)", (binding,))
            db.commit()
            for row in rows:
                if db.execute("SELECT 1 FROM labels WHERE key=?", (row["key"],)).fetchone():
                    continue
                if maximum is not None and completed >= maximum:
                    break
                _, audit = teacher.analyze(chess.Board(row["fen"]))
                value = float(audit["stockfish_evaluation"])
                if not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("Invalid Stockfish value label")
                payload = {"value": value, "audit": audit}
                with db:
                    db.execute(
                        "INSERT INTO labels VALUES (?,?)",
                        (
                            row["key"],
                            json.dumps({"payload": payload, "checksum": signature(payload)}),
                        ),
                    )
                completed += 1
                if completed % 100 == 0:
                    remaining = len(rows) - db.execute("SELECT count(*) FROM labels").fetchone()[0]
                    eta = (time.perf_counter() - started) / completed * remaining / 3600
                    print(
                        f"New puzzle value labels {completed}; remaining {remaining}; ETA {eta:.2f} hours",
                        flush=True,
                    )
            total = db.execute("SELECT count(*) FROM labels").fetchone()[0]
        seconds = time.perf_counter() - started
        result = {
            "labeled_this_invocation": completed,
            "labeled_total": total,
            "required": len(rows),
            "remaining": len(rows) - total,
            "seconds": seconds,
            "estimated_remaining_hours": seconds / completed * (len(rows) - total) / 3600
            if completed
            else None,
            "training_updates": 0,
        }
        atomic_json(root / "label_progress.json", result)
        return result


def puzzle_example(row: Mapping[str, Any], payload: Mapping[str, Any]) -> TrainingExample:
    board = chess.Board(row["fen"])
    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
    for uci in row["targets"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("Illegal puzzle target")
        policy[MoveEncoder().encode(move)] = 1 / len(row["targets"])
    return TrainingExample(
        BoardEncoder().encode(board),
        policy,
        payload["value"],
        {
            "fen": row["fen"],
            "position_key": row["key"],
            "split": row["split"],
            "group_id": row["group_id"],
            "game_id": row["group_id"],
            "origin": "lichess_puzzle",
            "puzzle_id": row["id"],
            "puzzle_rating": row["rating"],
            "themes": row["themes"],
            "policy_source": "lichess_puzzle_solver_root",
            "value_target_kind": STOCKFISH_VALUE_KIND,
            "teacher_audit": payload["audit"],
        },
    )


def _manifest(path: Path, shards: list[tuple[Path, int]], splits: Mapping[str, int]) -> None:
    entries = []
    for shard, count in shards:
        relative = shard.resolve().relative_to(path.parent.resolve()).as_posix()
        entries.append({"path": relative, "examples": count, "sha256": sha256_file(shard)})
    atomic_json(
        path,
        {
            "format": SHARDED_DATASET_FORMAT,
            "version": 1,
            "value_target_spec": SPEC,
            "examples": sum(splits.values()),
            "splits": dict(splits),
            "shards": entries,
            "confirmation_exported": False,
        },
    )


def finalize(root: Path) -> dict[str, Any]:
    with run_lock(root):
        plan = _verified(root)
        if (root / "training_ready.json").exists():
            return verify_ready(root)
        rows = [r for r in checked(root / "selection.json")["rows"] if r["split"] != "confirmation"]
        shards = []
        with closing(_connect(root)) as db:
            binding = db.execute("SELECT payload FROM bindings WHERE name='teacher'").fetchone()
            if not binding or json.loads(binding[0]).get("plan") != sha256_file(root / "plan.json"):
                raise ValueError("Puzzle labels are not bound to this experiment")
            if db.execute("SELECT count(*) FROM labels").fetchone()[0] != len(rows):
                raise ValueError("Finish all puzzle value labels before finalizing")
            for index in range(0, len(rows), 2048):
                batch = rows[index : index + 2048]
                examples = []
                for row in batch:
                    saved = db.execute(
                        "SELECT payload FROM labels WHERE key=?", (row["key"],)
                    ).fetchone()
                    if saved is None:
                        raise ValueError("Missing puzzle value label")
                    raw = json.loads(saved[0])
                    if signature(raw["payload"]) != raw["checksum"]:
                        raise ValueError("Puzzle label checksum mismatch")
                    examples.append(puzzle_example(row, raw["payload"]))
                path = root / "shards" / f"puzzles_{index // 2048:06d}.pt"
                # Partial finalization regenerates only new-run outputs; old corpus is never written.
                save_dataset(path, examples, metadata={"value_target_spec": SPEC}, format_version=2)
                shards.append((path, len(examples)))
                print(f"Puzzle shards exported: {len(shards)}", flush=True)
        splits = dict(Counter(r["split"] for r in rows))
        puzzle_path = root / "puzzles.json"
        _manifest(puzzle_path, shards, splits)
        corpus = Path(plan["corpus"]) / "dataset.json"
        old = json.loads(corpus.read_text(encoding="utf-8"))
        old_shards = [(corpus.parent / entry["path"], entry["examples"]) for entry in old["shards"]]
        combined_path = root.parent / f"{root.name}_combined.json"
        combined_splits = {split: old["splits"][split] + splits[split] for split in splits}
        _manifest(combined_path, old_shards + shards, combined_splits)
        configs = {}
        for arm in ("scratch", "finetune"):
            config = copy.deepcopy(plan["training_template"])
            config.update(seed=plan["seed"], device="cuda", model=plan["model"])
            config["training"].update(
                dataset_path=str(combined_path if arm == "scratch" else puzzle_path),
                checkpoint_dir=str(Path("checkpoints") / f"{root.name}_{arm}"),
                metrics_path=str(Path("data/metrics") / f"{root.name}_{arm}.jsonl"),
                epochs=100 if arm == "scratch" else 40,
                learning_rate=0.001 if arm == "scratch" else 0.00001,
                early_stopping_patience=0 if arm == "scratch" else 10,
                checkpoint_every=5 if arm == "scratch" else 2,
                freeze_batch_norm_statistics=arm == "finetune",
                baseline_eligible=arm == "finetune",
                source_sampling_multipliers={},
                game_balanced_sampling=False,
                validation_source_key="origin",
                use_dataset_split=True,
                value_target_spec=SPEC,
            )
            TrainingConfig.from_mapping(config["training"], seed=plan["seed"], device="cuda")
            path = root / "configs" / f"train_{arm}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            configs[arm] = str(path)
        outputs = [
            puzzle_path,
            combined_path,
            *map(Path, configs.values()),
            *[p for p, _ in shards],
        ]
        receipt = {
            "format": FORMAT + ".ready",
            "version": 1,
            "plan_sha256": sha256_file(root / "plan.json"),
            "configs": configs,
            "outputs": {str(path): sha256_file(path) for path in outputs},
            "combined_splits": combined_splits,
            "puzzle_splits": splits,
            "training_updates": 0,
            "promotion_performed": False,
        }
        signed(root / "training_ready.json", receipt)
        return receipt


def verify_ready(root: Path) -> dict[str, Any]:
    _verified(root)
    receipt = checked(root / "training_ready.json")
    if receipt["plan_sha256"] != sha256_file(root / "plan.json"):
        raise ValueError("Puzzle readiness belongs to a different plan")
    for path, digest in receipt["outputs"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Ready puzzle output changed: {path}")
    return receipt


def train(root: Path, arm: str, *, resume: bool = False, device: str = "cuda") -> None:
    if arm not in {"scratch", "finetune"}:
        raise ValueError("Expected scratch or finetune")
    with run_lock(root):
        receipt = verify_ready(root)
        config = Path(receipt["configs"][arm])
        checkpoint_dir = Path(load_config(config)["training"]["checkpoint_dir"])
        command = [
            sys.executable,
            "-m",
            "chess_ai",
            "train",
            "--config",
            str(config),
            "--device",
            device,
        ]
        if resume:
            last = checkpoint_dir / "last.pt"
            if not last.is_file():
                raise ValueError("No last.pt to resume for this arm")
            command += ["--resume", str(last)]
        else:
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise ValueError("Arm has checkpoints already; use --resume")
            if arm == "finetune":
                command += ["--init-checkpoint", checked(root / "plan.json")["champion"]]
        subprocess.run(command, check=True)


def evaluate(
    root: Path,
    checkpoint: Path,
    *,
    limit: int = 1000,
    device: str = "cuda",
    split: str = "validation",
) -> dict[str, Any]:
    """Resumable first-solver-move benchmark, not a whole-line solve-rate claim."""
    if split not in {"validation", "confirmation"} or limit < 1:
        raise ValueError("Use validation/confirmation and a positive limit")
    plan = _verified(root)
    rows = [r for r in checked(root / "selection.json")["rows"] if r["split"] == split][:limit]
    if not rows:
        raise ValueError("Empty puzzle evaluation split")
    digest = sha256_file(checkpoint)
    output = root / "evaluation" / f"{split}_{len(rows)}_{digest[:16]}"
    binding = {
        "format": FORMAT + ".evaluation",
        "version": 1,
        "selection_sha256": plan["selection_sha256"],
        "checkpoint_sha256": digest,
        "split": split,
        "ids": [r["id"] for r in rows],
        "search_simulations": 32,
        "c_puct": 1.5,
        "seed": plan["seed"],
    }
    with run_lock(output):
        output.mkdir(parents=True, exist_ok=True)
        saved = output / "progress.json"
        if saved.exists():
            previous = checked(saved)
            if previous["binding"] != binding:
                raise ValueError("Puzzle benchmark binding changed")
            predictions = previous["predictions"]
        else:
            predictions = {}
        agent = NeuralMCTSAgent(
            checkpoint, device=device, seed=plan["seed"], simulations=32, c_puct=1.5
        )
        for row in rows:
            if row["id"] in predictions:
                continue
            move = agent.choose_move(chess.Board(row["fen"]))
            predictions[row["id"]] = {"move": move.uci(), "correct": move.uci() in row["targets"]}
            if len(predictions) % 25 == 0 or len(predictions) == len(rows):
                signed(saved, {"binding": binding, "predictions": predictions})
                print(f"Puzzle first-move audit: {len(predictions)}/{len(rows)}", flush=True)
        correct = sum(predictions[row["id"]]["correct"] for row in rows)
        result = {
            **binding,
            "positions": len(rows),
            "correct_first_moves": correct,
            "first_move_accuracy": correct / len(rows),
            "complete": True,
            "full_puzzle_solve_rate_measured": False,
            "training_updates": 0,
            "report": str(output / "report.json"),
        }
        atomic_json(output / "report.json", result)
        return result
