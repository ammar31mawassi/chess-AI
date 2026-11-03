"""Resumable relabeling of explicitly listed training games and positions.

SQLite is the inventory and analysis journal. Labels are committed per position;
the final manifest references bounded tensor shards instead of stacking the corpus.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import torch

from chess_ai.data.examples import DATASET_FORMAT, TrainingExample, load_dataset, save_dataset
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.data.value_targets import SHARDED_DATASET_FORMAT, STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature

REBUILD_FORMAT = "self-improving-chess-ai.stockfish-rebuild"
BLOCKED_PARTS = {
    "evaluation",
    "standard_evaluation",
    "external",
    "imported_external",
    "gate",
    "gates",
    "holdout",
    "benchmark",
    "benchmarks",
    "standard_regression",
}


@dataclass(frozen=True)
class RebuildConfig:
    run_dir: Path
    datasets: tuple[Path, ...]
    pgn_roots: tuple[Path, ...]
    stockfish: StockfishConfig
    seed: int = 21102036
    validation_fraction: float = 0.1
    shard_size: int = 2048
    pilot_positions: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "datasets", tuple(Path(p) for p in self.datasets))
        object.__setattr__(self, "pgn_roots", tuple(Path(p) for p in self.pgn_roots))
        if not (self.datasets or self.pgn_roots) or not 0 < self.validation_fraction < 1:
            raise ValueError(
                "Rebuild requires datasets or PGN roots and a validation fraction between zero and one"
            )
        if self.shard_size <= 0 or self.pilot_positions < 2:
            raise ValueError("shard_size must be positive and pilot_positions at least two")
        if not self.stockfish.store_evaluation:
            raise ValueError("Stockfish rebuild requires store_evaluation=true")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RebuildConfig:
        settings = dict(raw)
        settings["stockfish"] = StockfishConfig.from_mapping(settings["stockfish"])
        return cls(**settings)

    @property
    def value_spec(self) -> dict[str, Any]:
        return {
            "kind": STOCKFISH_VALUE_KIND,
            "perspective": "side_to_move",
            "scale_cp": self.stockfish.evaluation_scale_cp,
        }

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "datasets": [str(p.resolve()) for p in self.datasets],
            "pgn_roots": [str(p.resolve()) for p in self.pgn_roots],
            "stockfish": self.stockfish.analysis_settings(),
            "binary_sha256": sha256_file(self.stockfish.executable),
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "shard_size": self.shard_size,
            "pilot_positions": self.pilot_positions,
            "value_target_spec": self.value_spec,
            "position_identity": "network_visible_position_fen_counters_reset_v1",
            "split": "game_group_hash_validation_wins_shared_position_v1",
        }


def _blocked(path: Path) -> bool:
    return any(part.lower() in BLOCKED_PARTS or "audit" in part.lower() for part in path.parts)


def _canonical(board: chess.Board) -> tuple[str, str]:
    if not board.is_valid():
        raise ValueError(f"Invalid training position: {board.fen()}")
    # These counters and game history are absent from the 18-plane network input.
    # One deterministic standalone label avoids conflicting labels for identical inputs.
    fields = board.fen(en_passant="fen").split()
    fen = " ".join([*fields[:4], "0", "1"])
    key = hashlib.sha256(BoardEncoder().encode(chess.Board(fen)).tobytes()).hexdigest()
    return key, fen


def _board_from_tensor(tensor: torch.Tensor) -> chess.Board:
    array = tensor.numpy()
    if array.shape != (18, 8, 8) or not np.isin(array, [0, 1]).all():
        raise ValueError("Cannot reconstruct a legacy position from non-binary board planes")
    board = chess.Board(None)
    for plane in range(12):
        for row, column in np.argwhere(array[plane]):
            square = chess.square(int(column), 7 - int(row))
            if board.piece_at(square) is not None:
                raise ValueError("Legacy board planes overlap pieces")
            board.set_piece_at(square, chess.Piece(plane % 6 + 1, plane < 6))
    for plane in range(12, 17):
        if not np.all(array[plane] == array[plane, 0, 0]):
            raise ValueError("Legacy global board plane is not constant")
    board.turn = bool(array[12, 0, 0])
    board.castling_rights = chess.BB_EMPTY
    for plane, square in ((13, chess.H1), (14, chess.A1), (15, chess.H8), (16, chess.A8)):
        if array[plane, 0, 0]:
            board.castling_rights |= chess.BB_SQUARES[square]
    ep = np.argwhere(array[17])
    if len(ep) > 1:
        raise ValueError("Legacy en-passant plane has multiple targets")
    board.ep_square = chess.square(int(ep[0, 1]), 7 - int(ep[0, 0])) if len(ep) else None
    _canonical(board)
    if not np.array_equal(BoardEncoder().encode(board), array):
        raise ValueError("Legacy board reconstruction failed its round-trip check")
    return board


def _split(config: RebuildConfig, group: str) -> str:
    number = int(hashlib.sha256(f"{config.seed}:{group}".encode()).hexdigest(), 16) / 2**256
    return "validation" if number < config.validation_fraction else "train"


def _insert(
    db: sqlite3.Connection,
    config: RebuildConfig,
    board: chess.Board,
    group: str,
    source: str,
    game_id: str,
) -> bool:
    if board.is_game_over(claim_draw=True):
        return False
    key, fen = _canonical(board)
    split = _split(config, group)
    # A repeated position can appear in distinct games. Validation wins collisions:
    # remove that input from training without splitting a game's own examples.
    db.execute(
        "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, NULL) "
        "ON CONFLICT(key) DO UPDATE SET split=excluded.split, group_id=excluded.group_id, "
        "source=excluded.source, game_id=excluded.game_id WHERE positions.split='train' AND excluded.split='validation'",
        (key, fen, "white" if board.turn else "black", split, group, source, game_id),
    )
    return True


def _raw_dataset(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if raw.get("format") != DATASET_FORMAT or raw.get("format_version") != 1:
        raise ValueError(f"Inventory accepts existing version-1 training sources only: {path}")
    if (
        raw.get("example_count") != len(raw.get("example_metadata_json", []))
        or len(raw["boards"]) != raw["example_count"]
    ):
        raise ValueError(f"Malformed source dataset: {path}")
    return raw


def _game_positions(game: chess.pgn.Game) -> Iterator[chess.Board]:
    if game.errors or not game.board().is_valid():
        raise ValueError("Training PGN contains parser errors or invalid initial position")
    board = game.board()
    if type(board) is not chess.Board or board.chess960:
        raise ValueError("Training PGN must use standard chess, not a variant")
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise ValueError("Training PGN contains an illegal move")
        yield board.copy(stack=True)
        board.push(move)


def _game_digest(game: chess.pgn.Game) -> str:
    return signature({"fen": game.board().fen(), "moves": [m.uci() for m in game.mainline_moves()]})


def inventory(config: RebuildConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.run_dir / "inventory.json"
    if report_path.exists():
        return _validate_inventory(config)
    catalog = config.run_dir / "catalog.sqlite"
    if catalog.exists():
        raise ValueError(
            "Incomplete inventory exists. Reserve a new run ID; it cannot be treated as complete."
        )
    with run_lock(config.run_dir), closing(sqlite3.connect(catalog)) as db:
        db.execute(
            "CREATE TABLE positions(key TEXT PRIMARY KEY, fen TEXT, color TEXT, split TEXT, group_id TEXT, source TEXT, game_id TEXT, label TEXT)"
        )
        db.execute("CREATE INDEX unlabeled ON positions(key) WHERE label IS NULL")
        records: list[dict[str, Any]] = []
        game_groups: dict[str, str] = {}
        csv_rows: dict[Path, set[int]] = {}
        counters: Counter[str] = Counter()
        for path in config.datasets:
            if _blocked(path):
                raise ValueError(
                    f"Evaluation/benchmark source is not authorized for this inventory: {path}"
                )
            raw = _raw_dataset(path)
            file_meta = raw.get("metadata", {})
            record: dict[str, Any] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": raw["example_count"],
                "kind": "dataset",
            }
            if file_meta.get("kind") == "explicit_weighted_dataset_composition":
                available = {p.resolve() for p in config.datasets}
                missing = [
                    entry["path"]
                    for entry in file_meta["sources"]
                    if Path(entry["path"]).resolve() not in available
                ]
                if missing:
                    raise ValueError(
                        f"Composed source has original datasets absent from inventory: {missing}"
                    )
                record["status"] = "composed_copy_skipped"
                records.append(record)
                continue
            source = (
                path.stem
                if path.parent.name == "datasets"
                else path.parts[path.parts.index("curricula") + 1]
                if "curricula" in path.parts
                else path.parent.name
            )
            reconstructed = 0
            for i, encoded in enumerate(raw["example_metadata_json"]):
                meta = json.loads(encoded)
                if (
                    str(meta.get("TrainingData", meta.get("training_data", "true"))).lower()
                    == "false"
                ):
                    raise ValueError(f"Evaluation row found in training dataset {path}")
                game_id = str(meta.get("game_id", f"{source}-position-{i}"))
                group = str(meta.get("opening_set_id", meta.get("opening_pair_id", game_id)))
                # Keep paired games together. Shared GameID connects archived PGNs to dataset groups.
                group = game_groups.setdefault(game_id, group)
                if meta.get("source") == "arevel/chess-games" and "source_row" in meta:
                    csv_path = Path(file_meta["config"]["source_path"])
                    csv_rows.setdefault(csv_path, set()).add(int(meta["source_row"]))
                if meta.get("fen"):
                    board = chess.Board(meta["fen"])
                    if not np.array_equal(BoardEncoder().encode(board), raw["boards"][i].numpy()):
                        if not np.array_equal(
                            BoardEncoder().encode(board)[:17], raw["boards"][i].numpy()[:17]
                        ):
                            raise ValueError(f"FEN/board tensor mismatch in {path} row {i}")
                        board = _board_from_tensor(raw["boards"][i])
                        reconstructed += 1
                else:
                    board = _board_from_tensor(raw["boards"][i])
                    reconstructed += 1
                counters["dataset_positions"] += int(
                    _insert(db, config, board, group, source, game_id)
                )
            record.update(status="included", reconstructed_positions=reconstructed)
            records.append(record)
            db.commit()
            print(f"Inventoried {path.name}: {raw['example_count']} rows", flush=True)
            del raw
        seen_games: set[str] = set()
        for root in config.pgn_roots:
            if not root.is_dir() or _blocked(root):
                raise ValueError(f"Missing or prohibited training PGN directory: {root}")
            for path in sorted(root.rglob("*.pgn")):
                if _blocked(path):
                    counters["excluded_evaluation_pgn_files"] += 1
                    continue
                included = 0
                with path.open(encoding="utf-8-sig") as handle:
                    while (game := chess.pgn.read_game(handle)) is not None:
                        if str(
                            game.headers.get("TrainingData", "true")
                        ).lower() == "false" or game.headers.get("EvaluationKind"):
                            counters["excluded_evaluation_games"] += 1
                            continue
                        saved_id = game.headers.get("GameID", game.headers.get("GameId"))
                        if (
                            game.headers.get("DataOrigin") == "HumanGUI"
                            and saved_id not in game_groups
                        ):
                            counters["excluded_unconfirmed_human_games"] += 1
                            continue
                        if game.errors:
                            raise ValueError("Training PGN contains parser errors")
                        if not game.variations:
                            counters["empty_pgn_games"] += 1
                            continue
                        digest = _game_digest(game)
                        if digest in seen_games:
                            counters["duplicate_games"] += 1
                            continue
                        seen_games.add(digest)
                        game_id = saved_id or f"pgn-{digest}"
                        group = game_groups.get(game_id, game_id)
                        for board in _game_positions(game):
                            counters["pgn_positions"] += int(
                                _insert(db, config, board, group, root.name, game_id)
                            )
                        included += 1
                records.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "kind": "pgn",
                        "games": included,
                    }
                )
            db.commit()
        csv.field_size_limit(16 * 1024 * 1024)
        for path, wanted in csv_rows.items():
            found: set[int] = set()
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row_index, row in enumerate(csv.DictReader(handle), 1):
                    if row_index not in wanted:
                        continue
                    game = chess.pgn.read_game(io.StringIO(row["AN"]))
                    if game is None:
                        raise ValueError("Previously imported CSV game is missing its moves")
                    game_id = f"kaggle-lichess-{row_index}"
                    for board in _game_positions(game):
                        counters["csv_positions"] += int(
                            _insert(db, config, board, game_groups[game_id], path.stem, game_id)
                        )
                    found.add(row_index)
                    if found == wanted:
                        break
            if found != wanted:
                raise ValueError("Could not recover every previously imported CSV game")
            records.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "kind": "csv_existing_games_only",
                    "games": len(found),
                }
            )
            db.commit()
        count = db.execute("SELECT count(*) FROM positions").fetchone()[0]
        splits = dict(db.execute("SELECT split,count(*) FROM positions GROUP BY split"))
        if not count or set(splits) != {"train", "validation"}:
            raise ValueError("Inventory needs nonempty training and validation splits")
        # All inputs, including intermediate positions, are excluded from future random starts.
        pairs = [
            {"opening_fen": row[0]} for row in db.execute("SELECT fen FROM positions ORDER BY key")
        ]
        atomic_json(
            config.run_dir / "training_manifest.json", {"pairs": pairs, "training_data": True}
        )
        report = {
            "format": REBUILD_FORMAT,
            "version": 1,
            "binding": config.binding,
            "inputs": records,
            "unique_positions": count,
            "splits": splits,
            "colors": dict(db.execute("SELECT color,count(*) FROM positions GROUP BY color")),
            "counts": dict(counters),
            "complete_pgn_games": len(seen_games),
            "training_updates": 0,
            "limitations": [
                "Legacy tensor-only games provide stored positions, not missing moves or history.",
                "Analysis uses standalone positions with counters reset to match the current network input.",
                "Exact duplicate inputs are removed; validation owns cross-split duplicates.",
            ],
        }
        report["catalog_signature"] = signature(
            {"binding": report["binding"], "inputs": records, "positions": pairs}
        )
        atomic_json(report_path, report)
        return report


def _validate_inventory(config: RebuildConfig) -> dict[str, Any]:
    report = json.loads((config.run_dir / "inventory.json").read_text(encoding="utf-8"))
    if (
        report.get("format") != REBUILD_FORMAT
        or report.get("version") != 1
        or report["binding"] != config.binding
    ):
        raise ValueError(
            "Rebuild configuration or Stockfish binary differs from the frozen inventory"
        )
    for record in report["inputs"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Source changed since inventory: {record['path']}")
    return report


def _connect(config: RebuildConfig) -> sqlite3.Connection:
    path = config.run_dir / "catalog.sqlite"
    if not path.is_file():
        raise ValueError("Run the inventory stage first")
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


def _label_row(teacher: StockfishTeacher, row: sqlite3.Row) -> dict[str, Any]:
    board = chess.Board(row["fen"])
    policy, audit = teacher.analyze(board)
    value = float(audit["stockfish_evaluation"])
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Stockfish supplied an invalid evaluation target")
    payload = {
        "policy": [[int(i), float(policy[i])] for i in np.flatnonzero(policy)],
        "value": value,
        "audit": audit,
    }
    payload["checksum"] = signature(payload)
    return payload


def pilot(config: RebuildConfig) -> dict[str, Any]:
    inventory_report = _validate_inventory(config)
    output = config.run_dir / "pilot.json"
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["catalog_signature"] != inventory_report["catalog_signature"]:
            raise ValueError("Pilot belongs to a different inventory")
        return saved
    with (
        run_lock(config.run_dir),
        closing(_connect(config)) as db,
        StockfishTeacher(config.stockfish) as teacher,
    ):
        rows: list[sqlite3.Row] = []
        for color in ("white", "black"):
            rows.extend(
                db.execute(
                    "SELECT * FROM positions WHERE color=? ORDER BY key LIMIT ?",
                    (color, config.pilot_positions // 2),
                )
            )
        if len(rows) < 2 or len({r["color"] for r in rows}) != 2:
            raise ValueError("Pilot requires positions for both White and Black")
        started = time.perf_counter()
        samples = []
        for index, row in enumerate(rows, 1):
            label = _label_row(teacher, row)
            db.execute(
                "UPDATE positions SET label=? WHERE key=?",
                (json.dumps(label, allow_nan=False), row["key"]),
            )
            db.commit()
            samples.append(
                {
                    "fen": row["fen"],
                    "color": row["color"],
                    "value": label["value"],
                    "best_move": label["audit"]["teacher_move_uci"],
                    "top_moves": label["audit"]["top_moves"],
                }
            )
            if index % 16 == 0:
                print(f"Pilot {index}/{len(rows)} positions", flush=True)
        seconds = time.perf_counter() - started
        report = {
            "catalog_signature": inventory_report["catalog_signature"],
            "positions": len(rows),
            "seconds": seconds,
            "estimated_serial_label_hours": seconds
            / len(rows)
            * inventory_report["unique_positions"]
            / 3600,
            "samples": samples,
            "teacher": teacher.provenance,
            "training_updates": 0,
        }
        atomic_json(output, report)
        return report


def label_positions(config: RebuildConfig, maximum: int | None = None) -> dict[str, Any]:
    inventory_report = _validate_inventory(config)
    if maximum is not None and maximum <= 0:
        raise ValueError("max-positions must be positive")
    with (
        run_lock(config.run_dir),
        closing(_connect(config)) as db,
        StockfishTeacher(config.stockfish) as teacher,
    ):
        count = 0
        started = time.perf_counter()
        while maximum is None or count < maximum:
            row = db.execute(
                "SELECT * FROM positions WHERE label IS NULL ORDER BY key LIMIT 1"
            ).fetchone()
            if row is None:
                break
            label = _label_row(teacher, row)
            db.execute(
                "UPDATE positions SET label=? WHERE key=?",
                (json.dumps(label, allow_nan=False), row["key"]),
            )
            db.commit()
            count += 1
            if count % 100 == 0:
                remaining = db.execute(
                    "SELECT count(*) FROM positions WHERE label IS NULL"
                ).fetchone()[0]
                print(f"Labeled {count} this invocation; {remaining} remaining", flush=True)
        remaining = db.execute("SELECT count(*) FROM positions WHERE label IS NULL").fetchone()[0]
        report = {
            "catalog_signature": inventory_report["catalog_signature"],
            "labeled_this_invocation": count,
            "remaining": remaining,
            "seconds": time.perf_counter() - started,
            "complete": remaining == 0,
            "training_updates": 0,
        }
        atomic_json(config.run_dir / "label_progress.json", report)
        return report


def finalize(config: RebuildConfig) -> dict[str, Any]:
    inventory_report = _validate_inventory(config)
    output = config.run_dir / "dataset.json"
    with run_lock(config.run_dir), closing(_connect(config)) as db:
        if db.execute("SELECT count(*) FROM positions WHERE label IS NULL").fetchone()[0]:
            raise ValueError("Finish labeling every inventoried position before finalizing")
        if output.exists():
            report = json.loads(output.read_text(encoding="utf-8"))
            if report["catalog_signature"] != inventory_report["catalog_signature"]:
                raise ValueError("Finalized dataset belongs to a different inventory")
            for entry in report["shards"]:
                if sha256_file(config.run_dir / entry["path"]) != entry["sha256"]:
                    raise ValueError("Finalized shard hash mismatch")
            return report
        shards: list[dict[str, Any]] = []
        cursor = db.execute("SELECT * FROM positions ORDER BY key")
        while rows := cursor.fetchmany(config.shard_size):
            path = config.run_dir / "shards" / f"positions_{len(shards):06d}.pt"
            binding = {
                "value_target_spec": config.value_spec,
                "catalog_signature": inventory_report["catalog_signature"],
                "position_keys": [row["key"] for row in rows],
            }
            if path.exists():
                existing = load_dataset(path)
                if existing.metadata != binding or len(existing) != len(rows):
                    raise ValueError(f"Existing shard is incompatible: {path}")
            else:
                examples = []
                for row in rows:
                    label = json.loads(row["label"])
                    checksum = label.pop("checksum")
                    if signature(label) != checksum:
                        raise ValueError("Committed position label checksum mismatch")
                    if (
                        _canonical(chess.Board(row["fen"]))[0] != row["key"]
                        or _split(config, row["group_id"]) != row["split"]
                    ):
                        raise ValueError("Position inventory key or split was modified")
                    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
                    for action, probability in label["policy"]:
                        policy[action] = probability
                    examples.append(
                        TrainingExample(
                            BoardEncoder().encode(chess.Board(row["fen"])),
                            policy,
                            label["value"],
                            {
                                **label["audit"],
                                "fen": row["fen"],
                                "game_id": row["game_id"],
                                "group_id": row["group_id"],
                                "composition_source": row["source"],
                                "position_key": row["key"],
                                "split": row["split"],
                                "value_target_kind": STOCKFISH_VALUE_KIND,
                                "player_to_move": row["color"],
                            },
                        )
                    )
                save_dataset(path, examples, metadata=binding, format_version=2)
            shards.append(
                {
                    "path": path.relative_to(config.run_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "examples": len(rows),
                }
            )
            print(f"Finalized shard {len(shards)} ({len(rows)} positions)", flush=True)
        report = {
            "format": SHARDED_DATASET_FORMAT,
            "version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "value_target_spec": config.value_spec,
            "catalog_signature": inventory_report["catalog_signature"],
            "examples": inventory_report["unique_positions"],
            "shards": shards,
            "splits": inventory_report["splits"],
            "training_updates": 0,
        }
        atomic_json(output, report)
        return report


def run_rebuild(config: RebuildConfig, stage: str, maximum: int | None = None) -> dict[str, Any]:
    if stage == "inventory":
        return inventory(config)
    if stage == "pilot":
        return pilot(config)
    if stage == "labels":
        return label_positions(config, maximum)
    if stage == "finalize":
        return finalize(config)
    raise ValueError(f"Unsupported rebuild stage: {stage}")
