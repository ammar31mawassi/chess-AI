"""An explicit, resumable union of compatible labels and full GM mainlines.

Old tensors are referenced, not copied into RAM during inventory. Validation owns
shared inputs; confirmation and evaluation inputs veto both exported splits.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import torch
import yaml

from chess_ai.config import load_config
from chess_ai.data.endgame_pilot import _checked_json, _file_record, load_prepared, orbit_keys
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.gm_endgames import _source_games
from chess_ai.data.stockfish_rebuild import _blocked, _board_from_tensor, _canonical, _label_row
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.data.value_targets import SHARDED_DATASET_FORMAT, STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import (
    atomic_json,
    replace_with_retry,
    run_lock,
    sha256_file,
    signature,
)
from chess_ai.training.trainer import TrainingConfig

FORMAT = "self-improving-chess-ai.full-stockfish-corpus"
SPEC = {"kind": STOCKFISH_VALUE_KIND, "perspective": "side_to_move", "scale_cp": 600.0}


def source_shards(path: Path) -> Iterator[Path]:
    if path.suffix != ".json":
        yield path.resolve()
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != SHARDED_DATASET_FORMAT
        or manifest.get("version") != 1
        or manifest.get("value_target_spec") != SPEC
    ):
        raise ValueError("Full corpus requires compatible Stockfish shard manifests")
    for entry in manifest["shards"]:
        shard = (path.parent / entry["path"]).resolve()
        if not shard.is_relative_to(path.parent.resolve()) or sha256_file(shard) != entry["sha256"]:
            raise ValueError("Source shard path/hash mismatch")
        yield shard


def _freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("format") != FORMAT or config.get("version") != 1:
        raise ValueError("Incompatible full-corpus configuration")
    root = Path(config["run_dir"])
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = _checked_json(plan_path)
        if plan["config"] != dict(config):
            raise ValueError("Full-corpus config changed; reserve another run")
    else:
        if (root / "catalog.sqlite").exists():
            raise ValueError(
                "Corpus catalog exists without its frozen plan; preserve it and use a new run"
            )
        files = {Path(config[k]).resolve() for k in ("champion", "champion_registry", "coverage")}
        files.add(Path(config["stockfish"]["executable"]).resolve())
        coverage = json.loads(Path(config["coverage"]).read_text(encoding="utf-8"))
        if config.get("collection_record"):
            collection_path = Path(config["collection_record"])
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            if collection.get("version") != 3 or collection["coverage_sha256"] != sha256_file(
                Path(config["coverage"])
            ):
                raise ValueError("Expanded collection coverage binding changed")
            files.add(collection_path.resolve())
        if coverage.get("version") != 3:
            raise ValueError("Use the expanded version-3 GM collection")
        if (
            coverage["selection_ownership"] != config["expected_ownership"]
            or coverage["added_ownership"] != config["expected_additions"]
        ):
            raise ValueError("GM quotas/additions differ from the reserved plan")
        base = Path(coverage["base_coverage"]["path"])
        if sha256_file(base) != coverage["base_coverage"]["sha256"]:
            raise ValueError("Original GM collection binding changed")
        original = json.loads(base.read_text(encoding="utf-8"))
        if sha256_file(Path(original["snapshot"]["path"])) != original["snapshot"]["sha256"]:
            raise ValueError("Original GM snapshot changed")
        if not set(original["selected_game_digests"]) <= set(coverage["selected_game_digests"]):
            raise ValueError("Expansion lost original GM games")
        if any(
            coverage["selected_game_owners"].get(k) != v
            for k, v in original["selected_game_owners"].items()
        ):
            raise ValueError("Expansion changed original GM ownership")
        files.update(
            (
                base.resolve(),
                Path(original["snapshot"]["path"]).resolve(),
                Path(coverage["snapshot"]["path"]).resolve(),
            )
        )
        if sha256_file(Path(coverage["snapshot"]["path"])) != coverage["snapshot"]["sha256"]:
            raise ValueError("Expanded GM snapshot changed")
        sources = []
        for item in config["sources"]:
            source = Path(item["path"])
            if _blocked(source):
                raise ValueError("Evaluation artifacts cannot be corpus sources")
            files.add(source.resolve())
            for shard in source_shards(source):
                files.add(shard)
                sources.append({"path": str(shard), "origin": item["name"]})
        for pilot in config["protected_pilots"]:
            files.add((Path(pilot) / "positions.json").resolve())
        rollouts = []
        for item in config["rollouts"]:
            for path in sorted(Path(item["folder"]).glob("case_*.json")):
                files.add(path.resolve())
                rollouts.append({"path": str(path.resolve()), "pilot": item["pilot"]})
        evaluation = sorted(
            {
                p.resolve()
                for folder in config["evaluation_roots"]
                for p in Path(folder).rglob("*.pgn")
            }
        )
        if not evaluation:
            raise ValueError("Historical evaluation exclusion cannot be omitted")
        files.update(evaluation)
        plan = {
            "format": FORMAT,
            "version": 1,
            "config": dict(config),
            "sources": sources,
            "rollouts": rollouts,
            "evaluation_files": [str(p) for p in evaluation],
            "inputs": [_file_record(p) for p in sorted(files)],
        }
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(plan_path, {"payload": plan, "sha256": signature(plan)})
    if plan.get("format") != FORMAT or plan.get("version") != 1:
        raise ValueError("Incompatible full-corpus plan")
    for entry in plan["inputs"]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Frozen corpus input changed: {entry['path']}")
    if sha256_file(Path(config["champion"])) != config["champion_sha256"]:
        raise ValueError("Frozen champion changed")
    return plan


def _connect(root: Path) -> sqlite3.Connection:
    db = sqlite3.connect(root / "catalog.sqlite")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE IF NOT EXISTS positions(key TEXT PRIMARY KEY,fen TEXT,split TEXT,group_id TEXT,origin TEXT,ref_path TEXT,ref_index INTEGER,bucket TEXT,label TEXT)"
    )
    db.execute("CREATE TABLE IF NOT EXISTS blocked(key TEXT PRIMARY KEY,kind INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS old_training(key TEXT PRIMARY KEY)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS fen_repairs(ref_path TEXT,ref_index INTEGER,source_fen TEXT,canonical_fen TEXT,PRIMARY KEY(ref_path,ref_index))"
    )
    db.execute("CREATE TABLE IF NOT EXISTS units(name TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE IF NOT EXISTS binding(name TEXT PRIMARY KEY,payload TEXT)")
    db.execute(
        "CREATE INDEX IF NOT EXISTS pending_labels ON positions(key) WHERE ref_path IS NULL AND label IS NULL"
    )
    db.commit()
    return db


def _ban(db: sqlite3.Connection, keys: set[str], kind: int) -> None:
    db.executemany(
        "INSERT INTO blocked VALUES (?,?) ON CONFLICT(key) DO UPDATE SET kind=max(kind,excluded.kind)",
        [(key, kind) for key in keys],
    )


def _bucket(board: chess.Board) -> str:
    pieces = [p.piece_type for p in board.piece_map().values() if p.piece_type != chess.KING]
    if len(pieces) == 1 and pieces[0] in (chess.QUEEN, chess.ROOK):
        return "basic_mate"
    return "endgame" if len(pieces) <= 8 else "general"


def _source_position(
    example: TrainingExample, producer: Mapping[str, Any], path: Path, index: int
) -> tuple[chess.Board, str, str, bool]:
    """Restore only a documented lossy FEN from the v2 offline collector.

    python-chess's default FEN omits an ep target when no legal ep capture exists.
    The collector's tensor/hash retained that square. All other planes, the tensor
    hash and the full legal-move set must agree before that legacy FEN is repaired.
    """
    board = chess.Board(str(example.metadata["fen"]))
    key, fen = _canonical(board)
    encoded = BoardEncoder().encode(board)
    tensor = np.asarray(example.board_tensor)
    stored_key = example.metadata["position_key"]
    if key == stored_key and np.array_equal(encoded, tensor):
        return board, key, fen, False
    context = f"{path}, row {index}"
    if (
        producer.get("format") == "self-improving-chess-ai.stockfish-offline-batch"
        and producer.get("version") == 2
        and board.ep_square is None
        and np.array_equal(encoded[:17], tensor[:17])
        and hashlib.sha256(tensor.tobytes()).hexdigest() == stored_key
    ):
        try:
            recovered = _board_from_tensor(torch.from_numpy(tensor))
        except ValueError as exc:
            raise ValueError(f"Source tensor/FEN/key mismatch at {context}: {exc}") from exc
        restored_key, restored_fen = _canonical(recovered)
        if (
            recovered.ep_square is not None
            and not recovered.has_legal_en_passant()
            and set(recovered.legal_moves) == set(board.legal_moves)
            and restored_key == stored_key
        ):
            return recovered, restored_key, restored_fen, True
    planes = np.unique(np.where(encoded != tensor)[0]).tolist()
    raise ValueError(
        f"Source tensor/FEN/key mismatch at {context}: stored_key={stored_key}, "
        f"fen_key={key}, differing_planes={planes}"
    )


def _put(db: sqlite3.Connection, row: dict[str, Any]) -> None:
    existing = db.execute("SELECT split FROM positions WHERE key=?", (row["key"],)).fetchone()
    if existing and (existing[0] == "validation" or row["split"] == "train"):
        return
    db.execute(
        "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,NULL)",
        tuple(
            row[k]
            for k in (
                "key",
                "fen",
                "split",
                "group_id",
                "origin",
                "ref_path",
                "ref_index",
                "bucket",
            )
        ),
    )


def _completed(db: sqlite3.Connection, name: str) -> bool:
    return db.execute("SELECT 1 FROM units WHERE name=?", (name,)).fetchone() is not None


def _mark(db: sqlite3.Connection, name: str) -> None:
    db.execute("INSERT INTO units VALUES (?)", (name,))
    db.commit()


def _catalog_signature(db: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in db.execute(
        "SELECT key,fen,split,group_id,origin,ref_path,ref_index,bucket FROM positions ORDER BY key"
    ):
        digest.update(json.dumps(tuple(row), separators=(",", ":")).encode())
    return digest.hexdigest()


def inventory(
    config: Mapping[str, Any], *, confirm_training_import: bool = False
) -> dict[str, Any]:
    if not confirm_training_import:
        raise PermissionError("Explicit --confirm-training-import is required")
    root = Path(config["run_dir"])
    with run_lock(root):
        plan = _freeze(config)
        with closing(_connect(root)) as db:
            bound = db.execute("SELECT payload FROM binding WHERE name='plan'").fetchone()
            plan_hash = signature(plan)
            if bound and bound[0] != plan_hash:
                raise ValueError("Corpus journal belongs to another plan")
            if not bound:
                db.execute("INSERT INTO binding VALUES ('plan',?)", (plan_hash,))
                db.commit()
            if (root / "inventory.json").exists():
                return _inventory_report(root, db)
            for index, item in enumerate(plan["sources"], 1):
                unit = "source:" + item["path"]
                if _completed(db, unit):
                    continue
                data = load_dataset(Path(item["path"]), mmap=True)
                if data.format_version != 2 or data.metadata.get("value_target_spec") != SPEC:
                    raise ValueError("Legacy outcome labels cannot enter the rebuild")
                if data.metadata.get("kind") in {
                    "stockfish_refinement_replay",
                    "explicit_weighted_dataset_composition",
                }:
                    raise ValueError("Use original label sources, not replay/composed copies")
                for offset, example in enumerate(data):
                    meta = example.metadata
                    if (
                        meta.get("split") not in {"train", "validation"}
                        or str(meta.get("TrainingData", meta.get("training_data", "true"))).lower()
                        == "false"
                    ):
                        raise ValueError("Source lacks fixed splits or identifies evaluation data")
                    board, key, fen, repaired = _source_position(
                        example, data.metadata, Path(item["path"]), offset
                    )
                    if repaired:
                        db.execute(
                            "INSERT OR REPLACE INTO fen_repairs VALUES (?,?,?,?)",
                            (item["path"], offset, str(meta["fen"]), fen),
                        )
                    if meta["split"] == "validation":
                        _ban(db, {key}, 1)
                    else:
                        db.execute("INSERT OR IGNORE INTO old_training VALUES (?)", (key,))
                    _put(
                        db,
                        {
                            "key": key,
                            "fen": fen,
                            "split": meta["split"],
                            "group_id": f"{item['origin']}:{meta['group_id']}",
                            "origin": item["origin"],
                            "ref_path": item["path"],
                            "ref_index": offset,
                            "bucket": _bucket(board),
                        },
                    )
                _mark(db, unit)
                print(f"Existing label shards {index}/{len(plan['sources'])}", flush=True)
            forced: dict[str, str] = {}
            priority = {"train": 0, "validation": 1, "confirmation": 2}
            for folder in config["protected_pilots"]:
                old = load_prepared(Path(folder))
                for row in old["rows"]:
                    if row["split"] != "train":
                        _ban(
                            db,
                            orbit_keys(chess.Board(row["fen"])),
                            3 if row["split"] == "confirmation" else 1,
                        )
                    if "source_game" in row:
                        previous = forced.get(row["source_game"], "train")
                        forced[row["source_game"]] = max(
                            (previous, row["split"]), key=priority.__getitem__
                        )
            coverage = json.loads(Path(config["coverage"]).read_text(encoding="utf-8"))
            gm_config = {
                "max_pieces": 64,
                "gm_import": {
                    "format": "self-improving-chess-ai.gm-endgame-import",
                    "version": 1,
                    "coverage": config["coverage"],
                    "coverage_sha256": sha256_file(Path(config["coverage"])),
                    "snapshot": coverage["snapshot"]["path"],
                    "games_per_player": config["games_per_player"],
                    "players": sorted(config["expected_ownership"]),
                },
            }
            games, _ = _source_games(gm_config)
            parent = {g["digest"]: g["digest"] for g in games}

            def find(group: str) -> str:
                while parent[group] != group:
                    parent[group] = parent[parent[group]]
                    group = parent[group]
                return group

            owners: dict[str, str] = {}
            for game in games:
                for row in game["rows"]:
                    board = chess.Board(row["fen"])
                    if len(board.piece_map()) <= 10:
                        for key in orbit_keys(board):
                            other = owners.setdefault(key, game["digest"])
                            left, right = find(other), find(game["digest"])
                            parent[max(left, right)] = min(left, right)
            component_forced: dict[str, str] = {}
            for game_digest, split in forced.items():
                if game_digest in parent:
                    group = find(game_digest)
                    component_forced[group] = max(
                        (component_forced.get(group, "train"), split), key=priority.__getitem__
                    )
            game_splits = {}
            for index, game in enumerate(games, 1):
                group = find(game["digest"])
                fraction = int(signature({"group": group, "seed": config["seed"]})[:8], 16) / 2**32
                split = component_forced.get(
                    group,
                    "train"
                    if fraction < 0.8
                    else "validation"
                    if fraction < 0.9
                    else "confirmation",
                )
                game_splits[game["digest"]] = {"split": split, "group": group}
                unit = "gm:" + game["digest"]
                if _completed(db, unit):
                    continue
                for row in game["rows"]:
                    key, fen = row["key"], row["fen"]
                    board = chess.Board(fen)
                    related = orbit_keys(board) if len(board.piece_map()) <= 10 else {key}
                    if split == "confirmation":
                        _ban(db, related, 3)
                        continue
                    if split == "validation":
                        # Previously trained inputs are not new validation evidence.
                        old_train = any(
                            db.execute(
                                "SELECT 1 FROM old_training WHERE key=?", (candidate,)
                            ).fetchone()
                            for candidate in related
                        )
                        _ban(db, related, 2 if old_train else 1)
                        if old_train:
                            continue
                    _put(
                        db,
                        {
                            "key": key,
                            "fen": fen,
                            "split": split,
                            "group_id": f"gm:{group}",
                            "origin": "gm_full",
                            "ref_path": None,
                            "ref_index": None,
                            "bucket": _bucket(chess.Board(fen)),
                        },
                    )
                _mark(db, unit)
                if index % 100 == 0:
                    print(f"GM full-game inventory {index}/{len(games)}", flush=True)
            atomic_json(
                root / "gm_game_splits.json", {"games": game_splits, "confirmation_exported": False}
            )
            for item in plan["rollouts"]:
                unit = "rollout:" + item["path"]
                if _completed(db, unit):
                    continue
                old = load_prepared(Path(item["pilot"]))
                starts = {row["key"]: row["fen"] for row in old["rows"]}
                state = _checked_json(Path(item["path"]))
                board = chess.Board(starts[state["case_key"]])
                for move in [*state["uci"], None]:
                    _ban(db, orbit_keys(board), 2)
                    if move is not None:
                        board.push_uci(move)
                _mark(db, unit)
            fens = {" ".join(r[0].split()[:4]) for r in db.execute("SELECT fen FROM positions")}
            for row in db.execute("SELECT fen FROM positions WHERE bucket!='general'"):
                board = chess.Board(row[0])
                fens.update(
                    " ".join(b.fen(en_passant="fen").split()[:4])
                    for b in (
                        board.mirror(),
                        board.transform(chess.flip_horizontal),
                        board.mirror().transform(chess.flip_horizontal),
                    )
                )
            for index, raw in enumerate(plan["evaluation_files"], 1):
                unit = "evaluation:" + raw
                if _completed(db, unit):
                    continue
                with Path(raw).open(encoding="utf-8-sig") as handle:
                    while (evaluation_game := chess.pgn.read_game(handle)) is not None:
                        board = evaluation_game.board()
                        if (
                            evaluation_game.errors
                            or type(board) is not chess.Board
                            or not board.is_valid()
                        ):
                            raise ValueError("Invalid evaluation exclusion PGN")
                        for move in [*evaluation_game.mainline_moves(), None]:
                            if " ".join(board.fen(en_passant="fen").split()[:4]) in fens:
                                related = (
                                    orbit_keys(board)
                                    if len(board.piece_map()) <= 10
                                    else {_canonical(board)[0]}
                                )
                                _ban(db, related, 2)
                            if move is not None:
                                if move not in board.legal_moves:
                                    raise ValueError("Illegal evaluation PGN move")
                                board.push(move)
                _mark(db, unit)
                if index % 100 == 0:
                    print(
                        f"Historical exclusions {index}/{len(plan['evaluation_files'])}", flush=True
                    )
            db.execute(
                "DELETE FROM positions WHERE key IN (SELECT key FROM blocked WHERE kind>=2) OR (split='train' AND key IN (SELECT key FROM blocked))"
            )
            db.commit()
            splits = dict(db.execute("SELECT split,count(*) FROM positions GROUP BY split"))
            if set(splits) != {"train", "validation"}:
                raise ValueError("Exclusions left an empty training/development split")
            if db.execute(
                "SELECT group_id FROM positions GROUP BY group_id HAVING count(DISTINCT split)>1"
            ).fetchone():
                raise ValueError("Corpus leaks a group across splits")
            report = {
                "format": FORMAT,
                "version": 1,
                "plan_sha256": plan_hash,
                "catalog_signature": _catalog_signature(db),
                "examples": sum(splits.values()),
                "splits": splits,
                "sources": dict(
                    db.execute("SELECT origin,count(*) FROM positions GROUP BY origin")
                ),
                "pending_labels": db.execute(
                    "SELECT count(*) FROM positions WHERE ref_path IS NULL"
                ).fetchone()[0],
                "source_fen_repairs": db.execute("SELECT count(*) FROM fen_repairs").fetchone()[0],
                "training_imported": True,
                "confirmation_exported": False,
                "training_updates": 0,
            }
            atomic_json(root / "inventory.json", report)
            return report


def _inventory_report(root: Path, db: sqlite3.Connection) -> dict[str, Any]:
    report = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
    if (
        report.get("format") != FORMAT
        or report.get("version") != 1
        or report["catalog_signature"] != _catalog_signature(db)
        or report["plan_sha256"]
        != db.execute("SELECT payload FROM binding WHERE name='plan'").fetchone()[0]
        or report["splits"]
        != dict(db.execute("SELECT split,count(*) FROM positions GROUP BY split"))
        or report["examples"] != db.execute("SELECT count(*) FROM positions").fetchone()[0]
    ):
        raise ValueError("Corpus inventory integrity check failed")
    return report


def labels(config: Mapping[str, Any], maximum: int | None = None) -> dict[str, Any]:
    if maximum is not None and maximum <= 0:
        raise ValueError("max-positions must be positive")
    root = Path(config["run_dir"])
    with run_lock(root):
        _freeze(config)
        with closing(_connect(root)) as db:
            report = _inventory_report(root, db)
            teacher_config = StockfishConfig.from_mapping(config["stockfish"])
            with StockfishTeacher(teacher_config) as teacher:
                binding = {"catalog": report["catalog_signature"], "teacher": teacher.provenance}
                saved = db.execute("SELECT payload FROM binding WHERE name='teacher'").fetchone()
                if saved and json.loads(saved[0]) != binding:
                    raise ValueError("Corpus teacher binding changed")
                if not saved:
                    db.execute("INSERT INTO binding VALUES ('teacher',?)", (json.dumps(binding),))
                    db.commit()
                count = 0
                started = time.perf_counter()
                while maximum is None or count < maximum:
                    row = db.execute(
                        "SELECT * FROM positions WHERE ref_path IS NULL AND label IS NULL ORDER BY key LIMIT 1"
                    ).fetchone()
                    if row is None:
                        break
                    payload = _label_row(teacher, row)
                    db.execute(
                        "UPDATE positions SET label=? WHERE key=?",
                        (json.dumps(payload), row["key"]),
                    )
                    db.commit()
                    count += 1
                    if count % 64 == 0:
                        remaining = db.execute(
                            "SELECT count(*) FROM positions WHERE ref_path IS NULL AND label IS NULL"
                        ).fetchone()[0]
                        print(
                            f"New labels {count}; remaining {remaining}; ETA {(time.perf_counter() - started) / count * remaining / 3600:.2f} h",
                            flush=True,
                        )
                seconds = time.perf_counter() - started
                remaining = db.execute(
                    "SELECT count(*) FROM positions WHERE ref_path IS NULL AND label IS NULL"
                ).fetchone()[0]
                result = {
                    "labeled_this_invocation": count,
                    "remaining": remaining,
                    "seconds": seconds,
                    "seconds_per_position": seconds / count if count else None,
                    "estimated_remaining_hours": seconds / count * remaining / 3600
                    if count
                    else None,
                    "teacher": teacher.provenance,
                    "training_updates": 0,
                }
                atomic_json(root / "label_progress.json", result)
                return result


def finalize(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["run_dir"])
    with run_lock(root):
        _freeze(config)
        with closing(_connect(root)) as db:
            report = _inventory_report(root, db)
            if db.execute(
                "SELECT 1 FROM positions WHERE ref_path IS NULL AND label IS NULL LIMIT 1"
            ).fetchone():
                raise ValueError("Finish pending Stockfish labels first")
            saved = db.execute("SELECT payload FROM binding WHERE name='teacher'").fetchone()
            if saved:
                binding = json.loads(saved[0])
                teacher = StockfishConfig.from_mapping(config["stockfish"])
                if (
                    binding["catalog"] != report["catalog_signature"]
                    or binding["teacher"]["analysis"] != teacher.analysis_settings()
                    or binding["teacher"]["executable_sha256"] != sha256_file(teacher.executable)
                ):
                    raise ValueError("Final corpus teacher binding changed")
            output = root / "dataset.json"
            if output.exists():
                result = json.loads(output.read_text(encoding="utf-8"))
                if result["catalog_signature"] != report["catalog_signature"]:
                    raise ValueError("Final corpus binding changed")
                for _path in source_shards(output):
                    pass
            else:
                shards: list[dict[str, Any]] = []
                cached_path = None
                cached = None
                cursor = db.execute("SELECT * FROM positions ORDER BY coalesce(ref_path,''),key")
                while rows := cursor.fetchmany(int(config["shard_size"])):
                    examples = []
                    for row in rows:
                        board = chess.Board(row["fen"])
                        meta = {
                            "fen": row["fen"],
                            "position_key": row["key"],
                            "split": row["split"],
                            "group_id": row["group_id"],
                            "game_id": row["group_id"],
                            "origin": row["origin"],
                            "sampling_bucket": row["bucket"],
                            "value_target_kind": STOCKFISH_VALUE_KIND,
                        }
                        if row["ref_path"] is not None:
                            if cached_path != row["ref_path"]:
                                cached_path = row["ref_path"]
                                cached = load_dataset(Path(cached_path), mmap=True)
                            assert cached is not None
                            original = cached[int(row["ref_index"])]
                            repair = db.execute(
                                "SELECT source_fen,canonical_fen FROM fen_repairs WHERE ref_path=? AND ref_index=?",
                                (row["ref_path"], row["ref_index"]),
                            ).fetchone()
                            if repair:
                                if repair[0] != original.metadata["fen"] or repair[1] != row["fen"]:
                                    raise ValueError("Source FEN repair provenance changed")
                                meta.update(
                                    source_fen=repair[0],
                                    source_fen_adjustment="inactive_en_passant_restored_v1",
                                )
                            examples.append(
                                replace(
                                    original,
                                    metadata={
                                        **original.metadata,
                                        **meta,
                                        "label_dataset": row["ref_path"],
                                    },
                                )
                            )
                        else:
                            payload = json.loads(row["label"])
                            checksum = payload.pop("checksum")
                            if signature(payload) != checksum:
                                raise ValueError("New label checksum mismatch")
                            policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
                            for action, probability in payload["policy"]:
                                policy[action] = probability
                            examples.append(
                                TrainingExample(
                                    BoardEncoder().encode(board),
                                    policy,
                                    payload["value"],
                                    {**meta, "teacher_audit": payload["audit"]},
                                )
                            )
                    path = root / "shards" / f"positions_{len(shards):06d}.pt"
                    metadata = {
                        "value_target_spec": SPEC,
                        "catalog_signature": report["catalog_signature"],
                        "position_keys": [r["key"] for r in rows],
                    }
                    if path.exists():
                        previous = load_dataset(path, mmap=True)
                        if previous.metadata != metadata or len(previous) != len(examples):
                            raise ValueError("Interrupted corpus shard differs")
                        for a, b in zip(previous, examples, strict=True):
                            if (
                                a.metadata != b.metadata
                                or a.target_value != float(np.float32(b.target_value))
                                or not np.array_equal(a.board_tensor, b.board_tensor)
                                or not np.array_equal(a.target_policy, b.target_policy)
                            ):
                                raise ValueError("Interrupted corpus shard content differs")
                    else:
                        save_dataset(path, examples, metadata=metadata, format_version=2)
                    shards.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "sha256": sha256_file(path),
                            "examples": len(examples),
                        }
                    )
                    print(f"Corpus shards {len(shards)}", flush=True)
                result = {
                    "format": SHARDED_DATASET_FORMAT,
                    "version": 1,
                    "value_target_spec": SPEC,
                    "catalog_signature": report["catalog_signature"],
                    "examples": report["examples"],
                    "splits": report["splits"],
                    "shards": shards,
                    "confirmation_exported": False,
                }
                atomic_json(output, result)
            counts = dict(
                db.execute(
                    "SELECT bucket,count(*) FROM positions WHERE split='train' GROUP BY bucket"
                )
            )
            shares = config["sampling_shares"]
            if set(counts) != set(shares) or abs(sum(shares.values()) - 1) > 1e-9:
                raise ValueError("All declared sampling buckets must be nonempty and sum to one")
            ready = {}
            for arm in ("scratch", "finetune"):
                template = root / "configs" / f"train_{arm}.yaml"
                train = load_config(template)
                train["training"].update(
                    sampling_source_key="sampling_bucket",
                    source_sampling_multipliers={key: shares[key] / counts[key] for key in shares},
                )
                TrainingConfig.from_mapping(
                    train["training"], seed=train["seed"], device=train["device"]
                )
                path = root / "configs" / f"ready_{arm}.yaml"
                text = yaml.safe_dump(train, sort_keys=False)
                if path.exists() and path.read_text(encoding="utf-8") != text:
                    raise ValueError("Ready training config changed")
                temporary = path.with_suffix(".yaml.tmp")
                try:
                    temporary.write_text(text, encoding="utf-8")
                    replace_with_retry(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                ready[arm] = {"path": str(path), "sha256": sha256_file(path)}
            atomic_json(
                root / "training_ready.json",
                {
                    "dataset_sha256": sha256_file(output),
                    "configs": ready,
                    "training_bucket_counts": counts,
                    "expected_sampling_shares": shares,
                    "training_updates": 0,
                },
            )
            atomic_json(
                root / "training_manifest.json",
                {
                    "pairs": [
                        {"opening_fen": r[0]}
                        for r in db.execute("SELECT fen FROM positions ORDER BY key")
                    ],
                    "training_data": True,
                },
            )
            return result
