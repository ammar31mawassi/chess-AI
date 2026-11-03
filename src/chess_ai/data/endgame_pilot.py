"""Small, isolated endgame curriculum using the existing Stockfish target semantics.

Source catalogs are read-only. Synthetic roots, mirrors and continuations share a
group. All historical validation inputs and archived evaluation positions veto
training; generated holdouts also veto every previously trained input.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import torch

from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.stockfish_rebuild import _blocked, _canonical, _label_row
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.data.value_targets import STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature

FORMAT = "self-improving-chess-ai.endgame-pilot"
VERSION = 1
MATERIALS = {
    "queen_mate": "KQk",
    "rook_mate": "KRk",
    "pawn": "KPk",
    "pawn_race": "KPPkp",
    "rook": "KRPkr",
    "bishop": "KBPkb",
    "knight": "KNPkn",
}


def family(board: chess.Board) -> str:
    """Descriptive material buckets, not chess rules or an endgame evaluator."""
    pieces = [p.piece_type for p in board.piece_map().values() if p.piece_type != chess.KING]
    if len(pieces) == 1 and pieces[0] in (chess.QUEEN, chess.ROOK):
        return "basic_mate"
    if all(p == chess.PAWN for p in pieces):
        return "pawn"
    if chess.ROOK in pieces and all(p in (chess.PAWN, chess.ROOK) for p in pieces):
        return "rook"
    if any(p in (chess.BISHOP, chess.KNIGHT) for p in pieces) and all(
        p in (chess.PAWN, chess.BISHOP, chess.KNIGHT) for p in pieces
    ):
        return "minor"
    return "mixed"


def score_band(cp: int | float) -> str:
    """Approximate teacher buckets; these are NOT exact WDL labels."""
    return "winning" if cp >= 300 else "losing" if cp <= -300 else "balanced"


def orbit_keys(board: chess.Board) -> set[str]:
    """Keep color mirrors and file reflections together, even across sources."""
    return {
        _canonical(b)[0]
        for b in (
            board,
            board.mirror(),
            board.transform(chess.flip_horizontal),
            board.mirror().transform(chess.flip_horizontal),
        )
    }


def generated_positions(seed: int, roots_per_material: int) -> list[dict[str, Any]]:
    if roots_per_material < 10 or roots_per_material % 10:
        raise ValueError("roots_per_material must be a positive multiple of 10")
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    for name, material in MATERIALS.items():
        for index in range(roots_per_material):
            # Exact 60/20/20 group counts within each material, fixed before labels.
            split = (
                "train" if index % 10 < 6 else "validation" if index % 10 < 8 else "confirmation"
            )
            for _ in range(10000):
                root = chess.Board(None)
                for square, symbol in zip(
                    rng.sample(range(64), len(material)), material, strict=True
                ):
                    root.set_piece_at(square, chess.Piece.from_symbol(symbol))
                root.turn = bool(rng.randrange(2))
                if (
                    root.is_valid()
                    and not root.is_game_over(claim_draw=True)
                    and not (orbit_keys(root) & seen)
                ):
                    break
            else:
                raise ValueError(f"Could not generate a valid {name} root")
            group = f"generated:{name}:{index}:{seed}"
            # Each child is reached by a python-chess legal move. Random placement
            # passes is_valid(), but is not a proof of reachability from move one.
            board = root.copy()
            group_rows = []
            group_orbits: set[str] = set()
            for ply in range(3):
                if board.is_game_over(claim_draw=True):
                    break
                for position in (board, board.mirror()):
                    key, fen = _canonical(position)
                    group_rows.append(
                        {
                            "key": key,
                            "fen": fen,
                            "group_id": group,
                            "split": split,
                            "source": "generated",
                            "material": name,
                            "family": family(position),
                            "root_ply": ply,
                        }
                    )
                group_orbits.update(orbit_keys(board))
                board.push(rng.choice(sorted(board.legal_moves, key=lambda m: m.uci())))
            # Discard the whole related family on any cross-group transposition.
            if not group_orbits & seen:
                rows.extend(group_rows)
                seen.update(group_orbits)
    return rows


def _read_catalog(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _checked_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw["payload"]
    if raw["sha256"] != signature(payload):
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return dict(payload)


def _write_checked(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, {"payload": payload, "sha256": signature(payload)})


def _dataset_metadata(path: Path) -> Iterator[dict[str, Any]]:
    """Stream metadata only; never copy the full historical tensors into memory."""
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest["shards"]:
            shard = (path.parent / entry["path"]).resolve()
            if (
                not shard.is_relative_to(path.parent.resolve())
                or sha256_file(shard) != entry["sha256"]
            ):
                raise ValueError(f"Historical shard path/hash mismatch: {shard}")
            yield from _dataset_metadata(shard)
        return
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if raw.get("format_version") != 2:
        raise ValueError("Endgame protection requires explicit version-2 datasets")
    for encoded in raw["example_metadata_json"]:
        meta = json.loads(encoded)
        if meta.get("split") not in {"train", "validation"}:
            raise ValueError("Historical dataset lacks an explicit split")
        yield meta


def prepare(
    config: Mapping[str, Any], *, external_plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Freeze a bounded pool; resumable evaluation exclusion scan, no UCI/training."""
    root = Path(config["run_dir"])
    with run_lock(root):
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "plan.json"
        if plan_path.exists():
            plan = _checked_json(plan_path)
            if plan["config"] != dict(config):
                raise ValueError("Preparation config changed; reserve a new run")
            for record in plan["inputs"]:
                if sha256_file(Path(record["path"])) != record["sha256"]:
                    raise ValueError(f"Preparation input changed: {record['path']}")
        elif external_plan is not None:
            if external_plan.get("config") != dict(config):
                raise ValueError("External position plan uses different settings")
            plan = external_plan
            for record in plan["inputs"]:
                if sha256_file(Path(record["path"])) != record["sha256"]:
                    raise ValueError(f"Preparation input changed: {record['path']}")
            _write_checked(plan_path, plan)
        else:
            if sha256_file(Path(config["champion"])) != config["champion_sha256"]:
                raise ValueError("Frozen champion hash mismatch")
            rows = generated_positions(int(config["seed"]), int(config["roots_per_material"]))
            generated_keys = set().union(*(orbit_keys(chess.Board(row["fen"])) for row in rows))
            blocked: set[str] = set()
            all_old: set[str] = set()
            inputs = [
                _file_record(Path(config["champion"])),
                _file_record(Path(config["champion_registry"])),
            ]
            pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
            rng = random.Random(int(config["seed"]))
            counts: Counter[str] = Counter()
            cap = int(config["natural_per_bucket"])
            for source in config["source_catalogs"]:
                path = Path(source)
                if _blocked(path):
                    raise ValueError("Evaluation catalogs cannot be training sources")
                inputs.append(_file_record(path))
                with closing(_read_catalog(path)) as db:
                    for row in db.execute(
                        "SELECT key, fen, split, group_id, source FROM positions ORDER BY key"
                    ):
                        if row["split"] not in {"train", "validation"}:
                            raise ValueError("Source catalog lacks fixed train/validation splits")
                        if row["key"] in generated_keys:
                            all_old.add(row["key"])
                        if row["split"] == "validation":
                            blocked.add(row["key"])
                            continue
                        if sum(c.isalpha() for c in row["fen"].split()[0]) > int(
                            config["max_pieces"]
                        ):
                            continue
                        board = chess.Board(row["fen"])
                        if not board.is_valid() or board.is_game_over(claim_draw=True):
                            continue
                        if _blocked(Path(row["source"])):
                            raise ValueError("Source catalog row identifies evaluation data")
                        label_raw = db.execute(
                            "SELECT label FROM positions WHERE key=?", (row["key"],)
                        ).fetchone()[0]
                        if label_raw is None:
                            raise ValueError("Natural source must be completely labeled")
                        label = json.loads(label_raw)
                        checksum = label.pop("checksum")
                        if checksum != signature(label):
                            raise ValueError("Source label checksum mismatch")
                        bucket = f"{family(board)}:{board.turn}:{score_band(label['audit']['top_moves'][0]['score_cp'])}"
                        counts[bucket] += 1
                        record = {
                            "key": row["key"],
                            "fen": row["fen"],
                            "split": "train",
                            "group_id": f"natural:{path.parent.parent.name}:{row['group_id']}",
                            "source": str(path),
                            "family": family(board),
                            "material": "natural",
                            "source_group": row["group_id"],
                        }
                        pool = pools[bucket]
                        if len(pool) < cap:
                            pool.append(record)
                        elif (chosen := rng.randrange(counts[bucket])) < cap:
                            pool[chosen] = record
                print(f"Read natural catalog {path}", flush=True)
            for source in config["protected_datasets"]:
                path = Path(source)
                inputs.append(_file_record(path))
                for meta in _dataset_metadata(path):
                    key = meta["position_key"]
                    if key in generated_keys:
                        all_old.add(key)
                    if meta["split"] == "validation":
                        blocked.add(key)
            rows.extend(row for pool in pools.values() for row in pool)
            # A related generated root is removed wholly, including every mirror.
            rejected_groups = {
                row["group_id"]
                for row in rows
                if row["source"] == "generated"
                and orbit_keys(chess.Board(row["fen"])) & (all_old | blocked)
            }
            seen: set[str] = set()
            kept = []
            for row in rows:
                if (
                    row["key"] in seen
                    or row["key"] in blocked
                    or row["group_id"] in rejected_groups
                ):
                    continue
                kept.append(row)
                seen.add(row["key"])
            plan = {
                "format": FORMAT,
                "version": VERSION,
                "config": dict(config),
                "inputs": inputs,
                "rows": kept,
                "natural_available_by_bucket": dict(counts),
                "generated_groups_rejected_for_history": len(rejected_groups),
            }
            _write_checked(plan_path, plan)

        # Scan every archived PGN under the explicit exclusion roots. Files are
        # hashed and journaled separately, so long scans resume after interruption.
        candidates = {key for row in plan["rows"] for key in row.get("related_keys", [row["key"]])}
        fens = {
            " ".join(fen.split()[:4])
            for row in plan["rows"]
            for fen in row.get("related_fens", [row["fen"]])
        }
        files = sorted(
            {
                p.resolve()
                for source in config["evaluation_roots"]
                for p in Path(source).rglob("*.pgn")
            }
        )
        if not files:
            raise ValueError("No evaluation PGNs found; historical exclusion cannot be skipped")
        started = time.perf_counter()
        with closing(sqlite3.connect(root / "exclusions.sqlite")) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, digest TEXT, keys_json TEXT)"
            )
            blocked = set()
            records = []
            openings: set[str] = set()
            for index, path in enumerate(files, 1):
                digest = sha256_file(path)
                saved = db.execute(
                    "SELECT digest,keys_json FROM files WHERE path=?", (str(path),)
                ).fetchone()
                if saved and saved[0] != digest:
                    raise ValueError(f"Protected evaluation PGN changed: {path}")
                if saved:
                    cached = json.loads(saved[1])
                    keys = cached["keys"]
                    starts = cached["starts"]
                else:
                    keys = set()
                    starts = []
                    with path.open(encoding="utf-8-sig") as handle:
                        while (game := chess.pgn.read_game(handle)) is not None:
                            if (
                                game.errors
                                or type(game.board()) is not chess.Board
                                or not game.board().is_valid()
                            ):
                                raise ValueError(f"Invalid exclusion PGN: {path}")
                            board = game.board()
                            starts.append(board.fen())
                            for move in [*game.mainline_moves(), None]:
                                # Avoid encoding all million historical inputs.
                                if " ".join(board.fen(en_passant="fen").split()[:4]) in fens:
                                    key = _canonical(board)[0]
                                    if key in candidates:
                                        keys.add(key)
                                if move is not None:
                                    if move not in board.legal_moves:
                                        raise ValueError(f"Illegal exclusion PGN move: {path}")
                                    board.push(move)
                    db.execute(
                        "INSERT INTO files VALUES (?,?,?)",
                        (str(path), digest, json.dumps({"keys": sorted(keys), "starts": starts})),
                    )
                    db.commit()
                blocked.update(keys)
                openings.update(starts)
                records.append({"path": str(path), "sha256": digest})
                if index % 250 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"Exclusion scan {index}/{len(files)}; ETA {elapsed / index * (len(files) - index):.0f}s (cached files included)",
                        flush=True,
                    )
        rejected = {
            row["group_id"]
            for row in plan["rows"]
            if blocked.intersection(row.get("related_keys", [row["key"]]))
        }
        rows = [row for row in plan["rows"] if row["group_id"] not in rejected]
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "config": dict(config),
            "inputs": plan["inputs"],
            "rows": rows,
            "evaluation_files": records,
            "excluded_evaluation_groups": len(rejected),
            "counts": dict(Counter(row["split"] for row in rows)),
            "training_updates": 0,
            "promotion_performed": False,
        }
        if "gm_import" in plan:
            payload["gm_import"] = plan["gm_import"]
        output = root / "positions.json"
        if output.exists() and _checked_json(output) != payload:
            raise ValueError("Prepared inputs/exclusion inventory changed; use a new run")
        _write_checked(output, payload)
        atomic_json(
            root / "training_manifest.json",
            {
                "format": FORMAT + ".exclusion-starts",
                "version": VERSION,
                "positions_sha256": sha256_file(output),
                "pairs": [
                    {"opening_fen": fen} for fen in sorted(openings | {row["fen"] for row in rows})
                ],
            },
        )
        return {
            key: payload[key]
            for key in ("counts", "excluded_evaluation_groups", "training_updates")
        }


def load_prepared(root: Path) -> dict[str, Any]:
    prepared = _checked_json(root / "positions.json")
    if prepared.get("format") != FORMAT or prepared.get("version") != VERSION:
        raise ValueError("Incompatible endgame pilot; reserve a new versioned run")
    config = prepared["config"]
    if sha256_file(Path(config["champion"])) != config["champion_sha256"]:
        raise ValueError("Frozen champion changed")
    return prepared


def label(root: Path, maximum: int | None = None) -> dict[str, Any]:
    if maximum is not None and maximum <= 0:
        raise ValueError("max-positions must be positive")
    prepared = load_prepared(root)
    config = prepared["config"]
    teacher_config = StockfishConfig.from_mapping(config["stockfish"])
    binding = {
        "positions": sha256_file(root / "positions.json"),
        "teacher": teacher_config.analysis_settings(),
        "binary": sha256_file(teacher_config.executable),
    }
    with (
        run_lock(root),
        closing(sqlite3.connect(root / "labels.sqlite")) as db,
        StockfishTeacher(teacher_config) as teacher,
    ):
        db.execute("CREATE TABLE IF NOT EXISTS binding(payload TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS labels(key TEXT PRIMARY KEY,payload TEXT)")
        binding["teacher_provenance"] = teacher.provenance
        saved = db.execute("SELECT payload FROM binding").fetchone()
        if saved and json.loads(saved[0]) != binding:
            raise ValueError("Label journal input/teacher binding changed")
        if not saved:
            db.execute("INSERT INTO binding VALUES (?)", (json.dumps(binding),))
            db.commit()
        done = {row[0] for row in db.execute("SELECT key FROM labels")}
        pending = [row for row in prepared["rows"] if row["key"] not in done]
        start = time.perf_counter()
        count = 0
        for row in pending[:maximum]:
            # Same canonical cp-tanh semantics as the champion; no tablebases.
            payload = _label_row(teacher, row)
            db.execute("INSERT INTO labels VALUES (?,?)", (row["key"], json.dumps(payload)))
            db.commit()
            count += 1
            if count % 16 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"Labels {len(done) + count}/{len(prepared['rows'])}; {elapsed / count:.3f}s/position; ETA {(len(pending) - count) * elapsed / count / 60:.1f} min",
                    flush=True,
                )
        elapsed = time.perf_counter() - start
        report = {
            "labeled_this_invocation": count,
            "remaining": len(pending) - count,
            "seconds": elapsed,
            "seconds_per_position": elapsed / count if count else None,
            "estimated_remaining_seconds": (len(pending) - count) * elapsed / count
            if count
            else None,
            "teacher": teacher.provenance,
            "training_updates": 0,
        }
        atomic_json(root / "label_progress.json", report)
        return report


def labeled_rows(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = load_prepared(root)
    rows = []
    with closing(_read_catalog(root / "labels.sqlite")) as db:
        binding = json.loads(db.execute("SELECT payload FROM binding").fetchone()[0])
        config = StockfishConfig.from_mapping(prepared["config"]["stockfish"])
        if (
            binding["positions"] != sha256_file(root / "positions.json")
            or binding["teacher"] != config.analysis_settings()
            or binding["binary"] != sha256_file(config.executable)
        ):
            raise ValueError("Label journal is bound to different inputs or teacher")
        for row in prepared["rows"]:
            record = db.execute("SELECT payload FROM labels WHERE key=?", (row["key"],)).fetchone()
            if record is None:
                raise ValueError("Finish all pilot labels first")
            payload = json.loads(record[0])
            checksum = payload.pop("checksum")
            if (
                signature(payload) != checksum
                or _canonical(chess.Board(row["fen"]))[0] != row["key"]
            ):
                raise ValueError("Label or position integrity check failed")
            rows.append(
                {
                    **row,
                    "label": payload,
                    "band": score_band(payload["audit"]["top_moves"][0]["score_cp"]),
                }
            )
    return prepared, rows


def finalize(root: Path) -> dict[str, Any]:
    """Cap each material/color/value bucket; keep confirmation outside the dataset."""
    with run_lock(root):
        prepared, rows = labeled_rows(root)
        config = prepared["config"]
        output = root / "fresh.pt"
        if output.exists() and (root / "finalized.json").exists():
            report = _checked_json(root / "finalized.json")
            if report["dataset_sha256"] != sha256_file(output) or report[
                "positions_sha256"
            ] != sha256_file(root / "positions.json"):
                raise ValueError("Finalized pilot changed")
            return report
        examples = []
        counts: Counter[str] = Counter()
        for row in sorted(rows, key=lambda r: r["key"]):
            if row["split"] == "confirmation":
                continue
            board = chess.Board(row["fen"])
            source = "generated" if row["source"] == "generated" else "natural"
            bucket = f"{row['split']}:{source}:{row['family']}:{'white' if board.turn else 'black'}:{row['band']}"
            if counts[bucket] >= int(config["final_per_bucket"]):
                continue
            counts[bucket] += 1
            policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
            for action, probability in row["label"]["policy"]:
                policy[action] = probability
            examples.append(
                TrainingExample(
                    BoardEncoder().encode(board),
                    policy,
                    row["label"]["value"],
                    {
                        "fen": row["fen"],
                        "position_key": row["key"],
                        "game_id": row["group_id"],
                        "group_id": row["group_id"],
                        "split": row["split"],
                        "value_target_kind": STOCKFISH_VALUE_KIND,
                        "endgame_family": row["family"],
                        "endgame_band": row["band"],
                        "endgame_source": source,
                        "teacher_audit": row["label"]["audit"],
                    },
                )
            )
        if {e.metadata["split"] for e in examples} != {"train", "validation"}:
            raise ValueError("Pilot needs nonempty training and development sets")
        spec = {"kind": STOCKFISH_VALUE_KIND, "perspective": "side_to_move", "scale_cp": 600.0}
        if config["stockfish"]["evaluation_scale_cp"] != 600.0:
            raise ValueError("This pilot requires the frozen champion's cp/600 semantics")
        metadata = {
            "kind": FORMAT,
            "version": VERSION,
            "explicit_split": True,
            "value_target_spec": spec,
            "dataset_signature": signature(
                {"positions": sha256_file(root / "positions.json"), "counts": dict(counts)}
            ),
        }
        if output.exists():
            existing = load_dataset(output)
            if existing.metadata != metadata or len(existing) != len(examples):
                raise ValueError("Interrupted finalization has an incompatible tensor artifact")
            for old, new in zip(existing, examples, strict=True):
                if (
                    old.metadata != new.metadata
                    or old.target_value != float(np.float32(new.target_value))
                    or not np.array_equal(old.board_tensor, new.board_tensor)
                    or not np.array_equal(old.target_policy, new.target_policy)
                ):
                    raise ValueError("Interrupted finalization tensor content differs")
        else:
            save_dataset(output, examples, metadata=metadata, format_version=2)
        report = {
            "examples": len(examples),
            "by_bucket": dict(counts),
            "dataset_sha256": sha256_file(output),
            "positions_sha256": sha256_file(root / "positions.json"),
            "confirmation_exported": False,
        }
        _write_checked(root / "finalized.json", report)
        return report


def teacher_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """Time 28 fixed synthetic positions at two budgets; no dataset is produced."""
    output = Path(config["run_dir"]) / "teacher_smoke.json"
    if output.exists():
        raise ValueError("Teacher smoke already exists; keep its report")
    pool = generated_positions(int(config["seed"]), 10)
    counts: Counter[str] = Counter()
    rows = []
    for row in pool:
        if counts[row["material"]] < 4:
            rows.append(row)
            counts[row["material"]] += 1
    runs: list[dict[str, Any]] = []
    for nodes in (50000, 200000):
        settings = replace(
            StockfishConfig.from_mapping(config["stockfish"]), nodes=nodes, cache_dir=None
        )
        with StockfishTeacher(settings) as teacher:
            start = time.perf_counter()
            scores = []
            for row in rows:
                _, payload = teacher.analyze(chess.Board(row["fen"]))
                scores.append(
                    {
                        "fen": row["fen"],
                        "material": row["material"],
                        "move": payload["teacher_move_uci"],
                        "cp": payload["top_moves"][0]["score_cp"],
                    }
                )
            elapsed = time.perf_counter() - start
            runs.append(
                {
                    "nodes": nodes,
                    "seconds": elapsed,
                    "seconds_per_position": elapsed / len(rows),
                    "teacher": teacher.provenance,
                    "scores": scores,
                }
            )
            print(f"Teacher smoke: {nodes} nodes, {elapsed / len(rows):.3f}s/position", flush=True)
    report = {
        "format": FORMAT + ".teacher-smoke",
        "version": 1,
        "positions": len(rows),
        "seed": config["seed"],
        "runs": runs,
        "best_move_changes": sum(
            a["move"] != b["move"]
            for a, b in zip(runs[0]["scores"], runs[1]["scores"], strict=True)
        ),
        "score_band_changes": sum(
            score_band(a["cp"]) != score_band(b["cp"])
            for a, b in zip(runs[0]["scores"], runs[1]["scores"], strict=True)
        ),
        "training_updates": 0,
        "note": "Cold-cache timing on a tiny correlated synthetic sample; disagreement does not prove the larger budget is correct.",
    }
    atomic_json(output, report)
    return report
