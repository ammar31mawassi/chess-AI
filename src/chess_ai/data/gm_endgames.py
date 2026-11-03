"""Explicit multi-player PGN positions for a bounded, grouped endgame pilot.

Human mainlines supply positions only. The existing endgame label journal supplies
Stockfish targets, and its dataset finalizer never exports confirmation positions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chess_ai.data.endgame_pilot import (
    FORMAT,
    VERSION,
    _checked_json,
    _dataset_metadata,
    _file_record,
    _read_catalog,
    _write_checked,
    family,
    load_prepared,
    orbit_keys,
    prepare,
)
from chess_ai.data.stockfish_rebuild import _blocked, _canonical, _game_digest
from chess_ai.storage.artifacts import sha256_file, signature

IMPORT_FORMAT = "self-improving-chess-ai.gm-endgame-import"


def _related(board: chess.Board) -> list[tuple[str, str]]:
    return [
        _canonical(position)
        for position in (
            board,
            board.mirror(),
            board.transform(chess.flip_horizontal),
            board.mirror().transform(chess.flip_horizontal),
        )
    ]


def _source_games(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = config["gm_import"]
    if source.get("format") != IMPORT_FORMAT or source.get("version") != 1:
        raise ValueError("Incompatible GM import configuration")
    coverage_path, snapshot = Path(source["coverage"]), Path(source["snapshot"])
    if _blocked(snapshot) or sha256_file(coverage_path) != source["coverage_sha256"]:
        raise ValueError("GM coverage changed or source identifies evaluation data")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if (
        coverage.get("format") != "self-improving-chess-ai.endgame-archive-coverage"
        or coverage.get("version") not in {2, 3}
        or coverage.get("require_recorded_moves") is not True
        or sha256_file(snapshot) != coverage["snapshot"]["sha256"]
        or str(snapshot) != coverage["snapshot"]["path"]
    ):
        raise ValueError("GM snapshot/coverage version or checksum mismatch")
    owners = coverage["selected_game_owners"]
    quota = int(source["games_per_player"])
    if quota <= 0 or coverage["selection_ownership"] != {p: quota for p in source["players"]}:
        raise ValueError("GM collection does not meet the explicit per-player quota")
    games = []
    seen: set[str] = set()
    with snapshot.open(encoding="utf-8-sig") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            board = game.board()
            if (
                game.errors
                or type(board) is not chess.Board
                or board.chess960
                or not board.is_valid()
                or game.headers.get("TrainingData", "true").lower() == "false"
                or game.headers.get("EvaluationKind")
                or game.headers.get("Result") not in {"1-0", "0-1", "1/2-1/2"}
            ):
                raise ValueError("Invalid or evaluation-only GM game")
            digest = _game_digest(game)
            participants = {game.headers.get(c, "").casefold() for c in ("White", "Black")}
            if digest in seen or owners.get(digest) not in participants:
                raise ValueError("Duplicate GM game or mismatched player ownership")
            seen.add(digest)
            positions = []
            moves = list(game.mainline_moves())
            if not moves:
                raise ValueError("Empty GM game")
            for ply, move in enumerate(moves):
                if move not in board.legal_moves:
                    raise ValueError("Illegal GM mainline move")
                if len(board.piece_map()) <= int(config["max_pieces"]) and not board.is_game_over(
                    claim_draw=True
                ):
                    key, fen = _canonical(board)
                    positions.append(
                        {
                            "key": key,
                            "fen": fen,
                            "family": family(board),
                            "material": "gm",
                            "source": str(snapshot),
                            "source_game": digest,
                            "root_ply": ply,
                            "owner": owners[digest],
                        }
                    )
                board.push(move)
            games.append({"digest": digest, "rows": positions})
    if (
        seen != set(coverage["selected_game_digests"])
        or len(games) != quota * len(source["players"])
        or Counter(owners[digest] for digest in seen) != coverage["selection_ownership"]
    ):
        raise ValueError("GM snapshot game inventory differs from the frozen collection")
    return games, coverage


def build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Join transposing games before splitting, then screen all their endgame inputs."""
    if int(config["max_positions_per_game"]) < 2:
        raise ValueError("max_positions_per_game must be at least two")
    games, coverage = _source_games(config)
    inputs = []

    def record(path: Path) -> None:
        item = _file_record(path)
        if item not in inputs:
            inputs.append(item)

    for path in (
        config["champion"],
        config["champion_registry"],
        config["gm_import"]["coverage"],
        config["gm_import"]["snapshot"],
    ):
        record(Path(path))
    if sha256_file(Path(config["champion"])) != config["champion_sha256"]:
        raise ValueError("Frozen champion hash mismatch")
    parent = {g["digest"]: g["digest"] for g in games}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    key_owner: dict[str, str] = {}
    related_by_game: dict[str, dict[str, str]] = {}
    for game in games:
        related = dict(pair for row in game["rows"] for pair in _related(chess.Board(row["fen"])))
        related_by_game[game["digest"]] = related
        for key in related:
            previous = key_owner.setdefault(key, game["digest"])
            left, right = find(previous), find(game["digest"])
            parent[max(left, right)] = min(left, right)
    candidates = set(key_owner)
    blocked: set[str] = set()
    for raw in config["source_catalogs"]:
        path = Path(raw)
        record(path)
        with closing(_read_catalog(path)) as db:
            blocked.update(
                r[0] for r in db.execute("SELECT key FROM positions") if r[0] in candidates
            )
    for raw in config["protected_datasets"]:
        path = Path(raw)
        record(path)
        if path.suffix == ".json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for entry in manifest["shards"]:
                record((path.parent / entry["path"]).resolve())
        blocked.update(
            str(meta["position_key"])
            for meta in _dataset_metadata(path)
            if meta["position_key"] in candidates
        )
    for raw in config["protected_pilots"]:
        pilot = Path(raw)
        record(pilot / "positions.json")
        old = load_prepared(pilot)
        for row in old["rows"]:
            blocked.update(orbit_keys(chess.Board(row["fen"])) & candidates)
        # Endgame rollouts are JSON rather than PGN. Protect their full paths too.
        for folder in config["protected_rollout_roots"]:
            for path in sorted(Path(folder).glob("case_*.json")):
                record(path)
                state = _checked_json(path)
                lookup = {r["key"]: r["fen"] for r in old["rows"]}
                if state["case_key"] not in lookup:
                    raise ValueError("Protected rollout does not belong to the source pilot")
                board = chess.Board(lookup[state["case_key"]])
                for uci in [*state["uci"], None]:
                    blocked.update(orbit_keys(board) & candidates)
                    if uci is not None:
                        board.push_uci(uci)
    rejected = {find(key_owner[key]) for key in blocked}
    related_by_group: dict[str, dict[str, str]] = defaultdict(dict)
    for digest, related in related_by_game.items():
        related_by_group[find(digest)].update(related)
    kept = []
    seen: set[str] = set()
    described: set[str] = set()
    for game in sorted(games, key=lambda g: g["digest"]):
        group = find(game["digest"])
        if group in rejected:
            continue
        # A component, including every game/continuation/mirror, has one split.
        fraction = int(signature({"seed": config["seed"], "group": group})[:8], 16) / 2**32
        split = "train" if fraction < 0.7 else "validation" if fraction < 0.85 else "confirmation"
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(
            game["rows"], key=lambda r: signature({"seed": config["seed"], "key": r["key"]})
        ):
            if row["key"] not in seen:
                board = chess.Board(row["fen"])
                buckets[f"{row['family']}:{board.turn}"].append(row)
        selected: list[dict[str, Any]] = []
        while buckets and len(selected) < int(config["max_positions_per_game"]):
            for bucket in sorted(list(buckets)):
                row = buckets[bucket].pop(0)
                if row["key"] not in seen:
                    selected.append(row)
                    seen.add(row["key"])
                if not buckets[bucket]:
                    del buckets[bucket]
                if len(selected) == int(config["max_positions_per_game"]):
                    break
        for row in selected:
            row.update(group_id=f"gm:{group}", split=split)
            # Screen unselected endgame inputs too; any collision vetoes the group.
            if group not in described:
                row["related_keys"] = sorted(related_by_group[group])
                row["related_fens"] = sorted(set(related_by_group[group].values()))
                described.add(group)
            kept.append(row)
    if {r["split"] for r in kept} != {"train", "validation", "confirmation"}:
        raise ValueError("GM pool needs nonempty whole-game train/development/confirmation groups")
    return {
        "format": FORMAT,
        "version": VERSION,
        "config": dict(config),
        "inputs": inputs,
        "rows": kept,
        "gm_import": {
            "format": IMPORT_FORMAT,
            "version": 1,
            "collection_games": coverage["selected_games"],
            "selection_ownership": coverage["selection_ownership"],
            "groups_rejected_for_history": len(rejected),
            "position_target_source": "local Stockfish only; no human moves/results as targets",
        },
    }


def prepare_gm(config: Mapping[str, Any], *, confirm_training_import: bool) -> dict[str, Any]:
    if not confirm_training_import:
        raise PermissionError("Explicit --confirm-training-import is required for GM games")
    root = Path(config["run_dir"])
    plan = None if (root / "plan.json").exists() else build_plan(config)
    report = prepare(config, external_plan=plan)
    if set(report["counts"]) != {"train", "validation", "confirmation"}:
        raise ValueError("Exclusions left an empty GM split; reserve a different pilot")
    _write_checked(
        root / "training_import.json",
        {
            "format": IMPORT_FORMAT,
            "version": 1,
            "explicit_training_import": True,
            "positions_sha256": sha256_file(root / "positions.json"),
            "config_signature": signature(dict(config)),
            "source": config["gm_import"]["snapshot"],
            "target_source": "local Stockfish only",
            "training_updates": 0,
        },
    )
    return report


def validate_inputs(root: Path) -> dict[str, Any]:
    prepared = load_prepared(root)
    imported = _checked_json(root / "training_import.json")
    if (
        imported.get("explicit_training_import") is not True
        or imported.get("positions_sha256") != sha256_file(root / "positions.json")
        or imported.get("config_signature") != signature(prepared["config"])
    ):
        raise ValueError("GM training import receipt changed")
    for record in [*prepared["inputs"], *prepared["evaluation_files"]]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Frozen GM/protection input changed: {record['path']}")
    return prepared
