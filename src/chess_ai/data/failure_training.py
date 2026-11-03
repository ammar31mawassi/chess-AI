"""Fresh bot-failure collection and explicit, holdout-protected teacher import.

This is supervised correction, not reinforcement learning. Evaluation games are
never accepted as collection input. All stages have separate resumable artifacts.
"""

from __future__ import annotations

import io
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import yaml

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.arena.match import run_match
from chess_ai.arena.search_experiments import (
    bind_run,
    implementation_signature,
    opening,
    read_cases,
)
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.refinement_replay import compose_refinement_replay
from chess_ai.data.stockfish_rebuild import _blocked, _canonical
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.data.value_targets import STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature
from chess_ai.training.trainer import TrainingConfig

FORMAT = "self-improving-chess-ai.failure-training"
SPEC = {"kind": STOCKFISH_VALUE_KIND, "perspective": "side_to_move", "scale_cp": 600.0}


def position_keys(board: chess.Board) -> set[str]:
    keys = {_canonical(board)[0]}
    if len(board.piece_map()) <= 10:
        for transformed in (
            board.mirror(),
            board.transform(chess.flip_horizontal),
            board.mirror().transform(chess.flip_horizontal),
        ):
            # File reflection is not a standard-chess symmetry when castling
            # rights remain. Let python-chess reject invalid transformed states.
            if transformed.is_valid():
                keys.add(_canonical(transformed)[0])
    return keys


def collect(
    checkpoint: Path,
    output: Path,
    *,
    seed: int,
    pairs: int,
    simulations: int,
    max_plies: int,
    device: str,
    opponent: str = "minimax",
) -> dict[str, Any]:
    """Generate new explicitly designated training games, never reuse audits."""
    if pairs < 2 or max_plies < 1 or opponent not in {"minimax", "self"}:
        raise ValueError("Use at least two pairs, positive plies, and minimax/self opponent")
    with run_lock(output):
        bind_run(
            output,
            {
                "stage": "failure_collection",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "seed": seed,
                "pairs": pairs,
                "simulations": simulations,
                "max_plies": max_plies,
                "device": device,
                "opponent": opponent,
            },
        )
        counts: Counter[str] = Counter()
        for index in range(pairs):
            # Split whole color-swapped pairs before playing. Never select the
            # validation games by outcome or by the teacher's analysis.
            split = "validation" if index % 5 == 0 else "train"
            board = opening(seed, index)
            if board.is_game_over(claim_draw=True):
                raise ValueError("Terminal generated opening; reserve another seed/run")
            for bot_white in (True, False):
                path = output / f"game_{index:05}_{int(bot_white)}.json"
                if path.exists():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if raw["checksum"] != signature(raw["game"]):
                        raise ValueError("Collection game checksum mismatch")
                else:
                    bot = NeuralMCTSAgent(checkpoint, device=device, simulations=simulations)
                    other = (
                        NeuralMCTSAgent(checkpoint, device=device, simulations=simulations)
                        if opponent == "self"
                        else MinimaxAgent(depth=2)
                    )
                    match = run_match(
                        bot if bot_white else other,
                        other if bot_white else bot,
                        starting_fen=board.fen(en_passant="fen"),
                        max_plies=max_plies,
                        extra_headers={
                            "Event": "FRESH TRAINING COLLECTION",
                            "TrainingOrigin": FORMAT,
                        },
                    )
                    if match.termination == "illegal_agent_move":
                        raise ValueError("Illegal move in collection")
                    game = {
                        "group": f"failure:{seed}:{index}",
                        "split": split,
                        "bot_white": bot_white,
                        "opponent": opponent,
                        "pgn": match.pgn,
                        "termination": match.termination,
                    }
                    atomic_json(path, {"game": game, "checksum": signature(game)})
                    del bot, other
                counts[split] += 1
        report = {
            "format": FORMAT + ".collection",
            "version": 1,
            "games": dict(counts),
            "game_files": {p.name: sha256_file(p) for p in sorted(output.glob("game_*.json"))},
            "training_updates": 0,
            "teacher_labels": 0,
        }
        atomic_json(output / "collection.json", report)
        return report


def _protect_board(board: chess.Board, blocked: set[str]) -> None:
    blocked.update(position_keys(board))


def protected_inputs(corpus: Path, suite: Path) -> tuple[set[str], set[str], dict[str, str]]:
    """Read finalized corpus + current historical evaluation paths without writes."""
    ready = json.loads((corpus / "training_ready.json").read_text(encoding="utf-8"))
    if sha256_file(corpus / "dataset.json") != ready["dataset_sha256"]:
        raise ValueError("Finalized replay manifest changed")
    files = {str(corpus / "dataset.json"): ready["dataset_sha256"], str(suite): sha256_file(suite)}
    blocked: set[str] = set()
    training: set[str] = set()
    with closing(
        sqlite3.connect((corpus / "catalog.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        blocked.update(r[0] for r in db.execute("SELECT key FROM blocked"))
        for _key, fen, split in db.execute("SELECT key,fen,split FROM positions"):
            if split == "train":
                training.update(position_keys(chess.Board(fen)))
            else:
                _protect_board(chess.Board(fen), blocked)
    # Include benchmark cases and their complete prescribed lines, not only roots.
    for case in read_cases(suite):
        board = chess.Board(case["fen"])
        _protect_board(board, blocked)
        for uci in [*case.get("setup_moves", []), *case.get("solution", [])]:
            board.push_uci(uci)
            _protect_board(board, blocked)
    for path in sorted(Path("data/games").rglob("*.pgn")):
        if not _blocked(path):
            continue
        files[str(path)] = sha256_file(path)
        with path.open(encoding="utf-8-sig") as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                if game.errors:
                    raise ValueError(f"Invalid evaluation PGN: {path}")
                board = game.board()
                _protect_board(board, blocked)
                for move in game.mainline_moves():
                    board.push(move)
                    _protect_board(board, blocked)
    # Existing per-ply endgame rollouts store their source pilot and full UCI path.
    for binding_path in sorted(Path("data/experiments").rglob("binding.json")):
        cases = sorted(binding_path.parent.glob("case_*.json"))
        if not cases:
            continue
        from chess_ai.data.endgame_pilot import _checked_json

        binding = _checked_json(binding_path)
        pilots = []
        for positions in Path("data/experiments").glob("*/positions.json"):
            if sha256_file(positions) == binding["positions_sha256"]:
                pilots.append(positions)
        if not pilots:
            raise ValueError(f"Unmapped protected rollout {binding_path}")
        positions = pilots[0]
        prepared = _checked_json(positions)
        roots = {r["key"]: r["fen"] for r in prepared["rows"]}
        files[str(positions)] = sha256_file(positions)
        files[str(binding_path)] = sha256_file(binding_path)
        for path in cases:
            state = _checked_json(path)
            board = chess.Board(roots[state["case_key"]])
            _protect_board(board, blocked)
            for uci in state["uci"]:
                board.push_uci(uci)
                _protect_board(board, blocked)
            files[str(path)] = sha256_file(path)
    # Search-ablation matches live outside historical arena directories. Their
    # embedded PGNs are still evaluation-only and must be protected in later imports.
    for path in sorted(Path("data/search_experiments").rglob("game_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "result" not in raw:
            continue
        if raw["checksum"] != signature(raw["result"]):
            raise ValueError("Search evaluation checksum mismatch")
        game = chess.pgn.read_game(io.StringIO(raw["result"]["pgn"]))
        if game is None or game.errors:
            raise ValueError("Invalid search evaluation PGN")
        board = game.board()
        _protect_board(board, blocked)
        for move in game.mainline_moves():
            board.push(move)
            _protect_board(board, blocked)
        files[str(path)] = sha256_file(path)
    return blocked, training, files


def label_failures(
    collection: Path,
    corpus: Path,
    suite: Path,
    output: Path,
    teacher: StockfishConfig,
    *,
    confirm_training_import: bool,
    seed: int,
    positions_per_game: int = 24,
    minimum_loss_cp: int = 150,
    maximum: int | None = None,
) -> dict[str, Any]:
    if not confirm_training_import:
        raise ValueError("Explicit --confirm-training-import is required")
    if positions_per_game < 1 or minimum_loss_cp < 1 or (maximum is not None and maximum < 1):
        raise ValueError("Use positive sampling, loss, and timing limits")
    if teacher.evaluation_scale_cp != 600 or not teacher.store_evaluation:
        raise ValueError("Failure labels require cp/600 evaluation targets")
    receipt = json.loads((collection / "collection.json").read_text(encoding="utf-8"))
    if receipt.get("format") != FORMAT + ".collection" or receipt.get("version") != 1:
        raise ValueError("Only fresh training collections are accepted")
    blocked, historical_train, files = protected_inputs(corpus, suite)
    with run_lock(output):
        bind_run(
            output,
            {
                "stage": "failure_labels",
                "collection_sha256": sha256_file(collection / "collection.json"),
                "protected_files": files,
                "protected_keys_sha256": signature({"keys": sorted(blocked)}),
                "historical_train_keys_sha256": signature({"keys": sorted(historical_train)}),
                "teacher": teacher.analysis_settings(),
                "teacher_binary_sha256": sha256_file(teacher.executable),
                "seed": seed,
                "positions_per_game": positions_per_game,
                "minimum_loss_cp": minimum_loss_cp,
            },
        )
        candidates = []
        for name, digest in receipt["game_files"].items():
            path = collection / name
            if path.parent.resolve() != collection.resolve() or sha256_file(path) != digest:
                raise ValueError("Collection game path/hash mismatch")
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = raw["game"]
            if raw["checksum"] != signature(record):
                raise ValueError("Collection record checksum mismatch")
            game = chess.pgn.read_game(io.StringIO(record["pgn"]))
            if game is None or game.errors or game.headers.get("TrainingOrigin") != FORMAT:
                raise ValueError("Input is not a valid fresh training game")
            board = game.board()
            positions = []
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError("Illegal collection mainline")
                if record["opponent"] == "self" or board.turn == record["bot_white"]:
                    if board.is_game_over(claim_draw=True):
                        raise ValueError("Collection continued after terminal position")
                    keys = position_keys(board)
                    if not keys & blocked and not (
                        record["split"] == "validation" and keys & historical_train
                    ):
                        positions.append(
                            {
                                "key": _canonical(board)[0],
                                "fen": board.fen(en_passant="fen"),
                                "move": move.uci(),
                                "group": record["group"],
                                "split": record["split"],
                                "result": game.headers["Result"],
                                "turn": board.turn,
                                "termination": record["termination"],
                            }
                        )
                board.push(move)
            rng = random.Random(f"{seed}:{name}")
            candidates.extend(rng.sample(positions, min(len(positions), positions_per_game)))
        # Validation owns transpositions and endgame mirrors across new pairs.
        new_validation: set[str] = set()
        for row in candidates:
            if row["split"] == "validation":
                new_validation.update(position_keys(chess.Board(row["fen"])))
        unique: dict[str, dict[str, Any]] = {}
        for row in sorted(
            candidates, key=lambda r: (r["split"] != "validation", r["group"], r["key"])
        ):
            if row["split"] == "train" and position_keys(chess.Board(row["fen"])) & new_validation:
                continue
            unique.setdefault(row["key"], row)
        with (
            closing(sqlite3.connect(output / "labels.sqlite")) as db,
            StockfishTeacher(teacher) as engine,
        ):
            db.execute("CREATE TABLE IF NOT EXISTS labels(key TEXT PRIMARY KEY,payload TEXT)")
            completed = 0
            for key, row in sorted(unique.items()):
                if db.execute("SELECT 1 FROM labels WHERE key=?", (key,)).fetchone():
                    continue
                if maximum is not None and completed >= maximum:
                    break
                board = chess.Board(row["fen"])
                policy, audit = engine.analyze(board)
                played = engine.analyze_move(board, chess.Move.from_uci(row["move"]))
                best_cp = audit["top_moves"][0]["score_cp"]
                loss = max(0, best_cp - played["score_cp"])
                actor_won = row["result"] == ("1-0" if row["turn"] else "0-1")
                failed_conversion = best_cp >= 200 and not actor_won
                selected = loss >= minimum_loss_cp or failed_conversion
                payload = {
                    **row,
                    "selected": selected,
                    "estimated_cp_loss": loss,
                    "failed_conversion_proxy": failed_conversion,
                    "policy": policy.tolist(),
                    "value": audit["stockfish_evaluation"],
                    "audit": audit,
                    "played_audit": played,
                }
                with db:
                    db.execute(
                        "INSERT INTO labels VALUES(?,?)",
                        (
                            key,
                            json.dumps(
                                {
                                    "payload": payload,
                                    "checksum": signature(payload),
                                },
                                sort_keys=True,
                            ),
                        ),
                    )
                completed += 1
            labeled = db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        report = {
            "format": FORMAT + ".labels",
            "version": 1,
            "candidate_positions": len(unique),
            "labeled": labeled,
            "complete": labeled == len(unique),
            "training_updates": 0,
            "limitations": "Teacher FEN labels omit repetition history. Cp loss and failed conversion are bounded-analysis proxies; capped games remain unresolved. Validation failures are diagnostic development data, never confirmation evidence.",
        }
        atomic_json(output / "label_progress.json", report)
        return report


def finalize_failures(
    output: Path, replay: Path, train_template: Path, *, seed: int
) -> dict[str, Any]:
    """Export isolated fresh/replay dataset and a candidate config, not an optimizer run."""
    with run_lock(output):
        progress = json.loads((output / "label_progress.json").read_text(encoding="utf-8"))
        if not progress["complete"]:
            raise ValueError("Finish failure labels before finalizing")
        binding = json.loads((output / "binding.json").read_text(encoding="utf-8"))
        if binding["implementation_sha256"] != implementation_signature():
            raise ValueError("Failure implementation changed after labeling; reserve a new import")
        if seed != binding["seed"]:
            raise ValueError("Finalization seed differs from label seed")
        for path, digest in binding["protected_files"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("Protected inputs changed; reserve a new import")
        if (
            str(replay) not in binding["protected_files"]
            or sha256_file(replay) != binding["protected_files"][str(replay)]
        ):
            raise ValueError("Replay must be the protected finalized corpus manifest")
        receipt_path = output / "training_ready.json"
        if receipt_path.exists():
            ready = json.loads(receipt_path.read_text(encoding="utf-8"))
            if any(sha256_file(Path(p)) != h for p, h in ready["files"].items()):
                raise ValueError("Ready failure artifacts changed")
            return ready
        examples = []
        with closing(sqlite3.connect(output / "labels.sqlite")) as db:
            for (encoded,) in db.execute("SELECT payload FROM labels ORDER BY key"):
                raw = json.loads(encoded)
                row = raw["payload"]
                if raw["checksum"] != signature(row):
                    raise ValueError("Failure label checksum mismatch")
                # Keep ALL sampled validation positions: selecting validation
                # only by mistakes would distort its measured distribution.
                if not row["selected"] and row["split"] != "validation":
                    continue
                examples.append(
                    TrainingExample(
                        BoardEncoder().encode(chess.Board(row["fen"])),
                        np.asarray(row["policy"], dtype=np.float32),
                        row["value"],
                        {
                            "fen": row["fen"],
                            "position_key": row["key"],
                            "split": row["split"],
                            "group_id": row["group"],
                            "game_id": row["group"],
                            "value_target_kind": STOCKFISH_VALUE_KIND,
                            "teacher_audit": row["audit"],
                            "estimated_cp_loss": row["estimated_cp_loss"],
                        },
                    )
                )
        if {str(e.metadata["split"]) for e in examples} != {"train", "validation"}:
            raise ValueError(
                "Need fresh failures in train and independent validation positions; collect more pairs"
            )
        fresh = output / "fresh.pt"
        if not fresh.exists():
            save_dataset(
                fresh,
                examples,
                format_version=2,
                metadata={
                    "explicit_split": True,
                    "value_target_spec": SPEC,
                    "kind": FORMAT,
                    "version": 1,
                },
            )
        else:
            saved = load_dataset(fresh)
            if len(saved) != len(examples) or any(
                a.metadata != b.metadata
                or not np.array_equal(a.board_tensor, b.board_tensor)
                or not np.array_equal(a.target_policy, b.target_policy)
                or not np.isclose(a.target_value, b.target_value)
                for a, b in zip(saved, examples, strict=True)
            ):
                raise ValueError("Existing failure dataset differs from committed labels")
        dataset = output / "replay.pt"
        if not dataset.exists():
            compose_refinement_replay(fresh, replay, dataset, seed=seed)
        else:
            old = load_dataset(dataset)
            expected = {
                "fresh_sha256": sha256_file(fresh),
                "replay_manifest_sha256": sha256_file(replay),
                "seed": seed,
                "fresh_fraction": 0.5,
            }
            if old.metadata.get("composition_binding") != expected:
                raise ValueError("Existing replay binding changed")
        template = yaml.safe_load(train_template.read_text(encoding="utf-8"))
        config = {**template, "seed": seed}
        candidate_id = "failure_" + signature({"output": str(output.resolve())})[:16]
        config["training"] = {
            **template["training"],
            "dataset_path": str(dataset),
            "checkpoint_dir": str(Path("checkpoints") / candidate_id),
            "metrics_path": str(Path("data/metrics") / f"{candidate_id}.jsonl"),
            "learning_rate": 1e-5,
            "epochs": 8,
            "checkpoint_every": 1,
            "early_stopping_patience": 3,
            "baseline_eligible": True,
            "freeze_batch_norm_statistics": True,
            "use_dataset_split": True,
            "value_target_spec": SPEC,
            "validation_source_key": "composition_source",
            "source_sampling_multipliers": {},
            "game_balanced_sampling": False,
        }
        TrainingConfig.from_mapping(
            config["training"], seed=seed, device=config.get("device", "cpu")
        )
        train_path = output / "train.yaml"
        text = yaml.safe_dump(config, sort_keys=False)
        if train_path.exists() and train_path.read_text(encoding="utf-8") != text:
            raise ValueError("Failure training config exists with different settings")
        train_path.write_text(text, encoding="utf-8")
        ready = {
            "format": FORMAT + ".ready",
            "version": 1,
            "files": {str(p): sha256_file(p) for p in (fresh, dataset, train_path)},
            "train_config": str(train_path),
            "training_updates": 0,
            "candidate_checkpoint": str(Path(config["training"]["checkpoint_dir"]) / "best.pt"),
        }
        atomic_json(receipt_path, ready)
        return ready
