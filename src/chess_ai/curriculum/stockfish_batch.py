"""Games first, independent UCI labels second, explicit immutable finalization third."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import chess
import chess.pgn

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.agents.opening_book_agent import OpeningBookAgent
from chess_ai.agents.protocol import ChessAgent
from chess_ai.arena.match import run_match
from chess_ai.arena.paired_audit import _excluded_opening_fens, _opening_suite
from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.stockfish_teacher import StockfishConfig, StockfishTeacher
from chess_ai.data.value_targets import STOCKFISH_VALUE_KIND
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file, signature
from chess_ai.storage.games import parse_pgn_text, save_pgn

LOGGER = logging.getLogger(__name__)
BATCH_FORMAT = "self-improving-chess-ai.stockfish-offline-batch"
BATCH_VERSION = 1
# Stockfish is exclusively a labeler. Six games share each opening split group.
COHORTS = (
    ("neural_vs_d1", "neural", "d1"),
    ("neural_vs_d2", "neural", "d2"),
    ("d2_vs_d1", "d2", "d1"),
)


class StockfishBatchError(RuntimeError):
    """A signed collection cannot be safely created, resumed, or finalized."""


@dataclass(frozen=True, slots=True)
class StockfishBatchConfig:
    run_dir: Path
    neural_checkpoint: Path
    stockfish: StockfishConfig
    opening_positions: int = 100
    opening_min_full_moves: int = 2
    opening_max_full_moves: int = 5
    normal_start_every_positions: int = 10
    max_plies: int = 200
    positions_per_game: int = 24
    tactical_share: float = 0.25
    d1_depth: int = 1
    d2_depth: int = 2
    win_weight: float = 2.0
    draw_weight: float = 1.5
    loss_weight: float = 1.0
    seed: int = 2035
    device: str = "cuda"
    collection_version: int = 1
    search_simulations: int = 0
    c_puct: float = 1.5
    opening_book: Path | None = None
    extra_standard_book_games: bool = False
    exclude_opening_manifests: tuple[Path, ...] = ()
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir).resolve())
        object.__setattr__(self, "neural_checkpoint", Path(self.neural_checkpoint).resolve())
        if self.opening_book is not None:
            object.__setattr__(self, "opening_book", Path(self.opening_book).resolve())
        object.__setattr__(
            self,
            "exclude_opening_manifests",
            tuple(Path(p).resolve() for p in self.exclude_opening_manifests),
        )
        if type(self.collection_version) is not int or self.collection_version not in (1, 2):
            raise ValueError(
                "collection_version must be 1 (legacy) or 2 (PUCT evaluation refinement)"
            )
        if type(self.search_simulations) is not int or self.search_simulations < 0:
            raise ValueError("search_simulations must be a non-negative integer")
        if not math.isfinite(self.c_puct) or self.c_puct <= 0:
            raise ValueError("c_puct must be finite and positive")
        if not isinstance(self.extra_standard_book_games, bool):
            raise ValueError("extra_standard_book_games must be a boolean")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.collection_version == 1 and (
            self.search_simulations
            or self.opening_book is not None
            or self.extra_standard_book_games
            or self.exclude_opening_manifests
            or self.c_puct != 1.5
            or self.validation_fraction != 0.1
        ):
            raise ValueError("New refinement settings require collection_version=2")
        if self.collection_version == 2:
            if self.search_simulations <= 0 or not self.stockfish.store_evaluation:
                raise ValueError("Version 2 requires PUCT and stored Stockfish evaluations")
            if self.normal_start_every_positions or self.opening_min_full_moves == 0:
                raise ValueError(
                    "Version 2 varied openings require positive bounds and normal_start_every_positions=0"
                )
            if self.extra_standard_book_games != (self.opening_book is not None):
                raise ValueError(
                    "The opening book is required exactly when extra_standard_book_games=true"
                )
        for name in (
            "opening_positions",
            "max_plies",
            "positions_per_game",
            "d1_depth",
            "d2_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "opening_min_full_moves",
            "opening_max_full_moves",
            "normal_start_every_positions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.opening_max_full_moves < self.opening_min_full_moves
            or self.opening_max_full_moves * 2 >= self.max_plies
        ):
            raise ValueError("Opening bounds must be ordered and shorter than max_plies")
        if (
            isinstance(self.tactical_share, bool)
            or not isinstance(self.tactical_share, (int, float))
            or not 0 <= self.tactical_share <= 0.5
        ):
            raise ValueError("tactical_share must be between zero and 0.5")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("win_weight", "draw_weight", "loss_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, device: str | None = None
    ) -> StockfishBatchConfig:
        values = dict(raw)
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown stockfish_batch settings: {', '.join(sorted(unknown))}")
        values["stockfish"] = StockfishConfig.from_mapping(values["stockfish"])
        if device is not None:
            values["device"] = device
        return cls(**values)

    def settings(self) -> dict[str, Any]:
        raw = asdict(self)
        raw.pop("stockfish")
        if self.collection_version == 1:
            for key in (
                "collection_version",
                "search_simulations",
                "c_puct",
                "opening_book",
                "extra_standard_book_games",
                "exclude_opening_manifests",
                "validation_fraction",
            ):
                raw.pop(key)
        else:
            raw["exclude_opening_manifests"] = [str(p) for p in self.exclude_opening_manifests]
        return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}

    @property
    def value_spec(self) -> dict[str, Any]:
        return {
            "kind": STOCKFISH_VALUE_KIND,
            "perspective": "side_to_move",
            "scale_cp": self.stockfish.evaluation_scale_cp,
        }


def opening_board(config: StockfishBatchConfig, index: int) -> tuple[chess.Board, list[str]]:
    if config.normal_start_every_positions and index % config.normal_start_every_positions == 0:
        return chess.Board(), []
    for attempt in range(100):
        rng = random.Random(config.seed + index * 1_000_003 + attempt)
        board, moves = chess.Board(), []
        count = rng.randint(config.opening_min_full_moves, config.opening_max_full_moves) * 2
        for _ in range(count):
            move = rng.choice(sorted(board.legal_moves, key=lambda item: item.uci()))
            moves.append(move.uci())
            board.push(move)
            if board.is_game_over(claim_draw=True):
                break
        if len(moves) == count and not board.is_game_over(claim_draw=True):
            return board, moves
    raise StockfishBatchError("Cannot create a non-terminal seeded opening")


def _game_specs(config: StockfishBatchConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if config.collection_version == 2:
        excluded = set().union(
            *(_excluded_opening_fens(p) for p in config.exclude_opening_manifests)
        )
        suite = _opening_suite(
            openings=config.opening_positions + 1,
            seed=config.seed,
            minimum_full_moves=config.opening_min_full_moves,
            maximum_full_moves=config.opening_max_full_moves,
            excluded_fens=excluded,
        )
        starts = [(i, board, moves, False) for i, (board, moves, _) in enumerate(suite[1:], 1)]
        if config.extra_standard_book_games:
            starts.append((0, chess.Board(), [], True))
    else:
        starts = [
            (i, *opening_board(config, i), False) for i in range(1, config.opening_positions + 1)
        ]
    for index, board, moves, book_enabled in starts:
        for cohort, agent_a, agent_b in COHORTS[:2] if config.collection_version == 2 else COHORTS:
            for color in ("white", "black"):
                number = len(specs) + 1
                specs.append(
                    {
                        "number": number,
                        "game_id": f"sf-{config.seed}-{number:06d}",
                        "opening_set_id": f"sf-{config.seed}-opening-{index:04d}",
                        "opening_index": index,
                        "opening_fen": board.fen(),
                        "opening_moves": moves,
                        "cohort": cohort,
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "agent_a_color": color,
                        "pgn": str(config.run_dir / "games" / f"game_{number:06d}.pgn"),
                        "shard": str(config.run_dir / "shards" / f"game_{number:06d}.pt"),
                    }
                )
                if config.collection_version == 2:
                    specs[-1]["book_enabled"] = book_enabled
    return specs


def _load_state(config: StockfishBatchConfig, teacher: StockfishTeacher) -> dict[str, Any]:
    state_path = config.run_dir / "state.json"
    neural_hash = sha256_file(config.neural_checkpoint)
    binding: dict[str, Any] = {
        "collection": config.settings(),
        "neural_sha256": neural_hash,
        "teacher": teacher.provenance,
    }
    if config.collection_version == 2:
        binding["opening_book_sha256"] = (
            sha256_file(config.opening_book) if config.opening_book else None
        )
        binding["excluded_manifest_sha256"] = {
            str(p): sha256_file(p) for p in config.exclude_opening_manifests
        }
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise StockfishBatchError("Stockfish state root must be a mapping")
            if (
                state.get("format") != BATCH_FORMAT
                or state.get("version") != config.collection_version
            ):
                raise StockfishBatchError("Incompatible Stockfish state format/version")
            if state.get("signature") != signature(binding) or state.get("binding") != binding:
                raise StockfishBatchError(
                    "Collection configuration, checkpoint, or teacher changed; use a new run_dir"
                )
            specs = _game_specs(config)
            if len(state["games"]) > len(specs) or len(state["labels"]) > len(state["games"]):
                raise StockfishBatchError("State has inconsistent committed counts")
            for expected, game in zip(specs, state["games"], strict=False):
                if any(game.get(key) != value for key, value in expected.items()):
                    raise StockfishBatchError(
                        "Committed game metadata does not match its signed schedule"
                    )
                if sha256_file(Path(game["pgn"])) != game["pgn_sha256"]:
                    raise StockfishBatchError("Committed PGN digest mismatch")
            for game, label in zip(state["games"], state["labels"], strict=False):
                if (
                    label["shard"] != game["shard"]
                    or sha256_file(Path(label["shard"])) != label["sha256"]
                ):
                    raise StockfishBatchError("Committed label shard digest mismatch")
            if state["complete"]:
                for name, digest in state["final_hashes"].items():
                    if sha256_file(config.run_dir / name) != digest:
                        raise StockfishBatchError("Finalized artifact digest mismatch")
            return state
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StockfishBatchError(f"Invalid Stockfish resume state: {exc}") from exc
    if config.run_dir.exists() and any(config.run_dir.iterdir()):
        raise StockfishBatchError("Output exists without state; choose an empty new run_dir")
    checkpoint = load_checkpoint(config.neural_checkpoint, map_location="cpu")
    if (
        config.collection_version == 2
        and checkpoint.metadata.get("value_target_spec") != config.value_spec
    ):
        raise StockfishBatchError(
            "Refinement requires a checkpoint with matching Stockfish value semantics"
        )
    state = {
        "format": BATCH_FORMAT,
        "version": config.collection_version,
        "signature": signature(binding),
        "binding": binding,
        "games": [],
        "labels": [],
        "pending": None,
        "complete": False,
    }
    atomic_json(state_path, state)
    return state


def _recover_pending(config: StockfishBatchConfig, state: dict[str, Any]) -> None:
    pending = state.get("pending")
    if pending is None:
        return
    if not isinstance(pending, dict) or "stage" not in pending or "paths" not in pending:
        raise StockfishBatchError("Malformed pending transaction")
    # Only a path provably reserved by this exact next transaction can be removed.
    stage = pending["stage"]
    specs = _game_specs(config)
    if stage == "games":
        owned = [Path(specs[len(state["games"])]["pgn"])]
    elif stage == "labels":
        owned = [Path(specs[len(state["labels"])]["shard"])]
    elif stage == "finalize":
        owned = [config.run_dir / "dataset.pt", config.run_dir / "manifest.json"]
    else:
        raise StockfishBatchError("Unknown pending transaction stage")
    if pending["paths"] != [str(path) for path in owned]:
        raise StockfishBatchError("Pending transaction paths do not match signed outputs")
    for path in owned:
        if not path.resolve().is_relative_to(config.run_dir):
            raise StockfishBatchError("Pending path escapes run_dir")
        path.unlink(missing_ok=True)
    state["pending"] = None
    atomic_json(config.run_dir / "state.json", state)


def _reserve(
    config: StockfishBatchConfig, state: dict[str, Any], stage: str, paths: list[Path]
) -> None:
    if any(path.exists() for path in paths):
        raise StockfishBatchError(
            "Uncommitted output collision; restore state or use a new run_dir"
        )
    state["pending"] = {"stage": stage, "paths": [str(path) for path in paths]}
    atomic_json(config.run_dir / "state.json", state)


def _actor(config: StockfishBatchConfig, kind: str, neural: NeuralAgent, seed: int) -> ChessAgent:
    if kind == "neural":
        return neural
    return MinimaxAgent(
        depth=config.d1_depth if kind == "d1" else config.d2_depth,
        deterministic=False,
        seed=seed,
        name=f"Minimax {kind}",
    )


def _generate_game(
    config: StockfishBatchConfig, spec: Mapping[str, Any], neural: NeuralAgent
) -> dict[str, Any]:
    board = chess.Board()
    for uci in spec["opening_moves"]:
        board.push_uci(uci)
    a = _actor(config, spec["agent_a"], neural, config.seed + spec["number"] * 31)
    b = _actor(config, spec["agent_b"], neural, config.seed + spec["number"] * 37)
    if spec.get("book_enabled"):
        assert config.opening_book is not None
        a = OpeningBookAgent(a, config.opening_book, seed=config.seed + spec["number"] * 31)
    white, black = (a, b) if spec["agent_a_color"] == "white" else (b, a)
    result = run_match(
        white,
        black,
        max_plies=config.max_plies - len(spec["opening_moves"]),
        starting_fen=board.fen(),
        seed=None,
    )
    if result.illegal_agent is not None:
        raise StockfishBatchError(f"Illegal gameplay output from {result.illegal_agent}")
    for move in result.moves:
        board.push_uci(move.uci)
    game = chess.pgn.Game.from_board(board)
    game.headers.update(
        {
            "Event": f"Stockfish v{config.collection_version} collection (teacher absent from gameplay)",
            "White": result.white_name,
            "Black": result.black_name,
            "Result": result.result,
            "Termination": result.termination,
            "GameID": spec["game_id"],
            "TrainingDuringCollection": "false",
            **(
                {
                    "TrainingData": "true",
                    "SearchSimulations": str(config.search_simulations),
                    "OpeningBook": str(config.opening_book) if spec.get("book_enabled") else "none",
                    "OpeningSetID": spec["opening_set_id"],
                }
                if config.collection_version == 2
                else {}
            ),
        }
    )
    save_pgn(game, Path(spec["pgn"]))
    return {
        **spec,
        "result": result.result,
        "termination": result.termination,
        "plies": len(board.move_stack),
        "pgn_sha256": sha256_file(Path(spec["pgn"])),
    }


def sample_positions(
    game: chess.pgn.Game, opening_plies: int, config: StockfishBatchConfig, game_number: int
) -> list[tuple[int, chess.Board, chess.Move]]:
    """Bounded, seeded coverage with a capped tactical reservation and FEN deduplication."""
    board = game.board()
    candidates: list[tuple[int, chess.Board, chess.Move]] = []
    seen: set[bytes] = set()
    for ply, move in enumerate(game.mainline_moves()):
        if move not in board.legal_moves:
            raise StockfishBatchError("Archived PGN contains an illegal move")
        encoded = BoardEncoder().encode(board).tobytes()
        if config.collection_version == 2 or (ply >= opening_plies and encoded not in seen):
            candidates.append((ply, board.copy(stack=False), move))
            seen.add(encoded)
        board.push(move)
    if config.collection_version == 2:
        return candidates
    rng = random.Random(config.seed + game_number * 101)
    count = min(len(candidates), config.positions_per_game)
    tactics = [
        item
        for item in candidates
        if item[1].is_check() or item[1].is_capture(item[2]) or item[2].promotion
    ]
    reserved = rng.sample(tactics, min(len(tactics), math.floor(count * config.tactical_share)))
    chosen = {item[0] for item in reserved}
    remaining = [item for item in candidates if item[0] not in chosen]
    selected = reserved + rng.sample(remaining, count - len(reserved))
    return sorted(selected, key=lambda item: item[0])


def _label_game(
    config: StockfishBatchConfig, record: Mapping[str, Any], teacher: StockfishTeacher
) -> dict[str, Any]:
    games = parse_pgn_text(Path(record["pgn"]).read_text(encoding="utf-8"))
    if len(games) != 1 or games[0].headers["Result"] != record["result"]:
        raise StockfishBatchError("Archived PGN result/count mismatch")
    examples = []
    for ply, board, actual in sample_positions(
        games[0], len(record["opening_moves"]), config, record["number"]
    ):
        if config.collection_version == 2:
            fen = " ".join([*board.fen(en_passant="fen").split()[:4], "0", "1"])
            board = chess.Board(fen)
        policy, audit = teacher.analyze(board)
        outcome_value = result_value_for_turn(record["result"], board.turn)
        value = (
            float(audit["stockfish_evaluation"])
            if config.collection_version == 2
            else outcome_value
        )
        played = teacher.analyze_move(board, actual) if config.collection_version == 2 else None
        weight = (
            config.win_weight
            if outcome_value > 0
            else config.loss_weight
            if outcome_value < 0
            else config.draw_weight
        )
        examples.append(
            TrainingExample(
                BoardEncoder().encode(board),
                policy,
                value,
                {
                    **audit,
                    "game_id": record["game_id"],
                    "opening_set_id": record["opening_set_id"],
                    "absolute_ply": ply,
                    "cohort": record["cohort"],
                    "result": record["result"],
                    "termination": record["termination"],
                    "value_target_kind": STOCKFISH_VALUE_KIND
                    if config.collection_version == 2
                    else "completed_game_outcome",
                    **(
                        {
                            "played_move_evaluation": played,
                            "estimated_cp_loss": max(
                                0, audit["top_moves"][0]["score_cp"] - played["score_cp"]
                            )
                            if played is not None
                            else None,
                            "best_score_cp": audit["top_moves"][0]["score_cp"],
                            "position_key": hashlib.sha256(
                                BoardEncoder().encode(board).tobytes()
                            ).hexdigest(),
                            "book_enabled": record["book_enabled"],
                            "completed_game_outcome": outcome_value,
                            "group_id": record["opening_set_id"],
                            "composition_source": "fresh",
                        }
                        if config.collection_version == 2
                        else {}
                    ),
                    "actual_move_uci": actual.uci(),
                    "actual_matched_teacher": actual.uci() == audit["teacher_move_uci"],
                    "actor_kind": record["agent_a"]
                    if (board.turn == chess.WHITE) == (record["agent_a_color"] == "white")
                    else record["agent_b"],
                    "actor_outcome_weight": weight,
                    "player_to_move": "white" if board.turn == chess.WHITE else "black",
                    "tactical": bool(
                        board.is_check() or board.is_capture(actual) or actual.promotion
                    ),
                    "target_policy_kind": "stockfish_multipv_score_softmax",
                },
            )
        )
        if config.collection_version == 2 and len(examples) % 25 == 0:
            LOGGER.info("Stockfish game=%s evaluated moves=%d", record["number"], len(examples))
    if not examples:
        raise StockfishBatchError("Complete game yielded no nonterminal continuation positions")
    save_dataset(
        Path(record["shard"]),
        examples,
        metadata={
            "teacher": teacher.provenance,
            "format": BATCH_FORMAT,
            "version": config.collection_version,
            "game_id": record["game_id"],
            **({"value_target_spec": config.value_spec} if config.collection_version == 2 else {}),
        },
        format_version=config.collection_version,
    )
    return {
        "shard": record["shard"],
        "sha256": sha256_file(Path(record["shard"])),
        "examples": len(examples),
    }


def _finalize(config: StockfishBatchConfig, state: dict[str, Any]) -> None:
    expected_games = config.opening_positions * (4 if config.collection_version == 2 else 6)
    expected_games += 4 if config.extra_standard_book_games else 0
    if len(state["games"]) != expected_games or len(state["labels"]) != expected_games:
        raise StockfishBatchError(
            "Finalize requires all complete games and all committed label shards"
        )
    dataset = config.run_dir / "dataset.pt"
    manifest = config.run_dir / "manifest.json"
    _reserve(config, state, "finalize", [dataset, manifest])
    examples = [example for label in state["labels"] for example in load_dataset(label["shard"])]
    metadata = {
        "format": BATCH_FORMAT,
        "version": config.collection_version,
        **state["binding"],
        "training_updates": 0,
        "value_target_kind": STOCKFISH_VALUE_KIND
        if config.collection_version == 2
        else "completed_game_outcome",
    }
    if config.collection_version == 2:
        unique: dict[str, TrainingExample] = {}
        for example in examples:
            group = example.metadata["group_id"]
            fraction = int(signature({"group": group, "seed": config.seed})[:16], 16) / 2**64
            split = "validation" if fraction < config.validation_fraction else "train"
            example = replace(example, metadata={**example.metadata, "split": split})
            key = str(example.metadata["position_key"])
            if key not in unique or (
                unique[key].metadata["split"] == "train" and split == "validation"
            ):
                unique[key] = example
        examples = list(unique.values())
        metadata.update(
            value_target_spec=config.value_spec,
            explicit_split=True,
            dataset_signature=signature({"labels": state["labels"], "binding": state["binding"]}),
            recorded_move_evaluations=sum(label["examples"] for label in state["labels"]),
        )
    save_dataset(dataset, examples, metadata=metadata, format_version=config.collection_version)
    pairs = []
    for spec in _game_specs(config)[:: (4 if config.collection_version == 2 else 6)]:
        pairs.append(
            {
                "opening_index": spec["opening_index"],
                "opening_fen": spec["opening_fen"],
                "opening_moves": spec["opening_moves"],
            }
        )
    atomic_json(
        manifest,
        {
            **metadata,
            "pairs": pairs,
            "games": state["games"],
            "labels": state["labels"],
            "dataset_sha256": sha256_file(dataset),
            "examples": len(examples),
        },
    )
    state["complete"] = True
    state["pending"] = None
    state["final_hashes"] = {
        "dataset.pt": sha256_file(dataset),
        "manifest.json": sha256_file(manifest),
    }
    atomic_json(config.run_dir / "state.json", state)


def run_stockfish_batch(
    config: StockfishBatchConfig, *, stage: Literal["all", "games", "labels", "finalize"] = "all"
) -> dict[str, Any]:
    """Never train; commit each complete PGN and each complete game's label shard."""
    if stage not in {"all", "games", "labels", "finalize"}:
        raise ValueError("Unknown Stockfish collection stage")
    with run_lock(config.run_dir), StockfishTeacher(config.stockfish) as teacher:
        state = _load_state(config, teacher)
        if not state["complete"]:
            _recover_pending(config, state)
            specs = _game_specs(config)
            if stage in {"all", "games"} and len(state["games"]) < len(specs):
                neural = (
                    NeuralMCTSAgent(
                        config.neural_checkpoint,
                        device=config.device,
                        simulations=config.search_simulations,
                        c_puct=config.c_puct,
                        seed=config.seed,
                    )
                    if config.collection_version == 2
                    else NeuralAgent(
                        config.neural_checkpoint,
                        device=config.device,
                        deterministic=True,
                        seed=config.seed,
                    )
                )
                for spec in specs[len(state["games"]) :]:
                    _reserve(config, state, "games", [Path(spec["pgn"])])
                    record = _generate_game(config, spec, neural)
                    state["games"].append(record)
                    state["pending"] = None
                    atomic_json(config.run_dir / "state.json", state)
                    LOGGER.info(
                        "Stockfish collection games=%d/%d (no teacher analysis)",
                        len(state["games"]),
                        len(specs),
                    )
                del neural
            if stage in {"all", "labels"}:
                if len(state["games"]) != len(specs):
                    raise StockfishBatchError(
                        "Finish the games stage before independent teacher labeling"
                    )
                for record in state["games"][len(state["labels"]) :]:
                    _reserve(config, state, "labels", [Path(record["shard"])])
                    label = _label_game(config, record, teacher)
                    state["labels"].append(label)
                    state["pending"] = None
                    atomic_json(config.run_dir / "state.json", state)
                    LOGGER.info("Stockfish labels games=%d/%d", len(state["labels"]), len(specs))
            if stage in {"all", "finalize"}:
                _finalize(config, state)
        return {
            "run_dir": str(config.run_dir),
            "complete": state["complete"],
            "games": len(state["games"]),
            "labeled_games": len(state["labels"]),
            "examples": sum(label["examples"] for label in state["labels"]),
            "dataset_path": str(config.run_dir / "dataset.pt"),
            "training_updates": 0,
        }
