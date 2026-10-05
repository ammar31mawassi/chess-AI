"""Reproducible supervised games produced by simple classical agents."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
from numpy.typing import NDArray

from chess_ai.agents.minimax_agent import MinimaxAgent
from chess_ai.agents.protocol import ChessAgent
from chess_ai.agents.random_agent import RandomAgent
from chess_ai.data.examples import ACTION_SIZE, TrainingExample, load_dataset, save_dataset
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.games import save_pgn

LOGGER = logging.getLogger(__name__)


class DataGenerationError(RuntimeError):
    """Raised when generation settings or an agent's move are invalid."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Bounded settings for a resumable supervised generation run.

    ``max_moves`` is counted in plies (individual turns), not pairs of moves.
    Optional random-opening plies count toward that limit but are not stored as
    training targets. This exposes the classical teachers to varied positions
    without teaching the random opening moves themselves.
    """

    dataset_path: Path = Path("data/datasets/dev_examples.pt")
    pgn_dir: Path | None = None
    games: int = 2
    white_agent: str = "minimax"
    black_agent: str = "random"
    minimax_depth: int = 1
    white_depth: int | None = None
    black_depth: int | None = None
    max_moves: int = 200
    max_examples: int | None = None
    save_every_games: int = 1
    alternate_agents: bool = False
    record_agent: str | None = None
    anchor_minimax_games: int = 0
    random_opening_min_plies: int = 0
    random_opening_max_plies: int = 0
    deduplicate_positions: bool = False
    seed: int = 0
    resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        if self.pgn_dir is not None:
            object.__setattr__(self, "pgn_dir", Path(self.pgn_dir))
        if self.games <= 0:
            raise ValueError("games must be positive")
        if self.minimax_depth <= 0:
            raise ValueError("minimax_depth must be positive")
        if self.white_depth is not None and self.white_depth <= 0:
            raise ValueError("white_depth must be positive when provided")
        if self.black_depth is not None and self.black_depth <= 0:
            raise ValueError("black_depth must be positive when provided")
        if self.max_moves <= 0:
            raise ValueError("max_moves must be positive")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided")
        if self.save_every_games <= 0:
            raise ValueError("save_every_games must be positive")
        if not isinstance(self.alternate_agents, bool):
            raise TypeError("alternate_agents must be a boolean")
        if self.record_agent is not None and self.record_agent.lower() not in {
            "minimax",
            "random",
        }:
            raise ValueError("record_agent must be minimax, random, or omitted")
        if self.anchor_minimax_games < 0 or self.anchor_minimax_games > self.games:
            raise ValueError("anchor_minimax_games must be between zero and games")
        if self.random_opening_min_plies < 0:
            raise ValueError("random_opening_min_plies cannot be negative")
        if self.random_opening_max_plies < self.random_opening_min_plies:
            raise ValueError("random_opening_max_plies must be at least random_opening_min_plies")
        if self.random_opening_max_plies >= self.max_moves:
            raise ValueError("random_opening_max_plies must be smaller than max_moves")
        if not isinstance(self.deduplicate_positions, bool):
            raise TypeError("deduplicate_positions must be a boolean")
        valid_agents = {"random", "minimax"}
        if self.white_agent.lower() not in valid_agents:
            raise ValueError(f"Unsupported white_agent {self.white_agent!r}: use random or minimax")
        if self.black_agent.lower() not in valid_agents:
            raise ValueError(f"Unsupported black_agent {self.black_agent!r}: use random or minimax")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        seed: int = 0,
        resume: bool | None = None,
    ) -> GenerationConfig:
        """Build settings from the YAML ``generation`` section."""

        values = dict(raw)
        values.setdefault("seed", seed)
        if resume is not None:
            values["resume"] = resume
        for path_name in ("dataset_path", "pgn_dir"):
            if values.get(path_name) is not None:
                values[path_name] = Path(str(values[path_name]))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown generation settings: {', '.join(unknown)}")
        return cls(**values)

    def reproducibility_metadata(self) -> dict[str, Any]:
        """Settings that must match before an existing file can be resumed."""

        metadata: dict[str, Any] = {
            "white_agent": self.white_agent.lower(),
            "black_agent": self.black_agent.lower(),
            "white_depth": self.white_depth or self.minimax_depth,
            "black_depth": self.black_depth or self.minimax_depth,
            "max_moves": self.max_moves,
            "seed": self.seed,
            "board_planes": 18,
            "action_size": ACTION_SIZE,
        }
        # Omitting disabled additions preserves resume compatibility with
        # datasets written before diversified openings were introduced.
        if self.random_opening_max_plies > 0:
            metadata["random_opening_min_plies"] = self.random_opening_min_plies
            metadata["random_opening_max_plies"] = self.random_opening_max_plies
        if self.deduplicate_positions:
            metadata["deduplicate_positions"] = True
        if self.alternate_agents:
            metadata["alternate_agents"] = True
        if self.record_agent is not None:
            metadata["record_agent"] = self.record_agent.lower()
        if self.anchor_minimax_games > 0:
            metadata["anchor_minimax_games"] = self.anchor_minimax_games
        if self.pgn_dir is not None:
            metadata["pgn_archive"] = True
        return metadata


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Counts and status returned even after a graceful interruption."""

    dataset_path: Path
    pgn_dir: Path | None
    completed_games: int
    examples: int
    duplicates_discarded: int
    unrecorded_agent_moves: int
    interrupted: bool
    resumed: bool


@dataclass(slots=True)
class _PendingExample:
    board_tensor: NDArray[np.float32]
    target_policy: NDArray[np.float32]
    turn: chess.Color
    metadata: dict[str, Any]


def result_value_for_turn(result: str | chess.Outcome, turn: chess.Color) -> float:
    """Return a final result from the stored position's active perspective."""

    result_text = result.result() if isinstance(result, chess.Outcome) else result
    if result_text == "1/2-1/2":
        return 0.0
    if result_text == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result_text == "0-1":
        return 1.0 if turn == chess.BLACK else -1.0
    raise ValueError(f"Cannot create a training value from unfinished result {result_text!r}")


# A descriptive alias is useful in lessons and preserves an obvious search term.
target_value_for_turn = result_value_for_turn


def _one_hot_policy(action: int, action_size: int = ACTION_SIZE) -> NDArray[np.float32]:
    if not 0 <= action < action_size:
        raise ValueError(f"Action {action} is outside [0, {action_size})")
    target = np.zeros(action_size, dtype=np.float32)
    target[action] = 1.0
    return target


def _finalize_examples(
    pending: Sequence[_PendingExample],
    result: str,
    *,
    termination: str,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for item in pending:
        metadata = dict(item.metadata)
        metadata["result"] = result
        metadata["termination"] = termination
        examples.append(
            TrainingExample(
                board_tensor=item.board_tensor,
                target_policy=item.target_policy,
                target_value=result_value_for_turn(result, item.turn),
                metadata=metadata,
            )
        )
    return examples


def generate_game_examples(
    white_agent: ChessAgent,
    black_agent: ChessAgent,
    *,
    game_id: str,
    max_moves: int = 200,
    starting_board: chess.Board | None = None,
    example_metadata: Mapping[str, Any] | None = None,
    board_encoder: BoardEncoder | None = None,
    move_encoder: MoveEncoder | None = None,
) -> list[TrainingExample]:
    """Play one bounded game and turn every pre-move position into an example.

    If the position remains unfinished at the limit, Phase 1 records a draw.
    This makes the bootstrap pipeline bounded without pretending to know who
    would eventually win.
    """

    if max_moves <= 0:
        raise ValueError("max_moves must be positive")
    if not game_id.strip():
        raise ValueError("game_id cannot be empty")
    boards = board_encoder or BoardEncoder()
    moves = move_encoder or MoveEncoder()
    if starting_board is not None and not isinstance(starting_board, chess.Board):
        raise TypeError("starting_board must be a python-chess Board")
    board = starting_board.copy(stack=True) if starting_board is not None else chess.Board()
    if not board.is_valid():
        raise ValueError(f"Invalid starting position: {board.fen()!r}")
    initial_ply = len(board.move_stack)
    shared_metadata = dict(example_metadata or {})
    pending: list[_PendingExample] = []

    for recorded_ply in range(max_moves):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return _finalize_examples(
                pending,
                outcome.result(),
                termination=outcome.termination.name.lower(),
            )
        active_agent = white_agent if board.turn == chess.WHITE else black_agent
        # A copy prevents an accidental agent push/pop bug from corrupting the
        # authoritative game used to label subsequent positions.
        chosen_move = active_agent.choose_move(board.copy(stack=False))
        if not isinstance(chosen_move, chess.Move):
            raise DataGenerationError(
                f"Agent {active_agent.name!r} returned {type(chosen_move).__name__}, not chess.Move"
            )
        if chosen_move not in board.legal_moves:
            raise DataGenerationError(
                f"Agent {active_agent.name!r} returned illegal move {chosen_move.uci()} "
                f"in position {board.fen()}"
            )
        action = moves.encode(chosen_move)
        pending.append(
            _PendingExample(
                board_tensor=boards.encode(board),
                target_policy=_one_hot_policy(action),
                turn=board.turn,
                metadata={
                    **shared_metadata,
                    "game_id": game_id,
                    "ply": initial_ply + recorded_ply,
                    "fen": board.fen(),
                    "move_uci": chosen_move.uci(),
                    "policy_action": action,
                    "player_to_move": "white" if board.turn == chess.WHITE else "black",
                    "white_agent": white_agent.name,
                    "black_agent": black_agent.name,
                },
            )
        )
        board.push(chosen_move)

    return _finalize_examples(pending, "1/2-1/2", termination="move_limit")


def _make_agent(kind: str, *, depth: int, seed: int, color_label: str) -> ChessAgent:
    normalized = kind.lower()
    if normalized == "random":
        return RandomAgent(seed=seed, name=f"Random ({color_label})")
    if normalized == "minimax":
        return MinimaxAgent(
            depth=depth,
            deterministic=True,
            seed=seed,
            name=f"Minimax depth {depth} ({color_label})",
        )
    raise DataGenerationError(f"Unsupported generation agent {kind!r}")


def _game_agent_specs(
    config: GenerationConfig,
    *,
    game_index: int,
) -> tuple[tuple[str, int], tuple[str, int], bool]:
    """Resolve color assignments while keeping an optional minimax anchor first."""

    if game_index < config.anchor_minimax_games:
        return (
            ("minimax", config.minimax_depth),
            ("minimax", config.minimax_depth),
            True,
        )

    white_spec = (config.white_agent.lower(), config.white_depth or config.minimax_depth)
    black_spec = (config.black_agent.lower(), config.black_depth or config.minimax_depth)
    alternating_index = game_index - config.anchor_minimax_games
    if config.alternate_agents and alternating_index % 2 == 1:
        white_spec, black_spec = black_spec, white_spec
    return white_spec, black_spec, False


def _random_opening(
    config: GenerationConfig,
    *,
    game_index: int,
) -> tuple[chess.Board, int, int]:
    """Return a reproducible, unfinished board after unlabelled random plies."""

    board = chess.Board()
    if config.random_opening_max_plies == 0:
        return board, 0, config.seed

    # Keep opening randomness independent from the configured game-agent seeds.
    opening_seed = config.seed + 1_000_000_007 + game_index * 2
    opening_agent = RandomAgent(seed=opening_seed, name="Unrecorded random opening")
    length_rng = np.random.default_rng(opening_seed + 1)
    requested_plies = int(
        length_rng.integers(
            config.random_opening_min_plies,
            config.random_opening_max_plies + 1,
        )
    )

    for _ in range(requested_plies):
        move = opening_agent.choose_move(board.copy(stack=False))
        candidate = board.copy(stack=True)
        candidate.push(move)
        # Never hand the recorded teachers a game that the random prefix has
        # already finished. The final random move is simply left out.
        if candidate.is_game_over(claim_draw=True):
            break
        board.push(move)
    return board, len(board.move_stack), opening_seed


def _encoded_position_digest(example: TrainingExample) -> bytes:
    """Hash exactly the planes visible to the network for deduplication."""

    tensor = example.board_tensor
    encoded = (
        tensor.tobytes()
        if isinstance(tensor, np.ndarray)
        else tensor.detach().cpu().numpy().tobytes()
    )
    return hashlib.sha256(encoded).digest()


def _game_pgn_path(config: GenerationConfig, game_index: int) -> Path:
    if config.pgn_dir is None:
        raise ValueError("Cannot create a generated-game PGN path without pgn_dir")
    return config.pgn_dir / f"game_{game_index + 1:04d}.pgn"


def _save_generated_game_pgn(
    config: GenerationConfig,
    *,
    game_index: int,
    game_id: str,
    opening_board: chess.Board,
    opening_plies: int,
    opening_seed: int,
    white_name: str,
    black_name: str,
    generated_examples: Sequence[TrainingExample],
    policy_source_examples: int,
    retained_examples: int,
) -> Path | None:
    """Archive the complete game, including its unrecorded random prefix."""

    if config.pgn_dir is None:
        return None

    board = opening_board.copy(stack=True)
    for example in generated_examples:
        move = chess.Move.from_uci(str(example.metadata["move_uci"]))
        if move not in board.legal_moves:
            raise DataGenerationError(
                f"Cannot archive {game_id}: recorded move {move.uci()} is illegal in {board.fen()}"
            )
        board.push(move)

    game = chess.pgn.Game.from_board(board)
    result = (
        str(generated_examples[0].metadata["result"])
        if generated_examples
        else board.result(claim_draw=True)
    )
    termination = (
        str(generated_examples[0].metadata["termination"]) if generated_examples else "opening_only"
    )
    game.headers["Event"] = "Supervised Minimax Teacher Game"
    game.headers["Site"] = "Local"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    game.headers["GameID"] = game_id
    game.headers["GeneratorSeed"] = str(config.seed)
    game.headers["OpeningSeed"] = str(opening_seed)
    game.headers["UnrecordedOpeningPlies"] = str(opening_plies)
    game.headers["TeacherPlies"] = str(len(generated_examples))
    game.headers["PolicySourceAgent"] = config.record_agent or "all"
    game.headers["UnrecordedAgentPlies"] = str(len(generated_examples) - policy_source_examples)
    game.headers["RetainedTrainingExamples"] = str(retained_examples)
    game.headers["DuplicatesDiscarded"] = str(policy_source_examples - retained_examples)
    return save_pgn(game, _game_pgn_path(config, game_index))


def _validate_pgn_archive(config: GenerationConfig, completed_games: int) -> None:
    if config.pgn_dir is None:
        return
    missing = [
        _game_pgn_path(config, game_index)
        for game_index in range(completed_games)
        if not _game_pgn_path(config, game_index).is_file()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise DataGenerationError(
            "Cannot resume because archived training-game PGNs are missing: "
            f"{preview}. Restore the archive or choose new dataset_path and pgn_dir values."
        )


def _signature(config: GenerationConfig) -> str:
    encoded = json.dumps(
        config.reproducibility_metadata(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_progress(
    config: GenerationConfig,
    examples: Sequence[TrainingExample],
    completed_games: int,
    *,
    duplicates_discarded: int,
    unrecorded_agent_moves: int,
    interrupted: bool,
) -> None:
    save_dataset(
        config.dataset_path,
        examples,
        metadata={
            "kind": "supervised_generation",
            "generator_signature": _signature(config),
            "generator_config": config.reproducibility_metadata(),
            "completed_games": completed_games,
            "duplicates_discarded": duplicates_discarded,
            "unrecorded_agent_moves": unrecorded_agent_moves,
            "requested_games": config.games,
            "interrupted": interrupted,
        },
    )


def generate_supervised_data(config: GenerationConfig) -> GenerationSummary:
    """Generate, periodically save, and practically resume a dataset.

    Each game receives seeds derived from its index. Therefore resuming at game
    N produces the same game N that an uninterrupted run would have produced.
    """

    examples: list[TrainingExample] = []
    completed_games = 0
    duplicates_discarded = 0
    unrecorded_agent_moves = 0
    resumed = False
    signature = _signature(config)
    if config.resume and config.dataset_path.exists():
        loaded = load_dataset(config.dataset_path)
        saved_signature = loaded.metadata.get("generator_signature")
        if saved_signature != signature:
            raise DataGenerationError(
                "Cannot resume because generation settings differ from the saved dataset. "
                "Choose another dataset path or run with resume disabled."
            )
        raw_completed = loaded.metadata.get("completed_games", 0)
        if not isinstance(raw_completed, int) or raw_completed < 0:
            raise DataGenerationError("Saved dataset has invalid completed_games metadata")
        completed_games = raw_completed
        raw_duplicates = loaded.metadata.get("duplicates_discarded", 0)
        if not isinstance(raw_duplicates, int) or raw_duplicates < 0:
            raise DataGenerationError("Saved dataset has invalid duplicates_discarded metadata")
        duplicates_discarded = raw_duplicates
        raw_unrecorded = loaded.metadata.get("unrecorded_agent_moves", 0)
        if not isinstance(raw_unrecorded, int) or raw_unrecorded < 0:
            raise DataGenerationError("Saved dataset has invalid unrecorded_agent_moves metadata")
        unrecorded_agent_moves = raw_unrecorded
        examples = list(loaded.examples)
        resumed = True
        _validate_pgn_archive(config, completed_games)
    elif config.pgn_dir is not None and config.pgn_dir.exists():
        existing_pgns = sorted(config.pgn_dir.glob("game_*.pgn"))
        if existing_pgns:
            raise DataGenerationError(
                "Refusing to overwrite an existing generated-game PGN archive. Choose a new "
                f"pgn_dir or resume its matching dataset: {config.pgn_dir}"
            )

    seen_positions = (
        {_encoded_position_digest(example) for example in examples}
        if config.deduplicate_positions
        else set()
    )

    if config.max_examples is not None and len(examples) > config.max_examples:
        raise DataGenerationError(
            f"Saved dataset already has {len(examples)} examples, above max_examples="
            f"{config.max_examples}"
        )

    interrupted = False
    try:
        while completed_games < config.games:
            if config.max_examples is not None and len(examples) >= config.max_examples:
                break
            game_index = completed_games
            game_seed = config.seed + game_index * 2
            white_spec, black_spec, is_anchor = _game_agent_specs(
                config,
                game_index=game_index,
            )
            white = _make_agent(
                white_spec[0],
                depth=white_spec[1],
                seed=game_seed,
                color_label="White",
            )
            black = _make_agent(
                black_spec[0],
                depth=black_spec[1],
                seed=game_seed + 1,
                color_label="Black",
            )
            if is_anchor:
                opening_board, opening_plies, opening_seed = chess.Board(), 0, config.seed
            else:
                opening_board, opening_plies, opening_seed = _random_opening(
                    config,
                    game_index=game_index,
                )
            per_game_limit = config.max_moves - opening_plies
            if config.max_examples is not None:
                per_game_limit = min(per_game_limit, config.max_examples - len(examples))
            game_id = f"generated-{config.seed}-{game_index:06d}"
            generated_examples = generate_game_examples(
                white,
                black,
                game_id=game_id,
                max_moves=per_game_limit,
                starting_board=opening_board,
                example_metadata={
                    "opening_plies": opening_plies,
                    "opening_seed": opening_seed,
                    "recorded_starting_fen": opening_board.fen(),
                    "anchor_game": is_anchor,
                },
            )
            policy_source_examples: list[TrainingExample] = []
            recorded_agent = (
                config.record_agent.lower() if config.record_agent is not None else None
            )
            annotated_examples: list[TrainingExample] = []
            for example in generated_examples:
                source_kind = (
                    white_spec[0]
                    if example.metadata["player_to_move"] == "white"
                    else black_spec[0]
                )
                annotated = TrainingExample(
                    board_tensor=example.board_tensor,
                    target_policy=example.target_policy,
                    target_value=example.target_value,
                    metadata={**example.metadata, "policy_source_agent": source_kind},
                )
                annotated_examples.append(annotated)
                if recorded_agent is None or source_kind == recorded_agent:
                    policy_source_examples.append(annotated)
                else:
                    unrecorded_agent_moves += 1
            generated_examples = annotated_examples
            game_examples = policy_source_examples
            if config.deduplicate_positions:
                unique_examples: list[TrainingExample] = []
                for example in game_examples:
                    digest = _encoded_position_digest(example)
                    if digest in seen_positions:
                        duplicates_discarded += 1
                        continue
                    seen_positions.add(digest)
                    unique_examples.append(example)
                game_examples = unique_examples
            _save_generated_game_pgn(
                config,
                game_index=game_index,
                game_id=game_id,
                opening_board=opening_board,
                opening_plies=opening_plies,
                opening_seed=opening_seed,
                white_name=white.name,
                black_name=black.name,
                generated_examples=generated_examples,
                policy_source_examples=len(policy_source_examples),
                retained_examples=len(game_examples),
            )
            examples.extend(game_examples)
            completed_games += 1
            LOGGER.info(
                "Generated game %d/%d (%d total examples)",
                completed_games,
                config.games,
                len(examples),
            )
            if completed_games % config.save_every_games == 0:
                _save_progress(
                    config,
                    examples,
                    completed_games,
                    duplicates_discarded=duplicates_discarded,
                    unrecorded_agent_moves=unrecorded_agent_moves,
                    interrupted=False,
                )
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Generation interrupted; saving %d completed games", completed_games)

    _save_progress(
        config,
        examples,
        completed_games,
        duplicates_discarded=duplicates_discarded,
        unrecorded_agent_moves=unrecorded_agent_moves,
        interrupted=interrupted,
    )
    return GenerationSummary(
        dataset_path=config.dataset_path,
        pgn_dir=config.pgn_dir,
        completed_games=completed_games,
        examples=len(examples),
        duplicates_discarded=duplicates_discarded,
        unrecorded_agent_moves=unrecorded_agent_moves,
        interrupted=interrupted,
        resumed=resumed,
    )


# Concise aliases for command code and notebooks.
generate_dataset = generate_supervised_data
