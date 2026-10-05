"""Safe persistence for training examples created by the human-play GUI.

The GUI dataset is deliberately separate from generated and external-game
artifacts.  Appending requires an explicit opt-in, validates provenance at
both file and example level, and delegates the actual atomic replacement to
the repository's versioned supervised-dataset writer.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final

import chess
import numpy as np

from chess_ai.data.dataset_generator import result_value_for_turn
from chess_ai.data.examples import (
    IncompatibleDatasetError,
    LoadedDataset,
    TrainingExample,
    load_dataset,
    save_dataset,
)
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.environment.move_encoder import MoveEncoder

HUMAN_GUI_DATASET_KIND: Final = "human_gui"
HUMAN_GUI_DATASET_VERSION: Final = 1
HUMAN_GUI_POLICY_SOURCE: Final = "human_moves_only"
DEFAULT_HUMAN_GUI_DATASET_PATH: Final = Path("data/datasets/human_gui.pt")


class HumanGuiDatasetError(IncompatibleDatasetError):
    """Raised when a file cannot safely be used as a human-GUI dataset."""


class ExplicitTrainingOptInRequired(PermissionError):
    """Raised unless a completed game's human moves are explicitly approved."""


@dataclass(frozen=True, slots=True)
class HumanGuiAppendResult:
    """Summary of one atomic human-game append."""

    path: Path
    game_id: str
    added_examples: int
    total_examples: int
    completed_games: int


@contextmanager
def _exclusive_append_lock(destination: Path) -> Iterator[None]:
    """Serialize the read-validate-replace cycle across GUI processes."""

    lock_path = destination.with_name(f".{destination.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = None
    try:
        stream = lock_path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            lock_api = import_module("msvcrt")
            lock_api.locking(stream.fileno(), lock_api.LK_LOCK, 1)
        else:  # pragma: no cover - exercised by non-Windows CI
            lock_api = import_module("fcntl")
            lock_api.flock(stream.fileno(), lock_api.LOCK_EX)
    except OSError as exc:
        if stream is not None:
            stream.close()
        raise HumanGuiDatasetError(
            f"Could not lock human-GUI dataset {destination} for append: {exc}"
        ) from exc

    try:
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                lock_api.locking(stream.fileno(), lock_api.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised by non-Windows CI
                lock_api.flock(stream.fileno(), lock_api.LOCK_UN)
        finally:
            stream.close()


def _require_human_gui_metadata(dataset: LoadedDataset, source: Path) -> None:
    metadata = dataset.metadata
    if metadata.get("kind") != HUMAN_GUI_DATASET_KIND:
        raise HumanGuiDatasetError(
            f"Refusing to mix human-GUI examples into {source}: dataset kind is "
            f"{metadata.get('kind')!r}, expected {HUMAN_GUI_DATASET_KIND!r}."
        )
    if metadata.get("human_gui_dataset_version") != HUMAN_GUI_DATASET_VERSION:
        raise HumanGuiDatasetError(
            f"Human-GUI dataset version {metadata.get('human_gui_dataset_version')!r} is "
            f"incompatible with version {HUMAN_GUI_DATASET_VERSION}."
        )
    if metadata.get("policy_source") != HUMAN_GUI_POLICY_SOURCE:
        raise HumanGuiDatasetError(
            "Refusing to append because the existing dataset is not marked as human-moves-only."
        )
    if metadata.get("training_opt_in_required") is not True:
        raise HumanGuiDatasetError(
            "Refusing to append because the existing dataset lacks the explicit opt-in marker."
        )


def _validate_example_provenance(example: TrainingExample, *, persisted: bool) -> None:
    metadata = example.metadata
    if metadata.get("origin") != HUMAN_GUI_DATASET_KIND:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} does not have human-GUI provenance."
        )
    if metadata.get("policy_source") != "human":
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} is not labelled from a human move."
        )
    if persisted and metadata.get("training_opt_in_confirmed") is not True:
        raise HumanGuiDatasetError(
            f"Stored example {example.game_id!r} lacks explicit training opt-in provenance."
        )
    result = metadata.get("result")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} does not belong to a completed game."
        )
    player_to_move = metadata.get("player_to_move")
    human_color = metadata.get("human_color")
    if player_to_move not in {"white", "black"} or human_color != player_to_move:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} is not a position where the human is to move."
        )

    fen = metadata.get("fen")
    move_text = metadata.get("move_uci")
    if not isinstance(fen, str) or not isinstance(move_text, str):
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} is missing its pre-move FEN or human move."
        )
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_text)
    except ValueError as exc:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} has invalid chess metadata: {exc}"
        ) from exc
    if not board.is_valid() or move not in board.legal_moves:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} does not contain a legal human move."
        )
    expected_board_tensor = BoardEncoder().encode(board)
    if not np.array_equal(example.board_tensor, expected_board_tensor):
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} board tensor does not match its recorded FEN."
        )
    expected_color = "white" if board.turn == chess.WHITE else "black"
    if player_to_move != expected_color:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} player-to-move metadata does not match its FEN."
        )

    selected_actions = np.flatnonzero(example.target_policy)
    if len(selected_actions) != 1:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} must contain a one-hot human-move policy."
        )
    encoded_action = MoveEncoder().encode(move)
    metadata_action = metadata.get("policy_action")
    if metadata_action != encoded_action or int(selected_actions[0]) != encoded_action:
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} policy does not match its recorded human move."
        )

    turn = chess.WHITE if player_to_move == "white" else chess.BLACK
    expected_value = result_value_for_turn(str(result), turn)
    if not np.isclose(example.target_value, expected_value):
        raise HumanGuiDatasetError(
            f"Example {example.game_id!r} value does not match the completed result perspective."
        )


def _approved_copy(example: TrainingExample) -> TrainingExample:
    metadata: dict[str, Any] = dict(example.metadata)
    metadata["training_opt_in_confirmed"] = True
    return TrainingExample(
        board_tensor=example.board_tensor,
        target_policy=example.target_policy,
        target_value=example.target_value,
        metadata=metadata,
    )


def load_human_gui_dataset(path: str | Path) -> LoadedDataset:
    """Load and fully validate a dedicated human-GUI dataset."""

    source = Path(path)
    dataset = load_dataset(source)
    _require_human_gui_metadata(dataset, source)
    for example in dataset:
        _validate_example_provenance(example, persisted=True)

    actual_games = len({example.game_id for example in dataset})
    if dataset.metadata.get("completed_games") != actual_games:
        raise HumanGuiDatasetError(
            "Human-GUI dataset completed_games metadata does not match its examples."
        )
    return dataset


def append_human_gui_game(
    path: str | Path,
    examples: Sequence[TrainingExample],
    *,
    confirm_training: bool,
) -> HumanGuiAppendResult:
    """Atomically append one completed game's explicitly approved human moves.

    The full existing file is validated before it is replaced.  The underlying
    :func:`save_dataset` call writes a temporary file and uses ``os.replace``,
    so interruption cannot leave a partially serialized dataset.
    """

    if confirm_training is not True:
        raise ExplicitTrainingOptInRequired(
            "Human moves are not added to training automatically. Confirm this completed game "
            "explicitly before appending it."
        )

    incoming = list(examples)
    if not incoming:
        raise HumanGuiDatasetError("A completed game has no human-move examples to append.")
    for example in incoming:
        _validate_example_provenance(example, persisted=False)

    game_ids = {example.game_id for example in incoming}
    if len(game_ids) != 1:
        raise HumanGuiDatasetError("One append must contain examples from exactly one game ID.")
    game_id = game_ids.pop()

    destination = Path(path)
    with _exclusive_append_lock(destination):
        existing: list[TrainingExample] = []
        if destination.exists():
            loaded = load_human_gui_dataset(destination)
            existing = list(loaded.examples)
            if game_id in {example.game_id for example in existing}:
                raise HumanGuiDatasetError(
                    f"Game {game_id!r} is already present; refusing to duplicate its examples."
                )

        approved = [_approved_copy(example) for example in incoming]
        combined = [*existing, *approved]
        completed_games = len({example.game_id for example in combined})
        saved_path = save_dataset(
            destination,
            combined,
            metadata={
                "kind": HUMAN_GUI_DATASET_KIND,
                "human_gui_dataset_version": HUMAN_GUI_DATASET_VERSION,
                "policy_source": HUMAN_GUI_POLICY_SOURCE,
                "training_opt_in_required": True,
                "completed_games": completed_games,
            },
        )
    return HumanGuiAppendResult(
        path=saved_path.resolve(),
        game_id=game_id,
        added_examples=len(approved),
        total_examples=len(combined),
        completed_games=completed_games,
    )
