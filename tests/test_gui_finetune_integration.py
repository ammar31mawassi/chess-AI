"""End-to-end coverage from a confirmed GUI game to fresh fine-tuning."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import chess

from chess_ai.cli import build_parser, command_train
from chess_ai.gui.controller import HumanNeuralGame
from chess_ai.model import PolicyValueNet, load_checkpoint, save_checkpoint


class ScriptedNeuralAgent:
    name = "Scripted neural integration agent"

    def __init__(self, moves: list[str]) -> None:
        self._moves: Iterator[str] = iter(moves)

    def choose_move(self, _board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(next(self._moves))


def test_confirmed_gui_game_fine_tunes_from_an_initial_checkpoint(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "human_gui.pt"
    game = HumanNeuralGame(
        ScriptedNeuralAgent(["e7e5", "d8h4"]),
        human_color=chess.WHITE,
        checkpoint_label="source/base.pt",
        training_enabled=True,
        session_seed=31,
        session_id="fine-tune-integration",
        pgn_dir=tmp_path / "games",
        dataset_path=dataset_path,
    )
    game.play_human_move("f2", "f3")
    game.play_ai_turn()
    game.play_human_move("g2", "g4")
    game.play_ai_turn()
    assert game.game_over
    append_result = game.append_training_examples(confirm_training=True)
    assert append_result.added_examples == 2

    source_path = save_checkpoint(
        tmp_path / "source" / "base.pt",
        PolicyValueNet(channels=4, residual_blocks=0),
        epoch=3,
    )
    output_dir = tmp_path / "fine_tuned"
    metrics_path = tmp_path / "metrics" / "fine_tune.jsonl"
    config_path = tmp_path / "fine_tune.yaml"
    config_path.write_text(
        "\n".join(
            (
                "seed: 31",
                "device: cpu",
                "training:",
                f"  dataset_path: {dataset_path.as_posix()}",
                f"  checkpoint_dir: {output_dir.as_posix()}",
                f"  metrics_path: {metrics_path.as_posix()}",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  gradient_clip: 1.0",
                "  validation_fraction: 0.0",
                "  scheduler: false",
                "  num_workers: 0",
                "  log_every: 100",
            )
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(config_path),
            "--init-checkpoint",
            str(source_path),
            "--device",
            "cpu",
        ]
    )
    assert command_train(args) == 0

    source = load_checkpoint(source_path)
    best = load_checkpoint(output_dir / "best.pt")
    latest = load_checkpoint(output_dir / "last.pt")
    assert source.epoch == 3
    assert best.epoch == latest.epoch == 1
    assert best.model.config.channels == latest.model.config.channels == 4
    assert best.model.config.residual_blocks == latest.model.config.residual_blocks == 0

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    for name in ("training_loss", "policy_loss", "value_loss"):
        assert math.isfinite(float(rows[0][name]))
        assert math.isfinite(float(best.metrics[name]))
        assert math.isfinite(float(latest.metrics[name]))
