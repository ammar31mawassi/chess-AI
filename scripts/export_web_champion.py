"""Stage the verified registered champion and its search for Vercel."""

from __future__ import annotations

import shutil
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.environment.board_encoder import BoardEncoder
from chess_ai.storage.artifacts import atomic_json, sha256_file
from chess_ai.storage.champion import load_gameplay_champion

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    champion = load_gameplay_champion(ROOT / "checkpoints")
    if champion is None:
        raise SystemExit("Register a champion before exporting.")
    web = ROOT / "web"
    model_path = web / "model/champion.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    source = NeuralAgent(champion.checkpoint, device="cpu")
    torch.set_num_threads(1)
    torch.onnx.export(
        source.model,
        (torch.zeros((1, 18, 8, 8)),),
        str(model_path),
        input_names=["board"],
        output_names=["policy", "value"],
        opset_version=17,
        dynamo=False,
    )
    exported = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    positions = [chess.Board()]
    board = chess.Board()
    for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(move)
        positions.append(board.copy())
    for board in positions:
        expected_policy, expected_value = source.evaluate(board)
        policy, value = exported.run(None, {"board": BoardEncoder().encode(board)[None]})
        np.testing.assert_allclose(policy[0], expected_policy, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(value.item(), expected_value, atol=1e-5)
    for relative in (
        "agents/neural_mcts_agent.py",
        "environment/board_encoder.py",
        "environment/move_encoder.py",
    ):
        target = web / "chess_ai" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "src/chess_ai" / relative, target)
    shutil.copyfile(web / "runtime_agent.py", web / "chess_ai/agents/neural_agent.py")
    for folder in (web / "chess_ai", web / "chess_ai/agents", web / "chess_ai/environment"):
        (folder / "__init__.py").touch()
    pieces = web / "public/pieces"
    pieces.mkdir(parents=True, exist_ok=True)
    for file in (ROOT / "src/chess_ai/gui/assets/chessnut").iterdir():
        if file.suffix == ".png" or file.name in {"COPYRIGHT.txt", "LICENSE.txt", "NOTICE.md"}:
            shutil.copyfile(file, pieces / file.name)
    atomic_json(
        model_path.with_suffix(".json"),
        {
            "checkpoint": champion.checkpoint.name,
            "checkpoint_sha256": sha256_file(champion.checkpoint),
            "model_sha256": sha256_file(model_path),
            "c_puct": champion.c_puct,
        },
    )
    print(f"Verified ONNX parity on {len(positions)} positions for {champion.checkpoint.name}")


if __name__ == "__main__":
    main()
