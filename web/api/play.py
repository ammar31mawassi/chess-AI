"""Stateless game turns for the public workbench."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import chess

from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent

MODEL = Path(__file__).resolve().parents[1] / "model/champion.onnx"
MAX_PLIES = 1000


def _play(board: chess.Board, uci: object) -> str:
    if not isinstance(uci, str) or len(uci) not in (4, 5):
        raise ValueError("Expected a legal UCI move.")
    if board.is_game_over(claim_draw=True):
        raise ValueError("This game has ended.")
    move = board.parse_uci(uci)
    if not move:
        raise ValueError("Passes are unavailable in chess.")
    san = board.san(move)
    board.push(move)
    return san


def turn(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history", [])
    human_side = payload.get("side", "white")
    if human_side not in ("white", "black"):
        raise ValueError("Choose White or Black.")
    if not isinstance(history, list) or len(history) > MAX_PLIES:
        raise ValueError("Invalid game history.")
    human = human_side == "white"
    board = chess.Board()
    moves = [_play(board, uci) for uci in history]
    requested_move = payload.get("move")
    if requested_move is not None:
        if board.turn != human:
            raise ValueError("Wait for the champion's move.")
        moves.append(_play(board, requested_move))
    outcome = board.outcome(claim_draw=True)
    if payload.get("reply") is True and outcome is None and board.turn != human:
        model = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
        agent = NeuralMCTSAgent(
            MODEL, simulations=1_000_000, c_puct=model["c_puct"], search_seconds=5.0
        )
        move = agent.choose_move(board)
        moves.append(_play(board, move.uci()))
        outcome = board.outcome(claim_draw=True)
    return {
        "fen": board.fen(),
        "history": [move.uci() for move in board.move_stack],
        "san": moves,
        "legal": [move.uci() for move in board.legal_moves] if outcome is None else [],
        "turn": "white" if board.turn else "black",
        "check": chess.square_name(king)
        if board.is_check() and (king := board.king(board.turn)) is not None
        else None,
        "result": outcome.result() if outcome else None,
        "termination": outcome.termination.name.lower().replace("_", " ") if outcome else None,
    }


class handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        self._json(200, turn({}))

    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 12_000:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a game request.")
            result = turn(payload)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            logging.exception("Champion request failed")
            self._json(503, {"error": "The champion is temporarily unavailable. Try again."})
        else:
            self._json(200, result)
