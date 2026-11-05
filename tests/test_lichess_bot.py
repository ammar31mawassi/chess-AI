"""Offline mocked Bot API tests; these never contact Lichess or start live games."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import urllib.error
from dataclasses import replace
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.model.checkpoint import save_checkpoint
from chess_ai.model.policy_value_net import PolicyValueNet
from chess_ai.online import credentials
from chess_ai.online.lichess_api import LichessAPI, LichessError, _NoRedirect
from chess_ai.online.lichess_bot import (
    BotConfig,
    BotRunner,
    ClockedPUCT,
    GameSession,
    GameStateError,
    challenge_reason,
    verify_account,
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return BotConfig("TestBot", "a" * 64, device="cpu")


def challenge():
    return {
        "id": "abcdef12",
        "direction": "in",
        "rated": False,
        "variant": {"key": "standard"},
        "challenger": {"id": "friend"},
        "timeControl": {"type": "clock", "limit": 300, "increment": 3},
    }


def state(moves="", status="started", **extra):
    return {
        "type": "gameState",
        "moves": moves,
        "status": status,
        "wtime": 300000,
        "btime": 300000,
        "winc": 3000,
        "binc": 3000,
        **extra,
    }


def full(*, bot_white=True, moves="", status="started", **extra):
    white, black = {"id": "testbot", "name": "TestBot"}, {"id": "friend", "name": "Friend"}
    return {
        "type": "gameFull",
        "id": "abcdefgh",
        "variant": {"key": "standard"},
        "rated": False,
        "white": white if bot_white else black,
        "black": black if bot_white else white,
        "initialFen": "startpos",
        "clock": {"initial": 300000, "increment": 3000},
        "state": state(moves, status),
        **extra,
    }


class FakeAPI(LichessAPI):
    def __init__(self):
        self.calls = []
        self.fail = False
        self.profile = {"id": "testbot", "username": "TestBot", "title": "BOT", "count": {"all": 0}}
        self.permissions = {"bot:play"}

    def post(self, path, fields=None):
        self.calls.append((path, fields))
        if self.fail:
            self.fail = False
            raise LichessError("Ambiguous connection failure")
        return {"ok": True}

    def account(self):
        return self.profile

    def scopes(self):
        return self.permissions


class FakeAgent:
    def __init__(self):
        self.calls = []

    def choose_timed(self, board, remaining_ms, increment_ms):
        self.calls.append((board.copy(stack=True), remaining_ms, increment_ms))
        return board.parse_uci("e2e4" if not board.move_stack else "g1f3")


def test_challenge_filter_accepts_only_supported_casual_human_clocks(config):
    assert challenge_reason(config, challenge(), busy=False) is None
    changes = [
        ({"rated": True}, "casual"),
        ({"variant": {"key": "atomic"}}, "variant"),
        ({"challenger": {"id": "other", "title": "BOT"}}, "noBot"),
        ({"timeControl": {"type": "correspondence"}}, "timeControl"),
        ({"timeControl": {"type": "clock", "limit": 60, "increment": 3}}, "tooFast"),
        ({"timeControl": {"type": "clock", "limit": 300, "increment": 0}}, "tooFast"),
        ({"timeControl": {"type": "clock", "limit": 7200, "increment": 3}}, "tooSlow"),
        ({"initialFen": "custom"}, "standard"),
    ]
    for change, expected in changes:
        assert challenge_reason(config, {**challenge(), **change}, busy=False) == expected
    assert challenge_reason(config, challenge(), busy=True) == "later"


def test_pending_challenge_reserves_the_single_slot(config):
    api = FakeAPI()
    runner = BotRunner(config, api, FakeAgent())
    runner.challenge({"challenge": challenge()})
    second = {**challenge(), "id": "abcdef13"}
    runner.challenge({"challenge": second})
    assert api.calls == [
        ("/api/challenge/abcdef12/accept", None),
        ("/api/challenge/abcdef13/decline", {"reason": "later"}),
    ]
    assert runner.pending[0] == "abcdef12"


def test_replay_deduplicates_moves_and_preserves_history(config):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full())
    assert not session.state_event(state())
    assert not session.state_event(state())
    assert len(api.calls) == 1
    session.state_event(state("e2e4"))  # Opponent's turn.
    session.state_event(state("e2e4 e7e5"))
    assert len(api.calls) == 2
    assert api.calls[-1][0].endswith("/move/g1f3")
    assert [m.uci() for m in agent.calls[-1][0].move_stack] == ["e2e4", "e7e5"]
    session.state_event(state())  # Delayed stale event must not resubmit a move.
    assert len(api.calls) == 2
    saved = json.loads((session.path / "state.json").read_text())
    assert saved["moves"] == "e2e4 e7e5" and saved["training_data"] is False


def test_ambiguous_post_reconnect_uses_server_history(config):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full())
    api.fail = True
    with pytest.raises(LichessError):
        session.state_event(state())
    session.full_event(full(moves="e2e4"))
    session.state_event(state("e2e4"))
    assert len(api.calls) == 1  # Move had reached server; no blind retry.


def test_restart_recovers_without_replaying_acknowledged_move(config):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full())
    session.state_event(state())
    restarted = GameSession(config, api, agent, "abcdefgh")
    restarted.full_event(full(moves="e2e4"))
    restarted.state_event(state("e2e4"))
    assert len(api.calls) == 1


def test_game_finished_while_offline_is_recovered_without_playing(config, monkeypatch):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full())
    session.state_event(state("f2f3"), play=False)
    final = full(moves="f2f3 e7e5 g2g4 d8h4", status="mate")
    final["state"]["winner"] = "black"

    def stream(path, stop):
        yield final

    monkeypatch.setattr(api, "stream", stream)
    BotRunner(config, api, agent).recover_finished(set())
    saved = json.loads((session.path / "state.json").read_text())
    assert saved["complete"] and saved["result"] == "0-1"
    assert not api.calls and not agent.calls


def test_terminal_server_result_is_archived_without_more_moves(config):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full(bot_white=False))
    assert session.state_event(state("f2f3 e7e5 g2g4 d8h4", "mate", winner="black"))
    game = chess.pgn.read_game(io.StringIO((session.path / "game.pgn").read_text()))
    assert game.headers["Result"] == "0-1"
    assert game.end().board().is_checkmate()
    assert not api.calls and not agent.calls


def test_draw_claim_uses_draw_offer_endpoint_not_disconnection_claim(config):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full())
    cycle = "g1f3 g8f6 f3g1 f6g8"
    session.state_event(state(cycle + " " + cycle))
    assert api.calls == [("/api/bot/game/abcdefgh/draw/yes", None)]
    assert not agent.calls


def test_intended_repetition_move_is_played_before_claim(config):
    api, agent = FakeAPI(), FakeAgent()
    session = GameSession(config, api, agent, "abcdefgh")
    session.full_event(full(bot_white=False))
    moves = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1"
    session.state_event(state(moves))
    assert api.calls[0][0].endswith("/move/f6g8")
    assert not agent.calls


def test_invalid_identity_variant_and_history_never_send_moves(config):
    api = FakeAPI()
    session = GameSession(config, api, FakeAgent(), "abcdefgh")
    for change in ({"variant": {"key": "chess960"}}, {"rated": True}, {"id": "otherid"}):
        with pytest.raises(GameStateError):
            session.full_event(full(**change))
    session.full_event(full())
    with pytest.raises(GameStateError, match="illegal move"):
        session.state_event(state("e2e5"))
    assert not api.calls


def test_account_pinning_and_bot_scope(config):
    api = FakeAPI()
    assert verify_account(config, api)["title"] == "BOT"
    api.profile["id"] = "someoneelse"
    with pytest.raises(LichessError, match="different account"):
        verify_account(config, api)
    api.profile["id"] = "testbot"
    api.permissions = set()
    with pytest.raises(LichessError, match="bot:play"):
        verify_account(config, api)
    api.permissions = {"bot:play"}
    api.profile["title"] = None
    with pytest.raises(LichessError, match="upgraded"):
        verify_account(config, api)
    assert verify_account(config, api, require_bot=False)["title"] is None


@pytest.mark.parametrize(
    "confirm,games,allowed", [(False, 0, False), (True, 1, False), (True, 0, True)]
)
def test_upgrade_is_separate_explicit_and_zero_game_only(
    config, tmp_path, monkeypatch, capsys, confirm, games, allowed
):
    path = Path(__file__).resolve().parents[1] / "scripts/lichess_bot.py"
    spec = importlib.util.spec_from_file_location("lichess_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = FakeAPI()
    api.profile["title"] = None
    api.profile["count"]["all"] = games
    monkeypatch.setattr(module, "load_token", lambda: "fake_secret_token_5112026")
    monkeypatch.setattr(module, "LichessAPI", lambda token: api)
    monkeypatch.setattr(module.BotConfig, "read", lambda path: config)
    monkeypatch.setattr(module.BotConfig, "checkpoint", lambda self: Path("frozen.pt"))
    monkeypatch.setattr(api, "get", lambda path: {"nowPlaying": []})
    original_post = api.post

    def post(path, fields=None):
        result = original_post(path, fields)
        api.profile["title"] = "BOT"
        return result

    monkeypatch.setattr(api, "post", post)
    monkeypatch.setattr(
        sys, "argv", ["lichess_bot.py", "upgrade"] + (["--confirm-bot-upgrade"] if confirm else [])
    )
    if allowed:
        module.main()
        assert api.calls == [("/api/bot/account/upgrade", None)]
        assert json.loads(capsys.readouterr().out)["ready"]
    else:
        with pytest.raises(ValueError):
            module.main()
        assert not api.calls


def test_clocked_agent_matches_production_with_ample_budget(tmp_path):
    torch.manual_seed(5112026)
    path = save_checkpoint(tmp_path / "model.pt", PolicyValueNet(channels=8, residual_blocks=0))
    online = ClockedPUCT(path, device="cpu", seed=5112026, max_seconds=20)
    production = NeuralMCTSAgent(path, device="cpu", simulations=32)
    board = chess.Board()
    assert online.choose_timed(board, 10000000, 3000) == production.choose_move(board)
    assert board == chess.Board()
    assert online.choose_timed(board, 200, 0) in board.legal_moves


def test_config_requires_evaluation_directory_and_valid_ranges(config):
    with pytest.raises(ValueError, match="evaluation"):
        replace(config, output_dir=Path("data/datasets/online"))
    with pytest.raises(ValueError, match="increment"):
        replace(config, minimum_increment=0)


def test_http_errors_redact_tokens_and_429_imposes_cooldown(monkeypatch):
    token = "fake_secret_token_5112026"
    api = LichessAPI(token)
    calls = []

    def reject(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, token, {}, io.BytesIO(token.encode()))

    monkeypatch.setattr(api._opener, "open", reject)
    with pytest.raises(LichessError) as error:
        api.account()
    assert token not in str(error.value) and error.value.status == 429
    with pytest.raises(LichessError, match="cooldown"):
        api.account()
    assert len(calls) == 1
    with pytest.raises(LichessError, match="redirect"):
        _NoRedirect().redirect_request(None, None, 302, "", None, "https://evil.invalid")


def test_stream_handles_blank_keepalive_and_closes_response(monkeypatch):
    api = LichessAPI("fake_secret_token_5112026")
    response = io.BytesIO(b'\n{"type":"gameStart","game":{"id":"abcdefgh"}}\n')
    monkeypatch.setattr(api, "_open", lambda path: response)
    assert list(api.stream("/api/stream/event", threading.Event())) == [
        {"type": "gameStart", "game": {"id": "abcdefgh"}}
    ]
    assert response.closed


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI")
def test_dpapi_roundtrip_stores_only_ciphertext(tmp_path, monkeypatch):
    path = tmp_path / "credential.dpapi"
    monkeypatch.setattr(credentials, "credential_path", lambda: path)
    monkeypatch.delenv("CHESS_AI_LICHESS_TOKEN", raising=False)
    token = "fake_secret_token_5112026"
    credentials.store_token(token)
    assert token.encode() not in path.read_bytes()
    assert credentials.load_token() == token
    monkeypatch.setenv("CHESS_AI_LICHESS_TOKEN", "different_fake_token_5112026")
    assert credentials.load_token() == "different_fake_token_5112026"
