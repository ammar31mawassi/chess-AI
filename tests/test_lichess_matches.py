"""Offline match scheduling, quota and scoreboard tests; no real challenges."""

import json
import sys
from dataclasses import replace

import chess
import pytest

from chess_ai.online import matches
from chess_ai.online.lichess_api import LichessError
from chess_ai.online.lichess_bot import BotConfig, GameSession, GameStateError
from chess_ai.online.matches import MatchRunner, recent_bot_games, result_record


class API:
    def __init__(self):
        self.calls = []
        self.games = []
        self.runner = None

    def stream(self, path, stop):
        self.calls.append(path)
        if path.startswith("/api/games/"):
            yield from self.games
        else:
            yield {"id": "otherbot", "title": "BOT"}

    def get(self, path):
        return {"nowPlaying": []}

    def post(self, path, fields=None):
        self.calls.append((path, fields))
        if self.runner:
            self.runner.stop.set()
        return {"challenge": {"id": "game1234"}}


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return BotConfig("TestBot", "a" * 64, device="cpu", opponent_kind="bot")


def test_quota_includes_bot_aborts_and_ongoing_but_not_humans():
    api = API()
    api.games = [
        {
            "id": "botgame",
            "status": "aborted",
            "players": {
                "white": {"user": {"id": "testbot", "title": "BOT"}},
                "black": {"user": {"id": "other", "title": "BOT"}},
            },
        },
        {
            "id": "human",
            "players": {
                "white": {"user": {"id": "testbot", "title": "BOT"}},
                "black": {"user": {"id": "friend"}},
            },
        },
    ]
    assert recent_bot_games(api, "TestBot", 200000) == {"botgame"}
    assert "since=113600000&ongoing=true&finished=true" in api.calls[0]


def test_quota_stops_without_challenging(config, monkeypatch):
    api = API()
    runner = MatchRunner(config, api, None, 100)
    runner.ready.set()
    monkeypatch.setattr(matches, "recent_bot_games", lambda *args: {str(i) for i in range(100)})
    runner.schedule()
    assert runner.stop.is_set()
    assert "daily allowance exhausted" in runner.reason
    assert not api.calls


def test_scheduler_creates_one_casual_challenge_and_reserves(config, monkeypatch):
    api = API()
    runner = MatchRunner(config, api, None, 100)
    api.runner = runner
    runner.ready.set()
    monkeypatch.setattr(matches, "recent_bot_games", lambda *args: set())
    runner.schedule()
    path, fields = api.calls[-1]
    assert path == "/api/challenge/otherbot"
    assert fields["rated"] == "false"
    assert fields["clock.limit"] == "300"
    assert runner.pending[0] == "game1234"


def test_local_starts_protect_against_export_delay(config, monkeypatch):
    runner = MatchRunner(config, API(), None, 100)
    runner.starts = {str(i): 200000 for i in range(100)}
    runner.ready.set()
    monkeypatch.setattr(matches.time, "time", lambda: 200001)
    monkeypatch.setattr(matches, "recent_bot_games", lambda *args: set())
    runner.schedule()
    assert "allowance exhausted" in runner.reason


@pytest.mark.parametrize(
    ("color", "winner", "expected"),
    [
        (chess.WHITE, "white", "won"),
        (chess.BLACK, "white", "lost"),
        (chess.BLACK, "black", "won"),
        (chess.WHITE, None, "draw"),
    ],
)
def test_scoreboard_and_terminal_summary(config, color, winner, expected):
    session = GameSession(config, API(), None, "game1234")
    session.color = color
    session.full = {"white": {"id": "white"}, "black": {"id": "black"}}
    session.path.mkdir(parents=True)
    result = "1-0" if winner == "white" else "0-1" if winner else "1/2-1/2"
    (session.path / "state.json").write_text(
        json.dumps({"result": result, "moves": "", "status": "resign" if winner else "draw"})
    )
    assert result_record(session)["outcome"] == expected
    runner = MatchRunner(config, API(), None, 1)
    runner.game_finished(session)
    assert runner.stop.is_set()
    summary = runner.save_summary()
    assert summary[expected] == 1
    assert summary["stop_reason"] == "target reached"
    assert summary["training_data"] is False


def test_bot_opponent_enabled_only_by_explicit_mode(config):
    event = {
        "id": "game1234",
        "variant": {"key": "standard"},
        "rated": False,
        "white": {"id": "testbot", "title": "BOT"},
        "black": {"id": "otherbot", "title": "BOT"},
    }
    GameSession(config, API(), None, "game1234").full_event(event)
    with pytest.raises(GameStateError):
        GameSession(replace(config, opponent_kind="human"), API(), None, "game1234").full_event(
            event
        )


def test_incoming_bot_challenges_declined_in_batch_mode(config):
    api = API()
    runner = MatchRunner(config, api, None, 100)
    runner.challenge({"challenge": {"direction": "in", "id": "incoming"}})
    assert api.calls == [("/api/challenge/incoming/decline", {"reason": "later"})]
    runner.challenge({"challenge": {"direction": "out", "id": "outgoing"}})
    assert len(api.calls) == 1


@pytest.mark.parametrize("level", range(1, 9))
@pytest.mark.parametrize("bot_white", [True, False])
def test_builtin_computer_identity_and_summary(config, level, bot_white):
    config = replace(config, opponent_kind="ai", ai_level=level)
    own, computer = {"id": "testbot", "name": "TestBot"}, {"aiLevel": level}
    event = {
        "id": "computer1",
        "variant": {"key": "standard"},
        "rated": False,
        "white": own if bot_white else computer,
        "black": computer if bot_white else own,
        "clock": {"initial": 600000, "increment": 5000},
    }
    session = GameSession(config, API(), None, "computer1")
    session.full_event(event)
    # Inputs and non-AI opponent rules are preserved.
    assert computer == {"aiLevel": level}
    session.full_event(event)  # Reconnecting must keep the same saved binding.
    session.state_event(
        {
            "moves": "",
            "status": "resign",
            "winner": "black" if bot_white else "white",
            "wtime": 600000,
            "btime": 600000,
            "winc": 5000,
            "binc": 5000,
        },
        play=False,
    )
    record = result_record(session)
    assert record["opponent"] == f"Lichess computer level {level}"
    assert record["outcome"] == "lost"
    assert record["reason"] == "resignation"
    binding = json.loads((session.path / "binding.json").read_text())
    assert binding["opponent_ai_level"] == level
    runner = MatchRunner(config, API(), None, 1)
    runner.game_finished(session)
    summary = runner.save_summary()
    assert summary["mode"] == "ai" and summary["ai_level"] == level
    assert summary["lost"] == 1
    assert f"Lichess computer level {level}" in (session.path / "game.pgn").read_text()


@pytest.mark.parametrize(
    "opponent",
    [
        {"aiLevel": 2},
        {"aiLevel": "3"},
        {"aiLevel": True},
        {"id": "human"},
        {"id": "fakecomputer", "aiLevel": 3},
    ],
)
def test_computer_mode_rejects_wrong_level_or_registered_opponent(config, opponent):
    config = replace(config, opponent_kind="ai", ai_level=3)
    with pytest.raises(GameStateError):
        GameSession(config, API(), None, "computer1").full_event(
            {
                "id": "computer1",
                "variant": {"key": "standard"},
                "rated": False,
                "white": {"id": "testbot"},
                "black": opponent,
            }
        )


@pytest.mark.parametrize("kind", ["human", "bot"])
def test_other_modes_reject_builtin_computers(config, kind):
    with pytest.raises(GameStateError):
        GameSession(replace(config, opponent_kind=kind), API(), None, "computer1").full_event(
            {
                "id": "computer1",
                "variant": {"key": "standard"},
                "rated": False,
                "white": {"id": "testbot"},
                "black": {"aiLevel": 3},
            }
        )


def test_computer_creation_and_restart_do_not_duplicate_post(config, monkeypatch, capsys):
    class ComputerAPI(API):
        def post(self, path, fields=None):
            self.calls.append((path, fields))
            return {"id": "computer1", "fullId": "PRIVATE_GAME_ACCESS"}

    config = replace(config, opponent_kind="ai", ai_level=4)
    api = ComputerAPI()
    runner = MatchRunner(config, api, None, 1)

    # Replace only the game worker entry: all scheduling below is offline.
    def started(game_id):
        runner.active = game_id

    monkeypatch.setattr(runner, "start_game", started)
    runner.challenge_computer()
    runner.challenge_computer()  # Already active: recover, do not challenge again.
    assert runner.active == "computer1"
    assert api.calls == [
        (
            "/api/challenge/ai",
            {
                "level": "4",
                "clock.limit": "600",
                "clock.increment": "5",
                "color": "random",
                "variant": "standard",
            },
        )
    ]
    assert "PRIVATE_GAME_ACCESS" not in capsys.readouterr().out
    assert not runner.ledger_path.exists()  # Not recorded as an account-vs-account BOT game.


def test_computer_recovers_existing_game_before_creation(config, monkeypatch):
    api = API()
    monkeypatch.setattr(api, "get", lambda path: {"nowPlaying": [{"gameId": "existing1"}]})
    runner = MatchRunner(replace(config, opponent_kind="ai", ai_level=1), api, None, 1)
    started = []
    monkeypatch.setattr(runner, "start_game", started.append)
    runner.challenge_computer()
    assert started == ["existing1"] and not api.calls


def test_computer_ambiguous_post_stops_without_retry(config, monkeypatch):
    api = API()
    attempts = []

    def fail(path, fields):
        attempts.append(path)
        raise LichessError("Connection to Lichess failed")

    monkeypatch.setattr(api, "post", fail)
    runner = MatchRunner(replace(config, opponent_kind="ai", ai_level=1), api, None, 1)
    runner.ready.set()
    runner.schedule()
    assert attempts == ["/api/challenge/ai"]
    assert runner.stop.is_set()


@pytest.mark.parametrize(
    "arguments",
    [
        ["lvl"],
        ["lvl", "0"],
        ["lvl", "9"],
        ["lvl", "three"],
        ["human", "3"],
        ["100", "3"],
    ],
)
def test_invalid_level_syntax_fails_before_credentials_or_network(monkeypatch, arguments):
    def forbidden():
        pytest.fail("Invalid input must not load credentials")

    monkeypatch.setattr(matches, "load_token", forbidden)
    monkeypatch.setattr(sys, "argv", ["chess-bot", *arguments])
    with pytest.raises(SystemExit) as error:
        matches.main()
    assert error.value.code == 2
