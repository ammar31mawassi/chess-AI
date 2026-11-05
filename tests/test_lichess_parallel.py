"""Offline concurrent-game, restart, quota and shared-model tests; no live play."""

import io
import json
import sys
import threading
import time
from collections import Counter

import chess
import pytest

from chess_ai.online import matches, parallel_matches
from chess_ai.online.lichess_api import LichessAPI, LichessError
from chess_ai.online.lichess_bot import BotConfig
from chess_ai.online.parallel_matches import ParallelMatchRunner, SerializedAgent
from chess_ai.storage.artifacts import atomic_json, run_lock


class Agent:
    def __init__(self):
        self.clocks = []

    def choose_timed(self, board, remaining_ms, increment_ms):
        self.clocks.append((remaining_ms, increment_ms))
        return next(iter(board.legal_moves))


class API:
    def __init__(self):
        self.calls = []
        self.ongoing = []
        self.barrier = None
        self.reject = None

    def get(self, path):
        return {"nowPlaying": list(self.ongoing)}

    def post(self, path, fields=None):
        self.calls.append((path, fields))
        if self.reject:
            raise LichessError("Mock request rejected", status=self.reject)
        return {"challenge": {"id": f"game{len(self.calls):04d}"}}

    def stream(self, path, stop, *, connected=None):
        if connected is not None:
            connected()
        if path.startswith("/api/bot/online"):
            yield from ({"id": f"bot{i}"} for i in range(20))
        elif path.startswith("/api/bot/game/stream/"):
            game_id = path.rsplit("/", 1)[1]
            if self.barrier:
                self.barrier.wait(timeout=5)
            yield {
                "type": "gameFull",
                "id": game_id,
                "variant": {"key": "standard"},
                "rated": False,
                "white": {"id": "testbot", "title": "BOT"},
                "black": {"id": "opponent", "title": "BOT"},
                "clock": {"initial": 300000, "increment": 3000},
                "state": {
                    "type": "gameState",
                    "status": "mate",
                    "winner": "black",
                    "moves": "f2f3 e7e5 g2g4 d8h4",
                    "wtime": 300000,
                    "btime": 300000,
                    "winc": 3000,
                    "binc": 3000,
                },
            }


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BotConfig("TestBot", "a" * 64, device="cpu", opponent_kind="bot")
    result = ParallelMatchRunner(config, API(), Agent(), 100)
    monkeypatch.setattr(parallel_matches, "recent_bot_games", lambda *args: set())
    return result


def record(game_id):
    return {
        "id": game_id,
        "outcome": "won",
        "opponent": "opponent",
        "color": "white",
        "result": "1-0",
        "reason": "checkmate",
        "url": "https://lichess.org/" + game_id,
    }


def test_default_fills_five_slots_and_pending_offers_reserve_target(runner):
    assert runner.concurrency == 5
    for _ in range(8):
        runner.tick()
    assert len(runner.api.calls) == 5 and len(runner.reservations) == 5
    assert len({call[0] for call in runner.api.calls}) == 5
    small = ParallelMatchRunner(runner.config, API(), Agent(), 3, concurrency=10)
    for _ in range(8):
        small.tick()
    assert small.concurrency == 3 and len(small.api.calls) == 3


def test_quota_includes_pending_reservations_and_local_starts(runner, monkeypatch):
    monkeypatch.setattr(
        parallel_matches, "recent_bot_games", lambda *args: {f"old{i}" for i in range(97)}
    )
    runner.starts["localdelayed"] = time.time()
    for _ in range(8):
        runner.tick()
    assert len(runner.api.calls) == 2
    assert len(runner.reservations) == 2


def test_five_game_workers_finish_independently_and_count_once(runner):
    runner.target = 5
    runner.api.barrier = threading.Barrier(6)
    for i in range(5):
        runner.start_game(f"active{i}")
        runner.start_game(f"active{i}")  # Replayed gameStart must not duplicate a stream.
    with runner.guard:
        threads = list(runner.workers.values())
        assert len(threads) == 5
    runner.api.barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(runner.results) == 5 and not runner.workers and runner.stop.is_set()
    assert Counter(row["outcome"] for row in runner.results) == {"lost": 5}
    summary = runner.save_summary()
    assert summary["version"] == 2 and summary["stop_reason"] == "target reached"
    for i in range(5):
        saved = json.loads((runner.config.output_dir / f"active{i}" / "state.json").read_text())
        assert saved["complete"] and saved["result"] == "0-1"
    assert not runner.api.calls  # Terminal recovery/test streams must never submit moves.


def test_quota_stop_drains_existing_games_instead_of_stopping_workers(runner, monkeypatch):
    monkeypatch.setattr(
        parallel_matches, "recent_bot_games", lambda *args: {str(i) for i in range(100)}
    )
    runner.workers["alreadyactive"] = threading.Thread()
    runner.tick()
    assert runner.no_new_games and not runner.stop.is_set()
    assert not runner.api.calls
    runner.release_game("alreadyactive")
    assert runner.stop.is_set()


def test_game_start_racing_post_does_not_leave_a_double_reservation(runner, monkeypatch):
    def post(path, fields):
        runner.assigned.add("racedgame")
        return {"challenge": {"id": "racedgame"}}

    monkeypatch.setattr(runner.api, "post", post)
    runner.tick()
    assert not runner.reservations


def test_startup_waits_for_legacy_outgoing_offers_before_new_challenges(runner):
    assert not runner.ready.is_set()
    runner.account_connected()
    assert runner.ready.is_set() and runner.startup_until > time.time()
    runner.tick()
    assert not runner.api.calls
    runner.handle_event({"type": "challenge", "challenge": {"id": "oldoffer", "direction": "out"}})
    assert "oldoffer" in runner.reservations
    runner.startup_until = 0
    for _ in range(8):
        runner.tick()
    assert len(runner.api.calls) == 4 and len(runner.reservations) == 5


def test_declined_or_expired_reservation_frees_its_slot(runner):
    runner.reservations["oldoffer"] = time.time() + 35
    runner.handle_event({"type": "challengeDeclined", "challenge": {"id": "oldoffer"}})
    assert not runner.reservations
    runner.reservations["expired1"] = time.time() - 1
    runner.tick()
    assert "expired1" not in runner.reservations
    assert len(runner.reservations) == 1


def test_session_resumes_results_target_pending_and_updated_concurrency(runner):
    runner.results = [record("oldgame")]
    runner.assigned = {"oldgame", "livegame"}
    runner.reservations["offered"] = time.time() + 35
    runner.attempted["opponent"] = time.time()
    runner.save_summary()
    resumed = ParallelMatchRunner(runner.config, API(), Agent(), 100, concurrency=2)
    resumed.load_session()
    assert resumed.summary_path == runner.summary_path
    assert resumed.results == runner.results and resumed.assigned == runner.assigned
    assert resumed.reservations == runner.reservations and resumed.concurrency == 2
    assert resumed.attempted == runner.attempted


def test_legacy_summary_migrates_without_changing_the_old_file(runner):
    old = runner.config.output_dir / "sessions" / "20261105_000000_000000.json"
    atomic_json(
        old,
        {
            "format": parallel_matches.SESSION_FORMAT,
            "version": 1,
            "mode": "bot",
            "requested": 100,
            "completed": 1,
            "games": [record("oldgame")],
        },
    )
    atomic_json(
        runner.config.output_dir / "oldgame" / "binding.json",
        {"username": runner.config.username, "checkpoint_sha256": runner.config.checkpoint_sha256},
    )
    before = old.read_bytes()
    runner.load_session()
    assert len(runner.results) == 1 and runner.summary_path != old
    assert runner.resumed_from == str(old)
    assert old.read_bytes() == before
    runner.save_summary()
    assert json.loads(runner.summary_path.read_text())["version"] == 2


def test_completed_latest_session_starts_fresh_without_resurrecting_older_batches(runner):
    runner.target = 2
    runner.results = [record("game0001"), record("game0002")]
    runner.save_summary()
    fresh = ParallelMatchRunner(runner.config, API(), Agent(), 2)
    fresh.load_session()
    assert fresh.results == [] and fresh.summary_path != runner.summary_path


def test_offline_finished_assigned_game_is_archived_and_counted_once(runner):
    runner.assigned.add("offline1")
    runner.recover([])
    runner.recover([])
    assert [row["id"] for row in runner.results] == ["offline1"]
    assert not runner.api.calls


def test_legacy_terminal_state_missing_from_summary_is_counted_on_migration(runner):
    old = runner.config.output_dir / "sessions" / "20261105_000000_000000.json"
    atomic_json(runner.config.output_dir / "orphan01" / "state.json", {"complete": True})
    atomic_json(
        old,
        {
            "format": parallel_matches.SESSION_FORMAT,
            "version": 1,
            "mode": "bot",
            "requested": 100,
            "completed": 0,
            "games": [],
        },
    )
    runner.load_session()
    runner.recover([])
    assert [row["id"] for row in runner.results] == ["orphan01"]
    runner.recover([])
    assert len(runner.results) == 1


def test_fresh_batch_does_not_absorb_unrelated_old_journals(runner):
    atomic_json(runner.config.output_dir / "oldother" / "state.json", {"complete": False})
    runner.recover([])
    assert not runner.results


def test_rate_limit_halts_matchmaking_but_allows_existing_workers_to_drain(runner, monkeypatch):
    runner.api.reject = 429
    runner.ready.set()
    runner.workers["livegame"] = threading.Thread()
    waits = []

    def wait(seconds):
        assert runner.workers and not runner.stop.is_set()
        assert runner.no_new_games
        waits.append(seconds)
        runner.stop.set()
        return True

    monkeypatch.setattr(runner.stop, "wait", wait)
    runner.schedule()
    assert waits == [60] and len(runner.api.calls) == 1
    assert "HTTP 429" in runner.reason


def test_shared_model_has_one_search_at_a_time_and_accounts_for_queue_delay():
    class Model:
        def __init__(self):
            self.active = 0
            self.maximum = 0
            self.remaining = []

        def choose_timed(self, board, remaining_ms, increment_ms):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.remaining.append(remaining_ms)
            time.sleep(0.03)
            self.active -= 1
            return next(iter(board.legal_moves))

    model = Model()
    shared = SerializedAgent(model)
    threads = [
        threading.Thread(target=shared.choose_timed, args=(chess.Board(), 300000, 3000))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert model.maximum == 1 and len(model.remaining) == 3
    assert min(model.remaining) < 300000
    shared.lock.acquire()
    try:
        assert shared.choose_timed(chess.Board(), 1, 0) in chess.Board().legal_moves
        assert len(model.remaining) == 3
    finally:
        shared.lock.release()


def test_short_api_requests_are_serialized_through_response_reading():
    api = LichessAPI("mock_token_never_sent")
    active = 0
    maximum = 0
    guard = threading.Lock()

    class Response(io.BytesIO):
        def read(self, *args):
            time.sleep(0.02)
            return super().read(*args)

        def __exit__(self, *args):
            nonlocal active
            with guard:
                active -= 1
            return super().__exit__(*args)

    class Opener:
        def open(self, request, timeout):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            return Response(b'{"ok":true}')

    api._opener = Opener()
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(api.get("/api/mock"))) for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert results == [{"ok": True}] * 3 and maximum == 1 and active == 0


def test_lock_collision_never_writes_session_or_starts_a_game(runner):
    with run_lock(runner.config.output_dir), pytest.raises(RuntimeError, match="locked"):
        runner.run()
    assert not runner.summary_path.exists() and not runner.api.calls


def test_run_recovers_five_existing_games_even_when_limit_is_lower(runner):
    runner.target = 5
    runner.concurrency = 2
    runner.api.ongoing = [{"gameId": f"recover{i}"} for i in range(5)]
    runner.run()
    assert len(runner.results) == 5 and not runner.api.calls
    assert not runner.workers and runner.stop.is_set()
    saved = json.loads(runner.summary_path.read_text())
    assert saved["completed"] == 5 and saved["concurrency"] == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["100", "--concurrency", "0"],
        ["100", "--concurrency", "-2"],
        ["human", "--concurrency", "5"],
        ["lvl", "3", "--concurrency", "5"],
    ],
)
def test_invalid_concurrency_rejected_before_loading_credentials(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["chess-bot", *arguments])
    monkeypatch.setattr(
        matches, "load_token", lambda: pytest.fail("Must reject before credentials")
    )
    with pytest.raises(SystemExit) as exc:
        matches.main()
    assert exc.value.code == 2
