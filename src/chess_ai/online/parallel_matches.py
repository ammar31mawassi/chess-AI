"""Resumable bounded BOT batches with concurrent games and one shared model.

All live activity is opt-in from the user's command. Each game keeps an independent
GameSession. Model search and short API requests are serialized, not duplicated.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess

from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.online.lichess_api import LichessAPI, LichessError
from chess_ai.online.lichess_bot import BotConfig, BotRunner, GameSession, MoveAgent, identifier
from chess_ai.online.matches import MatchRunner, recent_bot_games, result_record
from chess_ai.storage.artifacts import atomic_json, run_lock

SESSION_FORMAT = "self-improving-chess-ai.lichess-session"


class SerializedAgent:
    """One CUDA model, with waiting time deducted from the requesting game's clock."""

    def __init__(self, agent: MoveAgent) -> None:
        self.agent = agent
        self.lock = threading.Lock()

    def choose_timed(self, board: chess.Board, remaining_ms: int, increment_ms: int) -> chess.Move:
        before = time.monotonic()
        if not self.lock.acquire(timeout=max(0.0, (remaining_ms - 500) / 1000)):
            return min(board.legal_moves, key=MoveEncoder().encode)
        try:
            elapsed_ms = int((time.monotonic() - before) * 1000)
            return self.agent.choose_timed(board, max(0, remaining_ms - elapsed_ms), increment_ms)
        finally:
            self.lock.release()


class ParallelMatchRunner(MatchRunner):
    """Five games by default; legacy human/computer service remains single-game."""

    def __init__(
        self,
        config: BotConfig,
        api: LichessAPI,
        agent: MoveAgent,
        target: int,
        concurrency: int = 5,
    ) -> None:
        if config.opponent_kind != "bot" or not 1 <= target <= 100 or concurrency < 1:
            raise ValueError("Use bot mode, 1-100 games and positive concurrency")
        # MatchRunner's constructor only establishes local paths/state; it does
        # not load or write journals while the old process is still running.
        BotRunner.__init__(self, config, api, SerializedAgent(agent))
        self.guard = threading.RLock()
        self.target = target
        self.concurrency = min(target, concurrency)
        self.results: list[dict[str, Any]] = []
        self.workers: dict[str, threading.Thread] = {}
        self.assigned: set[str] = set()
        self.reservations: dict[str, float] = {}
        self.attempted: dict[str, float] = {}
        self.starts: dict[str, float] = {}
        self.reason = "running"
        self.no_new_games = False
        self.startup_until = 0.0
        self.resumed_from: str | None = None
        self.ledger_path = config.output_dir / "bot_starts.json"
        # Reuse the legacy filename convention so latest-session migration is simple.
        self.summary_path = (
            config.output_dir
            / "sessions"
            / (datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f") + ".json")
        )

    def save_summary(self) -> dict[str, Any]:
        with self.guard:
            counts = Counter(record["outcome"] for record in self.results)
            summary = {
                "format": SESSION_FORMAT,
                "version": 2,
                "mode": "bot",
                "username": self.config.username,
                "checkpoint_sha256": self.config.checkpoint_sha256,
                "requested": self.target,
                "concurrency": self.concurrency,
                "completed": len(self.results),
                "won": counts["won"],
                "lost": counts["lost"],
                "draw": counts["draw"],
                "unfinished": counts["unfinished"],
                "points": counts["won"] + counts["draw"] / 2,
                "stop_reason": self.reason,
                "games": list(self.results),
                "assigned_ids": sorted(self.assigned),
                "pending": dict(self.reservations),
                "attempted_opponents": dict(self.attempted),
                "resumed_from": self.resumed_from,
                "training_data": False,
            }
            atomic_json(self.summary_path, summary)
            return summary

    def load_session(self) -> None:
        """Called only after taking the account run lock; preserve legacy files."""
        if self.ledger_path.exists():
            raw_starts = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            self.starts = {identifier(key): float(value) for key, value in raw_starts.items()}
        candidates = []
        for path in sorted((self.config.output_dir / "sessions").glob("*.json"), reverse=True):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if (
                raw.get("format") == SESSION_FORMAT
                and raw.get("mode") == "bot"
                and raw.get("requested") == self.target
            ):
                candidates.append((path, raw))
        if not candidates:
            return
        path, saved = candidates[0]
        # A completed latest batch means a new invocation requests a new batch.
        # Never resurrect an older interrupted legacy batch after this one finishes.
        if saved["completed"] >= self.target:
            return
        if saved.get("version") not in {1, 2}:
            raise LichessError("Unsupported bot session version")
        records = saved["games"]
        ids = [identifier(record["id"]) for record in records]
        if len(ids) != len(set(ids)) or len(ids) != saved["completed"]:
            raise LichessError("Saved bot session has duplicate/missing results")
        if saved["version"] == 2:
            if (
                saved.get("username", "").casefold() != self.config.username.casefold()
                or saved.get("checkpoint_sha256") != self.config.checkpoint_sha256
            ):
                raise LichessError("Saved batch belongs to another account/champion")
            self.summary_path = path
            self.assigned = {identifier(value) for value in saved["assigned_ids"]} | set(ids)
            self.reservations = {
                identifier(key): float(value) for key, value in saved.get("pending", {}).items()
            }
            self.attempted = {
                identifier(key): float(value)
                for key, value in saved.get("attempted_opponents", {}).items()
            }
            self.resumed_from = saved.get("resumed_from")
        else:
            # Version 1 did not pin identity in the summary. Its per-game bindings do.
            for game_id in ids:
                binding = json.loads(
                    (self.config.output_dir / game_id / "binding.json").read_text(encoding="utf-8")
                )
                if (
                    binding.get("username", "").casefold() != self.config.username.casefold()
                    or binding.get("checkpoint_sha256") != self.config.checkpoint_sha256
                ):
                    raise LichessError("Legacy batch game belongs to another account/champion")
            self.assigned = set(ids)
            self.resumed_from = str(path)
        self.results = records
        print(
            f"Resuming batch: {len(records)}/{self.target} finished; concurrency {self.concurrency}.",
            flush=True,
        )

    def start_game(self, game_id: str) -> None:
        game_id = identifier(game_id)
        with self.guard:
            self.reservations.pop(game_id, None)
            if game_id in self.workers or any(row["id"] == game_id for row in self.results):
                return
            self.assigned.add(game_id)
            self.starts.setdefault(game_id, time.time())
            atomic_json(self.ledger_path, self.starts)
            worker = threading.Thread(target=self._game, args=(game_id,), daemon=True)
            self.workers[game_id] = worker
            # Persist ownership before a worker submits its first move.
            self.save_summary()
            worker.start()
            if len(self.workers) > self.concurrency:
                # Existing games must be served even after reducing concurrency.
                self.log("recovering_above_new_concurrency", active=len(self.workers))

    def game_finished(self, session: GameSession) -> None:
        record = result_record(session)
        with self.guard:
            if any(row["id"] == session.game_id for row in self.results):
                return
            self.assigned.add(session.game_id)
            self.results.append(record)
            print(
                f"Game {len(self.results)}/{self.target}: {record['outcome'].upper()} "
                f"{record['result']} vs {record['opponent']} as {record['color']} "
                f"— {record['reason']} | {record['url']}",
                flush=True,
            )
            self.save_summary()

    def release_game(self, game_id: str) -> None:
        with self.guard:
            self.workers.pop(game_id, None)
            self.maybe_finish()

    def maybe_finish(self) -> None:
        """Never stop a live game merely because quota or the batch target was reached."""
        with self.guard:
            if len(self.results) >= self.target:
                self.no_new_games = True
                self.reason = "target reached"
            if self.no_new_games and not self.workers and not self.reservations:
                self.stop.set()
            self.save_summary()

    def recover(self, ongoing: list[dict[str, Any]]) -> None:
        """Start live streams first; then reconcile games completed while offline."""
        active_ids = {identifier(game["gameId"]) for game in ongoing}
        for game_id in active_ids:
            self.start_game(game_id)
        completed_ids = {record["id"] for record in self.results}
        recover_ids = self.assigned - active_ids - completed_ids
        # Version 1 did not save active IDs. Its journal timestamps identify
        # games from that invocation, including a crash after terminal state was
        # saved but before its result reached the old summary. Do not count old
        # orphans from unrelated batches in a fresh version-2 session.
        if self.resumed_from:
            legacy_path = Path(self.resumed_from)
            try:
                since = (
                    datetime.strptime(legacy_path.stem, "%Y%m%d_%H%M%S_%f")
                    .replace(tzinfo=UTC)
                    .timestamp()
                )
            except ValueError:
                raise LichessError("Cannot identify the legacy batch's journal interval") from None
            until = legacy_path.stat().st_mtime + 1
            for path in self.config.output_dir.glob("*/state.json"):
                if since <= path.stat().st_mtime <= until and path.parent.name not in (
                    active_ids | completed_ids
                ):
                    recover_ids.add(identifier(path.parent.name))
        for game_id in sorted(recover_ids):
            session = GameSession(self.config, self.api, self.agent, game_id)
            with closing(self.api.stream(f"/api/bot/game/stream/{game_id}", self.stop)) as stream:
                full = next(stream, None)
                if full is None or full.get("type") != "gameFull":
                    raise LichessError("Could not recover interrupted bot game")
                own = self.config.username.casefold()
                opponent = (
                    full["black"]
                    if str(full["white"].get("id", "")).casefold() == own
                    else full["white"]
                )
                if opponent.get("title") != "BOT":
                    if game_id in self.assigned:
                        raise LichessError("Batch contains an unexpected non-BOT game")
                    continue  # A journal from the separate human/computer modes.
                session.full_event(full)
                done = session.state_event(full["state"], play=False)
            if done:
                self.game_finished(session)
            else:
                self.start_game(game_id)
        # Pending challenges persist across restart and reserve their allowance.
        for game_id in active_ids:
            self.reservations.pop(game_id, None)

    def tick(self) -> float:
        """One serialized matchmaking decision, separate from game workers."""
        ongoing = self.api.get("/api/account/playing").get("nowPlaying", [])
        for game in ongoing:
            self.start_game(game["gameId"])
        now = time.time()
        with self.guard:
            if now < self.startup_until:
                # The account stream replays outstanding outgoing offers. Old
                # version-1 processes did not persist those reservations. Serve
                # ongoing games now, but allow offers to replay/expire before
                # spending any more slots or daily allowance.
                self.maybe_finish()
                return 2
            for game_id, deadline in list(self.reservations.items()):
                if deadline <= now:
                    # Real-time challenges expire after 20s. We reserve for 35s
                    # and reconcile current games before releasing the slot.
                    self.reservations.pop(game_id)
            if len(self.results) + len(self.workers) + len(self.reservations) >= self.target:
                self.maybe_finish()
                return 2
            if self.no_new_games or len(self.workers) + len(self.reservations) >= self.concurrency:
                self.maybe_finish()
                return 2
        recent = recent_bot_games(self.api, self.config.username, now)
        with self.guard:
            used = (
                recent
                | {key for key, value in self.starts.items() if value > now - 86400}
                | set(self.workers)
            )
            reserved = set(self.reservations) - used
            remaining = 100 - len(used) - len(reserved)
            if remaining <= 0:
                if self.reservations:
                    return 2  # An expiring offer might free quota without starting a game.
                self.no_new_games = True
                self.reason = "daily allowance exhausted (conservative rolling 24 hours)"
                self.maybe_finish()
                return 2
        bots = []
        with closing(self.api.stream("/api/bot/online?nb=512", self.stop)) as stream:
            for bot in stream:
                name = identifier(bot["id"])
                if (
                    name.casefold() != self.config.username.casefold()
                    and now - self.attempted.get(name, 0) >= 1800
                ):
                    bots.append(name)
        random.Random(f"{self.config.seed}:{len(self.attempted)}").shuffle(bots)
        if not bots:
            self.log(
                "waiting_for_eligible_bot", active=len(self.workers), concurrency=self.concurrency
            )
            return 30
        opponent = bots[0]
        with self.guard:
            # Reserve before POST; the gameStart event can race its response.
            self.reservations["posting"] = now + 35
            self.attempted[opponent] = now
            self.save_summary()
        try:
            response = self.api.post(
                f"/api/challenge/{opponent}",
                {
                    "rated": "false",
                    "variant": "standard",
                    "color": "random",
                    "clock.limit": "300",
                    "clock.increment": "3",
                },
            )
        except LichessError as exc:
            if exc.status == 400:
                with self.guard:
                    self.reservations.pop("posting", None)
                    self.save_summary()
                return 5
            # Ambiguous POST and 429 halt new challenges, but active games drain.
            raise
        game_id = identifier(response.get("challenge", response)["id"])
        with self.guard:
            deadline = self.reservations.pop("posting", now + 35)
            if game_id not in self.assigned:
                self.reservations[game_id] = deadline
            self.save_summary()
        print(
            f"Challenged {opponent}: casual 5+3 | concurrency {self.concurrency} | "
            f"allowance remaining before offer: {remaining}",
            flush=True,
        )
        return 5

    def schedule(self) -> None:
        while not self.ready.wait(0.2):
            if self.stop.is_set():
                return
        while not self.stop.is_set():
            try:
                pause = self.tick()
            except Exception as exc:
                status = exc.status if isinstance(exc, LichessError) else None
                with self.guard:
                    self.no_new_games = True
                    self.reason = f"Matchmaking halted: {type(exc).__name__}" + (
                        f" HTTP {status}" if status else ""
                    )
                    self.maybe_finish()
                self.log("matchmaking_halted_draining_games", http_status=status)
                pause = 60 if status == 429 else 5
            self.stop.wait(pause)

    def handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "challenge":
            challenge = event.get("challenge", {})
            if challenge.get("direction") == "out":
                game_id = identifier(challenge["id"])
                with self.guard:
                    if game_id not in self.assigned:
                        self.reservations[game_id] = time.time() + 35
                        self.save_summary()
            else:
                self.challenge(event)
        elif kind == "gameStart":
            self.start_game(event["game"]["id"])
        elif kind in {"challengeCanceled", "challengeDeclined"}:
            with self.guard:
                self.reservations.pop(event.get("challenge", {}).get("id"), None)
                self.maybe_finish()

    def account_connected(self) -> None:
        """Do not challenge until outstanding offers have replayed on a real stream."""
        with self.guard:
            self.startup_until = time.time() + 35
        self.ready.set()

    def run(self) -> None:
        # Crucially, failure to acquire this lock never writes a session/ledger.
        with run_lock(self.config.output_dir):
            scheduler = None
            try:
                self.load_session()
                ongoing = self.api.get("/api/account/playing").get("nowPlaying", [])
                self.recover(ongoing)
                self.maybe_finish()
                self.log(
                    "bot_ready",
                    username=self.config.username,
                    target=self.target,
                    completed=len(self.results),
                    concurrency=self.concurrency,
                    active=len(self.workers),
                )
                scheduler = threading.Thread(target=self.schedule, daemon=True)
                scheduler.start()
                while not self.stop.is_set():
                    try:
                        for event in self.api.stream(
                            "/api/stream/event", self.stop, connected=self.account_connected
                        ):
                            if self.stop.is_set():
                                break
                            self.handle_event(event)
                    except (LichessError, OSError) as exc:
                        status = exc.status if isinstance(exc, LichessError) else None
                        if status in (401, 403):
                            self.reason = "Lichess authorization failed"
                            raise LichessError(self.reason) from None
                        self.log("account_reconnecting", http_status=status)
                        self.stop.wait(60 if status == 429 else 3)
                    else:
                        self.stop.wait(2)
            finally:
                self.stop.set()
                if scheduler:
                    scheduler.join(timeout=25)
                with self.guard:
                    workers = list(self.workers.values())
                deadline = time.monotonic() + 25
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                if any(worker.is_alive() for worker in workers) or (
                    scheduler and scheduler.is_alive()
                ):
                    # Keep the account lock until no writer can mutate journals.
                    self.log("waiting_for_stream_shutdown")
                    for worker in workers:
                        worker.join()
                    if scheduler:
                        scheduler.join()
                if self.reason == "running":
                    self.reason = "interrupted"
                summary = self.save_summary()
                print(
                    f"\nSUMMARY: {summary['completed']}/{self.target} games | Won {summary['won']} | "
                    f"Lost {summary['lost']} | Draw {summary['draw']} | "
                    f"Unfinished/aborted {summary['unfinished']} | Points {summary['points']}\n"
                    f"Stopped: {self.reason}\nFull results: {self.summary_path}",
                    flush=True,
                )
