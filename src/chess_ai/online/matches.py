"""Bounded, user-launched Lichess matches with conservative daily accounting."""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess

from chess_ai.online.credentials import load_token
from chess_ai.online.lichess_api import LichessAPI, LichessError
from chess_ai.online.lichess_bot import (
    BotConfig,
    BotRunner,
    ClockedPUCT,
    GameSession,
    identifier,
    verify_account,
)
from chess_ai.storage.artifacts import atomic_json


def recent_bot_games(api: LichessAPI, username: str, now: float) -> set[str]:
    """Count all started bot games over 24 hours, conservatively including aborts."""
    path = (
        f"/api/games/user/{identifier(username)}?since={int((now - 86400) * 1000)}"
        "&ongoing=true&finished=true&moves=false"
    )
    result = set()
    with closing(api.stream(path, threading.Event())) as games:
        for game in games:
            players = game["players"]
            opponents = [
                player.get("user", {})
                for player in players.values()
                if str(player.get("user", {}).get("id", "")).casefold() != username.casefold()
            ]
            if any(player.get("title") == "BOT" for player in opponents):
                result.add(identifier(game["id"]))
    return result


def result_record(session: GameSession) -> dict[str, Any]:
    saved = json.loads((session.path / "state.json").read_text(encoding="utf-8"))
    assert session.full is not None
    opponent = session.full["black" if session.color else "white"]
    result = saved["result"]
    outcome = (
        "draw"
        if result == "1/2-1/2"
        else "unfinished"
        if result == "*"
        else "won"
        if result == ("1-0" if session.color else "0-1")
        else "lost"
    )
    board = chess.Board()
    for move in saved["moves"].split():
        board.push_uci(move)
    reason = saved["status"]
    if reason == "draw":
        if board.is_repetition(3):
            reason = "threefold repetition"
        elif board.is_fifty_moves():
            reason = "fifty-move rule"
        elif board.is_insufficient_material():
            reason = "insufficient material"
        else:
            reason = "draw (server did not provide a more specific reason)"
    reason = {"mate": "checkmate", "resign": "resignation", "outoftime": "time expired"}.get(
        reason, reason
    )
    return {
        "id": session.game_id,
        "opponent": opponent.get("name", opponent["id"])
        if opponent.get("aiLevel")
        else opponent["id"],
        "color": "white" if session.color else "black",
        "outcome": outcome,
        "result": result,
        "reason": reason,
        "url": f"https://lichess.org/{session.game_id}",
    }


class MatchRunner(BotRunner):
    def __init__(self, config: BotConfig, api: LichessAPI, agent: ClockedPUCT, target: int):
        super().__init__(config, api, agent)
        self.target = target
        self.results: list[dict[str, Any]] = []
        self.rng = random.Random(config.seed)
        self.attempted: dict[str, float] = {}
        self.reason = "interrupted"
        self.ledger_path = config.output_dir / "bot_starts.json"
        self.starts: dict[str, float] = {}
        self.summary_path = (
            config.output_dir
            / "sessions"
            / (datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f") + ".json")
        )

    def challenge(self, event: dict[str, Any]) -> None:
        if self.config.opponent_kind in {"bot", "ai"}:
            item = event.get("challenge", {})
            if item.get("direction", "in") == "in":
                self.api.post(
                    f"/api/challenge/{identifier(item['id'])}/decline", {"reason": "later"}
                )
        else:
            super().challenge(event)

    def start_game(self, game_id: str) -> None:
        if self.config.opponent_kind == "bot":
            self.starts[identifier(game_id)] = time.time()
            atomic_json(self.ledger_path, self.starts)
        super().start_game(game_id)

    def game_finished(self, session: GameSession) -> None:
        record = result_record(session)
        self.results.append(record)
        print(
            f"Game {len(self.results)}/{self.target}: {record['outcome'].upper()} "
            f"{record['result']} vs {record['opponent']} as {record['color']} "
            f"— {record['reason']} | {record['url']}",
            flush=True,
        )
        if len(self.results) >= self.target:
            self.reason = "target reached"
            self.stop.set()
        self.save_summary()

    def save_summary(self) -> dict[str, Any]:
        counts = Counter(record["outcome"] for record in self.results)
        summary = {
            "format": "self-improving-chess-ai.lichess-session",
            "version": 1,
            "mode": self.config.opponent_kind,
            "requested": self.target,
            "completed": len(self.results),
            "won": counts["won"],
            "lost": counts["lost"],
            "draw": counts["draw"],
            "unfinished": counts["unfinished"],
            "points": counts["won"] + counts["draw"] / 2,
            "stop_reason": self.reason,
            "games": self.results,
            "training_data": False,
        }
        if self.config.opponent_kind == "ai":
            summary["ai_level"] = self.config.ai_level
        atomic_json(self.summary_path, summary)
        return summary

    def challenge_computer(self) -> None:
        """Create exactly one game; recover an existing game without another POST."""
        with self.guard:
            if self.active is not None or self.stop.is_set():
                return
        # Account-stream reconnection can lag a game created by an interrupted POST.
        ongoing = self.api.get("/api/account/playing").get("nowPlaying", [])
        if ongoing:
            if len(ongoing) != 1:
                raise LichessError("Multiple ongoing games; resolve these before playing")
            self.start_game(ongoing[0]["gameId"])
            return
        response = self.api.post(
            "/api/challenge/ai",
            {
                "level": str(self.config.ai_level),
                "clock.limit": "600",
                "clock.increment": "5",
                "color": "random",
                "variant": "standard",
            },
        )
        game_id = identifier(response["id"])
        print(
            f"Lichess computer level {self.config.ai_level}: casual 10+5 | "
            f"https://lichess.org/{game_id}",
            flush=True,
        )
        # The API's gameStart may arrive before the POST response. The runner
        # deduplicates an active ID; a completed game must not be restarted.
        if not self.stop.is_set():
            self.start_game(game_id)

    def schedule(self) -> None:
        while not self.ready.wait(0.2):
            if self.stop.is_set():
                return
        try:
            if self.config.opponent_kind == "ai":
                self.challenge_computer()
                return
            while not self.stop.is_set():
                with self.guard:
                    busy = self.active is not None
                    pending = self.pending
                if busy or (pending and pending[1] > time.monotonic()):
                    self.stop.wait(1)
                    continue
                # A late acceptance after an expired challenge must be recovered
                # before another outgoing challenge is attempted.
                ongoing = self.api.get("/api/account/playing").get("nowPlaying", [])
                if ongoing:
                    if len(ongoing) != 1:
                        raise LichessError("Multiple ongoing games; stop and resolve them")
                    self.start_game(ongoing[0]["gameId"])
                    continue
                with self.guard:
                    self.pending = None
                now = time.time()
                recent = recent_bot_games(self.api, self.config.username, now)
                recent.update(key for key, started in self.starts.items() if started > now - 86400)
                remaining = max(0, 100 - len(recent))
                if not remaining:
                    self.reason = "daily allowance exhausted (conservative rolling 24 hours)"
                    self.stop.set()
                    return
                bots = []
                with closing(self.api.stream("/api/bot/online?nb=512", self.stop)) as stream:
                    for bot in stream:
                        name = identifier(bot["id"])
                        if name.casefold() != self.config.username.casefold() and (
                            now - self.attempted.get(name, 0) >= 1800
                        ):
                            bots.append(name)
                self.rng.shuffle(bots)
                if not bots:
                    print(
                        "Waiting for an eligible online bot (30-minute opponent cooldown).",
                        flush=True,
                    )
                    self.stop.wait(60)
                    continue
                opponent = bots[0]
                self.attempted[opponent] = now
                # Reserve before POST: a gameStart can race the response.
                with self.guard:
                    self.pending = ("outgoing", time.monotonic() + 35)
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
                            self.pending = None
                        print(
                            f"Challenge unavailable: {opponent}; trying another after 30 seconds.",
                            flush=True,
                        )
                        self.stop.wait(30)
                        continue
                    # Ambiguous POST/429: stop instead of risking duplicate games
                    # or retrying a platform daily restriction indefinitely.
                    raise
                challenge_id = identifier(response.get("challenge", response)["id"])
                with self.guard:
                    if self.pending and self.pending[0] == "outgoing":
                        self.pending = (challenge_id, time.monotonic() + 35)
                print(
                    f"Challenged {opponent}: casual 5+3; allowance remaining: {remaining}",
                    flush=True,
                )
                self.stop.wait(30)
        except Exception as exc:
            self.reason = f"Matchmaking stopped: {type(exc).__name__}"
            print(self.reason + "; no further challenges will be sent.", flush=True)
            self.stop.set()

    def run(self) -> None:
        if self.ledger_path.exists():
            self.starts = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        scheduler = None
        if self.config.opponent_kind in {"bot", "ai"}:
            scheduler = threading.Thread(target=self.schedule, daemon=True)
            scheduler.start()
        else:
            print(
                "BOT accounts cannot enter human matchmaking. Waiting for ONE human "
                "casual standard rapid challenge, 10+5.\n"
                f"Opponent: visit https://lichess.org/@/{self.config.username} and challenge the bot.\n"
                "Leave this terminal open until the game finishes.",
                flush=True,
            )
        try:
            super().run()
        finally:
            self.stop.set()
            if scheduler:
                scheduler.join(timeout=25)
            summary = self.save_summary()
            print(
                f"\nSUMMARY: {summary['completed']}/{self.target} games | "
                f"Won {summary['won']} | Lost {summary['lost']} | Draw {summary['draw']} | "
                f"Unfinished/aborted {summary['unfinished']} | Points {summary['points']}\n"
                f"Stopped: {self.reason}\nFull results: {self.summary_path}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", help="1-100 bot games, human, or lvl <1-8> (one computer game)")
    parser.add_argument("level", nargs="?", type=int, choices=range(1, 9))
    parser.add_argument("--concurrency", type=int, help="Simultaneous BOT games (default: 5)")
    args = parser.parse_args()
    human = args.mode == "human"
    computer = args.mode == "lvl"
    if args.concurrency is not None and (
        args.concurrency < 1 or ((human or computer) and args.concurrency != 1)
    ):
        parser.error("Concurrency must be positive; human/lvl modes support one game")
    if computer != (args.level is not None):
        parser.error("Use chess-bot lvl <1-8>; other modes take no level argument")
    try:
        target = 1 if human or computer else int(args.mode)
        if not 1 <= target <= 100:
            raise ValueError
    except ValueError:
        parser.error("Use lvl <1-8>, human, or a number from 1 through 100")
    os.chdir(Path(__file__).resolve().parents[3])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    config = replace(
        BotConfig.read(Path("configs/lichess_bot.yaml")),
        opponent_kind="ai" if computer else "human" if human else "bot",
        ai_level=args.level,
    )
    if human:
        config = replace(
            config,
            minimum_seconds=600,
            maximum_seconds=600,
            minimum_increment=5,
            maximum_increment=5,
        )
    api = LichessAPI(load_token())
    verify_account(config, api)
    agent = ClockedPUCT(
        config.checkpoint(),
        device=config.device,
        seed=config.seed,
        max_seconds=config.max_search_seconds,
    )
    agent.evaluate(chess.Board())
    if not human and not computer:
        from chess_ai.online.parallel_matches import ParallelMatchRunner

        ParallelMatchRunner(config, api, agent, target, concurrency=args.concurrency or 5).run()
    else:
        MatchRunner(config, api, agent, target).run()


def cli() -> None:
    try:
        main()
    except (ValueError, LichessError) as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        print(
            "Stopped. Restart the same command promptly to recover ongoing games; results stay saved."
        )


if __name__ == "__main__":
    cli()
