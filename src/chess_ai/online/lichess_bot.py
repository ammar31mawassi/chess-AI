"""Opt-in Lichess Bot API service; no browser, teacher, training, or promotion."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chess
import chess.pgn

from chess_ai.agents.neural_agent import resolve_device
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent, _Node
from chess_ai.config import load_config
from chess_ai.online.lichess_api import LichessAPI, LichessError
from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file
from chess_ai.storage.champion import load_gameplay_champion
from chess_ai.storage.games import save_pgn

FORMAT = "self-improving-chess-ai.lichess-bot"
ACTIVE = {"created", "started"}


class GameStateError(LichessError):
    """A game cannot safely be handled by this pilot."""


def identifier(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,30}", value) is None:
        raise LichessError("Invalid Lichess identifier")
    return value


@dataclass(frozen=True)
class BotConfig:
    username: str
    checkpoint_sha256: str
    device: str = "cuda"
    output_dir: Path = Path("data/games/evaluation/lichess_bot")
    minimum_seconds: int = 300
    maximum_seconds: int = 1800
    minimum_increment: int = 3
    maximum_increment: int = 30
    max_search_seconds: float = 1.0
    seed: int = 5112026
    opponent_kind: str = "human"
    ai_level: int | None = None

    def __post_init__(self) -> None:
        identifier(self.username)
        if self.opponent_kind not in {"human", "bot", "ai"}:
            raise ValueError("opponent_kind must be human, bot or ai")
        if self.opponent_kind == "ai":
            if type(self.ai_level) is not int or not 1 <= self.ai_level <= 8:
                raise ValueError("Lichess computer level must be an integer from 1 to 8")
        elif self.ai_level is not None:
            raise ValueError("ai_level requires opponent_kind ai")
        if re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256) is None:
            raise ValueError("Expected a pinned checkpoint SHA256")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "minimum_seconds",
            "maximum_seconds",
            "minimum_increment",
            "maximum_increment",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 0 < self.minimum_seconds <= self.maximum_seconds:
            raise ValueError("Invalid initial clock range")
        if not 0 < self.minimum_increment <= self.maximum_increment:
            raise ValueError("A positive increment is required")
        if not math.isfinite(self.max_search_seconds) or self.max_search_seconds <= 0:
            raise ValueError("max_search_seconds must be finite and positive")
        # Keep online games within the existing evaluation-only import boundary.
        if not self.output_dir.resolve().is_relative_to(Path("data/games/evaluation").resolve()):
            raise ValueError("Online games must stay under data/games/evaluation")

    @classmethod
    def read(cls, path: Path) -> BotConfig:
        raw = load_config(path)["lichess_bot"]
        if raw.get("format") != FORMAT or raw.get("version") != 1:
            raise ValueError("Unsupported Lichess bot configuration")
        return cls(**{k: v for k, v in raw.items() if k not in {"format", "version"}})

    def checkpoint(self) -> Path:
        champion = load_gameplay_champion()
        if (
            champion is None
            or sha256_file(champion.checkpoint) != self.checkpoint_sha256
            or champion.search_simulations != 32
            or champion.c_puct != 1.5
        ):
            raise ValueError(
                "Registered champion/settings changed; explicitly update the bot config"
            )
        resolve_device(self.device)
        return champion.checkpoint


def challenge_reason(config: BotConfig, challenge: Mapping[str, Any], *, busy: bool) -> str | None:
    if busy:
        return "later"
    if challenge.get("variant", {}).get("key") != "standard":
        return "variant"
    if challenge.get("rated") is not False:
        return "casual"
    challenger = challenge.get("challenger", {})
    if challenger.get("title") == "BOT" or not challenger.get("id"):
        return "noBot"
    if str(challenger["id"]).casefold() == config.username.casefold():
        return "generic"
    clock = challenge.get("timeControl", {})
    if clock.get("type") != "clock":
        return "timeControl"
    seconds, increment = clock.get("limit"), clock.get("increment")
    if type(seconds) is not int or type(increment) is not int:
        return "timeControl"
    if seconds < config.minimum_seconds or increment < config.minimum_increment:
        return "tooFast"
    if seconds > config.maximum_seconds or increment > config.maximum_increment:
        return "tooSlow"
    initial = challenge.get("initialFen")
    if initial not in (None, "startpos", chess.STARTING_FEN):
        return "standard"
    return None


class MoveAgent(Protocol):
    def choose_timed(
        self, board: chess.Board, remaining_ms: int, increment_ms: int
    ) -> chess.Move: ...


class ClockedPUCT(NeuralMCTSAgent):
    """Production PUCT32 with an online-only soft clock budget between traversals."""

    def __init__(self, checkpoint: Path, *, device: str, seed: int, max_seconds: float) -> None:
        super().__init__(checkpoint, device=device, seed=seed, simulations=32, c_puct=1.5)
        self.max_seconds = max_seconds

    def choose_timed(self, board: chess.Board, remaining_ms: int, increment_ms: int) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise LichessError("Cannot search a terminal board")
        # Preserve a network/transport reserve. Emergency move selection is legal
        # and deterministic; normal clocks use the audited PUCT32 traversal rule.
        if remaining_ms <= 500:
            return min(board.legal_moves, key=self._move_encoder.encode)
        budget = min(self.max_seconds, (remaining_ms - 500) / 40000 + increment_ms / 10000)
        deadline = time.perf_counter() + max(0.005, budget)
        search_board = board.copy(stack=True)
        root = _Node()
        self._expand(search_board, root)
        for _ in range(self.simulations):
            if time.perf_counter() >= deadline:
                break
            self._simulate(search_board, root)
        action, _ = max(
            root.edges.items(),
            key=lambda item: (
                item[1].visits,
                item[1].mean_value,
                item[1].prior,
                -item[0],
            ),
        )
        move = self._move_encoder.decode(action)
        if move not in board.legal_moves:
            raise LichessError("Search selected an illegal move")
        return move


class GameSession:
    """Rebuild authoritative server history; deduplicate acknowledged submissions."""

    def __init__(self, config: BotConfig, api: LichessAPI, agent: MoveAgent, game_id: str) -> None:
        self.config, self.api, self.agent = config, api, agent
        self.game_id = identifier(game_id)
        self.full: dict[str, Any] | None = None
        self.color = chess.WHITE
        self.submitted: str | None = None
        self.history: tuple[str, ...] = ()
        self.path = config.output_dir / self.game_id

    def full_event(self, event: dict[str, Any]) -> None:
        if event.get("id") != self.game_id or event.get("variant", {}).get("key") != "standard":
            raise GameStateError("Unsupported game identity/variant")
        if event.get("rated") is not False:
            raise GameStateError("Unexpected rated game; this pilot is casual-only")
        white, black = event["white"], event["black"]
        own = self.config.username.casefold()
        if str(white.get("id", "")).casefold() == own:
            self.color = chess.WHITE
            opponent = black
        elif str(black.get("id", "")).casefold() == own:
            self.color = chess.BLACK
            opponent = white
        else:
            raise GameStateError("Bot is not a player in this game")
        if self.config.opponent_kind == "ai":
            if (
                type(opponent.get("aiLevel")) is not int
                or opponent["aiLevel"] != self.config.ai_level
                or opponent.get("id")
            ):
                raise GameStateError("Unexpected Lichess computer level/identity")
            # Built-in computers have aiLevel, but no registered user ID/name.
            # Normalize only the saved representation; never submit this label to an API.
            opponent = {
                **opponent,
                "id": f"lichess-ai-level-{self.config.ai_level}",
                "name": f"Lichess computer level {self.config.ai_level}",
            }
            event = {**event, "black" if self.color else "white": opponent}
            white, black = event["white"], event["black"]
        elif (
            opponent.get("aiLevel") is not None
            or not opponent.get("id")
            or (opponent.get("title") == "BOT") != (self.config.opponent_kind == "bot")
        ):
            raise GameStateError("Unexpected opponent type")
        fen = event.get("initialFen", "startpos")
        if fen not in ("startpos", chess.STARTING_FEN):
            raise GameStateError("Custom starting positions are disabled")
        binding = {
            "format": FORMAT + ".game",
            "version": 1,
            "id": self.game_id,
            "username": self.config.username,
            "white": white["id"],
            "black": black["id"],
            "rated": False,
            "initial_fen": chess.STARTING_FEN,
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "search_simulations": 32,
            "c_puct": 1.5,
            "max_search_seconds": self.config.max_search_seconds,
            "clock": event.get("clock"),
            "training_data": False,
        }
        if self.config.opponent_kind == "ai":
            binding["opponent_ai_level"] = self.config.ai_level
        path = self.path / "binding.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != binding:
            raise GameStateError("Saved online game binding differs")
        atomic_json(path, binding)
        self.full = event

    def state_event(self, state: dict[str, Any], *, play: bool = True) -> bool:
        if self.full is None:
            raise GameStateError("Game state arrived before full game data")
        moves = state.get("moves")
        if not isinstance(moves, str):
            raise GameStateError("Invalid server move list")
        history = tuple(moves.split())
        if len(history) < len(self.history):
            return False  # Ignore a stale event; takebacks are not accepted.
        if history[: len(self.history)] != self.history:
            raise GameStateError("Server history changed without an accepted takeback")
        board = chess.Board()
        try:
            for uci in moves.split():
                board.push_uci(uci)
        except ValueError:
            raise GameStateError("Server history contains an illegal move") from None
        self.history = history
        status = state.get("status")
        if not isinstance(status, str):
            raise GameStateError("Server game status missing")
        for key in ("wtime", "btime", "winc", "binc"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise GameStateError("Invalid server clock")
        done = status not in ACTIVE
        winner = state.get("winner")
        result = (
            "1-0"
            if winner == "white"
            else "0-1"
            if winner == "black"
            else "1/2-1/2"
            if status in {"draw", "stalemate", "outoftime", "insufficientMaterialClaim"}
            else "*"
        )
        game = chess.pgn.Game.from_board(board)
        game.headers.update(
            {
                "Event": "Lichess BOT evaluation only",
                "Site": f"https://lichess.org/{self.game_id}",
                "White": self.full["white"].get("name", self.full["white"]["id"]),
                "Black": self.full["black"].get("name", self.full["black"]["id"]),
                "Result": result,
                "Termination": status,
                "ModelSHA256": self.config.checkpoint_sha256,
            }
        )
        save_pgn(game, self.path / "game.pgn")
        atomic_json(
            self.path / "state.json",
            {
                "format": FORMAT + ".state",
                "version": 1,
                "game_id": self.game_id,
                "moves": moves,
                "status": status,
                "result": result,
                "clocks": {k: state[k] for k in ("wtime", "btime", "winc", "binc")},
                "complete": done,
                "training_data": False,
            },
        )
        if (
            done
            or not play
            or status != "started"
            or board.turn != self.color
            or moves == self.submitted
        ):
            return done
        if board.is_repetition(3) or board.is_fifty_moves():
            # draw/yes claims an existing repetition. claim-draw is a different
            # endpoint for disconnected opponents, not a chess-rule claim.
            self.api.post(f"/api/bot/game/{self.game_id}/draw/yes")
            self.submitted = moves
            return False
        if board.is_game_over(claim_draw=False):
            return False  # Await authoritative terminal event.
        remaining = state["wtime" if self.color else "btime"]
        increment = state["winc" if self.color else "binc"]
        before = time.perf_counter()
        intended_draw = None
        if board.can_claim_draw():
            # Offline search treats an available draw claim as terminal. If the
            # claim requires an intended move, play that legal move first; Lichess
            # automatically claims repetitions for BOT accounts.
            for candidate in sorted(board.legal_moves, key=lambda m: m.uci()):
                board.push(candidate)
                try:
                    if board.is_repetition(3) or board.is_fifty_moves():
                        intended_draw = candidate
                        break
                finally:
                    board.pop()
        move = intended_draw or self.agent.choose_timed(
            board.copy(stack=True), remaining, increment
        )
        if move not in board.legal_moves:
            raise GameStateError("Agent output failed python-chess legality validation")
        # A failed/ambiguous POST is never blindly retried. The worker reconnects
        # for a new authoritative gameFull before deciding whether to submit.
        self.api.post(f"/api/bot/game/{self.game_id}/move/{board.uci(move)}")
        self.submitted = moves
        with (self.path / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "ply": board.ply(),
                        "move": board.uci(move),
                        "seconds": time.perf_counter() - before,
                        "remaining_ms": remaining,
                        "simulations_cap": 32,
                    }
                )
                + "\n"
            )
        return False


class BotRunner:
    """One account stream and at most one game worker; only explicit run starts play."""

    def __init__(self, config: BotConfig, api: LichessAPI, agent: MoveAgent) -> None:
        self.config, self.api, self.agent = config, api, agent
        self.stop = threading.Event()
        self.guard = threading.RLock()
        self.active: str | None = None
        self.pending: tuple[str, float] | None = None
        self.worker: threading.Thread | None = None
        self.ready = threading.Event()

    def game_finished(self, session: GameSession) -> None:
        """Hook for bounded user-run sessions, after a terminal state is saved."""

    def release_game(self, game_id: str) -> None:
        """Release a completed worker's slot; concurrent runners override this."""
        with self.guard:
            self.active = None

    @staticmethod
    def log(event: str, **values: Any) -> None:
        print(json.dumps({"event": event, **values}), flush=True)

    def challenge(self, event: dict[str, Any]) -> None:
        challenge = event.get("challenge", {})
        if challenge.get("direction", "in") != "in":
            return
        challenge_id = identifier(challenge.get("id"))
        with self.guard:
            if self.pending is not None and self.pending[1] < time.monotonic():
                self.pending = None
            busy = self.active is not None or self.pending is not None
            reason = challenge_reason(self.config, challenge, busy=busy)
            if reason is None:
                self.pending = (challenge_id, time.monotonic() + 30)
        if reason is not None:
            self.api.post(f"/api/challenge/{challenge_id}/decline", {"reason": reason})
            self.log("challenge_declined", id=challenge_id, reason=reason)
            return
        try:
            self.api.post(f"/api/challenge/{challenge_id}/accept")
        except LichessError:
            with self.guard:
                self.pending = None
            raise
        self.log("challenge_accepted", id=challenge_id)

    def start_game(self, game_id: str) -> None:
        game_id = identifier(game_id)
        with self.guard:
            if self.active == game_id:
                return
            if self.active is not None:
                self.log("unexpected_second_game", id=game_id)
                return
            self.active = game_id
            self.pending = None
            self.worker = threading.Thread(target=self._game, args=(game_id,), daemon=True)
            self.worker.start()

    def _game(self, game_id: str) -> None:
        session = GameSession(self.config, self.api, self.agent, game_id)
        finished = False
        try:
            while not self.stop.is_set() and not finished:
                try:
                    for event in self.api.stream(f"/api/bot/game/stream/{game_id}", self.stop):
                        if event.get("type") == "gameFull":
                            session.full_event(event)
                            finished = session.state_event(event["state"])
                        elif event.get("type") == "gameState":
                            finished = session.state_event(event)
                        if finished:
                            break
                except GameStateError:
                    self.log("unsupported_game_state", id=game_id)
                    self.stop.set()
                    break
                except (LichessError, OSError) as exc:
                    status = exc.status if isinstance(exc, LichessError) else None
                    self.log("game_reconnecting", id=game_id, http_status=status)
                    if status in (401, 403):
                        self.stop.set()
                    if self.stop.wait(60 if status == 429 else 3):
                        break
                else:
                    if not finished:
                        self.stop.wait(2)
        except Exception as exc:
            self.log("game_worker_failed", id=game_id, error_type=type(exc).__name__)
            self.stop.set()
        finally:
            if finished:
                self.game_finished(session)
                self.log("game_finished", id=game_id)
                self.release_game(game_id)
            else:
                # Keep the slot reserved when a live game could not finish.
                self.log("game_worker_stopped", id=game_id)

    def recover_finished(self, active_ids: set[str]) -> None:
        """Archive games that finished while this user-controlled process was offline."""
        for path in sorted(self.config.output_dir.glob("*/state.json")):
            game_id = identifier(path.parent.name)
            if game_id in active_ids:
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("complete"):
                continue
            session = GameSession(self.config, self.api, self.agent, game_id)
            with closing(self.api.stream(f"/api/bot/game/stream/{game_id}", self.stop)) as stream:
                first = next(stream, None)
                if first is None or first.get("type") != "gameFull":
                    raise LichessError("Cannot recover the final state of an interrupted game")
                session.full_event(first)
                session.state_event(first["state"], play=False)
            self.log("interrupted_game_archived", id=game_id)

    def run(self) -> None:
        with run_lock(self.config.output_dir):
            ongoing = self.api.get("/api/account/playing").get("nowPlaying", [])
            if len(ongoing) > 1:
                raise LichessError(
                    "More than one existing game; resolve these before starting the pilot"
                )
            self.recover_finished({game["gameId"] for game in ongoing})
            for game in ongoing:
                self.start_game(game["gameId"])
            self.log(
                "bot_ready",
                username=self.config.username,
                casual_only=True,
                min_clock_seconds=self.config.minimum_seconds,
                min_increment=self.config.minimum_increment,
            )
            try:
                self.ready.set()
                while not self.stop.is_set():
                    try:
                        for event in self.api.stream("/api/stream/event", self.stop):
                            if self.stop.is_set():
                                break
                            kind = event.get("type")
                            if kind == "challenge":
                                self.challenge(event)
                            elif kind == "gameStart":
                                self.start_game(event["game"]["id"])
                            elif kind in {"challengeCanceled", "challengeDeclined"}:
                                with self.guard:
                                    if (
                                        self.pending
                                        and event.get("challenge", {}).get("id") == self.pending[0]
                                    ):
                                        self.pending = None
                    except (LichessError, OSError) as exc:
                        status = exc.status if isinstance(exc, LichessError) else None
                        self.log("account_reconnecting", http_status=status)
                        if status in (401, 403):
                            raise LichessError(
                                "Lichess authorization failed; check/rotate the token"
                            ) from None
                        self.stop.wait(60 if status == 429 else 3)
                    else:
                        self.stop.wait(2)
            finally:
                self.stop.set()
                if self.worker is not None:
                    self.worker.join(timeout=25)


def verify_account(
    config: BotConfig, api: LichessAPI, *, require_bot: bool = True
) -> dict[str, Any]:
    account = api.account()
    if str(account.get("id", "")).casefold() != config.username.casefold():
        raise LichessError("Token belongs to a different account than the pinned bot username")
    if "bot:play" not in api.scopes():
        raise LichessError("Token needs bot:play permission")
    if require_bot and account.get("title") != "BOT":
        raise LichessError("Account must be upgraded to BOT before running")
    return account
