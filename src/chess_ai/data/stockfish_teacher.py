"""Independent, bounded UCI position analysis; no gameplay or optimizer access."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import numpy.typing as npt

from chess_ai.environment.move_encoder import MoveEncoder
from chess_ai.storage.artifacts import atomic_json, sha256_file, signature

TEACHER_FORMAT = "self-improving-chess-ai.stockfish-teacher"
TEACHER_VERSION = 1


class StockfishTeacherError(RuntimeError):
    """Invalid provenance, engine failure, timeout, or invalid analysis."""


@dataclass(frozen=True, slots=True)
class StockfishConfig:
    executable: Path
    source_commit: str
    nodes: int | None = 20_000
    depth: int | None = None
    multipv: int = 5
    threads: int = 1
    hash_mb: int = 64
    timeout_seconds: float = 30.0
    temperature_cp: float = 100.0
    mate_score_cp: int = 100_000
    evaluation_scale_cp: float = 600.0
    store_evaluation: bool = True
    cache_dir: Path | None = None
    mate_distance_temperature: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "executable", Path(self.executable))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if (
            not isinstance(self.source_commit, str)
            or len(self.source_commit) != 40
            or any(c not in "0123456789abcdef" for c in self.source_commit)
        ):
            raise ValueError("source_commit must be the exact 40-character Stockfish Git SHA")
        if (self.nodes is None) == (self.depth is None):
            raise ValueError("Configure exactly one fixed nodes or depth limit")
        for name in ("nodes", "depth", "multipv", "threads", "hash_mb", "mate_score_cp"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("timeout_seconds", "temperature_cp", "evaluation_scale_cp"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.store_evaluation, bool):
            raise ValueError("store_evaluation must be a boolean")
        if self.mate_distance_temperature is not None and (
            isinstance(self.mate_distance_temperature, bool)
            or not isinstance(self.mate_distance_temperature, (int, float))
            or not math.isfinite(self.mate_distance_temperature)
            or self.mate_distance_temperature <= 0
        ):
            raise ValueError("mate_distance_temperature must be finite and positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StockfishConfig:
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown stockfish settings: {', '.join(sorted(unknown))}")
        return cls(**dict(raw))

    def analysis_settings(self) -> dict[str, Any]:
        settings = asdict(self)
        settings.pop("executable")
        settings.pop("cache_dir")
        # Preserve all legacy cache/provenance signatures when the option is off.
        if self.mate_distance_temperature is None:
            settings.pop("mate_distance_temperature")
        return {
            **settings,
            "clear_hash": True,
            "score_iteration": "last_complete_exact_multipv_depth",
            "independent_fen_without_history": True,
            "syzygy_path": "",
            "limit_strength": False,
            "skill_level": 20,
        }


def convert_score(
    score: chess.engine.PovScore, turn: chess.Color, config: StockfishConfig
) -> dict[str, int | float | None]:
    """Signed mate distance and cp from the player to move; evaluation is metadata."""
    relative = score.pov(turn)
    mate = relative.mate()
    cp = relative.score()
    # python-chess handles MateGiven and mate zero signs consistently.
    scaled = relative.score(mate_score=config.mate_score_cp)
    if scaled is None:
        raise StockfishTeacherError("Engine supplied an unconvertible score")
    if mate is None:
        scaled = max(-config.mate_score_cp + 1, min(config.mate_score_cp - 1, scaled))
    return {
        "cp": cp,
        "mate": mate,
        "score_cp": scaled,
        "normalized_evaluation": math.tanh(scaled / config.evaluation_scale_cp),
    }


def policy_from_rows(
    board: chess.Board, rows: Sequence[Mapping[str, Any]], config: StockfishConfig
) -> tuple[npt.NDArray[np.float32], list[dict[str, Any]]]:
    """Validate the entire PV and normalize only the analyzed legal root moves."""
    expected = min(config.multipv, board.legal_moves.count())
    if len(rows) != expected:
        raise StockfishTeacherError(f"Expected {expected} MultiPV lines; received {len(rows)}")
    trace: list[dict[str, Any]] = []
    seen: set[chess.Move] = set()
    for row in rows:
        pv, score = row.get("pv"), row.get("score")
        if not isinstance(pv, (list, tuple)) or not pv:
            raise StockfishTeacherError("Engine output is missing a principal variation")
        if not isinstance(score, chess.engine.PovScore):
            raise StockfishTeacherError("Engine output is missing a valid score")
        if row.get("lowerbound") or row.get("upperbound"):
            raise StockfishTeacherError("Bound-only scores cannot supply soft policy targets")
        check = board.copy(stack=False)
        for move in pv:
            if not isinstance(move, chess.Move) or move not in check.legal_moves:
                raise StockfishTeacherError("Engine principal variation contains an illegal move")
            check.push(move)
        if pv[0] in seen:
            raise StockfishTeacherError("Engine supplied duplicate MultiPV root moves")
        seen.add(pv[0])
        converted = convert_score(score, board.turn, config)
        trace.append(
            {
                "move_uci": pv[0].uci(),
                "pv": [move.uci() for move in pv],
                **converted,
                "depth": row.get("depth"),
                "nodes": row.get("nodes"),
            }
        )
    trace.sort(key=lambda item: (-item["score_cp"], item["move_uci"]))
    scores = np.array([item["score_cp"] for item in trace], dtype=np.float64)
    weights = np.exp((scores - scores.max()) / config.temperature_cp)
    if config.mate_distance_temperature is not None:
        winning_mates = np.array(
            [item["mate"] is not None and item["score_cp"] > 0 for item in trace]
        )
        losing_mates = all(item["mate"] is not None and item["score_cp"] < 0 for item in trace)
        if winning_mates.any():
            # Mate scores differ by only one cp per reported mate move in the
            # legacy mapping. Prefer shorter known mates without changing values.
            weights = np.zeros(len(trace), dtype=np.float64)
            distances = np.array(
                [
                    abs(item["mate"])
                    for item in trace
                    if item["mate"] is not None and item["score_cp"] > 0
                ],
                dtype=np.float64,
            )
            weights[winning_mates] = np.exp(
                -(distances - distances.min()) / config.mate_distance_temperature
            )
        elif losing_mates:
            distances = np.array([abs(item["mate"]) for item in trace], dtype=np.float64)
            weights = np.exp((distances - distances.max()) / config.mate_distance_temperature)
    probabilities = weights / weights.sum()
    policy = np.zeros(MoveEncoder.action_size, dtype=np.float32)
    encoder = MoveEncoder()
    for item, probability in zip(trace, probabilities, strict=True):
        item["probability"] = float(probability)
        policy[encoder.encode(chess.Move.from_uci(item["move_uci"]))] = probability
    return policy, trace


class StockfishTeacher:
    """One reliably closed process with cold-hash analysis and signed disk caching."""

    def __init__(self, config: StockfishConfig) -> None:
        self.config = config
        self.engine: chess.engine.SimpleEngine | None = None
        if not config.executable.is_file():
            raise StockfishTeacherError(f"Stockfish executable does not exist: {config.executable}")
        self.binary_sha256 = sha256_file(config.executable)
        sidecar = config.executable.with_suffix(".provenance.json")
        if sidecar.exists():
            try:
                build = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(build, dict):
                    raise ValueError("build provenance must be a mapping")
                if (
                    build["source_commit"] != config.source_commit
                    or build["executable_sha256"] != self.binary_sha256
                ):
                    raise ValueError("build commit/binary digest does not match configuration")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise StockfishTeacherError(
                    f"Invalid executable provenance {sidecar}: {exc}"
                ) from exc
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                str(config.executable.resolve()),
                timeout=config.timeout_seconds,
                cwd=str(config.executable.resolve().parent),
            )
            if "stockfish" not in self.engine.id.get("name", "").lower():
                raise StockfishTeacherError("Configured executable does not identify as Stockfish")
            required = {"Threads", "Hash", "MultiPV", "Clear Hash"}
            if not required.issubset(self.engine.options):
                raise StockfishTeacherError("Stockfish is missing required UCI options")
            settings: dict[str, Any] = {"Threads": config.threads, "Hash": config.hash_mb}
            for name, value in (
                ("SyzygyPath", ""),
                ("UCI_LimitStrength", False),
                ("Skill Level", 20),
            ):
                if name in self.engine.options:
                    settings[name] = value
            self.engine.configure(settings)
            nnue_hashes = {}
            for name in ("EvalFile", "EvalFileSmall"):
                if name in self.engine.options:
                    filename = self.engine.options[name].default
                    if filename:
                        net = config.executable.resolve().parent / str(filename)
                        if net.is_file():
                            nnue_hashes[str(net)] = sha256_file(net)
            self.provenance = {
                "format": TEACHER_FORMAT,
                "version": TEACHER_VERSION,
                "source_repository": "https://github.com/official-stockfish/Stockfish",
                "source_commit": config.source_commit,
                "engine_id": dict(self.engine.id),
                "executable_sha256": self.binary_sha256,
                "analysis": config.analysis_settings(),
                "uci_settings": settings,
                "uci_defaults": {
                    name: option.default for name, option in self.engine.options.items()
                },
                "license": "GPL-3.0",
                "external_nnue_sha256": nnue_hashes,
            }
        except BaseException as exc:
            self.close()
            if isinstance(exc, (KeyboardInterrupt, SystemExit, StockfishTeacherError)):
                raise
            raise StockfishTeacherError(f"Stockfish startup/configuration failed: {exc}") from exc

    def __enter__(self) -> StockfishTeacher:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except (TimeoutError, chess.engine.EngineError):
                pass
            finally:
                self.engine.close()
                self.engine = None

    def cache_signature(self, fen: str) -> str:
        return signature({"fen": fen, "teacher": self.provenance})

    def _analyse(
        self, board: chess.Board, *, root_moves: Sequence[chess.Move] | None = None
    ) -> list[chess.engine.InfoDict]:
        assert self.engine is not None
        engine = self.engine

        # SimpleEngine.analyse does not enforce its timeout for nodes/depth limits.
        # Put a deadline around python-chess's protocol coroutine, without adding
        # a wall-clock search limit (which would change the generated targets).
        async def collect() -> list[chess.engine.InfoDict]:
            count = min(
                self.config.multipv,
                board.legal_moves.count() if root_moves is None else len(root_moves),
            )
            analysis = await engine.protocol.analysis(
                board,
                chess.engine.Limit(nodes=self.config.nodes, depth=self.config.depth),
                multipv=count,
                game=object(),
                root_moves=root_moves,
            )
            current: list[chess.engine.InfoDict] = []
            complete: list[chess.engine.InfoDict] = []
            try:
                async for row in analysis:
                    if not {"depth", "score", "pv"}.issubset(row):
                        if "multipv" in row:
                            current = []
                        continue
                    rank = row.get("multipv", 1)
                    # Stockfish emits ordered rank 1..N snapshots. During an
                    # interrupted iteration, ranks can have different depths
                    # and reordered roots. Never merge those rows by depth.
                    if rank == 1:
                        current = []
                    if rank != len(current) + 1:
                        current = []
                        continue
                    current.append(row)
                    if len(current) != count:
                        continue
                    if (
                        len({line["depth"] for line in current}) == 1
                        and all(
                            line["pv"] and not line.get("lowerbound") and not line.get("upperbound")
                            for line in current
                        )
                        and len({line["pv"][0] for line in current}) == count
                        and (not complete or current[0]["depth"] >= complete[0]["depth"])
                    ):
                        complete = current.copy()
                await analysis.wait()
            finally:
                analysis.stop()
            if not complete:
                raise StockfishTeacherError(
                    "No complete exact MultiPV iteration; increase the fixed node/depth budget"
                )
            return complete

        async def bounded() -> list[chess.engine.InfoDict]:
            return await asyncio.wait_for(collect(), timeout=self.config.timeout_seconds)

        future = asyncio.run_coroutine_threadsafe(bounded(), engine.protocol.loop)
        return future.result(timeout=self.config.timeout_seconds + 1.0)

    def analyze(self, board: chess.Board) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        if not board.is_valid() or board.is_game_over(claim_draw=False):
            raise StockfishTeacherError("Cannot analyze an invalid or terminal position")
        # FEN-only labels intentionally disregard repetition history.
        board = chess.Board(board.fen())
        key = self.cache_signature(board.fen())
        cache = self.config.cache_dir / f"{key}.json" if self.config.cache_dir else None
        if cache is not None and cache.exists():
            try:
                raw = json.loads(cache.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("cache root must be a mapping")
                if raw.get("signature") != key or raw.get("payload_sha256") != signature(
                    raw["payload"]
                ):
                    raise ValueError("cache signature or content digest mismatch")
                payload = raw["payload"]
                cached_rows: list[chess.engine.InfoDict] = []
                for item in payload["top_moves"]:
                    score = (
                        (
                            chess.engine.MateGiven
                            if item["mate"] == 0 and item["score_cp"] > 0
                            else chess.engine.Mate(item["mate"])
                        )
                        if item["mate"] is not None
                        else chess.engine.Cp(item["cp"])
                    )
                    cached_rows.append(
                        {
                            "pv": [chess.Move.from_uci(uci) for uci in item["pv"]],
                            "score": chess.engine.PovScore(score, board.turn),
                            "depth": item["depth"],
                            "nodes": item["nodes"],
                        }
                    )
                if payload["fen"] != board.fen() or payload["teacher_signature"] != signature(
                    self.provenance
                ):
                    raise ValueError("cache FEN or teacher identity mismatch")
                policy, _ = policy_from_rows(board, cached_rows, self.config)
                return policy, payload
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise StockfishTeacherError(f"Invalid Stockfish cache {cache}: {exc}") from exc
        if self.engine is None:
            raise StockfishTeacherError("Stockfish process is closed")
        try:
            self.engine.configure({"Clear Hash": None})
            rows = self._analyse(board)
        except (TimeoutError, concurrent.futures.TimeoutError) as exc:
            self.engine.close()
            self.engine = None
            raise StockfishTeacherError(
                "Stockfish analysis timed out; game shard was not committed"
            ) from exc
        except chess.engine.EngineError as exc:
            self.close()
            raise StockfishTeacherError(f"Stockfish crashed or rejected analysis: {exc}") from exc
        policy, trace = policy_from_rows(board, rows, self.config)
        payload = {
            "fen": board.fen(),
            "teacher_signature": signature(self.provenance),
            "top_moves": trace,
            "teacher_move_uci": trace[0]["move_uci"],
            "policy_entropy": float(-sum(p * math.log(p) for p in policy if p > 0)),
        }
        if self.config.store_evaluation:
            payload["stockfish_evaluation"] = trace[0]["normalized_evaluation"]
        if cache is not None:
            atomic_json(
                cache, {"signature": key, "payload": payload, "payload_sha256": signature(payload)}
            )
        return policy, payload

    def analyze_move(self, board: chess.Board, move: chess.Move) -> dict[str, Any]:
        """Score a legal played move, forcing that root when outside the top MultiPV."""
        if move not in board.legal_moves:
            raise StockfishTeacherError("Cannot score an illegal played move")
        _, best = self.analyze(board)
        for row in best["top_moves"]:
            if row["move_uci"] == move.uci():
                return {**row, "analysis_kind": "multipv_root"}
        independent = chess.Board(board.fen())
        key = signature({"fen": independent.fen(), "move": move.uci(), "teacher": self.provenance})
        cache = self.config.cache_dir / f"forced_{key}.json" if self.config.cache_dir else None
        rows: list[chess.engine.InfoDict]
        if cache is not None and cache.exists():
            try:
                raw = json.loads(cache.read_text(encoding="utf-8"))
                payload = raw["payload"]
                if raw["signature"] != key or raw["payload_sha256"] != signature(payload):
                    raise ValueError("forced-root cache signature or digest mismatch")
                item = payload["score"]
                score = (
                    (
                        chess.engine.MateGiven
                        if item["mate"] == 0 and item["score_cp"] > 0
                        else chess.engine.Mate(item["mate"])
                    )
                    if item["mate"] is not None
                    else chess.engine.Cp(item["cp"])
                )
                rows = [
                    {
                        "pv": [chess.Move.from_uci(uci) for uci in item["pv"]],
                        "score": chess.engine.PovScore(score, independent.turn),
                        "depth": item["depth"],
                        "nodes": item["nodes"],
                    }
                ]
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise StockfishTeacherError(f"Invalid forced-root cache {cache}: {exc}") from exc
        else:
            if self.engine is None:
                raise StockfishTeacherError("Stockfish process is closed")
            try:
                self.engine.configure({"Clear Hash": None})
                rows = self._analyse(independent, root_moves=[move])
            except (TimeoutError, concurrent.futures.TimeoutError, chess.engine.EngineError) as exc:
                self.close()
                raise StockfishTeacherError(
                    f"Stockfish forced-root analysis failed: {exc}"
                ) from exc
        _, trace = policy_from_rows(independent, rows, replace(self.config, multipv=1))
        if trace[0]["move_uci"] != move.uci():
            raise StockfishTeacherError("Forced-root analysis returned a different move")
        result = {**trace[0], "analysis_kind": "forced_root"}
        if cache is not None:
            payload = {"score": result}
            atomic_json(
                cache, {"signature": key, "payload": payload, "payload_sha256": signature(payload)}
            )
        return result
