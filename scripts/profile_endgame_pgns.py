"""Measure a capped multi-player archive sample; never import or label training data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chess_ai.data.endgame_pilot import family
from chess_ai.data.stockfish_rebuild import _canonical, _game_digest
from chess_ai.storage.artifacts import atomic_json, replace_with_retry, sha256_file, signature


def profile(
    sources: list[Path],
    usernames: list[str],
    output: Path,
    *,
    seed: int = 23102036,
    maximum: int = 300,
    minimum_base_seconds: int = 180,
    maximum_per_player: int | None = None,
    snapshot_dir: Path | None = None,
    base_coverage: Path | None = None,
) -> dict[str, Any]:
    if output.exists() or maximum <= 0 or not usernames:
        raise ValueError("Use a new output, positive cap and explicit player usernames")
    if maximum_per_player is not None and maximum_per_player <= 0:
        raise ValueError("maximum_per_player must be positive")
    players = {name.casefold() for name in usernames}
    files = sorted({p for source in sources for p in source.glob("*.pgn")})
    if not files:
        raise ValueError("No local PGNs found")
    seen: set[str] = set()
    seen_urls: dict[str, str] = {}
    candidates = []
    counts: Counter[str] = Counter()
    inputs = []
    for path in files:
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8-sig") as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                if game.errors:
                    raise ValueError(f"Invalid PGN: {path}")
                if game.headers.get("TrainingData", "true").lower() == "false" or game.headers.get(
                    "EvaluationKind"
                ):
                    raise ValueError("Evaluation games cannot enter an archive coverage sample")
                if not players & {game.headers.get(c, "").casefold() for c in ("White", "Black")}:
                    raise ValueError(f"Game does not match an explicitly supplied username: {path}")
                board = game.board()
                if type(board) is not chess.Board or board.chess960 or not board.is_valid():
                    counts["excluded_variants"] += 1
                    continue
                if game.headers.get("Result") not in {"1-0", "0-1", "1/2-1/2"}:
                    counts["excluded_unfinished"] += 1
                    continue
                counts["records"] += 1
                move_count = 0
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        raise ValueError("Illegal archived move")
                    board.push(move)
                    move_count += 1
                if not move_count:
                    counts["excluded_empty_games"] += 1
                    continue
                digest = _game_digest(game)
                url = game.headers.get("Link", "")
                if url and url in seen_urls and seen_urls[url] != digest:
                    raise ValueError("Same game URL contains conflicting movetext")
                if url:
                    seen_urls[url] = digest
                if digest in seen:
                    counts["duplicate_movetext"] += 1
                    continue
                seen.add(digest)
                control = game.headers.get("TimeControl", "?")
                # Preserve the exact PGN control; do not guess rapid/blitz classes.
                try:
                    base = int(control.split("+")[0])
                except ValueError:
                    counts["excluded_unknown_or_multistage_control"] += 1
                    continue
                if base < minimum_base_seconds:
                    counts["excluded_short_base_control"] += 1
                    continue
                key = signature({"seed": seed, "game": digest})
                candidates.append((key, digest, game))
    ownership: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    chosen = []
    base_report: dict[str, Any] | None = None
    if base_coverage is not None:
        base_report = json.loads(base_coverage.read_text(encoding="utf-8"))
        if base_report.get("version") != 2 or base_report.get("usernames") != sorted(players):
            raise ValueError("Expansion requires a compatible original version-2 collection")
        for entry in [*base_report["inputs"], base_report["snapshot"]]:
            if sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("Original GM collection changed")
        by_digest = {item[1]: item for item in candidates}
        for digest in base_report["selected_game_digests"]:
            if digest not in by_digest:
                raise ValueError("Original GM game is absent from the eligible expansion inputs")
            owner = base_report["selected_game_owners"][digest]
            ownership[owner] += 1
            assignments[digest] = owner
            chosen.append(by_digest[digest])
        if (
            maximum_per_player is None
            or any(ownership[p] >= maximum_per_player for p in players)
            or len(chosen) >= maximum
        ):
            raise ValueError("Expanded quotas must exceed the frozen original quotas")
    for item in sorted(candidates, key=lambda item: item[0]):
        if len(chosen) == maximum:
            break
        if item[1] in assignments:
            continue
        game_players = players & {item[2].headers.get(c, "").casefold() for c in ("White", "Black")}
        eligible = [
            name
            for name in game_players
            if maximum_per_player is None or ownership[name] < maximum_per_player
        ]
        if not eligible:
            continue
        # One owner per unique game prevents a game between tracked GMs from
        # filling two quotas. Participation counts remain separately inspectable.
        owner = min(eligible, key=lambda name: (ownership[name], name))
        ownership[owner] += 1
        assignments[item[1]] = owner
        chosen.append(item)
    unique: set[str] = set()
    buckets: Counter[str] = Counter()
    controls: Counter[str] = Counter()
    selected_players: Counter[str] = Counter()
    for _, _, game in chosen:
        controls[game.headers["TimeControl"]] += 1
        for name in players & {game.headers.get(c, "").casefold() for c in ("White", "Black")}:
            selected_players[name] += 1
        board = game.board()
        reached = False
        for move in game.mainline_moves():
            if len(board.piece_map()) <= 10 and not board.is_game_over(claim_draw=True):
                reached = True
                key = _canonical(board)[0]
                if key not in unique:
                    unique.add(key)
                    buckets[family(board)] += 1
                    counts["unique_at_most_five_pieces"] += len(board.piece_map()) <= 5
            board.push(move)
        counts["selected_games_with_endgames"] += reached
        counts["selected_decisive_nonterminal_endings"] += game.headers[
            "Result"
        ] != "1/2-1/2" and not board.is_game_over(claim_draw=True)
    report = {
        "format": "self-improving-chess-ai.endgame-archive-coverage",
        "version": 2,
        "inputs": inputs,
        "usernames": sorted(players),
        "seed": seed,
        "maximum_games": maximum,
        "maximum_per_player": maximum_per_player,
        "selection_ownership": dict(ownership),
        "minimum_base_seconds": minimum_base_seconds,
        "require_recorded_moves": True,
        "eligible_games": len(candidates),
        "selected_games": len(chosen),
        "counts": dict(counts),
        "unique_endgame_positions": len(unique),
        "families": dict(buckets),
        "time_controls": dict(controls),
        "players": dict(selected_players),
        "selected_game_digests": [digest for _, digest, _ in chosen],
        "selected_game_owners": assignments,
        "training_imported": False,
        "teacher_calls": 0,
        "note": "Nonterminal decisive endings include resignation, timeout and other causes. No training split or labels are created. An explicit multi-player importer and historical exclusion pass are still required before training.",
    }
    if base_report is not None:
        assert base_coverage is not None
        report.update(
            version=3,
            base_coverage={"path": str(base_coverage), "sha256": sha256_file(base_coverage)},
            retained_original_games=len(base_report["selected_game_digests"]),
            added_ownership={
                p: ownership[p] - base_report["selection_ownership"][p] for p in sorted(players)
            },
        )
    if snapshot_dir is not None:
        if maximum_per_player is None or any(
            ownership[name] != maximum_per_player for name in players
        ):
            raise ValueError("A snapshot requires complete explicit per-player quotas")
        content = (
            "\n\n".join(
                game.accept(
                    chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                )
                for _, _, game in chosen
            )
            + "\n"
        ).encode("utf-8")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / "selected_games.pgn"
        if snapshot.exists():
            if snapshot.read_bytes() != content:
                raise ValueError("Existing GM snapshot differs from this selection")
        else:
            temporary = snapshot.with_suffix(".pgn.part")
            temporary.write_bytes(content)
            replace_with_retry(temporary, snapshot)
        report["snapshot"] = {
            "path": str(snapshot),
            "sha256": sha256_file(snapshot),
            "games": len(chosen),
        }
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--username", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=23102036)
    parser.add_argument("--minimum-base-seconds", type=int, default=180)
    parser.add_argument("--max-games-per-player", type=int)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--base-coverage", type=Path)
    args = parser.parse_args()
    report = profile(
        args.source_dir,
        args.username,
        args.output,
        seed=args.seed,
        maximum=args.max_games,
        minimum_base_seconds=args.minimum_base_seconds,
        maximum_per_player=args.max_games_per_player,
        snapshot_dir=args.snapshot_dir,
        base_coverage=args.base_coverage,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                **{
                    k: report[k]
                    for k in (
                        "selected_games",
                        "selection_ownership",
                        "unique_endgame_positions",
                        "families",
                        "training_imported",
                    )
                },
            },
            indent=2,
        )
    )
