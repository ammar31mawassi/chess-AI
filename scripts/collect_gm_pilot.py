"""Collect exact per-GM archive quotas through the public API; never train/import."""

from __future__ import annotations

import argparse
import json
import runpy
import urllib.request
from datetime import date
from pathlib import Path

from chess_ai.storage.artifacts import atomic_json, run_lock, sha256_file

PLAYERS = {"hikaru": 15448422, "magnuscarlsen": 3889224, "fabianocaruana": 11177810}


def collect(
    raw_dir: Path,
    report_dir: Path,
    *,
    games_per_player: int = 500,
    latest_month: str = "2026-09",
    max_months: int = 12,
    seed: int = 23102036,
    minimum_base_seconds: int = 180,
    base_coverage: Path | None = None,
) -> dict:
    if games_per_player <= 0 or max_months <= 0:
        raise ValueError("Game quotas and month cap must be positive")
    latest = date.fromisoformat(latest_month + "-01")
    download = runpy.run_path(str(Path(__file__).with_name("download_chesscom_games.py")))[
        "download"
    ]
    profile = runpy.run_path(str(Path(__file__).with_name("profile_endgame_pgns.py")))["profile"]
    spec = {
        "coverage_version": 2,
        "raw_dir": str(raw_dir),
        "players": PLAYERS,
        "games_per_player": games_per_player,
        "latest_month": latest_month,
        "max_months": max_months,
        "seed": seed,
        "minimum_base_seconds": minimum_base_seconds,
    }
    if base_coverage is not None:
        spec.update(
            coverage_version=3,
            base_coverage=str(base_coverage),
            base_coverage_sha256=sha256_file(base_coverage),
        )
    with run_lock(report_dir):
        report_dir.mkdir(parents=True, exist_ok=True)
        config_path = report_dir / "collection_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != spec:
            raise ValueError("GM collection settings changed; use another report directory")
        atomic_json(config_path, spec)
        profiles = []
        for name, expected_id in PLAYERS.items():
            url = f"https://api.chess.com/pub/player/{name}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "chess-ai-archive-coverage/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            if data.get("player_id") != expected_id or data.get("title") != "GM":
                raise ValueError(f"Public account identity changed: {name}")
            profiles.append(
                {
                    "url": url,
                    "username": name,
                    "player_id": expected_id,
                    "name": data.get("name"),
                    "title": data["title"],
                }
            )
        counts = dict.fromkeys(PLAYERS, 0)
        months = []
        for offset in range(max_months):
            number = latest.year * 12 + latest.month - 1 - offset
            month = date(number // 12, number % 12 + 1, 1)
            months.append(month.strftime("%Y-%m"))
            for name in PLAYERS:
                if counts[name] < games_per_player and download(name, raw_dir, month, month):
                    raise RuntimeError(f"Archive download failed: {name} {month}")
            progress_path = report_dir / f"coverage_{month:%Y-%m}.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                for entry in progress["inputs"]:
                    if sha256_file(Path(entry["path"])) != entry["sha256"]:
                        raise ValueError("Previously profiled GM archive changed")
            else:
                progress = profile(
                    [raw_dir / name for name in PLAYERS],
                    list(PLAYERS),
                    progress_path,
                    maximum=len(PLAYERS) * games_per_player,
                    maximum_per_player=games_per_player,
                    minimum_base_seconds=minimum_base_seconds,
                    seed=seed,
                    base_coverage=base_coverage,
                )
            counts = {name: progress["selection_ownership"].get(name, 0) for name in PLAYERS}
            print(f"Validated unique game quotas after {month:%Y-%m}: {counts}", flush=True)
            if all(count == games_per_player for count in counts.values()):
                break
        else:
            raise ValueError(
                f"Month cap reached before filling quotas: {counts}; no complete snapshot created"
            )
        final_path = report_dir / "coverage.json"
        if final_path.exists():
            final = json.loads(final_path.read_text())
            for entry in [*final["inputs"], final["snapshot"]]:
                if sha256_file(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("Frozen GM collection changed")
        else:
            final = profile(
                [raw_dir / name for name in PLAYERS],
                list(PLAYERS),
                final_path,
                maximum=len(PLAYERS) * games_per_player,
                maximum_per_player=games_per_player,
                minimum_base_seconds=minimum_base_seconds,
                seed=seed,
                snapshot_dir=report_dir / "source_games",
                base_coverage=base_coverage,
            )
        result = {
            "format": "self-improving-chess-ai.gm-archive-collection",
            "version": 3 if base_coverage is not None else 2,
            "profiles": profiles,
            "months_considered": months,
            "archive_months_loaded": sorted(
                {Path(entry["path"]).stem for entry in final["inputs"]}
            ),
            "selection_ownership": counts,
            "games": final["selected_games"],
            "coverage": str(final_path),
            "coverage_sha256": sha256_file(final_path),
            "snapshot": final["snapshot"],
            "training_imported": False,
            "teacher_calls": 0,
        }
        atomic_json(report_dir / "collection.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--games-per-player", type=int, default=500)
    parser.add_argument("--latest-month", default="2026-09")
    parser.add_argument("--max-months", type=int, default=12)
    parser.add_argument("--minimum-base-seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=23102036)
    parser.add_argument("--base-coverage", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.raw_dir,
                args.report_dir,
                games_per_player=args.games_per_player,
                latest_month=args.latest_month,
                max_months=args.max_months,
                seed=args.seed,
                base_coverage=args.base_coverage,
                minimum_base_seconds=args.minimum_base_seconds,
            ),
            indent=2,
        )
    )
