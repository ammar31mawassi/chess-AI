"""Download public monthly Chess.com PGNs without importing them into training."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "downloaded_datasets" / "chesscom"


def month_arg(value: str) -> date:
    """Parse an inclusive calendar month."""
    if re.fullmatch(r"\d{4}-\d{2}", value):
        try:
            return date.fromisoformat(value + "-01")
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("Use a valid month in YYYY-MM format")


def months_between(start: date, end: date) -> list[str]:
    first = start.year * 12 + start.month - 1
    last = end.year * 12 + end.month - 1
    return [f"{index // 12:04d}/{index % 12 + 1:02d}" for index in range(first, last + 1)]


def fetch(url: str, user_agent: str) -> bytes:
    """Make serial requests with bounded retries for transient failures."""
    for attempt in range(5):
        delay = float(2**attempt)
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code != 429 and not 500 <= exc.code <= 599:
                raise RuntimeError(
                    f"HTTP {exc.code}: {url}. Check the username for 404; "
                    "for 403, try again later with --user-agent containing contact details."
                ) from exc
            retry_after = exc.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == 4:
                raise RuntimeError(f"Network failure for {url}: {exc}") from exc
        if attempt == 4:
            raise RuntimeError(f"Request failed after 5 attempts: {url}")
        print(f"Retrying in {delay:g}s: {url}", flush=True)
        time.sleep(delay)
    raise AssertionError("Unreachable")


def check_pgn(payload: bytes) -> None:
    """Reject empty/HTML/error responses; full chess validation belongs to import."""
    text = payload.decode("utf-8-sig")
    if not text.lstrip().startswith('[Event "'):
        raise ValueError("Response is empty or lacks a PGN Event header")


def download(
    username: str,
    output: Path,
    start: date,
    end: date,
    *,
    refresh: bool = False,
    user_agent: str | None = None,
) -> int:
    """Save one file per available month; return nonzero if any download fails."""
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]{1,100}", username):
        raise ValueError("Username must contain only letters, digits, underscores, or hyphens")
    current = datetime.now(UTC).date().replace(day=1)
    if start > end or end > current:
        raise ValueError("Start must not follow end, and end must not be in the future")
    agent = user_agent or f"chess-ai-pgn-downloader/1.0 (Chess.com username: {username})"
    base = f"https://api.chess.com/pub/player/{username}/games"
    index = json.loads(fetch(f"{base}/archives", agent))
    if not isinstance(index, dict) or not isinstance(index.get("archives"), list):
        raise ValueError("Unexpected archive index: expected an archives list")
    if not all(isinstance(url, str) for url in index["archives"]):
        raise ValueError("Unexpected archive index: archive URLs must be strings")
    available = {url.lower() for url in index["archives"]}
    destination = output / username
    destination.mkdir(parents=True, exist_ok=True)
    saved = skipped = empty = failed = 0
    print(f"Saving raw PGNs to {destination.resolve()}", flush=True)
    for month in months_between(start, end):
        archive = f"{base}/{month}"
        if archive not in available:
            print(f"{month}: no archive listed", flush=True)
            empty += 1
            continue
        target = destination / (month.replace("/", "-") + ".pgn")
        temporary = target.with_suffix(".pgn.part")
        try:
            if target.exists() and not refresh and month != current.strftime("%Y/%m"):
                check_pgn(target.read_bytes())
                print(f"{month}: already downloaded", flush=True)
                skipped += 1
                continue
            payload = fetch(archive + "/pgn", agent)
            check_pgn(payload)
            temporary.write_bytes(payload)
            temporary.replace(target)
            saved += 1
            print(f"{month}: saved {len(payload):,} bytes", flush=True)
        except (RuntimeError, ValueError, OSError) as exc:
            failed += 1
            print(f"{month}: FAILED: {exc}", file=sys.stderr, flush=True)
    print(f"Finished: {saved} saved, {skipped} existing, {empty} unlisted, {failed} failed.")
    if failed:
        print("Rerun to retry failures. Use --refresh to replace invalid existing PGNs.")
    return int(failed > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Your Chess.com username (no password)")
    parser.add_argument("--start", type=month_arg, default=month_arg("2018-11"))
    parser.add_argument("--end", type=month_arg, default=datetime.now(UTC).date().replace(day=1))
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Parent download folder"
    )
    parser.add_argument("--refresh", action="store_true", help="Download existing months again")
    parser.add_argument("--user-agent", help="Optional application name and contact information")
    args = parser.parse_args()
    try:
        return download(
            args.username,
            args.output,
            args.start,
            args.end,
            refresh=args.refresh,
            user_agent=args.user_agent,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
