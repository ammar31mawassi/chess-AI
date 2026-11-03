"""Atomic JSON and content signatures for independently resumable artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


def replace_with_retry(source: str | Path, destination: str | Path) -> None:
    """Atomically replace a file, tolerating brief Windows sharing/access locks.

    Retry the same complete temporary file. Never unlink the destination: the last
    committed artifact must remain readable if every attempt fails.
    """
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.8))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """One writer per run. A hard-kill lock requires explicit manual recovery."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = run_dir.with_name(f".{run_dir.name}.lock")
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "run_dir": str(run_dir)}, handle)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Collection is locked: {lock}. After a hard kill, verify the recorded process has exited before removing this lock."
        ) from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
