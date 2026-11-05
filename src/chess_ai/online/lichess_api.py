"""Small fixed-origin Lichess HTTP client with sanitized errors and NDJSON streams."""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator, Mapping
from typing import Any


class LichessError(RuntimeError):
    """A sanitized API failure; never includes credentials or response bodies."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise LichessError("Unexpected redirect from Lichess; request stopped")


class LichessAPI:
    def __init__(self, token: str) -> None:
        self._token = token
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._lock = threading.RLock()
        self._cooldown = 0.0

    def _open(
        self,
        path: str,
        *,
        data: bytes | None = None,
        method: str = "GET",
        content_type: str = "application/x-www-form-urlencoded",
    ) -> Any:
        if not path.startswith("/") or path.startswith("//") or "\n" in path:
            raise LichessError("Invalid API path")
        request = urllib.request.Request(
            "https://lichess.org" + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self._token,
                "User-Agent": "chess-ai-epoch99/1.0 (Lichess Bot API)",
                "Content-Type": content_type,
                "Accept": "application/x-ndjson, application/json",
            },
        )
        with self._lock:
            try:
                if time.monotonic() < self._cooldown:
                    raise LichessError("Lichess rate-limit cooldown is active", status=429)
                return self._opener.open(request, timeout=20)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    self._cooldown = time.monotonic() + 60.0
                status = exc.code
                exc.close()
                raise LichessError(f"Lichess returned HTTP {status}", status=status) from None
            except (OSError, urllib.error.URLError):
                raise LichessError("Connection to Lichess failed") from None

    def get(self, path: str) -> dict[str, Any]:
        with self._lock, self._open(path) as response:
            raw = json.load(response)
        if not isinstance(raw, dict):
            raise LichessError("Unexpected Lichess response shape")
        return raw

    def post(self, path: str, fields: Mapping[str, str] | None = None) -> dict[str, Any]:
        data = urllib.parse.urlencode(fields or {}).encode()
        with self._lock, self._open(path, data=data, method="POST") as response:
            raw = json.load(response)
        if not isinstance(raw, dict):
            raise LichessError("Unexpected Lichess response shape")
        return raw

    def account(self) -> dict[str, Any]:
        return self.get("/api/account")

    def scopes(self) -> set[str]:
        with (
            self._lock,
            self._open(
                "/api/token/test",
                data=self._token.encode(),
                method="POST",
                content_type="text/plain",
            ) as response,
        ):
            raw = json.load(response)
        item = raw.get(self._token) if isinstance(raw, dict) else None
        if not isinstance(item, dict):
            raise LichessError("Lichess token is invalid")
        return set(str(item.get("scopes", "")).split(","))

    def stream(
        self, path: str, stop: threading.Event, *, connected: Callable[[], None] | None = None
    ) -> Generator[dict[str, Any], None, None]:
        with self._open(path) as response:
            if connected is not None:
                connected()
            while not stop.is_set():
                try:
                    line = response.readline(1_000_001)
                except (OSError, http.client.HTTPException):
                    raise LichessError("Lichess stream connection interrupted") from None
                if not line:
                    return
                if len(line) > 1_000_000:
                    raise LichessError("Oversized Lichess event")
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    raise LichessError("Malformed Lichess stream event") from None
                if not isinstance(raw, dict):
                    raise LichessError("Unexpected Lichess event shape")
                yield raw
