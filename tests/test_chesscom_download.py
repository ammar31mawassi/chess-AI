"""Archive range, resume, and failure handling without contacting Chess.com."""

import importlib.util
import json
from datetime import UTC, date, datetime
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_chesscom_games.py"
PGN = b'[Event "Live Chess"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n'
BASE = "https://api.chess.com/pub/player/test_user/games"


@pytest.fixture
def downloader():
    spec = importlib.util.spec_from_file_location("chesscom_downloader", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_month_range_crosses_year_inclusively(downloader):
    assert downloader.months_between(date(2018, 11, 1), date(2019, 1, 1)) == [
        "2018/11",
        "2018/12",
        "2019/01",
    ]


def test_resume_preserves_history_and_refreshes_current(downloader, tmp_path, monkeypatch):
    current = datetime.now(UTC).date().replace(day=1)
    month = current.strftime("%Y/%m")
    destination = tmp_path / "test_user"
    destination.mkdir()
    old = destination / "2018-11.pgn"
    old.write_bytes(PGN)
    recent = destination / f"{current:%Y-%m}.pgn"
    recent.write_bytes(PGN)
    fetch = Mock(
        side_effect=[
            json.dumps({"archives": [BASE + "/2018/11", BASE + "/" + month]}).encode(),
            PGN + b"\n",
        ]
    )
    monkeypatch.setattr(downloader, "fetch", fetch)
    assert downloader.download("Test_User", tmp_path, date(2018, 11, 1), current) == 0
    assert old.read_bytes() == PGN
    assert recent.read_bytes() == PGN + b"\n"
    assert [call.args[0] for call in fetch.call_args_list] == [
        BASE + "/archives",
        BASE + "/" + month + "/pgn",
    ]


def test_bad_response_preserves_old_file_and_continues(downloader, tmp_path, monkeypatch):
    destination = tmp_path / "test_user"
    destination.mkdir()
    old = destination / "2018-11.pgn"
    old.write_bytes(PGN)
    monkeypatch.setattr(
        downloader,
        "fetch",
        Mock(
            side_effect=[
                json.dumps({"archives": [BASE + "/2018/11", BASE + "/2018/12"]}).encode(),
                b"<html>server error</html>",
                PGN,
            ]
        ),
    )
    assert (
        downloader.download(
            "test_user", tmp_path, date(2018, 11, 1), date(2018, 12, 1), refresh=True
        )
        == 1
    )
    assert old.read_bytes() == PGN
    assert (destination / "2018-12.pgn").read_bytes() == PGN
    assert not list(destination.glob("*.part"))


def test_rate_limit_retry_after(downloader, monkeypatch):
    headers = Message()
    headers["Retry-After"] = "3"
    opener = Mock(side_effect=[HTTPError(BASE, 429, "rate limit", headers, None), BytesIO(PGN)])
    sleeper = Mock()
    monkeypatch.setattr(downloader, "urlopen", opener)
    monkeypatch.setattr(downloader.time, "sleep", sleeper)
    assert downloader.fetch(BASE, "test-client") == PGN
    sleeper.assert_called_once_with(3.0)


def test_bad_username_never_requests(downloader, tmp_path, monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(downloader, "fetch", fetch)
    with pytest.raises(ValueError, match="Username"):
        downloader.download("../other", tmp_path, date(2018, 11, 1), date(2018, 12, 1))
    fetch.assert_not_called()
