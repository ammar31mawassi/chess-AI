"""User-scoped Windows DPAPI storage; tokens never belong in project files."""

from __future__ import annotations

import ctypes
import os
import re
from ctypes import wintypes
from pathlib import Path

PREFIX = b"chess-ai-lichess-dpapi-v1\n"


def credential_path() -> Path:
    if os.name != "nt":
        raise ValueError("Use CHESS_AI_LICHESS_TOKEN on non-Windows systems")
    return Path(os.environ["LOCALAPPDATA"]) / "chess-ai" / "lichess-token.dpapi"


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise ValueError("DPAPI credentials require Windows")
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = _Blob(len(data), buffer)
    output = _Blob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    function.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise ValueError("Windows could not protect/unprotect this user's credential")
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel32.LocalFree(output.data)


def validate_token(token: str) -> str:
    token = token.strip()
    if re.fullmatch(r"[A-Za-z0-9_]{12,512}", token) is None:
        raise ValueError("Invalid token shape; enter only the token itself")
    return token


def store_token(token: str) -> Path:
    path = credential_path()
    encrypted = PREFIX + _crypt(validate_token(token).encode(), decrypt=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encrypted)
    os.replace(temporary, path)
    return path


def load_token() -> str:
    supplied = os.environ.get("CHESS_AI_LICHESS_TOKEN")
    if supplied:
        return validate_token(supplied)
    path = credential_path()
    if not path.is_file():
        raise ValueError("No Lichess credential. Run scripts/lichess_bot.py store-token first")
    encoded = path.read_bytes()
    if not encoded.startswith(PREFIX):
        raise ValueError("Unsupported credential file")
    return validate_token(_crypt(encoded[len(PREFIX) :], decrypt=True).decode())
