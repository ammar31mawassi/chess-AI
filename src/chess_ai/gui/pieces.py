"""Offline, size-cached Chessnut artwork for the Tk board."""

import tkinter as tk
from pathlib import Path

import chess
from PIL import Image, ImageTk

ASSET_DIRECTORY = Path(__file__).with_name("assets") / "chessnut"


class PieceImages:
    """Retain Tk image references, caching only the current rendered size."""

    def __init__(self, master: tk.Misc) -> None:
        self.master = master
        self._sources: dict[str, Image.Image] = {}
        self._images: dict[str, ImageTk.PhotoImage] = {}
        self._size = 0
        for color in "wb":
            for kind in "kqrbnp":
                name = color + kind.upper()
                with Image.open(ASSET_DIRECTORY / f"{name}.png") as source:
                    self._sources[name] = source.convert("RGBA")

    def get(self, piece: chess.Piece, size: int) -> ImageTk.PhotoImage:
        """Resize from the original art, never from an already resized image."""

        size = max(1, size)
        if size != self._size:
            self._images.clear()
            self._size = size
        name = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()
        if name not in self._images:
            resized = self._sources[name].resize((size, size), Image.Resampling.LANCZOS)
            self._images[name] = ImageTk.PhotoImage(resized, master=self.master)
        return self._images[name]
