"""Explicit maintainer-only download and rasterization of the bundled Chessnut art.

Install resvg-py==0.5.0 before running. Gameplay never calls this script or the network.
"""

from pathlib import Path
from urllib.request import urlopen

import resvg_py

REVISION = "2b8eaf14a31edad7e9deb53b1473e1d4857868a9"
SOURCE = f"https://raw.githubusercontent.com/LexLuengas/chessnut-pieces/{REVISION}"
DESTINATION = Path(__file__).resolve().parents[1] / "src/chess_ai/gui/assets/chessnut"


def main() -> None:
    """Keep original vectors and license notices alongside 256px transparent PNGs."""

    DESTINATION.mkdir(parents=True, exist_ok=True)
    names = [f"{color}{piece}.svg" for color in "wb" for piece in "KQRBNP"]
    for name in [*names, "LICENSE.txt", "COPYRIGHT.txt"]:
        with urlopen(f"{SOURCE}/{name}", timeout=30) as response:
            (DESTINATION / name).write_bytes(response.read())
    for name in names:
        svg = (DESTINATION / name).read_text(encoding="utf-8")
        png = resvg_py.svg_to_bytes(svg_string=svg, width=256, height=256)
        (DESTINATION / name).with_suffix(".png").write_bytes(png)


if __name__ == "__main__":
    main()
