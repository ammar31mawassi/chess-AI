# Download your Chess.com game archive

The standalone downloader uses Python's standard library and the public read-only API.
No password, API key, browser, or additional packages are needed. It checks every month
from November 2018 through the current UTC month, using the archive index to skip months
without games. Only completed games available through that API are included.

Run in PowerShell:

```powershell
Set-Location "C:\Users\ammar\OneDrive\Desktop\chess-AI"
$chessUsername = Read-Host "Your Chess.com username"
.\.venv\Scripts\python.exe .\scripts\download_chesscom_games.py --username "$chessUsername" --start 2018-11
```

The default destination is `downloaded_datasets/chesscom/<username>/`, relative to the
repository regardless of your working directory. Each available month gets a separate
`YYYY-MM.pgn` file containing that month's games. This directory is ignored by Git.
The script creates the account subfolder automatically.

Rerun the command after interruption: existing historical PGNs are retained, while the
current month is downloaded again to pick up newly completed games. Use `--refresh` to
redownload all selected months (also useful for repairing damaged files or refreshing a
previously partial month after the calendar changes). `--start YYYY-MM`, `--end YYYY-MM`,
and `--output "D:\ChessArchive"` override the inclusive range and parent destination.
Avoid running two downloaders against the same destination at once.

Requests run sequentially with a 60-second timeout and up to five attempts for network,
429, and server errors. Numeric Retry-After headers are respected. Failures are printed
and produce exit code 1; other months continue. A missing/invalid username or archive
index failure stops the run immediately. Writes use `.pgn.part` followed by replacement,
so an interrupted transfer does not become a completed PGN. If Chess.com rejects your
client, `--user-agent "my-chess-archive/1.0 (contact: you@example.com)"` supplies contact
information. API caching can delay visibility of recent games.

These are raw external PGNs. The downloader checks their encoding and Event header,
but does not validate chess moves, run Stockfish, or import games into training.
Before later teacher analysis/import, explicitly validate complete games using
`python-chess`, handle variants, deduplicate, and assign whole games to train/holdout
groups. Keep personal archive games separate from existing evaluation games.

Sources: [Chess.com PGN download guide](https://support.chess.com/en/articles/8705305-how-do-i-get-a-pgn-of-my-game)
and [official Published-Data API documentation](https://www.chess.com/news/view/published-data-api).
## Stockfish labeling after download

The [personal-game Stockfish workflow](personal_stockfish_training.md) performs a separate explicit
import, validates/freeze-snapshots the local PGNs, and labels every unique position. Downloaded moves
are position sources; Stockfish move preferences and evaluations supply the training targets.
