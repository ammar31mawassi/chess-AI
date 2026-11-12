# Lichess BOT connection

The user authorized the dedicated `ZaatarTheCat` account's official Bot API integration on
2026-11-05. This is an explicit exception to the offline boundary for that account and API only.
It does not authorize browser/GUI automation, assistance on human accounts, or automatic training.
The user starts and stops the live service; coding agents may configure it and verify readiness.

## Connected model and challenge policy

For `chess-bot 100`, `chess-bot human`, and `chess-bot lvl <1-8>`, see
[the terminal commands](lichess_terminal.md).
Default `run` service behavior continues to accept only incoming casual human challenges.
Numeric terminal batches now default to five concurrent BOT games; `--concurrency N` overrides
the limit and repeating the same target resumes the latest incomplete batch, including legacy
single-game sessions. Human/computer terminal modes remain single-game.

`configs/lichess_bot.yaml` pins the registered full-corpus-plus-puzzles scratch epoch-70 champion,
SHA256 `c10650a082e04cb6f8685ea7f4e00f7241a1e5c06061be08e808eb3145fc0c9a`.
Startup verifies the champion checksum and registry settings. The model loads once and
warms up before accepting challenges. No Stockfish, book, tablebase, or online evaluation lookup
participates in choosing moves. There is no live training, checkpoint update, or automatic promotion.

The pilot accepts registered **human opponents**, **casual standard chess**, initial clocks
from **5 to 30 minutes**, and increments from **3 to 30 seconds**. Use **5+3** or **10+5** initially.
One pending challenge or active game reserves the single slot. Other challenges are declined with
a reason. The default `run` service sends no outgoing challenges. The bounded `chess-bot N`
command explicitly challenges bots. Neither mode sends chats, accepts takebacks, enters
tournaments, or plays rated games.

The API adapter reconstructs the full server move history with `python-chess` on every state update.
Production PUCT's side-to-move values and move ordering are unchanged. An online-only subclass
checks a soft deadline between traversals. The configuration sets a five-second maximum and a
one-million-traversal cap. The remaining clock and increment can shorten the budget. At 500 ms
or less it chooses a deterministic legal emergency move. A traversal in progress can overrun the
deadline. GPU training should run separately from live play.

## Credentials

The token is never stored in YAML, source, Git, PGNs, or logs. On Windows the hidden-input
installer stores a versioned DPAPI-encrypted credential under:

```text
%LOCALAPPDATA%\chess-ai\lichess-token.dpapi
```

Only the same Windows user can decrypt it through DPAPI. `CHESS_AI_LICHESS_TOKEN` is an optional
environment override, also usable on other operating systems. Tokens pasted into chat should be
rotated before public use. Create the replacement while logged in as the BOT account at
[Lichess token settings](https://lichess.org/account/oauth/token), with **bot:play** permission.
Enter it through the hidden prompt; do not paste it into commands or commit it.

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
& .\.venv\Scripts\python.exe scripts/lichess_bot.py store-token
if ($LASTEXITCODE -ne 0) { throw 'Credential storage failed' }
& .\.venv\Scripts\python.exe scripts/lichess_bot.py check
if ($LASTEXITCODE -ne 0) { throw 'Lichess readiness check failed' }
```

`check` verifies account identity, bot permission, checkpoint hash/settings, device availability,
and ongoing-game count. It prints public account information only. It never starts a stream,
accepts a challenge, submits a move, or loads the model for inference. It reports `ready: false`
until the account has the BOT title.

## Account upgrade

Lichess BOT conversion is **permanent** and requires an account with zero previous games.
The setup command pins the account identity, checks `bot:play`, checks zero games, and requires
an explicit upgrade flag. It is a separate command from running the bot and does not begin play.
It is idempotent when the account is already a BOT.

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
& .\.venv\Scripts\python.exe scripts/lichess_bot.py upgrade --confirm-bot-upgrade
if ($LASTEXITCODE -ne 0) { throw 'BOT upgrade failed' }
```

## Bring the bot online

The user launches this process. Keep the computer awake and connected while games are running.

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& .\.venv\Scripts\python.exe scripts/lichess_bot.py run
if ($LASTEXITCODE -ne 0) { throw 'Lichess bot stopped with an error' }
```

Wait for `bot_ready`, then visit [ZaatarTheCat](https://lichess.org/@/ZaatarTheCat) yourself and send
a standard, casual 5+3 or 10+5 challenge from your human account. Share the same profile link with
friends. The service uses API streams only; it never controls a browser or chess GUI.

Press Ctrl+C once to stop. Prefer stopping between games: clocks keep running if the service is
stopped during a game. Restarting the same command recovers ongoing games from the server and
archives recorded games that finished while offline. A second process is rejected by the run lock.
After a hard kill, verify the PID in the sibling `.lichess_bot.lock` is no longer running before
manually deleting only that stale lock. Never delete the game journals to fix a lock.

## Reliability and output formats

Account and game streams reconnect on transient failures. HTTP 429 establishes a 60-second
cooldown. Authentication failures stop the service. Move submissions are not blindly retried:
after an uncertain response the game stream reconnects and rebuilds the authoritative position.
Duplicate/stale states do not resubmit acknowledged moves. Unsupported state or history changes
stop the pilot rather than continuing with a corrupt board. Opponent chat is ignored.

Lichess automatically claims threefold repetition for BOT accounts. When a claim requires an
intended move, the adapter finds that move through python-chess and plays it. The `draw/yes`
action handles existing repetition claims; `claim-draw` is for disconnected opponents and is not
used as a repetition/fifty-move shortcut.

Each game is recorded under `data/games/evaluation/lichess_bot/<game-id>/`:

- `binding.json`: version-1 identity, checkpoint, clock/search settings, evaluation-only provenance.
- `state.json`: version-1 latest server history, clocks, result/status, completion marker.
- `game.pgn`: complete server move history, atomic updates, model checksum.
- `decisions.jsonl`: move, elapsed time, remaining clock, and traversal cap.

Unfinished/aborted games retain `*`; a server winner or draw status supplies the recorded result.
These artifacts remain **evaluation-only** and are already inside the project's protected
evaluation directory. They are not imported into any dataset or used to update weights.
Checkpoint and dataset formats are unchanged; the new bot config and journal formats are version 1.

Offline tests mock all Bot API moves, streams, and challenges. Coverage includes challenge rules,
single-slot reservation, retries/duplicates, restart recovery, legal move/history validation,
draw handling, terminal PGNs, clocked-search equivalence with adequate time, HTTP/token redaction,
and DPAPI encryption round trips. Readiness can be verified live without playing a game.

Primary protocol references: [Bot API](https://lichess.org/api#tag/Bot),
[BOT upgrade](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-account-upgrade.yaml),
[game stream](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-game-stream-gameId.yaml),
[Lichess draw handling](https://github.com/lichess-org/lila/blob/master/modules/round/src/main/Drawer.scala).
