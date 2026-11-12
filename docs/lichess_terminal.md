# Bounded Lichess matches from PowerShell

The user launches live jobs. Setup and tests do not send challenges or play games.
The existing encrypted token and pinned scratch champion are reused. No training or promotion
occurs; all games remain under `data/games/evaluation/lichess_bot`.

Install the shortcut (idempotent; preserves and backs up existing profiles):

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
. .\scripts\install_chess_bot_command.ps1
```

This adds a dot-source line to the current user's all-host profiles for Windows PowerShell
and PowerShell. For another already-open terminal, load the wrapper once:

```powershell
. 'C:\Users\ammar\OneDrive\Desktop\chess-AI\scripts\chess-bot.ps1'
```

New terminals load it automatically unless started with `-NoProfile` or profile execution is
disabled by system policy. The command works from any working directory.

```powershell
chess-bot 100
```

Requests a 100-game bot batch with **up to five simultaneous games**, casual standard 5+3,
random colors. Repeating the same target resumes the latest incomplete matching batch and
retains its finished results. A completed latest batch starts a new batch, subject to the
remaining daily allowance. Smaller targets from 1 through 100 work too. Override concurrency:

```powershell
chess-bot 100 --concurrency 3
```

The override must be a positive integer and is capped at the batch target. Concurrency is a
maximum: opponents may decline, take time to accept, or be unavailable. Online bots may decline these
settings, so neither finding 100 opponents nor finishing in one day is guaranteed. A bot is
not challenged again for 30 minutes in the same session. Incoming challenges are declined in
this mode so they cannot bypass the match counter or daily check. Games print WON/LOST/DRAW,
score, opponent, color, server reason and public link. Aborts are reported separately as
unfinished and conservatively count toward the allowance.

[Lichess bot client documentation](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot#note-on-daily-game-limits)
documents 100 bot-vs-bot games per day. Human games do not consume that allowance. Since a
precise reset boundary is not documented there, the scheduler conservatively counts the last
24 hours from server history plus the local started-game ledger. This can stop earlier than
the server would. Restarting does not reset the ledger. Server HTTP 429, authorization failure,
or an ambiguous outgoing POST stops new matchmaking rather than risking duplicate challenges.
Remaining allowance is checked before each challenge; it does not sleep overnight to reset.
Pending offers reserve both a game slot and daily allowance, so five outstanding offers cannot
spend the last available game five times. Outgoing requests are sequential, at least five seconds
apart. Slots refill as games finish. On quota exhaustion or a matchmaking error, new challenges
halt while already-live games finish. A shared model/search lock protects CUDA inference; its
queue time is deducted from each game's clock. The bot uses a soft deadline of up to five seconds
per move and a one-million-traversal cap. The game clock can shorten the search budget.

```powershell
chess-bot human
```

BOT accounts cannot enter the normal human matchmaking pool. This command **waits for one
incoming human challenge**, casual standard rapid **10+5**, then exits after reporting the
result. It prints [the bot profile](https://lichess.org/@/ZaatarTheCat) for a human to open and
challenge manually. It does not search or message random people. The normal unlimited
incoming-human service remains available separately with `scripts/lichess_bot.py run`.

Run only one service for this account at a time; the shared run lock rejects a second process.
Ctrl+C stops the process and prints the partial summary. If a game was live, restart the same
mode promptly to recover it. **Server clocks do not pause while this process is offline.**
Wait for shutdown to return to the PowerShell prompt before restarting; the account lock stays
held until all game writers have stopped. Use the same numeric target to resume its remaining
games; concurrency can change between restarts. For example, 37 finished games resume at game
38/100. If reducing concurrency below the number of existing games, those games are all recovered
first; new offers wait for slots to free. Existing game journals and quota ledger are retained.
On startup, live games reconnect immediately; new matchmaking waits approximately 35 seconds
for the account stream to replay outstanding offers from the old process and for expired offers
to clear. This also protects the five-slot/daily limit when migrating a version-1 pending offer.

For the upgrade from the original single-game runner, leave it running while code is prepared.
After the update is ready, Ctrl+C, wait for the prompt, then restart:

```powershell
. 'C:\Users\ammar\OneDrive\Desktop\chess-AI\scripts\chess-bot.ps1'
chess-bot 100
```

Its latest incomplete version-1 100-game summary is migrated into a new version-2 summary.
Finished games retain their results; per-game bindings verify the account/champion. The old
summary is preserved. Ongoing games reconnect from server history; games that finished while
offline are archived and counted. Restart promptly to reduce the risk of losing games on time.
No second bot process should run alongside the first.

## Built-in Lichess computer

```powershell
chess-bot lvl 3
```

Replace `3` with any integer from **1 through 8**. This plays **one** casual standard **10+5**
game against the built-in Lichess computer, with random color. It prints WON/LOST/DRAW and
the reason, saves the PGN and session summary, then exits. These are Lichess computer strength
levels, separate from our offline Stockfish depth D1/D2 evaluations.

This uses `POST /api/challenge/ai` with the existing `bot:play` permission. No browser or GUI
is controlled. Other incoming challenges are declined while this mode runs. The computer's
`aiLevel` must match the requested level; an ordinary user or different level is rejected.
Lichess omits registered user IDs/names for its built-in computer, so the saved game binding
uses `lichess-ai-level-N` and the PGN uses `Lichess computer level N` for that player.
Version-1 AI game bindings add `opponent_ai_level`; session summaries add `ai_level` and
use mode `ai`. Existing human/BOT artifacts remain compatible. AI games are not added to the
account-vs-account bot ledger; any server limits still apply and rejected requests stop the
session. Restart with the same `lvl N` to recover an ongoing game. Creation is never blindly
retried after an ambiguous POST response. Setup and offline tests do not create live games.

[Official computer challenge API](https://lichess.org/api#tag/Challenges/operation/challengeAi).

Session summaries are saved in `data/games/evaluation/lichess_bot/sessions/*.json`
with requested/completed counts, wins/losses/draws/unfinished games, points, stop reason, and
each result/link. `bot_starts.json` maps started game IDs to Unix timestamps and supplements
server history against export delays. Per-game version-1 bindings, state, PGN and decision
journals are unchanged. A generic server draw stays generic unless python-chess can identify
repetition, the fifty-move rule or insufficient material from the complete move history.

Numeric BOT batches use **version 2**: account/checkpoint pins, target, concurrency, assigned game
IDs, pending reservations, attempted opponents and optional legacy migration source are added.
Results are de-duplicated by game ID under a lock. Human/computer summaries remain version 1;
their commands still play one game. Incompatible batch versions or mismatched account/champion
bindings fail clearly. Daily allowance never resets just because a process is restarted.

Concurrency verification: Ruff formatting/checks and mypy passed; the full suite passed
480 tests with 6 expected environment-dependent skips. Offline coverage includes five independent
workers, legacy migration, offline terminal recovery, duplicate events, pending-offer quota/slot
reservation, startup offer replay, lower-limit restart recovery, serialized model/HTTP access,
clock queue deduction, draining on quota/rate limits and lock collisions. All seven PowerShell
blocks parse. Existing saved legacy game bindings were checked read-only against the configured
account/champion. No live process was stopped or launched and no games were created during setup.

Protocol references: [online bots](https://lichess.org/api#tag/Bot/operation/apiBotOnline),
[user games](https://lichess.org/api#tag/Games/operation/apiGamesUser),
[challenges](https://lichess.org/api#tag/Challenges/operation/challengeCreate).
