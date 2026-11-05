# Repository guidance for coding agents

- Read `docs/architecture.md` and `docs/training_guide.md` before major changes.
- Delegate every chess-legality decision to `python-chess`; do not create a second rules engine.
- Preserve explicit deterministic seeds, especially in tests and data generation.
- Never automate, inspect, click, or control an external chess GUI app or website.
  The only authorized exception is a local Stockfish executable, separately executed through
  python-chess UCI facilities for offline teacher analysis. This does not authorize browser,
  online-chess, or GUI automation. Stockfish uses CPU; our neural model may use CUDA.
- User-authorized exception (2026-11-05): the dedicated `ZaatarTheCat` BOT account may
  use Lichess's official Bot API through the opt-in online adapter. This authorizes
  API account setup, incoming challenge handling, and bot moves only; no browser/GUI
  automation or engine assistance on human accounts. The user launches live bot,
  training, and evaluation jobs; agents prepare commands and may verify account/setup.
- User-authorized extension (2026-11-05): user-run `chess-bot N` may send casual
  standard challenges to online BOT accounts, up to the remaining daily allowance.
  User-authorized concurrency (2026-11-05): default five simultaneous games for
  `chess-bot N`, with an explicit `--concurrency N` override and resumable batches.
  `chess-bot human` waits for one human rapid challenge; no human matchmaking pool access.
- User-authorized extension (2026-11-05): user-run `chess-bot lvl <1-8>` may create
  one casual standard game against the built-in Lichess computer via the official API.
- Never silently mix external benchmark games into training data. Import must be explicit and validated.
- Run Ruff formatting/checks, mypy, and the full pytest suite before declaring completion.
- Update docs whenever commands, schemas, checkpoint formats, or dataset formats change.
- Prefer readable educational code over premature optimization.
- Version incompatible dataset/checkpoint changes and fail clearly when old artifacts cannot load.
- Whenever the user needs to run commands, include the exact commands directly in the response
  in copyable PowerShell code blocks, even when they are also documented in a runbook.
  Do not provide only a runbook link.
