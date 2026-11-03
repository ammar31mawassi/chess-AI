# Repository guidance for coding agents

- Read `docs/architecture.md` and `docs/training_guide.md` before major changes.
- Delegate every chess-legality decision to `python-chess`; do not create a second rules engine.
- Preserve explicit deterministic seeds, especially in tests and data generation.
- Never automate, inspect, click, or control an external chess GUI app or website.
  The only authorized exception is a local Stockfish executable, separately executed through
  python-chess UCI facilities for offline teacher analysis. This does not authorize browser,
  online-chess, or GUI automation. Stockfish uses CPU; our neural model may use CUDA.
- Never silently mix external benchmark games into training data. Import must be explicit and validated.
- Run Ruff formatting/checks, mypy, and the full pytest suite before declaring completion.
- Update docs whenever commands, schemas, checkpoint formats, or dataset formats change.
- Prefer readable educational code over premature optimization.
- Version incompatible dataset/checkpoint changes and fail clearly when old artifacts cannot load.
- Whenever the user needs to run commands, include the exact commands directly in the response
  in copyable PowerShell code blocks, even when they are also documented in a runbook.
  Do not provide only a runbook link.
