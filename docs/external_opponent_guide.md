# Manual external-opponent guide

External mode is a human-operated benchmark. It never reads, clicks, inspects, or controls The Chess
Lv.100 (or any other application).

## Before a game

1. Choose the checkpoint and record enough training/arena evidence to justify testing it.
2. Open the external chess application yourself.
3. Select a color and difficulty. Do not change either mid-game.
4. Run `python -m chess_ai external --checkpoint CHECKPOINT --ai-color white` (or `black`).
5. Enter an opponent label and difficulty exactly enough to group the result later.

When the AI moves, the terminal prints a UCI move such as `e2e4`. Enter it manually in the external
application. Then type the external opponent's reply into the terminal.

## Session commands

- `show`: print the current board again.
- `fen`: print the exact position string, useful for checking transcription.
- `undo`: undo the last full turn where possible; synchronize the external app manually too.
- `help`: print the command summary.
- `resign`: record that the side to move resigned, after confirmation.
- `quit`: leave without inventing a chess result; the partial PGN can still be retained.

Illegal and malformed moves are rejected without changing the board. Promotion needs a suffix:
`e7e8q`, `e7e8r`, `e7e8b`, or `e7e8n`.

## Saved evidence

The PGN headers include date, checkpoint/model label, external opponent, AI color, and result. A
separate append-only JSONL record includes opponent, difficulty, checkpoint, color, result, move
count, and UTC timestamp. `python -m chess_ai report` aggregates these by checkpoint and level.

If the board reaches a claimable draw or the external program declares a different result, the
terminal asks for confirmation rather than guessing. Always make the recorded PGN and external app
agree.

## Training separation

External games are evaluation data by default. They are never read by `train` or appended to a
dataset automatically. To validate a PGN for a later, deliberate workflow, use:

```powershell
python -m chess_ai import-external --pgn data/games/external/GAME.pgn `
  --destination-dir data/games/imported_external `
  --confirm-evaluation-data-import
```

The command replays every move through `python-chess`, checks the result/header, and writes a
separately labeled PGN plus an `evaluation_origin=true` manifest. It deliberately does not create a
training dataset. Review it before any later conversion. Keeping benchmark games out of the training
set preserves the meaning of later evaluation.

## Interpreting results

One win is anecdotal. Alternate colors, repeat the same opponent level, retain losses and draws, and
compare cumulative score with uncertainty in mind. A fixed opponent alone can reward narrow
adaptation. Also test random/minimax agents, earlier checkpoints, varied openings, and eventually
tactical/UCI benchmarks from the roadmap.
