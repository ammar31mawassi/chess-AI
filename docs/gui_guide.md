# GUI human-play and fine-tuning guide

The GUI is a local human-versus-neural chess workbench. It makes repeated manual games and an
explicit end-of-session fine-tune comfortable without turning another chess application into an
input source or silently treating every click as training data.

## Why this interface

The agreed Phase 1 track uses Tkinter, which ships with ordinary Python on Windows and keeps the
project local and lightweight. A browser server, JavaScript build, account, cloud database, and
network connection are unnecessary. It is one single-window workbench: the chessboard is the primary
surface, while a compact side panel groups model/game setup, current status, move history, data
collection, and game actions.

The UI reflects the same boundaries as the rest of the repository:

- `python-chess` remains the only authority for legal moves, check, mate, draws, promotion, and
  results.
- The opponent is always a checkpoint-backed neural agent. Random and minimax setup is intentionally
  absent from this screen.
- Neural inference runs outside Tkinter's event loop. While the model is thinking, move inputs are
  disabled and the status is explicit; the window should remain responsive.
- No external chess program, website, screen, window, process, or accessibility API is inspected or
  controlled.

## Start a game

From the repository root with the virtual environment active:

```powershell
python -m chess_ai gui `
  --human-color white `
  --device cuda `
  --training-enabled
```

Use `--human-color black` to play the other side. If `--checkpoint` is omitted, discovery prefers the
newest successful `checkpoints/human_sessions/**/best.pt`, then checks
`checkpoints/gpu_first/best.pt`, `checkpoints/dev/best.pt`, the newest other
`checkpoints/**/best.pt`, and finally the newest `checkpoints/**/last.pt`. If none exists, the field
stays blank for you to browse. `--device auto` uses CUDA when the installed PyTorch build can access
it and otherwise uses CPU.

Every launch reserves a unique GUI session ID before training can begin. The ID names a generated
configuration, a new checkpoint directory, and a new metrics file, so clicking the training button
cannot collide with a previous run:

```text
data/gui_sessions/<session-id>/training.yaml
checkpoints/human_sessions/<session-id>/
data/metrics/human_sessions/<session-id>.jsonl
```

Starting another game does not discard these reservations. After one training cycle finishes, the
workbench reserves fresh paths for the next cycle.

The command options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint PATH` | local discovery | Neural model used for the opponent |
| `--human-color white\|black` | `white` | Human's color for the first game |
| `--device auto\|cpu\|cuda` | `auto` | PyTorch inference device |
| `--pgn-dir PATH` | `data/games/human_gui` | Independent game-record directory |
| `--dataset-path PATH` | `data/datasets/human_gui.pt` | Dedicated confirmed-human dataset |
| `--training-enabled` | off | Preselect the visible collection opt-in |

Omitting `--training-enabled` does not prevent play or PGN saving. It means the human-data option
begins off.

## Board and workbench behavior

The window is titled **Neural Chess Workbench**. Its right side contains **Game setup**, **Game
controls**, **Automatic training session**, **Status**, and **Move history** sections. Setup keeps the
**Checkpoint**, **Play as**, **Device**, **Learn from this game (confirm at end)**, and **Training dataset** choices visible;
**Browse...** selects another checkpoint. Settings take effect when **New Game** starts. Controls also
provide **Undo Turn**, **Resign**, **Flip Board**, and **I'm Done — Train AI**. The last control ends
the collection cycle, not the current chess game. It remains disabled until the current game is
finished or resigned and its save/collection decision is complete, so it cannot discard the board.

Click one of your pieces, then click a highlighted legal destination. Clicking another movable piece
changes the selection. A promotion opens an explicit queen/rook/bishop/knight choice rather than
guessing. Board orientation initially follows the selected human color and can be flipped without
changing the game. Status and move history report whose turn it is, check/game-over state, model
activity, and played moves.

**Undo Turn** removes the most recent human decision and any neural reply; an undone move is also
removed from the pending human labels. **Resign** records a real loss, saves its PGN, and—only when
the two collection confirmations are satisfied—can label that completed game's human positions.
If saving the PGN or confirmed examples fails, that button becomes **Retry Save** so a transient
failure does not silently discard the completed result.

The workbench uses a small, predictable state model:

1. **Ready**: the checkpoint and game settings are valid and the board accepts the human move.
2. **AI thinking**: board/game-changing inputs are temporarily disabled until one legal neural move
   returns or an error is reported.
3. **Game complete**: the result and termination are fixed; the PGN is saved and the separate
   training-data decision is offered.
4. **New game**: transient selection/history state is reset while the chosen model and cumulative
   dataset path remain visible.
5. **Training**: **I'm Done — Train AI** launches a background training process. Training progress is
   shown without running neural optimization in Tkinter's event loop. Starting another training run
   is disabled, and closing the window is blocked, until that process exits.

Keyboard shortcuts mirror visible controls: `Ctrl+N` starts a new game, `Ctrl+O` opens the checkpoint
browser, `Ctrl+Z` undoes the last human turn, `Ctrl+F` flips the board, `Ctrl+Enter` starts confirmed
session training when eligible, and `Escape` clears the selected square. A shortcut is never
required to complete a game.

## What is saved

PGN evidence and training examples are deliberately independent. Every normally completed or
resigned game is saved automatically under `data/games/human_gui/`, whether data collection is on or
off. A PGN does not become a training dataset merely because it exists.

Training collection has two visible gates:

1. Select **Learn from this game (confirm at end)** before starting the game (or launch with
   `--training-enabled`).
2. After the game has a completed result, answer a second confirmation whose safe default is **No**.
   It warns that only human moves are added and that mistakes or losing moves can be poor labels.

When both gates are satisfied:

- only positions immediately before a **human** move become examples;
- the policy target is one-hot for the move the human chose;
- AI moves never become policy labels;
- a human win labels those human-to-move positions `+1`, a loss labels them `-1`, and a draw labels
  them `0`;
- checkmate, rule-based draws, or an explicitly recorded resignation can make the in-memory examples
  eligible, but only the post-result confirmation persists them;
- reset, abandoned, or otherwise unfinished games contribute zero examples.

The dedicated file is cumulative across GUI launches and training cycles. It records human-GUI
provenance and rejects incompatible dataset kinds, old human
GUI versions, missing opt-in markers, duplicate game IDs, or non-human policy sources. Existing data
is validated before append, concurrent GUI writers are serialized with a file lock, and the
replacement is atomic. This prevents accidental mixing with generated or external benchmark
datasets and avoids lost updates when two local windows finish together.

## Build a less biased small dataset

Alternate colors rather than playing every game as White:

```powershell
python -m chess_ai gui --human-color white --device cuda --training-enabled

python -m chess_ai gui --human-color black --device cuda --training-enabled
```

Keep losses and draws if they are genuine; selecting only wins distorts both the position mix and
value targets. Still review every opt-in. More manual games add examples, but repeated openings from
one player are highly correlated and are not equivalent to broad, high-quality supervision.

## Train from the workbench

After the completed games you want have passed both collection gates, click **I'm Done — Train AI**.
If another game is active, finish or resign it and complete the post-game decision first.
The workbench launches `chess_ai train` in a background process using:

- the checkpoint selected for the current GUI training cycle as `--init-checkpoint`;
- the cumulative confirmed-human dataset, normally `data/datasets/human_gui.pt`;
- the selected `auto`, `cpu`, or `cuda` device; and
- that cycle's reserved `data/gui_sessions/<session-id>/training.yaml`.

The generated configuration points to `checkpoints/human_sessions/<session-id>/` and
`data/metrics/human_sessions/<session-id>.jsonl`. These paths are unique, so training never
overwrites the source checkpoint or an earlier human candidate. The source remains available for
comparison even if training fails.

Training is a separate process rather than work performed by the Tk event loop. The window reports
that it is training and blocks closing until the process finishes, preventing an accidental exit
from abandoning a live run. When training succeeds, its `best.pt` is selected in the current window
so **New Game** uses the new neural candidate. A new session ID and fresh config/checkpoint/metrics
paths are then reserved for the next training cycle, whose source is the checkpoint selected at that
time.

The button does not bypass collection consent. It only sees examples already present in the
cumulative dataset. An unfinished game, a game whose collection checkbox was off, or a completed
game declined at the post-game prompt contributes nothing. Previously confirmed games remain in the
dataset and are trained again in later cycles unless you deliberately select a different dataset
path.

Even the low-learning-rate settings can overfit a tiny personal dataset. Warning signs include
falling training loss alongside worsening validation loss, memorized openings, less varied play, or
worse arena results. A cumulative dataset can still be narrow and correlated; adding more games from
the same openings does not guarantee improvement and can degrade the model. Collect at least two
confirmed games before expecting a validation split. With only one game, training keeps it in the
training split and reports validation metrics as unavailable; two games are still far too few to
establish improvement.

## Manual CLI fine-tuning remains available

The GUI button is a convenience around the existing protected initialization mode. To run a
reserved session yourself, use its generated YAML and the source checkpoint shown in the workbench:

```powershell
python -m chess_ai train `
  --config data/gui_sessions/<session-id>/training.yaml `
  --init-checkpoint <source-checkpoint> `
  --device cuda
```

Replace both angle-bracket placeholders with real paths before running the command. Alternatively,
copy `configs/human_gui.yaml`, change `training.checkpoint_dir` and `training.metrics_path` to unused
run-specific paths, and invoke the same command. `--init-checkpoint` loads the source architecture
and weights but begins with a fresh optimizer, metrics file, and epoch 1. It is distinct from
`--resume`, and the two flags cannot be combined. Manual initialization refuses to overwrite its
source, an existing candidate, or an existing metrics file.

Compare a frozen session candidate with its preserved source across both colors before deciding it
is better. Twenty games are only a quick screen, not proof of improvement; retain both checkpoints
and review the PGNs, validation metrics, and broader opponents before any manual promotion decision.

## Troubleshooting

- **No checkpoint found**: pass an explicit existing `--checkpoint` path.
- **CUDA requested but unavailable**: install a CUDA-enabled PyTorch build or use `--device cpu`;
  `python -m chess_ai doctor --device cuda` reports the current state.
- **Moves do not respond**: check the status first. Inputs are intentionally locked during the AI
  turn and after game completion.
- **Dataset does not exist after playing**: a PGN save is independent. The game must finish, data
  collection must be enabled, and the post-result append must be confirmed.
- **Training button has no eligible data**: finish at least one game with collection enabled and
  accept the post-game confirmation. An unfinished or declined game is intentionally absent.
- **The window will not close**: closing is intentionally blocked while the background trainer is
  active. Wait for its success or error result so the run is not abandoned accidentally.
- **A successful model is not used**: verify that the training status reports success. The resulting
  `checkpoints/human_sessions/<session-id>/best.pt` should become the selected checkpoint for the next
  game; a failed run keeps the prior source selected.
- **Validation metrics are unavailable (`n/a`/`null`)**: collect at least two completed games with
  distinct game IDs. Whole games are kept together during the train/validation split; training can
  still run with one game, but that result is not a meaningful evaluation.
- **Initialization refuses the output directory**: keep the source checkpoint where it is and set
  `training.checkpoint_dir` to a different candidate directory. If it reports an earlier candidate,
  also choose a new `training.metrics_path` or intentionally archive the old run first.
