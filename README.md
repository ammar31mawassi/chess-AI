# Self-Improving Chess AI

An educational, local Python project for learning how a chess AI is assembled from reliable rules,
position/move encodings, classical search, a policy-value neural network, supervised data, training,
and controlled evaluation.

**Phase 1 is a supervised bootstrap, not a full AlphaZero implementation.** It is designed to run end
to end on an ordinary computer and make every major component readable. Strong chess performance is
not guaranteed.

## Current capabilities

- Uses `python-chess` as the authority for legal moves, special moves, outcomes, draws, FEN, and PGN.
- Encodes positions as documented 18-plane tensors and moves in a deterministic 4,208-action space.
- Provides seeded random, terminal human, alpha-beta minimax, and checkpoint-backed neural agents.
- Trains a compact configurable PyTorch residual policy-value network on versioned supervised data.
- Generates one-hot policy labels from ordinary classical-agent games and soft top-k policy labels
  from exact D2 root scores in the offline teacher-batch workflow. Outcome values retain the correct
  player-to-move perspective.
- Runs a resumable teacher-correction curriculum in which a neural student plays a shallow minimax
  opponent while a stronger minimax teacher relabels the student's decision positions.
- Collects a generation-only 500-game batch from a frozen champion against D1, with paired colors,
  2-3-full-move random openings, D2 annotations on every played ply, and no training between games.
- Collects a separate 400-game, four-matchup batch on 50 shared short openings and stores per-actor
  win/draw/loss weights for later game-balanced training.
- Selects among retained epoch checkpoints using fixed D1 and D2 gameplay before final fresh audits;
  validation loss is only a gameplay tie-breaker and selection never auto-promotes.
- Splits validation by complete game, logs metrics, resumes training, and saves atomic versioned
  checkpoints.
- Plays color-switched agent tournaments, saves PGNs, measures move time, and reports approximate
  score-derived Elo without automatically promoting a model.
- Runs a human-in-the-loop external benchmark and aggregates results by checkpoint/opponent level.
- Provides a local Tkinter chess workbench for human-versus-neural games, legal-move highlighting,
  independent PGN saving, explicit opt-in collection of the human's moves, and session-scoped
  background fine-tuning that preserves its source checkpoint.
- Includes tests, a doctor command, tiny development configuration, and learning-oriented docs.

## Phase 1 non-goals

This phase does not include Monte Carlo Tree Search, PUCT, Dirichlet noise, self-play reinforcement
learning, parallel workers, automatic model promotion, Stockfish integration, online play, cloud
services, or a database. The GUI is intentionally a local human-versus-neural workbench, not a
general tournament frontend or a bridge to another chess application. The external application is
**not automated**. The Chess Lv.100 is used only as a manually operated benchmark: a person copies
the AI move into it and types its reply.

Training against one fixed opponent alone is not sufficient evidence of general improvement.

## Architecture at a glance

`python-chess` supplies rules. The `environment` layer wraps games and creates stable numeric inputs.
All agents implement one `choose_move(board)` interface. Classical agents generate versioned
examples; the trainer learns a policy and value; the greedy neural agent masks illegal policy
actions; the optional `neural-mcts` agent combines policy priors and value estimates with PUCT; the
arena and external workflow evaluate fixed checkpoints; the GUI supplies local human neural-play
input.
PGN, JSONL, datasets, and checkpoints remain local.

See [architecture](docs/architecture.md) for dependency flow, formats, orientation, and boundaries.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m chess_ai doctor
```

Python 3.11 or newer is supported. PyTorch runs on CPU; CUDA is optional and used only when available
and selected.

## Commands

```powershell
python -m chess_ai --help
python -m chess_ai doctor
python -m chess_ai gui --checkpoint checkpoints/dev/best.pt
python -m chess_ai play --white human --black random
python -m chess_ai play --white minimax --black random --depth 2
python -m chess_ai generate-data --config configs/dev.yaml
python -m chess_ai generate-data --config configs/minimax_d2_diverse_500.yaml
python -m chess_ai generate-data --config configs/minimax_d2_vs_random_teacher_500.yaml
python -m chess_ai teacher-cycle --config configs/d1_opponent_d2_teacher_100.yaml
python -m chess_ai teacher-batch --config configs/gen1_vs_d1_d2_teacher_500.yaml
python -m chess_ai mixed-batch --config configs/gen2_mixed_d1_d2_400.yaml
python -m chess_ai mixed-batch --config configs/minimax_d1_d2_d3_3000.yaml
python -m chess_ai gated-refinement --config configs/d1_gated_refinement_from_epoch399.yaml
python -m chess_ai gated-audit --config configs/d1_gated_refinement_from_epoch399.yaml
python -m chess_ai paired-audit --candidate checkpoints/candidate/best.pt --champion checkpoints/d1_gated_from_epoch399/champions/generation_0001.pt
python -m chess_ai gameplay-select --config configs/gen2_mixed_d1_d2_400_select.yaml
python -m chess_ai train --config configs/dev.yaml
python -m chess_ai arena --white minimax --black random --games 20
python -m chess_ai arena --white neural --white-checkpoint checkpoints/dev/best.pt --black random --games 20
python -m chess_ai evaluate --candidate checkpoints/dev/best.pt --champion checkpoints/older.pt --games 20
python -m chess_ai external --checkpoint checkpoints/dev/best.pt --ai-color white
python -m chess_ai report
```

Use `python -m chess_ai COMMAND --help` for command-specific arguments. Generated files go beneath
`data/` and `checkpoints/` unless overridden.

The diversified depth-2 generation configuration stores its dataset in
`data/datasets/minimax_d2_diverse_500.pt` and archives all 500 source games beneath
`data/games/minimax_d2_diverse_500/`.

The teacher-cycle configuration plays exactly 100 seeded, color-switched pairs (200 games): the
student faces Minimax D1 and Minimax D2 supplies correction labels on student turns. Cycles 1-99
perform small replay-protected updates; cycle 100 is held out so the final reported checkpoint is
actually tested. The command preserves every PGN and correction dataset, resumes from atomic state,
never auto-promotes, and reports D1 mastery only after five consecutive two-color sweeps.
After completion, extend the same state safely with `--cycles 200` (the new total target). The
runner consumes the old endpoint's held-out corrections before playing the next pair; no artifact
paths or checkpoint files need manual editing.
The current refinement also uses a normal starting position every fifth pair, letting D2 directly
correct the standard Black line without giving up the varied openings in the other pairs. Repeated
anchor lessons are retained across cycles, 25% of each correction batch is reserved for them, and a
separate two-color standard-start regression pair is saved after every trained anchor. Continue the
existing 500-cycle state with `--cycles 600`; no output path or existing file needs manual editing.

After the cycle-600 regression audit, the prepared gated continuation treats epoch 399 as a separate
gameplay champion. Candidates train in ten-pair blocks from the current champion and are compared
with it on the same fixed D1 opening suite in both colors. A candidate is promoted only for a real
score gain with no standard-start regression; rejected weights are archived and never feed the next
round. This gameplay champion is intentionally distinct from training-loss `best.pt`.
The first unseen 20-opening holdout gave generation one a narrow 4.5-to-4.0 point advantage over
generation zero without standard regression. The prepared configuration now extends the completed
10-round state to a total target of 20; gameplay gating remains active for every new candidate.

The next offline stage is defined by `configs/gen1_vs_d1_d2_teacher_500.yaml`. It freezes generation
one, plays 250 openings twice for 500 games, records complete PGNs and per-pair trace shards, and
asks D2 to score every played position after the short random prefix. The generator has no optimizer
path. `configs/gen1_vs_d1_d2_teacher_500_train.yaml` later starts a fresh three-epoch policy-only
candidate with game-balanced sampling, pair-grouped validation, and a 2x loss weight for champion
wins. `paired-audit` compares that candidate with generation one on unseen starts without promotion.

The next mixed stage uses `configs/gen2_mixed_d1_d2_400.yaml`: 50 shared two-to-three-full-move
positions feed D1-vs-D2, D2-vs-D2, neural-vs-D1, and neural-vs-D2, with both Agent-A colors. D2
supplies soft labels on every continuation ply. Winner moves receive weight 2, draw moves 1.5, and
loser moves 1. After offline training, `gameplay-select` evaluates every epoch against both D1 and
D2 and names a provisional checkpoint without modifying the champion.

For a larger classical league dataset, `configs/minimax_d1_d2_d3_3000.yaml` creates 500 shared
opening sets. Each is played as D2-vs-D1, D2-vs-D3, and D1-vs-D3 with colors reversed, producing
3,000 games. Every tenth set is a normal-start anchor; the other sets begin after 2-5 random full
moves. An independent D3 teacher supplies all soft policy targets, and collection remains resumable
without training during the games.

## Graphical human-versus-neural games

Launch the local workbench with collection enabled. It automatically selects the newest successful
human-session candidate, falling back to the original GPU checkpoint when none exists:

```powershell
python -m chess_ai gui --device cuda --training-enabled
```

Choose White or Black in the window, click a piece and destination, and use **New Game** for the
next game. Every completed or resigned game is saved as a PGN independently. Training-data
collection is a separate, visible opt-in: only the human's moves become policy targets, and only
completed games whose result you confirm are eligible. Unfinished games are excluded. GUI data stays
in the cumulative `data/datasets/human_gui.pt`; it is never mixed into another dataset silently.

Each GUI launch reserves a unique session ID and non-overwriting paths for its generated training
configuration, candidate checkpoints, and metrics:

```text
data/gui_sessions/<session-id>/training.yaml
checkpoints/human_sessions/<session-id>/
data/metrics/human_sessions/<session-id>.jsonl
```

After collecting the games you want, click **I'm Done — Train AI**. Training runs in a background
process against the cumulative confirmed-human dataset while the board stays responsive. The model
selected for that session is the immutable source. If training succeeds, the new `best.pt` is
selected in the current window for the next game; the next training cycle receives another fresh
set of paths. Closing the window is blocked while that process is active.

The equivalent manual command remains available for scripted experiments:

```powershell
python -m chess_ai train --config configs/human_gui.yaml `
  --init-checkpoint checkpoints/gpu_first/best.pt --device cuda
```

Manual fresh runs must also use unused output and metrics paths. See the [GUI guide](docs/gui_guide.md)
before treating a small set of personal moves as useful training evidence: the GUI dataset is
cumulative, but a larger count does not guarantee useful variety, and fine-tuning on tiny or narrow
data can make the model worse.

## Training workflow

1. Generate labels from reproducibly seeded minimax/random games.
2. Inspect the versioned dataset metadata and game mix.
3. Train with a fixed whole-game or whole-opening-pair validation split.
4. Inspect policy/value losses, top-k policy accuracy, and value error—not only total loss.
5. Compare the frozen candidate with baselines and the current champion across switched colors.
6. Promote or rename a checkpoint only after a human reviews enough evidence.

For exact commands, expected artifacts, resume behavior, and troubleshooting, follow the
[training guide](docs/training_guide.md). The tiny `configs/dev.yaml` proves plumbing quickly;
`configs/train.yaml` is only a reasonable experiment starter and does not promise a strong engine.
The reference-informed joint policy/value workflow is documented in
[model improvement comparison](docs/model_improvement_comparison.md).

## Manual external-opponent workflow

```powershell
python -m chess_ai external --checkpoint checkpoints/dev/best.pt --ai-color white
```

Open The Chess Lv.100 yourself. Enter each printed AI UCI move there, then type the opponent's move
back into the terminal. The workflow displays the board, validates input, supports `undo`, `show`,
`fen`, `help`, `resign`, and `quit`, and saves a PGN plus benchmark record. It never clicks, reads,
screenshots, modifies, or controls the external app.

External games remain evaluation-only. `import-external` is a separate explicit validation step;
nothing silently mixes benchmark games into training. See the
[external-opponent guide](docs/external_opponent_guide.md).

## Testing and quality checks

```powershell
python -m ruff format .
python -m ruff check .
python -m mypy
python -m pytest
```

Tests cover rules-wrapper behavior, exact encodings, promotions and special moves, agent legality,
search tactics, model/backprop/checkpoints, value perspective, finite tiny training, PGN round trips,
external-input rejection, arena accounting, and CPU/CUDA paths where available.

## Directory structure

```text
configs/                    tiny and starter YAML configurations
docs/                       architecture, concepts, workflows, experiment log, glossary
src/chess_ai/
  environment/              ChessGame and 18-plane / 4,208-action encoders
  agents/                   human, random, minimax, neural agents
  model/                    residual policy-value network and checkpoints
  data/                     examples, replay storage, supervised generation
  training/                 losses, split, trainer, metrics
  curriculum/               online correction cycles and generation-only teacher batches
  gui/                      neural-play workbench, human-game collection, session training launcher
  arena/                    matches, tournaments, external sessions, approximate ratings
  storage/                  PGN, JSONL metrics, benchmark records
  cli.py                    command-line composition
tests/                      deterministic unit and integration tests
data/{games,datasets,metrics,gui_sessions}/ generated local artifacts (ignored by Git)
checkpoints/                generated model/training state, including human sessions (ignored by Git)
```

## Current limitations

- Classical one-hot labels and D2 score-softmax labels both inherit the blind spots of shallow
  minimax. D2 root scores are richer than one move but are not MCTS visit distributions.
- The ordinary `neural` agent is still a policy-only greedy baseline. `neural-mcts` uses both heads,
  but currently evaluates leaves sequentially and does not yet create self-play visit targets.
- The compact network and tiny configurations prioritize learnability and runtime over strength.
- A small human-only dataset can overfit quickly and imitate one player's mistakes or narrow opening
  choices; keep the source checkpoint and compare the fine-tuned candidate before promoting it.
- Opening-game supervision teaches move patterns from strong games, not verbal principles or a
  proof that every line is objectively best. The optional compiled opening book guarantees only
  that selected book moves are legal and present in the reviewed source. Always retain broad replay
  and test the normal start in both colors.
- Approximate Elo is a descriptive transform of observed score, not an official rating or confidence
  interval.
- Bit-for-bit reproducibility across different CPU/GPU, CUDA, PyTorch, and driver versions is not
  guaranteed.
- A move limit is scored as an arena/data-generation draw safeguard, not an official chess outcome.

## Reproducibility

Record the YAML file, seed, dataset format/version, checkpoint metadata, Python/PyTorch versions,
hardware/device, Git commit, and exact opponents in [experiments](docs/experiments.md). Do not mix
positions from one game between training and validation. Keep checkpoints immutable during an arena
run. A deterministic seed reproduces pseudo-random choices within the same software environment; it
cannot remove all hardware-dependent floating-point variation.

## Roadmap

- Policy-guided Monte Carlo Tree Search and PUCT
- Dirichlet exploration noise
- Self-play replay buffer and full reinforcement-learning loop
- Candidate-versus-champion promotion criteria and parallel self-play workers
- Stockfish/UCI and tactical-puzzle benchmarks
- Optional approved online-bot integration, kept separate from manual external mode

These are documented future layers, not hidden or partial Phase 1 features.

## Dependencies and licenses

The direct runtime dependencies are PyTorch, NumPy, PyYAML, and python-chess; pytest, Ruff, mypy,
pytest-cov, and types-PyYAML are development tools. They are separate projects with their own
licenses. Before redistributing an application or bundled environment, review the exact installed
versions and their license files (for example with `python -m pip show PACKAGE`). No pretrained
weights, remote datasets, or hidden network downloads are included.

## Learning resources in this repository

- [Neural networks for beginners](docs/neural_networks_for_beginners.md)
- [Training guide](docs/training_guide.md)
- [GUI human-play and fine-tuning guide](docs/gui_guide.md)
- [External-opponent guide](docs/external_opponent_guide.md)
- [Architecture](docs/architecture.md)
- [Experiment template](docs/experiments.md)
- [Glossary](docs/glossary.md)
