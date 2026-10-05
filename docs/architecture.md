# Architecture

## Phase 1 boundary

This repository is a supervised-bootstrap vertical slice. It has a reliable chess environment,
classical agents, a compact policy-value network, local data/training tools, a focused neural-play
GUI, controlled evaluation, and a small deterministic PUCT inference agent. It deliberately does
**not** claim to be AlphaZero: search does not yet generate self-play training targets, inference is
not batched, and there is no reinforcement-learning loop or automated interaction with external
chess software.

## Dependency flow

```text
python-chess
    -> environment (rules, board and move encoding)
    -> agents (human, random, minimax, greedy neural, policy/value PUCT)
    -> arena / external sessions / local GUI

environment + agents
    -> supervised dataset generation
    -> PyTorch trainer -> versioned checkpoints
    -> NeuralAgent or NeuralMCTSAgent -> arena / controlled evaluation

local GUI + collection checkbox + post-game confirmation
    -> cumulative human-move dataset
    -> independently saved PGNs

GUI session coordinator
    -> unique generated training config
    -> background trainer -> session checkpoints + metrics
    -> successful best checkpoint selected for later games

arena and training
    -> storage (PGN, JSONL metrics and benchmark records)

Neural student vs shallow minimax opponent
    -> stronger minimax correction on every student turn
    -> replay-protected policy update -> next seeded two-color pair

frozen gameplay champion
    -> block of fresh D1/D2 corrections -> conservative candidate
    -> identical D1 opening suite for champion and candidate
    -> promote only on score improvement with no standard-start regression

frozen gameplay champion
    -> 250 short seeded opening pairs vs D1 (500 games, no training)
    -> exact D2 root scores on every played ply -> soft policy trace dataset
    -> game-balanced, outcome-weighted offline candidate training
    -> collision-checked candidate/champion D1 holdout audit

frozen gameplay champion
    -> 50 shared short openings x 4 matchups x 2 colors (400 games, no training)
    -> all-ply D2 soft targets + winner/draw/loser actor weights
    -> opening-grouped offline training with every epoch retained
    -> gameplay-first epoch selection against both D1 and D2
    -> fresh paired audits before any manual promotion

seeded minimax league
    -> 500 shared openings x 3 depth pairings x 2 colors (3,000 games)
    -> independent D3 soft targets on every continuation position
    -> six-game atomic opening shards + complete PGNs + resumable manifest

reviewed local Kaggle CSVs
    -> explicit confirmation + python-chess validation + bounded deterministic sampling
    -> separate value, human-game, and tactical policy datasets
    -> staged low-learning-rate candidates -> focused D3-teacher fine-tuning
```

Keeping these arrows mostly one-way makes it possible to learn and test one layer at a time.

## Main packages

- `environment/game.py` is a small stateful wrapper around `chess.Board`. `python-chess` remains
  the only rules authority.
- `environment/board_encoder.py` converts a position to 18 binary planes. Array coordinates are
  `[plane, rank, file]`: rank index `0` is chess rank 8 and file index `0` is file `a`.
- `environment/move_encoder.py` provides the stable 4,208-action policy vocabulary. The first
  4,032 entries represent every distinct source/destination pair; 176 additional entries preserve
  queen, rook, bishop, and knight promotion identity for every valid promotion geometry.
- `agents/` provides a shared protocol. Minimax is intentionally readable alpha-beta search.
  `NeuralAgent` is the fast greedy policy baseline. `NeuralMCTSAgent` masks illegal actions, uses
  legal policy probabilities as PUCT priors, evaluates leaves with the value head, negates values
  at every ply, and selects the most-visited root action. Its tree is rebuilt for each move and its
  leaf inference is sequential, which keeps the implementation auditable but limits throughput.
  `OpeningBookAgent` may wrap any agent. It samples only legal continuations for the exact current
  position using a private seeded RNG. The first unmatched position permanently disables book mode
  for that game and delegates all remaining moves to the wrapped agent.
- `model/` contains the residual policy-value network and versioned, atomic checkpoints.
- `data/` contains versioned examples and supervised generation. Generation may apply a seeded,
  unrecorded random opening before classical teachers take control, and may deduplicate the exact
  18-plane inputs visible to the network across games. An optional PGN archive atomically preserves
  every complete source game and is verified when dataset generation resumes. Agent colors may be
  alternated while policy examples are filtered to a named teacher; optional normal-start minimax
  anchor games protect a known opening line. A stored value is always from the perspective of the
  player to move in that stored position.
- `training/` splits by complete game ID or another configured metadata group, optimizes policy and
  value losses, records JSONL metrics, and writes epoch/best checkpoints. Optional per-example loss
  weights and game-balanced sampling support outcome-aware offline batches without allowing long
  games to dominate merely because they contain more positions. Weighted and ordinary unweighted
  metrics are both recorded. Optional legal-policy masking uses FEN plus `python-chess` so training
  matches NeuralAgent's inference-time legality mask.
  Checkpoint selection can target total, policy, value, or policy-top-1 validation performance.
  Fresh initialization may keep epoch zero eligible, so an update that never improves the selected
  metric cannot silently replace its source. Optional patience/minimum-delta stopping bounds long
  runs, while sparse periodic epoch retention avoids writing hundreds of redundant checkpoints.
- `curriculum/` runs the controlled student/opponent/teacher loop. Each cycle gives one seeded
  post-opening position to the student as White and Black against a fixed shallow minimax opponent.
  A stronger minimax teacher labels only positions where the neural student moved. Corrections are
  mixed with a reproducibly sampled historical dataset, while PGNs, per-cycle correction datasets,
  metrics, checkpoints, and atomic resume state remain separate artifacts. The final pair is held
  out, and a mastery gate is reported without automatic promotion or difficulty changes.
  A configured periodic normal-start interval can anchor a known deterministic weakness while the
  remaining cycles keep random-opening diversity. Adding the first interval to a completed legacy
  run is an explicit, one-time stage migration; later resumes bind it into the state signature.
  Normal-start corrections are deduplicated within a pair but deliberately retained across anchor
  cycles, and a configured share of each correction batch is reserved for their rehearsal. After a
  trained anchor, a separate two-color standard-start regression pair is saved and attached to the
  cycle history; it is evaluation-only and does not affect training or mastery accounting. A
  completed periodic-anchor run may adopt this replay refinement once during an upward extension.
  The separate gated refinement starts each block from an immutable gameplay champion, uses a fresh
  optimizer, and evaluates candidate and champion independently against D1 from identical fixed
  positions in both colors. A candidate must improve the fixed-suite score without regressing on
  the standard pair. Accepted candidates receive immutable generation snapshots; rejected
  candidates remain archived and cannot become a later training source. Gameplay champion state is
  deliberately separate from the trainer's loss-selected `best.pt`.
  The offline teacher batch is a separate generation-only path. It freezes one champion, creates
  250 seeded 2-3-full-move openings, plays each as both colors against D1, and asks D2 for exact
  root scores on every subsequent ply. The strongest five moves become a soft policy target. Every
  pair receives an atomic trace shard and two complete PGNs before resume state advances; final
  assembly produces a frozen dataset and manifest. No optimizer or checkpoint output is reachable
  from this command. Champion-win examples carry a configured 2x training weight while draw/loss
  examples retain weight 1.
  The mixed offline batch reuses 50 seeded positions across D1-vs-D2, D2-vs-D2, neural-vs-D1, and
  neural-vs-D2. Each matchup uses both Agent-A colors for 400 games total. D2 annotates every
  continuation position; examples are weighted according to the outcome of the actor currently to
  move (win 2, draw 1.5, loss 1). Seeded minimax variation is restricted to equally scored best
  moves. All eight games from one opening share an `opening_set_id` so they cannot straddle the
  training/validation boundary.
  The minimax-league mode uses the same collector without loading a neural checkpoint. It plays
  D2-vs-D1, D2-vs-D3, and D1-vs-D3 in both color assignments from each shared position. D3 teacher
  analysis is cached within each six-game opening set. Every tenth position can be a normal-start
  anchor while the remaining positions use a configured random full-move range.
  A separate explicit finalization command may permanently close a stopped collection after at
  least one full opening set. It assembles only committed shards, records both requested and
  finalized opening counts, marks the state complete, and prevents a later resume from silently
  adding more games.
- `data/kaggle_import.py` contains explicit adapters for the reviewed local Kaggle chess CSVs.
  Evaluation and tactic FENs, tactical moves, SAN games, ratings, time controls, and results are
  validated before conversion. Imports are deterministically bounded, retain source provenance,
  refuse overwrite, and require a CLI acknowledgement that external data is intentionally entering
  training. Evaluation-only rows receive a neutral legal policy solely to satisfy dataset shape;
  their training configuration sets policy loss to zero. External sources remain separate so their
  quality and effect can be measured before focused minimax-teacher fine-tuning. Game imports may
  set `maximum_ply` to construct an opening-only sample while still parsing and validating the
  complete source games.
  `data/compose.py` explicitly copies reviewed datasets into a source-labelled artifact and assigns
  one normalized training-weight field. Related color-swapped teacher games retain a shared
  composition group for leak-free validation; a source may name its grouping metadata through
  `group_key`. Source files are never edited.
  `data/opening_book.py` explicitly validates named external opening lines through `python-chess`,
  aggregates transpositions into soft policy/value examples, and writes a versioned JSON runtime
  book. Training import and runtime use are both opt-in; neither artifact is created implicitly.
- `gui/` is a local Tkinter human-versus-neural workbench. Its controller delegates legality and
  outcomes to `python-chess`, runs neural inference away from the Tk event loop, saves PGNs
  independently, and can collect only explicitly confirmed human moves in a separate cumulative
  dataset. Its session coordinator reserves collision-free training artifacts and starts the
  ordinary trainer in a background process.
- `arena/` plays agents through ordinary `chess.Board` copies, detects illegal agent output, switches
  colors, records move time, and reports an explicitly approximate Elo difference. Its paired audit
  compares candidate and champion against D1 on identical starts in both colors, can exclude every
  opening FEN recorded by a training manifest, and never promotes or trains a checkpoint. Gameplay
  selection evaluates every retained epoch and the champion on one collision-checked opening suite
  against both D1 and D2. It rejects per-depth or standard-start regressions, uses validation loss
  only as a final gameplay tie-breaker, and remains provisional until fresh paired audits pass.
- `storage/` writes PGN and append-only benchmark/metric records locally.
- `cli.py` is the composition root; modules do not parse command-line arguments themselves.

## Human GUI data boundary

The workbench is an input surface, not a second chess engine. `python-chess` supplies legal moves,
check, mate, draws, promotion rules, and final outcomes. The GUI may highlight or disable actions,
but those are presentations of the board's legal-move set rather than independently implemented
rules.

PGN saving and training-data collection are separate actions. A finished game can be preserved as a
PGN even when collection is disabled or declined. When collection is visibly enabled, the user is
asked again after a completed game before it is appended to `data/datasets/human_gui.pt`. Only
positions where the human chose the move receive policy labels. A confirmed completed result
supplies value labels from the player-to-move perspective. Abandoned, reset, or otherwise unfinished
games never supply examples. The human dataset is cumulative across GUI launches and training
cycles; the trainer sees all accepted examples at the selected dataset path, not only games played
since the last run. Alternating the human's color across games reduces, but does not eliminate,
small-dataset bias.

Every workbench launch reserves a unique training session ID. The reservation binds three artifact
locations before a GUI-initiated run begins:

```text
data/gui_sessions/<session-id>/training.yaml
checkpoints/human_sessions/<session-id>/
data/metrics/human_sessions/<session-id>.jsonl
```

The generated YAML carries a session format name, integer version, ID, and UTC creation time. The
coordinator rejects edited path metadata, unsafe IDs, incompatible versions, stale resume targets,
and any collision with prior training artifacts.

**I'm Done — Train AI** invokes the ordinary training entry point in a background process with the
checkpoint selected for that cycle as its initialization source. This keeps optimization outside
the Tk event loop. The workbench blocks window closing for the lifetime of the child process. A
successful `best.pt` becomes the selected checkpoint in the current window, then the coordinator
reserves a fresh ID and paths for a possible next cycle. A failed run leaves the source selected.
Nothing automatically overwrites or promotes the source checkpoint.

Both GUI and manual fine-tuning use `train --init-checkpoint`: checkpoint architecture and weights
are loaded, then a new optimizer, scheduler, metrics file, and epoch count begin at epoch 1. This
differs from `--resume`, which restores optimizer/scheduler and continues the saved epoch. The two
modes are mutually exclusive. An initialization run must use a checkpoint output directory other
than the source directory so the starting model remains immutable for comparison, and it refuses to
overwrite an existing candidate or metrics file. The GUI satisfies those constraints with its
session reservation; the manual CLI remains available when the caller supplies unused paths.

## Artifact compatibility

Datasets and checkpoints carry a format name, integer version, and shape-defining metadata. Loaders
reject unknown formats and incompatible shapes rather than guessing. Adding fields compatibly is
fine; changing their meaning requires a new version. Writes use a temporary file and `os.replace`
where practical so interruption does not leave a half-written checkpoint or dataset.

Generated artifacts live under `data/` and `checkpoints/` and are ignored by Git. Empty `.gitkeep`
files retain the expected directory layout.

Teacher-cycle state has its own format name/version and a signature over all reproducibility
settings. Resuming rejects edited settings, missing PGNs/corrections, mismatched checkpoint epochs,
and output files that exist without their matching state. Correction datasets retain the ordinary
dataset format: policy targets are teacher moves, while values remain completed-game outcomes from
the stored player-to-move perspective.

Gated-refinement state independently binds the initial champion, round/block sizes, fixed gate,
training settings, and artifact paths. Its `champion.json` records the currently accepted immutable
generation plus a SHA-256 digest. Gate PGNs and scores are evaluation artifacts and never enter the
correction or replay datasets. A separate collision-checked holdout suite compares generation zero
and the current champion on unseen seeded starts without participating in promotion; its report and
PGNs are likewise evaluation-only.

Offline teacher-batch state binds every generation setting plus the champion SHA-256. A committed
pair is accepted on resume only when both PGNs and its trace shard still match their recorded
digests. The final manifest lists every opening and game, and the final dense dataset is assembled
only after all configured pairs are committed. Paired-audit output is deliberately separate and is
marked `TrainingData=false`.

Mixed-batch state uses an immutable boundary at one complete opening set: eight games for the neural
mix or six for the minimax league. Resume validates any configured neural checkpoint hash, every
committed PGN, and every opening shard. Its manifest is compatible with the paired-audit
opening-exclusion check. Gameplay-selection reports and PGNs are evaluation artifacts with
`promotion_performed: false`; they never enter the training dataset.

An early-finalized mixed batch keeps its original signed collection configuration but records
`finalized_early`, `requested_opening_positions`, and `finalized_opening_positions` in its state and
manifest. The assembled dataset repeats those provenance fields. External Kaggle imports use their
own format/version marker inside ordinary supervised dataset metadata; changing interpretation of
evaluation perspective or source fields requires a new importer version.

## Reproducibility

Commands accept or read a seed. Random agents own their own RNG, PyTorch/NumPy are seeded by the
training entry point, and train/validation assignment is deterministic by game ID or configured
opening-pair ID. Determinism is best effort across hardware: different PyTorch/CUDA versions can
still introduce small numeric differences, so record versions alongside experiments.

## External opponent safety boundary

External mode prints the AI's UCI move and waits for a human to enter the opponent's move. It has no
browser, screenshot, accessibility, process-control, or GUI-automation dependency. Benchmark PGNs
are evaluation artifacts. A separate explicit import command validates them before producing any
training-compatible output.

## Future layers (not implemented)

The intended next stages are batched PUCT leaf evaluation, root Dirichlet exploration, search-visit
policy targets, a bounded self-play replay window, confidence-aware candidate promotion, parallel
self-play, UCI/Stockfish and tactical-puzzle benchmarks, then a full reinforcement-learning loop.
Optional online play would require an approved bot API and a separate safety/design review.
