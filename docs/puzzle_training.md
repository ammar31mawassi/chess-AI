# Two isolated Lichess puzzle training experiments

The user authorized import of 250,000 puzzles and two separate experiments on 2026-11-05.
The frozen CC0 snapshot is `downloaded_datasets/lichess_puzzles/hf_461725966ea4`, revision
`461725966ea46898ec6481372a6326de3281bfbd` (6,157,341 puzzles). This is supervised training,
not reinforcement learning. The registered scratch epoch-99 champion remains unchanged.

## Selection and boundaries

Run ID: `puzzles_250k_v1_20261105_01`; seed: `5112041`. We select **250,000 puzzles total**:
225,000 training, 12,500 development/validation, and 12,500 untouched confirmation puzzles.
These are **one solver-root position per puzzle**, not every move in every solution.
Confirmation positions never enter training/validation tensor shards or Stockfish labeling.

The deterministic candidate pool samples across every Parquet shard. Quality gates require
at least 100 plays, popularity at least 80, and rating deviation at most 100. There is no rating
ceiling or deliberate theme oversampling. Selection records its actual rating range/themes.
Entire lines are validated with python-chess. The first UCI move is the opponent's setup move;
the second is the solver's first move. Null/illegal moves, terminal roots, duplicate inputs,
and repeated source games are rejected. All legal mates are accepted for mate-in-one roots.

The existing protection helper reads the finalized full-GM corpus, validation/confirmation bans,
search-development suite, historical evaluation PGNs, endgame rollouts and search-ablation games.
New puzzles exclude both historical training inputs and protected inputs, including supported
endgame mirrors/reflections. Only one puzzle per source game is selected, and root orbits are
disjoint across all new splits. Old corpus tensors, labels and train/validation splits are reused.
Evaluation provenance and exclusion-key digests are snapshotted at reservation. Live online
evaluation journals may subsequently grow; those mutable files are not frozen training inputs.
New evaluations must still check for overlap with this experiment's training positions.

Snapshots, approved old shards, champion/registry, selection, teacher binary, ready configs and
new shards have hash guards. An edited config/input fails clearly; reserve a new run to change it.
No evaluation PGN is imported as training. No checkpoint is automatically promoted.

## Labels and model schedules

Puzzle solutions supply the policy targets. Puzzle ratings are never used as value targets.
The existing local Stockfish teacher supplies **root position values**, from the side to move,
using the existing `tanh(cp/600)`/mate semantics. Teacher settings inherit the approved corpus's
pinned executable/commit and fixed 200,000-node budget; MultiPV is reduced to 1 because we need
only its scalar evaluation. Puzzle solutions remain the policy authority. This bounded value
estimate may disagree with a deeper puzzle analysis; it is not exact tablebase WDL.
Analysis is CPU-only through python-chess UCI, one position at a time. A SQLite journal commits
each completed label and checks teacher provenance on resume. The 237,500 train/validation
values are computed once and reused by both arms. No per-position JSON cache is created.

| Arm | Initialization | Training positions | Epochs | Learning rate |
|---|---|---|---|---|
| `scratch` | Random weights, fresh optimizer | All 680,747 approved old training rows + 225,000 puzzle roots = **905,747** | 100 | 0.001, cosine scheduler |
| `finetune` | Frozen registered scratch epoch-99 champion, fresh optimizer | **225,000 puzzle roots only** | Up to 40, patience 10 | 0.00001, cosine scheduler |

Scratch uses uniform shuffled exposure to the full combined dataset, with no source sampler
multipliers or game balancing. Thus puzzles are about 24.8% of its training rows. The combined
manifest has 1,007,834 examples: 905,747 train and 102,087 validation (89,587 old + 12,500 puzzle).
The puzzle-only manifest has 237,500 examples (225,000 train + 12,500 validation). Fine-tuning
fixes BatchNorm running statistics, allows affine parameters and both heads to learn, and keeps
epoch zero eligible as a baseline. Both retain best/last checkpoints; periodic retention is every
5 scratch epochs and every 2 fine-tune epochs. Validation loss selects `best.pt`; gameplay is
still required before any promotion. Puzzle-only training may forget general chess skills.

Old shards are referenced from a combined manifest at
`data/experiments/puzzles_250k_v1_20261105_01_combined.json`, whose parent contains both sources.
No existing tensor file is copied or edited. New puzzle tensors need roughly 5 GB plus metadata;
labels/checkpoints need additional space. Existing version-2 tensors, version-1 shard manifests,
18 planes, 4,208 actions and checkpoint formats are unchanged. New signed plan, selection,
readiness, value-label payloads and puzzle-audit journals are version 1.

## Exact PowerShell commands

Preparation/selection is already performed by the agent for the run above. If reproducing it
in a new checkout/run, install the optional Parquet reader and explicitly acknowledge import:

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$puzzleRun = 'puzzles_250k_v1_20261105_01'
& $py -m pip install -e '.[puzzles]'
if ($LASTEXITCODE -ne 0) { throw 'Puzzle dependency installation failed' }
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage prepare --confirm-training-import
if ($LASTEXITCODE -ne 0) { throw 'Puzzle preparation failed' }
```

The user runs all labeling, training and evaluation jobs. Stop any live bot service normally
before CUDA training/evaluation so it does not compete for GPU resources. Set variables in each
fresh terminal; do not reuse the old `$run` variable from earlier experiments.

First label 1,000 positions to measure actual throughput and ETA:

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$puzzleRun = 'puzzles_250k_v1_20261105_01'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage labels --max-positions 1000
if ($LASTEXITCODE -ne 0) { throw 'Puzzle value-label pilot stopped' }
```

Finish values, then materialize compatible shards and the two training configs:

```powershell
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage labels
if ($LASTEXITCODE -ne 0) { throw 'Puzzle labeling stopped; rerun to resume' }
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage finalize
if ($LASTEXITCODE -ne 0) { throw 'Puzzle finalization stopped; rerun after resolving the error' }
```

Run the arms sequentially. This intentionally starts new candidates and never edits the champion:

```powershell
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage train --arm scratch --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Scratch training stopped; use its resume command' }
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage train --arm finetune --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Puzzle-only fine-tuning stopped; use its resume command' }
```

For a stopped arm, use its own `last.pt` via the wrapper. Completed epochs resume with optimizer,
scheduler and RNG state; a partly completed epoch is replayed from the last saved checkpoint.
Labeling resumes per position. Finalization can be rerun after interruption. Normal Ctrl+C
cleans up locks; after a hard kill, verify the recorded PID exited before removing a stale lock.

```powershell
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage train --arm scratch --device cuda --resume
if ($LASTEXITCODE -ne 0) { throw 'Scratch resume stopped' }
```

```powershell
& $py scripts/puzzle_training.py --run-id $puzzleRun --stage train --arm finetune --device cuda --resume
if ($LASTEXITCODE -ne 0) { throw 'Fine-tune resume stopped' }
```

When both finish, compare the unchanged champion and both candidates on the same first 1,000
development puzzles with production PUCT32, `c_puct=1.5`, no book. This measures **first solver
move accuracy**, not full-line solve rate or Elo. The JSON progress journal resumes every 25
positions. Ordinary D1/D2 full-game paired tests and the old endgame suites are still necessary
to assess forgetting/general strength; do those before nominating one candidate for confirmation.

```powershell
$champion = (Get-Content "data/experiments/$puzzleRun/plan.json" -Raw | ConvertFrom-Json).payload.champion
if (-not $champion) { throw 'Champion registry has no checkpoint path' }
$candidates = @(
    $champion,
    "checkpoints/${puzzleRun}_scratch/best.pt",
    "checkpoints/${puzzleRun}_finetune/best.pt"
)
foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate)) { throw "Missing checkpoint: $candidate" }
    & $py scripts/puzzle_training.py --run-id $puzzleRun --stage evaluate `
        --checkpoint $candidate --limit 1000 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Puzzle development evaluation stopped: $candidate" }
}
```

Keep the 12,500 confirmation puzzles untouched during development. The explicit
`--split confirmation` option exists for a later, single nominated candidate after reviewing
development evidence. It must not become another epoch-selection set.

## Implementation verification

Ruff formatting/checks and mypy passed. The full suite passed 458 tests with 6 expected
environment-dependent skips (local UCI smoke checks and the CUDA-unavailable path on a CUDA
machine). Focused offline tests cover setup/solution semantics, legal mate alternatives, seeded
deduplication and protected splits, label resume/provenance, old-corpus preservation, confirmation
exclusion, correct scratch/fine-tune initialization, safe resume, tamper detection and puzzle-audit
resume. All seven PowerShell blocks were parsed without executing their jobs. No Stockfish value
labeling, live gameplay, optimizer update or promotion was performed during setup.
