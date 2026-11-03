# Stockfish training from the 1,500-game GM collection

The prepared run is `gm_endgame_training_v1_20261024_01`. It uses the frozen,
validated collection of 500 games each from Hikaru Nakamura, Magnus Carlsen and
Fabiano Caruana. All 1,500 mainlines were revalidated. No labeling or optimization
has run. The existing champion and all earlier experiments remain unchanged.

The preliminary plan contains 6,399 positions: 4,586 train, 917 validation and 896
confirmation. Eight connected game groups were excluded for historical overlap.
These counts are **before the historical evaluation-PGN exclusion scan**; that scan
can remove more complete groups. A missing split fails rather than changing the seed.

## Import and target boundary

`src/chess_ai/data/gm_endgames.py` validates coverage version 2, the exact snapshot
SHA-256, nonempty legal standard-chess mainlines, player ownership, unique game
digests and each player's quota. `scripts/gm_endgame_training.py` reserves configs
without importing training data. `prepare --confirm-training-import` explicitly
crosses that boundary and writes a checksum-bound `training_import.json` receipt.

Games sharing any endgame input, color mirror or horizontal reflection are joined
into connected components **before** deterministic 70/15/15 train/validation/
confirmation assignment. Splits are by component, not by move. At most 16 distinct
positions per game are chosen by deterministic family/color round-robin sampling.
This bounds correlation without assuming every online GM move is a good label.
Sampling does not guarantee equal counts for rare families or score bands.

The importer rejects a whole component if any of its endgame inputs or mirrors
overlap earlier Stockfish catalogs, protected datasets, the original endgame pilot
(including its untouched confirmation), historical evaluation PGNs or recorded
endgame rollout paths. Evaluation-PGN scanning includes **unselected** endgame
positions from each imported game. It never copies historical evaluation inputs
into training. The per-file scan journal resumes through the existing pilot code.

Raw games supply positions only. CPU Stockfish supplies MultiPV policy targets and
side-to-move `stockfish_cp_tanh_v1` values with scale 600, 200,000 nodes, MultiPV 5,
one thread and 128 MiB hash. The existing mate-distance preference is 2.0; Syzygy
remains disabled. Canonical labels reset counters consistently with the champion's
18-plane encoder, which lacks clocks/history. Human moves/results are never targets.

The compatible endgame-pilot position schema is reused with additive `gm_import`,
game digest, owner and related-position exclusion fields. The new importer/experiment/
authorization formats are version 1 and reject unknown versions. Tensor version 2
and checkpoint/value semantics are unchanged. Confirmation rows are labeled for
later evaluation but never exported to `fresh.pt` or `replay.pt`.

Finalization caps each split/family/color/approximate-score bucket at 256. Replay
draws one historical example per fresh example in each split, retaining historical
split protection. Training starts from the frozen epoch-60 champion with fresh
optimizer, LR `1e-5`, batch 128, at most eight epochs, patience three, and eligible
epoch-zero baseline. Fixed BatchNorm statistics preserve the normalization buffers;
affine weights still train. This setting follows the retention diagnostic and is
not a claim that the rejected retention candidate improved gameplay.

## Commands in execution order

The run/configs and preliminary plan already exist. Run these from PowerShell.
Preparation still needs its complete historical evaluation exclusion pass:

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$run = 'gm_endgame_training_v1_20261024_01'
$root = "data/experiments/$run"
$champion = 'checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt'

& $py scripts/gm_endgame_training.py --run-id $run --stage prepare --confirm-training-import
if ($LASTEXITCODE -ne 0) { throw 'GM import/exclusion scan failed; repeat to resume' }
```

Time a small first labeling batch. Its JSON report includes measured seconds per
position and an estimated remaining time; cached inputs can make estimates optimistic:

```powershell
& $py scripts/gm_endgame_training.py --run-id $run --stage labels --max-positions 64
if ($LASTEXITCODE -ne 0) { throw 'GM teacher pilot stopped; repeat to resume' }
```

Finish CPU labeling, assemble the fresh tensor dataset, and compose 50/50 replay:

```powershell
foreach ($stage in @('labels', 'finalize', 'compose')) {
  & $py scripts/gm_endgame_training.py --run-id $run --stage $stage
  if ($LASTEXITCODE -ne 0) { throw "GM stage stopped: $stage; repeat to resume" }
}
```

Start CUDA training once the stages succeed:

```powershell
& $py -m chess_ai train --config "$root/configs/train.yaml" --device cuda `
  --init-checkpoint $champion
if ($LASTEXITCODE -ne 0) { throw 'GM training stopped; use the resume command' }
```

For an interrupted training run with a committed `last.pt`, use this instead of
fresh initialization. Do not extend a completed early-stopped run:

```powershell
& $py -m chess_ai train --config "$root/configs/train.yaml" --device cuda `
  --resume "checkpoints/$run/last.pt"
if ($LASTEXITCODE -ne 0) { throw 'GM training resume stopped' }
```

Use the loss-selected `best.pt` as the single provisional candidate. Compare it to
the champion on the new GM validation roots under identical unassisted PUCT32:

```powershell
$candidate = "checkpoints/$run/best.pt"
foreach ($model in @('champion', 'candidate')) {
  $checkpoint = if ($model -eq 'champion') { $champion } else { $candidate }
  & $py scripts/endgame_pilot.py --config "$root/configs/pilot.yaml" evaluate `
    --checkpoint $checkpoint --output "$root/development_$model" --device cuda
  if ($LASTEXITCODE -ne 0) { throw "GM development stopped: $model; repeat to resume" }
}
```

Also retain the original synthetic stress test, including its scarce basic mates:

```powershell
& $py scripts/endgame_pilot.py `
  --config data/experiments/endgame_pilot_v1_20261023_01/configs/pilot.yaml evaluate `
  --checkpoint $candidate --output "$root/synthetic_development_candidate" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Synthetic development stopped; repeat to resume' }
```

Compare the synthetic result with the original pilot's `development_champion/report.json`.
If `best.pt` remains epoch zero, it is the unchanged champion and no improvement was
learned. If development regresses, stop; do not try confirmation or more epochs.
Whole-game D1/D2 non-regression must pass before nominating one candidate for fresh
confirmation. No promotion or confirmation command is part of this initial sequence.

## Resume, cost and additional runs

Repeat interrupted preparation/labeling/finalization/composition after the previous
process exits. SQLite commits every label. Replay composition verifies its input
binding and recovers an interrupted receipt write without replacing completed tensors.
Input/config/teacher/champion changes fail rather than invalidating progress silently.
Ctrl+C releases locks; after a hard kill verify the PID in the sibling lock file has
exited before manually removing that lock. Never delete the label journal to resume.

There are at most 6,399 planned teacher calls in this run, reduced by evaluation
exclusions. Use the 64-position timing instead of extrapolating older synthetic
timings. Stockfish uses CPU; CUDA helps training and neural evaluation. Allow roughly
1 GiB free for tensors, caches, journals and retained checkpoints; actual usage depends
on exclusions, bucket counts and stopping epoch. Evaluation can cost more than training
because PUCT is sequential and every played neural move is reviewed.

To reserve a separate experiment later, choose an unused ID:

```powershell
& $py scripts/gm_endgame_training.py --run-id YOUR_NEW_ID --stage reserve
```

The registered champion is verified during reservation. Existing runs are never
overwritten. Every new run still requires its explicit preparation/import stage.

Validation: Ruff formatting/checks and mypy passed. Full pytest with the pinned
local Stockfish smoke passed 347 tests with one expected CUDA-availability skip.
Coverage includes connected game/mirror splits, per-game caps, historical input
and unselected evaluation collisions, protected confirmation/rollout paths, label
resume, source edits, external-import acknowledgement and replay receipt recovery.
All eight documented PowerShell blocks parse successfully. The champion registry
and checkpoint SHA-256 remain unchanged. The real evaluation exclusion scan,
Stockfish labeling and optimizer updates are pending the commands above.
