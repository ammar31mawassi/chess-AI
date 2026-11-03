# Full-corpus rebuild with 3,000 GM games

The user authorized a second 500 games each from Hikaru Nakamura, Magnus Carlsen
and Fabiano Caruana, retaining the previous collection and training a new candidate
from random weights on all compatible available position sources.

The expanded collection is complete at `data/experiments/gm_3000_v1_20261031_01/`:
**3,000 unique games, 1,000 owned by each GM, all 1,500 original games retained with
unchanged ownership, and exactly 500 added per player**. The public profiles were
rechecked against the previously verified stable account IDs. Only serial public
archive API requests were used. September 2026 was the latest completed month at
collection setup; the bounded backward scan reached July 2025 for Caruana. Original
raw files, collection metadata and snapshot remain unchanged.

Coverage describes 997 games reaching ten pieces or fewer, 30,549 unique endgame
inputs and 5,608 inputs with at most five pieces. These are mostly online fast games,
and many end before mate. They supply encountered positions, never human move or
game-result targets. `validation.json` independently checks the exported snapshot's
legal nonempty mainlines, game digests, ownership, exact retention and checksum.
The snapshot contains 279,933 recorded pre-move positions before canonical
deduplication, claimable-terminal filtering and holdout exclusions.

## What the new experiment changes

The reserved run is **`full_gm_rebuild_v1_20261031_01`**. Its inventory includes:

- The original 480,953-row Stockfish evaluation dataset, including its fixed split.
- The 57,983-row personal-game Stockfish dataset.
- The compatible Stockfish refinement dataset.
- The original targeted endgame pilot's `fresh.pt`.
- The previous GM endgame pilot's `fresh.pt`.
- Every eligible nonterminal pre-move position from both players in the combined
  3,000-game snapshot, covering openings, middlegames and endgames without a
  per-game position cap. Claimable terminal positions are skipped by python-chess.

The original rebuild already relabeled the earlier approved legacy sources. We
reuse those compatible labels rather than concatenate legacy outcome datasets or
duplicate replay/composition artifacts. Exact network inputs have a single owner.
Existing tensor bytes, labels and teacher metadata are preserved and referenced in
the inventory. New unique GM inputs receive independent local Stockfish labels at
200,000 nodes, MultiPV 5, one CPU thread, 128 MiB hash, mate-distance preference 2.0.
Older labels may have different recorded teacher budgets/preferences; the corpus
does not claim all labels were regenerated at 200,000 nodes.

All values retain `stockfish_cp_tanh_v1`, side to move, scale 600. Canonical FEN
counters reset consistently with the existing encoder. No Syzygy probing, outcome
labels, new model heads, runtime assistance or promotion are introduced.

## Splits and protection

Existing source splits stay fixed. Related GM games are joined through endgame
transpositions/color mirrors/file reflections before splitting; inherited GM
train/validation/confirmation assignments take precedence, with the more protected
assignment winning if newly connected games bridge old groups. New GM components
use a deterministic 80/10/10 hash split. Early opening transpositions do not join
every normal-start game into one giant component; shared inputs are deduplicated
with validation ownership instead.

Prior validation inputs veto training. Previously trained inputs that collide with
new GM validation are dropped from both exported splits, not repackaged as new
validation evidence. Reserved original and GM endgame confirmation inputs/mirrors
remain protected. Entire GM confirmation games supply exclusions, not training
examples or labels. All archived evaluation PGNs and JSON endgame rollout paths
also veto exported inputs. Endgame mirror equivalents are screened as well.
Some historical rows are therefore removed for protection; "all positions" means
all compatible, deduplicated, permitted training positions, excluding holdouts.

`full_corpus.py` journals each source shard, GM game, rollout and evaluation PGN in
SQLite. It freezes exact input/config/champion/teacher-binary hashes, validates
source tensors against FENs and keys, checks fixed group ownership and signs the
position inventory. Each new label commits separately with a checksum and a bound
teacher provenance record. Finalization writes bounded atomic tensor shards,
verifies/reuses committed shards after interruption, and preserves source-level
label provenance. Dataset tensor version 2 and shard-manifest version 1 are reused;
the full-corpus plan/experiment/inventory use their own version-1 format. Expanded
GM coverage/collection use version 3; old version-2 collections remain readable.

The v2 offline refinement collector has a known lossy FEN serialization: its board
tensor and `position_key` can retain an en-passant target while default FEN omits it
because no legal en-passant capture exists. Import reconstructs that target only for
this producer/version, only when the tensor hash matches the saved key, the first
17 planes agree, python-chess validates the reconstructed board and both boards
have identical legal-move sets. Legal en-passant omissions, piece/turn/castling
mismatches, wrong hashes and unknown producer variants still fail with source path
and row index. No source tensors or policy/value targets are changed.

Repairs are journaled atomically with their source unit in the additive `fen_repairs`
table and counted by `inventory.json.source_fen_repairs`. Exported metadata records
the original `source_fen` and `source_fen_adjustment: inactive_en_passant_restored_v1`.
The current source audit found 612 such rows in the 16,313-row refinement dataset;
all remaining 22,644 source rows passed validation through the adapter. See the
run's `source_fen_diagnostic.json`. Existing committed source units remain valid:
after the old `Source tensor/FEN/key mismatch` error, rerun the inventory command
above without deleting the SQLite catalog, plan or completed units. Original source
artifacts, schemas and value-target semantics remain unchanged.
The compatibility fix passed Ruff, mypy and the full suite: 360 tests passed with
one expected CUDA-availability skip. Regression tests cover restored inactive
targets, rejection of lost legal captures and unrelated corruption, import resume,
unchanged source bytes and exact preservation of exported policy/value targets.

## Training and development protocol

The scratch arm uses the champion's architecture (32 channels, two residual blocks),
random parameters, fresh optimizer, LR `1e-3`, batch 128, at most 100 epochs, patience
15 and updating BatchNorm statistics. It receives **no initialization checkpoint**.
The second practical comparison starts from the unchanged champion with LR `1e-5`,
at most eight epochs, patience three, eligible baseline and fixed BatchNorm statistics.
These are appropriate separate schedules, not an initialization-only causal ablation.
Both use seed 31102039, identical data and identical exposure targets.

Expected training draws are 65% general positions, 30% other endgames (at most ten
pieces) and 5% basic KQK/KRK positions, including either side to move. Finalization
computes per-bucket inverse-count multipliers through the existing sampler. It
does not duplicate stored rows or change validation/loss weights. Sampling uses
replacement, so availability of all rows does not mean each row is drawn each epoch.
The explicit 5% mating allocation is a pilot choice motivated by the previous GM
sample's 23 mating training rows, not an established optimal mixture. Empty declared
buckets fail clearly. A later mixture change requires a separate experiment.

Only `best.pt` from each arm goes to development. Use the same original synthetic
and GM endgame suites as before, unassisted PUCT32, `c_puct=1.5`, local D2 opponent,
and matching seeds/limits. Compare with their saved champion reports. Whole-game
D1/D2 non-regression is a subsequent gate. Nominate at most one candidate before
fresh confirmation; do not search confirmation across epochs, arms or budgets.
The registered champion and its checkpoint are preserved throughout.

## PowerShell: run in order

The collection and experiment reservation are complete. Start with explicit import
and the complete source/holdout inventory. This may take time and resumes by unit:

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$run = 'full_gm_rebuild_v1_20261031_01'
$root = "data/experiments/$run"
$champion = 'checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt'

& $py scripts/full_gm_rebuild.py --run-id $run --stage inventory --confirm-training-import
if ($LASTEXITCODE -ne 0) { throw 'Full inventory stopped; repeat to resume' }
```

Measure 128 new labels before launching the remaining CPU work. The printed JSON
contains the actual pending count, time per position and estimated remaining hours:

```powershell
& $py scripts/full_gm_rebuild.py --run-id $run --stage labels --max-positions 128
if ($LASTEXITCODE -ne 0) { throw 'Teacher timing batch stopped' }
```

Complete labeling and finalization. The latter produces signed ready-config hashes
and verifies the actual sampling bucket counts:

```powershell
foreach ($stage in @('labels', 'finalize')) {
  & $py scripts/full_gm_rebuild.py --run-id $run --stage $stage
  if ($LASTEXITCODE -ne 0) { throw "Full corpus stage stopped: $stage" }
}
```

Train from scratch. The wrapper verifies the frozen inputs, finalized manifest and
ready config; it invokes the ordinary trainer without `--init-checkpoint`:

```powershell
& $py scripts/full_gm_rebuild.py --run-id $run --stage train --arm scratch --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Scratch training stopped; use resume if interrupted' }
```

Then train the champion-initialized comparison, if running both planned arms:

```powershell
& $py scripts/full_gm_rebuild.py --run-id $run --stage train --arm finetune --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Fine-tuning comparison stopped' }
```

For an interrupted run with a committed `last.pt`, use this instead of another fresh
start. Set the arm to the interrupted run. Do not extend completed early-stopped runs:

```powershell
$arm = 'scratch'
& $py scripts/full_gm_rebuild.py --run-id $run --stage train --arm $arm --resume --device cuda
if ($LASTEXITCODE -ne 0) { throw "Training resume stopped: $arm" }
```

Evaluate the two candidates on both existing endgame development suites:

```powershell
foreach ($arm in @('scratch', 'finetune')) {
  $candidate = "checkpoints/${run}_$arm/best.pt"
  foreach ($suite in @('synthetic', 'gm')) {
    $cfg = if ($suite -eq 'synthetic') {
      'data/experiments/endgame_pilot_v1_20261023_01/configs/pilot.yaml'
    } else {
      'data/experiments/gm_endgame_training_v1_20261024_01/configs/pilot.yaml'
    }
    & $py scripts/endgame_pilot.py --config $cfg evaluate --checkpoint $candidate `
      --output "$root/development_${arm}_$suite" --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Endgame development stopped: $arm / $suite" }
  }
}
```

If only the scratch arm was trained, use `@('scratch')` in that loop. Both champion
baselines already exist: original endgame pilot `development_champion/report.json`
and previous GM training run `development_champion/report.json`. Repeating endgame
evaluation resumes per ply only when the checkpoint, case suite and budget match.

Only after development supports one candidate, choose its arm and run whole-game
development checks. These are local minimax depths, not Stockfish depths:

```powershell
$arm = 'scratch'
$candidate = "checkpoints/${run}_$arm/best.pt"
foreach ($depth in @(1, 2)) {
  & $py -m chess_ai paired-audit --candidate $candidate --champion $champion `
    --openings 16 --audit-seed 31102040 --opponent-depth $depth `
    --search-simulations 32 --c-puct 1.5 --device cuda `
    --exclude-manifest "$root/training_manifest.json" `
    --exclude-additional-manifest 'data/experiments/stockfish_rebuild_eval_v2_20261011_01/relabel/training_manifest.json' `
    --pgn-dir "data/games/evaluation/${run}_${arm}_development_d$depth"
  if ($LASTEXITCODE -ne 0) { throw "Whole-game development stopped: D$depth" }
}
```

Whole-game audits cannot resume partial PGNs; keep the checkpoint/seed fixed and
use a new output suffix for an interrupted attempt. No confirmation or promotion
command is part of this setup.

## Collection verification, costs and recovery

The collection command is already completed. To verify/retry its exact frozen run:

```powershell
& $py scripts/collect_gm_pilot.py `
  --raw-dir downloaded_datasets/chesscom_gm_pilot_20261024 `
  --report-dir data/experiments/gm_3000_v1_20261031_01 `
  --games-per-player 1000 --latest-month 2026-09 --max-months 24 `
  --minimum-base-seconds 180 --seed 31102038 `
  --base-coverage data/experiments/gm_endgame_1500_v2_20261024_01/coverage.json
if ($LASTEXITCODE -ne 0) { throw 'Expanded GM collection verification failed' }
```

The expanded corpus is much larger than the earlier 9,676-row replay. New labeling
can take hours; the 128-position timing uses the actual retained full-game pool.
Existing labels are reused, not regenerated. Allow roughly 25 GiB of additional
disk space initially; actual tensor size is approximately 21,440 bytes per unique
row before metadata, plus label journals/caches/checkpoints. Inventory reports the
actual retained count. Training uses memory-mapped shards, but example metadata and
legal masks still consume RAM. CUDA accelerates the neural model; Stockfish stays
on CPU. The setup does not launch exhaustive labeling or optimization.

Repeat an interrupted inventory/label/finalize stage after the prior process exits.
Input hashes and source checks repeat before work resumes. Ctrl+C releases locks;
after a hard kill verify the recorded PID has exited before removing only the stale
sibling lock. Keep the SQLite catalog and completed shards. For another experiment,
use `scripts/full_gm_rebuild.py --run-id YOUR_NEW_ID --stage reserve`; output IDs and
training paths are never reused automatically.

## Setup verification

Ruff formatting/checks and mypy passed, including the new/extended scripts. Full
pytest with the pinned local Stockfish smoke passed **354 tests with one expected
CUDA-availability skip**. Tests cover pinned expansion ownership/quotas, compatible
label reuse, interrupted game inventory, per-position label resume, confirmation
game protection, group separation, source/label edits, fixed sampling shares,
scratch-versus-initialized command construction, orphaned catalogs and preservation
of protected rollout bytes. All nine documented PowerShell blocks parse. The five
real source formats/metadata and reserved config bindings were checked. Champion
registry/checkpoint hashes remain unchanged. Exhaustive inventory, new-label
analysis and optimizer updates are intentionally left for the commands above.
