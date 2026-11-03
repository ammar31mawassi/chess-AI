# Independent search and failure-training pilots

Implemented 2026-11-03 after reviewing [Syed Taha's engine experiments](https://syedtaha22.github.io/blog/chess-bitboards-and-benchmarks/).
Items 1–5 are opt-in experiments; item 6 (specialized endgame value heads) is excluded.
His C++/NNUE alpha-beta speedups are not predictions for our Python/CNN/PUCT implementation.
The checkpoint, production PUCT, champion registration, historical holdouts, and running full-GM
corpus job remain unchanged. Nothing automatically promotes a checkpoint.

## Independent interventions

| Arm/workflow | Change | Experimental defaults |
|---|---|---|
| `baseline` | Instrumented equivalent of production sequential PUCT | 32 traversals; c_puct 1.5 |
| `cache` | Checkpoint-local LRU of exact 18-plane network predictions | 4,096 entries; about 140 MiB logits plus input keys/overhead |
| `batch` | Pending PUCT leaves evaluated together | Up to 8 leaves; temporary virtual loss |
| `tactical` | Bounded tactical negamax at evaluated leaves | Depth 2; at most 32 extra nodes per leaf |
| `reuse` | Retain the subtree reached through actually played moves | Matching root FEN and complete move-history prefix |
| failure training | Fresh bot games, independent teacher labels, replay, new candidate | 50 paired openings; at most 24 sampled bot turns/game |

Arms share the same model architecture and checkpoint, but never stack interventions. Each match
run selects one arm. Production `NeuralMCTSAgent` is unchanged. The new `ExperimentalPUCT` agent
is exposed through `scripts/search_experiments.py` and its Python API, not the GUI/default agent.

Cache entries contain network outputs only, copied defensively. Terminal adjudication always uses
python-chess with clocks/history; neither backed-up values nor terminal outcomes are cached by
placement. Changing weights requires a new agent. Batched search reserves visits using virtual
loss, flushes on duplicate pending leaves, and removes reservations even on inference failure.
Except for an exact immediate-mate shortcut, fixed-work runs complete the requested traversals.
Scheduling and numerical batching may
change moves; equal traversals do not imply equal trees or neural work.

The tactical arm first checks all legal root moves for an immediate, python-chess-proven mate.
If found it returns that move without neural work or simulations, recording `proven_mate_in_one`.
Otherwise leaf verification considers captures, promotions, and **all legal evasions while in check**.
Stand pat is forbidden in check. At depth/work limits the network is an approximate fallback.
It does not search every quiet check or assert a tactical proof. Root values are unused, so the
root gets no wasted tactical extension. Reuse follows played edges only, without a transposition
table. Identical FENs with different starting history do not inherit a tree. Reused visits are
reported separately; every move also receives its fresh traversal budget.

## Development benchmark

`configs/search_development_v1.json` is schema version 1: format, version, unique `cases` with
`id`, `category`, `fen`; optional `setup_moves`, `expected_moves`, `criterion: mate_in_one`, or
`solution` (alternating solver/opponent UCI moves). Python-chess validates prescribed lines.
Different immediate mating moves are accepted. Only explicitly scored cases measure correctness;
other positions measure speed. Conversion needs full games, not a single unscored move.
The 14 synthetic cases cover openings, mates, conversion, material, promotions, check evasions,
en passant, castling, near-fifty-move states, and repetition history. They are development only.

The runner warms each model, then clears cache/tree before every case. This cold-root benchmark
cannot demonstrate cross-move cache/reuse benefits; games do. Reports include median/p95 time,
network evaluations, batches, hits, and encoding/network/terminal/legal/selection times. Total
time also includes push/pop, copies, normalization, bookkeeping, and instrumentation. CPU output
transfers synchronize CUDA before the inference timer stops. Network positions/second uses total
search time, not raw GPU throughput. Repeated timings are not independent chess evidence.

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$champion = 'checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt'
$run = 'data/search_experiments/pilot_20261103_01'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
foreach ($device in @('cpu', 'cuda')) {
    & $py scripts/search_experiments.py benchmark --checkpoint $champion `
        --device $device --simulations 32 --repeats 3 --seed 31102041 `
        --output "${run}/benchmark_${device}_s32"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark stopped: $device" }
}
```

Read `report.json` → `summary_by_arm`. Compare hardware under stable load. Fixed-work choices
are deterministic on the same numerical setup; wall-clock runs are not. The optional equal-time
benchmark uses a **soft** deadline, checked between traversals/batches. At least one traversal
completes unless the exact immediate-mate shortcut returns first; batches/tactical work can
overshoot. Actual elapsed time is recorded.

```powershell
& $py scripts/search_experiments.py benchmark --checkpoint $champion `
    --device cuda --simulations 1000000 --seconds 0.25 --repeats 3 --seed 31102041 `
    --output "${run}/benchmark_cuda_t250ms"
if ($LASTEXITCODE -ne 0) { throw 'Timed benchmark stopped' }
```

## Paired gameplay (expensive)

Each variant plays baseline from 100 seeded fresh openings in both colors: 200 games/variant,
800 games for all four variants per budget mode. The shared checkpoint intentionally isolates
search. Openings are development positions, not checked against all historical training starts.
The tactical arm spends more neural work at equal traversals, making equal-time tests necessary.

```powershell
foreach ($arm in @('cache', 'batch', 'tactical', 'reuse')) {
    & $py scripts/search_experiments.py matches --checkpoint $champion --arm $arm `
        --device cuda --simulations 32 --openings 100 --max-plies 240 --seed 31102042 `
        --output "${run}/matches_${arm}_s32"
    if ($LASTEXITCODE -ne 0) { throw "Fixed-work matches stopped: $arm" }
}
```

```powershell
foreach ($arm in @('cache', 'batch', 'tactical', 'reuse')) {
    & $py scripts/search_experiments.py matches --checkpoint $champion --arm $arm `
        --device cuda --simulations 1000000 --seconds 0.25 `
        --openings 100 --max-plies 240 --seed 31102042 `
        --output "${run}/matches_${arm}_t250ms"
    if ($LASTEXITCODE -ne 0) { throw "Timed matches stopped: $arm" }
}
```

Version-1 JSON stores full PGN text and timing by arm. Confidence intervals bootstrap whole
opening pairs. Capped games score half a point for accounting, but remain separately reported
as unresolved. This is not a rating estimate or confirmation. Investigate tactical regressions
before accepting a speed improvement. Do not repeatedly peek at confirmation while tuning.

Cases/games are individually committed with checksums. Repeat an identical command to resume;
changed code/checkpoint/suite/arm/device/seed/budget requires another output directory. Source hashes
bind the actual implementation, not just a version label. Ctrl+C releases
the writer lock. Interrupted cases/games restart; completed ones are reused.

## Separate failure-driven training

Collection can run independently. Labeling/finalization require the current full-GM corpus to
finish its `finalize` stage first (`training_ready.json` and `dataset.json`). That corpus is read
without modification. Collection uses unchanged production PUCT, keeping the data intervention
separate from search changes. Default opponent is unchanged minimax D2; `--opponent self` tests
self-play separately and samples both actors. New collection settings require new directories.

```powershell
$collection = "${run}/failure_collection"
$failures = "${run}/failure_labels"
$corpus = 'data/experiments/full_gm_rebuild_v1_20261031_01'
& $py scripts/search_experiments.py collect --checkpoint $champion `
    --device cuda --simulations 32 --pairs 50 --max-plies 240 --seed 31102043 `
    --opponent minimax --output $collection
if ($LASTEXITCODE -ne 0) { throw 'Fresh training collection stopped' }
```

Import requires explicit acknowledgement and accepts only this collector's frozen, marked fresh
training games, never evaluation PGNs. Protection includes finalized corpus blocked/validation
inputs, current historical evaluation PGNs, recorded endgame rollout paths, this suite and its
prescribed lines, and search-match PGNs under `data/search_experiments`. Keep evaluation outputs
under that standard directory: custom locations are not automatically discovered. Endgame mirrors
are protected when python-chess validates the transformation; some castling states cannot reflect.

Whole opening/color pairs split before results are known. New validation owns transpositions and
endgame mirrors; training duplicates are dropped. New validation matching historical training is
excluded. Protected keys/files and teacher binary/settings are bound to the journal. Finish
evaluations before import and keep that snapshot stable; changed protections reject resume.

Training positions are selected for estimated cp loss >=150 or a winning-position/non-win
conversion proxy. Capped games remain unresolved. **Outcomes select positions, never value labels.**
All sampled eligible validation positions are retained, rather than selecting only mistakes.
The existing teacher supplies policy and tanh(cp/600) values, using 200k nodes, MultiPV 5,
mate-distance preference, CPU UCI, and no runtime tablebase assistance. Each label commits
individually with audits and forced-played-move analysis. FEN labels omit repetition history.
This is supervised correction, not reinforcement learning.

First time 32 labels, then repeat without the cap to finish:

```powershell
& $py scripts/search_experiments.py labels --collection $collection --corpus $corpus `
    --teacher-config 'data/experiments/endgame_pilot_v1_20261023_01/configs/pilot.yaml' `
    --output $failures --seed 31102043 --positions-per-game 24 --minimum-loss-cp 150 `
    --max-positions 32 --confirm-training-import
if ($LASTEXITCODE -ne 0) { throw 'Failure label timing batch stopped' }
```

```powershell
& $py scripts/search_experiments.py labels --collection $collection --corpus $corpus `
    --teacher-config 'data/experiments/endgame_pilot_v1_20261023_01/configs/pilot.yaml' `
    --output $failures --seed 31102043 --positions-per-game 24 --minimum-loss-cp 150 `
    --confirm-training-import
if ($LASTEXITCODE -ne 0) { throw 'Failure labeling stopped' }
& $py scripts/search_experiments.py finalize --output $failures `
    --replay "${corpus}/dataset.json" `
    --train-template "${corpus}/configs/train_finetune.yaml" --seed 31102043
if ($LASTEXITCODE -ne 0) { throw 'Failure finalization stopped' }
```

The existing replay composer keeps one historical example per fresh example in each split,
preserving labels and fixed cp/600 semantics. Finalization fails if independent new validation
or training failures are absent; collect more in a new run rather than relaxing exclusions.
Exports use existing version-2 tensors. Collection/label/ready records and journal payloads use
version 1. Separate checkpoint/metrics paths prevent overwriting other candidates.
The generated config uses LR 1e-5, up to 8 epochs, patience 3, frozen BatchNorm statistics,
baseline eligibility, and no source multipliers. It is a separate fine-tuning candidate,
not an extension of the current scratch/full-GM experiment.

```powershell
& $py -m chess_ai train --config "${failures}/train.yaml" `
    --init-checkpoint $champion --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Failure candidate training stopped' }
```

Only for interrupted training:

```powershell
$ready = Get-Content "${failures}/training_ready.json" -Raw | ConvertFrom-Json
$last = Join-Path (Split-Path $ready.candidate_checkpoint) 'last.pt'
& $py -m chess_ai train --config "${failures}/train.yaml" --resume $last --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Failure candidate resume stopped' }
```

Evaluate the candidate with unchanged production search to isolate the data intervention:

```powershell
$ready = Get-Content "${failures}/training_ready.json" -Raw | ConvertFrom-Json
foreach ($suite in @('synthetic', 'gm')) {
    $pilot = if ($suite -eq 'synthetic') {
        'data/experiments/endgame_pilot_v1_20261023_01/configs/pilot.yaml'
    } else {
        'data/experiments/gm_endgame_training_v1_20261024_01/configs/pilot.yaml'
    }
    & $py scripts/endgame_pilot.py --config $pilot evaluate `
        --checkpoint $ready.candidate_checkpoint --device cuda --simulations 32 `
        --output "${run}/failure_candidate_${suite}"
    if ($LASTEXITCODE -ne 0) { throw "Failure candidate development stopped: $suite" }
}
```

Review development before nominating at most one confirmation candidate under the existing protocol.
No new value-head architecture, rules engine, external GUI automation, or promotion is implemented.
Tests cover baseline/cache parity across 14 cases, batch visit counts and failure cleanup,
tactical budgets/evasions, history-safe reuse, draw/cache separation, deadlines, checksummed resume,
and fresh collection → mocked teacher → replay/config plus evaluation/mirror exclusions.
Unit tests establish correctness, not strength. Generated artifacts stay local; source, tests,
the fixed development suite, and this README are committed.

## Implementation verification and first smoke result

Ruff formatting/checks and mypy (`src/chess_ai scripts`, 83 source files) pass.
The full pytest suite with the pinned local Stockfish configured reports **391 passed, 1 skipped**.
The skip tests the CUDA-unavailable error path on this CUDA-capable machine. All ten PowerShell
blocks above were parsed with PowerShell's own parser. Champion/registry SHA256 values still match
the original frozen values. The prior accumulated source/docs were preserved in commit `9b2ae16`;
the opt-in experiment implementation and tests are in `1663aed`.

A real-champion CPU smoke run used 32 traversals, one timing repeat, seed 31102041, and the 14
development cases. Local report: `data/search_experiments/implementation_smoke_20261103_02/report.json`.

| Arm | Scored cases solved | Neural positions | Median move seconds |
|---|---:|---:|---:|
| baseline | 2/5 | 394 | 0.074 |
| cache | 2/5 | 350 | 0.493 |
| batch | 3/5 | 383 | 0.831 |
| tactical | 5/5 | 906 | 0.716 |
| reuse (cold roots) | 2/5 | 394 | 0.549 |

Cache matched every baseline move and avoided 44 neural evaluations. Batch inference averaged
5.39 positions/call. The tactical safeguard took all three immediate mates that baseline missed;
the other two scored cases were solved by both. This tiny synthetic result supports the safeguard's
correctness, not a general strength claim. No full games, training, or promotion were performed.
The current corpus-labeling job was running during this smoke pass, and repeated model loads/order
can affect timing. No speedup is established: cache/batch/reuse were slower in this run despite
some reduced inference work. Use repeated CPU/CUDA and equal-time games before accepting them.
