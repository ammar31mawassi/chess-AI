# Endgame pilot — 2026-10-23

Use targeted offline Stockfish supervision first. Keep epoch 60 as champion. Add
Syzygy as a separately designed reference later, and measure a capped GM archive
sample before importing it. This is a hypothesis-driven pilot, not a strength claim.

Follow-up on 2026-10-24: see the [matched retention experiment and completed
500-games-per-GM collection](endgame_retention.md). The original pilot below remains
archived with its rejected updates and unused confirmation pool.

## What the repository and small diagnostics show

The registry was verified against checkpoint SHA-256
`7b2972f1aa5a1a9caa73db7c1c2f60a48061820b66cca5cdc452e99ba0cdc37a`.
Its default remains PUCT 32, `c_puct=1.5`, without a book. Both heads affect search.
The 18 planes omit halfmove clocks and repetition history. PUCT nevertheless retains
the actual board history and uses python-chess for terminal/claimable draws.

Read-only catalog counts:

| Existing source | Training inputs | Training inputs with at most 10 pieces |
|---|---:|---:|
| Original evaluation rebuild | 426,028 | 30,070 |
| Approved personal-game relabel | 51,186 | 5,984 |

The personal-data **selection** suite's frozen champion reached at most 10 pieces
in 22/64 games: 14/32 against local minimax D1 and 8/32 against D2. Across all 64,
25 games ended by repetition; one stopped at the move cap. These are not Stockfish
depths, and these old evaluation games remain exclusion/diagnostic data only.

A seeded eight-game sample (two per opponent-depth/neural-color stratum) reviewed
56 neural decisions with the pinned Stockfish at 50,000 nodes, MultiPV 1. Seven
entries were estimated above +300 cp: three became wins and four repetition draws.
One entry was already below -300 cp and lost. Five sampled decisions lost at least
200 mapped cp; two were in the already-lost game, two abandoned a reported forced
mate while retaining an advantage, and one changed an estimated win to a draw.
Mate-to-cp differences are not ordinary material losses. No balanced entry was
sampled, so this does **not** establish a defense rate or overall endgame skill.

For example, `champion_d1/opening_001_neural_black.pgn` entered at +626 cp for Black.
The sample includes `b3f7` instead of `g8e7` in a reported mating position, and later
`b4d3` instead of `c5c2`, with an estimated +599-to-0 change. This supports testing
conversion/mating technique rather than treating every endgame loss as a late error.
King moves and promotions are tagged, but tags alone do not measure their quality.

Reports are local and evaluation-only:

- `data/experiments/endgame_pilot_v1_20261023_01/diagnosis/report.json`
- `data/experiments/endgame_pilot_v1_20261023_01/teacher_smoke.json`
- `data/experiments/endgame_pilot_v1_20261023_01/search_budget_smoke.json`

The teacher smoke used 28 synthetic positions: 50,000 nodes averaged **0.086 s**;
200,000 nodes averaged **0.213 s** per position on this host. Fifteen best moves
changed, but none changed the coarse +/-300-cp category. Disagreement is not proof
that the larger search is correct. The pilot spends the larger budget on a small pool.

A separate search-only stress check restored eight retrospectively selected
positions with their PGN history. CPU PUCT32 took 0.620 seconds total and PUCT128
2.724 seconds; each had two >=200-cp errors. This tiny biased check does not establish
a search-budget winner. Keep candidate/champion at PUCT32 for the main comparison.
The registered model/search settings were not changed.

## Why these targets

The existing mate mapping differs by only one mapped centipawn for each reported
mate move. At policy temperature 100 cp, mate-in-one and mate-in-five consequently
get nearly equal weight. The new optional `stockfish.mate_distance_temperature: 2.0`
assigns exponentially more policy mass to shorter **reported winning mates**. If
every analyzed move is a losing mate, it prefers delaying mate. Ordinary cp cases
retain the original softmax. Several equally short mates retain equal probability.
These are bounded-search reports, not tablebase proof.

The value remains `stockfish_cp_tanh_v1`, side-to-move `tanh(cp/600)`. It still
saturates near forced mates and cannot teach distance by itself. No new head,
encoder, or incompatible checkpoint/dataset format is introduced. The teacher
option defaults to `None` and is omitted from legacy provenance when disabled,
preserving existing cache signatures. Enabling it binds the temperature into the
teacher/cache identity; the pilot has its own cache. Old runs are unaffected.

Stockfish's own [UCI option implementation](https://github.com/official-stockfish/Stockfish/blob/master/src/engine.cpp)
separately exposes MultiPV, strength, thread, and Syzygy settings. We retain a pinned
binary, one thread, 128 MB hash, full strength, cold searches, five PVs, and a fixed
node budget. Fixed nodes improve reproducibility, but do not guarantee a proof.

## Syzygy decision

Use local tablebases first for an offline evaluation reference, after verifying
downloads and implementing a separately versioned annotation path. No runtime
probing or tablebase downloading is enabled by this pilot.

Per [python-chess's Syzygy API](https://python-chess.readthedocs.io/en/stable/syzygy.html),
WDL is side-to-move: +2 win, 0 draw, -2 loss; +1/-1 are cursed wins/blessed losses
drawn under the fifty-move rule. `probe_wdl` assumes a just-zeroed counter. DTZ is
distance to a capture/pawn move, **not distance to mate**, and rounding matters near
the rule boundary. Castling positions are unsupported. Captures/promotions can
require additional material tables; missing tables must be reported, never treated
as draws. DTZ probing requires both WDL and DTZ files.

A future label adapter should retain raw WDL/DTZ and missing-coverage status.
Do not regress exact WDL directly into the existing cp-tanh value head. Either use
a separate auxiliary objective or version the value semantics and add the needed
clock/history representation. Preserve all equally outcome-preserving legal moves
before considering progress; child scores reverse perspective. Do not choose a
move by simply negating/minimizing raw child DTZ: zeroing moves, mate, clocks and
repetition need explicit handling delegated to python-chess and a validated adapter.
Stockfish's [root tablebase code](https://github.com/official-stockfish/Stockfish/blob/master/src/syzygy/tbprobe.cpp)
provides a primary reference for why root ranking is more involved than a scalar lookup.

The [tablebase generator's size figures](https://github.com/syzygy1/tb#tablebase-files)
are about **378 MB WDL + 561 MB DTZ** through five pieces, versus **68.2 GB + 81.9 GB**
for the six-piece files, with smaller dependencies also needed. Start with a
checksum-verified 3–5-piece set, allowing roughly 1–2 GB free. Six pieces needs
roughly 151 GB before working space; seven is outside this pilot's disk scope.
The download index checked during research was inaccessible, so no guessed bulk
tablebase command is supplied. A later assisted agent must be compared with both
models equally assisted and reported separately from unassisted learning gains.

## GM archive assessment

Public API profiles and archive indexes were checked on 2026-10-23, without a browser
or playing interface. Account mappings come from the API's name, title and stable
player ID fields; no alias was guessed:

| Player / public account | Player ID | Available archive months | Recorded rapid / blitz / bullet games |
|---|---:|---:|---:|
| [Hikaru Nakamura / hikaru](https://api.chess.com/pub/player/hikaru) | 15448422 | 154, Jan 2014–Oct 2026 | 494 / 45,959 / 20,815 |
| [Magnus Carlsen / magnuscarlsen](https://api.chess.com/pub/player/magnuscarlsen) | 3889224 | 67, Dec 2014–Oct 2026 | 277 / 6,951 / 2,306 |
| [Fabiano Caruana / fabianocaruana](https://api.chess.com/pub/player/fabianocaruana) | 11177810 | 104, Mar 2013–Oct 2026 | 459 / 4,809 / 1,596 |

Archive counts are available months, not a continuous range. Game counts are sums
of public win/loss/draw records from each account's `/stats` endpoint and can change.
Index endpoints are `/games/archives`. These sources are overwhelmingly fast chess;
they are not a classical tournament corpus. Longer time controls are worth favoring
for natural planning/defense positions, but coverage still needs measurement.

GM positions could add realistic rook/minor endings and strong resistance. They
can also stop at resignation or timeout before demonstrating mate/promotion, and
the same game can appear in two players' downloads. Fame and the final result do
not make either the moves or values suitable targets. Keep Stockfish supervision
for both colors. Report unique endgame positions per selected game and per teacher
minute, material diversity, exact time controls, duplicates, and nonterminal endings.

Recommendation: after the existing-position pilot, download **one completed month
per player**, then profile at most 300 eligible games in total. Prefer base clocks
of at least three minutes for the first sample; inspect the exact controls and
consider a slower-only follow-up. Expand only if coverage adds material families
or realistic defense that this curriculum lacks. No GM games have been downloaded
or imported here. `new_personal_stockfish_training.py` assumes one username; do not
point it at a combined GM directory or silently import the coverage report.

The [Published-Data API documentation](https://www.chess.com/news/view/published-data-api)
provides read-only profiles, monthly archives, PGNs, and time-control fields. Our
downloader uses serial requests and retries; a recognizable user agent is supported.

## Pilot construction and limitations

The reserved ID is **`endgame_pilot_v1_20261023_01`**. Its configs already exist.
Preparation, labeling, fine-tuning and development evaluation were subsequently run
by the user. See the recorded outcome below before continuing to confirmation.

### Development outcome — 2026-10-24

`best.pt` retained the eligible **epoch-0 baseline**. Every model-state tensor was
checked against the registered epoch-60 champion and is identical; file hashes
differ because the checkpoint metadata differs. The trained epoch-3/epoch-4 files
contain updated state, but did not beat baseline validation loss:

| Checkpoint | Combined validation loss |
|---|---:|
| Baseline / selected `best.pt` | 2.607528 |
| Epoch 1 | 2.655739 |
| Epoch 3 | 2.671214 |
| Epoch 4 / `last.pt` | 2.666579 |

The original run stopped early at epoch 3; its subsequent resume produced epoch 4.
Neither update displaced the baseline. Read-only source diagnostics on the same
644 validation examples found a small fresh-source improvement, outweighed by
historical replay regression:

| Source (322 validation rows each) | Baseline total loss | Epoch-4 total loss |
|---|---:|---:|
| Fresh endgames | 2.412585 | 2.405197 |
| Historical replay | 2.802475 | 2.927961 |

Fresh value MSE improved from 0.182304 to 0.167304 while fresh policy loss worsened
from 2.175806 to 2.183423. Replay policy and value loss both worsened. This suggests
a retention tradeoff in the trained update; it does not establish its gameplay strength.
The diagnostic made no optimizer updates and is saved as
`data/experiments/endgame_pilot_v1_20261023_01/source_diagnostics_20261024.json`.

The selected baseline and champion consequently produced all 41 identical endgame
move sequences: each converted 4/16 estimated winning starts, drew the other 12,
and recorded 17 win-to-nonwin proxy decisions. Whole-game development tied at
11.5/16 versus D1 and 6/16 versus D2. These results compare identical weights;
they do not test the trained epoch-4 model's gameplay.

**Stop this pilot before confirmation.** Keep the registered champion. The reserved
confirmation positions and whole-game seed remain unused. A further development
experiment should address retention before another candidate is nominated; do not
resume simply to accumulate epochs or search confirmation for a favorable checkpoint.

- Natural pool: read only the **training** rows of the original and approved personal
  catalogs, at most 10 pieces, capped at 48 per material/color/old-score bucket.
  The 50k labels choose strata only; the new teacher independently relabels the pool.
- Synthetic pool: 40 roots for each of KQK, KRK, KPK, KPPKP, KRPKR, KBPKB and KNPKN.
  Each root has a three-position legal continuation and color mirrors. All related
  positions share one group. Per material, 60/20/20 percent of root indices are
  assigned to training/development/confirmation **before** labeling.
- Python-chess validates positions/moves. Randomly placed roots pass `is_valid`;
  that is not proof of historical reachability from move one. Natural positions
  supply the reachable component. No random/human move is a target.
- Exact inputs, transpositions, color mirrors, and file reflections are tracked.
  Cross-root generated collisions discard an entire group. Generated roots that
  intersect old train **or** validation inputs are removed as whole groups.
- Original/personal catalog validation and the finalized original/personal/refinement
  datasets protect old validation. Every PGN under the saved evaluation/gate/holdout/
  audit roots vetoes overlapping groups. Evaluation games never supply training rows.
  This preserves the historical artifacts and splits; it does not undo any leakage
  that predates this experiment. Unknown external holdouts must be added to the
  exclusion configuration before preparation, not after training.
- Finalization caps each split/source/material/color/new-score bucket at 24. It
  reports actual coverage rather than duplicating rare examples to pretend perfect
  balance. +/-300 cp defines approximate winning/balanced/losing strata, **not WDL**.
  Review missing buckets before spending on training. Development/confirmation are
  synthetic challenge sets; they do not estimate the natural distribution of games.
- `fresh.pt` contains only training/development. Confirmation stays in the separately
  tagged position/label journal. The existing composer adds equal historical replay
  counts per split. Generated holdouts are screened against that historical corpus.
- Fine-tune a fresh optimizer from the frozen champion: LR `1e-5`, weights 1/1,
  batch 128, eight epochs maximum, patience three, baseline eligible. Compare only
  `best.pt` against champion on development. Retained epochs are debugging artifacts,
  not invitations to keep trying candidates on confirmation.

The new preparation/report schemas use named version 1 formats. Supervised tensors
remain version 2. Frozen positions, source hashes, teacher provenance, journal
checksums, and checkpoint hashes protect resume. The expensive exclusion scan
commits each PGN; labeling commits each position; rollouts commit each ply. Preparation
can repeat its initial catalog/protection pass if interrupted before `plan.json`.
Completed reports are retained. Changing the corpus, seed, teacher, checkpoint or
evaluation settings requires a new run/output. Confirmation locks one candidate
hash and PUCT32 settings in `confirmation_pair.json`.

## PowerShell execution order

Run these blocks from the actual repository. Each expensive stage is explicit.

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$run = 'endgame_pilot_v1_20261023_01'
$root = "data/experiments/$run"
$cfg = "$root/configs"
$champion = 'checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt'
$candidate = "checkpoints/$run/best.pt"
```

1. Prepare the bounded pool and scan historical exclusions. This hashes large source
catalogs and scans roughly 11,000 archived evaluation PGNs; it can take minutes.
Rerun the identical command after interruption.

```powershell
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" prepare
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed; inspect before labeling' }
```

2. Label 64 positions first for timing, then continue the same journal. CPU only.
The pilot pool is capped at roughly 3,120 positions before exclusion/deduplication.
The measured small-position rate implies about 11 minutes of serial analysis at
that upper bound, plus hashing/cache/I/O and harder positions. Use the actual ETA.

```powershell
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" labels --max-positions 64
if ($LASTEXITCODE -ne 0) { throw 'Bounded labeling failed' }
Get-Content "$root/label_progress.json"
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" labels
if ($LASTEXITCODE -ne 0) { throw 'Labeling stopped; repeat the same command to resume' }
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" finalize
if ($LASTEXITCODE -ne 0) { throw 'Finalization failed' }
Get-Content "$root/finalized.json"
```

Inspect family/color/score coverage. If it is inadequate, reserve a new configuration
before training; do not reclassify confirmation positions as training data.

3. Compose 50/50 replay once, then train. Allow about **1 GB free for this experiment**:
roughly tens to low hundreds of MB for tensors/journals/cache, about 7 MB per current
checkpoint, plus potentially larger evaluation logs. Source catalogs are not copied.
Training time depends on device; use the first epoch's timing. CUDA is optional.

```powershell
& $py scripts/compose_stockfish_refinement.py --config "$cfg/compose.yaml"
if ($LASTEXITCODE -ne 0) { throw 'Replay composition failed' }
& $py -m chess_ai train --config "$cfg/train.yaml" --device cuda --init-checkpoint $champion
if ($LASTEXITCODE -ne 0) { throw 'Training stopped; inspect and resume a saved epoch' }
```

Keep an existing `replay.pt`; do not compose it again. After a saved epoch, resume:

```powershell
& $py -m chess_ai train --config "$cfg/train.yaml" --device cuda --resume "checkpoints/$run/last.pt"
```

4. Development endgames: same starts/budget, no book/tablebase assistance. Each side
is the neural player in separate color-balanced roots; the opponent is local D2.
The command reviews neural decisions with Stockfish offline, but Stockfish never
chooses a gameplay reply. Rerun an identical command to resume saved plies.

```powershell
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" evaluate --checkpoint $champion `
  --output "$root/development_champion" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Champion endgames failed' }
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" evaluate --checkpoint $candidate `
  --output "$root/development_candidate" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Candidate endgames failed' }
```

Up to two roots per material/color/score bucket, with one representative per generated
group, are chosen independently of the checkpoint. Coverage can be uneven. This
can take much longer than labeling: up to 160 plies per root plus neural searches,
minimax and offline teaching. Reports distinguish natural draws from unresolved caps,
entry categories, conversion outcomes, losses from balanced positions, promotion,
stalemate/repetition/fifty-move tags, and neural time. A king-move count is descriptive.
For a search-only ablation, repeat development with `--simulations 128` and new output
paths for **both** checkpoints; do not fold those results into the PUCT32 learning claim.

5. Whole-game development regression check, still just the same one candidate:

```powershell
foreach ($depth in @(1, 2)) {
  & $py -m chess_ai paired-audit --candidate $candidate --champion $champion `
    --openings 8 --audit-seed 25102037 --opponent-depth $depth `
    --search-simulations 32 --c-puct 1.5 --device cuda `
    --exclude-manifest "$root/training_manifest.json" `
    --exclude-additional-manifest 'data/experiments/stockfish_rebuild_eval_v2_20261011_01/relabel/training_manifest.json' `
    --pgn-dir "data/games/evaluation/${run}_development_d$depth"
  if ($LASTEXITCODE -ne 0) { throw "Development audit failed: D$depth" }
}
```

Inspect the two endgame reports and two whole-game reports **before** the next block.
If `best.pt` is still the unchanged eligible baseline, stop: there is no new candidate
to confirm. Do not scan later epochs against the reserved confirmation set.
Proceed only if conversion improves, defense does not regress where measured, and
whole-game aggregate/per-depth/standard-start scores do not regress. Inspect concrete
win-to-draw/loss mistakes and unresolved games, not just mean loss or D1/D2 wins.
With sparse buckets, report uncertainty; a few successes do not establish strength.

6. Freeze this single candidate, then confirm once on reserved endgames:

```powershell
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" evaluate --checkpoint $candidate `
  --split confirmation --output "$root/confirmation_candidate" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Candidate confirmation stopped' }
& $py scripts/endgame_pilot.py --config "$cfg/pilot.yaml" evaluate --checkpoint $champion `
  --split confirmation --output "$root/confirmation_champion" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Champion confirmation stopped' }
```

7. Fresh whole-game confirmation: exclude development starts too. The helper below
writes UTF-8 without a BOM, including on Windows PowerShell 5.1.

```powershell
$devStarts = foreach ($depth in @(1, 2)) {
  $report = Get-Content "data/games/evaluation/${run}_development_d$depth/report.json" -Raw | ConvertFrom-Json
  foreach ($opening in $report.openings) { @{ opening_fen = $opening.fen } }
}
$startsJson = @{ pairs = @($devStarts) } | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "$root/development_starts.json"), $startsJson, [System.Text.UTF8Encoding]::new($false))
foreach ($depth in @(1, 2)) {
  $direct = @()
  if ($depth -eq 2) { $direct = @('--direct-baseline') }
  & $py -m chess_ai paired-audit --candidate $candidate --champion $champion `
    --openings 32 --audit-seed 27102036 --opponent-depth $depth `
    --search-simulations 32 --c-puct 1.5 --device cuda @direct `
    --exclude-manifest "$root/training_manifest.json" `
    --exclude-additional-manifest "$root/development_starts.json" `
    --exclude-additional-manifest 'data/experiments/stockfish_rebuild_eval_v2_20261011_01/relabel/training_manifest.json' `
    --pgn-dir "data/games/evaluation/${run}_confirmation_d$depth"
  if ($LASTEXITCODE -ne 0) { throw "Confirmation audit failed: D$depth" }
}
```

The existing whole-game audit does not resume partial games and refuses existing
PGNs. After interruption use a new output suffix with the **same checkpoint, seed
and settings**, recording the abandoned attempt. Do not test another candidate on
these confirmation suites. Review paired root/game results and uncertainty; count
both natural completion and cap rates. Confirmation must support endgame improvement
and whole-game non-regression before a separate, explicitly authorized promotion.
No command above promotes a checkpoint.

After a hard kill, inspect a reported lock's PID and remove only that stale lock
once its process has exited. Ctrl+C releases locks normally. Do not delete journals.

## Optional GM coverage pilot, separate from training

These download commands fetch raw archives only. September 2026 is a completed month;
the downloader checks availability. Run serially, then measure coverage:

```powershell
foreach ($player in @('hikaru', 'magnuscarlsen', 'fabianocaruana')) {
  & $py scripts/download_chesscom_games.py --username $player --start 2026-09 --end 2026-09
  if ($LASTEXITCODE -ne 0) { throw "Archive download failed: $player" }
}
& $py scripts/profile_endgame_pgns.py `
  --source-dir downloaded_datasets/chesscom/hikaru `
  --source-dir downloaded_datasets/chesscom/magnuscarlsen `
  --source-dir downloaded_datasets/chesscom/fabianocaruana `
  --username hikaru --username magnuscarlsen --username fabianocaruana `
  --max-games 300 --minimum-base-seconds 180 --seed 23102036 `
  --output "$root/gm_coverage.json"
```

The profiler validates standard-chess mainlines, rejects evaluation markers, removes
duplicate movetext across accounts, detects conflicting duplicate game URLs, samples
whole games deterministically, and reports exact controls/material coverage. It
creates **no training dataset or Stockfish labels**. Its cap is a total cap, so inspect
player counts; one prolific account can dominate. A future import should cap each
player/time-control stratum, preserve whole-game groups, screen all historic/new
holdouts, and compare an equal-position GM arm against this curriculum. Only after
that explicit import should the shared relabeler be used. Do not collect all careers
or use game results as value targets.

## Validation

Verified on 2026-10-23: Ruff formatting/checks and mypy passed; full pytest passed
**336 tests with one skip** (the CUDA-unavailable error-path test is skipped because
CUDA is available). The pinned local UCI smoke tests passed. Checks use
`.venv\Scripts\python.exe`, with the local UCI smoke enabled when `CHESS_AI_STOCKFISH` and
`CHESS_AI_STOCKFISH_COMMIT` are set. Tests cover deterministic legal generation,
group/symmetry isolation, old holdout exclusion, immutable inputs, label checksums,
partial labeling/finalization/rollout recovery, one-candidate confirmation, archive
deduplication, stalemate/cap handling, and opt-in mate-policy cache separation.
