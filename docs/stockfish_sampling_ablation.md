# Stockfish sampling diagnosis and next training experiment

The 2026-10-07 collection, composition, 60-epoch training run, and gameplay selection are complete.
Epoch 120 remains the champion. This 2026-10-11 investigation evaluated existing checkpoints and
archived evaluation PGNs; it performed no new training or game generation.

## Findings

Source diagnostics use the original seed 2037, `composition_group_id` split, legal masks, and
stored loss weights. All 18,088 held-out examples were evaluated for epoch 120, candidate epoch 25,
and the validation-selected candidate epoch 43. These are development diagnostics on an already
used holdout, not fresh evidence for promotion.

| Source | Training examples | Expected training draws | Expected weighted draw mass | Held-out examples |
| --- | ---: | ---: | ---: | ---: |
| Historical replay | 59,904 | 86.64% | 65.22% | 15,380 |
| Named openings | 1,892 | 10.55% | 31.71% | 449 |
| New Stockfish continuations | 11,809 | **2.81%** | **3.08%** | 2,259 |

Game-balanced sampling equalizes game IDs across the combined dataset, not sources. The teacher
source has only 504 training games versus 15,544 replay games and 1,892 opening example IDs.
Consequently, the teacher's 16.04% share of training rows becomes 2.81% of expected draws. Loss
weights operate after sampling and do not fix that exposure. Weighted draw mass is a ratio of
expected masses; minibatch weight normalization means it is not an exact gradient contribution.

| Weighted held-out metric | Epoch 120 | Candidate epoch 43 |
| --- | ---: | ---: |
| Replay move top-1 accuracy | 22.55% | 21.42% |
| Opening move top-1 accuracy | 16.67% | 25.83% |
| Stockfish move top-1 accuracy | 14.18% | 14.54% |
| Stockfish policy cross-entropy | 3.2280 | 3.1803 |
| Stockfish outcome-value MAE | 0.6810 | 0.7508 |

Unweighted Stockfish top-1 accuracy moved from 15.76% to 16.07%. The largest measured learning gain
was on openings; replay accuracy regressed and teacher accuracy barely changed. These observations
make teacher underexposure a useful hypothesis to test, rather than proving it caused every loss.
Top-1 accuracy compares the greedy move with the target argmax; a different reasonable teacher move
still counts as a mismatch. Outcome-value MAE measures the archived game result, not Stockfish cp.

The local UCI review covered the standard opening plus the first two varied openings at D1/D2,
both colors, for each model: 24 archived games and 344 sampled neural moves. There were 36 estimated
losses of at least 200 cp in 175 champion moves and 38 in 169 candidate moves. These include repeated
positions: 159 distinct FENs for the champion and 151 for the candidate. This is a bounded
descriptive review, not an independent statistical comparison.

Recurring themes were king moves, missed preferred captures, and unrecapturable piece captures in
the teacher PV. Both models still made `Qf3xb7` on varied opening 2, where the 5,000-node teacher
preferred `Nb1c3` and estimated an 899-cp loss for the queen capture. Both played `Kf2e3` on opening 1
instead of the teacher's `e2e4`, with an estimated 246-cp loss. The review also found allowed mates
in one. Best-move and forced-root scores come from separate fixed-node searches; their difference
is an estimate. PV capture tags do not prove a piece is hanging in every continuation. Already-lost
positions can conceal additional mistakes. All reviewed PGNs and FENs remain evaluation-only.

Detailed local reports:

- `data/diagnostics/stockfish_teacher_v1_20261007_201114_88fd0daf_20261011/sources.json`
- `data/diagnostics/stockfish_teacher_v1_20261007_201114_88fd0daf_20261011/games.json`

## Prepared matched experiment

Configs are already reserved under
`data/experiments/stockfish_sampling_ablation_20261011_01/configs/`. Do not rerun preparation for
this ID. The experiment manifest records the paths and single changed factor.

Both variants independently start from epoch 120, reuse the frozen 91,693-example replay, and use
seed 2037, the original grouped holdout, LR `5e-7`, policy/value weights `1/0.1`, legal masks,
game-balanced draws and stored outcome weights. They both run 10 epochs with the same cosine
schedule, retain epochs 5/10 plus best/last/baseline, disable patience stopping, and select `best.pt`
by validation policy loss. Compared with the old 60-epoch run, these shared settings define a shorter
experiment; the comparison that isolates the factor is **new control versus new teacher_focus**.

- `control`: original source exposure.
- `teacher_focus`: multiplies only `stockfish_continuations` sampling mass by eight.

The focus run should draw about **18.78%** Stockfish positions instead of 2.81%, with an expected
weighted draw mass share of 20.25%. It still retains historical/opening rehearsal. It changes
sampling probabilities, not the saved policies, value targets, or per-example loss weights. No new
collection, composition, external imports, or architecture change is required.

Run these commands from the repository root. Both training commands initialize from the same
checkpoint; do not initialize the focus variant from the control's output.

```powershell
$ablationCfg = "data/experiments/stockfish_sampling_ablation_20261011_01/configs"
$baseline = "checkpoints/joint_policy_value_v3/epoch_0120.pt"

.\.venv\Scripts\python.exe -m chess_ai train --config "$ablationCfg/control_train.yaml" `
  --init-checkpoint $baseline --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Control training failed; inspect before proceeding' }

.\.venv\Scripts\python.exe -m chess_ai train --config "$ablationCfg/teacher_focus_train.yaml" `
  --init-checkpoint $baseline --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Teacher-focus training failed; inspect before proceeding' }
```

The old epoch took roughly 19–24 seconds on this machine. New per-source evaluation adds a second
held-out pass per epoch, plus initial loading/mask construction. Allow several minutes per training
variant; this is a rough estimate, not a measured duration for the new configuration.

The JSONL metrics now include `validation_by_source`. Each source records row count, weighted and
unweighted cross-entropy, MSE, top-1/top-5 accuracy and value MAE using the same definitions as the
global metrics. Check teacher gains alongside replay/opening regressions.

```powershell
Get-Content data/metrics/stockfish_sampling_ablation_20261011_01_control.jsonl -Tail 1
Get-Content data/metrics/stockfish_sampling_ablation_20261011_01_teacher_focus.jsonl -Tail 1
```

After both trainings, run the two selectors sequentially. They use the same **fresh** seed
10112035 and eight shared openings with both colors at D1/D2, excluding the same training manifests.
Eight openings are a small diagnostic suite, not sufficient evidence for promotion. Selection can
take substantially longer than the training; it plays up to 160 games per variant, depending on
which checkpoints duplicate a retained epoch. It performs no training and does not auto-promote.

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select --config "$ablationCfg/control_select.yaml" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Control selection failed' }

.\.venv\Scripts\python.exe -m chess_ai gameplay-select --config "$ablationCfg/teacher_focus_select.yaml" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Teacher-focus selection failed' }
```

Compare points, D1/D2 and standard-start regressions, and source metrics. If focus improves teacher
metrics but loses more games or forgets replay, keep epoch 120 and reject it. If it improves gameplay
and passes the existing gates, confirm with fresh paired greedy and PUCT audits before manual
promotion. Keep value-loss removal and learning-rate changes for later separate experiments.

If interrupted after a complete epoch, resume the **same** variant:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train --config "$ablationCfg/control_train.yaml" `
  --resume checkpoints/stockfish_sampling_ablation_20261011_01_control/last.pt --device cuda

.\.venv\Scripts\python.exe -m chess_ai train --config "$ablationCfg/teacher_focus_train.yaml" `
  --resume checkpoints/stockfish_sampling_ablation_20261011_01_teacher_focus/last.pt --device cuda
```

Do not rerun fresh initialization into existing outputs. If selection is interrupted, retain its
partial evaluation PGNs and choose new report/PGN paths before restarting; selection has no resume
journal. For another matched experiment, reserve a different ID:

```powershell
.\.venv\Scripts\python.exe scripts/new_stockfish_ablation.py --run-id YOUR_NEW_ID `
  --source-train data/curricula/stockfish_teacher_v1_20261007_201114_88fd0daf/configs/train.yaml `
  --source-select data/curricula/stockfish_teacher_v1_20261007_201114_88fd0daf/configs/select.yaml
```

## Reproducing diagnostics and schema additions

Choose new output paths for each diagnostic run; existing reports are never overwritten.

```powershell
.\.venv\Scripts\python.exe -m chess_ai diagnose-training `
  --config data/curricula/stockfish_teacher_v1_20261007_201114_88fd0daf/configs/train.yaml `
  --checkpoint checkpoints/joint_policy_value_v3/epoch_0120.pt `
  --checkpoint checkpoints/stockfish_teacher_v1_20261007_201114_88fd0daf_candidate/best.pt `
  --output data/diagnostics/YOUR_NEW_ID/sources.json --device cuda

.\.venv\Scripts\python.exe -m chess_ai diagnose-games `
  --config data/curricula/stockfish_teacher_v1_20261007_201114_88fd0daf/configs/collect.yaml `
  --selection-report data/games/evaluation/stockfish_teacher_v1_20261007_201114_88fd0daf_selection/report.json `
  --candidate checkpoints/stockfish_teacher_v1_20261007_201114_88fd0daf_candidate/best.pt `
  --output data/diagnostics/YOUR_NEW_ID/games.json `
  --nodes 5000 --openings 2 --positions-per-game 16 --seed 2040
```

`diagnose-training` runs inference on the exact training-config holdout, checks explicit source
labels, records dataset/checkpoint SHA-256, split settings and holdout-group digest, and reports
expected sampler coverage. It never optimizes or changes the dataset. `diagnose-games` validates
evaluation-only PGNs, legal PVs and checkpoint digests, and separately invokes local Stockfish for
the best root and archived actual root. Hashes are cleared between searches and the existing deadline
applies; these forced-root scores bypass training caches. The review config overrides only analysis
nodes/depth/MultiPV/cache for this distinct diagnostic provenance. It does not edit the signed run.

New optional training settings are `validation_source_key` (default null), `sampling_source_key`
(default `composition_source`), and `source_sampling_multipliers` (default empty mapping). Source
multipliers must be finite positive numbers; configured sources must exist in the training split,
and missing source labels fail clearly. Unspecified sources use multiplier 1. The sampler remains
seeded per epoch. Validation is evaluated sequentially without resampling or source multipliers.
Legacy configs retain their sampling behavior and avoid extra source inference by default.

Source/game diagnostic reports and experiment manifests each have independent format names and
version 1. The new metrics/config fields are additive; dataset and checkpoint tensor semantics and
format versions remain unchanged. Neither report is a training-compatible dataset.

Verification on 2026-10-11: Ruff formatting/checks and mypy passed; full pytest passed **267 tests
with one CUDA-availability skip**, including real local UCI forced-root and MultiPV regressions.
At the preparation stage, both train/select configs were parsed and validated with unused output
paths. Training and subsequent evaluation results are recorded below. No commit or push was made.

## Completed ablation and fresh audits (2026-10-11)

Both matched variants completed ten epochs. Neither gameplay selector found an eligible replacement
for `checkpoints/joint_policy_value_v3/epoch_0120.pt`. Control epoch 10 scored 4/32 versus the
baseline's 3.5/32 on the small selection suite; this half-point gain was below the required
one-point improvement. Teacher-focus epoch 10 scored 3/32 on that same suite.

Control epoch 10 was then audited against epoch 120 using two fresh seeds, 11112035 and 12112035,
with 16 openings per seed, D1 and D2 minimax opponents, both colors, and greedy neural policy
(`search_simulations=0`). The 30 varied opening positions were disjoint from recorded training
starts, previous selection suites, and each other. The standard starting position remained a
repeated regression anchor. Each model played 32 games per seed/depth combination.

| Seed | Opponent | Control epoch 10 points / 32 | Epoch 120 points / 32 |
| --- | --- | --- | --- |
| 11112035 | D1 | 5.0 | 5.0 |
| 11112035 | D2 | 0.0 | 0.0 |
| 12112035 | D1 | 3.5 | 4.5 |
| 12112035 | D2 | 0.0 | 0.0 |

No suite passed the promotion gate. Across D1 games, control scored 8.5/64 versus epoch 120's
9.5/64; its Black score fell to 3.5/32 versus 5.5/32. Both models lost every D2 game, so that
comparison has a floor effect and does not establish equal strength. The paired bootstrap over
30 varied D1 openings also spans zero improvement. The earlier small gain did not replicate:
**retain epoch 120**.

The audit plan, four reports, all 256 evaluation-only PGNs, and verified aggregate summary are
saved under `data/games/evaluation/stockfish_control_epoch10_fresh_20261011_01/`. Every saved game
was parsed with python-chess and checked against its report, including legality, result, color,
checkpoint, starting FEN, and move count. Both checkpoint SHA-256 hashes remained unchanged.
No audit games were imported into training, no training updates were performed by these audits,
and no checkpoint was promoted. Ruff formatting/checks, mypy, and the full pytest suite were
rerun successfully: **267 passed, one CUDA-availability skip**.
