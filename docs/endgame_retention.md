# Endgame retention experiment and 1,500-game GM collection

## Matched training experiment — 2026-10-24

The original endgame pilot retained the champion as its eligible baseline. Its
trained epoch-4 model slightly improved fresh endgame validation but regressed on
historical replay. A read-only diagnostic restored only the champion's BatchNorm
buffers in that trained model, without writing a candidate checkpoint:

| In-memory diagnostic | Fresh validation loss | Replay validation loss |
|---|---:|---:|
| Champion | 2.412585 | 2.802475 |
| Original trained epoch 4 | 2.405197 | 2.927961 |
| Same epoch-4 weights with champion BatchNorm buffers | 2.406087 | 2.803765 |

This identifies normalization drift as a plausible contributor, rather than
assuming the learning rate or lack of GM games explains the regression. It does
not prove a gameplay improvement. The diagnostic is saved at
`data/experiments/endgame_pilot_v1_20261023_01/batch_norm_diagnostic_20261024.json`.

[PyTorch's BatchNorm documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)
distinguishes learnable affine parameters from running statistics used for inference.
The statistics update separately from the optimizer's learning rate. Small optimizer
steps can therefore accompany a substantial change in inference normalization.

The completed training experiment is **`endgame_retention_v1_20261024_01`**:

- Both arms start from the frozen epoch-60 champion with a fresh optimizer and the
  same seed 23102036, exact 2,986-row replay artifact, 50/50 source counts, explicit
  splits, batch order, LR `1e-5`, loss weights, and eight-epoch limit/patience three.
- The only treatment difference is `training.freeze_batch_norm_statistics`.
  `false` is the existing behavior. `true` places BatchNorm layers in evaluation
  mode during optimization, retaining their running means/variances/counters.
  Their affine parameters and the remaining model parameters still train.
- The new option defaults to false and is omitted from checkpoint metadata when
  disabled. Enabled checkpoints record it, and resume rejects a change of mode.
  Tensor/checkpoint shapes, target semantics and version compatibility are unchanged.
- The same source pilot development sets are reused. Confirmation remains reserved.
  No new labeling, GM data or transpositions were added to either training arm.

| Arm | Stopped after epoch | Selected epoch | Selected validation loss |
|---|---:|---:|---:|
| Champion baseline | — | 0 | 2.607528 |
| Control: ordinary BatchNorm | 3 | 0 | 2.607528 |
| Fixed BatchNorm statistics | 5 | 2 | **2.604783** |

The control reproduced the prior first three epoch metrics. The treatment selected
an actual update: 27 model-state tensors differ from the champion, while all 21
BatchNorm buffers are identical. The selected treatment's fresh/replay losses are
2.403820 / 2.805746. Replay retention is much better than the original update but
is not exact, and the combined gain is small. This is a development hypothesis,
not evidence for promotion.

Artifacts:

- `data/experiments/endgame_retention_v1_20261024_01/experiment.json`: frozen input,
  champion, config hashes and the declared comparison.
- `data/experiments/endgame_retention_v1_20261024_01/source_diagnostics.json`:
  read-only baseline/control/treatment validation metrics.
- `checkpoints/endgame_retention_v1_20261024_01_frozen_bn/best.pt`: selected epoch 2.
- `data/experiments/endgame_retention_v1_20261024_01/development_candidate/`:
  per-ply resumable PUCT32 evaluation on the original 41 development roots.

The epoch-0 control does not need redundant gameplay evaluation. The champion's
existing development report provides the matched baseline. The completed evaluation
verified identical roots, labels, device, seed, search budget and opponent settings:

| Endgame development outcome | Champion | Fixed-statistics candidate |
|---|---:|---:|
| Estimated winning starts converted | 4/16 | 3/16 |
| Balanced starts: wins / draws / losses / unresolved | 2 / 9 / 0 / 0 | 2 / 7 / 0 / 2 |
| Losing starts: draws / losses | 10 / 4 | 9 / 5 |
| Win-to-nonwin proxy decisions | 17 | 11 |
| Large mapped-cp losses / reviewed moves | 82 / 1,070 | 83 / 1,233 |

The candidate improved one pawn-ending draw to a win, but changed another pawn
win and one basic-mating win to draws, and one losing minor ending from draw to
loss. Two balanced positions hit the cap and remain unresolved. Thirteen of the
41 move sequences were unchanged. Fewer per-move proxy mistakes did not translate
into better conversion outcomes.

**Verdict: reject this candidate at development.** Keep the champion and leave
confirmation unused. The normalization intervention improved retention on validation
targets but has not improved measured play. Do not run the conditional whole-game
commands below for this rejected candidate or select another epoch on confirmation.
The full comparison and artifact digests are saved in `results.json` in the run directory.

## PowerShell commands

The two training arms have already run. Keep their files and metrics. These commands
document the exact fresh invocations; do not run them over existing outputs:

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
$py = '.\.venv\Scripts\python.exe'
$run = 'endgame_retention_v1_20261024_01'
$root = "data/experiments/$run"
$pilot = 'data/experiments/endgame_pilot_v1_20261023_01'
$champion = 'checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt'
$candidate = "checkpoints/${run}_frozen_bn/best.pt"
```

For a separate experiment, reserve a new ID with
`scripts/new_endgame_retention.py --run-id YOUR_NEW_ID`. Its two YAMLs differ only in
normalization behavior and output paths. The creation script verifies the registered
champion and that the existing replay excludes reserved confirmation positions.

```powershell
# Fresh training only: both invocations are already completed for this run.
foreach ($arm in @('control', 'frozen_bn')) {
  & $py -m chess_ai train --config "$root/configs/train_$arm.yaml" `
    --device cuda --init-checkpoint $champion
  if ($LASTEXITCODE -ne 0) { throw "Retention training failed: $arm" }
}
```

Do not extend the stopped runs just to add epochs. For an interrupted run that had
not completed its planned stopping decision, resume its matching mode and paths:

```powershell
$arm = 'frozen_bn'
& $py -m chess_ai train --config "$root/configs/train_$arm.yaml" --device cuda `
  --resume "checkpoints/${run}_$arm/last.pt"
```

Repeat the following evaluation only after an existing invocation has exited.
It resumes committed plies and does not touch training or confirmation:

```powershell
& $py scripts/endgame_pilot.py --config "$pilot/configs/pilot.yaml" evaluate `
  --checkpoint $candidate --output "$root/development_candidate" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Development endgames stopped; repeat to resume' }
```

Compare that report with `$pilot/development_champion/report.json`. For a future
candidate with a useful development improvement, the expensive whole-game development
checks use the same seed and exclusion manifests as the original experiment:

```powershell
foreach ($depth in @(1, 2)) {
  & $py -m chess_ai paired-audit --candidate $candidate --champion $champion `
    --openings 8 --audit-seed 25102037 --opponent-depth $depth `
    --search-simulations 32 --c-puct 1.5 --device cuda `
    --exclude-manifest "$pilot/training_manifest.json" `
    --exclude-additional-manifest 'data/experiments/stockfish_rebuild_eval_v2_20261011_01/relabel/training_manifest.json' `
    --pgn-dir "data/games/evaluation/${run}_development_d$depth"
  if ($LASTEXITCODE -ne 0) { throw "Development whole-game audit failed: D$depth" }
}
```

The existing whole-game runner cannot resume partial PGNs: keep the same model/seed
and use a new output suffix if interrupted. Only a nominated candidate that improves
endgames without whole-game regression should proceed to the source pilot's still-unused
confirmation protocol. Candidate/champion use identical PUCT32 settings throughout.
No automatic promotion is implemented.

## GM collection: the requested 500 per player

The user increased the target to **500 unique games each** for Hikaru Nakamura,
Magnus Carlsen and Fabiano Caruana: **1,500 selected games total**. Collection is
complete under **`data/experiments/gm_endgame_1500_v2_20261024_01/`**.

Public API profiles were rechecked against stable player IDs 15448422, 3889224 and
11177810. Requests were serial. The collector started with the last completed month,
September 2026, then moved backward only for players below quota. It needed March
through September for Magnus/Caruana and August/September for Hikaru. The raw archives
occupy about 7.3 MiB and remain in `downloaded_datasets/chesscom_gm_pilot_20261024/`.

The selected sample uses base clocks of at least 180 seconds and complete standard
chess PGNs. Python-chess validates each mainline; duplicate movetext across accounts
is removed, and conflicting game URLs are rejected. One owner is assigned per game,
so an encounter between two tracked GMs is counted once toward collection quotas.
Each player owns exactly 500 selections. Participation counts can exceed 500 because
they can also appear as another selected player's opponent.

Final round-trip validation caught a selected archive record that was abandoned
before any move. The superseded v1 snapshot is retained for traceability. The v2
coverage/collection formats explicitly require recorded moves and exclude two empty
raw records before deduplication. The active v2 snapshot was independently reparsed
and contains exactly 1,500 nonempty legal games with all three 500-game quotas filled.

| Selected-sample coverage | Count |
|---|---:|
| Unique games | 1,500 |
| Games reaching at most 10 pieces | 496 |
| Unique nonterminal endgame inputs | 15,105 |
| Rook-family inputs | 4,221 |
| Minor-piece-family inputs | 3,529 |
| Pawn-family inputs | 705 |
| Basic-mating inputs | 113 |
| Mixed-material inputs | 6,537 |
| Inputs with at most five pieces | 2,907 |

Exact time controls: 1,101 at 180 seconds, 11 at 180+1, 298 at 300 seconds, and 90 at
600 seconds. These online archives remain mostly blitz; their coverage should not be
described as a classical GM corpus. There are 1,217 decisive nonterminal endings,
which include resignation, timeout and other causes. The scarcity of basic mating
positions supports retaining the targeted curriculum alongside natural GM positions.
All counts are descriptive, without Stockfish quality labels or a full historical
holdout/novelty exclusion pass for training.

A preliminary read-only lookup found **15,095 of these inputs absent from both
prior Stockfish catalogs**. Ten matched old training inputs; none matched their
validation inputs or the reserved synthetic endgame confirmation inputs. This
supports the collection's novelty, but still does not replace the full historical
evaluation-PGN and dataset exclusion pass required before training. See `novelty.json`.

The collection contains:

- `source_games/selected_games.pgn`: exactly 1,500 selected legal mainlines. Original
  comments/variations are omitted from this normalized snapshot; raw monthly PGNs
  retain the source content.
- `coverage.json`: input/snapshot SHA-256, exact controls, material counts, selection
  digests and their assigned player quotas.
- `collection.json`: verified identities, months and final collection status.
- `coverage_YYYY-MM.json`: resumable per-month coverage decisions.

The snapshot and metadata are **raw external position sources, not training data**.
There have been zero Stockfish labeling calls or optimizer updates on GM games.
An explicit multi-player import must next reserve whole-game train/development/
confirmation groups, exclude historical evaluation inputs and the current endgame
confirmation pool, deduplicate transpositions, cap correlated positions per game,
and use teacher targets for both colors. Never train on game results or the coverage
report, and never point the single-username personal importer at this combined PGN.

The [Published-Data API](https://www.chess.com/news/view/published-data-api) supplies
the public profile/monthly PGN endpoints. The collector creates no account sessions
and uses no browser or playing interface.

To verify/resume this exact collection, run:

```powershell
& $py scripts/collect_gm_pilot.py `
  --raw-dir downloaded_datasets/chesscom_gm_pilot_20261024 `
  --report-dir data/experiments/gm_endgame_1500_v2_20261024_01 `
  --games-per-player 500 --latest-month 2026-09 --max-months 12 `
  --minimum-base-seconds 180 --seed 23102036
if ($LASTEXITCODE -ne 0) { throw 'GM collection stopped; repeat the same command' }
```

The bounded month cap fails clearly if quotas cannot be filled. Historical downloads
are reused; an edited archive or selected snapshot fails hash validation. Use a new
report directory for a different selection. The profiler also accepts optional
`--max-games-per-player` and `--snapshot-dir`; it refuses a snapshot with incomplete
per-player quotas, and keeps deterministic ownership separate from participation.

## Validation

Ruff formatting/checks and mypy passed. Full pytest passed **340 tests with one
expected CUDA-unavailable-test skip**; the pinned local Stockfish smoke passed.
Additional coverage verifies unchanged BatchNorm buffers across optimization/resume,
updated affine parameters, default behavior, rejection of changed resume mode, exact
per-player quotas, cross-account deduplication, and snapshot round-trip/recovery.
The champion registry/checkpoint and source datasets are preserved.
