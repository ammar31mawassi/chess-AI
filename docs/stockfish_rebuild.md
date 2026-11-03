# Stockfish position-evaluation rebuild

This experiment trains a new model from seeded random weights on Stockfish move and position-value
targets. Epoch 120 stays immutable. Stockfish runs locally on CPU through python-chess UCI;
neural training/inference may use CUDA. There is no online chess, external GUI automation,
automatic promotion, or import of evaluation games.

The prepared ID is `stockfish_rebuild_eval_v2_20261011_01`. Its configs already exist. Run stages
from the repository root; preparation has not started full analysis or training.

## Inventory and pilot

```powershell
$cfg = "data/experiments/stockfish_rebuild_eval_v2_20261011_01/configs"
.\.venv\Scripts\python.exe -m chess_ai stockfish-rebuild --config "$cfg/relabel.yaml" --stage inventory
if ($LASTEXITCODE -ne 0) { throw 'Inventory failed; inspect before proceeding' }
.\.venv\Scripts\python.exe -m chess_ai stockfish-rebuild --config "$cfg/relabel.yaml" --stage pilot
if ($LASTEXITCODE -ne 0) { throw 'Pilot failed; inspect before proceeding' }
```

`relabel/inventory.json` binds exact source paths, SHA-256 digests, counts, and limitations. SQLite
stores unique positions; a separate manifest excludes training positions from future random starts.
The preparation snapshot lists all currently saved training datasets and curriculum dataset shards.
Composed copies are skipped only if their originals are listed. Explicit training PGN directories
are listed separately. Evaluation/holdout/gate/external paths and PGNs marked `TrainingData=false`
or `EvaluationKind` are excluded. Human GUI PGNs are recovered only for game IDs already present in
the confirmed human dataset. Previously imported CSV games are recovered by their stored
`source_row`; no additional CSV games are imported.

Every nonterminal pre-move position in recovered games is included for both colors. Standalone
tactical/evaluation rows are also relabeled. Legacy binary tensors can reconstruct pieces, turn,
castling rights and en passant with python-chess and an encoding round-trip check. Without a PGN
they cannot recover missing moves or history; these sources contribute only their stored positions.

The pilot analyzes 128 positions balanced between White and Black using the full run's fixed
50,000-node, five-PV, single-thread settings. `relabel/pilot.json` records teacher provenance,
inspectable samples, elapsed time, and approximate serial analysis hours. Pilot labels are reused.
After interrupted/cached pilot analysis, this estimate can be optimistic. Source hashing, CSV
scanning, finalization and training time are additional. Neither stage trains the neural model.

## Relabel every position and finalize

```powershell
$cfg = "data/experiments/stockfish_rebuild_eval_v2_20261011_01/configs"
.\.venv\Scripts\python.exe -m chess_ai stockfish-rebuild --config "$cfg/relabel.yaml" --stage labels
if ($LASTEXITCODE -ne 0) { throw 'Labels stopped; rerun the same command to continue' }
.\.venv\Scripts\python.exe -m chess_ai stockfish-rebuild --config "$cfg/relabel.yaml" --stage finalize
if ($LASTEXITCODE -ne 0) { throw 'Finalization failed; inspect the error' }
```

Labels are committed per position. Ctrl+C releases the lock and retains completed labels; rerun
`labels` to analyze missing positions. Add `--max-positions 20000` for a bounded invocation.
Finalization requires zero missing labels and reuses completed compatible shards after interruption.
Stages check frozen source digests, teacher settings, and binary hash. After a hard kill, inspect
the PID in the reported lock and remove that lock only after verifying its process has exited.

Inventory has a separate completion boundary. If it stops before `inventory.json` exists, reserve
a new ID; an incomplete catalog cannot be accepted as a complete inventory. Source artifacts are
never overwritten. The complete corpus can require much more disk and RAM than the old replay:
finalization uses 2,048-position shards, and training memory-maps their tensors, but example metadata
and legal masks still live in RAM.

## Target semantics and split

Policy labels are soft distributions over Stockfish's MultiPV root moves (100-cp temperature),
not the historical player's move. Values are `tanh(score_cp / 600)` from the player-to-move
perspective. Signed mate scores use the existing explicit conversion and saturate near +/-1.
Raw centipawn/mate scores, PVs and best moves remain inspectable. This scalar is a scaled evaluation,
not a calibrated win probability or exact mate-distance prediction.

Exact duplicate network inputs receive one label. Analysis uses standalone FENs with counters
reset to `0 1`, because clocks/history are absent from the current 18-plane input. This is an
explicit current-model limitation. Games and known opening-pair groups use a deterministic hash
split (seed 21102036, 10% validation). Validation retains cross-split duplicate positions while
their training occurrences are removed. Identical inputs cannot cross the split, and every
retained example owned by a game/group stays on one side. Training weights unique positions
uniformly; previous winner/opening weights and teacher underexposure are not carried forward.

`relabel/dataset.json` uses `self-improving-chess-ai.stockfish-evaluation-shards` version 1 and
references SHA-256-bound supervised tensor shards of version 2. Version 2 requires a
`value_target_spec` and compatible per-row target-kind tags. New checkpoints use version 2 with
the same specification. Version-1 artifacts remain loadable for legacy work; mismatched dataset,
initialization and resume semantics fail clearly. Resume also binds the dataset manifest digest.
The 18 planes and 4,208 policy actions are unchanged.
Legacy `compose-datasets` refuses version-2 inputs; evaluation rows cannot be saved or loaded as
version 1. Use this rebuild's finalized manifest directly rather than mixing old value labels.

## Train from scratch

```powershell
$cfg = "data/experiments/stockfish_rebuild_eval_v2_20261011_01/configs"
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" --device cuda
```

There is deliberately no `--init-checkpoint`. Architecture matches epoch 120 for the existing
selector. Initial LR is 0.001 with cosine decay, batch size 128, policy/value loss weights 1/1,
legal masks and up to 100 epochs. Validation-total-loss patience is 15 epochs (minimum improvement
0.0001). Retained checkpoints include epochs 50/100 if reached, best and last. JSONL metrics report
each source. These are initial experimental settings, not a guarantee of improved gameplay.

Resume after a completed epoch with the same config:

```powershell
$cfg = "data/experiments/stockfish_rebuild_eval_v2_20261011_01/configs"
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" --device cuda `
  --resume checkpoints/stockfish_rebuild_eval_v2_20261011_01/last.pt
```

Fresh training refuses existing checkpoints/metrics and seeds initialization before construction.
If no complete epoch was saved, reserve fresh training output paths rather than claiming a resume.

Inspect the evaluation head (optionally add `--fen "YOUR_FEN"`):

```powershell
.\.venv\Scripts\python.exe -m chess_ai evaluate-position `
  --checkpoint checkpoints/stockfish_rebuild_eval_v2_20261011_01/best.pt --device cuda
```

Output includes side-to-move and White-perspective values, approximate inverse-scaled centipawns,
the target specification and legal policy preferences. A saturated scalar does not identify mate.

## Compare against epoch 120

```powershell
$cfg = "data/experiments/stockfish_rebuild_eval_v2_20261011_01/configs"
foreach ($mode in @("greedy", "puct", "book")) {
  .\.venv\Scripts\python.exe -m chess_ai gameplay-select --config "$cfg/select_$mode.yaml" --device cuda
  if ($LASTEXITCODE -ne 0) { throw "Selection failed: $mode" }
}
```

Comparisons use the same eight starts, both colors, D1/D2, and a one-point gate with depth and
standard-start regression checks. Greedy uses policy alone. PUCT uses both heads with 32 traversals
per move. Book mode uses that same budget plus the existing compiled opening book for both models,
recording its hash. The wrapper resets per game and hands control to search permanently after the
first unknown position. Book-assisted strength is separate from model-only strength. PUCT is
sequential and may take substantially longer than greedy evaluation.

Fresh confirmation audits run only for modes whose selector found an eligible candidate:

```powershell
foreach ($mode in @("greedy", "puct", "book")) {
  .\.venv\Scripts\python.exe scripts/audit_stockfish_rebuild.py `
    --experiment data/experiments/stockfish_rebuild_eval_v2_20261011_01/experiment.json `
    --mode $mode --device cuda
  if ($LASTEXITCODE -ne 0) { throw "Audit failed: $mode" }
}
```

Eligible modes use fresh seeds 23102036/24102036, 16 openings, both colors and D1/D2. Training
positions, selection starts and the first seed's varied starts are excluded from later fresh
starts. D1/D2 share each seed's suite; standard start is a repeated regression anchor. PGNs remain
evaluation-only, and no checkpoint is promoted. Completed reports can be reused; partially written
PGNs require a fresh `--audit-id retry1` suffix before restarting. Improvement must be assessed through gameplay,
not lower loss or opening-book gains alone.

## Local play with both heads and opening guidance

On 2026-10-15, the user authorized promotion of epoch 60 (`best.pt`, SHA-256
`7b2972f1aa5a1a9caa73db7c1c2f60a48061820b66cca5cdc452e99ba0cdc37a`). All 12 fresh D1/D2
audits passed; direct matches with seeds 25102036/26102036 scored 48 wins, 16 draws, and zero
losses against epoch 120 with PUCT 32 for both models. The repeated standard-start pair drew in
both seeds. These are local-opponent results, not a general chess rating.

The frozen champion is
`checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt`.
`checkpoints/champion.json` selects it for GUI launches with PUCT 32, `c_puct=1.5`, and no book.
The hash-bound evidence and rollback path are recorded in
`data/experiments/stockfish_rebuild_eval_v2_20261011_01/promotion.json`.

```powershell
.\.venv\Scripts\python.exe -m chess_ai gui --device cuda
```

Epoch 120 is preserved. To play it explicitly with the same search budget:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gui `
  --checkpoint checkpoints/joint_policy_value_v3/epoch_0120.pt --device cuda `
  --search-simulations 32
```

The following command explicitly enables the optional book; it differs from promoted default play.

```powershell
.\.venv\Scripts\python.exe -m chess_ai gui `
  --checkpoint checkpoints/stockfish_rebuild_eval_v2_20261011_01/best.pt --device cuda `
  --search-simulations 32 --opening-book data/opening_books/all_chess_openings_v1.json
```

This opens the repository's own local workbench. `best.pt` is validation-selected; use an audited
candidate for later play. For another experiment, reserve a different ID with
`python scripts/new_stockfish_rebuild.py --run-id YOUR_NEW_ID`. Preparation trains nothing. Reuse
the current reserved configs instead of preparing this ID again.

Verification on 2026-10-11: Ruff formatting/checks and mypy passed; full pytest with the pinned local
UCI executable passed **281 tests with one CUDA-availability skip**. Coverage includes real teacher
relabeling, interruption/resume, CSV recovery boundaries, evaluation exclusion, explicit splits,
checkpoint semantics, shared search/book settings and book reset per game. All five prepared configs
were validated and the champion/book hashes were unchanged. Full corpus analysis and training were
not started; the user's commands above start those stages.
