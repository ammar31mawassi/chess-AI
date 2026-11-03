# Train on Stockfish targets from personal Chess.com games

The explicitly authorized experiment is `chesscom_xxkiller31_20261018_01`. Its local source is
`downloaded_datasets/chesscom/xxkiller31/`: 23 monthly PGNs, 1,060 archived game records, and 66,510
recorded pre-move positions across both colors before deduplication. Empty game records have no
position targets; they are counted separately. No website, external chess GUI, or downloader is
used by this workflow.

The completed inventory contains 57,983 unique network inputs: 28,997 White-to-move and 28,986
Black-to-move positions. There are 51,186 training and 6,797 validation rows. One empty game and
one duplicate movetext are excluded from position ingestion, leaving 1,058 distinct playable games.

The historical move sequence supplies positions only. Local Stockfish supplies both targets:
MultiPV move preferences/best move and `tanh(score_cp/600)` evaluation from the player-to-move
perspective. Human moves, final results, and any evaluations already embedded in the PGN are not
used as training targets. Losing games are valid position sources. No new model head is needed:
the champion already has a policy head and a compatible evaluation head, both used by PUCT.

## Explicit validation and snapshots

Preparation verifies every game's parsing, player identity, completed result, standard-chess
starting position, and legal mainline through python-chess. Evaluation-only markers and variants
are rejected. Duplicate game movetexts and empty game records are counted. All original files stay
unchanged; exact-byte copies under `source_games/` freeze this experiment against later downloads.
`personal_import.json` records source paths, snapshot names, SHA-256 digests, counts, and the
explicit training-import boundary. It creates no human-move dataset.

The prepared configs already exist. To create a different experiment later:

```powershell
.\.venv\Scripts\python.exe scripts/new_personal_stockfish_training.py `
  --source-dir downloaded_datasets/chesscom/xxkiller31 --username xxkiller31 `
  --run-id YOUR_NEW_ID --confirm-training-import
```

Preparation refuses an existing run ID. An invalid PGN prevents snapshot creation. The current
user request explicitly authorized the local PGN import; the flag makes that scope visible in the
reproducible command. Existing benchmark/evaluation games are not imported.

## Analyze every unique position

This extends `stockfish-rebuild` to accept PGN-only input: `datasets: []` and an explicit nonempty
`pgn_roots` list. Existing dataset-based configurations retain their original binding and behavior.
The actual run's inventory was prepared separately from labeling; rerunning inventory verifies it.

```powershell
$cfg = 'data/experiments/chesscom_xxkiller31_20261018_01/configs'
foreach ($stage in @('inventory', 'pilot', 'labels', 'finalize')) {
  .\.venv\Scripts\python.exe -m chess_ai stockfish-rebuild `
    --config "$cfg/relabel.yaml" --stage $stage
  if ($LASTEXITCODE -ne 0) { throw "Personal Stockfish stage failed: $stage" }
}
```

Every recorded nonterminal pre-move position is included, for both players, with exact duplicate
network inputs analyzed once. The terminal position has no next-move policy target. Stockfish uses
the existing pinned binary, 50,000 fixed nodes, five MultiPV lines, and a compatible cache.
Analysis runs on CPU; CUDA is used for subsequent neural training/inference. The pilot analyzes
128 positions and reuses its labels. Cached pilot timings can underestimate the remaining time.

Raw signed cp/mate scores, principal variations, best moves, and normalized targets remain in the
label audit metadata. The existing encoder lacks history/clocks, so labels use independent FENs
with counters reset to `0 1`, consistently with the champion and original replay.

SQLite commits each position separately. After Ctrl+C, repeat `labels`; after power loss, verify
the recorded process has exited before removing only the stale `.relabel.lock`. Never delete the
catalog to recover progress. Source snapshot changes fail resume validation.

Games are assigned wholly to deterministic training/validation groups. Validation owns an input
shared between otherwise separate groups. Finalization writes version-2 shards plus `dataset.json`
with hashes, explicit splits, and the existing `stockfish_cp_tanh_v1` semantics.

## Replay and fine-tuning

```powershell
$cfg = 'data/experiments/chesscom_xxkiller31_20261018_01/configs'
.\.venv\Scripts\python.exe scripts/compose_stockfish_refinement.py --config "$cfg/compose.yaml"
if ($LASTEXITCODE -ne 0) { throw 'Replay composition failed' }

.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" --device cuda `
  --init-checkpoint checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt
```

The existing replay composer accepts the finalized personal-data manifest. It protects original
validation inputs from training and original training inputs from fresh validation, removes
cross-source duplicates, and draws equal fresh/historical position counts in each split. Only
compatible Stockfish evaluation labels enter the 50/50 mixture; no legacy outcome targets enter it.

Start from the registered epoch-60 champion, with a fresh optimizer, learning rate `1e-5`, batch
128, policy/value weights 1/1, at most 20 epochs, and patience 5. The unchanged baseline remains
eligible. This is a separate candidate; the rejected refinement's epoch 14 is not an initialization
source. After an interrupted completed epoch, use the same config with:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" --device cuda `
  --resume checkpoints/chesscom_xxkiller31_20261018_01/last.pt
```

Do not compose again if `replay.pt` already exists. Keep it and proceed to training/resume.

## Check gameplay before promotion

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select --config "$cfg/select_puct.yaml" --device cuda
```

The selector compares retained candidates to the frozen champion using PUCT 32 without a book,
fresh openings, D1/D2, and both colors. It excludes all imported training positions, original
training starts, and recorded historical selection/confirmation starts. Selection stays provisional;
fresh paired/direct audits are required before a later manual promotion. The current champion,
old checkpoints, downloaded PGNs, and existing datasets are not overwritten.
