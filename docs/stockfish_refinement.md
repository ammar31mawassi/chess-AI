# Champion refinement with all-move Stockfish supervision

The prepared experiment is `stockfish_refinement_v2_20261015_01`. It freezes the promoted
epoch-60 champion and plays 204 games before independent Stockfish labeling or optimization.
The champion's checkpoint and the runtime opening book are bound by content hashes in collection
state. No collection, composition, training, or selection command changes the registered champion.

## Games and opening book

- 50 seeded, collision-checked varied openings, each played against D1 and D2 as White and Black:
  200 games with PUCT 32, `c_puct=1.5`, and no book.
- Four additional games from the ordinary starting position: White/Black against each depth.
  The champion uses the opening book in these four games, then PUCT after leaving theory.
  Each game gets a fresh book wrapper. D1/D2 remain minimax opponents and do not use the book.
- Orthodox chess legality/outcomes come from python-chess. All games have the existing 200-ply
  cap; reaching it is recorded as a move-limit draw rather than a completed natural result.
- Every color/depth game from the same opening shares one split group. The four standard-start
  games share a separate group. Historical evaluation starts are excluded from varied starts;
  the ordinary start remains an explicit repeated anchor.

Run from the repository root:

```powershell
$cfg = 'data/experiments/stockfish_refinement_v2_20261015_01/configs'
foreach ($stage in @('games', 'labels', 'finalize')) {
  .\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/collect.yaml" --stage $stage
  if ($LASTEXITCODE -ne 0) { throw "Collection failed: $stage" }
}
```

`games` uses CUDA for neural inference and CPU for minimax. Stockfish is opened only as an offline
UCI teacher; it performs no searches during gameplay. `labels` uses the pinned local CPU Stockfish
at 50,000 nodes and MultiPV 5. The original rebuild's compatible cache is reused; cache signatures
bind the FEN, engine identity, binary, and analysis settings. Already analyzed positions can reuse
their analysis. Every recorded ply, including seeded opening moves and book moves for both sides,
receives an audit row. The final terminal position has no next-move target and is not analyzed.

## Labels and versioning

Version 2 explicitly uses all recorded plies; legacy `positions_per_game` and `tactical_share`
sampling fields do not subsample this mode. Per-game version-2 shards preserve every move record,
including repeated positions. Each record contains the played move, teacher top moves, signed raw
centipawn/mate scores, normalized evaluation, color, opening group, and result. A played move outside
the MultiPV set receives a separate forced-root search; otherwise its existing root score is reused.
`estimated_cp_loss` compares these bounded searches and is an estimate, particularly for mate scores.
The score guides diagnosis; the policy target remains the teacher distribution over preferred moves.

Value targets are `stockfish_cp_tanh_v1`, `tanh(score_cp/600)`, from the player-to-move perspective,
matching the champion. Completed-game outcomes remain audit metadata. The existing 18-plane encoder
does not represent history or clocks, so standalone FEN clocks reset to `0 1` and repeated network
inputs share a canonical label, as in the original rebuild.

Finalization deduplicates network inputs into `collection/dataset.pt`, retains the original all-move
shards, and writes `collection/manifest.json`. A deterministic opening-group hash fixes train/validation
assignment; validation owns a duplicate input shared by otherwise separate groups. The new tensor
dataset is version 2 with explicit split metadata and a resume signature over the frozen inputs.

Collection state/manifest use `self-improving-chess-ai.stockfish-offline-batch`, version 2, selected
by `collection_version: 2`. Version-1 configs/states keep their exact original signature shape,
six-game cohorts, sampled positions, and outcome targets. Incompatible resume versions or edited
settings, checkpoint, book, exclusion lists, engine, PGNs, or label shards fail clearly.

## Replay and candidate training

Compose only after finalization:

```powershell
$cfg = 'data/experiments/stockfish_refinement_v2_20261015_01/configs'
.\.venv\Scripts\python.exe scripts/compose_stockfish_refinement.py --config "$cfg/compose.yaml"
if ($LASTEXITCODE -ne 0) { throw 'Replay composition failed' }
```

The replay composer verifies the original evaluation manifest and shard hashes, protects **all**
old validation inputs from fresh training, and samples old examples deterministically. Historical
training inputs are excluded from fresh validation to avoid inflating the new holdout score. It keeps
exactly one old example per retained fresh example in each split, giving 50/50 fresh/replay position
coverage without copying the entire original corpus. Cross-source duplicate inputs are excluded;
group/split identities are retained. Sources and split counts are recorded in the output metadata.
No legacy outcome datasets, evaluation PGNs, or new external benchmark games enter this dataset.

Train a separate candidate with a fresh optimizer initialized from the frozen champion:

```powershell
$cfg = 'data/experiments/stockfish_refinement_v2_20261015_01/configs'
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" --device cuda `
  --init-checkpoint checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt
```

The initial learning rate is `1e-5`, maximum epochs 20, patience 5, batch 128, and policy/value
weights 1/1. Sampling is uniform over the bounded 50/50 composition. Legal masks, explicit splits,
per-source validation metrics, and every-epoch retention are enabled. `baseline_eligible: true`
keeps the unchanged initialization eligible; a lower validation loss alone does not promote it.

After interruption during training, resume with the same config:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" --device cuda `
  --resume checkpoints/stockfish_refinement_v2_20261015_01/last.pt
```

Fresh composition/training refuse existing outputs. Keep an already composed replay artifact and
proceed to training instead of composing again. For collection interruption, repeat the same stage:
committed games/shards are verified; a reserved uncommitted artifact is retried, with per-position
teacher caches preserving completed analysis. Ctrl+C releases the writer lock. After power loss,
verify the PID in the sibling `.collection.lock` has exited before removing only that stale lock.

On Windows, sync or antivirus software can briefly lock a destination during atomic replacement
(`WinError 5`, `32`, or `33`). JSON state/cache, PGN, dataset, and checkpoint writes retry the same
complete temporary file up to eight times with bounded backoff (3.15 seconds total waiting).
Persistent errors still stop the command while preserving the previous committed destination.
Rerun the same collection stage; committed work is verified and a reserved incomplete game is retried.
There is no need to delete the collection or run PowerShell as administrator for a transient lock.

## Gameplay selection

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select --config "$cfg/select_puct.yaml" --device cuda
```

Selection uses fresh starts against D1/D2, both colors, with PUCT 32 for candidate and champion.
It excludes the new collection starts, original training starts, and historical evaluation starts.
It is provisional: fresh paired and direct audits still precede a later manual promotion.

Only configs are prepared in this request. To reserve another experiment later, use
`python scripts/new_stockfish_refinement.py --run-id YOUR_NEW_ID`; do not recreate the reserved ID.
