# Stockfish teacher v1: ordered PowerShell runbook

For the promoted evaluation-head model, use [the version-2 refinement runbook](stockfish_refinement.md).
The version-1 commands below document the earlier sampled/outcome-label collection.

Run from `C:\Users\ammar\OneDrive\Desktop\chess-AI`. No long collection, training, selection, or
audit was run during implementation. A six-game/eight-ply smoke collection was allowed and passed.

Local inspection on 2026-10-05 found the opening dataset/book, v4 composition, all 100 training
epochs, and v4 gameplay-selection report. All 21 v4 candidates were rejected by that selector;
the older epoch-2 gate remained selected. This does not establish a stronger replacement for epoch
120. The existing epoch-120 PUCT audit improved varied-opening points but failed its standard-start
gate. Preserve both source checkpoints and all completed artifacts.

The templates are `configs/stockfish_teacher_v1{,_smoke,_compose,_train,_select}.yaml`. The helper
below copies them to unique run configs; rerunning collection with those same generated configs
means resume, while a new run ID means a separate experiment. Keep the printed run ID/config path.

## 1. Update/build or configure Stockfish

This machine already had MSYS2's GNU compiler and make. The tested build uses two CPU jobs and a
generic x86-64 target with embedded NNUE. Updating source may select a newer upstream commit;
never update an executable used by an unfinished collection. Builds go into commit-named directories.
The script refuses dirty/wrong-origin/wrong-branch checkouts and uses fast-forward only. It deepens
history when necessary rather than resetting the checkout. It refuses a different binary digest
in an existing immutable build directory.

```powershell
.\scripts\prepare_stockfish.ps1 -Build
if ($LASTEXITCODE -ne 0) { throw 'Stockfish preparation failed' }
$sfCommit = (git -C references/github/stockfish rev-parse HEAD).Trim()
$stockfish = "references/github/stockfish-build/$sfCommit/stockfish.exe"
$runId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
.\.venv\Scripts\python.exe scripts/new_stockfish_run.py `
  --run-id $runId --executable $stockfish --source-commit $sfCommit
if ($LASTEXITCODE -ne 0) { throw 'Run configuration reservation failed' }
$name = "stockfish_teacher_v1_$runId"
$runRoot = "data/curricula/$name"
$cfg = "$runRoot/configs"
$baseline = 'checkpoints/joint_policy_value_v3/epoch_0120.pt'
```

If MSYS2/compiler are absent, install them explicitly following the
[official compilation guide](https://official-stockfish.github.io/docs/stockfish-wiki/Compiling-from-source.html)
before the first command. The script does not install software. With default `C:\msys64`:

```powershell
winget install MSYS2.MSYS2
$env:MSYSTEM = 'MINGW64'
& 'C:\msys64\usr\bin\bash.exe' -lc 'pacman -S --needed --noconfirm make mingw-w64-x86_64-gcc'
```

To use an existing locally built binary, set `$stockfish` to its path and `$sfCommit` to its actual
source SHA, then run the reservation commands. For an official downloaded release, use its exact
release source commit, never the unrelated current master SHA. Engine version strings alone cannot
prove binary/source correspondence. The build helper writes and the teacher checks a
`stockfish.provenance.json` sidecar when present. Manual binaries without a sidecar rely on your
explicit source-commit declaration. The executable hash is always recorded independently.

## 2. Verify the executable

```powershell
.\.venv\Scripts\python.exe -m chess_ai stockfish-check `
  --executable $stockfish --source-commit $sfCommit
if ($LASTEXITCODE -ne 0) { throw 'UCI verification failed' }
```

This starts one process, uses 1,000 nodes on the initial position, prints engine/version/hash/settings
and three legal PVs, then closes the process. Stockfish is CPU-based. Its own NNUE is unrelated to
our CUDA model.

## 3. Tiny smoke collection

```powershell
.\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/smoke.yaml" --stage all
if ($LASTEXITCODE -ne 0) { throw 'Smoke collection failed' }
```

Six eight-ply games produce at most 12 labels. Smoke runs use CPU for both agents and Stockfish and
never train. Results under `"$runRoot/smoke/collection"` are isolated from the full experiment.
Rerunning a completed smoke verifies final artifact hashes without adding games.

## 4. Full games, then independent labels

```powershell
.\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/collect.yaml" --stage games
if ($LASTEXITCODE -ne 0) { throw 'Games incomplete; resume the same command' }
.\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/collect.yaml" --stage labels
if ($LASTEXITCODE -ne 0) { throw 'Labels incomplete; resume the same command' }
```

The default is 100 openings × three matchups × both colors = **600 games**. Ten opening sets are
normal-start anchors; the other 90 use seeded 2–5 full moves. D1/D2 use seeded equal-score tie breaks;
the frozen epoch-120 policy is greedy. Stockfish never chooses gameplay moves. CUDA serves neural
inference during games; minimax and all Stockfish labeling use CPU. No optimizer runs in either stage.

Labels use at most 24 positions/game (14,400 total), deduplicate exact network inputs within each
game, and reserve up to 25% of the sample for positions in check or where the played move is a
capture/promotion. The remaining positions are uniformly sampled; naturally tactical games may
still exceed 25% tactical positions. The reservation caps oversampling, not the game's natural mix.
Every sampled move retains actual-versus-teacher disagreement, top PVs, cp/mate scores, normalized
evaluations, entropy, and final outcome provenance. This is a bounded diversity sample rather than
training on every teacher move. Neural-policy uncertainty sampling is a later ablation.

Defaults are 20,000 fixed nodes, MultiPV 5, Threads 1, Hash 64 MB, and a 30-second failure deadline.
Use exactly one `nodes` or `depth`; fixed depth requires `nodes: null`. Clear Hash and a fresh UCI
game precede each independent FEN. The deepest **complete exact** MultiPV iteration supplies the
labels if the node budget stops the next iteration early. Each candidate is one contiguous ordered
rank 1–N snapshot, with distinct root moves, exact scores, and the same depth for every rank.
Partial updates can reorder moves and reuse earlier-depth scores; those rows are never merged with
an earlier snapshot. A budget too small to produce a complete exact snapshot
iteration fails clearly. Raising budgets/settings requires a **new run**, not an edited resume.

Cache keys bind exact FEN (including clocks), binary SHA, declared source commit, reported identity,
complete analysis configuration, UCI defaults/settings, and external default-NNUE hashes when used.
The build helper embeds NNUE into the executable, so its network is covered by the binary digest.
Analysis has no repetition history; archived full games retain their moves and final outcomes.
Reproducibility is strongest with one thread on the same CPU/software. Multiple threads are supported
but are not promised bit-for-bit deterministic.

## 5. Resume after Ctrl+C

Rerun whichever stage stopped, with its original generated config:

The same procedure recovers from the previous collector's `Engine supplied duplicate MultiPV root
moves` error. The snapshot-assembly fix preserves the existing version-1 analysis convention and
signatures. Keep the original run/config/cache: committed games remain valid, and the failed game's
uncommitted label transaction is retried automatically. No node-budget or engine change is needed.

```powershell
.\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/collect.yaml" --stage games
.\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/collect.yaml" --stage labels
```

Committed complete PGNs/shards are verified and skipped; only the current journaled transaction is
discarded and retried. Changing the checkpoint, teacher, settings, seed, or output path fails.
Ctrl+C closes the teacher and releases the writer lock. A hard process kill can leave
`"$runRoot/.collection.lock"`: read its JSON PID, verify that process has exited, then explicitly
remove that single lock with `Remove-Item -LiteralPath`. Never remove a live writer's lock or edit
state to bypass provenance checks. If reopening PowerShell, restore `$runId`, `$name`, `$runRoot`,
and `$cfg` to the saved run rather than generating new IDs.

## 6. Finalize

```powershell
.\.venv\Scripts\python.exe -m chess_ai stockfish-batch --config "$cfg/collect.yaml" --stage finalize
if ($LASTEXITCODE -ne 0) { throw 'Finalization failed' }
```

This requires all games and label shards, writes `collection/dataset.pt` and `collection/manifest.json`,
records digests, and permanently closes collection. There is intentionally no early partial finalization
in v1, because incomplete six-game opening sets would alter the planned mixture. Finalization writes
no checkpoint. State/manifest version 1 is separate from unchanged supervised dataset/checkpoint
formats. Unknown versions fail rather than migrating artifacts silently.

Values remain the completed archived game's result from the player to move: win +1, draw 0, loss −1.
Move-limit safeguard draws are explicitly identified by termination and are not official chess draws.
`stockfish_evaluation` is separate metadata; the trainer never substitutes it for the outcome. Loss
weights remain win 2, draw 1.5, loss 1 for the actor currently to move. The soft policy normalizes only
the requested legal MultiPV roots; other legal roots receive zero, as do all illegal actions.

## 7. Compose historical replay and openings

The existing v3 replay and opening dataset were inspected and are already present. This command
does not import external data or evaluation PGNs:

```powershell
.\.venv\Scripts\python.exe -m chess_ai compose-datasets --config "$cfg/compose.yaml"
if ($LASTEXITCODE -ne 0) { throw 'Composition failed' }
```

Sources are the new Stockfish labels, full v3 historical replay, and the named-opening dataset. The
output is `"data/datasets/${name}_replay.pt"`. Stockfish outcomes, existing joint-training weights,
and opening-frequency weights are carried explicitly. Source files remain immutable. The old/new
source ratio is determined by retained game counts and those weights, not a promised 50/50 quota.
This v1 rehearsal is deliberately conservative; compare source sampling counts before increasing
teacher weight. Whole source groups remain intact. Fixed normal-start anchors overlap by design;
use excluded fresh varied openings as the broader generalization test.

## 8. Train the fresh candidate on CUDA

```powershell
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" `
  --init-checkpoint $baseline --device cuda
```

The new run uses learning rate `5e-7`, policy/value loss weights 1/0.1, legal training masks,
game-balanced sampling, opening/source-grouped validation, at most 60 epochs, patience 20, and
every fifth epoch retained. Initialization also saves `baseline.pt`; gameplay selection now includes
that file explicitly. Epoch 120 is never written. `best.pt` is a validation candidate, not a promotion.

After Ctrl+C, resume the same candidate's last completely saved epoch:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train --config "$cfg/train.yaml" `
  --resume "checkpoints/${name}_candidate/last.pt" --device cuda
```

Do not rerun `--init-checkpoint` into existing outputs. If stopped before any complete epoch was saved,
keep the partial experiment for inspection and reserve new candidate/config/metric paths. Optimization
inside an interrupted epoch is discarded; the previous atomic checkpoint remains valid.

## 9. Select promising epochs by gameplay

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select --config "$cfg/select.yaml" --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Gameplay selection failed' }
$selectionPath = "data/games/evaluation/${name}_selection/report.json"
$selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
$candidate = $selection.selected_checkpoint
```

The 16-opening greedy suite compares every retained candidate and initialization baseline against
epoch 120 at D1 and D2, both colors, with a standard-position gate. It excludes new collection starts
and the two historical teacher manifests. A candidate needs aggregate improvement and no per-depth
or standard-start regression. If `selected_is_candidate` is false, keep epoch 120; do not audit a
checkpoint against itself or force a lower-loss candidate into the GUI. If selection is interrupted,
preserve the partial PGNs and choose fresh selection report/PGN paths before restarting.

## 10. Fresh paired and direct-baseline audits

Run only when the selector returned a distinct candidate. This is a **long** evaluation matrix:
three seeds × D1/D2 × greedy/32-simulation PUCT. Each call plays 64 minimax games plus 32 direct
candidate-versus-baseline games. Together, that is 1,152 games. Every call gets unique artifact paths.
For an initial diagnostic, run one call first; the whole matrix is the stronger confirmation step.

```powershell
if (-not $selection.selected_is_candidate) { throw 'No improved candidate selected; retain epoch 120' }
$auditTag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
foreach ($freshSeed in @(11060205, 12060205, 13060205)) {
  foreach ($depth in @(1, 2)) {
    foreach ($simulations in @(0, 32)) {
      $auditSeed = $freshSeed + $depth * 1000 + $simulations
      $auditDir = "data/games/evaluation/${name}_${auditTag}_seed${auditSeed}_d${depth}_s${simulations}"
      .\.venv\Scripts\python.exe -m chess_ai paired-audit `
        --candidate $candidate --champion $baseline `
        --openings 16 --audit-seed $auditSeed --opponent-depth $depth `
        --opening-min-full-moves 2 --opening-max-full-moves 5 `
        --search-simulations $simulations --c-puct 1.5 --device cuda `
        --direct-baseline --exclude-manifest "$runRoot/collection/manifest.json" `
        --exclude-additional-manifest data/curricula/gen1_vs_d1_d2_teacher_500/manifest.json `
        --exclude-additional-manifest data/curricula/minimax_d1_d2_d3_3000/manifest.json `
        --pgn-dir $auditDir
      if ($LASTEXITCODE -ne 0) { throw "Audit interrupted/failed: $auditDir" }
    }
  }
}
```

Reports contain W/D/L/points, per-color and standard/varied slices, paired opening-bootstrap
uncertainty, regression flags, and separate direct-baseline results. Read D1/D2 and each inference
style separately. A normal-start or per-color loss must be reported even if total points increase.
Small samples are flagged; the interval is descriptive and does not correct selection bias. The
single-suite `supports_promotion` field is only a provisional gate. Require consistent evidence over
fresh seeds and direct matches, then make any promotion manually. Nothing auto-promotes here.
PUCT/direct evaluation is substantially slower than greedy inference. Identical deterministic
standard starts across seeds are repeated checks, not extra independent observations. A separate
limited-strength Stockfish gameplay benchmark is deferred until these suites pass; never benchmark
on cached teacher/training positions. Audit PGNs always say `TrainingData=false`.

## 11. Play in the local GUI

After reviewing the evidence, play the selected candidate (or set `$candidate = $baseline` to keep
epoch 120):

```powershell
.\.venv\Scripts\python.exe -m chess_ai gui --checkpoint $candidate `
  --opening-book data/opening_books/all_chess_openings_v1.json `
  --search-simulations 32 --c-puct 1.5 --device cuda
```

The existing opening book remains active until the opponent leaves known theory. Evaluation above
omits it for both contestants so checkpoint comparisons measure the learned model/search itself.
No external GUI, chess site, or online application is inspected or controlled.

## Cost and implementation validation

| Stage | CPU | GPU | Disk / expected duration |
| --- | --- | --- | --- |
| Build/check | Compiler/Stockfish | None | About 103 MB embedded executable here; build took about a minute |
| Smoke | Minimax + 1k-node labels | None | Six PGNs, 12 examples; roughly 3 seconds after startup here |
| Full games | D1/D2 search | Frozen neural inference | 600 PGNs; could take hours; time a few completed games first |
| Labels | Stockfish 1 thread | None | ≤14,400 analyses at 20k nodes; CPU/hardware dependent, minutes to hours |
| Finalize/compose | Serialization + RAM | None | New dense labels ≤310 MB plus similar shards/cache; replay copy is roughly 2 GB |
| Train | Loading/legal masks | CUDA optimization | Roughly 100 MB retained model checkpoints plus metrics; measure epoch time |
| Select/audit | Minimax + match loop | Neural/PUCT leaves | Many PGNs; hours are plausible; sequential PUCT can underuse CUDA |

No elapsed-time estimate proves label quality or model improvement. Dense replay composition holds
examples in RAM and may require several GB; retain disk headroom for source, shards, and composed
copies. Teacher cp scores are a heuristic scale and top-MultiPV distributions omit weaker legal
alternatives. Outcome labels can disagree with teacher evaluations because later actors make mistakes.

Implementation commands actually run: `git pull --ff-only origin main`; safe Stockfish fetch and
fast-forward (the existing shallow checkout needed an unshallow recovery); `prepare_stockfish.ps1
-Build`; UCI verification; unit tests; Ruff format/check; mypy; full pytest; and the real six-game
smoke. The first smoke exposed partial bound scores and was retained as an incomplete debugging
artifact; the succeeding smoke is `data/curricula/stockfish_teacher_v1_implementation_smoke2_20261005/`.
No full generation, composition of the large replay, training, gameplay selection, or long audits
were executed. No project commit or push was made.

Final verification on 2026-10-05: Ruff formatting/checks passed (95 Python files), mypy passed
(53 source files), and the complete pytest suite passed **245 tests with one skip** in 15.81 seconds.
The local Stockfish integration smoke was enabled for that run. The skipped test covers unavailable
CUDA behavior and skips because CUDA is available on this machine. Without the optional engine
environment variables, ordinary tests require no Stockfish and its integration smoke skips clearly.

MultiPV snapshot fix verification on 2026-10-07: Ruff format/check and mypy passed; the full suite
passed **251 tests with one CUDA-availability skip**, including the real Stockfish regression at the
failed collection's FEN. A read-only check validated that collection's unchanged resume binding and
all 600 PGN digests, then successfully analyzed its first game's 24 sampled positions with the
original 20,000-node/MultiPV-5 settings. No collection state, label shard, or cache entry was written
by this check. Resume the labels command to process the remaining collection.

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
$env:CHESS_AI_STOCKFISH = (Resolve-Path $stockfish).Path
$env:CHESS_AI_STOCKFISH_COMMIT = $sfCommit
.\.venv\Scripts\python.exe -m pytest
```

Stockfish build provenance: commit `49ea5ded38315cff8e67f4a677a9e7811612fbf6`, reported name
`Stockfish dev-20261005-nogit`, executable SHA-256
`fa289b7c744dfae94fcb98ca69ee4686242a2e0a7d66cd1e5dd17be46455e5d1`. `nogit` is the compiler's
version-string output; the separate build sidecar records the verified source checkout commit.

The engine remains separately licensed [GPL-3.0](https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt).
Keep the license and exact source pointer with any distributed binary. No engine or NNUE is
committed. UCI behavior follows [python-chess's engine API](https://python-chess.readthedocs.io/en/latest/engine.html),
with our independent deadline and complete-iteration checks added around it.
