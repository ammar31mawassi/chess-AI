# Training guide (Windows PowerShell)

These commands assume PowerShell is open in the repository root. Phase 1 is a supervised bootstrap;
the tiny run proves the software path, not playing strength.

## 1. Create and activate an environment

Python 3.11 or newer is required. If the Python launcher is installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks the activation script, allow it for only the current process and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`python -m pip` deliberately ties installation to the active interpreter.

## 2. Verify the source and environment

```powershell
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m chess_ai doctor
```

Doctor prints package versions, CUDA availability, selected device, artifact directories, and the
result of a sample `(1, 18, 8, 8)` forward pass. CPU is fully supported.

## 3. Generate a tiny supervised dataset

```powershell
python -m chess_ai generate-data --config configs/dev.yaml
```

The development configuration plays only a few shallow games and writes
`data/datasets/dev_examples.pt`. Output reports games, examples, and the path. If interrupted after a
periodic save, rerun with the same configuration and `--resume`; generation continues from the
validated partial dataset where possible.

### Generate 500 varied depth-2 teacher games

Deterministic minimax against deterministic minimax from the normal starting position repeats one
game, regardless of the requested game count. The dedicated configuration avoids that leakage by
playing 8-12 reproducibly random opening plies without recording them, then letting minimax depth 2
control both colors. Only the minimax continuation becomes policy data, and duplicate neural input
positions are discarded globally:

```powershell
python -m chess_ai generate-data --config configs/minimax_d2_diverse_500.yaml
```

It writes `data/datasets/minimax_d2_diverse_500.pt` and archives every complete teacher game as
`data/games/minimax_d2_diverse_500/game_0001.pgn` through `game_0500.pgn`. Each PGN includes the
unrecorded opening and the minimax continuation, with headers stating how many positions were kept
after deduplication. Dataset progress is saved atomically every five completed games, and the command
prints the retained example count plus `duplicates_discarded`. These paths are deliberately different
from older minimax artifacts.

If generation is interrupted, run the same command again; `resume: true` continues only when all
reproducibility settings match and every PGN for the already committed games is still present. A
missing archive file stops the resume with an actionable error instead of silently leaving the
training record incomplete.

The relevant generation settings are:

- `random_opening_min_plies` and `random_opening_max_plies`: legal random prefix length. These plies
  count toward `max_moves` but never become training labels.
- `deduplicate_positions: true`: retain the first occurrence of each exact 18-plane network input
  and discard later occurrences, including duplicates found after resuming.
- `pgn_dir`: atomically archive the complete game behind every generated training-game ID.

Do not replace one teacher with Random unless random moves are intentionally desired as policy
targets: ordinary generation records every move made after the unrecorded opening.

### Generate depth-2 teacher moves against Random

For a candidate intended to convert mistakes from unpredictable opponents, use the teacher-only
configuration:

```powershell
python -m chess_ai generate-data --config configs/minimax_d2_vs_random_teacher_500.yaml
```

It creates 500 total games. Game 1 is a standard minimax-D2-versus-minimax-D2 anchor that preserves
the known opening/repetition line. In games 2-500, depth-2 minimax plays Random and alternates colors;
only minimax moves become policy labels. Random moves remain in the archived PGN and determine later
positions, but never become imitation targets. Exact duplicate network inputs are discarded.

Artifacts are kept separate from every earlier experiment:

```text
data/datasets/minimax_d2_vs_random_teacher_500.pt
data/games/minimax_d2_vs_random_teacher_500/game_0001.pgn ... game_0500.pgn
```

The new configuration fields are:

- `alternate_agents: true`: swap the configured White and Black agent specifications every game
  after any anchor games.
- `record_agent: minimax`: retain policy examples only on turns controlled by minimax.
- `anchor_minimax_games: 1`: generate one normal-start minimax-versus-minimax game first. Anchor
  games always record both minimax sides and do not use random-opening diversification.

After generation completes, fine-tune a new candidate from the preserved V1 champion:

```powershell
python -m chess_ai train `
  --config configs/minimax_d2_vs_random_teacher_500_train.yaml `
  --init-checkpoint checkpoints/minimax_d2_diverse_500_v1/best.pt `
  --device cuda
```

This writes `checkpoints/minimax_d2_vs_random_teacher_500_v1/` and
`data/metrics/minimax_d2_vs_random_teacher_500_v1.jsonl`. It does not overwrite the source champion.

### Run 100 D1-opponent/D2-teacher correction pairs

The controlled correction loop starts from the preserved diverse V1 champion. For every cycle it
creates one seeded 4-8-ply opening, then plays that same position with the neural student once as
White and once as Black against Minimax D1. Minimax D2 is queried on every student turn; its move,
not the student's move, becomes the policy label.

Run the prepared 100-pair experiment on CUDA:

```powershell
.\.venv\Scripts\python.exe -m chess_ai teacher-cycle `
  --config configs/d1_opponent_d2_teacher_100.yaml `
  --device cuda
```

This is a long local run. If it is interrupted, run exactly the same command again. Atomic state and
completed artifacts are validated before continuing. Do not change the configuration paths, seed,
depths, or training settings while resuming.

The run creates:

```text
data/curricula/d1_opponent_d2_teacher_100/state.json
data/curricula/d1_opponent_d2_teacher_100/corrections/cycle_0001.pt ... cycle_0100.pt
data/games/d1_opponent_d2_teacher_100/cycle_0001_student_white.pgn ...
checkpoints/d1_opponent_d2_teacher_100/last.pt
data/metrics/d1_opponent_d2_teacher_100.jsonl
```

Cycles 1-99 train one low-learning-rate update using 20% cumulative teacher corrections and 80%
reproducibly sampled positions from `minimax_d2_diverse_500.pt`. The policy loss is calculated only
over legal actions, matching neural inference, and the value loss is down-weighted to `0.1`. Cycle
100 is deliberately held out: its two games test the returned `last.pt`, while its corrections are
saved but not used. Therefore the command plays exactly 200 games and returns a tested checkpoint.

A first win is reported as progress. It is not mastery. `ready_for_next_stage` becomes true only if
the student wins both colors for the final five consecutive seeded openings (10 straight wins).
The command never switches to D2, changes the teacher to D3, or promotes a checkpoint automatically;
review the PGNs before creating that next stage.

To add another 100 pairs after the initial run, do not edit or copy checkpoints, state, datasets, or
output paths. Raise the **total** target from 100 to 200 on the command line:

```powershell
.\.venv\Scripts\python.exe -m chess_ai teacher-cycle `
  --config configs/d1_opponent_d2_teacher_100.yaml `
  --cycles 200 `
  --device cuda
```

`--cycles 200` means cycles 1-200 in total, so a completed 100-cycle run adds cycles 101-200: 100
new pairs and 200 new games. The runner first trains once on cycle 100's previously held-out
corrections, then starts cycle 101. Cycle 200 becomes the new held-out final pair. You can use any
higher total such as `--cycles 150` or `--cycles 300`; lowering the target or changing another
reproducibility setting is rejected. Run the same extension command after interruption.

#### Refinement after cycle 400: periodic normal-start pairs

The cycle-400 checkpoint beats Minimax D1 as White from the standard position but loses as Black.
The prepared configuration now sets `normal_start_every_cycles: 5`. When the legacy completed
cycle-400 state is extended, this is accepted as a one-time curriculum refinement: cycles 405, 410,
and every fifth cycle thereafter start from the normal position in both colors. The other four out
of every five pairs retain seeded 4-8-ply openings, preventing the replay from collapsing onto one
deterministic line.

Continue through cycle 500 with:

```powershell
.\.venv\Scripts\python.exe -m chess_ai teacher-cycle `
  --config configs/d1_opponent_d2_teacher_100.yaml `
  --cycles 500 `
  --device cuda
```

This adds cycles 401-500 (200 games). Before cycle 401, the runner trains once on cycle 400's saved
holdout corrections. Cycle 500 is itself a periodic normal-start pair and remains held out, so the
returned checkpoint receives a direct, two-color standard-position test. No prior PGN, correction,
metric, or numbered epoch checkpoint is replaced; `last.pt` advances as the run resumes. After this
migration, keep the periodic-anchor setting unchanged for later extensions.

#### Refinement after cycle 500: retained and reserved anchor rehearsal

The cycle-500 audit found that the periodic tests produced more draws, but not more wins. The cause
was in the curriculum sampling, not a failure to update the network: exact standard-start positions
were globally deduplicated, so later anchor cycles frequently saved no reusable corrections. The
small number of old anchor examples was then sampled from the same pool as roughly 25,000 varied
corrections, making it easy for an update to forget the standard line.

The prepared configuration now adds two safeguards:

- Repeated standard-start corrections are retained across different anchor cycles. Exact duplicates
  are still removed within one two-color pair so a repetition loop cannot dominate a single file.
- `anchor_correction_share: 0.25` reserves one quarter of the correction portion of every update for
  standard-start lessons. Because corrections are 20% of the full batch, the reservation is about
  5% of the complete update. Unused reserved space is filled from the other correction pool.

After every trained normal-start pair, the runner also plays a fresh two-color standard-position
regression pair and saves it beneath:

```text
data/games/d1_opponent_d2_teacher_100/standard_evaluation/
  after_cycle_0500_student_white.pgn
  after_cycle_0500_student_black.pgn
```

Its W/D/L, score, and whether the update regressed relative to the pre-training pair are stored in
that cycle's state history. These two diagnostic games are not training data, do not query D2 for
labels, and are not included in `games_played` or the mastery gate.

Continue from the completed 500-cycle state through cycle 600 with:

```powershell
.\.venv\Scripts\python.exe -m chess_ai teacher-cycle `
  --config configs/d1_opponent_d2_teacher_100.yaml `
  --cycles 600 `
  --device cuda
```

The first extension step trains the saved cycle-500 holdout using the new reserved replay mix, then
saves its immediate standard regression pair before cycle 501. Cycles 505, 510, and every fifth
cycle thereafter use the refined anchors. Cycle 600 remains the new untrained holdout. Existing
numbered checkpoints, PGNs, and corrections are preserved; metrics and state resume append/update
behavior, and `last.pt` advances normally. Rerun the same command after an interruption. A completed
state that already contains periodic anchors is allowed this one-time anchor-replay migration, after
which the new settings are locked into its signature.

The old files contain only 148 retained anchor corrections because the earlier global filter already
discarded the others; the first refined update therefore uses all 148 rather than inventing duplicate
examples. New anchor files retain their repeated lessons, so the reserved quota fills naturally.
Each trained cycle records its exact replay/correction/anchor batch counts under `training_batch` in
`state.json`.

#### Gameplay-gated restart from epoch 399

The cycle-600 audit showed that better imitation-batch loss did not mean better chess: cycles
501-600 scored 6.5%, down from 9.25% in cycles 401-500. The post-anchor checks caught direct
draw-to-loss regressions at cycles 515 and 530, but the old loop only reported them and continued
updating `last.pt`. Epoch 399 remains the strongest observed standard-start checkpoint, so the gated
run starts from that immutable numbered file rather than the training-loss-selected `best.pt`.

The gameplay champion has a separate meaning and location:

```text
checkpoints/d1_gated_from_epoch399/champions/generation_0000.pt
checkpoints/d1_gated_from_epoch399/champions/champion.json
```

Generation zero is a byte-identical snapshot of `epoch_0399.pt`. `champion.json` is the authoritative
gameplay selection record; it does not overwrite or redefine any existing `best.pt`. Initialize only
the champion and paths, without playing or training, with:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gated-refinement `
  --config configs/d1_gated_refinement_from_epoch399.yaml `
  --initialize-only `
  --device cuda
```

Run or resume the prepared refinement:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gated-refinement `
  --config configs/d1_gated_refinement_from_epoch399.yaml `
  --device cuda
```

Each round has four explicit phases:

1. The current frozen champion plays 10 fresh D1 pairs; D2 labels the champion's turns.
2. One conservative candidate starts from the champion with a fresh optimizer and trains once on
   cumulative gated corrections plus the preserved diverse replay dataset.
3. Candidate and champion independently face D1 from the exact same eight starts, once per color:
   the standard position plus seven fixed seeded openings.
4. The candidate is promoted only if it gains at least 0.5 total point and does not score worse than
   the champion on the two standard-start games. Rejected candidates remain archived but never
   become the source of the next round.

Each ten-round segment collects 200 training games and plays 320 evaluation-only gate games. The
configuration now targets 20 total rounds; with rounds 1-10 complete, the next invocation adds
rounds 11-20. Gate PGNs are never training data. Training uses a smaller `0.000002` learning rate
and a fresh optimizer per candidate. Its value loss is disabled because the current NeuralAgent
chooses moves from policy only, while a correction game's final result mostly measures the
student's later mistakes rather than the D2-labelled move. Policy targets remain legality-masked D2
choices.

Artifacts are separated by round:

```text
data/curricula/d1_gated_from_epoch399/state.json
data/curricula/d1_gated_from_epoch399/corrections/round_####/
data/games/d1_gated_from_epoch399/{training,gate}/round_####/
data/metrics/d1_gated_from_epoch399/round_####.jsonl
checkpoints/d1_gated_from_epoch399/candidates/round_####/
```

Run the same command after interruption. `rounds: 20` is a total target, not 20 additional rounds;
only rounds 11-20 are pending. All gate decisions, scores, checkpoint hashes, and promotion reasons
remain in atomic state and the champion manifest.

Because the eight promotion starts are reused, confirm a promoted generation on a separate suite
before extending. This command compares generation zero and the current champion against D1 on the
same 20 starts in both colors: the standard position plus 19 collision-checked random openings that
do not occur in the promotion gate.

```powershell
.\.venv\Scripts\python.exe -m chess_ai gated-audit `
  --config configs/d1_gated_refinement_from_epoch399.yaml `
  --openings 20 `
  --device cuda
```

The audit plays 40 games per checkpoint, saves 80 PGNs and `report.json` below
`data/games/d1_gated_from_epoch399/holdout/`, and never promotes or modifies a checkpoint. Its seed,
opening FENs/moves, W/D/L, standard scores, point delta, and continuation verdict are recorded so the
comparison can be reproduced exactly.

The generation-1 audit used seed `90002045`. Across 40 games per checkpoint, generation zero scored
4.0 points (1W/6D/33L) and generation one scored 4.5 (1W/7D/32L), with both scoring 1.0 on the
standard pair. Three baseline losses became draws while two baseline draws became losses, so the
net improvement is real but narrow rather than a broad dominance claim. The audit had no random
opening overlap with the promotion gate and performed no promotion. This supports the guarded
round-20 continuation, followed by another fresh-seed audit.

### Offline 500-game generation-1/D1 batch with all-ply D2 teaching

This stage deliberately separates collection from optimization. Generation one stays frozen while
250 seeded openings are each played twice against Minimax D1, once per champion color. Every opening
contains two or three **full moves** (four or six plies). Minimax D2 observes and labels every played
position after that prefix, including both the champion's turns and D1's turns, but never chooses an
actual game move.

D2 scores every legal root move with the same full search window. The strongest five scores become
a temperature-softened policy target. This is richer than a one-hot best move, although it is still
not the MCTS visit distribution used by AlphaZero. Collection creates no optimizer, metrics file, or
checkpoint and reports `training_updates: 0`.

Generate exactly 500 games:

```powershell
.\.venv\Scripts\python.exe -m chess_ai teacher-batch `
  --config configs/gen1_vs_d1_d2_teacher_500.yaml `
  --device cuda
```

CUDA runs only the champion's neural inference. D1 and D2 are classical CPU searches, so D2 can
remain the runtime bottleneck. If interrupted, run the identical command again. The runner validates
the frozen champion hash plus every committed trace/PGN digest before continuing.

Artifacts are isolated from the earlier online curricula:

```text
data/curricula/gen1_vs_d1_d2_teacher_500/state.json
data/curricula/gen1_vs_d1_d2_teacher_500/manifest.json
data/curricula/gen1_vs_d1_d2_teacher_500/pairs/pair_0001.pt ... pair_0250.pt
data/games/gen1_vs_d1_d2_teacher_500/game_0001_champion_white.pgn ... game_0500_champion_black.pgn
data/datasets/gen1_vs_d1_d2_teacher_500.pt
```

The pair shards are the resume boundary. The final dataset is assembled only after all 250 pairs
and 500 PGNs are committed. Each example records the actual move, D2's preferred move and score,
the soft top-five policy, actor role, opening-pair ID, champion color/result, and value from the
player-to-move perspective. Complete PGNs include the random opening prefix rather than discarding
it from the archive.

Champion-win games carry `champion_outcome_weight: 2.0`; draws and losses carry `1.0`. Losses are
not down-weighted because they contain important corrections. The trainer samples complete games
with equal expected mass before applying this outcome weight, preventing a long game from dominating
only because it supplied more positions. Both color-swapped games from one opening share an
`opening_pair_id` and therefore remain together in training or validation.

After generation completes, start a new offline candidate from generation one:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/gen1_vs_d1_d2_teacher_500_train.yaml `
  --init-checkpoint checkpoints/d1_gated_from_epoch399/champions/generation_0001.pt `
  --device cuda
```

This creates `checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/` and
`data/metrics/gen1_vs_d1_d2_teacher_500_candidate_v1.jsonl`. It uses three conservative epochs at
learning rate `0.000002`, legality-masked policy loss, no value loss, game-balanced training
sampling, an opening-pair-grouped 80/20 split, and the stored outcome weight. Existing metric names
are the weighted values when weighting is configured; each record also contains explicitly named
`unweighted_*` losses and accuracies. The immutable generation-one source is never overwritten.

Evaluate on a first collision-checked holdout. Opening zero is the standard position; the remaining
19 openings use a new seed and are rejected if their FEN occurs in the training manifest. Candidate
and champion independently play D1 as both colors from every identical start:

```powershell
.\.venv\Scripts\python.exe -m chess_ai paired-audit `
  --candidate checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/best.pt `
  --champion checkpoints/d1_gated_from_epoch399/champions/generation_0001.pt `
  --openings 20 `
  --audit-seed 9300205 `
  --opponent-depth 1 `
  --device cuda `
  --exclude-manifest data/curricula/gen1_vs_d1_d2_teacher_500/manifest.json `
  --pgn-dir data/games/evaluation/gen1_vs_d1_d2_teacher_500_candidate_v1_holdout_a
```

Confirm with a second unseen suite rather than promoting from one narrow result:

```powershell
.\.venv\Scripts\python.exe -m chess_ai paired-audit `
  --candidate checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/best.pt `
  --champion checkpoints/d1_gated_from_epoch399/champions/generation_0001.pt `
  --openings 20 `
  --audit-seed 9400205 `
  --opponent-depth 1 `
  --device cuda `
  --exclude-manifest data/curricula/gen1_vs_d1_d2_teacher_500/manifest.json `
  --pgn-dir data/games/evaluation/gen1_vs_d1_d2_teacher_500_candidate_v1_holdout_b
```

Each audit saves 40 PGNs per checkpoint plus `report.json`; none becomes training data. A report's
`supports_promotion` requires at least a 0.5-point gain and no regression on the standard two-color
pair. Treat both suites together: keep generation one if the apparent gain disappears or standard
play regresses. Promotion remains a separate human decision.

This adapts three practical lessons from the
[AlphaZero paper](https://arxiv.org/abs/1712.01815): learn from every visited state, use a policy
distribution produced by search rather than only one selected move, and keep policy/value targets
separate. It does not claim to implement AlphaZero: D2 is not policy/value-guided MCTS, the current
value head does not guide play, and the published system operated at vastly greater scale. The later
architecture stage remains policy/value-guided MCTS and self-play.

### Mixed 400-game generation-2 batch and gameplay-first epoch selection

This stage starts from the gameplay-approved epoch 2 and creates 50 reproducible positions after
two or three full random moves. Every position is reused by four matchups, each with Agent A once as
White and once as Black:

```text
Minimax D1 vs Minimax D2       100 games
Minimax D2 vs Minimax D2       100 games
Neural generation 2 vs D1      100 games
Neural generation 2 vs D2      100 games
                               ---------
                               400 games
```

The D2-vs-D2 actors use seeded random selection only among moves with the same best minimax score;
the seed remains reproducible. D2 independently scores every legal root move on every continuation
ply for the soft top-five policy target. Actual gameplay chooses the state distribution but does not
replace the teacher target.

Start collection, with no optimizer or training during the games:

```powershell
.\.venv\Scripts\python.exe -m chess_ai mixed-batch `
  --config configs/gen2_mixed_d1_d2_400.yaml `
  --device cuda
```

If interrupted, run the identical command again. One opening is committed only after all eight
games, their PGNs, and its trace shard are saved. The final artifacts are:

```text
data/curricula/gen2_mixed_d1_d2_400/state.json
data/curricula/gen2_mixed_d1_d2_400/manifest.json
data/curricula/gen2_mixed_d1_d2_400/openings/opening_0001.pt ... opening_0050.pt
data/games/gen2_mixed_d1_d2_400/<matchup>/*.pgn
data/datasets/gen2_mixed_d1_d2_400.pt
```

Weights belong to the actor at each example: an eventual winner's moves carry `2.0`, both actors'
moves in a draw carry `1.5`, and an eventual loser's moves carry `1.0`. All eight games from one
opening share an `opening_set_id`, keeping the position out of the opposite validation split.

After all 400 games finish, train five conservative epochs from epoch 2. Every epoch is retained:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/gen2_mixed_d1_d2_400_train.yaml `
  --init-checkpoint checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt `
  --device cuda
```

Do not treat this run's `best.pt` as the gameplay winner. Run the separate selection suite:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select `
  --config configs/gen2_mixed_d1_d2_400_select.yaml `
  --device cuda
```

The selector tests the champion and each `epoch_*.pt` against D1 and D2 from identical unseen
positions. A candidate must improve aggregate points and may not regress in total score or standard
score at either depth. Gameplay score ranks candidates first; validation loss only breaks a
gameplay tie. The selected path is provisional: use new-seed `paired-audit` runs against D1 and D2
before manually promoting it. Selection produces no training update and never copies or overwrites
a champion checkpoint.

### Generate the 3,000-game D1/D2/D3 league

This long CPU experiment creates 500 opening sets. Every set is played six times: D2-vs-D1,
D2-vs-D3, and D1-vs-D3, with each pairing color-reversed. Every tenth set starts normally, while
the other 450 begin after 2-5 complete random legal moves. An independent deterministic D3 teacher
labels every subsequent position with a soft top-five policy. No optimizer runs during collection.

```powershell
.\.venv\Scripts\python.exe -m chess_ai mixed-batch `
  --config configs/minimax_d1_d2_d3_3000.yaml `
  --device cpu
```

Minimax search runs on the CPU, so CUDA does not accelerate this command. It can take a long time.
If interrupted, rerun the identical command: resume verifies and keeps every fully committed
six-game opening set before continuing.

To intentionally stop permanently after the currently committed opening sets, run this once:

```powershell
.\.venv\Scripts\python.exe -m chess_ai finalize-mixed-batch `
  --config configs/minimax_d1_d2_d3_3000.yaml `
  --device cpu
```

This does not play another game. It creates the combined dataset and manifest from only the
committed six-game shards, marks the state complete, and prevents later continuation. For the
current experiment, 65 openings produced 390 games and 18,427 examples in
`data/datasets/minimax_d1_d2_d3_3000.pt`.

### Import the reviewed Kaggle datasets and train in stages

The source CSVs remain under `downloaded_datasets/` and are never modified. Every import requires
an explicit acknowledgement, validates positions and moves with `python-chess`, records provenance,
uses deterministic bounded sampling, and refuses to overwrite an existing output.

Create the three separate datasets:

```powershell
.\.venv\Scripts\python.exe -m chess_ai import-kaggle `
  --config configs/kaggle_value_15000.yaml `
  --confirm-external-training-data

.\.venv\Scripts\python.exe -m chess_ai import-kaggle `
  --config configs/kaggle_games_15000.yaml `
  --confirm-external-training-data

.\.venv\Scripts\python.exe -m chess_ai import-kaggle `
  --config configs/kaggle_tactics_15000.yaml `
  --confirm-external-training-data
```

The value importer stratifies by game phase and evaluation band. Its neutral legal policy is a
shape placeholder only; the matching trainer sets policy loss to zero. The human-game importer
keeps only complete normal games with both ratings at least 2000 and a base clock of at least ten
minutes, then takes at most four deterministic positions per accepted game. The tactics importer
requires the supplied UCI move to be legal. The evaluation signs in these files are treated as
side-to-move scores and centipawns are mapped smoothly into `[-1, 1]`; mate scores map to the ends.

Train conservatively from the gameplay-approved generation-2 checkpoint, carrying each stage into
the next:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/kaggle_value_15000_train.yaml `
  --init-checkpoint checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt `
  --device cuda

.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/kaggle_games_15000_train.yaml `
  --init-checkpoint checkpoints/kaggle_value_15000_candidate_v1/best.pt `
  --device cuda

.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/kaggle_tactics_15000_train.yaml `
  --init-checkpoint checkpoints/kaggle_games_15000_candidate_v1/best.pt `
  --device cuda

.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/minimax_d1_d2_d3_65_train.yaml `
  --init-checkpoint checkpoints/kaggle_tactics_15000_candidate_v1/best.pt `
  --device cuda
```

The stages are intentionally ordered value knowledge, broad filtered human policy, stronger
tactical policy, then the project-specific D3 teacher. Preserve the old champion and treat the
final `best.pt` only as a candidate until fixed-seed D1 and D2 gameplay audits show non-regression.
If any stage regresses badly, skip that stage and initialize the following stage from the last
checkpoint that passed gameplay evaluation.

### Patience-based value plus combined-policy v2

The v2 correction avoids sequential policy forgetting. It keeps value-only pretraining separate,
then rehearses all policy sources in every epoch. Human examples carry weight `0.5`, tactical
Stockfish moves `1.5`, and D3 examples retain their actor outcome weights `1.0/1.5/2.0`.

Compose the already imported policy datasets:

```powershell
.\.venv\Scripts\python.exe -m chess_ai compose-datasets `
  --config configs/combined_policy_v2.yaml
```

Run value-only training for at most 200 epochs. Epoch zero remains eligible, validation value loss
selects `best.pt`, and training stops after 30 epochs without an improvement of at least `0.0001`:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/kaggle_value_patience_v2_train.yaml `
  --init-checkpoint checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt `
  --device cuda
```

Train the combined policy dataset for at most 200 epochs. Value loss is disabled because the
current NeuralAgent selects moves with policy only. Validation policy loss selects `best.pt`:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/combined_policy_patience_v2_train.yaml `
  --init-checkpoint checkpoints/kaggle_value_patience_v2/best.pt `
  --device cuda
```

Only every tenth epoch is retained, plus `best.pt`, `last.pt`, and the early-stop epoch. Evaluate
the retained checkpoints and the validation-selected best against D1 and D2:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select `
  --config configs/combined_policy_patience_v2_select.yaml `
  --device cuda
```

This selection is provisional. Run fresh paired audits using the checkpoint reported by the
selector; do not assume `best.pt` is the gameplay winner.

### Joint policy/value v3 with PUCT evaluation

The v3 stage applies the most useful ideas from the reviewed AlphaZero-family projects without
pretending that supervised teacher data is self-play reinforcement learning. It restores the
targeted D1/D2 dataset that produced the proven champion, mixes it with human, tactical, and D3
replay, trains policy and value together, and evaluates both heads through deterministic PUCT.

Compose the replay dataset once:

```powershell
.\.venv\Scripts\python.exe -m chess_ai compose-datasets `
  --config configs/joint_policy_value_v3.yaml
```

Start a fresh optimizer from the strongest audited checkpoint. Do not train from scratch: the model
shape is unchanged, and the source already contains demonstrated D1 skill.

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/joint_policy_value_v3_train.yaml `
  --init-checkpoint checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt `
  --device cuda
```

`best.pt` is only the validation-selected candidate. First test the raw policy with ordinary
`neural`. Before the final audits, rank every retained epoch through the same D1/D2 gameplay suite:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select `
  --config configs/joint_policy_value_v3_select.yaml `
  --device cuda
```

Use the `selected_checkpoint` printed by that command as the candidate below; `best.pt` is shown as
a placeholder in case validation and gameplay choose the same file. Then test policy plus value with
`neural-mcts`. A modest 32-simulation audit is a practical first pass; raise it to 64 after the
pipeline is behaving as expected.

```powershell
.\.venv\Scripts\python.exe -m chess_ai paired-audit `
  --candidate checkpoints/joint_policy_value_v3/best.pt `
  --champion checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt `
  --openings 30 `
  --audit-seed 9900205 `
  --opponent-depth 1 `
  --device cuda `
  --exclude-manifest data/curricula/gen1_vs_d1_d2_teacher_500/manifest.json `
  --pgn-dir data/games/evaluation/joint_policy_value_v3_greedy_d1

.\.venv\Scripts\python.exe -m chess_ai paired-audit `
  --candidate checkpoints/joint_policy_value_v3/best.pt `
  --champion checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt `
  --openings 30 `
  --audit-seed 9910205 `
  --opponent-depth 1 `
  --search-simulations 32 `
  --c-puct 1.5 `
  --device cuda `
  --exclude-manifest data/curricula/gen1_vs_d1_d2_teacher_500/manifest.json `
  --pgn-dir data/games/evaluation/joint_policy_value_v3_puct_d1
```

Repeat both audit styles against D2 with fresh seeds and `--opponent-depth 2` before promotion.
PUCT currently performs sequential leaf evaluations, so CUDA is supported but may be underutilized;
batched leaf inference is the next performance step.

### Opening refinement from v3 epoch 120

Epoch 120's stronger varied-opening PUCT result can be preserved while repairing its normal-start
weakness. This stage does not overwrite epoch 120. The reviewed `all-chess-openings` CSV contains
named legal lines, source game counts, and result percentages. The importer validates every line
with `python-chess`, merges transpositions into soft move distributions, and emits two separate
artifacts:

- a supervised policy/value dataset for low-rate refinement;
- a runtime opening book for exact, seeded-random continuation choices.

The runtime book provides the requested color-aware behaviour naturally. From the starting
position, White samples among known first moves using source game frequency. After White moves,
Black looks up that resulting position and samples only compatible replies. If either player leaves
known theory, the wrapper permanently changes that game to the configured neural or neural-PUCT
agent. It cannot re-enter book mode later in the middlegame. Full v3 replay remains in every
training epoch to reduce catastrophic forgetting.

Explicitly import the reviewed external source:

```powershell
.\.venv\Scripts\python.exe -m chess_ai import-openings `
  --config configs/all_chess_openings_import.yaml `
  --confirm-external-training-data
```

This creates `data/datasets/all_chess_openings_v1.pt` and
`data/opening_books/all_chess_openings_v1.json` without modifying the downloaded CSV.

Compose the opening emphasis with the complete v3 replay:

```powershell
.\.venv\Scripts\python.exe -m chess_ai compose-datasets `
  --config configs/opening_refinement_from_epoch120_v4.yaml
```

Start a fresh low-rate run from epoch 120:

```powershell
.\.venv\Scripts\python.exe -m chess_ai train `
  --config configs/opening_refinement_from_epoch120_v4_train.yaml `
  --init-checkpoint checkpoints/joint_policy_value_v3/epoch_0120.pt `
  --device cuda
```

Finally, rank the retained candidates through D1/D2 games. The original epoch-2 champion remains
the gate, so a v4 candidate must recover its standard-start score as well as improve aggregate play:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gameplay-select `
  --config configs/opening_refinement_from_epoch120_v4_select.yaml `
  --device cuda
```

The selected checkpoint is still provisional. Confirm it with fresh-seed greedy and 32-simulation
PUCT paired audits before promotion. Keep the source epoch 120 and epoch 2 files immutable.

To make an epoch-120-based agent use the exact opening book immediately, before refinement finishes:

```powershell
.\.venv\Scripts\python.exe -m chess_ai arena `
  --white neural-mcts `
  --white-checkpoint checkpoints/joint_policy_value_v3/epoch_0120.pt `
  --white-opening-book data/opening_books/all_chess_openings_v1.json `
  --black minimax `
  --black-depth 1 `
  --games 10 `
  --switch-colors `
  --simulations 32 `
  --device cuda `
  --pgn-dir data/games/evaluation/epoch120_opening_book_d1
```

When the neural agent is configured as Black, pass the same file through `--black-opening-book`.
Arena recreates agents with per-game seeds, so repeated games can choose different learned lines
while remaining reproducible for the same tournament seed.

The same one-way book-to-PUCT transition is available while playing in the GUI:

```powershell
.\.venv\Scripts\python.exe -m chess_ai gui `
  --checkpoint checkpoints/joint_policy_value_v3/epoch_0120.pt `
  --opening-book data/opening_books/all_chess_openings_v1.json `
  --search-simulations 32 `
  --c-puct 1.5 `
  --device cuda
```

Per-cycle epoch checkpoints use roughly the same space as the source checkpoint. With the current
model, expect approximately 700 MB for 99 retained epoch files, plus datasets and PGNs.

## 4. Train a tiny model

```powershell
python -m chess_ai train --config configs/dev.yaml
```

This uses a game-grouped train/validation split and one small epoch. Expect finite policy/value loss,
not good chess. Epoch checkpoints, `last.pt`, and `best.pt` are written beneath `checkpoints/dev/`;
metrics are appended to `data/metrics/dev_training.jsonl`.

To resume until the total epoch target in the YAML is reached:

```powershell
python -m chess_ai train --config configs/dev.yaml --resume checkpoints/dev/last.pt
```

Long-run controls are optional and backward-compatible:

- `selection_metric`: `validation_loss`, `validation_policy_loss`,
  `validation_value_loss`, or `validation_policy_top1_accuracy`.
- `early_stopping_patience`: zero disables patience stopping.
- `early_stopping_min_delta`: minimum metric change that resets patience.
- `checkpoint_every`: retain an `epoch_*.pt` every N epochs; `best.pt` and `last.pt` remain current.
- `baseline_eligible: true` writes `baseline.pt` for a run started with `--init-checkpoint` and
  keeps epoch zero eligible as `best.pt` until a new epoch improves the configured metric.

The checkpoint must match the model configuration. Optimizer state is restored when present. If the
checkpoint already reached `training.epochs`, the command performs no new optimizer steps; increase
that YAML value to continue farther.

To start a new run from existing weights instead of resuming its optimizer and epoch state, use
`--init-checkpoint`:

```powershell
python -m chess_ai train --config configs/human_gui.yaml `
  --init-checkpoint checkpoints/dev/best.pt
```

This reconstructs the architecture from the checkpoint, loads its weights, and starts with a fresh
optimizer, scheduler, metrics file, and epoch 1. When `--init-checkpoint` is present, its checkpoint
architecture is authoritative and the YAML `model` section is ignored. Without that flag, the YAML
model settings apply—including when `--resume` is used, where they must match the saved architecture.
`--resume` and `--init-checkpoint` are mutually exclusive. The new run's
`training.checkpoint_dir` must differ from the source checkpoint's directory so the source remains
unchanged. Fresh initialization also refuses to overwrite an existing candidate checkpoint or
metrics file. For another experiment, archive the old candidate or choose new run-specific values
for both `training.checkpoint_dir` and `training.metrics_path`.

## 5. Evaluate the model

Run a quick color-switched arena:

```powershell
python -m chess_ai arena --white neural --white-checkpoint checkpoints/dev/best.pt --black random --games 4 --switch-colors
```

Compare candidate and champion checkpoints without automatically promoting either:

```powershell
python -m chess_ai evaluate --candidate checkpoints/dev/best.pt --champion checkpoints/dev/last.pt --games 10
```

Arena output includes wins, draws, losses, score rate, and an explicitly approximate Elo difference.
Use many games and more than one opponent before drawing conclusions.

## 6. Play against an agent

Play White against a neural checkpoint:

```powershell
python -m chess_ai play --white human --black neural --black-checkpoint checkpoints/dev/best.pt
```

Enter moves in UCI form such as `e2e4`, `e7e8q` for promotion, or `e1g1` for castling. The board and
legal-input errors are printed in the terminal.

For a classical sanity check:

```powershell
python -m chess_ai play --white human --black minimax --depth 2
```

## 7. Collect human moves in the graphical workbench

Launch a neural game as White with collection visibly preselected:

```powershell
python -m chess_ai gui `
  --human-color white `
  --device cuda `
  --training-enabled
```

Use `--human-color black` for the next set of games so the dataset does not contain only one color.
The GUI saves game PGNs independently. Collection is opt-in and receives a second confirmation only
after a completed result: only human-chosen moves become policy labels, the confirmed result becomes
their value label, and unfinished games are excluded. The default paths are
`data/games/human_gui/` and `data/datasets/human_gui.pt`. The dataset is cumulative: every accepted
game is appended for later GUI training cycles, while unconfirmed and unfinished games never enter
it.

Each GUI launch reserves a unique session ID and these non-overlapping artifacts:

```text
data/gui_sessions/<session-id>/training.yaml
checkpoints/human_sessions/<session-id>/
data/metrics/human_sessions/<session-id>.jsonl
```

When you have collected the games you want, click **I'm Done — Train AI**. It starts a background
training process using the current session's source checkpoint, the cumulative confirmed-human
dataset, and the reserved YAML. The source checkpoint is preserved. The window remains responsive,
but closing is blocked while training is running. On success, the session's `best.pt` is selected in
the same window for the next game, and a fresh set of paths is reserved for the next training cycle.
The button remains disabled while a chess game is active; finish or resign and complete its save
decision first.

Collection consent remains separate from this button. A game contributes examples only when it is
complete, **Learn from this game (confirm at end)** was enabled, and the post-game confirmation was
accepted. Clicking the training button cannot add an unfinished or declined game.

The manual CLI remains available. You can use an unused GUI-generated session configuration:

```powershell
python -m chess_ai train `
  --config data/gui_sessions/<session-id>/training.yaml `
  --init-checkpoint <source-checkpoint> `
  --device cuda
```

Replace the placeholders with the generated session ID and the checkpoint selected when that cycle
was reserved. You can instead copy `configs/human_gui.yaml`, but both
`training.checkpoint_dir` and `training.metrics_path` must be changed to unused paths for every
fresh run.

Collect at least two confirmed games before expecting a validation split. A one-game dataset can
train, but all of that game remains in the training split and validation metrics are unavailable.
Five epochs can still overfit a handful of games. More cumulative examples are not automatically
better when they repeat one player's openings and mistakes; tiny or narrow personal data can make a
candidate weaker. Preserve the source model, inspect validation metrics, and compare candidates
before any manual promotion.

See `docs/gui_guide.md` for controls, collection semantics, and small-data cautions.

## 8. Manually benchmark with The Chess Lv.100

No external application is automated. Open it yourself, start a game with the requested color, and
run:

```powershell
python -m chess_ai external --checkpoint checkpoints/dev/best.pt --ai-color white
```

Type the opponent label and difficulty when prompted. Copy each printed AI move into the external
application, then type its reply into this terminal. Commands available instead of a move are
`undo`, `show`, `fen`, `help`, `resign`, and `quit`. The session saves a PGN and an independent
benchmark record. It is not added to training data.

See `docs/external_opponent_guide.md` for result handling and explicit import validation.

## 9. Inspect metrics and reports

```powershell
Get-Content data/metrics/dev_training.jsonl
python -m chess_ai report
```

Each JSONL line is one machine-readable record. `report` groups external benchmark games by opponent
level and checkpoint so one isolated result is not presented as proof.

## 10. Move beyond the tiny settings

After the development pipeline succeeds, inspect and adjust `configs/train.yaml`:

```powershell
python -m chess_ai generate-data --config configs/train.yaml
python -m chess_ai train --config configs/train.yaml
```

Start with small increases. Dataset generation by minimax can dominate runtime, and a larger network
does not compensate for narrow or low-quality labels.

## Common errors

- **`No module named chess_ai`**: activate the environment and run
  `python -m pip install -e ".[dev]"` from the repository root.
- **Configuration file not found**: check `Get-Location`; relative paths are resolved from the current
  directory.
- **Dataset not found**: run `generate-data` before `train`, and verify the configured path.
- **Incompatible checkpoint**: use the same channels, residual-block count, input planes, and action
  space that created it. Do not rename an unrelated `.pt` file to bypass validation.
- **Initialization output matches its source directory**: change `training.checkpoint_dir` in the
  fine-tuning YAML. Initialization intentionally refuses to risk overwriting its source model.
- **Initialization finds an earlier candidate**: archive that experiment or use new
  `training.checkpoint_dir` and `training.metrics_path` values. Fresh initialization never
  overwrites those artifacts automatically. The GUI training button avoids this collision by
  reserving unique human-session paths for each cycle.
- **Human GUI dataset not found**: finish and confirm at least one collected GUI game before running
  training. Saving a PGN alone does not create training data.
- **GUI refuses to close**: wait for the background training process to finish. Closing is blocked
  during training so a live fine-tune is not abandoned accidentally.
- **CUDA requested but unavailable**: set `device: cpu` or pass `--device cpu`. Doctor shows what the
  installed PyTorch build can use.
- **CUDA out of memory**: reduce batch size or model channels, close other GPU workloads, and resume
  from the last valid checkpoint. The trainer adds this guidance to detected OOM errors.
- **Loss is `nan` or infinite**: stop the run; lower the learning rate, inspect the dataset, and use
  doctor/tests before resuming. Invalid numeric loss is treated as an error.
- **Teacher cycle finds existing outputs without state**: restore the matching `state.json`, or
  choose fresh `state_path`, `corrections_dir`, `checkpoint_dir`, `metrics_path`, and `pgn_dir`
  values. The curriculum will not guess whether orphaned artifacts belong to the requested run.
- **Teacher cycle produces no win**: the run cannot guarantee a win. Keep the source checkpoint,
  inspect correction disagreement and PGN terminations, and do not advance difficulty unless the
  explicit mastery gate passes.
- **Teacher batch finds outputs without state**: restore the matching
  `data/curricula/gen1_vs_d1_d2_teacher_500/state.json`, or choose fresh state, manifest, dataset,
  pair-shard, and PGN paths. It will not guess whether orphaned games belong to this experiment.
- **Teacher batch is slow despite CUDA**: D1 and exact-root D2 run on the CPU. CUDA accelerates only
  champion inference; do not raise teacher depth without first timing a small separate experiment.
- **Paired audit reports an existing-output collision**: choose a new `--pgn-dir` for every seed.
  Evaluation artifacts are never silently overwritten.
- **No validation games**: generate at least two distinct complete game IDs or lower the validation
  fraction carefully. Positions from one game are never split between both groups.
- **Very slow minimax generation**: lower `minimax_depth`, `games`, or `max_moves`. Depth grows the
  search tree rapidly.
- **External move rejected**: use UCI coordinates, include a promotion suffix, and compare the printed
  FEN/board with the external app. `undo` can repair a transcription error.
