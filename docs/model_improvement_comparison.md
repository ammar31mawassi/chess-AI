# Model improvement comparison

Current champion (2026-10-15): the Stockfish rebuild's epoch 60, frozen under
`checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt`, with PUCT 32 and no
book. It passed twelve D1/D2 audits and scored 48 wins, 16 draws, and zero losses in direct games
against epoch 120. See [the promotion record and playback commands](stockfish_rebuild.md).
The sections below document the earlier experiments and their rationale.

This comparison was made from pinned local snapshots listed in
[`references/github/README.md`](../references/github/README.md). They are research references only;
no external source code, weights, or unreviewed games were copied into this project.

## What mature projects do differently

| Area | Previous project behaviour | Reference pattern | Change in this stage |
| --- | --- | --- | --- |
| Move choice | One policy forward pass, then legal argmax | AlphaZero General, LCZero, OpenSpiel, and MCTX combine policy priors with value-guided PUCT | Added deterministic `neural-mcts` play with legal priors, leaf values, alternating-perspective backup, and visit-count selection |
| Objective | Value-only and policy-only stages could move the shared trunk in different directions | AlphaZero-family trainers learn policy and value together | Added a low-rate joint objective: policy `1.0`, value `0.25` |
| Replay | A later narrow dataset could replace an earlier skill distribution | KataGo and MuZero-style pipelines retain bounded mixtures of old and new experience | Added an explicit v3 mix of targeted D1/D2 replay, tactics, human positions, and the D3 league |
| Model selection | Validation loss found candidates but did not predict chess strength reliably | Chess AlphaZero and AlphaZero General gate candidates through arena games | Epoch zero stays eligible, but promotion still requires paired D1/D2 gameplay audits |
| Policy labels | Human moves, shallow teacher moves, or teacher score softmax | AlphaZero uses MCTS root visit counts | Not yet changed; search-visit training targets are the next data-format stage |
| GPU use | Training is batched; game generation and greedy inference are mostly serial | LCZero, KataGo, and MCTX batch many leaf or game evaluations | Not yet changed; current PUCT is intentionally sequential and modest-sized |

Primary references: [AlphaZero General](https://github.com/suragnair/alpha-zero-general),
[Chess AlphaZero](https://github.com/Zeta36/chess-alpha-zero),
[LCZero](https://github.com/LeelaChessZero/lc0),
[OpenSpiel](https://github.com/google-deepmind/open_spiel),
[KataGo](https://github.com/lightvector/KataGo), and
[MCTX](https://github.com/google-deepmind/mctx).

## Why this run should start from the champion

The network shape has not changed, and `epoch_0002.pt` already passed two independent paired audits
against the previous champion. Starting from scratch would discard that demonstrated D1 skill and
require much more data before the value-guided search becomes useful. The safer experiment is a
fresh optimizer at a `1e-6` learning rate, initialized from:

`checkpoints/gen1_vs_d1_d2_teacher_500_candidate_v1/epoch_0002.pt`

The source checkpoint remains immutable. `baseline_eligible: true` means epoch zero can remain the
validation winner if all updates are worse. This is a candidate-building step, not automatic
promotion.

## Implemented workflow

1. `configs/joint_policy_value_v3.yaml` composes four explicit supervised sources. Evaluation PGNs
   are excluded. Per-source group keys keep color-swapped games from the same opening on one side
   of the train/validation boundary.
2. `configs/joint_policy_value_v3_train.yaml` jointly trains both heads for at most 200 epochs,
   stops after 30 non-improving epochs, and keeps periodic checkpoints.
3. `neural-mcts` can be selected in `play` and `arena`; `--simulations` controls the search budget.
4. `configs/joint_policy_value_v3_select.yaml` ranks retained epochs through fixed D1/D2 games;
   validation loss only breaks gameplay ties.
5. `paired-audit --search-simulations N` evaluates candidate and champion with the same search
   budget on identical unseen openings.

Validation loss is useful for narrowing candidates, but a checkpoint should become champion only
after fresh-seed paired audits against D1 and D2 show non-regression. Search and training are two
separate improvements: also test greedy `neural` play so search cannot hide a policy regression.

## Stockfish-stage roadmap

This ordering is a project-specific assessment after inspecting our collector, trainer, replay,
greedy/PUCT agents, and gameplay gates. Benefit estimates are hypotheses for ablation, not measured
strength gains. The current stage adds bounded independent Stockfish MultiPV supervision, outcome
weights, a capped tactical reservation, explicit disagreement/entropy audit fields, replay/opening
rehearsal, initialization-baseline selection, richer result slices, and descriptive paired uncertainty.
It retains legal masking already implemented in both the trainer and agents. It does not alter the
network, reinterpret value targets, or implement a reinforcement-learning loop.

| Priority / improvement | Expected benefit | Implementation risk | Compute cost | Fit and next experiment |
| --- | --- | --- | --- | --- |
| 1. Legal masking in training/inference | High correctness benefit | Low; already tested | Small CPU mask preparation | Enabled in the new training config; illegal teacher actions fail before saving |
| 2. Bounded tactical-position oversampling | Medium/high label efficiency | Low with a capped reservation | Lower than all-ply labeling | Implemented: ≤24 examples/game and 25% tactical reservation; compare with uniform-only sampling |
| 3. Teacher-student disagreement sampling | Potentially high correction efficiency | Medium: selection bias, weak value signal | Neural scoring is cheap; disagreements still require teacher analysis | Store actual move, teacher move, actor kind, and disagreement now; next oversample neural disagreements with a uniform coverage floor |
| 4. Historical/old-new replay balance and bounded replay window | High protection against forgetting | Low/medium: source/group sampling must remain leak-free | Disk/RAM more limiting than math | Explicit full v3 + openings retained now, game-balanced sampling enabled; next add measured source quotas and a bounded whole-game window, not arbitrary position truncation |
| 5. Automated multi-seed regression suites | High reliability, indirect strength benefit | Low | High evaluation game count | Copy-paste three-seed greedy/PUCT D1/D2/direct-baseline matrix in the runbook; next automate report aggregation and resume at completed suite boundaries |
| 6. Confidence-aware checkpoint promotion | High decision quality | Medium: correlated openings and selection bias | More fresh games | Paired opening-bootstrap diagnostics and small-sample flags implemented; promotion stays manual. Next define paired statistical non-regression margins, multiple-seed confirmation, and repeated-testing policy |
| 7. Better value targets and calibration | High for PUCT if current outcome labels are noisy | Medium/high: changing semantics requires a versioned target | Moderate teacher and calibration work | Keep outcome targets and separate teacher evaluations now; next compare reliability curves/Brier/MSE by game phase, with explicit teacher-value auxiliary targets rather than silent substitution |
| 8. Policy entropy and top-k calibration metrics | Medium diagnostic benefit | Low/medium | Cheap validation batches | Teacher entropy and existing top-1/top-5 imitation accuracy available. Next add student legal entropy, target cross-entropy/KL, top-k target mass, and calibration against held-out teacher labels; do not call top-k imitation chess strength |
| 9. Opening/middlegame/tactical/endgame evaluation separation | Medium/high diagnosis | Low/medium: require legal curated positions and clean holdouts | Moderate extra matches | Standard versus varied starts and colors now reported. Next add independent fixed phase/tactical suites with provenance, excluding training/cached FENs |
| 10. Batched neural inference during PUCT | High throughput benefit | High: tree coordination/virtual losses change behavior | Better GPU utilization; more RAM | Next focused performance project: benchmark sequential reference first, batch across independent games or pending leaves, verify deterministic backups |
| 11. Search-visit policy targets | Potentially high alignment with actual PUCT choices | Medium/high: collector needs root visit export and target provenance | High self-play/search generation cost | Current Stockfish score softmax is not a visit target. Add an explicit target-kind/version and tested root visit normalization in a later data stage |
| 12. Root Dirichlet noise and temperature | Medium/high self-play diversity | Medium: unsuitable for deterministic evaluation | Modest overhead but longer diverse games | Enable only in a new seeded self-play mode; keep audits deterministic, separate training exploration settings from inference settings |
| 13. Valid chess symmetry augmentation | Uncertain/moderate data efficiency | High correctness risk: castling, pawn direction, promotions, en-passant and action mapping | Low extra inference, more training samples | Later prove one transform end-to-end through python-chess and move/plane mappings. Do not apply Go-style rotations or blindly mirror files when castling rights exist |
| 14. Larger/better architecture | Potentially high only with sufficiently broad data | Medium/high compatibility risk | High GPU memory/training cost | Defer until label quality/calibration/search bottlenecks are measured. Changing channels/actions/input semantics needs compatible fresh configs or explicit checkpoint versions |
| 15. PUCT-visit + self-play outcome loop | Highest long-term ceiling, uncertain at this scale | High, several interacting layers | Very high sustained generation/training/evaluation | Combine 10–12 with bounded replay and tested gameplay gates as a separately versioned educational AlphaZero-style loop; do not replace this supervised baseline all at once |

The [AlphaZero paper](https://arxiv.org/abs/1712.01815) motivates search-distribution policy targets,
self-play outcomes, and exploration, but its compute scale does not justify assuming our small
supervised student will reproduce its strength. [LCZero](https://github.com/LeelaChessZero/lc0)
provides a chess-specific policy/value search reference, while
[MCTX](https://github.com/google-deepmind/mctx) demonstrates accelerator-oriented batched search.
Those designs suggest directions; this table's ordering and risk judgments are our inference for
the present educational repository.

Stockfish is a separately executed [official UCI teacher](https://github.com/official-stockfish/Stockfish),
not a new chess-rules authority or a gameplay participant here. Read the
[v1 runbook](stockfish_teacher.md) before running its long experiments. The safest next decision
is whether these stronger, bounded labels improve fresh D1/D2 and direct-baseline games without
standard-start regression; architecture expansion should follow that evidence.
