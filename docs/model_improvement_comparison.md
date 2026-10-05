# Model improvement comparison

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
