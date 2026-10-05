# External neural-chess references

These repositories were shallow-cloned on 2026-09-21 for architecture and
training research. They are reference material only: none of their code,
weights, or datasets are imported into this project or its training data.
Star counts are a point-in-time snapshot and will change.

| Repository | Stars | Pinned commit | License | Most useful lesson for this project |
| --- | ---: | --- | --- | --- |
| [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) | 3,222 | `1227b4c8c233` | GPL-3.0 | Batched neural evaluation, PUCT search, self-play, and production chess-engine design. |
| [LeelaChessZero/lczero-training](https://github.com/LeelaChessZero/lczero-training) | 190 | `7c5d756ea6bb` | No repository-level license found; study only | Canonical LCZero policy/value training, data loading, learning-rate tuning, and checkpoint tools. |
| [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero) | 2,226 | `db5961e6bc0e` | MIT | A compact chess-specific self-play, optimization, evaluation, and promotion loop. |
| [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | 4,515 | `f1a78e0505c6` | MIT | Readable PUCT MCTS, replay history, arena comparison, and champion gating. |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | 5,497 | `48401890ee98` | Apache-2.0 | Reference-quality AlphaZero algorithms, evaluation patterns, and game abstractions. |
| [lightvector/KataGo](https://github.com/lightvector/KataGo) | 5,133 | `3c144b3e7971` | MIT for the main code; bundled dependencies vary | Efficient self-play pipelines, replay-window management, auxiliary targets, and gatekeeping. |
| [CSSLab/maia-chess](https://github.com/CSSLab/maia-chess) | 1,242 | `749204cf5979` | GPL-3.0 | Large-scale supervised chess move prediction and careful human-game stratification. |
| [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish) | 16,689 | `17a6c8f1eb0d` | GPL-3.0 | Strong teacher/evaluation behavior, reproducible search tests, and engine benchmarking. |
| [google-deepmind/mctx](https://github.com/google-deepmind/mctx) | 2,666 | `88f92056a420` | Apache-2.0 | Clean, batched, accelerator-friendly MCTS policies and search interfaces. |
| [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) | 2,870 | `0825bd544fc1` | MIT | Replay buffers, self-play orchestration, checkpointing, and evaluation dashboards. |

## Recommended reading order

1. `alpha-zero-general/MCTS.py` and `alpha-zero-general/Coach.py` for the
   clearest small implementation of search plus a training loop.
2. `chess-alpha-zero/src/chess_zero/worker/evaluate.py` for chess-specific
   candidate-versus-champion promotion.
3. `lc0/src/search/` and `lc0/src/selfplay/` for a production implementation.
4. `lczero-training/docs/README.md` and `lczero-training/docs/training_tuple.md`
   for modern policy/value data and training design.
5. `KataGo/SelfplayTraining.md` and `KataGo/docs/KataGoMethods.md` for efficient
   replay, self-play, and gating ideas that transfer beyond Go.
6. `mctx/` and OpenSpiel's AlphaZero examples for tested search APIs.

## Immediate adaptations worth considering

- Add a small PUCT search around the existing policy and value heads instead
  of choosing the largest policy logit directly.
- Train policy and value jointly so policy fine-tuning does not erase the
  shared representation learned by value training.
- Generate policy targets from search visit counts rather than only a single
  teacher move.
- Maintain a replay window containing champion, new self-play, tactical, and
  standard-start examples instead of replacing one distribution with another.
- Use paired candidate-versus-champion games and confidence-aware promotion;
  keep validation loss as a diagnostic rather than the promotion criterion.
- Batch neural evaluations across simultaneous games/search leaves to keep the
  RTX GPU busy while CPU workers handle chess tree expansion.

Before adapting code, inspect the repository's license and reimplement the
idea in this project's educational style. In particular, do not copy code from
repositories with GPL or missing license terms into this project without an
explicit licensing decision.
