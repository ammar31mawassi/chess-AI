# Prompt for the next chat: endgame improvement

Continue development of my educational neural-network chess AI in:
`C:\Users\ammar\OneDrive\Desktop\chess-AI`.

I think the opening has improved and the middlegame is acceptable. My next priority is the endgame. Treat this as a hypothesis to measure, rather than an established diagnosis. Research practical ways to improve endgame play and assess whether importing games from Hikaru Nakamura, Magnus Carlsen, and other GMs would help. Inspect the repository, investigate the evidence, and implement the justified next step. Give me exact PowerShell commands for expensive data collection, labeling, training, and evaluation; I will run those jobs myself.

## Read first and preserve

- Read `AGENTS.md`, `docs/architecture.md`, `docs/training_guide.md`, `docs/stockfish_rebuild.md`, `docs/stockfish_refinement.md`, `docs/personal_stockfish_training.md`, and `docs/chesscom_download.md`.
- There are many existing uncommitted changes. Inspect them and preserve unrelated work. Do not reset the repository or overwrite existing experiments.
- Use `.venv\Scripts\python.exe`. Stockfish runs on CPU; the neural model can use CUDA.
- Delegate all chess rules and legality decisions to python-chess.
- Do not inspect, automate, or control any external chess GUI, online game, or playing interface. Public API documentation research is separate from GUI interaction. Assess public archive APIs and provide download commands if appropriate; do not launch bulk downloads as part of the initial investigation.
- External training imports must be explicit, validated, and traceable. Evaluation games and benchmark positions must stay out of training. Respect deterministic seeds and whole-game/group splits.
- Run Ruff formatting/checks, mypy, and the full pytest suite after implementation. Update docs and version incompatible dataset/checkpoint changes.

## Current champion and inference

Read and verify `checkpoints/champion.json` rather than selecting the newest checkpoint by filename.

- Champion: `checkpoints/champions/stockfish_rebuild_eval_v2_20261011_01_epoch_0060.pt`.
- SHA256: `7b2972f1aa5a1a9caa73db7c1c2f60a48061820b66cca5cdc452e99ba0cdc37a`.
- Runtime: PUCT 32 simulations, `c_puct=1.5`, no opening book in the registered default. An optional opening-book implementation exists. Thirty-two simulations are tree traversals, not depth 32.
- Both policy and value heads affect PUCT. This is supervised teacher training, not an AlphaZero reinforcement-learning loop.
- Current evaluation labels: `stockfish_cp_tanh_v1`, side-to-move perspective, `tanh(cp/600)`. Human game results are not these value targets.
- Check the encoder's available information, especially halfmove clocks and repetition history, before proposing rule-dependent endgame targets or search behavior.

## Completed experiments: do not repeat rejected promotions

1. `data/experiments/stockfish_refinement_v2_20261015_01`: epoch 14 won development selection, but fresh confirmation did not establish improvement. It was not promoted.
2. `data/experiments/chesscom_xxkiller31_20261018_01`: personal Chess.com PGNs were validated and relabeled by Stockfish. Human moves supply encountered positions; Stockfish supplies policy/value targets for both colors. Training initialized from the epoch-60 champion, with fresh optimizer, learning rate `1e-5`, policy/value coefficients 1/1, and equal fresh/historical replay position counts. It stopped at epoch 12; lowest validation loss was epoch 7.
3. Gameplay selection chose personal-data epoch 10 with +6 points. Report: `data/games/evaluation/chesscom_xxkiller31_20261018_01_puct_selection_retry2/report.json`.
4. Fresh confirmation rejected promotion: across two seeds, epoch 10 scored 49 versus champion 51.5 against D1, and 19 versus 21 against D2: total 68 versus 72.5 across 128 opponent games per checkpoint. Direct matches: 13 wins, 36 draws, 15 losses, or 31/64 points for epoch 10 versus 33/64 for champion.
5. Confirmation plan: `data/experiments/chesscom_xxkiller31_20261018_01/configs/confirmation_plan.json`. Reports: `data/games/evaluation/chesscom_xxkiller31_20261018_01_puct_confirmation_{20112036,21112036}_d{1,2}/report.json`.

D1/D2 in these audit implementations are our local minimax opponents; check the code and do not describe them as Stockfish depth settings. Selection is provisional. Keep epoch 60 unless independent evidence supports a later explicitly authorized promotion. Do not keep testing alternative checkpoints on the same confirmation holdout until one happens to win.

## Existing tools to reuse

- `scripts/download_chesscom_games.py`: public monthly archive downloader; raw PGN download is separate from import and labeling.
- `scripts/new_personal_stockfish_training.py` and `src/chess_ai/data/personal_pgn.py`: explicit validated snapshots and personal-PGN experiment preparation. Inspect assumptions tied to a single username before adapting for multiple GMs.
- `src/chess_ai/data/stockfish_rebuild.py`: inventory, pilot, SQLite-journaled resumable position labeling, finalize, deduplicated grouped shards. PGN-only inputs are supported.
- `src/chess_ai/data/stockfish_teacher.py`: pinned teacher, analysis cache, MultiPV policy labels, position and played-move analysis. Existing teacher is 50,000 nodes, MultiPV 5, one thread, 128 MB hash; inspect actual configs. Syzygy is currently disabled in the teacher settings, so do not silently enable it without changing provenance/cache bindings.
- `src/chess_ai/data/refinement_replay.py`: protected replay and explicit compatible value semantics.
- `src/chess_ai/arena/blunder_diagnostics.py`, `src/chess_ai/training/diagnostics.py`, and `docs/model_improvement_comparison.md`: inspect before inventing another diagnostics pipeline.
- Local Stockfish: `references/github/stockfish-build/49ea5ded38315cff8e67f4a677a9e7811612fbf6/stockfish.exe`.
- Original replay: `data/experiments/stockfish_rebuild_eval_v2_20261011_01/relabel/dataset.json`.

Collection/labeling has resume mechanisms. Gameplay selection currently does not resume partial games and refuses existing output PGNs; interrupted attempts used new folders. Design any new lengthy endgame stages with resumable state, atomic writes, checkpoint/input hashes, progress reporting, and useful ETA information. Keep paths relative to the actual `chess-AI` repository, not a nonexistent `chessAI` folder.

## Investigation and proposed experiment

1. Measure the weakness using a small reproducible diagnostic sample. Separate reaching an already-lost endgame from throwing away a won or drawn endgame. Examine conversion, defense, pawn promotion, king activity, mating technique, stalemate, repetition, and move-limit draws. Existing evaluation games may inform diagnosis but must remain excluded from training.
2. Research current primary sources and link them. Compare targeted Stockfish-labeled endgame positions, naturally reached endgames from GM PGNs, reproducible legal endgame generation, Syzygy training labels, and optional runtime tablebase probing. Consider search budgets separately from learning improvements.
3. Assess GM archives specifically: verify public accounts and archive availability without guessing identities; compare time controls, useful endgame coverage, early resignations, duplicate games across players, labeling cost, and diversity. Do not assume famous-player moves or the game's result are ideal labels. Keep using Stockfish or appropriately modeled tablebase supervision. Recommend whether a capped GM pilot adds value over existing positions and a targeted curriculum before collecting everything.
4. Investigate Syzygy WDL/DTZ semantics carefully: side-to-move signs, cursed wins/blessed losses, the fifty-move counter, captures/promotions, missing material tables, and multiple equally good moves. DTZ is not distance to mate. Do not silently mix exact WDL targets with the current centipawn-tanh regression targets. Specify compatibility or a versioned migration. Estimate disk requirements before proposing large tablebase downloads.
5. Distinguish network learning from optional external assistance at inference. If a runtime tablebase is added, report unassisted and assisted performance separately and compare candidates under identical assistance. An assisted playing improvement alone does not prove improved network endgame understanding.
6. Prepare a small, balanced, reproducible pilot: both colors; winning/drawn/losing positions; pawn, rook, minor-piece, and basic mating families. Prefer reachable positions and validate them with python-chess. Prevent leakage across games, transpositions, identical network inputs, and closely related generated families. Preserve all historical holdouts and reserve new independent endgame holdouts before training.
7. Fine-tune a new candidate from the frozen champion with justified replay/sampling and learning settings. Avoid forgetting opening/middlegame skills. Select a limited number of candidates on development data, then confirm once on fresh endgame and whole-game suites. Include position quality, conversion/defense rates, draw-rule mistakes, runtime, and overall non-regression; D1/D2 wins alone are insufficient endgame evidence.

Start with repository findings and a concise evidence-based recommendation. Implement and test the smallest useful pilot pipeline, document expected disk/time costs, then give me commands in execution order. Do not launch long training, exhaustive labeling, massive downloads, or promotion during initial setup. Do not promise that more GM games or more epochs will necessarily make the model stronger.

Primary research starting points, checked on 2026-10-23:

- python-chess Syzygy documentation: https://python-chess.readthedocs.io/en/stable/syzygy.html
- Chess.com Published-Data API documentation: https://www.chess.com/news/view/published-data-api

Recheck current documentation and repository state rather than treating this handoff as immutable.
