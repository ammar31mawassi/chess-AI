# Neural networks for beginners

This chapter follows one position through the exact Phase 1 code. The model is intentionally small;
understanding the data flow matters more than making it large.

## 1. A tensor is an arranged collection of numbers

A scalar is one number (`3`). A vector is a row of numbers (`[3, 5]`). A matrix has rows and
columns. A **tensor** generalizes those ideas to any number of axes. `BoardEncoder.encode()` in
`src/chess_ai/environment/board_encoder.py` returns a NumPy tensor shaped `(18, 8, 8)`. The trainer
stacks several boards into a PyTorch tensor shaped `(batch, 18, 8, 8)`.

For batch size 4, there are `4 × 18 × 8 × 8 = 4,608` input numbers. Their shape tells us which game,
plane, rank, and file each number describes.

## 2. The board becomes 18 numerical planes

`BoardEncoder.encode()` writes `1.0` where a feature is present and `0.0` elsewhere:

- planes 0–5: white pawn, knight, bishop, rook, queen, king;
- planes 6–11: the same six black pieces;
- plane 12: all ones when White moves, otherwise all zeros;
- planes 13–16: one full plane per castling right;
- plane 17: one at the en-passant target square, if any.

Coordinates are `[plane, rank, file]`, with index 0 meaning rank 1 or file `a`. At the initial
position, `board[0, 1, 4] == 1.0` describes the white pawn on e2, while
`board[11, 7, 4] == 1.0` describes the black king on e8. This encoding preserves facts that a plain
piece diagram omits, such as whose turn it is and whether castling is still legal.

## 3. What one neuron computes

A neuron forms a weighted sum and usually applies an activation. With inputs `[2, 3]`, weights
`[0.5, -1]`, and bias `1`, the pre-activation is:

```text
2 × 0.5 + 3 × (-1) + 1 = -1
```

ReLU, used in `PolicyValueNet`, returns `max(0, x)`, so this neuron's output becomes 0.

## 4. Weights and biases

**Weights** decide how strongly an input matters; their sign decides whether it raises or lowers an
activation. A **bias** shifts the result even when inputs are zero. PyTorch owns these trainable
numbers in each `nn.Conv2d` and `nn.Linear` in
`src/chess_ai/model/policy_value_net.py`. Training changes them; ordinary inference does not.

## 5. Convolution

A convolution slides a small learned window over a board. The first `3 × 3` convolution in
`PolicyValueNet.input_block` can combine a square with its neighbors—for example, a pawn and the
squares it attacks. The same window weights are reused across all 64 squares. This is both smaller
and better matched to board geometry than giving every location a completely unrelated rule.

Padding of 1 keeps the board at `8 × 8`. With 32 configured channels, the input changes from
`18 × 8 × 8` into `32 × 8 × 8` learned features.

## 6. Feature maps

Each output channel of a convolution is a **feature map**. It is not assigned a human label, but it
may learn patterns such as open lines, defended pieces, or king exposure. A map is still an 8 by 8
grid: a large activation at one square means that learned pattern is present strongly there.

## 7. Residual blocks

`ResidualBlock.forward()` applies two convolutions and then adds the original input:

```python
output = relu(transformed_features + original_features)
```

The shortcut gives gradients a direct route through deeper networks and lets a block preserve useful
features easily. `configs/dev.yaml` uses only two blocks so CPU experiments remain approachable.

## 8. The policy output

`PolicyValueNet.policy_head` produces 4,208 numbers—one **logit** per action defined by
`MoveEncoder` in `src/chess_ai/environment/move_encoder.py`. A larger number means the model prefers
that move, but it is not yet a probability and it does not mean the move is legal.

During supervised bootstrap, `generate_game_examples()` (called by `generate_supervised_data()`) in
`src/chess_ai/data/dataset_generator.py` stores a one-hot target: the classical agent's selected
action is 1 and all other actions are 0. Policy cross-entropy trains the selected action's logit to
become relatively larger.

## 9. The value output

`PolicyValueNet.value_head` produces one number per board. `tanh` limits it to `[-1, 1]`: values near
1 favor the player to move, near -1 favor that player's opponent, and near 0 suggest an even/drawn
result. The label in each `TrainingExample` is the completed game's result from the perspective of
the player who was to move in that saved position—not always White's perspective.

## 10. Logits and softmax

Softmax converts logits into probabilities. For logits `[2, 1]`:

```text
exp(2) / (exp(2) + exp(1)) ≈ 0.73
exp(1) / (exp(2) + exp(1)) ≈ 0.27
```

Adding the same number to both logits changes neither probability. Training therefore cares about
relative preference. PyTorch's cross-entropy accepts logits directly and performs the stable softmax
calculation internally.

## 11. Legal-move masking

Chess legality is not learned or guessed. `NeuralAgent.choose_move()` in
`src/chess_ai/agents/neural_agent.py` calls `MoveEncoder.legal_action_mask(board)`, excludes every
zero entry, and chooses only among legal indexes. If a model gives an illegal move the largest raw
logit, that logit is ignored. `python-chess` remains the final source of truth.

## 12. Loss functions

`src/chess_ai/training/losses.py` combines:

```text
policy cross-entropy + value mean-squared error + L2 regularization
```

If the target value is 1.0 and the prediction is 0.4, its squared error is
`(1.0 - 0.4)² = 0.36`. Policy loss measures move-classification error. L2 discourages unnecessarily
large weights. The components are logged separately so one cannot hide the behavior of another.

## 13. Gradient descent

A gradient says how a small parameter change would change loss. Gradient descent nudges parameters
in the direction that lowers loss. The learning rate controls step size. `Trainer` uses AdamW, which
adapts steps using recent gradient history and implements weight decay cleanly.

Too large a learning rate can make loss jump or become non-finite; too small can make learning
imperceptibly slow. `configs/dev.yaml` starts at `0.001` as a pipeline setting, not a universal best
value.

## 14. Backpropagation

PyTorch records how the forward output was computed. Calling `loss.backward()` applies the chain
rule backward through value head, policy head, residual tower, and input convolution, filling each
parameter's `.grad`. `Trainer` then clips excessively large gradients and calls `optimizer.step()`.
Tests verify that this backward pass produces finite gradients.

## 15. Epochs and batches

A **batch** is the subset processed in one optimizer step. An **epoch** is one pass over all training
examples. With 100 examples and batch size 8, one epoch has 13 batches (the final one has four
examples). Smaller batches use less memory; more epochs revisit the data more times.

## 16. Training versus inference

During training, PyTorch tracks gradients and batch-normalization layers update their statistics.
During inference, `load_model(..., eval_mode=True)` calls `model.eval()`, and
`NeuralAgent.choose_move()` uses `torch.inference_mode()`. This is faster and prevents accidental
weight updates. Training answers “how should weights change?”; inference asks only “what does this
fixed checkpoint predict?”

## 17. Overfitting

Overfitting occurs when training performance improves while performance on unseen positions does
not. A model might memorize positions or quirks of one minimax depth instead of learning reusable
chess structure. More model capacity is not automatically better. Diverse games, held-out
validation, regularization, and arena opponents help expose overfitting.

## 18. Validation data

`split_examples_by_game()` in `src/chess_ai/data/examples.py`, called by
`Trainer._prepare_examples()`, assigns complete game IDs to either training or validation. It never
scatters adjacent positions from one game across both sets. That would leak nearly identical boards
and make validation look misleadingly good. The fixed seed makes the assignment reproducible.

## 19. Checkpoints

`save_checkpoint()` in `src/chess_ai/model/checkpoint.py` stores model weights, architecture, epoch,
metrics, and optional optimizer/scheduler state. It writes a temporary file and atomically replaces
the destination. `load_checkpoint()` checks the format version and architecture before applying
weights, which turns silent shape mistakes into readable errors. `best.pt` means best held-out loss
within that run; it does not mean proven strongest chess play.

## 20. Why one win proves very little

Chess results are noisy: colors, opening, sampling, and one tactical mistake can decide a game. If a
new checkpoint wins once, it may simply have been lucky. `arena` switches colors and aggregates many
games; external reports group results by checkpoint and opponent level. Even a 6–4 score has high
uncertainty. Record the experiment, run more games against multiple opponents, and keep promotion a
deliberate human decision.
