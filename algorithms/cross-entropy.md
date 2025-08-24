---
title: 'cross entropy loss'
type: 'algorithm'
version: '1'
date: '2025-08-24'
license: 'cc-by-nc-sa-4.0'
---

# Cross Entropy Loss

### Purpose

Cross entropy loss measures the difference between the predicted unnormalized
logits (outputs of your model) and the target labels. It is used for
classification tasks with **C** classes, and supports **class weighting** to
handle unbalanced datasets.

### Input & Target

- **Input:** Tensor of logits (not probabilities), shape:

  - Unbatched: `(C,)`
  - Batched: `(minibatch, C)`
  - Higher-dimensional (e.g., images): `(minibatch, C, d1, d2, ..., dK)`, with
    `K ≥ 1`

- **Target:**

  - **Class indices:** Each entry is an integer in `[0, C)` (or `ignore_index`
    if specified)
  - **Probabilities:** Each entry is a probability distribution over classes
    (for label smoothing, blended labels, etc.)

### Class Index Targets (Most Common Case)

For targets as class indices (integer labels):

$$
\ell(x, y) = L = \{ l_1, \ldots, l_N \}^T, \quad l_n = -w_{y_n} \cdot \log \left( \frac{\exp(x_{n, y_n})}{\sum_{c=1}^C \exp(x_{n, c})} \right) \cdot 1\{y_n \neq \text{ignore\_index}\}
$$

- $x$: Logits

- $y$: Target class index

- $w$: Optional class weight

- $N$: Number of items (batch \* spatial dims)

- $C$: Number of classes

- **Reduction** (default = `'mean'`):

  - `'none'`: Returns `L`, the unreduced per-sample loss
  - `'mean'`: Averages over non-ignored elements, optionally weighted
  - `'sum'`: Sums all loss values

**Note:** This is equivalent to applying `LogSoftmax` to the logits, followed
by `NLLLoss` (Negative Log Likelihood Loss).

### Probability Targets (Soft/Blended Labels)

If your targets are probability distributions over classes (e.g., for label
smoothing):

$$
\ell(x, y) = L = \{ l_1, \ldots, l_N \}^T, \quad l_n = -\sum_{c=1}^C w_c \cdot \log \left( \frac{\exp(x_{n, c})}{\sum_{i=1}^C \exp(x_{n, i})} \right) y_{n, c}
$$

- $y_{n, c}$: Probability assigned to class $c$ for item $n$

- **Reduction**:

  - `'none'`: Per-sample loss
  - `'mean'`: Average over batch
  - `'sum'`: Sum over batch

### Additional Notes

- **Class Weights**: Pass a weight tensor to emphasize/ignore specific classes.
- **ignore_index**: Loss is not computed for targets equal to `ignore_index`.
- **Input**: Should be raw logits (no softmax applied).
