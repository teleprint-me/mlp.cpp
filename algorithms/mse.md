---
title: 'mean squared error'
type: 'algorithm'
version: '1'
date: '2025-08-24'
license: 'cc-by-nc-sa-4.0'
---

# Mean Squared Error (MSE) Criterion

This criterion measures the mean squared error (squared L2 norm) between each
element in the input `x` and the target `y`.

#### **Unreduced Loss (`reduction='none'`):**

For input tensors `x` and `y` of any shape (each containing `N` elements):

$$
\ell(x, y) = L = [l_1, \ldots, l_N]^T, \quad l_n = (x_n - y_n)^2
$$

#### **Reduction Modes:**

- **`reduction='mean'` (default):**

  - Returns the mean of all element-wise losses:

    $$
    \ell(x, y) = \frac{1}{N} \sum_{n=1}^{N} (x_n - y_n)^2
    $$

- **`reduction='sum'`:**

  - Returns the sum of all element-wise losses:

    $$
    \ell(x, y) = \sum_{n=1}^{N} (x_n - y_n)^2
    $$

- **`reduction='none'`:**

  - Returns the vector of element-wise squared differences:

    $$
    \ell(x, y) = [(x_1 - y_1)^2, \ldots, (x_N - y_N)^2]^T
    $$

**Note:**

- The mean reduction divides by the total number of elements `N`.
- To avoid division by `N`, set `reduction='sum'`.
