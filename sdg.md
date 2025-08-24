---
title: 'stochastic gradient descent'
type: 'algorithm'
version: '1'
date: '2025-08-24'
license: 'cc-by-nc-sa-4.0'
---

# Stochastic Gradient Descent

## References

- [On the importance of initialization and momentum in deep learning](https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)

## SDG Algorithm

Implements stochastic gradient descent (optionally with momentum).

**Input**:

- $\gamma$ — learning rate (lr)
- $\theta_0$ — initial parameters
- $f(\theta)$ — objective function
- $\lambda$ — weight decay (L2 regularization)
- $\mu$ — momentum
- $\tau$ — dampening
- **Optional**: `nesterov` (boolean), `maximize` (boolean)

**Initialize**:

$$
b_0 \leftarrow 0 \quad \text{(momentum buffer, if momentum is used)}
$$

**For** $t = 1, 2, \dots$:

1. **Compute gradient**:

$$
g_t \leftarrow \nabla_{\theta} f_t(\theta_{t-1})
$$

If maximizing instead of minimizing:

$$
g_t \leftarrow -g_t
$$

2. **Apply weight decay (L2 regularization)** (if $\lambda \neq 0$):

$$
g_t \leftarrow g_t + \lambda \, \theta_{t-1}
$$

3. **Apply momentum (if $\mu \neq 0$)**:

- **If $t>1$**:

$$
b_t \leftarrow \mu \, b_{t-1} + (1 - \tau) g_t
$$

- **Else**:

$$
b_t \leftarrow g_t
$$

4. **Apply Nesterov accelerated gradient (if `nesterov=True`)**:

$$
g_t \leftarrow g_t + \mu \, b_t
$$

Otherwise:

$$
g_t \leftarrow b_t
$$

5. **Update parameters**:

$$
\theta_t \leftarrow
\begin{cases}
\theta_{t-1} + \gamma g_t, & \text{if maximize=True} \\
\theta_{t-1} - \gamma g_t, & \text{otherwise}
\end{cases}
$$

**Return**: $\theta_t$
