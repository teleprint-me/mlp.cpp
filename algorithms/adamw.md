---
title: 'decay regularization'
type: 'algorithm'
version: '1'
date: '2025-07-22'
license: 'cc-by-nc-sa-4.0'
---

# Decay Regularization

## References

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

## AdamW Algorithm

Implements AdamW algorithm, where weight decay does not accumulate in the
momentum nor variance.

**Input**:

- $\gamma$ — learning rate (lr)
- $\beta_1, \beta_2$ — exponential decay rates for moments (betas)
- $\theta_0$ — initial parameters
- $f(\theta)$ — objective function
- $\epsilon$ — numerical stability (epsilon)
- $\lambda$ — weight decay
- **Optional**: `amsgrad` (boolean), `maximize` (boolean)

**Initialize**:

$$
m_0 \leftarrow 0 \quad \text{(first moment, mean of gradients)} \\
v_0 \leftarrow 0 \quad \text{(second moment, uncentered variance)} \\
v_0^{\text{max}} \leftarrow 0 \quad \text{(only if AMSGrad is used)}
$$

**For** $t = 1, 2, \dots$:

1. **Compute gradient**:

$$
g_t \leftarrow
\begin{cases}
-\nabla_{\theta} f_t(\theta_{t-1}), & \text{if maximize=True} \\
\nabla_{\theta} f_t(\theta_{t-1}), & \text{otherwise}
\end{cases}
$$

2. **Apply decoupled weight decay**:

$$
\theta_t \leftarrow \theta_{t-1} - \gamma \lambda \, \theta_{t-1}
$$

3. **Update biased first moment estimate**:

$$
m_t \leftarrow \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t
$$

4. **Update biased second raw moment estimate**:

$$
v_t \leftarrow \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2
$$

5. **Compute bias-corrected estimates**:

$$
\widehat{m}_t \leftarrow \frac{m_t}{1 - \beta_1^t}, \quad
\widehat{v}_t \leftarrow
\begin{cases}
\frac{v_t^{\text{max}}}{1 - \beta_2^t}, & \text{if AMSGrad=True} \\
\frac{v_t}{1 - \beta_2^t}, & \text{otherwise}
\end{cases}
$$

where:

$$
v_t^{\text{max}} \leftarrow \max(v_{t-1}^{\text{max}}, v_t) \quad (\text{only if AMSGrad=True})
$$

6. **Update parameters**:

$$
\theta_t \leftarrow \theta_t - \gamma \, \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t} + \epsilon}
$$

**Return**: $\theta_t$ (optimized parameters)

For further details regarding the algorithm we refer to Decoupled Weight Decay
Regularization.
