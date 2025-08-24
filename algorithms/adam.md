---
title: 'stochastic optimization'
type: 'algorithm'
version: '1'
date: '2025-07-22'
license: 'cc-by-nc-sa-4.0'
---

# Stochastic Optimization

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## Adam Algorithm

Implements Adam algorithm.

**Input**:

- $\gamma$ — learning rate (lr)
- $\beta_1, \beta_2$ — exponential decay rates for moments (betas)
- $\theta_0$ — initial parameters
- $f(\theta)$ — objective function
- $\lambda$ — weight decay (L2 regularization)
- $\epsilon$ — numerical stability
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

2. **Apply weight decay (if $\lambda \neq 0$)**:

$$
g_t \leftarrow g_t + \lambda \, \theta_{t-1}
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
\theta_t \leftarrow \theta_{t-1} - \gamma \, \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t} + \epsilon}
$$

**Return**: $\theta_t$ (optimized parameters)
