/**
 * Copyright © 2025 Austin Berrio
 * @file mlp/mlp.cpp
 * @brief Multi-layer perceptron implementation in C with minimal C++ features.
 * - Classes and templates are **not** allowed! C-like structs are preferred.
 *   - Setting default field parameters for `struct` is encouraged.
 * - `auto` keyword usage is **not** allowed!
 *   - Data types must be explicitly declared! Hiding types is discouraged.
 * - Using std::vector is allowed to simplify memory management.
 *   - std::vector usage is preferred to simplify vector and matrix operations.
 * - Function signatures must be explicit and C-like.
 *   - Function parameters must use explicit types and shapes, e.g. f(float* x, size_t n).
 *   - Pointer refs signal the data is mutable to some capacity unless const qualified.
 * - Simplicity rules them all!
 *   - Implementations should prioritize simplicity at all costs.
 *   - Abstractions are deferred until absolutely necessary.
 *   - Abstractions will reveal themselves through prototyping.
 */

#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mlp.h"

/**
 * MLP Utils
 * @{
 */

// Get layer dimension in
size_t mlp_layer_dim_in(struct MLP* mlp, size_t layer) {
    assert(layer < mlp->dim.n_layers);
    return (layer == 0) ? mlp->dim.n_in : mlp->dim.n_hidden;
}

// Get layer dimension out
size_t mlp_layer_dim_out(struct MLP* mlp, size_t layer) {
    assert(layer < mlp->dim.n_layers);
    return (layer == mlp->dim.n_layers - 1) ? mlp->dim.n_out : mlp->dim.n_hidden;
}

/** @} */

/**
 * Logging
 * @{
 */

// LOG(struct, field, format)
#define MLP_LOG_PARAM_SIZE(s, f, fmt) printf("%s->%s = " fmt "\n", #s, #f, (size_t) (s)->f);
#define MLP_LOG_PARAM_FLOAT(s, f, fmt) printf("%s->%s = " fmt "\n", #s, #f, (double) (s)->f);
#define MLP_LOG_PARAM_BOOL(s, f, fmt) \
    printf("%s->%s = " fmt "\n", #s, #f, (int) (s)->f ? "true" : "false");

void mlp_log_dims(struct MLP* mlp) {
    printf("Dimensions:\n");
    struct MLPParams* dim = &mlp->dim;
    MLP_LOG_PARAM_SIZE(dim, n_layers, "%zu");
    MLP_LOG_PARAM_SIZE(dim, n_in, "%zu");
    MLP_LOG_PARAM_SIZE(dim, n_hidden, "%zu");
    MLP_LOG_PARAM_SIZE(dim, n_out, "%zu");
    MLP_LOG_PARAM_SIZE(dim, seed, "%zu");
    MLP_LOG_PARAM_FLOAT(dim, bias, "%f");
    printf("\n");
}

void mlp_log_opts(struct MLP* mlp) {
    printf("Optimizer:\n");
    struct SGDParams* opt = &mlp->opt;
    MLP_LOG_PARAM_SIZE(opt, epochs, "%zu");
    MLP_LOG_PARAM_SIZE(opt, log_every, "%zu");
    MLP_LOG_PARAM_FLOAT(opt, tolerance, "%f");
    MLP_LOG_PARAM_FLOAT(opt, lr, "%f");
    MLP_LOG_PARAM_FLOAT(opt, weight_decay, "%f");
    MLP_LOG_PARAM_FLOAT(opt, momentum, "%f");
    MLP_LOG_PARAM_FLOAT(opt, dampening, "%f");
    MLP_LOG_PARAM_BOOL(opt, nesterov, "%s");
    printf("\n");
}

void mlp_log_vector(const char* title, const float* x, size_t n) {
    printf("%s: ", title);
    printf("(n_out,) = (%zu,)\n", n);
    for (size_t i = 0; i < n; i++) {
        printf("    %6.6f", (double) x[i]);
    }
    printf("\n\n");  // pad output
}

// Print a row-major matrix (rows x cols)
void mlp_log_matrix(const char* title, const float* W, int rows, int cols) {
    printf("%s: ", title);
    printf("(n_out, n_in) = (%d, %d)\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("    %.6f", (double) W[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void mlp_log_layers(struct MLP* mlp) {
    // Output initialized weights and biases
    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        // y = (n_out,)
        // W = (n_out, n_in)
        // x = (n_in,)
        // b = (n_out,)
        struct MLPLayer* L = &mlp->layers[i];
        printf("Layer %zu:\n", i);

        // n_in = input dim = columns
        size_t n_in = mlp_layer_dim_in(mlp, i);
        // n_out = output dim = rows
        size_t n_out = mlp_layer_dim_out(mlp, i);

        // matrix is row-major (n_out, n_in) = (rows, columns)
        mlp_log_matrix("W", L->W.data(), n_out, n_in);
        mlp_log_vector("b", L->b.data(), n_out);
    }
}

/** @} */

/**
 * Initialization
 * @{
 */

void mlp_init_input(struct MLP* mlp, float* x_in, size_t n) {
    assert(n == mlp->dim.n_in);

    mlp->x.resize(n);
    for (size_t i = 0; i < mlp->x.size(); i++) {
        mlp->x[i] = x_in[i];
    }
}

void mlp_init_input_random(struct MLP* mlp) {
    mlp->x.resize(mlp->dim.n_in);
    for (size_t i = 0; i < mlp->x.size(); i++) {
        mlp->x[i] = (float) rand() / (float) RAND_MAX;  // Normalize input
    }
}

// Returns a sample from N(-1, 1)
float rand_uniform(size_t n_in, size_t n_out) {
    float a = sqrtf(6.0f / (n_in + n_out));  // scaling factor
    float ud = 2 * ((float) rand() / (float) RAND_MAX) - 1;  // uniform
    return ud * a;
}

// Returns a sample from N(0, 1)
float rand_normal(size_t n_in, size_t n_out) {
    // Box-Muller transform
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);  // avoid log(0)
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float) M_PI * u2);
    float stddev = sqrtf(2.0f / (n_in + n_out));
    return z0 * stddev;
}

// https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
// https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization
void mlp_init_xavier(struct MLP* mlp) {
    struct MLPParams* dim = &mlp->dim;

    // Initialize model layers
    mlp->layers.resize(dim->n_layers);
    for (size_t i = 0; i < dim->n_layers; i++) {
        // Get the current layer
        struct MLPLayer* L = &mlp->layers[i];

        // Get current layer dimensions
        size_t n_in = mlp_layer_dim_in(mlp, i);
        size_t n_out = mlp_layer_dim_out(mlp, i);

        // Calculate current layer dimensions
        size_t W_d = n_in * n_out;  // Weights (n_in x n_out)
        size_t b_d = n_out;  // Biases (n_out)

        // Resize weights and biases
        L->W.resize(W_d);
        L->b.resize(b_d);

        // Initialize weights
        for (size_t j = 0; j < W_d; j++) {
            L->W[j] = rand_normal(n_in, n_out);
        }

        // Initialize biases
        for (size_t j = 0; j < b_d; j++) {
            L->b[j] = mlp->dim.bias;  // Can be 0 or small real value
        }
    }
}

/** @} */

/**
 * Forward pass (inference)
 * @{
 */

// Sigmoid Activation Function
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));  // creates non-linearity
}

void sigmoid_vector(float* v, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        v[i] = sigmoid(v[i]);
    }
}

// Apply row-major matrix multiplication (y = Wx + b)
void matmul(float* y, float* W, float* x, float* b, size_t n_out, size_t n_in) {
#pragma omp parallel for
    for (size_t i = 0; i < n_out; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < n_in; j++) {
            sum += W[i * n_in + j] * x[j];
        }
        y[i] = sum + b[i];
    }
}

void mlp_forward(struct MLP* mlp, float* x_in, size_t n) {
    struct MLPParams* dim = &mlp->dim;

    // Copy the input vector
    std::vector<float> x(x_in, x_in + n);

    // Apply the forward pass
    for (size_t i = 0; i < dim->n_layers; i++) {
        struct MLPLayer* L = &mlp->layers[i];

        // Current layer dimensions
        size_t n_in = mlp_layer_dim_in(mlp, i);
        size_t n_out = mlp_layer_dim_out(mlp, i);

        // Resize the output vector
        mlp->y.resize(n_out);

        // Ensure the weights and biases are correctly sized
        assert(L->W.size() == n_in * n_out);
        assert(L->b.size() == n_out);

        // Apply matrix multiplication
        matmul(mlp->y.data(), L->W.data(), x.data(), L->b.data(), n_out, n_in);

        // Apply activation function
        sigmoid_vector(mlp->y.data(), n_out);

        // Cache post-activation
        L->a = mlp->y;  // required for chain-rule

        // Copy the output of the current layer to the input
        x = mlp->y;
    }
}

/** @} */

/**
 * Backward propagation (training)
 * @{
 */

/**************************************************
 *      Calculus for Machine Learning Reference    *
 **************************************************
 *
 * 1. Slope of a Line
 * ------------------
 * The slope between points (x₁, y₁) and (x₂, y₂) is:
 *      slope = (y₂ - y₁) / (x₂ - x₁),  where x₂ ≠ x₁
 * - Geometrically: "rise over run" (average rate of change).
 *
 * 2. Derivative at a Point
 * ------------------------
 * The derivative f'(a) gives the instantaneous rate of change of f at x = a:
 *      f'(a) ≈ (f(b) - f(a)) / (b - a), for b near a
 *      f'(a) = lim (b→a) (f(b) - f(a)) / (b - a)
 * - Interpreted as the slope of the tangent at x = a.
 * - If f'(a) > 0, f is increasing at a.
 * - If f'(a) < 0, f is decreasing at a.
 * - |f'(a)| is the magnitude of change.
 *
 * 3. Leibniz Notation
 * -------------------
 * If y = f(x), then:
 *      Δy / Δx → dy/dx = f'(x)
 *      d/dx (y) = derivative of y with respect to x
 *
 * 4. Tangent Line Approximation (Linearization)
 * ---------------------------------------------
 * For small Δx:
 *      Δy ≈ f'(a)·Δx
 *      f(x) ≈ f(a) + f'(a)·(x - a)
 * - Used for quick estimates and numerical updates.
 *
 * 5. The Exponential Rule
 * -----------------------
 * For any positive constant a:
 *      d/dx (a^x) = (ln a)·a^x
 * - Special case: d/dx (e^x) = e^x
 *
 * 6. The Chain Rule (Core of Backpropagation)
 * -------------------------------------------
 * If y = f(z) and z = g(t), then:
 *      dy/dt = (dy/dz)·(dz/dt)
 * - In ML, for composite functions (layered neural nets):
 *      dL/dw = dL/da · da/dz · dz/dw
 *   where:
 *      - L is the loss function
 *      - a is the activation output (e.g. σ(z))
 *      - z is the pre-activation (linear combination)
 *      - w is a weight
 * - For sigmoid activation:
 *      σ(z) = 1 / (1 + exp(-z))
 *      d/dz σ(z) = σ(z)·(1 - σ(z))
 * - For SE (Squared Error) loss:
 *      L = ½(a - y)²
 *      dL/da = (a - y)
 * - Output layer error term (delta):
 *      delta = (a - y) · σ'(z)
 *   This gives the gradient of the loss with respect to z.
 *
 * 7. Batch Aggregation (not part of chain rule)
 * ---------------------------------------------
 * - In ML, we often sum or average loss/gradients over a batch:
 *      MSE = (1/n) ∑ (a_i - y_i)²
 *   This "mean" is separate from the chain rule and does not affect the chain itself.
 *
 **************************************************/

// Derivative of sigmoid for backpropagation
float sigmoid_prime(float x) {
    return x * (1.0f - x);
}

// Compute the multi-layer gradients (aka deltas)
// Each layer’s deltas are calculated using the deltas from the next layer.
// The shape of deltas always matches the number of outputs for that layer.
void mlp_compute_gradients(struct MLP* mlp, float* y_true, size_t n) {
    // Get the final layer index
    size_t last_layer = mlp->dim.n_layers - 1;
    // Get the final layer dimension
    size_t last_dim = mlp_layer_dim_out(mlp, last_layer);
    // Get the final layer
    struct MLPLayer* L_last = &mlp->layers[last_layer];

    // Initialize the output deltas
    L_last->d.resize(last_dim);

    // Ensure the output dimensions match
    // @note mlp.y == L_last->a. They must be equivalent.
    assert(mlp->y.size() == n && "Output dim must match label size");
    assert(L_last->d.size() == n && "Output dim must match label size");

    // Backpropagate output layer deltas
#pragma omp parallel for
    for (size_t i = 0; i < last_dim; i++) {
        // Delta output layer: δ_i = (a_i - y_i) ⋅ σ'(a_i)
        L_last->d[i] = (L_last->a[i] - y_true[i]) * sigmoid_prime(L_last->a[i]);
    }

    // Backpropagate hidden layer deltas
    for (int l = last_layer - 1; l >= 0; l--) {
        // Current hidden layer
        struct MLPLayer* L = &mlp->layers[l];
        // Next hidden layer
        struct MLPLayer* L_next = &mlp->layers[l + 1];

        // Get the current output dimension (current row)
        size_t n_out = mlp_layer_dim_out(mlp, l);  // delta shape
        // Get the next output dimension (next row)
        size_t n_out_next = mlp_layer_dim_out(mlp, l + 1);  // next delta

        // Resize to the current hidden layer
        L->d.resize(n_out);

        // Apply the chain rule
#pragma omp parallel for
        for (size_t i = 0; i < n_out; i++) {
            float sum = 0.0f;

            // Use next layer’s weights and deltas to compute the current layer’s deltas.
            for (size_t j = 0; j < n_out_next; j++) {
                // W_T[j * rows + i] = W[i * cols + j];
                // Weight from this layer (i) to next layer (j)
                // W_T = W[j * current_row + i]
                sum += L_next->W[j * n_out + i] * L_next->d[j];
            }

            L->d[i] = sum * sigmoid_prime(L->a[i]);
        }
    }
}

// Update weights and biases
void mlp_update_params(struct MLP* mlp) {
    const float lr = mlp->opt.lr;
    const float mu = mlp->opt.momentum;
    const float tau = mlp->opt.dampening;
    const float lambda = mlp->opt.weight_decay;
    const bool nesterov = mlp->opt.nesterov;

    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        // Get the current input layer
        struct MLPLayer* L = &mlp->layers[i];

        // Get the current input dimension
        size_t n_in = mlp_layer_dim_in(mlp, i);  // column
        // Get the current output dimension
        size_t n_out = mlp_layer_dim_out(mlp, i);  // row

        // Get activations from the previous layer
        std::vector<float> &a = (i == 0) ? mlp->x : mlp->layers[i - 1].a;

        // Only initialize moment if it's set
        if (mu > 0) {  // b_0 <- 0
            // Initialize weight momentum
            if (L->vW.size() != L->W.size()) {
                L->vW.assign(L->W.size(), 0.0f);
            }

            // Initialize bias momentum
            if (L->vb.size() != L->b.size()) {
                L->vb.assign(L->b.size(), 0.0f);
            }
        }

        // Apply stochastic gradient descent
#pragma omp parallel for
        for (size_t j = 0; j < n_out; j++) {
            // Update weights
            for (size_t k = 0; k < n_in; k++) {
                // Current parameter (θ)
                size_t idx = j * n_in + k;
                // Compute gradient (g_t ← ∇_{θ} f_{t} (θ_{t - 1}))
                float gw = L->d[j] * a[k];
                // Sanity check
                assert(!std::isnan(gw) && !std::isinf(gw));

                // Coupled L2 regularization (g_t + λ * θ_{t - 1})
                if (lambda > 0.0f) {
                    gw += lambda * L->W[idx];
                }

                // Apply momentum
                if (mu > 0.0f) {
                    // b_t = μ * b_{t - 1} + (1 - τ) * g_t
                    L->vW[idx] = mu * L->vW[idx] + (1.0f - tau) * gw;

                    // Apply accelerated gradient if enabled
                    if (nesterov) {
                        // g_t = g_t + μ * b_t
                        gw += mu * L->vW[idx];
                    } else {
                        // g_t = b_t
                        gw = L->vW[idx];
                    }
                }

                // θ_t = θ_{t - 1} - γ * g_t
                L->W[idx] -= lr * gw;
            }

            // Update biases
            {
                float gb = L->d[j];
                // Sanity check
                assert(!std::isnan(gb) && !std::isinf(gb));

                // L2 regularization (g_t + λ * θ_{t - 1})
                if (lambda > 0.0f) {
                    gb += lambda * L->b[j];
                }

                if (mu > 0.0f) {
                    // b_t = μ * b_{t - 1} + (1 - τ) * g_t
                    L->vb[j] = mu * L->vb[j] + (1.0f - tau) * gb;

                    if (nesterov) {
                        // g_t = g_t + μ * b_t
                        gb += mu * L->vb[j];
                    } else {
                        // g_t = b_t
                        gb = L->vb[j];
                    }
                }

                // θ_t = θ_{t - 1} - γ * g_t
                L->b[j] -= lr * gb;
            }
        }
    }
}

// loss with reduction as mean
float mse(float* y_pred, float* y_true, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / n;
}

// Transpose a row-major matrix (rows x cols) into (cols x rows)
void transpose(const float* W, float* W_T, int rows, int cols) {
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            W_T[j * rows + i] = W[i * cols + j];
        }
    }
}

/** @} */
