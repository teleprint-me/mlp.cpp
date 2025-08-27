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
#include <vector>

/**
 * Structures
 * @{
 */

// Model dimensions
struct MLPParams {
    size_t seed = 1337;  // Random seed for reproducibility
    size_t n_layers = 3;  // Number of hidden layers
    size_t n_in = 8;  // Input features (e.g., XOR has 4 samples by 2 inputs)
    size_t n_hidden = 8;  // Number of hidden units (4 states per sampled pair)
    size_t n_out = 4;  // Output units (e.g., XOR has 4 samples by 1 output)
};

// Model optimization
struct SGDParams {
    float lr = 0.01f;  // Learning rate (gamma)
    float weight_decay = 0.0f;  // L2 regularization (lambda)
    float momentum = 0.0f;  // Momentum coefficient (mu)
    float dampening = 0.0f;  // Dampening coefficient (tau)
    bool nesterov = false;  // Nesterov acceleration
};

// Model layers
struct MLPLayer {
    std::vector<float> W;  // Weights (n_out x n_in)
    std::vector<float> b;  // Biases (n_out)

    std::vector<float> a;  // post-activation (sigmoid(z))
    std::vector<float> d;  // delta (δ_n = ε_n * a_n​)

    std::vector<float> vW;  // Weights momentum
    std::vector<float> vb;  // Biases momentum
};

// Model
struct MLP {
    // Model layers
    std::vector<struct MLPLayer> layers;

    // Input/output tensors
    std::vector<float> x;  // 1D input vector
    std::vector<float> y;  // 1D output vector

    // Model dimensions
    struct MLPParams dim{};

    // Model optimization
    struct SGDParams opt{};
};

/** @} */

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

        // Calculate the scaling factor
        float a = sqrtf(6.0f / (n_in + n_out));

        // Initialize weights
        for (size_t j = 0; j < W_d; j++) {
            float ud = 2 * ((float) rand() / (float) RAND_MAX) - 1;  // uniform
            L->W[j] = ud * a;  // [-a, +a]
        }

        // Initialize biases
        for (size_t j = 0; j < b_d; j++) {
            L->b[j] = 1e-5f;  // Can be 0 or small real value
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
void mlp_compute_gradients(struct MLP* mlp, float* y_true) {
    // Get the final layer index
    size_t last_layer = mlp->dim.n_layers - 1;
    // Get the final layer dimension
    size_t last_dim = mlp_layer_dim_out(mlp, last_layer);
    // Get the final layer
    struct MLPLayer* L_last = &mlp->layers[last_layer];

    // Initialize the output deltas
    L_last->d.resize(last_dim);

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
    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        // Get the current input layer
        struct MLPLayer* L = &mlp->layers[i];

        // Get the current input dimension (col)
        size_t n_in = mlp_layer_dim_in(mlp, i);
        // Get the current output dimension (row)
        size_t n_out = mlp_layer_dim_out(mlp, i);

        // Get the previous activation
        std::vector<float> &a = (i == 0) ? mlp->x : mlp->layers[i - 1].a;

        // Only initialize moment if it's set
        if (mlp->opt.momentum > 0) {
            // Initialize weight momentum
            if (L->vW.size() != L->W.size()) {
                L->vW.assign(L->W.size(), 0.0f);
            }

            // Initialize bias momentum
            if (L->vb.size() != L->b.size()) {
                L->vb.assign(L->b.size(), 0.0f);
            }
        }

        // before updating parameters
        constexpr float GRAD_EPS = 1e-15f;

        // Apply stochastic gradient descent
        // #pragma omp parallel for
        for (size_t j = 0; j < n_out; j++) {
            // Update the weights
            for (size_t k = 0; k < n_in; k++) {
                // Get the current parameter
                size_t idx = j * n_in + k;
                // Calculate the base gradient
                float g = L->d[j] * a[k];

                // Apply weight decay
                if (mlp->opt.weight_decay > 0) {
                    g += mlp->opt.weight_decay * L->W[idx];
                }

                // Apply dampening if set
                g *= (1.0f - mlp->opt.dampening);

                if (!(fabsf(g) > GRAD_EPS) || std::isnan(g)) {
                    printf(
                        "Warning: small or NaN gradient detected: %f at idx %zu\n", (double) g, idx
                    );
                    // assert(0 && "Gradient vanished or NaN!");
                }

                // Apply momentum
                if (mlp->opt.momentum > 0) {
                    L->vW[idx] = mlp->opt.momentum * L->vW[idx] + g;
                    L->W[idx] -= mlp->opt.lr * L->vW[idx];
                } else {  // Otherwise, update
                    L->W[idx] -= mlp->opt.lr * g;
                }
            }

            // Update the biases
            float db = (1.0f - mlp->opt.dampening) * L->d[j];

            if (!(fabsf(db) > GRAD_EPS) || std::isnan(db)) {
                printf(
                    "Warning: small or NaN bias gradient detected: %f at idx %zu\n", (double) db, j
                );
                // assert(0 && "Gradient vanished or NaN!");
            }

            if (mlp->opt.momentum > 0) {
                L->vb[j] = mlp->opt.momentum * L->vb[j] + db;
                L->b[j] -= mlp->opt.lr * L->vb[j];
            } else {
                L->b[j] -= mlp->opt.lr * db;
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

int main(void) {
    /**
     * Initialize the model
     */

    // Create the model
    MLP mlp{};

    // Seed random number generator
    if (mlp.dim.seed > 0) {
        srand(mlp.dim.seed);
    } else {
        srand(time(NULL));
    }

    // Create input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

    // Expected XOR inputs (n_samples, n_in) = 4 * 2 = 8
    // The input vector must be flat!
    std::vector<float> inputs = {
        // [0, 0]
        0.0f,
        0.0f,
        // [0, 1]
        0.0f,
        1.0f,
        // [1, 0]
        1.0f,
        0.0f,
        // [1, 1]
        1.0f,
        1.0f
    };

    // Randomly initialize the input vector
    mlp_init_input(&mlp, inputs.data(), inputs.size());

    // Log initialized input vector
    mlp_log_matrix("x", mlp.x.data(), mlp.dim.n_out, mlp.dim.n_in / mlp.dim.n_out);

    // Apply xavier-glorot initialization to model layers
    mlp_init_xavier(&mlp);

    // Log initialized weights and biases
    mlp_log_layers(&mlp);

    /**
     * Perform a forward pass
     */

    // Execute the forward pass
    mlp_forward(&mlp, mlp.x.data(), mlp.x.size());

    // Output results
    mlp_log_vector("y", mlp.y.data(), mlp.y.size());

    /**
     * Perform a backward pass
     * A gradient is the error weighted by the derivative of the
     * activation function at the output.
     */

    // Expected XOR outputs (n_samples * n_out) = 4 * 1 = 4
    std::vector<float> y_true = {0.0f, 1.0f, 1.0f, 0.0f};

    // Ensure the output dimensions match
    assert(mlp.y.size() == y_true.size());

    float loss_0 = mse(mlp.y.data(), y_true.data(), mlp.y.size());
    printf("loss_0: %.6f\n", (double) loss_0);

    // Compute output layer gradients (aka deltas)
    mlp_compute_gradients(&mlp, y_true.data());

    // Update weights and biases
    mlp_update_params(&mlp);

    mlp_forward(&mlp, mlp.x.data(), mlp.x.size());

    float loss_1 = mse(mlp.y.data(), y_true.data(), mlp.y.size());
    printf("loss_1: %.6f\n", (double) loss_1);
    printf(
        "loss_diff: %.6f - %.6f = %.6f\n\n",
        (double) loss_1,
        (double) loss_0,
        (double) (loss_1 - loss_0)
    );

    mlp_log_layers(&mlp);
    mlp_log_vector("y", mlp.y.data(), mlp.y.size());

    return 0;
}
