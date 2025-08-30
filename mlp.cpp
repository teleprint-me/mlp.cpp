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

/**
 * Structures
 * @{
 */

// Model dimensions
struct MLPParams {
    size_t n_layers = 3;  // Number of hidden layers
    size_t n_in = 2;  // Input features (e.g., XOR has 4 samples by 2 inputs)
    size_t n_hidden = 3;  // Number of hidden units (4 states per sampled pair)
    size_t n_out = 1;  // Output units (e.g., XOR has 4 samples by 1 output)
    size_t seed = 1337;  // Random seed for reproducibility
    float bias = 0.0f;  // Initial tensor biases
};

// Model optimization
struct SGDParams {
    size_t epochs = 1000;  // Training cycles
    size_t log_every = 100;  // Log epoch every n cycles
    float tolerance = 1e-3;  // Stop loss
    float lr = 1e-1f;  // Learning rate (gamma)
    float weight_decay = 0.0f;  // L2 regularization (lambda)
    float momentum = 0.0f;  // Momentum coefficient (mu)
    float dampening = 0.0f;  // Dampening coefficient (tau)
    bool nesterov = false;  // @todo Nesterov acceleration
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
    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        // Get the current input layer
        struct MLPLayer* L = &mlp->layers[i];

        // Get the current input dimension
        size_t n_in = mlp_layer_dim_in(mlp, i);  // column
        // Get the current output dimension
        size_t n_out = mlp_layer_dim_out(mlp, i);  // row

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

        // Apply stochastic gradient descent
#pragma omp parallel for
        for (size_t j = 0; j < n_out; j++) {
            // Update weights
            for (size_t k = 0; k < n_in; k++) {
                // Current parameter (θ)
                size_t idx = j * n_in + k;
                // Base gradient (δL / δW)
                float g = L->d[j] * a[k];
                // Sanity check
                assert(!std::isnan(g) && !std::isinf(g));

                // L2 regularization (g + λW)
                if (mlp->opt.weight_decay > 0) {
                    g += mlp->opt.weight_decay * L->W[idx];
                }

                // Apply momentum
                if (mlp->opt.momentum > 0) {
                    // (1 - τ) * g
                    g *= (1.0f - mlp->opt.dampening);

                    // μv + g
                    L->vW[idx] = mlp->opt.momentum * L->vW[idx] + g;

                    if (mlp->opt.nesterov) {
                        // g + μv
                        g += mlp->opt.momentum * L->vW[idx];
                    } else {
                        // g = v
                        g = L->vW[idx];
                    }
                }

                // θ - γg
                L->W[idx] -= mlp->opt.lr * g;
            }

            // Update biases
            float db = L->d[j];
            // Sanity check
            assert(!std::isnan(db) && !std::isinf(db));

            if (mlp->opt.momentum > 0) {
                // (1 - τ) * g
                db *= (1.0f - mlp->opt.dampening);

                // μv + g
                L->vb[j] = mlp->opt.momentum * L->vb[j] + db;

                if (mlp->opt.nesterov) {
                    // g + μv
                    db += mlp->opt.momentum * L->vb[j];
                } else {
                    // g = v
                    db = L->vb[j];
                }
            }

            // θ - γg
            L->b[j] -= mlp->opt.lr * db;
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

void print_usage(struct MLP* mlp, const char* prog) {
    const char options[] = "[--seed N] [--layers N] [--hidden N] [--epochs N] [--lr F] [...]";
    printf("Usage: %s %s\n", prog, options);
    printf("  --seed      N    Random seed (default: %zu)\n", mlp->dim.seed);
    printf("  --bias      F    Initial bias (default: %f)\n", (double) mlp->dim.bias);
    printf("  --layers    N    Number of layers (default: %zu)\n", mlp->dim.n_layers);
    printf("  --hidden    N    Hidden units per layer (default: %zu)\n", mlp->dim.n_hidden);
    printf("  --epochs    N    Training epochs (default: %zu)\n", mlp->opt.epochs);
    printf("  --log-every N    Log every N epochs (default: %zu)\n", mlp->opt.log_every);
    printf("  --lr        F    Learning rate (default: %f)\n", (double) mlp->opt.lr);
    printf("  --tolerance F    Stop loss (default: %f)\n", (double) mlp->opt.tolerance);
    printf("  --decay     F    L2 regularization (default: %f)\n", (double) mlp->opt.weight_decay);
    printf("  --momentum  F    Momentum coefficient (default: %f)\n", (double) mlp->opt.momentum);
    printf("  --dampening F    Momentum coefficient (default: %f)\n", (double) mlp->opt.dampening);
}

int main(int argc, const char* argv[]) {
    // Create the model
    MLP mlp{};

    // Simple manual CLI parse
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            mlp.dim.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bias") == 0 && i + 1 < argc) {
            mlp.dim.bias = atof(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            mlp.dim.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            mlp.dim.n_hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            mlp.opt.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            mlp.opt.log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            mlp.opt.lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            mlp.opt.tolerance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--decay") == 0 && i + 1 < argc) {
            mlp.opt.weight_decay = atof(argv[++i]);
        } else if (strcmp(argv[i], "--momentum") == 0 && i + 1 < argc) {
            mlp.opt.momentum = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dampening") == 0 && i + 1 < argc) {
            mlp.opt.dampening = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(&mlp, argv[0]);
            return 0;
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            print_usage(&mlp, argv[0]);
            return 1;
        }
    }

    if (mlp.dim.n_layers < 3) {
        fprintf(stderr, "[ERROR] Minimum layers = 3\n\t(┛ಠ_ಠ)┛彡┻━┻\n");
        return 1;
    }

    if (mlp.dim.n_hidden < 2) {
        fprintf(stderr, "[ERROR] Minimum hidden dims = 2\n\t┻━┻︵ \\(°□°)/ ︵ ┻━┻\n");
        return 1;
    }

    printf("╔══════════════════════════════╗\n");
    printf("║  MLP XOR Lab  (by Austin)    ║\n");
    printf("╚══════════════════════════════╝\n");

    // Log model parameters
    mlp_log_dims(&mlp);
    mlp_log_opts(&mlp);

    // Seed random number generator
    if (mlp.dim.seed > 0) {
        srand(mlp.dim.seed);
    } else {
        srand(time(NULL));
    }

    // Create input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

    // Apply xavier-glorot initialization to model layers
    mlp_init_xavier(&mlp);
    // Do a sanity check when initializing the model
    mlp_log_layers(&mlp);

    // XOR dataset: 4 samples, each sample has 2 inputs, 1 output
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
    };
    std::vector<std::vector<float>> outputs = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f},
    };

    // For per-epoch stats
    std::vector<float> y_pred(mlp.dim.n_out);

    // Execute training loop
    for (size_t epoch = 0; epoch < mlp.opt.epochs; epoch++) {
        float loss_epoch = 0.0f;

        // For each XOR sample
        for (size_t i = 0; i < inputs.size(); i++) {
            // Set the current inputs
            mlp.x = inputs[i];

            // Set the current outputs
            std::vector<float> &y_true = outputs[i];

            // Compute forward propagation (predictions)
            mlp_forward(&mlp, mlp.x.data(), mlp.x.size());

            // Compute loss per sample
            float loss = mse(mlp.y.data(), y_true.data(), y_true.size());
            loss_epoch += loss;

            // Compute backward propagation (gradients)
            mlp_compute_gradients(&mlp, y_true.data(), y_true.size());

            // Update parameters using computed gradients
            mlp_update_params(&mlp);
        }

        // Average loss over all samples
        loss_epoch /= inputs.size();

        // Log every n epochs
        if (epoch % mlp.opt.log_every == 0 || loss_epoch < mlp.opt.tolerance) {
            printf("epoch[%zu] Σ(-᷅_-᷄๑) = %f\n", epoch, (double) loss_epoch);
        }

        // Stop loss
        if (loss_epoch < mlp.opt.tolerance) {
            printf("epoch[%zu] (◕‿◕✿) (loss < %f)\n", epoch, (double) mlp.opt.tolerance);
            break;
        }
    }

    // Log predictions
    printf("\n-=≡Σ<|°_°|>:\n");
    for (size_t i = 0; i < inputs.size(); i++) {
        mlp.x = inputs[i];
        mlp_forward(&mlp, mlp.x.data(), mlp.x.size());
        printf(
            "Input: [%f %f] -> Predicted: %f, Target: %f\n",
            (double) inputs[i][0],
            (double) inputs[i][1],
            (double) mlp.y[0],
            (double) outputs[i][0]
        );
    }

    return 0;
}
