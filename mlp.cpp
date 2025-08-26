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
    size_t n_hidden = 16;  // Number of hidden units (4 states per sampled pair)
    size_t n_out = 4;  // Output units (e.g., XOR has 4 samples by 1 output)
};

// Model optimization
struct SGDParams {
    float lr = 0.01f;  // Learning rate (gamma)
    float weight_decay = 0.0f;  // L2 regularization (lambda)
    float momentum = 0.0f;  // Momentum coefficient (mu)
    float dampening = 0.0f;  // Dampening coefficient (tau)
    bool nesterov = false;  // Nesterov acceleration
    bool maximize = false;  // Minimize or maximize loss
};

// Model layers
struct MLPLayer {
    std::vector<float> W;  // Weights (n_out x n_in)
    std::vector<float> b;  // Biases (n_out)
    std::vector<float> a;  // post-activation (sigmoid(z))
    std::vector<float> d;  // delta (δ_n = ε_n * a_n​)
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
            float nd = (float) rand() / (float) RAND_MAX;  // [0, 1]
            L->W[j] = nd * a;  // scale by normal dist
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
    return 1.0f / (1.0f + expf(-x));
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
        L->a = mlp->y;

        // Copy the output of the current layer to the input
        x = mlp->y;
    }
}

/** @} */

/**
 * Backward propagation (training)
 * @{
 */

// Derivative of sigmoid for backpropagation
float sigmoid_prime(float x) {
    return x * (1.0f - x);
}

void sigmoid_prime_vector(float* v, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        v[i] = sigmoid_prime(v[i]);
    }
}

// reduction as mean
float mse(float* y_pred, float* y_true, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / n;
}

void sgd(float* w, const float* grad, size_t n, float lr, float weight_decay) {
    for (size_t i = 0; i < n; i++) {
        float g = grad[i] + weight_decay * w[i];  // add L2 penalty if needed
        w[i] -= lr * g;
    }
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

    // Assume single input and target vector for now
    std::vector<float> y_pred = mlp.y;  // original predicitions

    // Ensure the output dimensions match
    assert(y_true.size() == y_pred.size());

    float loss = mse(mlp.y.data(), y_true.data(), mlp.y.size());
    printf("Initial loss: %.6f\n", (double) loss);

    // Compute output layer deltas/gradients
    size_t last_layer = mlp.dim.n_layers - 1;
    size_t last_dim = mlp_layer_dim_out(&mlp, last_layer);
    struct MLPLayer* L_last = &mlp.layers[last_layer];
    L_last->d.resize(last_dim);
    for (size_t i = 0; i < last_dim; i++) {
        // Delta output layer: δ_i = (a_i - y_i) ⋅ σ'(a_i)
        float a = L_last->a[i];  // post-activation
        float y = y_true[i];  // target
        L_last->d[i] = (a - y) * sigmoid_prime(a);
    }

    // Back-propagate hidden layer deltas/gradients
    for (int l = last_layer - 1; l >= 0; l--) {
        // Current hidden layer
        struct MLPLayer* L = &mlp.layers[l];
        // Next hidden layer
        struct MLPLayer* L_next = &mlp.layers[l + 1];

        // Get the current hidden dimensions
        size_t n_out = mlp_layer_dim_out(&mlp, l);
        // Get the next hidden dimensions
        size_t n_next = mlp_layer_dim_out(&mlp, l + 1);

        // Resize to the current hidden layer
        L->d.resize(n_out);

        for (size_t i = 0; i < n_out; i++) {
            float sum = 0.0f;

            for (size_t j = 0; j < n_next; j++) {
                // Weight from i (this layer) to j (next layer)
                sum += L_next->W[j * n_out + i] * L_next->d[j];
            }

            L->d[i] = sum * sigmoid_prime(L->a[i]);
        }
    }

    // Update weights and biases
    for (size_t i = 0; i < mlp.dim.n_layers; i++) {
        struct MLPLayer* L = &mlp.layers[i];

        size_t n_in = mlp_layer_dim_in(&mlp, i);
        size_t n_out = mlp_layer_dim_out(&mlp, i);

        std::vector<float>* a_prev = (i == 0) ? &mlp.x : &mlp.layers[i - 1].a;

        for (size_t j = 0; j < n_out; j++) {
            for (size_t k = 0; k < n_in; k++) {
                L->W[j * n_in + k] -= mlp.opt.lr * L->d[j] * (*a_prev)[k];
            }
            L->b[j] -= mlp.opt.lr * L->d[j];
        }
    }

    return 0;
}
