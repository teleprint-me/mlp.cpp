/**
 * Copyright © 2025 Austin Berrio
 * @file mlp/mlp.h
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

#ifndef MLP_H
#define MLP_H

#include <cstdlib>
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
    bool nesterov = false;  // Nesterov acceleration
};

// Model layers
struct MLPLayer {
    std::vector<float> W;  // Weights (n_out x n_in)
    std::vector<float> b;  // Biases (n_out)

    std::vector<float> a;  // post-activation (sigmoid(z))
    std::vector<float> d;  // delta (δ_n = ε_n * a_n​)

    std::vector<float> vW;  // Weight velocity (momentum)
    std::vector<float> vb;  // Biase velocity (momentum)
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
size_t mlp_layer_dim_in(struct MLP* mlp, size_t layer);

// Get layer dimension out
size_t mlp_layer_dim_out(struct MLP* mlp, size_t layer);

/** @} */

/**
 * Logging
 * @{
 */

void mlp_log_dims(struct MLP* mlp);

void mlp_log_opts(struct MLP* mlp);

void mlp_log_vector(const char* title, const float* x, size_t n);

// Print a row-major matrix (rows x cols)
void mlp_log_matrix(const char* title, const float* W, int rows, int cols);

void mlp_log_layers(struct MLP* mlp);

/** @} */

/**
 * Initialization
 * @{
 */

void mlp_init_input(struct MLP* mlp, float* x_in, size_t n);

void mlp_init_input_random(struct MLP* mlp);

// Returns a sample from N(-1, 1)
float rand_uniform(size_t n_in, size_t n_out);

// Returns a sample from N(0, 1)
float rand_normal(size_t n_in, size_t n_out);

void mlp_init_xavier(struct MLP* mlp);

/** @} */

/**
 * Forward pass (inference)
 * @{
 */

// Sigmoid Activation Function
float sigmoid(float x);

// Apply row-major matrix multiplication (y = Wx + b)
void matmul(float* y, float* W, float* x, float* b, size_t n_out, size_t n_in);

void mlp_forward(struct MLP* mlp, float* x_in, size_t n);

/** @} */

/**
 * Backward pass (training)
 * @{
 */

// Derivative of sigmoid for backpropagation
float sigmoid_prime(float x);

// Compute the multi-layer gradients (aka deltas)
// Each layer’s deltas are calculated using the deltas from the next layer.
// The shape of deltas always matches the number of outputs for that layer.
void mlp_compute_gradients(struct MLP* mlp, float* y_true, size_t n);

// Update weights and biases
void mlp_update_params(struct MLP* mlp);

// loss with reduction as mean
float mse(float* y_pred, float* y_true, size_t n);

/** @} */

#endif  // MLP_H
