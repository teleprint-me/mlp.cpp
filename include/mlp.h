/**
 * @file    mlp/mlp.h
 * @brief   Multi-layer perceptron (MLP) in C with minimal C++ features.
 * @author  Austin Berrio
 * @copyright Copyright © 2025
 *
 * Coding rules:
 *   - No classes/templates; use C-style structs.
 *   - No 'auto'; all types explicit.
 *   - Prefer std::vector for storage.
 *   - Explicit, C-style function signatures.
 *   - Pointer args: mutable unless const.
 *   - Simplicity first; abstraction only as needed.
 */

#ifndef MLP_H
#define MLP_H

#include <cstdlib>
#include <vector>

/** @defgroup MLPStructs MLP Data Structures
 *  @{
 */

/** @brief Model dimensions and basic configuration. */
struct MLPParams {
    size_t n_layers = 3;  ///< Number of layers (hidden + output)
    size_t n_in = 2;  ///< Input features
    size_t n_hidden = 3;  ///< Hidden units per layer
    size_t n_out = 1;  ///< Output units
    size_t seed = 1337;  ///< RNG seed for reproducibility
    float bias = 0.0f;  ///< Initial bias value
};

/** @brief Stochastic Gradient Descent (SGD) optimizer settings. */
struct SGDParams {
    size_t epochs = 1000;  ///< Training epochs
    size_t log_every = 100;  ///< Logging interval (in epochs)
    float tolerance = 1e-3;  ///< Stop if loss <= tolerance
    float lr = 1e-1f;  ///< Learning rate (gamma)
    float weight_decay = 0.0f;  ///< L2 regularization (lambda)
    float momentum = 0.0f;  ///< SGD momentum
    float dampening = 0.0f;  ///< Dampening for momentum
    bool nesterov = false;  ///< Enable Nesterov acceleration
};

/** @brief Parameters, activations, and velocity buffers for one layer. */
struct MLPLayer {
    std::vector<float> W;  ///< Weights (row-major: n_out x n_in)
    std::vector<float> b;  ///< Biases (n_out)
    std::vector<float> a;  ///< Activations (post-activation, e.g., sigmoid)
    std::vector<float> d;  ///< Deltas (gradients)
    std::vector<float> vW;  ///< Velocity (momentum) for weights
    std::vector<float> vb;  ///< Velocity (momentum) for biases
};

/** @brief Main MLP model object. */
struct MLP {
    std::vector<MLPLayer> layers;  ///< Layers (n_layers)
    std::vector<float> x;  ///< Current input vector (n_in)
    std::vector<float> y;  ///< Current output vector (n_out)
    MLPParams dim{};  ///< Model dimensions
    SGDParams opt{};  ///< Optimizer parameters
};

/** @} */

/** @defgroup MLPUtils Utility Functions
 *  @{
 */

/** @brief Get number of input features for a given layer. */
size_t mlp_layer_dim_in(struct MLP* mlp, size_t layer);

/** @brief Get number of output features for a given layer. */
size_t mlp_layer_dim_out(struct MLP* mlp, size_t layer);

/** @} */

/** @defgroup MLPLogging Logging and Debugging
 *  @{
 */

/** @brief Print model dimensions. */
void mlp_log_dims(struct MLP* mlp);

/** @brief Print optimizer parameters. */
void mlp_log_opts(struct MLP* mlp);

/** @brief Print vector with label. */
void mlp_log_vector(const char* title, const float* x, size_t n);

/** @brief Print matrix (row-major: rows x cols). */
void mlp_log_matrix(const char* title, const float* W, int rows, int cols);

/** @brief Print weights and biases for all layers. */
void mlp_log_layers(struct MLP* mlp);

/** @} */

/** @defgroup MLPInit Model Initialization
 *  @{
 */

/** @brief Initialize weights/biases (Glorot/Xavier init). */
void mlp_init_xavier(struct MLP* mlp);

/** @} */

/** @defgroup MLPForward Forward Pass
 *  @{
 */

/** @brief Sigmoid activation function. */
float sigmoid(float x);

void softmax(float* x, int n);

/** @brief Matrix-vector multiply (row-major): y = Wx + b. */
void matmul(float* y, float* W, float* x, float* b, size_t n_out, size_t n_in);

/** @brief Compute model output from input. */
void mlp_forward(struct MLP* mlp, float* x_in, size_t n);

/** @} */

/** @defgroup MLPBackward Backward Pass and Training
 *  @{
 */

/** @brief Derivative of sigmoid (for backprop). */
float sigmoid_prime(float x);

/** @brief Compute output and hidden layer gradients. */
void mlp_compute_gradients(struct MLP* mlp, float* y_true, size_t n);

/** @brief Update weights and biases using gradients/optimizer. */
void mlp_update_params(struct MLP* mlp);

/** @brief Mean squared error (reduction=mean). */
float mse(float* y_pred, float* y_true, size_t n);

// y_pred: predicted probabilities (softmax output), shape (n,)
// y_true: target one-hot vector, shape (n,)
// n: number of classes
float cross_entropy(const float* y_pred, const float* y_true, size_t n);

std::vector<float> one_hot_encode(int label, int n_classes);

/** @} */

#endif  // MLP_H
