/**
 * Copyright © 2025 Austin Berrio
 * @file mlp/mlp.cpp
 * @brief Multi-layer perceptron implementation in C with minimal C++ features.
 * - No classes or templates used
 * - auto keyword is banned
 * - minimal vector usage
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
    size_t n_layers = 3;  // Number of hidden layers
    size_t n_in = 4;  // Input features
    size_t n_hidden = 8;  // Hidden units
    size_t n_out = 2;  // Output units
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
    std::vector<float> W;  // Output layer weights (n_out x n_in)
    std::vector<float> b;  // Output layer biases (n_out)
};

// Layer cache
struct MLPLayerCache {
    std::vector<float> z;  // pre-activation (Wx + b)
    std::vector<float> a;  // post-activation (sigmoid(z))
};

// Model
struct MLP {
    // Model layers
    std::vector<struct MLPLayer> layers;

    // Layer cache
    std::vector<struct MLPLayerCache> cache;

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
 * Logging
 * @{
 */

void mlp_log_input(struct MLP* mlp) {
    for (size_t i = 0; i < mlp->x.size(); i++) {
        printf("mlp.x[%zu] = %.6f\n", i, (double) mlp->x[i]);
    }
    printf("\n");  // pad output
}

void mlp_log_output(struct MLP* mlp) {
    for (size_t i = 0; i < mlp->y.size(); i++) {
        printf("mlp.y[%zu] = %.6f\n", i, (double) mlp->y[i]);
    }
    printf("\n");  // pad output
}

void mlp_log_weights_and_biases(struct MLP* mlp) {
    // Output initialized weights and biases
    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        struct MLPLayer* L = &mlp->layers[i];

        printf("Layer %zu:\n", i);

        for (size_t j = 0; j < L->W.size(); j++) {
            printf("  W[%zu] = %.6f\n", j, (double) L->W[j]);
        }

        for (size_t j = 0; j < L->b.size(); j++) {
            printf("  b[%zu] = %.6f\n", j, (double) L->b[j]);
        }

        printf("\n");
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

        // Current layer dimensions
        size_t n_in = (i == 0) ? dim->n_in : dim->n_hidden;
        size_t n_out = (i == dim->n_layers - 1) ? dim->n_out : dim->n_hidden;

        // Layer dimensions
        size_t W_d = n_in * n_out;  // Weights (n_in x n_out)
        size_t b_d = n_out;  // Biases (n_out)

        // Initialize weights and biases
        L->W.resize(W_d);
        L->b.resize(b_d);

        // Scaling factor
        float a = sqrtf(6.0f / (n_in + n_out));

        // Initialize weights
        for (size_t j = 0; j < W_d; j++) {
            float rd = 2.0f * ((float) rand() / (float) RAND_MAX) - 1.0f;
            L->W[j] = rd * a;  // [-a, +a] range
        }

        // Initialize biases
        for (size_t j = 0; j < b_d; j++) {
            float rd = 2.0f * ((float) rand() / (float) RAND_MAX) - 1.0f;
            L->b[j] = rd;  // Can be 0 or small real value
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

// Create a simple matmul function (y = Wx + b)
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

    // Initialize cached layers
    mlp->cache.resize(dim->n_layers);

    // Apply the forward pass
    for (size_t i = 0; i < dim->n_layers; i++) {
        struct MLPLayer* L = &mlp->layers[i];
        struct MLPLayerCache* C = &mlp->cache[i];

        // Current layer dimensions
        size_t n_in = (i == 0) ? dim->n_in : dim->n_hidden;
        size_t n_out = (i == dim->n_layers - 1) ? dim->n_out : dim->n_hidden;

        // Resize the output vector
        mlp->y.resize(n_out);

        // Ensure the weights and biases are correctly sized
        assert(L->W.size() == n_in * n_out);
        assert(L->b.size() == n_out);

        // Apply matrix multiplication
        matmul(mlp->y.data(), L->W.data(), x.data(), L->b.data(), n_out, n_in);
        C->z = mlp->y;  // cache pre-activation

        // Apply activation function
        sigmoid_vector(mlp->y.data(), n_out);
        C->a = mlp->y;  // cache post-activation

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
    for (size_t i = 0; i < n; ++i) {
        float g = grad[i] + weight_decay * w[i];  // add L2 penalty if needed
        w[i] -= lr * g;
    }
}

/** @} */

int main(void) {
    srand(time(NULL));  // Seed random number generator

    // Initialize model
    MLP mlp{};

    // Initialize input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

    // Input vector
    mlp_init_input_random(&mlp);

    // Output initialized input vector
    mlp_log_input(&mlp);

    // Initialize model layers
    mlp_init_xavier(&mlp);

    // Output initialized weights and biases
    mlp_log_weights_and_biases(&mlp);

    // Execute the forward pass
    mlp_forward(&mlp, mlp.x.data(), mlp.x.size());

    // Output results
    mlp_log_output(&mlp);

    return 0;
}
