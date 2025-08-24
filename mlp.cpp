// mlp/mlp.cpp
// Multi-layer perceptron implementation in C with minimal C++ features
// No classes or templates used, minimal vector usage

#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>

// Model dimensions
struct MLPParams {
    int n_layers = 3;  // Number of hidden layers
    int n_in = 8;  // Input features
    int n_hidden = 16;  // Hidden units
    int n_out = 4;  // Output units
};

// Model layers
struct MLPLayer {
    std::vector<float> W;  // Output layer weights (n_out x n_in)
    std::vector<float> b;  // Output layer biases (n_out)
};

// Model
struct MLP {
    // Model layers
    std::vector<MLPLayer> layers;

    // Input/output tensors
    std::vector<float> x;  // 1D input vector
    std::vector<float> y;  // 1D output vector

    // Model dimensions
    struct MLPParams params{};
};

int main(void) {
    srand(time(NULL));  // Seed random number generator

    // Initialize model
    MLP mlp{};

    // Input vector
    mlp.x.resize(mlp.params.n_in);
    for (size_t i = 0; i < mlp.x.size(); i++) {
        mlp.x[i] = (float) rand() / (float) RAND_MAX;  // Normalize input
    }

    // Output results
    for (size_t i = 0; i < mlp.x.size(); i++) {
        printf("mlp.x[%zu] = %.6f\n", i, (double) mlp.x[i]);
    }

    // Initialize model layers
    mlp.layers.resize(mlp.params.n_layers);

    // Xavier-Glorot initialization
    for (int i = 0; i < mlp.params.n_layers; i++) {
        // Get the current layer
        struct MLPLayer* L = &mlp.layers[i];
    
        // Current layer dimensions
        size_t n_in = (i == 0) ? mlp.params.n_in : mlp.params.n_hidden;
        size_t n_out = (i == mlp.params.n_layers - 1) ? mlp.params.n_out : mlp.params.n_hidden;

        // Layer dimensions
        size_t W_d = n_in * n_out;  // Weights (n_in x n_out)
        size_t b_d = n_out;  // Biases (n_out)

        // Initialize weights and biases
        L->W.resize(W_d);
        L->b.resize(b_d);

        // Xavier-Glorot initialization
        float a = sqrtf(6.0f / (n_in + n_out));  // Scaling factor

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

    return 0;
}
