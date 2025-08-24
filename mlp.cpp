// mlp/mlp.cpp
// a multi-layer perceptron in C++ completely from scratch.
// the code will lean into C heavily while leveraging C++ utils to reduce complexity.
// no classes or templates are allowed. minimal usage of vectors, maps, etc. is valid.
// auto is banned from usage! it is an anti-pattern to hide data types!
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>

// Model dimensions
struct MLPParams {
    int n_layers = 3;
    int n_in = 8;
    int n_hidden = 16;
    int n_out = 4;
};

// Model layers
struct MLPLayer {
    std::vector<float> W;  // n_out x n_in
    std::vector<float> b;  // n_out
};

// Model
struct MLP {
    // setup model layers
    std::vector<struct MLPLayer> layers;

    // input/output tensors
    std::vector<float> x;  // 1d input vector
    std::vector<float> y;  // 1d output vector

    // setup model params
    struct MLPParams params{};
};

int main(void) {
    srand(time(NULL));

    // initialize the model
    MLP mlp{};

    // initialize the input vector
    mlp.x.resize(mlp.params.n_in);
    for (size_t i = 0; i < mlp.x.size(); i++) {
        mlp.x[i] = (float) rand() / (float) RAND_MAX;  // normalized distribution
    }

    // dump the input results
    for (size_t i = 0; i < mlp.x.size(); i++) {
        printf("mlp.x[%zu] = %.6f\n", i, (double) mlp.x[i]);
    }

    // zero-initialize model layers
    mlp.layers.resize(mlp.params.n_layers);

    // calculate layer dimensions
    size_t W_d = mlp.params.n_out * mlp.params.n_in;
    size_t b_d = mlp.params.n_out;

    // xavier-glorot initialization
    for (int i = 0; i < mlp.params.n_layers; i++) {
        // get the current layer
        MLPLayer* L = &mlp.layers[i];

        // initialize the weights and biases
        L->W.resize(W_d);
        L->b.resize(b_d);

        // real distribution
        float a = sqrtf(6.f / (W_d + b_d));  // fan_in + fan_out

        // apply distribution to weights
        for (size_t j = 0; j < W_d; j++) {
            float rd = 2 * ((float) rand() / RAND_MAX) - 1;
            L->W[j] = rd * a;
        }

        // apply distribution to biases
        for (size_t j = 0; j < b_d; j++) {
            float rd = 2 * ((float) rand() / RAND_MAX) - 1;
            L->b[j] = rd * a;
        }
    }

    return 0;
}
