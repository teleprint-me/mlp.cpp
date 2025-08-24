// mlp/mlp.cpp
// a multi-layer perceptron in C++ completely from scratch.
// the code will lean into C heavily while leveraging C++ utils to reduce complexity.
// no classes or templates are allowed. minimal usage of vectors, maps, etc. is valid.
// auto is banned from usage! it is an anti-pattern to hide data types!
#include <ctime>
#include <cstdlib>
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
    float* W = nullptr;  // n_out x n_in
    float* b = nullptr;  // n_out
};

// Model
struct MLP {
    // setup model layers
    std::vector<struct MLPLayer> layers;

    // input/output tensors
    float* x = nullptr;  // 1d input vector
    float* y = nullptr;  // 1d output vector

    // setup model params
    struct MLPParams params{};
};

int main(void) {
    srand(time(NULL));

    // initialize the model
    MLP mlp{};
    // initialize the model layers
    mlp.layers.resize(mlp.params.n_layers);

    // initialize the input vector
    std::vector<float> x(mlp.params.n_in);
    mlp.x = x.data();
    for (size_t i = 0; i < x.size(); i++) {
        mlp.x[i] = (float) rand() / (float) RAND_MAX;  // normalized distribution
    }

    for (int i = 0; i < mlp.params.n_in; i++) {
        printf("mlp.x[%d] = %.6f\n", i, (double) mlp.x[i]);
    }
    return 0;
}
