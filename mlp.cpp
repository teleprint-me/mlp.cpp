// mlp/mlp.cpp
// a multi-layer perceptron in C++ completely from scratch.
// the code will lean into C heavily while leveraging C++ utils to reduce complexity.
// no classes or templates are allowed. minimal usage of vectors, maps, etc. is valid.
#include <cstdio>

// Model dimensions
struct MLPParams {
    int n_layers = 3;
    int n_in = 8;
    int n_hidden = 16;
    int n_out = 4;
};

// Model layers
struct MLPLayers {
    float* W = nullptr;  // flat 2d matrix
    float* b = nullptr;  // 1d vector
};

// Model
struct MLP {
    float* x = nullptr;  // 1d input vector
    float* y = nullptr;  // 1d output vector

    struct MLPLayers* layers;

    struct MLPParams {};
};

int main(void) {
    printf("Hello, world!\n");
    return 0;
}
