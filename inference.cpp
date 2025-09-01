// @file inference.cpp
#include <cassert>
#include <cstring>
#include <cstdio>

#include "mlp.h"
#include "ckpt.h"

void cli_usage(const char* prog) {
    char fname[MLP_MAX_FNAME];
    mlp_ckpt_name(fname, MLP_MAX_FNAME, 0);

    printf("Usage: %s %s\n", prog, "[--ckpt S] ...");
    printf("--ckpt S Checkpoint path (default: %s)\n", fname);
}

void cli_parse(int argc, const char* argv[], char* file_path[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            *file_path = (char*) argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cli_usage(argv[0]);
            exit(0);
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            cli_usage(argv[0]);
            exit(1);
        }
    }
}

int main(int argc, const char* argv[]) {
    struct MLP mlp {};

    // Parse user input
    char* file_path = nullptr;
    cli_parse(argc, argv, &file_path);

    printf("╔══════════════════════════════╗\n");
    printf("║  XOR INFERENCE  (by Austin)  ║\n");
    printf("╚══════════════════════════════╝\n");

    // Construct the checkpoint file path
    char ckpt_path[MLP_MAX_FNAME];
    mlp_ckpt_path(ckpt_path, MLP_MAX_FNAME, file_path);

    // Ensure the model file exists
    assert(mlp_ckpt_exists(ckpt_path));

    // Read the model file
    assert(mlp_ckpt_load(&mlp, ckpt_path));

    // Log model dims to stdout
    mlp_log_dims(&mlp);

    // Create input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

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

    // Compute forward propagation (predictions)
    mlp_forward(&mlp, mlp.x.data(), mlp.x.size());

    // Log predictions
    printf("-=≡Σ<|°_°|>:\n");
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
