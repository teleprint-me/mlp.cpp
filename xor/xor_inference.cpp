/**
 * @file      xor/inference.cpp
 * @brief     Inference a pre-trained multi-layer perceptron.
 * @author    Austin Berrio
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

#include <cassert>
#include <cstring>
#include <cstdio>

#include "path.h"
#include "mlp.h"
#include "ckpt.h"

void cli_usage(const char* prog) {
    char fname[MLP_MAX_FNAME];
    mlp_ckpt_path(fname, MLP_MAX_FNAME, "models", "mlp-latest.bin");

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

    // Create a checkpoint path
    if (!file_path) {
        file_path = strdup("./xor-latest.bin");
    }

    // Create a working directory
    char* dirname = mlp_ckpt_dirname(file_path);
    // Invalid working directory
    if (!dirname) {
        free(file_path);
        return 1;
    }

    // Create a file name
    char* basename = mlp_ckpt_basename(file_path);
    // No file name was given
    if (!basename) {
        basename = strdup("xor-latest.bin");  // Default to latest model file
    }

    // Calculate the maximum length for the ckeckpoint path
    size_t max_path_len = mlp_ckpt_max_path_len(file_path);

    // Allocate memory to the checkpoint path
    char* ckpt_path = (char*) malloc(max_path_len + 1);

    // Write the file path to the checkpoint path
    mlp_ckpt_path(ckpt_path, max_path_len, dirname, basename);

    // Log the resultant checkpoint path
    fprintf(stderr, "(☞ﾟヮﾟ)☞ %s\n\n", ckpt_path);

    // Ensure the model file exists
    assert(path_is_file(ckpt_path));

    // Read the model file
    assert(mlp_ckpt_load(&mlp, ckpt_path));

    // Free path resources
    free(dirname);
    free(basename);
    free(ckpt_path);

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
    printf("-=≡Σ<|°_°|>\n");
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
