/**
 * @file      xor/inspect.cpp
 * @brief     Inspect a pre-trained multi-layer perceptron.
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
    printf("║  XOR INSPECTOR  (by Austin)  ║\n");
    printf("╚══════════════════════════════╝\n");

    // Create a checkpoint path
    char* dirname = nullptr;
    char* basename = nullptr;

    // user supplied path
    // user supplied path
    if (path_exists(file_path)) {
        dirname = path_dirname(file_path);  // models/
        basename = path_basename(file_path);  // mlp-latest.bin
    } else {
        fprintf(stderr, "Path does not exist: %s\n", file_path ? file_path : "NULL");
        return 1;
    }

    // no file name was given (edge case)
    if (!*basename) {
        free(basename);
        basename = strdup("mlp-latest.bin");
    }

    // Construct the checkpoint file path
    char ckpt_path[MLP_MAX_FNAME];
    mlp_ckpt_path(ckpt_path, MLP_MAX_FNAME, dirname, basename);

    // Log the resultant checkpoint path
    fprintf(stderr, "ckpt (☞ﾟヮﾟ)☞ %s\n\n", ckpt_path);

    // Ensure the model file exists
    assert(path_is_file(ckpt_path));

    // Read the model file
    assert(mlp_ckpt_load(&mlp, ckpt_path));

    // free path resources
    free(dirname);
    free(basename);

    // Log model data to stdout
    mlp_log_dims(&mlp);
    mlp_log_layers(&mlp);

    return 0;
}
