// @file inspect.cpp
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

    // Construct the checkpoint file path
    char ckpt_path[MLP_MAX_FNAME];
    mlp_ckpt_path(ckpt_path, MLP_MAX_FNAME, file_path);

    // Read the model file
    mlp_ckpt_load(&mlp, ckpt_path);

    // Log model data to stdout
    mlp_log_dims(&mlp);
    mlp_log_layers(&mlp);

    return 0;
}
