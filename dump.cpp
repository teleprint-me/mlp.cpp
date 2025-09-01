// @file dump.cpp
#include <cstring>
#include <cstdio>

#include "mlp.h"
#include "ckpt.h"

void print_usage(const char* prog) {
    const char options[] = "[--seed N] [--layers N] [--hidden N] [--epochs N] [--lr F] [...]";

    char fname[MLP_MAX_FNAME];
    mlp_ckpt_name(fname, MLP_MAX_FNAME, 0);

    printf("Usage: %s %s\n", prog, options);
    printf("--ckpt S Checkpoint path (default: %s)\n", fname);
}

int main(int argc, const char* argv[]) {
    struct MLP mlp {};

    char* file_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            file_path = (char*) argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    fprintf(stderr, "file path: %s\n", file_path);

    return 0;
}
