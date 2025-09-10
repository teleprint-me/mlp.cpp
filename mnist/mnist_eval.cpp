/**
 * @file      mnist/mnist_eval.cpp
 * @brief     Evaluate a pre-trained multi-layer perceptron for digit classification.
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

#include <algorithm>
#include <vector>

#include "mlp.h"
#include "ckpt.h"
#include "mnist_data.h"

struct CLIParams {
    struct MLP mlp {};

    const char** argv = nullptr;
    int argc = -1;

    char* ckpt_path = nullptr;
    char* data_path = nullptr;
    int n_samples_per_class = 10;
};

void cli_usage(struct CLIParams* cli) {
    printf("Usage: %s [--ckpt S] [--data S] [--samples N] ...\n", cli->argv[0]);
    printf("--ckpt      S Checkpoint path (default: ./mnist-latest.bin)\n");
    printf("--data      S Dataset path (default: ./data/mnist/testing)\n");
    printf("--samples   N Number of samples per class (default: %d)\n", cli->n_samples_per_class);
}

void cli_parse(struct CLIParams* cli) {
    const int argc = cli->argc;
    const char** argv = cli->argv;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            cli->ckpt_path = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            cli->data_path = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            cli->n_samples_per_class = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cli_usage(cli);
            exit(0);
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            cli_usage(cli);
            exit(1);
        }
    }
}

int main(int argc, const char* argv[]) {
    // Create CLI parameters
    struct CLIParams cli{};

    // Initialize model layers (do this before parsing!)
    cli.mlp.dim.n_in = 768;  // 28x28 for each image
    cli.mlp.dim.n_hidden = 10;  // default to a sane value
    cli.mlp.dim.n_out = 10;  // one-hot for each digit

    // Initialize and parse cli parameters
    cli.argc = argc;
    cli.argv = argv;
    cli_parse(&cli);

    // Set a sample count per class threshold
    if (1 > cli.n_samples_per_class) {
        cli.n_samples_per_class = 1;
    }

    // Create default checkpoint path
    if (!cli.ckpt_path) {
        cli.ckpt_path = strdup("./mnist-latest.bin");
    }

    // Create default dataset path
    if (!cli.data_path) {
        cli.data_path = strdup("./data/mnist/testing");
    }

    // Copy configured model to runtime space
    struct MLP mlp = cli.mlp;

    printf("╔═══════════════════════════╗\n");
    printf("║  MNIST EVAL  (by Austin)  ║\n");
    printf("╚═══════════════════════════╝\n");

    // Create a working directory
    char* dirname = mlp_ckpt_dirname(cli.ckpt_path);
    // Invalid working directory
    if (!dirname) {
        free(cli.ckpt_path);
        return 1;
    }

    // Create a file name
    char* basename = mlp_ckpt_basename(cli.ckpt_path);
    // No file name was given
    if (!basename) {
        basename = strdup("mnist-latest.bin");  // Default to latest model file
    }

    // Calculate the maximum length for the ckeckpoint path
    size_t ckpt_max_path_len = mlp_ckpt_max_path_len(cli.ckpt_path);

    // Allocate memory to the checkpoint path
    char* ckpt_path = (char*) malloc(ckpt_max_path_len + 1);

    // Write the file path to the checkpoint path
    mlp_ckpt_path(ckpt_path, ckpt_max_path_len, dirname, basename);

    // Log the resultant checkpoint path
    fprintf(stderr, "Paths:\n");
    fprintf(stderr, "(ಥ⌣ಥ) %s\n", cli.data_path);
    fprintf(stderr, "(☞ﾟヮﾟ)☞ %s\n\n", ckpt_path);

    // Load and initialize a pre-trained model
    assert(mlp_ckpt_load(&mlp, ckpt_path));

    // Log model initialization as a sanity check
    mlp_log_dims(&mlp);

    // Load and initialize the MNIST dataset
    std::vector<MNISTSample> samples{}; /**< Array of MNIST samples. */
    fprintf(stderr, "Loading:\n");
    assert(mnist_load_samples(cli.data_path, cli.n_samples_per_class, samples));
    fprintf(stderr, "(ʘ‿ʘ)~ Loaded %d images\n\n", cli.n_samples_per_class * 10);

    // Run evaluation for digit classification
    std::vector<MNISTSample> test_samples{};
    mnist_load_samples("./data/mnist/testing", cli.n_samples_per_class, test_samples);

    size_t correct = 0;
    for (size_t i = 0; i < test_samples.size(); i++) {
        mlp.x = test_samples[i].pixels;
        mlp_forward(&mlp, mlp.x.data(), mlp.x.size());
        int pred = std::distance(mlp.y.begin(), std::max_element(mlp.y.begin(), mlp.y.end()));
        if (pred == test_samples[i].label) {
            correct++;
        }
    }
    printf("Test Accuracy: %.2f%%\n", 100.0 * correct / test_samples.size());

    // Clean up
    free(basename);
    free(dirname);
    free(ckpt_path);
    free(cli.ckpt_path);
    free(cli.data_path);

    return 0;
}
