/**
 * @file      mnist/mnist_train.cpp
 * @brief     Pre-train a multi-layer perceptron on hand-written digit classification.
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
#include <ctime>
#include <cstring>
#include <cstdio>

#include <algorithm>
#include <vector>

#include <unistd.h>
#include <sys/stat.h>

#include "xorshift.h"
#include "path.h"
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
    struct MLP* mlp = &cli->mlp;

    const char options[] = "[--seed N] [--layers N] [--hidden N] [--epochs N] [--lr F] [...]";
    const char* nest = (mlp->opt.nesterov) ? "true" : "false";

    printf("Usage: %s %s\n", cli->argv[0], options);
    printf("--ckpt      S Checkpoint path (default: ./mnist-latest.bin)\n");
    printf("--data      S Dataset path (default: ./data/mnist/training)\n");
    printf("--samples   N Number of samples per class (default: 10)\n");
    printf("--seed      N Random seed (default: %zu)\n", mlp->dim.seed);
    printf("--bias      F Initial bias (default: %f)\n", (double) mlp->dim.bias);
    printf("--layers    N Number of layers (default: %zu)\n", mlp->dim.n_layers);
    printf("--hidden    N Hidden units per layer (default: %zu)\n", mlp->dim.n_hidden);
    printf("--epochs    N Training epochs (default: %zu)\n", mlp->opt.epochs);
    printf("--log-every N Log every N epochs (default: %zu)\n", mlp->opt.log_every);
    printf("--lr        F Learning rate (default: %f)\n", (double) mlp->opt.lr);
    printf("--tolerance F Stop loss (default: %f)\n", (double) mlp->opt.tolerance);
    printf("--decay     F L2 regularization (default: %f)\n", (double) mlp->opt.weight_decay);
    printf("--momentum  F Momentum coefficient (default: %f)\n", (double) mlp->opt.momentum);
    printf("--dampening F Dampening coefficient (default: %f)\n", (double) mlp->opt.dampening);
    printf("--nesterov  N Nesterov acceleration (default: %s)\n", nest);
}

void cli_parse(struct CLIParams* cli) {
    const int argc = cli->argc;
    const char** argv = cli->argv;

    // Simple manual CLI parse
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            cli->ckpt_path = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            cli->data_path = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            cli->n_samples_per_class = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cli->mlp.dim.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bias") == 0 && i + 1 < argc) {
            cli->mlp.dim.bias = atof(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            cli->mlp.dim.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            cli->mlp.dim.n_hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            cli->mlp.opt.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            cli->mlp.opt.log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            cli->mlp.opt.lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            cli->mlp.opt.tolerance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--decay") == 0 && i + 1 < argc) {
            cli->mlp.opt.weight_decay = atof(argv[++i]);
        } else if (strcmp(argv[i], "--momentum") == 0 && i + 1 < argc) {
            cli->mlp.opt.momentum = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dampening") == 0 && i + 1 < argc) {
            cli->mlp.opt.dampening = atof(argv[++i]);
        } else if (strcmp(argv[i], "--nesterov") == 0 && i + 1 < argc) {
            cli->mlp.opt.nesterov = atoi(argv[++i]) ? 1 : 0;
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
        cli.data_path = strdup("./data/mnist/training");
    }

    // Copy configured model to runtime space
    struct MLP mlp = cli.mlp;

    if (mlp.dim.n_layers < 3) {
        fprintf(stderr, "[ERROR] Minimum layers = 3\n\t(┛ಠ_ಠ)┛彡┻━┻\n");
        return 1;
    }

    if (mlp.dim.n_hidden < 10) {
        fprintf(stderr, "[ERROR] Minimum hidden units = 10\n\t┻━┻︵ \\(°□°)/ ︵ ┻━┻\n");
        return 1;
    }

    if (mlp.opt.lr <= 0.0f) {
        fprintf(stderr, "[ERROR] Learning rate > 0\n\t(╯°Д°)╯︵/(.□ . \\)\n");
        return 1;
    }

    if (mlp.opt.dampening < 0.0f || mlp.opt.dampening >= 1.0f) {
        fprintf(stderr, "[ERROR] 0 <= dampening < 1\n\t(^ಠ_ಠ^)ﾉ\n");
    }

    if (mlp.opt.momentum < 0.0f || mlp.opt.momentum >= 1.0f) {
        fprintf(stderr, "[ERROR] 0 <= momentum < 1\n\t┬─┬ノ( º _ ºノ)\n");
        return 1;
    }

    if (mlp.opt.nesterov && mlp.opt.momentum == 0.0f) {
        fprintf(stderr, "[Error] Momentum > 0\n\t(˚Õ˚)ر ~~~~┻━┻\n");
        return 1;
    }

    printf("╔══════════════════════════════╗\n");
    printf("║  MNIST TRAINER  (by Austin)  ║\n");
    printf("╚══════════════════════════════╝\n");

    // Seed random number generator
    if (mlp.dim.seed > 0) {
        xorshift_init(mlp.dim.seed);
    } else {
        xorshift_init(time(NULL));
    }

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

    // Initialize the model if it does not exist already
    if (path_is_file(ckpt_path)) {
        // Load and initialize a pre-trained model
        assert(mlp_ckpt_load(&mlp, ckpt_path));
    } else {
        // Apply xavier-glorot initialization to model layers
        mlp_init_xavier(&mlp);
    }

    // Log model initialization as a sanity check
    mlp_log_dims(&mlp);
    mlp_log_opts(&mlp);
    // mlp_log_layers(&mlp);

    // Load and initialize the MNIST dataset
    std::vector<MNISTSample> samples{}; /**< Array of MNIST samples. */
    fprintf(stderr, "Loading:\n");
    mnist_load_samples(cli.data_path, cli.n_samples_per_class, samples);
    fprintf(stderr, "(ʘ‿ʘ)~ Loaded %d images\n\n", cli.n_samples_per_class * 10);

    // Initialize input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

    // For per-epoch stats
    std::vector<float> y_pred(mlp.dim.n_out);

    // Execute training loop
    fprintf(stderr, "Training:\n");
    for (size_t epoch = 0; epoch < mlp.opt.epochs; epoch++) {
        float loss_epoch = 0.0f;

        // Shuffle the selected samples
        xorshift_yates(samples.data(), samples.size(), sizeof(MNISTSample));

        // For each XOR sample
        for (size_t i = 0; i < samples.size(); i++) {
            // Copy input
            mlp.x = samples[i].pixels;

            // Set the current outputs
            std::vector<float> y_true = one_hot_encode(samples[i].label, mlp.dim.n_out);

            // Compute forward propagation (predictions)
            mlp_forward(&mlp, mlp.x.data(), mlp.x.size());

            // Compute loss per sample
            float loss = mse(mlp.y.data(), y_true.data(), y_true.size());
            loss_epoch += loss;

            // Compute backward propagation (gradients)
            mlp_compute_gradients(&mlp, y_true.data(), y_true.size());

            // Update parameters using computed gradients
            mlp_update_params(&mlp);
        }

        // Average loss over all samples
        loss_epoch /= samples.size();

        // Log every n epochs
        if (epoch % mlp.opt.log_every == 0) {
            printf("epoch[%zu] Σ(-᷅_-᷄๑) %f\n", epoch, (double) loss_epoch);
            mlp_ckpt_stamp(ckpt_path, ckpt_max_path_len, dirname, "mnist", epoch);
            mlp_ckpt_save(&mlp, ckpt_path);
        }

        // Stop loss
        if (loss_epoch < mlp.opt.tolerance) {
            printf("epoch[%zu] (◕‿◕✿) (loss < %f)\n", epoch, (double) mlp.opt.tolerance);
            break;
        }
    }

    // Always save the lastest checkpoint with a time stamp as a backup
    mlp_ckpt_stamp(ckpt_path, ckpt_max_path_len, dirname, "mnist", mlp.opt.epochs);
    mlp_ckpt_save(&mlp, ckpt_path);

    // Always save the latest checkpoint to the same file
    mlp_ckpt_path(ckpt_path, ckpt_max_path_len, dirname, basename);
    mlp_ckpt_save(&mlp, ckpt_path);

    size_t correct = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        mlp.x = samples[i].pixels;
        mlp_forward(&mlp, mlp.x.data(), mlp.x.size());
        int pred = std::distance(mlp.y.begin(), std::max_element(mlp.y.begin(), mlp.y.end()));
        if (pred == samples[i].label) {
            correct++;
        }
    }
    printf("Accuracy: %.2f%%\n", 100.0 * correct / samples.size());

    // Clean up
    free(basename);
    free(dirname);
    free(ckpt_path);
    free(cli.ckpt_path);
    free(cli.data_path);

    return 0;
}
