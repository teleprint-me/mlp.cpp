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

#include <unistd.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "xorshift.h"
#include "path.h"
#include "mlp.h"
#include "ckpt.h"

void cli_usage(struct MLP* mlp, const char* prog) {
    const char options[] = "[--seed N] [--layers N] [--hidden N] [--epochs N] [--lr F] [...]";
    const char* nest = (mlp->opt.nesterov) ? "true" : "false";

    printf("Usage: %s %s\n", prog, options);
    printf("--data      S Dataset path (default: ./data/mnist)\n");
    printf("--ckpt      S Checkpoint path (default: ./mnist-latest.bin)\n");
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

void cli_parse(int argc, const char* argv[], struct MLP* mlp, char* file_path[]) {
    // Simple manual CLI parse
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            *file_path = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            mlp->dim.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bias") == 0 && i + 1 < argc) {
            mlp->dim.bias = atof(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            mlp->dim.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            mlp->dim.n_hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            mlp->opt.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            mlp->opt.log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            mlp->opt.lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            mlp->opt.tolerance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--decay") == 0 && i + 1 < argc) {
            mlp->opt.weight_decay = atof(argv[++i]);
        } else if (strcmp(argv[i], "--momentum") == 0 && i + 1 < argc) {
            mlp->opt.momentum = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dampening") == 0 && i + 1 < argc) {
            mlp->opt.dampening = atof(argv[++i]);
        } else if (strcmp(argv[i], "--nesterov") == 0 && i + 1 < argc) {
            mlp->opt.nesterov = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cli_usage(mlp, argv[0]);
            exit(0);
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            cli_usage(mlp, argv[0]);
            exit(1);
        }
    }
}

int main(int argc, const char* argv[]) {
    // Create the model
    struct MLP mlp {};

    mlp.dim.n_in = 768;  // 28x28 for each image
    mlp.dim.n_out = 10;  // one-hot for each digit
    mlp.dim.n_hidden = 128;  // default to a sane value

    char* file_path = nullptr;
    cli_parse(argc, argv, &mlp, &file_path);

    if (mlp.dim.n_layers < 3) {
        fprintf(stderr, "[ERROR] Minimum layers = 3\n\t(┛ಠ_ಠ)┛彡┻━┻\n");
        return 1;
    }

    if (mlp.dim.n_hidden < 32) {
        fprintf(stderr, "[ERROR] Minimum hidden units = 32\n\t┻━┻︵ \\(°□°)/ ︵ ┻━┻\n");
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

    if (!file_path) {
        file_path = strdup("./mnist-latest.bin");
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
        basename = strdup("mnist-latest.bin");  // Default to latest model file
    }

    // Calculate the maximum length for the ckeckpoint path
    size_t max_path_len = mlp_ckpt_max_path_len(file_path);

    // Allocate memory to the checkpoint path
    char* ckpt_path = (char*) malloc(max_path_len + 1);

    // Write the file path to the checkpoint path
    mlp_ckpt_path(ckpt_path, max_path_len, dirname, basename);

    // Log the resultant checkpoint path
    fprintf(stderr, "(☞ﾟヮﾟ)☞ %s\n\n", ckpt_path);

    // Initialize the model if it does not exist already
    if (path_is_file(ckpt_path)) {
        // Load and initialize a pre-trained model
        assert(mlp_ckpt_load(&mlp, ckpt_path));
    } else {
        // Apply xavier-glorot initialization to model layers
        mlp_init_xavier(&mlp);
    }

    // Log model parameters
    mlp_log_dims(&mlp);
    mlp_log_opts(&mlp);

    // Create input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

    // Do a sanity check after initializing the model
    mlp_log_layers(&mlp);

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

    // For per-epoch stats
    std::vector<float> y_pred(mlp.dim.n_out);

    // Execute training loop
    for (size_t epoch = 0; epoch < mlp.opt.epochs; epoch++) {
        float loss_epoch = 0.0f;

        // For each XOR sample
        for (size_t i = 0; i < inputs.size(); i++) {
            // Set the current inputs
            mlp.x = inputs[i];

            // Set the current outputs
            std::vector<float> &y_true = outputs[i];

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
        loss_epoch /= inputs.size();

        // Log every n epochs
        if (epoch % mlp.opt.log_every == 0) {
            printf("epoch[%zu] Σ(-᷅_-᷄๑) %f\n", epoch, (double) loss_epoch);
            mlp_ckpt_stamp(ckpt_path, max_path_len, dirname, "mnist", epoch);
            mlp_ckpt_save(&mlp, ckpt_path);
        }

        // Stop loss
        if (loss_epoch < mlp.opt.tolerance) {
            printf("epoch[%zu] (◕‿◕✿) (loss < %f)\n", epoch, (double) mlp.opt.tolerance);
            break;
        }
    }

    // Always save the lastest checkpoint with a time stamp as a backup
    mlp_ckpt_stamp(ckpt_path, max_path_len, dirname, "mnist", mlp.opt.epochs);
    mlp_ckpt_save(&mlp, ckpt_path);

    // Always save the latest checkpoint to the same file
    mlp_ckpt_path(ckpt_path, max_path_len, dirname, basename);
    mlp_ckpt_save(&mlp, ckpt_path);

    free(file_path);
    free(basename);
    free(dirname);
    free(ckpt_path);

    // Log predictions
    printf("\n-=≡Σ<|°_°|>\n");
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
