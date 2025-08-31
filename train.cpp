/**
 * Copyright © 2025 Austin Berrio
 * @file mlp/train.cpp
 * @brief Multi-layer perceptron implementation in C with minimal C++ features.
 * - Classes and templates are **not** allowed! C-like structs are preferred.
 *   - Setting default field parameters for `struct` is encouraged.
 * - `auto` keyword usage is **not** allowed!
 *   - Data types must be explicitly declared! Hiding types is discouraged.
 * - Using std::vector is allowed to simplify memory management.
 *   - std::vector usage is preferred to simplify vector and matrix operations.
 * - Function signatures must be explicit and C-like.
 *   - Function parameters must use explicit types and shapes, e.g. f(float* x, size_t n).
 *   - Pointer refs signal the data is mutable to some capacity unless const qualified.
 * - Simplicity rules them all!
 *   - Implementations should prioritize simplicity at all costs.
 *   - Abstractions are deferred until absolutely necessary.
 *   - Abstractions will reveal themselves through prototyping.
 */

#include <ctime>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include "mlp.h"

#define MLP_MAGIC 0x6D6C7000  // 'mlp\0'
#define MLP_VERSION 1

bool mlp_save(struct MLP* mlp, const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "[ERROR] File is unwritable.\n\t(⊙.☉)7");
        return false;
    }

    uint32_t magic = MLP_MAGIC;
    uint32_t version = MLP_VERSION;

    // Write header
    fwrite(&magic, sizeof(uint32_t), 1, file);
    fwrite(&version, sizeof(uint32_t), 1, file);
    fwrite(&mlp->dim.n_layers, sizeof(uint32_t), 1, file);
    fwrite(&mlp->dim.n_in, sizeof(uint32_t), 1, file);
    fwrite(&mlp->dim.n_hidden, sizeof(uint32_t), 1, file);
    fwrite(&mlp->dim.n_out, sizeof(uint32_t), 1, file);

    // Write each layer
    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        const MLPLayer* L = &mlp->layers[i];

        size_t W_count = mlp_layer_dim_out(mlp, i) * mlp_layer_dim_in(mlp, i);
        size_t b_count = mlp_layer_dim_out(mlp, i);

        fwrite(L->W.data(), sizeof(float), W_count, file);
        fwrite(L->b.data(), sizeof(float), b_count, file);
    }

    fclose(file);
    return true;
}

bool mlp_load(struct MLP* mlp, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "[ERROR] File is unreadable.\n\t(⊙.☉)7\n");
        return false;
    }

    uint32_t magic = 0;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (MLP_MAGIC != magic) {
        fprintf(stderr, "[ERROR] File is not an MLP.\n\tಡ_ಡ\n");
        return false;
    }

    uint32_t version = 0;
    fread(&version, sizeof(uint32_t), 1, file);
    if (MLP_VERSION != version) {
        fprintf(stderr, "[ERROR] Unsupported MLP version.\n\t(－‸ლ)\n");
        return false;
    }

    // Read header
    fread(&mlp->dim.n_layers, sizeof(uint32_t), 1, file);
    fread(&mlp->dim.n_in, sizeof(uint32_t), 1, file);
    fread(&mlp->dim.n_hidden, sizeof(uint32_t), 1, file);
    fread(&mlp->dim.n_out, sizeof(uint32_t), 1, file);

    // Reset the model
    mlp->layers.clear();
    mlp->layers.resize(mlp->dim.n_layers);

    // Read each layer
    for (size_t i = 0; i < mlp->dim.n_layers; i++) {
        MLPLayer* L = &mlp->layers[i];

        size_t W_count = mlp_layer_dim_out(mlp, i) * mlp_layer_dim_in(mlp, i);
        size_t b_count = mlp_layer_dim_out(mlp, i);

        fread(L->W.data(), sizeof(float), W_count, file);
        fread(L->b.data(), sizeof(float), b_count, file);
    }

    fclose(file);
    return true;
}

void print_usage(struct MLP* mlp, const char* prog) {
    const char options[] = "[--seed N] [--layers N] [--hidden N] [--epochs N] [--lr F] [...]";
    printf("Usage: %s %s\n", prog, options);
    printf("  --seed      N    Random seed (default: %zu)\n", mlp->dim.seed);
    printf("  --bias      F    Initial bias (default: %f)\n", (double) mlp->dim.bias);
    printf("  --layers    N    Number of layers (default: %zu)\n", mlp->dim.n_layers);
    printf("  --hidden    N    Hidden units per layer (default: %zu)\n", mlp->dim.n_hidden);
    printf("  --epochs    N    Training epochs (default: %zu)\n", mlp->opt.epochs);
    printf("  --log-every N    Log every N epochs (default: %zu)\n", mlp->opt.log_every);
    printf("  --lr        F    Learning rate (default: %f)\n", (double) mlp->opt.lr);
    printf("  --tolerance F    Stop loss (default: %f)\n", (double) mlp->opt.tolerance);
    printf("  --decay     F    L2 regularization (default: %f)\n", (double) mlp->opt.weight_decay);
    printf("  --momentum  F    Momentum coefficient (default: %f)\n", (double) mlp->opt.momentum);
    printf("  --dampening F    Dampening coefficient (default: %f)\n", (double) mlp->opt.dampening);
    printf(
        "  --nesterov  N    Nesterov acceleration (default: %s)\n",
        (mlp->opt.nesterov) ? "true" : "false"
    );
}

int main(int argc, const char* argv[]) {
    // Create the model
    MLP mlp{};

    // Simple manual CLI parse
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            mlp.dim.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bias") == 0 && i + 1 < argc) {
            mlp.dim.bias = atof(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            mlp.dim.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            mlp.dim.n_hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            mlp.opt.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            mlp.opt.log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            mlp.opt.lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            mlp.opt.tolerance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--decay") == 0 && i + 1 < argc) {
            mlp.opt.weight_decay = atof(argv[++i]);
        } else if (strcmp(argv[i], "--momentum") == 0 && i + 1 < argc) {
            mlp.opt.momentum = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dampening") == 0 && i + 1 < argc) {
            mlp.opt.dampening = atof(argv[++i]);
        } else if (strcmp(argv[i], "--nesterov") == 0 && i + 1 < argc) {
            mlp.opt.nesterov = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(&mlp, argv[0]);
            return 0;
        } else {
            printf("Unknown or incomplete option: %s\n", argv[i]);
            print_usage(&mlp, argv[0]);
            return 1;
        }
    }

    if (mlp.dim.n_layers < 3) {
        fprintf(stderr, "[ERROR] Minimum layers = 3\n\t(┛ಠ_ಠ)┛彡┻━┻\n");
        return 1;
    }

    if (mlp.dim.n_hidden < 2) {
        fprintf(stderr, "[ERROR] Minimum hidden dims = 2\n\t┻━┻︵ \\(°□°)/ ︵ ┻━┻\n");
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
    printf("║  MLP XOR Lab  (by Austin)    ║\n");
    printf("╚══════════════════════════════╝\n");

    // Log model parameters
    mlp_log_dims(&mlp);
    mlp_log_opts(&mlp);

    // Seed random number generator
    if (mlp.dim.seed > 0) {
        srand(mlp.dim.seed);
    } else {
        srand(time(NULL));
    }

    // Create input and output vectors
    mlp.x.resize(mlp.dim.n_in);
    mlp.y.resize(mlp.dim.n_out);

    // Apply xavier-glorot initialization to model layers
    mlp_init_xavier(&mlp);
    // Do a sanity check when initializing the model
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
        if (epoch % mlp.opt.log_every == 0 || loss_epoch < mlp.opt.tolerance) {
            printf("epoch[%zu] Σ(-᷅_-᷄๑) = %f\n", epoch, (double) loss_epoch);
        }

        // Stop loss
        if (loss_epoch < mlp.opt.tolerance) {
            printf("epoch[%zu] (◕‿◕✿) (loss < %f)\n", epoch, (double) mlp.opt.tolerance);
            break;
        }
    }

    // Log predictions
    printf("\n-=≡Σ<|°_°|>:\n");
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
