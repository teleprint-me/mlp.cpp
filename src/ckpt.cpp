// @file src/ckpt.cpp
#include <ctime>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include <unistd.h>
#include <sys/stat.h>

#include "mlp.h"
#include "ckpt.h"

size_t mlp_ckpt_stamp(char* out, size_t n) {
    time_t t = time(NULL);
    struct tm* local = localtime(&t);
    return strftime(out, n, "%Y-%m-%dT%H-%M-%S", local);
}

bool mlp_ckpt_exists(const char* path) {
    struct stat buffer;
    return stat(path, &buffer) == 0;
}

void mlp_ckpt_path(char* buffer, size_t n, const char* file_path) {
    snprintf(buffer, n, "%s", file_path);
    fprintf(stderr, "ckpt (☞ﾟヮﾟ)☞ %s\n\n", buffer);
}

void mlp_ckpt_name(char* buffer, size_t n, size_t epoch) {
    char stamp[MLP_MAX_STAMP];
    mlp_ckpt_stamp(stamp, MLP_MAX_STAMP);
    snprintf(buffer, n, "mlp-%s-ep%zu.bin", stamp, epoch);
}

bool mlp_ckpt_save(struct MLP* mlp, const char* path) {
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
        // Get the current layer
        const MLPLayer* L = &mlp->layers[i];

        // Weight and bias dimensions
        size_t W_d = mlp_layer_dim_out(mlp, i) * mlp_layer_dim_in(mlp, i);
        size_t b_d = mlp_layer_dim_out(mlp, i);

        // Write the current layer to the file
        fwrite(L->W.data(), sizeof(float), W_d, file);
        fwrite(L->b.data(), sizeof(float), b_d, file);
    }

    fclose(file);
    return true;
}

bool mlp_ckpt_load(struct MLP* mlp, const char* path) {
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
        // Get the current layer
        MLPLayer* L = &mlp->layers[i];

        // Weight and bias dimensions
        size_t W_d = mlp_layer_dim_out(mlp, i) * mlp_layer_dim_in(mlp, i);
        size_t b_d = mlp_layer_dim_out(mlp, i);

        // Initialize the matrix and vector
        L->W.resize(W_d);
        L->b.resize(b_d);

        // Read the current layer from the file
        fread(L->W.data(), sizeof(float), W_d, file);
        fread(L->b.data(), sizeof(float), b_d, file);
    }

    fclose(file);
    return true;
}
