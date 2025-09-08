/**
 * @file      mnist/mnist_data.cpp
 * @brief     Load the MNIST dataset into memory for training or inference.
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

#include <cstdint>
#include <stdlib.h>
#include <cstdio>

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "path.h"

// MNIST image dimensions
#define IMAGE_SIZE 28 * 28  // Flattened size of MNIST images

/**
 * @brief Represents a single MNIST image and its label.
 */
struct MNISTSample {
    int label = -1; /**< Label representing the digit (0-9). */
    std::vector<float> pixels{}; /**< Flattened pixel data (grayscale values). */
};

/**
 * @brief Represents a dataset of MNIST samples.
 */
struct MNISTDataset {
    uint32_t n_samples = 0; /**< Number of loaded samples. */
    std::vector<MNISTSample> samples{}; /**< Array of MNIST samples. */
};

int main(void) {
    char dataset_path[] = "data/mnist/training";

    size_t dirs_count;
    char** dirs = path_list_dirs(dataset_path, &dirs_count);
    if (!dirs) {
        return 1;
    }

    for (size_t i = 0; i < dirs_count; i++) {
        printf("dirs[%zu] %s\n", i, dirs[i]);
    }

    path_free_parts(dirs, dirs_count);
    return 0;
}
