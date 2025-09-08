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

#include "xorshift.h"
#include "path.h"

// MNIST image length
#define IMAGE_LEN 28
// Flattened size of MNIST images
#define IMAGE_PIXELS ((IMAGE_LEN) * (IMAGE_LEN))

/**
 * @brief Represents a single MNIST image and its label.
 */
struct MNISTSample {
    int label = -1; /**< Label representing the digit (0-9). */
    std::vector<float> pixels{}; /**< Flattened pixel data (grayscale values). */
};

// shuffle an array of indices [0, n)
void shuffle_indices(size_t* values, size_t n, uint32_t (*rng)(void)) {
    for (size_t i = n - 1; i > 0; i--) {
        // random index [0, i]
        size_t j = rng() % (i + 1);
        // swap indices i with j
        size_t temp = values[i];
        values[i] = values[j];
        values[j] = temp;
    }
}

int main(void) {
    uint64_t seed = 1337;
    xorshift_init(seed);

    char dataset_path[] = "data/mnist/training";

    std::vector<MNISTSample> samples{}; /**< Array of MNIST samples. */

    size_t dirs_count;
    char** dirs = path_list_dirs(dataset_path, &dirs_count);
    if (!dirs) {
        return 1;
    }

    size_t n_samples = 10;  // select up to 10 samples
    for (size_t i = 0; i < dirs_count; i++) {
        // extract label from dirname
        char* label_str = path_basename(dirs[i]);
        int label = atoi(label_str);

        // extract files from dirname
        size_t files_count;
        char** files = path_list_files(dirs[i], &files_count);

        // Build a list of indices
        std::vector<size_t> indices(files_count);
        for (size_t k = 0; k < files_count; ++k) {
            indices[k] = k;
        }

        shuffle_indices(indices.data(), indices.size(), xorshift_gen_int32);
        size_t max_samples = std::min(n_samples, files_count);
        for (size_t j = 0; j < max_samples; j++) {
            // select images at random and bind them to the file count
            uint32_t idx = indices[j];

            // load image and force grayscale
            int width, height, channels;
            uint8_t* data = stbi_load(files[idx], &width, &height, &channels, 1);
            if (!data || IMAGE_LEN != width || IMAGE_LEN != height) {
                printf("[skip] %s (bad image)\n", files[idx]);
                if (data) {
                    stbi_image_free(data);
                }
                continue;
            }

            std::vector<float> pixels(IMAGE_PIXELS);
            for (int k = 0; k < IMAGE_PIXELS; k++) {
                pixels[k] = data[k] / 255.0f;
            }
            samples.push_back({label, pixels});
            stbi_image_free(data);
        }

        path_free_parts(files, files_count);
        path_free(label_str);
    }

    path_free_parts(dirs, dirs_count);
    return 0;
}
