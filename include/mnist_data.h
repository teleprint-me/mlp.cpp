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
 *
 * Usage example:
 *   uint64_t seed = 1337;
 *   xorshift_init(seed);
 *   char dataset_path[] = "data/mnist/training";
 *   std::vector<MNISTSample> samples{}; ///< Array of MNIST samples.
 *   mnist_load_samples(samples, 10, dataset_path);
 */

#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <cstdint>

#include <vector>

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

std::vector<float> mnist_one_hot(int label, int n_classes);

// load image and force grayscale
uint8_t* mnist_load_image(const char* filename);

struct MNISTSample mnist_new_sample(const uint8_t* data, int label);

bool mnist_load_samples(
    const char* dirname, size_t n_samples_per_class, std::vector<MNISTSample> &out
);

#endif  // MNIST_DATA_H
