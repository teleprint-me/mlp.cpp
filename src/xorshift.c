/**
 * @license    cc-by-sa-4.0
 * @file       xorshift.h
 * @brief      Thread-safe Xorshift PRNG utility for C/C++ (POSIX compatible).
 *
 * Provides a minimal, high-quality PRNG with thread-local state,
 * floating-point output, and ML/NN initialization utilities.
 * See: https://en.wikipedia.org/wiki/Xorshift
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "xorshift.h"

struct Xorshift {
    uint64_t seed;
};

thread_local struct Xorshift state = {0};

void xorshift_init(uint64_t seed) {
    // Avoid seed==0 (degenerate cycle)
    state.seed = seed ? seed : 0xDEADBEEFCAFEBABEull;
}

static inline uint64_t xorshift_next(void) {
    state.seed ^= state.seed >> 12;
    state.seed ^= state.seed << 25;
    state.seed ^= state.seed >> 27;
    return state.seed * 0x2545F4914F6CDD1Dull;
}

// 32-bit random integer (high bits)
uint32_t xorshift_uint32(void) {
    return (uint32_t) (xorshift_next() >> 32);
}

// 24-bit uniform float [0,1)
float xorshift_float(void) {
    uint32_t x = xorshift_uint32();
    return (x >> 8) / 16777216.0f;
    // Or, for all 32 bits:
    // return x / 4294967296.0f;
}

// Xavier/Glorot uniform
float xorshift_xavier(size_t in, size_t out) {
    float a = sqrtf(6.0f / (in + out));
    float ud = 2.0f * xorshift_float() - 1.0f;
    return ud * a;
}

// Box-Muller normal
float xorshift_muller(size_t in, size_t out) {
    float u1 = xorshift_float();
    if (u1 < 1e-7f) {
        u1 = 1e-7f;  // avoid 0
    }
    float u2 = xorshift_float();
    if (u2 < 1e-7f) {
        u2 = 1e-7f;  // avoid 0
    }
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float) M_PI * u2);
    float stddev = sqrtf(2.0f / (in + out));
    return z0 * stddev;
}

bool xorshift_yates(void* base, size_t n, size_t size) {
    if (!base || n < 2) {
        return false;  // redundant
    }

    uint8_t* arr = (uint8_t*) base;
    uint8_t* tmp = (uint8_t*) malloc(size);
    if (!tmp) {
        return false;  // malloc failed
    }

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = xorshift_uint32() % (i + 1);
        /// @note use memmove for safe overlapping memory
        memmove(tmp, arr + i * size, size);
        memmove(arr + i * size, arr + j * size, size);
        memmove(arr + j * size, tmp, size);
    }

    free(tmp);
    return true;
}
