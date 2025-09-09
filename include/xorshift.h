/**
 * @license    cc-by-sa-4.0
 * @file       xorshift.h
 * @brief      Thread-safe Xorshift PRNG utility for C/C++ (POSIX compatible).
 *
 * Provides a minimal, high-quality PRNG with thread-local state,
 * floating-point output, and ML/NN initialization utilities.
 * See: https://en.wikipedia.org/wiki/Xorshift
 */

#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * @def _Thread_local
 * @brief Compatibility macro for C11 `_Thread_local` keyword.
 */
#if defined(__GNUC__) && !defined(_Thread_local)
    #define _Thread_local __thread
#endif

/**
 * @def thread_local
 * @brief Portable alias for thread-local storage.
 */
#ifndef thread_local
    #define thread_local _Thread_local
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize thread-local Xorshift PRNG state.
 * @param seed 64-bit seed for RNG state. Must be nonzero.
 */
void xorshift_init(uint64_t seed);

/**
 * @brief Generate next 32 random bits.
 * @return Unsigned 32-bit random integer in [0, 2^32).
 */
uint32_t xorshift_uint32(void);

/**
 * @brief Generate a random float in [0, 1).
 * @return 32-bit float uniformly distributed in [0, 1).
 */
float xorshift_float(void);

/**
 * @brief Generate a Glorot/Xavier-scaled uniform random value in [-a, a].
 *
 * Uses the formula: a = sqrt(6 / (in + out))
 * Output is uniform in [-a, a]. Suitable for ML weight initialization.
 *
 * @param in  Input dimension (fan-in)
 * @param out Output dimension (fan-out)
 * @return 32-bit float uniform in [-a, a]
 */
float xorshift_xavier(size_t in, size_t out);

/**
 * @brief Generate a Glorot-scaled normal random value using Box-Muller transform.
 *
 * Standard normal (mean=0, variance=1), scaled by sqrt(2 / (in + out)).
 * Suitable for ML weight initialization.
 *
 * @param in  Input dimension (fan-in)
 * @param out Output dimension (fan-out)
 * @return 32-bit float from N(0, stddev^2)
 */
float xorshift_muller(size_t in, size_t out);

/**
 * @brief In-place Fisher–Yates shuffle of an array.
 *
 * Shuffles a buffer of @p n elements, each of size @p size bytes.
 *
 * @param base Pointer to buffer to shuffle
 * @param n    Number of elements in buffer
 * @param size Size in bytes of each element
 * @return true on success, false on failure
 */
bool xorshift_yates(void* base, size_t n, size_t size);

#ifdef __cplusplus
}
#endif

#endif  // XORSHIFT_H
