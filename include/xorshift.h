/**
 * @license cc-by-sa-4.0
 * @file mlp/include/xorshift.h
 * @ref https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
 */

#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <cstdint>

struct XORShiftState {
    uint64_t seed;
};

void xorshift_init(uint64_t seed);

void xorshift_gen(void);

/**
 * xorshift rng: generate next step in sequence [0, 2^64).
 */
uint32_t xorshift_gen_int32(void);

/**
 * xorshift rng: normalize rng state [0, 1).
 */
float xorshift_gen_float(void);

#endif  // RNG_XORSHIFT_H
