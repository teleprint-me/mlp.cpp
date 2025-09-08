/**
 * @license cc-by-sa-4.0
 * @file xorshift.c
 * @ref https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
 */

#include "xorshift.h"

struct XORShiftState state = {0};

void xorshift_init(uint64_t seed) {
    state.seed = seed;
}

void xorshift_gen(void) {
    state.seed ^= state.seed >> 12;
    state.seed ^= state.seed << 25;
    state.seed ^= state.seed >> 27;
}

uint32_t xorshift_gen_int32(void) {
    xorshift_gen();
    return (state.seed * 0x2545F4914F6CDD1Dull) >> 32;
}

float xorshift_gen_float(void) {
    return (xorshift_gen_int32() >> 8) / 16777216.0f;
}
