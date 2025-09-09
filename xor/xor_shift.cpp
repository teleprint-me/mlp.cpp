// @file xor/xor_shift.cpp
#include "xorshift.h"
#include <cstdio>
#include <vector>

#define SEED 1337
#define GENERATIONS 20

int main(void) {
    // integer [0, UINT32_MAX)
    xorshift_init(SEED);
    printf("xorshift_uint32\n");
    for (size_t i = 0; i < GENERATIONS; i++) {
        uint32_t x = xorshift_uint32();
        printf("x[%zu] %u\n", i, x);
    }
    printf("\n");

    // float [0, 1]
    xorshift_init(SEED);
    printf("xorshift_float\n");
    for (size_t i = 0; i < GENERATIONS; i++) {
        float x = xorshift_float();
        printf("x[%zu] %f\n", i, x);
    }
    printf("\n");

    // float [-a, +a)
    xorshift_init(SEED);
    printf("xorshift_xavier\n");
    for (size_t i = 0; i < GENERATIONS; i++) {
        float x = xorshift_xavier(/* in */ 3, /* out */ 2);
        printf("x[%zu] %f\n", i, x);
    }
    printf("\n");

    // float [0, 1)
    xorshift_init(SEED);
    printf("xorshift_muller\n");
    for (size_t i = 0; i < GENERATIONS; i++) {
        float x = xorshift_muller(/* in */ 3, /* out */ 2);
        printf("x[%zu] %f\n", i, x);
    }
    printf("\n");

    xorshift_init(SEED);
    printf("xorshift_yates\n");

    printf("before\n");
    std::vector<size_t> x(GENERATIONS);
    for (size_t i = 0; i < GENERATIONS; i++) {
        x[i] = i + 1;
        printf("%zu ", x[i]);
    }

    printf("\nafter\n");
    xorshift_yates(x.data(), x.size(), sizeof(size_t));
    for (size_t i = 0; i < x.size(); i++) {
        printf("%zu ", x[i]);
    }
    printf("\n");

    return 0;
}
