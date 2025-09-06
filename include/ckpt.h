// @file include/ckpt.h
#ifndef MLP_CKPT_H
#define MLP_CKPT_H

#include <cstdlib>

#include "mlp.h"

#define MLP_MAGIC 0x6D6C7000  // 'mlp\0'
#define MLP_VERSION 1

#define MLP_MAX_STAMP 64  // timestamp length
#define MLP_MAX_FNAME 256  // buffer length

// apply a formatted file path to buffer with length n
void mlp_ckpt_path(char* buffer, size_t n, const char* dirname, const char* basename);

// apply a formatted timestamp to buffer with length n
void mlp_ckpt_stamp(char* buffer, size_t n, const char* dirname, size_t epoch);

bool mlp_ckpt_save(struct MLP* mlp, const char* path);

bool mlp_ckpt_load(struct MLP* mlp, const char* path);

#endif  // MLP_CKPT_H
