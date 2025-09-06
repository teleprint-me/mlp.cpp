// @file include/ckpt.h
#ifndef MLP_CKPT_H
#define MLP_CKPT_H

#include <cstdlib>

#include "mlp.h"

#define MLP_MAGIC 0x6D6C7000  // 'mlp\0'
#define MLP_VERSION 1

#define MLP_MAX_STAMP 64  // timestamp length
#define MLP_MAX_FNAME 256  // buffer length

// apply a formatted timestamp to out with length n
size_t mlp_ckpt_stamp(char* out, size_t n);

void mlp_ckpt_path(char* buffer, size_t n, const char* file_path);

void mlp_ckpt_name(char* buffer, size_t n, size_t epoch);

bool mlp_ckpt_save(struct MLP* mlp, const char* path);

bool mlp_ckpt_load(struct MLP* mlp, const char* path);

#endif  // MLP_CKPT_H
