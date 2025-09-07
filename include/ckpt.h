/**
 * @file      mlp/include/ckpt.h
 * @author    Austin Berrio
 * @copyright Copyright © 2025
 */

#ifndef MLP_CKPT_H
#define MLP_CKPT_H

#include <cstdlib>

#include "mlp.h"

#define MLP_MAGIC 0x6D6C7000  // 'mlp\0'
#define MLP_VERSION 1

#define MLP_MAX_STAMP 64  // timestamp length
#define MLP_MAX_FNAME 256  // buffer length

size_t mlp_ckpt_max_path_len(const char* path);

char* mlp_ckpt_dirname(const char* path);

char* mlp_ckpt_basename(const char* path);

void mlp_ckpt_path(char* buffer, size_t n, const char* dirname, const char* basename);

void mlp_ckpt_stamp(char* buffer, size_t n, const char* dirname, const char* name, size_t epoch);

bool mlp_ckpt_save(struct MLP* mlp, const char* path);

bool mlp_ckpt_load(struct MLP* mlp, const char* path);

#endif  // MLP_CKPT_H
