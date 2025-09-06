/**
 * @file      mlp/include/path.h
 * @author    Austin Berrio
 * @copyright Copyright © 2025
 */

#ifndef MLP_PATH_H
#define MLP_PATH_H

bool path_is_valid(const char* path);

// Checks if a path exists
bool path_exists(const char* path);

// Checks if a path is a directory
bool path_is_dir(const char* path);

// Checks if a path is a regular file
bool path_is_file(const char* path);

// Saner mkdir wrapper
int path_mkdir(const char* path);

char* path_dirname(const char* path);

char* path_basename(const char* path);

#endif  // MLP_PATH_H
