/**
 * @file      mlp/src/path.cpp
 * @author    Austin Berrio
 * @copyright Copyright © 2025
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <unistd.h>
#include <sys/stat.h>

#include "path.h"

// Path existence and checks
bool path_is_valid(const char* path) {
    return path && *path != '\0';
}

// Checks if a path exists
bool path_exists(const char* path) {
    struct stat buffer;
    return path_is_valid(path) && stat(path, &buffer) == 0;
}

// Checks if a path is a directory
bool path_is_dir(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        return false;
    }
    return S_ISDIR(buffer.st_mode);
}

// Checks if a path is a regular file
bool path_is_file(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        return false;
    }
    return S_ISREG(buffer.st_mode);
}

// Saner mkdir wrapper
int path_mkdir(const char* path) {
    if (mkdir(path, 0755) == -1 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

char* path_dirname(const char* path) {
    if (!path_is_valid(path)) {
        return strdup("");  // Invalid input -> empty string
    }

    // Find the last slash
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) {
        return strdup(".");  // No slash -> current directory
    }

    // Handle root case (e.g., "/")
    if (last_slash == path) {
        return strdup("/");
    }

    // Extract the directory part
    size_t length = last_slash - path;
    char* dir = (char*) malloc(length + 1);
    if (!dir) {
        return strdup("");  // Fallback on allocation failure
    }

    strncpy(dir, path, length);
    dir[length] = '\0';
    return dir;
}

char* path_basename(const char* path) {
    if (!path_is_valid(path)) {
        return strdup("");  // Invalid input -> empty string
    }

    // Find the last slash
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) {
        return strdup(path);  // No slash -> whole path is basename
    }

    // Return the part after the last slash
    return strdup(last_slash + 1);
}

// Splits a path into components
char** path_split(const char* path, size_t* count) {
    if (!path || !*path) {
        return nullptr;
    }

    *count = 0;
    char** parts = nullptr;

    // Estimate components length and allocate memory
    char* temp = strdup(path);
    char* token = strtok(temp, "/");
    while (token) {
        parts = (char**) realloc(parts, (*count + 1) * sizeof(char*));
        parts[*count] = strdup(token);
        *count += 1;
        token = strtok(NULL, "/");
    }

    free(temp);
    return parts;
}

// Frees split path components
void path_split_free(char** parts, size_t count) {
    if (parts) {
        for (size_t i = 0; i < count; i++) {
            free(parts[i]);
        }
        free(parts);
    }
}
