/**
 * @file      mlp/src/path.c
 * @author    Austin Berrio
 * @copyright Copyright © 2025
 * @brief     A POSIX C pathlib interface.
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include "path.h"

// Checks if path is valid input
bool path_is_valid(const char* path) {
    return path && *path != '\0';
}

// Checks if a path exists
bool path_exists(const char* path) {
    struct stat buffer;
    return path_is_valid(path) && stat(path, &buffer) == 0;
}

// Checks if path is a directory
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

// Checks if path is a regular file
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

// Returns the directory path
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

// Returns the file name
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

// Concatenate path components
char* path_cat(const char* dst, const char* src) {
    if (!path_is_valid(dst) || !path_is_valid(src)) {
        return NULL;  // Invalid inputs
    }

    // Allocate mem to new path
    size_t path_len = strlen(dst) + strlen(src) + 1;
    char* path = (char*) malloc(path_len);
    if (!path) {
        return NULL;
    }

    // Glue dst and src to path
    strcpy(path, dst);
    strcat(path, src);
    return path;
}

// Splits a path into components
char** path_split(const char* path, size_t* count) {
    if (!path_is_valid(path)) {
        return NULL;
    }

    *count = 0;
    char** parts = NULL;

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

// Read directory contents into memory
char** path_list(const char* path, size_t* count) {
    if (!path_is_dir(path)) {
        return NULL;
    }

    DIR* dir = opendir(path);
    if (!dir) {
        return NULL;
    }

    *count = 0;
    char** list = NULL;

    return list;
}
