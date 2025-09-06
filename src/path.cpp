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

// Path existence and checks

bool path_is_valid(const char* path) {
    return path && *path != '\0';
}

// Checks if a path exists
bool path_exists(const char* path) {
    return path_is_valid(path) && access(path, F_OK) == 0;
}

// Checks if a path is a directory
bool path_is_directory(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        fprintf(stderr, "Failed to stat path '%s': %s\n", path, strerror(errno));
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
        fprintf(stderr, "Failed to stat path '%s': %s\n", path, strerror(errno));
        return false;
    }
    return S_ISREG(buffer.st_mode);
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
