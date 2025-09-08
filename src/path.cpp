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

// Concatenate two path components, inserting a '/' if needed
char* path_cat(const char* dir, const char* file) {
    if (!path_is_valid(dir) || !path_is_valid(file)) {
        return NULL;
    }

    size_t len_dir = strlen(dir);
    size_t len_file = strlen(file);
    int needs_slash = (len_dir > 0 && dir[len_dir - 1] != '/');
    size_t total_len = len_dir + needs_slash + len_file + 1;

    char* path = (char*) malloc(total_len);
    if (!path) {
        return NULL;
    }

    strcpy(path, dir);
    if (needs_slash) {
        strcat(path, "/");
    }
    strcat(path, file);

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
char** path_list_files(const char* path, size_t* count) {
    if (!path_is_dir(path)) {
        return NULL;
    }

    DIR* dir = opendir(path);
    if (!dir) {
        return NULL;
    }

    *count = 0;
    char** files = NULL;

    struct dirent* dir_entry;
    while ((dir_entry = readdir(dir))) {
        int current = strcmp(".", dir_entry->d_name);
        int previous = strcmp("..", dir_entry->d_name);

        char* entry = path_cat(path, dir_entry->d_name);
        if (0 == current || 0 == previous || !path_is_file(entry)) {
            free(entry);
            continue;
        }

        files = (char**) realloc(files, (*count + 1) * sizeof(char*));
        files[*count] = entry;
        *count += 1;
    }

    return files;
}
