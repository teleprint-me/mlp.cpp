# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(mlp VERSION 0.1.0 DESCRIPTION "MLP written completely from scratch")

# clang: Invalid argument '-std=gnu17' not allowed with 'C++'
# C17 conflicts with C++17. The only way to resolve this is to set to C11.
# This seems to be a clang related bug. See cmake discourse forum for more info.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Required for clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(WARN "-Wall -Wextra -Wpedantic -Werror -Wformat-security -Wshadow -fexceptions")
set(EXTRA_WARN "-Wformat -Wnull-dereference -Wdouble-promotion")
set(SANITIZE "-fsanitize=address,undefined -fno-omit-frame-pointer")
set(ANALYSIS "-Wanalyzer-double-free -Wanalyzer-file-leak -Wanalyzer-malloc-leak -Wanalyzer-null-dereference -Wanalyzer-out-of-bounds -Wanalyzer-va-list-leak")
set(COMMON "-D_FILE_OFFSET_BITS=64 ${OpenMP_C_FLAGS} ${WARN}")

set(DEBUG "${COMMON} -fopenmp -g3 ${EXTRA_WARN} ${SANITIZE} ${ANALYSIS}")
set(RELEASE "${COMMON} -fopenmp -Ofast -march=native")

find_package(OpenMP REQUIRED)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DEBUG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEBUG}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RELEASE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RELEASE}")
endif()

# Set the output directory for built binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add core library as a dependency
# add_library(mlp SHARED
#     src/rng.cpp
# )
# target_include_directories(mlp PUBLIC include)
# target_link_libraries(mlp PRIVATE m rt)

# Add xor executable
add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE m)

# Add mlp executable
add_executable(mlp mlp.cpp)
target_link_libraries(mlp PRIVATE m)

add_custom_target(run_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all generated files"
)
