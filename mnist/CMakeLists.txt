# mlp/mnist/CMakeLists.txt

set(BIN
    "mnist_train"
    "mnist_eval"
    # "mnist_inference"
)

set(INPUT_DIR ${PROJECT_SOURCE_DIR}/mnist)
set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build/mnist)

add_library(mnist SHARED
    "mnist_data.cpp"
)
target_include_directories(mnist PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mnist PRIVATE m)

foreach(exe IN LISTS BIN)
    add_executable(${exe} ${INPUT_DIR}/${exe}.cpp)
    target_link_libraries(${exe} mlp mnist)
    target_include_directories(${exe} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
endforeach()
