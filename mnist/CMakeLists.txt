# mlp/mnist/CMakeLists.txt

set(BIN
    "mnist_train"
    # "inspect"
    # "inference"
)

set(INPUT_DIR ${PROJECT_SOURCE_DIR}/mnist)
set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build/mnist)

foreach(exe IN LISTS BIN)
    add_executable(${exe} ${INPUT_DIR}/${exe}.cpp)
    target_link_libraries(${exe} "mlp")
    target_include_directories(${exe} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
endforeach()
