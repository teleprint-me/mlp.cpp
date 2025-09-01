# mlp/xor/CMakeLists.txt

set(BIN
    "train"
    "inspect"
    "inference"
)

set(INPUT_DIR ${PROJECT_SOURCE_DIR}/xor)
set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build/xor)

foreach(exe IN LISTS BIN)
    add_executable(${exe} ${INPUT_DIR}/${exe}.cpp)
    target_link_libraries(${exe} "mlp")
    target_include_directories(${exe} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
endforeach()
