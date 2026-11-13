# Graph traversal and codec adapters are header-only. Computational kernels
# are compiled once by Turbo, including its per-ISA source flags.
add_library(core_quantized_graph INTERFACE)
target_include_directories(core_quantized_graph INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/core)
target_link_libraries(core_quantized_graph INTERFACE
    core_framework zvec_ailego zvec_turbo rabitqlib)
