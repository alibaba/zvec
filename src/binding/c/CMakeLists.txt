include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── nlohmann/json (header-only) ──────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests   OFF CACHE BOOL "" FORCE)
set(JSON_Install      OFF CACHE BOOL "" FORCE)
set(JSON_MultipleHeaders OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ── Build zvec_c as a shared library ─────────────────────────────────────────
add_library(zvec_c SHARED zvec_c.cc)

target_include_directories(zvec_c
    PRIVATE
        ${PROJECT_ROOT_DIR}/src
        ${PROJECT_ROOT_DIR}/src/include
)

target_link_libraries(zvec_c PRIVATE nlohmann_json::nlohmann_json)

# Replicate the same whole-archive / force_load strategy used by the Python
# binding so every algorithm implementation is included in the shared lib.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(zvec_c PRIVATE
        -Wl,--whole-archive
        $<TARGET_FILE:core_knn_flat_static>
        $<TARGET_FILE:core_knn_flat_sparse_static>
        $<TARGET_FILE:core_knn_hnsw_static>
        $<TARGET_FILE:core_knn_hnsw_sparse_static>
        $<TARGET_FILE:core_knn_ivf_static>
        $<TARGET_FILE:core_knn_cluster_static>
        $<TARGET_FILE:core_mix_reducer_static>
        $<TARGET_FILE:core_metric_static>
        $<TARGET_FILE:core_utility_static>
        $<TARGET_FILE:core_quantizer_static>
        -Wl,--no-whole-archive
        zvec_db
    )
elseif (APPLE)
    target_link_libraries(zvec_c PRIVATE
        -Wl,-force_load,$<TARGET_FILE:core_knn_flat_static>
        -Wl,-force_load,$<TARGET_FILE:core_knn_flat_sparse_static>
        -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_static>
        -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_sparse_static>
        -Wl,-force_load,$<TARGET_FILE:core_knn_ivf_static>
        -Wl,-force_load,$<TARGET_FILE:core_knn_cluster_static>
        -Wl,-force_load,$<TARGET_FILE:core_mix_reducer_static>
        -Wl,-force_load,$<TARGET_FILE:core_metric_static>
        -Wl,-force_load,$<TARGET_FILE:core_utility_static>
        -Wl,-force_load,$<TARGET_FILE:core_quantizer_static>
        zvec_db
    )
endif()

# Install so that Rust's build.rs (cmake crate) can find it under $OUT_DIR/lib
install(TARGETS zvec_c
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib   # Windows DLL goes here too
)
install(FILES zvec_c.h DESTINATION include)
