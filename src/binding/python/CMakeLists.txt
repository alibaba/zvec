include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

set(SRC_LISTS
        binding.cc
        model/python_collection.cc
        model/python_doc.cc
        model/param/python_param.cc
        model/schema/python_schema.cc
        model/common/python_config.cc
        typing/python_type.cc
)

pybind11_add_module(_zvec ${SRC_LISTS})

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(_zvec PRIVATE
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
    target_link_options(_zvec PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
elseif (APPLE)
    target_link_libraries(_zvec PRIVATE
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
    target_link_libraries(_zvec PRIVATE
            -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.mac
    )
endif ()

target_include_directories(_zvec PRIVATE ${PYBIND11_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/binding/python/include)
