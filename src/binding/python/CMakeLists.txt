include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

set(SRC_LISTS
        binding.cc
        model/python_collection.cc
        model/python_doc.cc
        model/python_reranker.cc
        model/param/python_param.cc
        model/schema/python_schema.cc
        model/common/python_config.cc
        typing/python_type.cc
)

pybind11_add_module(_zvec ${SRC_LISTS})

# Ensure any change to the linker version script (exports.map) triggers a
# re-link of _zvec.so. target_link_options() alone is a command-line flag
# and does not register the script file as a build dependency, so stale
# wheels would silently keep the old export set otherwise.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET _zvec APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
elseif (APPLE)
    set_property(TARGET _zvec APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/exports.mac")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(_zvec PRIVATE
            -Wl,--whole-archive
            $<TARGET_FILE:core_knn_flat_static>
            $<TARGET_FILE:core_knn_flat_sparse_static>
            $<TARGET_FILE:core_knn_hnsw_static>
            $<TARGET_FILE:core_knn_hnsw_rabitq_static>
            $<TARGET_FILE:core_knn_hnsw_sparse_static>
            $<TARGET_FILE:core_knn_ivf_static>
            $<TARGET_FILE:core_knn_vamana_static>
            $<TARGET_FILE:core_knn_cluster_static>
            $<TARGET_FILE:core_knn_diskann_static>
            $<TARGET_FILE:core_mix_reducer_static>
            $<TARGET_FILE:core_metric_static>
            $<TARGET_FILE:core_utility_static>
            $<TARGET_FILE:core_quantizer_static>
            -Wl,--no-whole-archive
            zvec
            ${CMAKE_DL_LIBS}
    )
    target_link_options(_zvec PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
    # DiskAnn is now statically linked into _zvec.so via --whole-archive, so
    # no separate runtime .so is shipped in the wheel. libaio is still loaded
    # at runtime via dlopen() (see libaio_loader.h); if it is missing, DiskAnn
    # fails cleanly with an actionable error while other index types
    # (HNSW/IVF/Flat/Vamana) remain fully functional.
elseif (APPLE)
    target_link_libraries(_zvec PRIVATE
            -Wl,-force_load,$<TARGET_FILE:core_knn_flat_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_flat_sparse_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_rabitq_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_sparse_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_ivf_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_vamana_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_cluster_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_diskann_static>
            -Wl,-force_load,$<TARGET_FILE:core_mix_reducer_static>
            -Wl,-force_load,$<TARGET_FILE:core_metric_static>
            -Wl,-force_load,$<TARGET_FILE:core_utility_static>
            -Wl,-force_load,$<TARGET_FILE:core_quantizer_static>
            zvec
    )
    target_link_libraries(_zvec PRIVATE
            -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.mac
    )
elseif (MSVC)
    set(_zvec_whole_archive_libs
            core_knn_flat_static
            core_knn_flat_sparse_static
            core_knn_hnsw_static
            core_knn_hnsw_sparse_static
            core_knn_ivf_static
            core_knn_vamana_static
            core_knn_cluster_static
            core_knn_diskann_static
            core_mix_reducer_static
            core_metric_static
            core_utility_static
            core_quantizer_static
    )
    target_link_libraries(_zvec PRIVATE
            ${_zvec_whole_archive_libs}
            zvec
    )
    foreach(_lib ${_zvec_whole_archive_libs})
        target_link_options(_zvec PRIVATE
                "/WHOLEARCHIVE:$<TARGET_FILE:${_lib}>"
        )
    endforeach()
endif ()

target_include_directories(_zvec PRIVATE ${PYBIND11_INCLUDE_DIR} ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/binding/python/include)