include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(cluster)
cc_directory(flat)
cc_directory(flat_sparse)
cc_directory(ivf)
cc_directory(hnsw)
cc_directory(hnsw_sparse)

# Only include omega when ZVEC_ENABLE_OMEGA is ON
if(ZVEC_ENABLE_OMEGA)
    cc_directory(omega)
endif()