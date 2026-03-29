include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/option.cmake)

cc_directories(cluster)
cc_directories(flat)
cc_directories(flat_sparse)
cc_directories(ivf)
cc_directories(hnsw)
cc_directories(hnsw_sparse)
