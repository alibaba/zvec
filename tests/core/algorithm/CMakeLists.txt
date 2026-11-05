include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_gtest(
  NAME streamer_layout_test
  STRICT
  LIBS zvec_ailego core_framework core_utility core_knn_hnsw core_knn_hnsw_sparse
  SRCS streamer_layout_test.cc streamer_sparse_layout_test.cc
  INCS ${PROJECT_ROOT_DIR}/src/core
)

cc_directories(cluster)
cc_directories(flat)
cc_directories(flat_sparse)
cc_directories(ivf)
cc_directories(hnsw)
cc_directories(hnsw_sparse)
cc_directories(vamana)

if(DISKANN_SUPPORTED)
  message(STATUS "build diskann tests")
  cc_directory(diskann)
else()
  message(STATUS "skip diskann tests (unsupported platform)")
endif()

if(RABITQ_SUPPORTED)
  cc_directories(hnsw_rabitq)
  cc_directories(ivf_rabitq)
endif()
