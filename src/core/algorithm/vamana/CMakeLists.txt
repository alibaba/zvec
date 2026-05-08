include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_knn_vamana
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework core_knn_hnsw sparsehash
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
