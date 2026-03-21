include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_knn_omega
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework core_knn_hnsw omega
    INCS . ${PROJECT_ROOT_DIR}/src/include ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm ${PROJECT_ROOT_DIR}/thirdparty/omega/OMEGALib/include
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
