include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_KNN_IVF_RABITQ_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_ivf_rabitq
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework rabitqlib core_knn_cluster core_knn_hnsw_rabitq
    INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    LDFLAGS "${CORE_KNN_IVF_RABITQ_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
