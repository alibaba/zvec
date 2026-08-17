include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_KNN_HNSW_RABITQ_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_hnsw_rabitq
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework rabitqlib sparsehash
    INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    CXXFLAGS "${_RABITQ_MARCH_AVX2}"
    LDFLAGS "${CORE_KNN_HNSW_RABITQ_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
