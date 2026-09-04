include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_KNN_VAMANA_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_vamana
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    # core_framework is intentionally not listed: cc_library already appends the
    # ALWAYS_LINK deps reached through core_knn_hnsw (core_utility, then
    # core_framework) in dependency order. Naming it here would pin it ahead of
    # core_utility, and CMake would then repeat the archive to keep the link
    # order valid (Apple's ld warns "ignoring duplicate libraries").
    LIBS core_knn_hnsw sparsehash
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    LDFLAGS "${CORE_KNN_VAMANA_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
