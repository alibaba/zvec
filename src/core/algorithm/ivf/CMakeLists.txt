include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_KNN_IVF_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_ivf STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework core_knn_cluster
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    LDFLAGS "${CORE_KNN_IVF_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
