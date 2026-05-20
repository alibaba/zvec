include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
#message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

if(NOT APPLE)
  set(CORE_KNN_FLAT_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_flat 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework 
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm ${PROJECT_ROOT_DIR}/src/core/framework
    LDFLAGS "${CORE_KNN_FLAT_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
