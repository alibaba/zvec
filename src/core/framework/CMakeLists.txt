include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_framework 
    STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
