include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_metric 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego zvec_turbo core_framework 
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
