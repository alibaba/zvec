include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_rotator
    STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework rabitqlib
    INCS . .. ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
