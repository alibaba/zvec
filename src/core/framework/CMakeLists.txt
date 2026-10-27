include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Use the SDK's checked-in C++ catalog as the default for component consumers.
# The SDK compiles framework sources directly, so this creates no target cycle.
# A component-only application can supply its own C++ catalog instead.
cc_library(
    NAME core_framework 
    STATIC STRICT
    SRCS *.cc
    LIBS zvec_ailego zvec_turbo zvec_core
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/turbo
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
