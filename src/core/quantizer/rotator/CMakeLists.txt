include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# No rabitqlib dependency: matrix_rotator.cc implements Householder QR
# manually with plain C++ to avoid ISA-sensitive Eigen inline functions
# that cause ODR violations (duplicate codegen with different -march flags)
# leading to SEGFAULT on linux-x64-clang.

cc_library(
    NAME core_rotator
    STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework
    INCS . .. ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
