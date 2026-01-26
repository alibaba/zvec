include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
        NAME core_interface STATIC STRICT ALWAYS_LINK
        SRCS *.cc indexes/*.cc
        INCS . ${PROJECT_ROOT_DIR}/src/ ${PROJECT_ROOT_DIR}/src/core
        LIBS zvec_ailego core_framework sparsehash magic_enum
        VERSION "${PROXIMA_ZVEC_VERSION}"
)
