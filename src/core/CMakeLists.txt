include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(framework)
cc_directory(algorithm)
cc_directory(metric)
cc_directory(quantizer)
cc_directory(utility)
cc_directory(interface)
cc_directory(mixed_reducer)

git_version(GIT_SRCS_VER ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ALL_CORE_SRCS *.cc *.c *.h)

cc_library(
    NAME zvec_core STATIC STRICT PACKED
    SRCS ${ALL_CORE_SRCS}
    LIBS zvec_ailego sparsehash magic_enum
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${GIT_SRCS_VER}"
)