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

# Exclude omega algorithm files when OMEGA is disabled
if(NOT ZVEC_ENABLE_OMEGA)
    list(FILTER ALL_CORE_SRCS EXCLUDE REGEX ".*/algorithm/omega/.*")
    list(FILTER ALL_CORE_SRCS EXCLUDE REGEX ".*/omega_index\\.cc$")
endif()

# Conditionally link omega library
set(CORE_LIBS zvec_ailego sparsehash magic_enum)
if(ZVEC_ENABLE_OMEGA)
    list(APPEND CORE_LIBS omega)
endif()

cc_library(
    NAME zvec_core STATIC STRICT PACKED
    SRCS ${ALL_CORE_SRCS}
    LIBS ${CORE_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${GIT_SRCS_VER}"
)