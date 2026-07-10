include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512 TURBO_MARCH_FLAG_AVX512FP16)
    elseif (HOST_ARCH MATCHES "^(arm|arm64)$")
        # ARM64 architecture - NEON is enabled by default on aarch64,
        # no special march flags needed.
        message(STATUS "turbo: ARM64 detected, NEON enabled by default")
    endif()
endif()

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

# Set per-file compile flags for SIMD sources.
# set_source_files_properties is directory-scoped, so it must be called in the
# same directory that adds the sources to a target (i.e. here, not in a
# subdirectory).
if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        # SSE
        file(GLOB_RECURSE SSE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/sse/*.cc)
        set_source_files_properties(
            ${SSE_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_SSE}"
        )

        # AVX2
        file(GLOB_RECURSE AVX2_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx2/*.cc)
        set_source_files_properties(
            ${AVX2_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX2}"
        )

        # AVX512 (avx512_vnni and avx512/fht both use AVX512 march)
        file(GLOB_RECURSE AVX512_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512/*.cc)
        set_source_files_properties(
            ${AVX512_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    endif()
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT ALWAYS_LINK PACKED
    SRCS ${ALL_SRCS}
    LIBS zvec_ailego
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distance ${PROJECT_ROOT_DIR}/src/include
)
