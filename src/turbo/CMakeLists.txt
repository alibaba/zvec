include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE2 TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512 TURBO_MARCH_FLAG_AVX512FP16)
        # kSSE2 is gated by SSE2 at runtime. Keep these translation units at
        # that exact baseline so auto-vectorized scalar tails cannot acquire
        # instructions from the broader corei7 profile returned above.
        if(MSVC AND HOST_ARCH STREQUAL "x86")
            set(TURBO_MARCH_FLAG_SSE2 "/arch:SSE2")
        elseif(NOT MSVC)
            if(HOST_ARCH STREQUAL "x64")
                set(TURBO_MARCH_FLAG_SSE2 "-march=x86-64")
            else()
                set(TURBO_MARCH_FLAG_SSE2 "-march=i686 -msse2")
            endif()
        endif()
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
        # SSE2
        file(GLOB_RECURSE SSE2_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/sse2/*.cc)
        set_source_files_properties(
            ${SSE2_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_SSE2}"
        )

        # AVX2
        file(GLOB_RECURSE AVX2_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx2/*.cc)
        set_source_files_properties(
            ${AVX2_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX2}"
        )

        # AVX512-FP16
        file(GLOB_RECURSE AVX512FP16_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_fp16/*.cc
        )
        set_source_files_properties(
            ${AVX512FP16_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512FP16}"
        )

        # AVX512
        file(GLOB_RECURSE AVX512_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/conversion/avx512/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512/*.cc
        )
        set_source_files_properties(
            ${AVX512_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    endif()
endif()

set(ZVEC_TURBO_LIBRARY_OPTIONS)
if(WIN32 AND (BUILD_ZVEC_CORE_SHARED OR BUILD_ZVEC_SHARED))
  list(APPEND ZVEC_TURBO_LIBRARY_OPTIONS OBJECTS)
  if(BUILD_ZVEC_CORE_SHARED)
    list(APPEND ZVEC_TURBO_LIBRARY_OPTIONS
      EXPORT_DEF ZVEC_TURBO_BUILD_SHARED
    )
  endif()
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT ALWAYS_LINK PACKED
    ${ZVEC_TURBO_LIBRARY_OPTIONS}
    SRCS ${ALL_SRCS}
    LIBS zvec_ailego
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distance ${PROJECT_ROOT_DIR}/src/include
)
