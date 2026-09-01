include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512 TURBO_MARCH_FLAG_AVX512FP16)
    elseif (HOST_ARCH MATCHES "^(arm|arm64)$")
        # ARM64 architecture - NEON is enabled by default on aarch64.
        # Probe the NEON FP16 march flag (armv8.2-a+fp16) for the runtime-
        # dispatched native FP16 kernels; falls back to armv8-a when the
        # compiler cannot target FEAT_FP16.
        setup_compiler_march_for_arm(TURBO_MARCH_FLAG_NEON TURBO_MARCH_FLAG_NEON_FP16)
        message(STATUS "turbo: ARM64 detected, NEON FP16 march flag: ${TURBO_MARCH_FLAG_NEON_FP16}")
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
    elseif (HOST_ARCH MATCHES "^(arm|arm64)$")
        # NEON FP16 (FEAT_FP16): selected at runtime in turbo.cc via the
        # CpuFeatures FP16 flag; when the compiler cannot target
        # armv8.2-a+fp16 the kernels compile as scalar delegates.
        file(GLOB_RECURSE NEON_FP16_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/neon_fp16/*.cc)
        set_source_files_properties(
            ${NEON_FP16_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_NEON_FP16}"
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
