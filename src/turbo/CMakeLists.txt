include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        # ARM64 architecture - no special march flags needed for now
        # NEON implementations can be added here if needed
        message(STATUS "turbo: ARM64 detected, skipping x86-specific optimizations")
    endif()
endif()

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

# Set per-file compile flags for AVX512-VNNI sources.
# set_source_files_properties is directory-scoped, so it must be called in the
# same directory that adds the sources to a target (i.e. here, not in a
# subdirectory).
if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
        file(GLOB_RECURSE AVX512_VNNI_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/avx512_vnni/*.cc)
        set_source_files_properties(
            ${AVX512_VNNI_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    endif()
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT PACKED
    SRCS ${ALL_SRCS}
    LIBS zvec_ailego
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/include
)
