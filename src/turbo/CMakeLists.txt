include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
include(CheckCXXSourceCompiles)

# HOST_ARCH is only populated by the shared AUTO_DETECT_ARCH path. Resolve the
# CMake target locally as well so Turbo's ARM dispatch also works for Android
# cross builds and explicitly selected architectures.
set(_TURBO_TARGET_ARCH "${HOST_ARCH}")
if(NOT _TURBO_TARGET_ARCH)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _TURBO_SYSTEM_PROCESSOR)
  if(_TURBO_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|x64)$")
    set(_TURBO_TARGET_ARCH x64)
  elseif(_TURBO_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)$")
    set(_TURBO_TARGET_ARCH x86)
  elseif(_TURBO_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_TURBO_TARGET_ARCH arm64)
  elseif(_TURBO_SYSTEM_PROCESSOR MATCHES "^(arm|armv7|armv7-a|armv7l)$")
    set(_TURBO_TARGET_ARCH arm)
  else()
    set(_TURBO_TARGET_ARCH unknown)
  endif()
endif()

if(NOT ANDROID AND AUTO_DETECT_ARCH AND
   _TURBO_TARGET_ARCH MATCHES "^(x86|x64)$")
  setup_compiler_march_for_x86(
    TURBO_MARCH_FLAG_SSE2
    TURBO_MARCH_FLAG_AVX2
    TURBO_MARCH_FLAG_AVX512
    TURBO_MARCH_FLAG_AVX512FP16)
  # kSSE2 is gated by SSE2 at runtime. Keep these translation units at
  # that exact baseline so auto-vectorized scalar tails cannot acquire
  # instructions from the broader corei7 profile returned above.
  if(MSVC AND _TURBO_TARGET_ARCH STREQUAL "x86")
    set(TURBO_MARCH_FLAG_SSE2 "/arch:SSE2")
  elseif(NOT MSVC)
    if(_TURBO_TARGET_ARCH STREQUAL "x64")
      set(TURBO_MARCH_FLAG_SSE2 "-march=x86-64")
    else()
      set(TURBO_MARCH_FLAG_SSE2 "-march=i686 -msse2")
    endif()
  endif()
endif()

set(ZVEC_TURBO_ARM_FP16_SUPPORTED OFF)
set(TURBO_MARCH_FLAG_NEON_FP16 "")
if(NOT IOS AND NOT MSVC AND (AUTO_DETECT_ARCH OR ANDROID) AND
   _TURBO_TARGET_ARCH MATCHES "^(arm|arm64)$")
  # Keep this probe local to Turbo: only its neon_fp16 distance kernels need
  # FEAT_FP16, while the plain NEON kernels (which widen to FP32) and the
  # existing Ailego distance implementations stay untouched.
  # Android is kept even though build_android.sh sets AUTO_DETECT_ARCH=OFF:
  # that switch guards the host-oriented march detection, while this probe
  # runs against the actual (cross) target toolchain.
  # No fallback march is injected on probe failure: armv8-a is already the
  # aarch64 default, and forcing it onto 32-bit arm targets would be wrong.
  set(_TURBO_SAVED_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS
    "${_TURBO_SAVED_CMAKE_REQUIRED_FLAGS} -march=armv8.2-a+fp16")
  check_cxx_source_compiles([=[
    #include <arm_neon.h>
    #if !defined(__aarch64__)
    #error "AArch64 target required"
    #endif
    #if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #error "ARM FP16 vector arithmetic macro missing"
    #endif
    int main() {
      float16x8_t value = vdupq_n_f16(1.0f);
      value = vfmaq_f16(value, value, value);
      return vgetq_lane_u16(vreinterpretq_u16_f16(value), 0) == 0;
    }
  ]=] ZVEC_TURBO_COMPILER_SUPPORTS_ARM_FP16)
  set(CMAKE_REQUIRED_FLAGS "${_TURBO_SAVED_CMAKE_REQUIRED_FLAGS}")

  if(ZVEC_TURBO_COMPILER_SUPPORTS_ARM_FP16)
    set(TURBO_MARCH_FLAG_NEON_FP16 "-march=armv8.2-a+fp16")
    set(ZVEC_TURBO_ARM_FP16_SUPPORTED ON)
  endif()

  message(STATUS
    "turbo: NEON FP16 march flag: ${TURBO_MARCH_FLAG_NEON_FP16}, "
    "native kernels: ${ZVEC_TURBO_ARM_FP16_SUPPORTED}")
endif()

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

# Set per-file compile flags for SIMD sources.
# set_source_files_properties is directory-scoped, so it must be called in the
# same directory that adds the sources to a target (i.e. here, not in a
# subdirectory).
if(NOT ANDROID AND AUTO_DETECT_ARCH AND
   _TURBO_TARGET_ARCH MATCHES "^(x86|x64)$")
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

if(NOT IOS AND NOT MSVC AND (AUTO_DETECT_ARCH OR ANDROID) AND
   _TURBO_TARGET_ARCH MATCHES "^(arm|arm64)$")
    file(GLOB_RECURSE NEON_FP16_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/neon_fp16/*.cc)
    set_source_files_properties(
        ${NEON_FP16_SRCS}
        PROPERTIES
        COMPILE_FLAGS "${TURBO_MARCH_FLAG_NEON_FP16}"
    )
endif()

set(ZVEC_TURBO_DEFS)
if(ZVEC_TURBO_ARM_FP16_SUPPORTED)
    list(APPEND ZVEC_TURBO_DEFS ZVEC_HAVE_ARM_FP16_KERNEL=1)
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
    DEFS ${ZVEC_TURBO_DEFS}
    LIBS zvec_ailego
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distance ${PROJECT_ROOT_DIR}/src/include
)
