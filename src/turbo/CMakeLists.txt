include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# HOST_ARCH is only populated by the shared AUTO_DETECT_ARCH path. Resolve the
# CMake target locally as well so Turbo's ARM dispatch also works for Android
# cross builds and explicitly selected architectures.
set(_TURBO_TARGET_ARCH "${HOST_ARCH}")
set(_TURBO_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  # An explicit target takes precedence over the build host (e.g. Intel Mac
  # cross-compiling for Apple Silicon). Multi-architecture builds fail the
  # single-ISA probes safely and retain the baseline kernels.
  set(_TURBO_TARGET_ARCH "")
  set(_TURBO_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(NOT _TURBO_TARGET_ARCH)
  string(TOLOWER "${_TURBO_SYSTEM_PROCESSOR}" _TURBO_SYSTEM_PROCESSOR)
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
  # Use explicit ISA sets instead of a whole CPU model. For example,
  # -march=icelake-server can emit instructions absent on Skylake AVX512 CPUs.
  if(MSVC)
    if(_TURBO_TARGET_ARCH STREQUAL "x86")
      set(TURBO_MARCH_FLAG_SSE2 "/arch:SSE2")
    endif()
    set(_TURBO_AVX2_FLAGS "/arch:AVX2")
    set(_TURBO_AVX512_FLAGS "/arch:AVX512")
    set(_TURBO_VNNI_FLAGS "/arch:AVX512")
    set(_TURBO_FP16_FLAGS "/arch:AVX512")
  else()
    if(_TURBO_TARGET_ARCH STREQUAL "x64")
      set(TURBO_MARCH_FLAG_SSE2 "-march=x86-64")
    else()
      set(TURBO_MARCH_FLAG_SSE2 "-march=i686 -msse2")
    endif()
    set(_TURBO_AVX2_FLAGS
      "${TURBO_MARCH_FLAG_SSE2} -mavx2 -mfma -mf16c")
    set(_TURBO_AVX512_FLAGS
      "${_TURBO_AVX2_FLAGS} -mavx512f -mavx512bw -mavx512dq -mavx512vl")
    set(_TURBO_VNNI_FLAGS "${_TURBO_AVX512_FLAGS} -mavx512vnni")
    set(_TURBO_FP16_FLAGS "${_TURBO_AVX512_FLAGS} -mavx512fp16")
  endif()
endif()

# Check the same feature macros used by the implementation, with the same
# per-source flags. A missing ISA produces empty explicit dispatch rather than
# selecting a no-op stub. This also covers AUTO_DETECT_ARCH=OFF cross builds.
function(turbo_probe_x86 ISA CONDITION)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "${_TURBO_${ISA}_FLAGS}")
  unset(ZVEC_TURBO_COMPILER_${ISA} CACHE)
  check_cxx_source_compiles("
    #include <immintrin.h>
    #if !(${CONDITION})
    #error Kernel ISA unavailable
    #endif
    int main() { return 0; }
  " ZVEC_TURBO_COMPILER_${ISA})
  cmake_pop_check_state()
  if(ZVEC_TURBO_COMPILER_${ISA})
    set(TURBO_MARCH_FLAG_${ISA} "${_TURBO_${ISA}_FLAGS}" PARENT_SCOPE)
  else()
    set(TURBO_MARCH_FLAG_${ISA} "" PARENT_SCOPE)
  endif()
endfunction()

turbo_probe_x86(AVX2 "defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER)) && (defined(__F16C__) || defined(_MSC_VER))")
turbo_probe_x86(AVX512 "defined(__AVX2__) && defined(__AVX512F__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)))")
turbo_probe_x86(VNNI "(defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__) && defined(__FMA__) && defined(__F16C__)) || (defined(_MSC_VER) && defined(__AVX512F__))")
turbo_probe_x86(FP16 "defined(__AVX512FP16__)")

set(ZVEC_TURBO_ARM_FP16_SUPPORTED OFF)
set(TURBO_MARCH_FLAG_NEON_FP16 "")
if(NOT IOS AND NOT MSVC AND
   _TURBO_TARGET_ARCH MATCHES "^(arm|arm64)$")
  # Keep this probe local to Turbo: only its neon_fp16 distance kernels need
  # FEAT_FP16, while the plain NEON kernels (which widen to FP32) and the
  # existing Ailego distance implementations stay untouched.
  # Android is kept even though build_android.sh sets AUTO_DETECT_ARCH=OFF:
  # that switch guards the host-oriented march detection, while this probe
  # runs against the actual (cross) target toolchain.
  # No fallback march is injected on probe failure: armv8-a is already the
  # aarch64 default, and forcing it onto 32-bit arm targets would be wrong.
  set(_TURBO_ARM_FP16_FLAGS "")
  if(AUTO_DETECT_ARCH OR ANDROID)
    set(_TURBO_ARM_FP16_FLAGS "-march=armv8.2-a+fp16")
  endif()
  set(_TURBO_SAVED_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS
    "${_TURBO_SAVED_CMAKE_REQUIRED_FLAGS} ${_TURBO_ARM_FP16_FLAGS}")
  unset(ZVEC_TURBO_COMPILER_SUPPORTS_ARM_FP16 CACHE)
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
    set(TURBO_MARCH_FLAG_NEON_FP16 "${_TURBO_ARM_FP16_FLAGS}")
    set(ZVEC_TURBO_ARM_FP16_SUPPORTED ON)
  endif()

  message(STATUS
    "turbo: NEON FP16 march flag: ${TURBO_MARCH_FLAG_NEON_FP16}, "
    "native kernels: ${ZVEC_TURBO_ARM_FP16_SUPPORTED}")
endif()

option(ZVEC_TURBO_ENABLE_SME2 "Build runtime-gated Apple SME2 batch kernels" ON)
set(ZVEC_TURBO_HAVE_SME2 OFF CACHE INTERNAL "Turbo SME2 kernels built" FORCE)
unset(ZVEC_TURBO_COMPILER_SUPPORTS_SME2 CACHE)
# Test the target toolchain, not the host CPU. Older SDKs and non-Apple targets
# retain the NEON/scalar dispatch without a dependency on SME support routines.
if(ZVEC_TURBO_ENABLE_SME2 AND APPLE AND NOT IOS AND
   CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
   _TURBO_TARGET_ARCH STREQUAL "arm64" AND
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  block(SCOPE_FOR VARIABLES)
    cmake_reset_check_state()
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    # Apple only supports streaming SVE. armv9-a would allow illegal ordinary
    # SVE instructions in an otherwise non-streaming function's prologue.
    set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sme2")
    check_cxx_source_compiles([=[
      #include <arm_sme.h>
      #if !defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
      #error Requires AArch64 SME2 without non-streaming SVE
      #endif
      __arm_locally_streaming __arm_new("za")
      __attribute__((noinline)) void probe(float *p) {
        svzero_za();
        const auto v = svld1_f32_x4(svptrue_c32(), p);
        svmls_single_za32_f32_vg1x4(0, v, svget4_f32(v, 0));
        svmls_za32_f32_vg1x4(0, v, v);
        svadd_za32_f32_vg1x4(0, svread_za32_f32_vg1x4(0));
        const auto r = svread_za32_f32_vg1x4(0);
        svst1_f32(svptrue_b32(), p, svget4_f32(r, 0));
      }
      int main() { float p[256] = {}; probe(p); return int(p[0]); }
    ]=] ZVEC_TURBO_COMPILER_SUPPORTS_SME2)
  endblock()
  if(ZVEC_TURBO_COMPILER_SUPPORTS_SME2)
    set(ZVEC_TURBO_HAVE_SME2 ON CACHE INTERNAL "Turbo SME2 kernels built" FORCE)
  endif()
endif()
message(STATUS "turbo: SME2 FP32 batch kernels: ${ZVEC_TURBO_HAVE_SME2}")

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)
file(GLOB_RECURSE SME2_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/sme2/*.cc)
if(ZVEC_TURBO_HAVE_SME2)
  # Only the kernel TU gets SME2 instructions. The public dispatcher, callers
  # and tests must remain runnable on every supported Apple Silicon Mac.
  set_source_files_properties(${SME2_SRCS} PROPERTIES
      COMPILE_FLAGS "-march=armv8.6-a+sme2")
elseif(SME2_SRCS)
  list(REMOVE_ITEM ALL_SRCS ${SME2_SRCS})
endif()

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
        COMPILE_FLAGS "${TURBO_MARCH_FLAG_FP16}"
    )

    # AVX512
    file(GLOB_RECURSE AVX512_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/conversion/avx512/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/fp16/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/raw_uint8/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512/*.cc
    )
    set_source_files_properties(
        ${AVX512_SRCS}
        PROPERTIES
        COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
    )
    file(GLOB_RECURSE VNNI_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/record_quantized_int8/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/uniform_uint4/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/uniform_uint7/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/uniform_uint8/*.cc)
    set_source_files_properties(${VNNI_SRCS} PROPERTIES
        COMPILE_FLAGS "${TURBO_MARCH_FLAG_VNNI}")
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
foreach(_isa AVX2 AVX512 VNNI FP16)
  if(ZVEC_TURBO_COMPILER_${_isa})
    list(APPEND ZVEC_TURBO_DEFS ZVEC_HAVE_${_isa}_KERNEL=1)
  endif()
endforeach()
if(ZVEC_TURBO_HAVE_SME2)
    list(APPEND ZVEC_TURBO_DEFS ZVEC_HAVE_SME2=1)
endif()
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

set(ZVEC_TURBO_RABITQ_LIBS "")
if(RABITQ_SUPPORTED)
  # Share Eigen's ABI settings with RaBitQ when both are linked into zvec.
  list(APPEND ZVEC_TURBO_RABITQ_LIBS rabitqlib::core)
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT ALWAYS_LINK PACKED
    ${ZVEC_TURBO_LIBRARY_OPTIONS}
    SRCS ${ALL_SRCS}
    DEFS ${ZVEC_TURBO_DEFS}
    LIBS zvec_ailego ${ZVEC_TURBO_RABITQ_LIBS}
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distance ${PROJECT_ROOT_DIR}/src/include
)

# Eigen (header-only, shipped inside the RaBitQ third-party tree) backs the OPQ
# rotator only. It must be a SYSTEM include: INCS above maps to a plain
# target_include_directories, and STRICT turns Eigen's own warnings into errors.
set(TURBO_EIGEN_INC
    ${PROJECT_ROOT_DIR}/thirdparty/RaBitQ-Library/RaBitQ-Library-0.7.0/include)
foreach(_turbo_tgt zvec_turbo zvec_turbo_objects zvec_turbo_export_objects)
  if(TARGET ${_turbo_tgt})
    target_include_directories(${_turbo_tgt} SYSTEM PRIVATE ${TURBO_EIGEN_INC})
  endif()
endforeach()
