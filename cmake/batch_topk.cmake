# Keep ISA flags local to BatchTopK kernels. The dispatcher is compiled for
# the baseline target and selects only kernels supported by the running CPU.
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

function(zvec_configure_batch_topk)
  set(_dir "${PROJECT_ROOT_DIR}/src/core/utility")
  set(_definitions)
  # CMake caches probes by variable name. Recheck when an existing build tree
  # changes target/SDK/flags, rather than retaining another target's kernels.
  string(SHA256 _signature
    "${CMAKE_CXX_COMPILER}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_COMPILER_TARGET}|${CMAKE_SYSTEM_NAME}|${CMAKE_SYSTEM_PROCESSOR}|${CMAKE_OSX_ARCHITECTURES}|${CMAKE_SYSROOT}|${CMAKE_OSX_SYSROOT}|${CMAKE_CXX_FLAGS}")
  if(NOT "${ZVEC_TOPK_PROBE_SIGNATURE}" STREQUAL "${_signature}")
    foreach(_probe TARGET_X86 COMPILE_SSE2 COMPILE_AVX2 COMPILE_AVX512 COMPILE_NEON COMPILE_ARMV8)
      unset(ZVEC_TOPK_${_probe} CACHE)
    endforeach()
    set(ZVEC_TOPK_PROBE_SIGNATURE "${_signature}" CACHE INTERNAL
      "BatchTopK compiler probe target" FORCE)
  endif()
  check_cxx_source_compiles("
    #if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64) && !defined(_M_IX86)
    #error Not an x86 target
    #endif
    int main() { return 0; }" ZVEC_TOPK_TARGET_X86)
  if(ZVEC_TOPK_TARGET_X86)
    if(MSVC)
      if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_baseline /arch:SSE2)
      else()
        set(_baseline /arch:IA32)
      endif()
      set(_sse2 /arch:SSE2)
      set(_avx2 /arch:AVX2)
      set(_avx512 /arch:AVX512)
    else()
      if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_baseline -march=x86-64)
      else()
        set(_baseline -march=i686)
      endif()
      set(_sse2 "${_baseline};-msse2")
      set(_avx2 "${_baseline};-mavx2")
      set(_avx512 "${_baseline};-mavx512f")
    endif()
    set_source_files_properties("${_dir}/batch_topk.cc"
      PROPERTIES COMPILE_OPTIONS "${_baseline}")
    foreach(_isa SSE2 AVX2 AVX512)
      string(TOLOWER "${_isa}" _lower)
      string(REPLACE ";" " " _flags "${_${_lower}}")
      check_cxx_compiler_flag("${_flags}" ZVEC_TOPK_COMPILE_${_isa})
      if(ZVEC_TOPK_COMPILE_${_isa})
        set_source_files_properties("${_dir}/batch_topk_${_lower}.cc"
          PROPERTIES COMPILE_OPTIONS "${_${_lower}}")
        list(APPEND _definitions "ZVEC_BATCH_TOPK_HAVE_${_isa}=1")
      endif()
    endforeach()
  endif()

  check_cxx_source_compiles("
    #if !defined(__aarch64__) && !defined(_M_ARM64)
    #error Not an ARM64 target
    #endif
    #include <arm_neon.h>
    int main() { return vaddvq_u32(vdupq_n_u32(0)); }"
    ZVEC_TOPK_COMPILE_NEON)
  if(ZVEC_TOPK_COMPILE_NEON)
    if(NOT MSVC)
      check_cxx_compiler_flag("-march=armv8-a" ZVEC_TOPK_COMPILE_ARMV8)
      if(ZVEC_TOPK_COMPILE_ARMV8)
        set_source_files_properties(
          "${_dir}/batch_topk.cc" "${_dir}/batch_topk_neon.cc"
          PROPERTIES COMPILE_OPTIONS "-march=armv8-a")
      endif()
    endif()
    list(APPEND _definitions ZVEC_BATCH_TOPK_HAVE_NEON=1)
  endif()
  set_source_files_properties("${_dir}/batch_topk.cc"
    PROPERTIES COMPILE_DEFINITIONS "${_definitions}")
endfunction()
