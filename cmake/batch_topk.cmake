# Keep ISA flags local to BatchTopK kernels. The dispatcher is compiled for
# the baseline target and selects only kernels supported by the running CPU.
# AUTO_DETECT_ARCH controls global architecture defaults. These kernels are
# probed against the target toolchain even when that option is disabled.

function(zvec_configure_batch_topk)
  set(_dir "${PROJECT_ROOT_DIR}/src/turbo/topk")
  set(_enabled scalar)
  # CMake caches probes by variable name. Recheck when an existing build tree
  # changes target/SDK/flags, rather than retaining another target's kernels.
  string(SHA256 _signature
    "${CMAKE_CXX_COMPILER}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_COMPILER_TARGET}|${CMAKE_SYSTEM_NAME}|${CMAKE_SYSTEM_PROCESSOR}|${CMAKE_OSX_ARCHITECTURES}|${CMAKE_SYSROOT}|${CMAKE_OSX_SYSROOT}|${CMAKE_CXX_FLAGS}")
  if(NOT "${ZVEC_TOPK_PROBE_SIGNATURE}" STREQUAL "${_signature}")
    foreach(_probe TARGET_X86 COMPILE_NEON COMPILE_ARMV8)
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
    set(_sse2_feature __SSE2__)
    set(_avx2_feature __AVX2__)
    set(_avx512_feature __AVX512F__)
    foreach(_isa SSE2 AVX2 AVX512)
      string(TOLOWER "${_isa}" _lower)
      configure_simd_kernel("${_dir}/batch_topk_${_lower}.cc"
        "${_dir}/batch_topk.cc" "${_${_lower}}" "
        #include <zvec/ailego/internal/platform.h>
        #if !defined(${_${_lower}_feature})
        #error Kernel ISA unavailable
        #endif
        int main() { return 0; }" ZVEC_BATCH_TOPK_HAVE_${_isa})
      if(ZVEC_BATCH_TOPK_HAVE_${_isa})
        list(APPEND _enabled "${_lower}")
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
    set(_neon_flags "")
    if(NOT MSVC)
      check_cxx_compiler_flag("-march=armv8-a" ZVEC_TOPK_COMPILE_ARMV8)
      if(ZVEC_TOPK_COMPILE_ARMV8)
        set(_neon_flags "-march=armv8-a")
        set_source_files_properties("${_dir}/batch_topk.cc"
          PROPERTIES COMPILE_OPTIONS "-march=armv8-a")
      endif()
    endif()
    configure_simd_kernel("${_dir}/batch_topk_neon.cc"
      "${_dir}/batch_topk.cc" "${_neon_flags}" "
      #include <zvec/ailego/internal/platform.h>
      #if !defined(AILEGO_ARM64_NEON)
      #error Kernel ISA unavailable
      #endif
      int main() { return vaddvq_u32(vdupq_n_u32(0)); }"
      ZVEC_BATCH_TOPK_HAVE_NEON)
    if(ZVEC_BATCH_TOPK_HAVE_NEON)
      list(APPEND _enabled neon)
    endif()
  endif()
  list(JOIN _enabled ", " _enabled)
  message(STATUS "BatchTopK kernels: ${_enabled} (runtime dispatch)")
endfunction()
