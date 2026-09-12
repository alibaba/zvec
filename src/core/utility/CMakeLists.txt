include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_UTILITY_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

# block_heap.cc uses AVX2 intrinsics (guarded by __AVX2__) for its
# push_block fast path. When the host toolchain supports it, compile this
# source with an AVX2-capable -march so the AVX2 codegen is enabled. On
# other hosts the scalar fallback inside __AVX2__ guards is compiled
# instead, and callers runtime-gate invocation on CpuFeatures::AVX2.
if(NOT ANDROID AND AUTO_DETECT_ARCH)
  if(HOST_ARCH MATCHES "^(x86|x64)$")
    setup_compiler_march_for_x86(
      _BLOCK_HEAP_MARCH_SSE _BLOCK_HEAP_MARCH_AVX2
      _BLOCK_HEAP_MARCH_AVX512 _BLOCK_HEAP_MARCH_AVX512FP16)
    if(_BLOCK_HEAP_MARCH_AVX2)
      set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/block_heap.cc
        PROPERTIES
        COMPILE_FLAGS "${_BLOCK_HEAP_MARCH_AVX2}"
      )
    endif()
  endif()
endif()

cc_library(
    NAME core_utility 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework
    INCS . ${PROJECT_ROOT_DIR}/src/core
    LDFLAGS "${CORE_UTILITY_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
