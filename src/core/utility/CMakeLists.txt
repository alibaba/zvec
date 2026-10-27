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

# Build optional top-k SIMD loops in isolated translation units. The dispatcher
# itself stays on the baseline target and checks CPU features and OS state.
set(ZVEC_BATCH_TOPK_DEFINITIONS "" CACHE INTERNAL "Compiled BatchTopK variants" FORCE)
set(_batch_topk_defs)
set(_batch_topk_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  set(_batch_topk_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()
string(TOLOWER "${_batch_topk_arch}" _batch_topk_arch)
if(NOT ANDROID AND NOT IOS AND _batch_topk_arch MATCHES "^(x86_64|amd64|i.86)$")
  include(CheckCXXCompilerFlag)
  if(MSVC)
    set(_batch_topk_avx2 "/arch:AVX2")
    set(_batch_topk_avx512 "/arch:AVX512")
  else()
    if(_batch_topk_arch MATCHES "^(x86_64|amd64)$")
      set(_batch_topk_baseline "-march=x86-64")
    else()
      set(_batch_topk_baseline "-march=i686 -msse2")
    endif()
    set(_batch_topk_avx2 "${_batch_topk_baseline} -mavx2")
    set(_batch_topk_avx512 "${_batch_topk_baseline} -mavx512f")
  endif()
  foreach(_isa avx2 avx512)
    string(TOUPPER "${_isa}" _isa_upper)
    check_cxx_compiler_flag("${_batch_topk_${_isa}}"
                           ZVEC_COMPILER_BATCH_TOPK_${_isa_upper})
    if(ZVEC_COMPILER_BATCH_TOPK_${_isa_upper})
      # zvec_core also glob-collects these files in its parent directory.
      set_source_files_properties(
          ${CMAKE_CURRENT_SOURCE_DIR}/batch_topk_${_isa}.cc
          DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/core
          PROPERTIES COMPILE_FLAGS "${_batch_topk_${_isa}}")
      list(APPEND _batch_topk_defs ZVEC_BATCH_TOPK_HAVE_${_isa_upper}=1)
    endif()
  endforeach()
  set(ZVEC_BATCH_TOPK_DEFINITIONS "${_batch_topk_defs}"
      CACHE INTERNAL "Compiled BatchTopK variants" FORCE)
endif()
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_topk.cc
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/core
    PROPERTIES COMPILE_DEFINITIONS "${ZVEC_BATCH_TOPK_DEFINITIONS}")

cc_library(
    NAME core_utility 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework
    INCS . ${PROJECT_ROOT_DIR}/src/core
    LDFLAGS "${CORE_UTILITY_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
