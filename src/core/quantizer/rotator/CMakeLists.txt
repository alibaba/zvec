include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
  if(HOST_ARCH MATCHES "^(x86|x64)$")
    setup_compiler_march_for_x86(
        _ROTATOR_SSE _ROTATOR_AVX2 _ROTATOR_AVX512 _ROTATOR_AVX512FP16)
    if(_ROTATOR_SSE)
      set_source_files_properties(fht_rotator_sse.cc
          PROPERTIES COMPILE_FLAGS "${_ROTATOR_SSE}")
    endif()
    if(_ROTATOR_AVX2)
      set_source_files_properties(fht_rotator_avx2.cc
          PROPERTIES COMPILE_FLAGS "${_ROTATOR_AVX2}")
    endif()
    if(_ROTATOR_AVX512)
      set_source_files_properties(fht_rotator_avx512.cc
          PROPERTIES COMPILE_FLAGS "${_ROTATOR_AVX512}")
    endif()
  elseif(HOST_ARCH MATCHES "^(arm|arm64)$" AND NOT MSVC)
    set_source_files_properties(fht_rotator_neon.cc
        PROPERTIES COMPILE_FLAGS "-march=armv8-a")
  endif()
endif()

cc_library(
    NAME core_rotator
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework rabitqlib
    INCS . .. ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
