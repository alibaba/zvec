include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_QUANTIZER_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

# x86: use AVX2/AVX512 arch flag from RABITQ detection
if(RABITQ_SUPPORTED AND RABITQ_ARCH_FLAG)
  set_source_files_properties(
      record_rotator.cc
      PROPERTIES
      COMPILE_FLAGS "${RABITQ_ARCH_FLAG}"
  )
# ARM aarch64: use armv8-a to enable NEON intrinsics
elseif(HOST_ARCH MATCHES "^(arm|arm64)$" AND NOT MSVC)
  set_source_files_properties(
      record_rotator.cc
      PROPERTIES
      COMPILE_FLAGS "-march=armv8-a"
  )
endif()

cc_library(
        NAME core_quantizer 
        STATIC SHARED STRICT ALWAYS_LINK
        SRCS *.cc
        LIBS zvec_ailego core_framework
        LIBS zvec_ailego zvec_turbo core_framework
        INCS . ${PROJECT_ROOT_DIR}/src/core
        LDFLAGS "${CORE_QUANTIZER_LDFLAGS}"
        VERSION "${PROXIMA_ZVEC_VERSION}"
)
