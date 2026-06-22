include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_QUANTIZER_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
        NAME core_quantizer 
        STATIC SHARED STRICT ALWAYS_LINK
        SRCS *.cc
        LIBS zvec_ailego zvec_turbo core_framework
        INCS . ${PROJECT_ROOT_DIR}/src/core
        LDFLAGS "${CORE_QUANTIZER_LDFLAGS}"
        VERSION "${PROXIMA_ZVEC_VERSION}"
)
