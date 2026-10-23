include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_knn_omega
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework core_knn_hnsw omega
    INCS . ${PROJECT_ROOT_DIR}/src/include ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm ${PROJECT_ROOT_DIR}/thirdparty/omega/OMEGALib/include
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )

if(TARGET core_knn_omega_objects)
  find_package(OpenMP QUIET)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(core_knn_omega_objects PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_compile_options(core_knn_omega_objects PRIVATE
      "$<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>"
      "$<$<CXX_COMPILER_ID:AppleClang>:-Wno-unknown-pragmas>"
      "$<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>"
    )
endif()
