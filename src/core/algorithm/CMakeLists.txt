include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(cluster)
cc_directory(flat)
cc_directory(flat_sparse)
cc_directory(ivf)
cc_directory(hnsw)
cc_directory(hnsw_sparse)
if(RABITQ_SUPPORTED)
  message(STATUS "BUILD RABITQ")
  cc_directory(hnsw_rabitq)
else()
  message(STATUS "NOT BUILD RABITQ")
  # Empty stub library for unsupported platforms
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rabitq_stub.cc
    "// Stub implementation for unsupported platforms\n"
    "// RaBitQ only supports Linux x86_64\n"
    "namespace zvec { namespace core { /* empty namespace for compatibility */ } }\n"
  )

  cc_library(
      NAME core_knn_hnsw_rabitq
      STATIC SHARED STRICT ALWAYS_LINK
      SRCS ${CMAKE_CURRENT_BINARY_DIR}/rabitq_stub.cc
      LIBS core_framework
      INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
      VERSION "${PROXIMA_ZVEC_VERSION}"
    )
endif()
