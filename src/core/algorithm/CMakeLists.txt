include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(cluster)
cc_directory(flat)
cc_directory(flat_sparse)
cc_directory(ivf)
cc_directory(hnsw)
cc_directory(hnsw_sparse)
cc_directory(vamana)

if(DISKANN_SUPPORTED)
  message(STATUS "build diskann")
  cc_directory(diskann)
else()
  message(STATUS "not build diskann")
  # Empty stub library for unsupported platforms
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/diskann_stub.cc
    "// Stub implementation for unsupported platforms\n"
    "// DiskAnn only supports Linux x86_64\n"
    "namespace zvec { namespace core { /* empty namespace for compatibility */ } }\n"
  )

  if(MSVC)
    # MSVC: STATIC-only stub to avoid creating an empty DLL with no exports
    # (MSVC linker fails to produce an import library when there are zero exports).
    # cc_library with STATIC-only creates target "core_knn_diskann" but NOT the
    # "core_knn_diskann_static" variant (that only happens with STATIC+SHARED).
    # The Python binding references core_knn_diskann_static on all platforms, so
    # create an ALIAS so the same target name works under MSVC as well.
    # ($<TARGET_FILE:> supports ALIAS targets since CMake 3.18.)
    cc_library(
        NAME core_knn_diskann
        STATIC STRICT ALWAYS_LINK
        SRCS ${CMAKE_CURRENT_BINARY_DIR}/diskann_stub.cc
        LIBS core_framework
        INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
        VERSION "${PROXIMA_ZVEC_VERSION}"
      )
    add_library(core_knn_diskann_static ALIAS core_knn_diskann)
  else()
    cc_library(
        NAME core_knn_diskann
        STATIC SHARED STRICT ALWAYS_LINK
        SRCS ${CMAKE_CURRENT_BINARY_DIR}/diskann_stub.cc
        LIBS core_framework
        INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
        VERSION "${PROXIMA_ZVEC_VERSION}"
      )
  endif()
endif()

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
