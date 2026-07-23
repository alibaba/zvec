include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB_RECURSE ALL_SRCS *.cc *.c)

# DiskAnn is compiled as a STATIC+SHARED library (matching the pattern used by
# core_knn_hnsw, core_knn_cluster, etc.). The static variant
# (core_knn_diskann_static) is whole-archived into _zvec.so for the Python
# wheel, so no separate runtime .so is needed. The shared variant is used by
# C++ tools and tests that link against it directly.
#
# libaio is loaded at runtime via dlopen()/dlsym() (see libaio_loader.h), so
# we do NOT link against -laio. The only system-level link dependency is
# ${CMAKE_DL_LIBS} for dlopen/dlsym/dlclose.
set(CORE_KNN_DISKANN_LIBS core_framework core_knn_cluster)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386")
    list(APPEND CORE_KNN_DISKANN_LIBS ${CMAKE_DL_LIBS})
endif()

if(NOT APPLE)
    set(CORE_KNN_DISKANN_LDFLAGS
        "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_diskann
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS ${CORE_KNN_DISKANN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    LDFLAGS "${CORE_KNN_DISKANN_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
)