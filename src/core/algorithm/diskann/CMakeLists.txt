include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB_RECURSE ALL_SRCS *.cc *.c)

# DiskAnn is compiled as a STATIC+SHARED library (matching the pattern used by
# core_knn_hnsw, core_knn_cluster, etc.). The static variant
# (core_knn_diskann_static) is whole-archived into _zvec.so for the Python
# wheel, so no separate runtime .so is needed. The shared variant is used by
# C++ tools and tests that link against it directly.
#
# io_uring uses raw Linux syscalls and has no liburing dependency. libaio is
# loaded at runtime via dlopen()/dlsym() (see libaio_loader.h), so we do NOT
# link against -laio. On Linux, ${CMAKE_DL_LIBS} provides the system-level
# dependency for dlopen/dlsym/dlclose. macOS ARM64 uses pread and needs no
# extra I/O backend library. Windows overlapped I/O is built in and likewise
# needs no extra backend library.
set(CORE_KNN_DISKANN_LIBS zvec_ailego core_framework core_knn_cluster)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|aarch64|arm64)$")
    list(APPEND CORE_KNN_DISKANN_LIBS ${CMAKE_DL_LIBS})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
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
if(MSVC)
    # NOMINMAX must reach the _objects target, since _add_library() compiles
    # sources there; the SHARED/STATIC targets only bundle object files and a
    # PRIVATE define on them never reaches the compiler. Without this, the
    # windows.h min/max macros collide with std::min/std::max and
    # numeric_limits<T>::max()/min() (error C4003).
    target_compile_definitions(core_knn_diskann_objects PRIVATE NOMINMAX)
endif()
