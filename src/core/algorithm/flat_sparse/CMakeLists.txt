include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# --exclude-libs is GNU ld / LLVM lld only; Apple ld does not support it.
# On macOS (Mach-O), symbol interposition works differently and the
# Arrow/Parquet double-free issue does not apply.
if(NOT APPLE)
  set(CORE_KNN_FLAT_SPARSE_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

cc_library(
    NAME core_knn_flat_sparse 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework 
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    LDFLAGS "${CORE_KNN_FLAT_SPARSE_LDFLAGS}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
