include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

# fts_bench depends on the all-in-one zvec_shared library.
# On MSVC, zvec_shared is never built (PE/COFF 65535-section limit, LNK1189),
# so this tool cannot be built on that platform.
if(NOT MSVC AND BUILD_ZVEC_SHARED)
cc_binary(
    NAME fts_bench PACKED
    SRCS fts_bench_main.cc
    LIBS
         zvec_shared
         gflags
         roaring
         rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)
endif()
