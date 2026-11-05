include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_binary(
    NAME fts_bench PACKED
    SRCS fts_bench_main.cc
    LIBS
         zvec
         gflags
         roaring
         rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)

# The diagnostic benchmark uses POSIX resource and thread CPU clocks.
if(UNIX)
    cc_binary(
        NAME inverted_range_cache_bench PACKED
        SRCS inverted_range_cache_bench.cc
        LIBS zvec roaring rocksdb
        INCS . ${PROJECT_SOURCE_DIR}/src
        LDFLAGS ${APPLE_FRAMEWORK_LIBS}
    )
endif()
