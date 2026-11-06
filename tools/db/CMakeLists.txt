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

cc_binary(
    NAME query_threading_bench
    SRCS query_threading_bench_main.cc
    LIBS zvec gflags roaring rocksdb
         core_knn_flat core_metric core_quantizer core_utility
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)
