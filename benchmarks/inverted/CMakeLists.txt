include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

set(INVERTED_BENCH_SRCS
    int64_cases.cc
    array_int32_cases.cc
    index_fixture.cc
    metrics.cc
    runner.cc
)

cc_binary(
    NAME inverted_bench PACKED
    SRCS inverted_bench_main.cc ${INVERTED_BENCH_SRCS}
    LIBS zvec zvec_ailego gflags roaring rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)

cc_gtest(
    NAME inverted_bench_test
    SRCS inverted_bench_test.cc ${INVERTED_BENCH_SRCS}
    LIBS zvec zvec_ailego roaring rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)
