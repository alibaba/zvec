include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

set(INVERTED_BENCH_HELPER_SRCS
    shared/index_fixture.cc
    shared/metrics.cc
    shared/report_utils.cc
)

cc_binary(
    NAME inverted_numeric_bench PACKED
    SRCS
        numeric_cases/main.cc
        numeric_cases/numeric_cases.cc
        ${INVERTED_BENCH_HELPER_SRCS}
    LIBS zvec zvec_ailego gflags roaring rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)

cc_gtest(
    NAME inverted_bench_test
    SRCS
        numeric_cases/numeric_cases.cc
        numeric_cases/numeric_cases_test.cc
        shared/helpers_test.cc
        ${INVERTED_BENCH_HELPER_SRCS}
    LIBS zvec zvec_ailego roaring rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)
