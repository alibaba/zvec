include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

set(INVERTED_BENCH_HELPER_SRCS
    shared/index_fixture.cc
    shared/json.cc
    shared/metrics.cc
    shared/query_helpers.cc
)

cc_binary(
    NAME inverted_numeric_bench PACKED
    SRCS
        numeric_cases/dataset.cc
        numeric_cases/main.cc
        numeric_cases/numeric_cases.cc
        numeric_cases/profile.cc
        ${INVERTED_BENCH_HELPER_SRCS}
    LIBS zvec zvec_ailego gflags roaring rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)
