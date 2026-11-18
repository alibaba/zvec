include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_binary(
    NAME fts_bench PACKED
    SRCS fts_bench_main.cc
    LIBS
         zvec
         zvec_core
         gflags
         roaring
         rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)

# Custom FTS uses the registered FileDumper and read-storage factories.
# Retain their registration objects, as the database tests do; ordinary
# transitive archive linking drops these otherwise-unreferenced objects.
if(MSVC)
  target_link_options(fts_bench PRIVATE
    "/WHOLEARCHIVE:$<TARGET_FILE:zvec_core>")
elseif(APPLE)
  target_link_options(fts_bench PRIVATE
    "-Wl,-force_load,$<TARGET_FILE:zvec_core>")
else()
  target_link_options(fts_bench PRIVATE
    "-Wl,--whole-archive,$<TARGET_FILE:zvec_core>,--no-whole-archive")
endif()

# Input decoding belongs to the benchmark, not to the production JSON API.
# Mobile CI disables BUILD_BENCHMARKS and does not build this target.
cc_gtest(
    NAME fts_bench_input_test
    SRCS ${PROJECT_ROOT_DIR}/tests/tools/fts_bench_input_test.cc
    INCS .
)
