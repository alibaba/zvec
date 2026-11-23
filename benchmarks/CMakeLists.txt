include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Benchmark suites
cc_directory(fts)
cc_directory(inverted)
cc_directory(vector)
