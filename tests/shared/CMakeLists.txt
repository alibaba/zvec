include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_test(
    NAME shared_api_test STRICT
    SRCS shared_api_test.cc
    LIBS zvec_shared
)
