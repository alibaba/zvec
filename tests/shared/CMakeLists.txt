include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

if(BUILD_ZVEC_SHARED)
    cc_test(
        NAME shared_api_test STRICT
        SRCS shared_api_test.cc
        LIBS zvec_shared
    )
endif()

if(BUILD_ZVEC_CORE_SHARED)
    cc_test(
        NAME core_shared_api_test STRICT
        SRCS core_shared_api_test.cc
        LIBS zvec_core_shared
    )
endif()
