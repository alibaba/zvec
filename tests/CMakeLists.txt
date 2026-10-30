include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# TODO(windows): this is for test_util.h, should be removed
include_directories(${PROJECT_ROOT_DIR})

cc_directories(ailego)
cc_directories(db)
cc_directories(core)
cc_directories(turbo)
cc_directories(static)
cc_directories(compatibility)
if((BUILD_ZVEC_SHARED OR BUILD_ZVEC_CORE_SHARED) AND NOT IOS AND NOT ANDROID)
    cc_directories(shared)
endif()
if(BUILD_C_BINDINGS)
    cc_directories(c)
endif()
