include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# TODO(windows): this is for test_util.h, should be removed
include_directories(${PROJECT_ROOT_DIR})

cc_directories(ailego)
cc_directories(db)
cc_directories(core)
cc_directories(turbo)
if(BUILD_ZVEC_SHARED AND NOT IOS AND NOT ANDROID)
    cc_directories(shared)
endif()
if(BUILD_C_BINDINGS)
    cc_directories(c)
endif()

if(NOT IOS AND NOT ANDROID)
    add_subdirectory(cmake/link_dependencies)
    add_test(NAME turbo_link_dependencies_test
        COMMAND ${CMAKE_COMMAND}
            -S "${CMAKE_CURRENT_SOURCE_DIR}/cmake/turbo_link_dependencies"
            -B "${CMAKE_CURRENT_BINARY_DIR}/cmake/turbo_link_dependencies"
            -G "${CMAKE_GENERATOR}")
endif()
