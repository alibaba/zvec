include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Retrieve version from git repository
git_version(ZVEC_VERSION ${CMAKE_CURRENT_SOURCE_DIR})

# Add repository
if(BUILD_PYTHON_BINDINGS)
    cc_directory(python)
endif()

if(BUILD_C_BINDINGS)
    cc_directory(c)
endif()