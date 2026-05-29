include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Retrieve version from git repository
git_version(ZVEC_VERSION ${CMAKE_CURRENT_SOURCE_DIR})

# Add repository
cc_directory(core)