include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Add repositories
if(BUILD_C_BINDINGS)
    cc_directory(c)
endif()

if(BUILD_PYTHON_BINDINGS)
    cc_directory(python)
endif()
# Must come after c: the extension links the zvec_c_api target.
if(BUILD_SQLITE_BINDINGS)
    cc_directory(sqlite)
endif()
