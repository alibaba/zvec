include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

# Add C bindings
if(BUILD_C_BINDINGS)
    cc_directory(c)
endif()

if(BUILD_PYTHON_BINDINGS)
    cc_directory(python)
endif()