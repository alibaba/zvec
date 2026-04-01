include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directories(ailego)
cc_directories(db)
cc_directories(core)
if(BUILD_C_BINDINGS)
    cc_directories(c)
endif()
