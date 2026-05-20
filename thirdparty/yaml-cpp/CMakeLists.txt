set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable testing in yaml-cpp" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable parse tools in yaml-cpp" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable contrib stuff in yaml-cpp" FORCE)

set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})
add_subdirectory(yaml-cpp-0.6.3)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

mark_target_includes_system(yaml-cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(yaml-cpp PRIVATE -Wno-shadow)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(yaml-cpp PRIVATE -Wno-effc++)
endif()