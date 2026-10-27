include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_test(NAME factory_abi_test STRICT
  SRCS factory_abi_test.cc factory_abi_legacy.cc
  LIBS gtest gtest_main zvec_ailego)
# Exercise code compiled from both headers, without LTO merging their bodies.
set_target_properties(factory_abi_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
if(NOT MSVC)
  target_compile_options(factory_abi_test PRIVATE -O2)
endif()

# Native CMake consumers must work without any cc_* registration helper.
add_library(legacy_quantizer_extension STATIC EXCLUDE_FROM_ALL legacy_quantizer_extension.cc)
target_include_directories(legacy_quantizer_extension PRIVATE ${PROJECT_ROOT_DIR}/src)
target_link_libraries(legacy_quantizer_extension PRIVATE zvec_turbo)
add_executable(component_factory_test EXCLUDE_FROM_ALL component_factory_test.cc)
target_link_libraries(component_factory_test PRIVATE core_metric core_framework
  legacy_quantizer_extension zvec_turbo zvec_ailego)
add_dependencies(unittest component_factory_test)
add_test(NAME component_factory_test COMMAND component_factory_test)

add_executable(turbo_factory_test EXCLUDE_FROM_ALL turbo_factory_test.cc)
target_include_directories(turbo_factory_test PRIVATE ${PROJECT_ROOT_DIR}/src)
target_link_libraries(turbo_factory_test PRIVATE zvec_turbo zvec_ailego)
add_dependencies(unittest turbo_factory_test)
add_test(NAME turbo_factory_test COMMAND turbo_factory_test)

add_library(prebuilt_extension_source STATIC EXCLUDE_FROM_ALL prebuilt_extension.cc)
target_include_directories(prebuilt_extension_source PRIVATE ${PROJECT_ROOT_DIR}/src)
add_library(prebuilt_extension STATIC IMPORTED)
# Use a real imported archive, so the test cannot rely on source scanning it.
set_target_properties(prebuilt_extension PROPERTIES IMPORTED_LOCATION
  "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}prebuilt_extension_source${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_dependencies(prebuilt_extension prebuilt_extension_source)
add_executable(prebuilt_extension_test EXCLUDE_FROM_ALL prebuilt_extension_test.cc)
target_link_libraries(prebuilt_extension_test PRIVATE zvec_core zvec_turbo zvec_ailego)
zvec_link_static_extension(TARGET prebuilt_extension_test LIBRARY prebuilt_extension
  INITIALIZER RegisterCompatibilityExtension)
add_dependencies(unittest prebuilt_extension_test)
add_test(NAME prebuilt_extension_test COMMAND prebuilt_extension_test)

cc_library(NAME legacy_initializer STATIC ALWAYS_LINK
  SRCS_NO_GLOB SRCS legacy_initializer.cc)
add_executable(legacy_initializer_test EXCLUDE_FROM_ALL legacy_initializer_test.cc)
target_link_libraries(legacy_initializer_test PRIVATE legacy_initializer)
add_dependencies(unittest legacy_initializer_test)
add_test(NAME legacy_initializer_test COMMAND legacy_initializer_test)

cc_test(NAME quantizer_compatibility_test STRICT SRCS quantizer_compatibility_test.cc
  LIBS zvec_core zvec_turbo zvec_ailego gtest gtest_main INCS ${PROJECT_ROOT_DIR}/src)
set_tests_properties(factory_abi_test component_factory_test turbo_factory_test
  prebuilt_extension_test legacy_initializer_test quantizer_compatibility_test
  PROPERTIES TIMEOUT 90)
