include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_test(NAME factory_abi_test STRICT
  SRCS factory_abi_test.cc factory_abi_legacy.cc
  LIBS gtest gtest_main zvec_ailego)
# Exercise code compiled from both headers, without LTO merging their bodies.
set_target_properties(factory_abi_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
if(NOT MSVC)
  target_compile_options(factory_abi_test PRIVATE -O2)
endif()

# Keep native CMake link coverage while sharing only test setup.
function(add_compatibility_test NAME)
  add_executable(${NAME} EXCLUDE_FROM_ALL ${NAME}.cc)
  target_compile_features(${NAME} PRIVATE cxx_std_17)
  target_include_directories(${NAME} PRIVATE ${PROJECT_ROOT_DIR}/src)
  target_link_libraries(${NAME} PRIVATE ${ARGN})
  add_dependencies(unittest ${NAME})
  add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

add_library(legacy_quantizer_extension STATIC EXCLUDE_FROM_ALL legacy_quantizer_extension.cc)
target_compile_features(legacy_quantizer_extension PRIVATE cxx_std_17)
target_include_directories(legacy_quantizer_extension PRIVATE ${PROJECT_ROOT_DIR}/src)
target_link_libraries(legacy_quantizer_extension PRIVATE zvec_turbo)
add_compatibility_test(component_factory_test core_metric_static core_framework
  legacy_quantizer_extension zvec_turbo zvec_ailego)
zvec_link_static_extension(TARGET component_factory_test
  LIBRARY legacy_quantizer_extension INITIALIZER RegisterLegacyQuantizerExtension)
add_compatibility_test(turbo_factory_test zvec_turbo zvec_ailego)
add_compatibility_test(component_registry_test core_metric_static core_framework)
add_compatibility_test(legacy_builtin_test zvec_core zvec_turbo zvec_ailego)
target_sources(legacy_builtin_test PRIVATE legacy_builtin_client.cc)
set_target_properties(legacy_builtin_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

add_library(prebuilt_extension_source STATIC EXCLUDE_FROM_ALL prebuilt_extension.cc)
# This archive intentionally has no SDK link dependency. Declare its language
# requirement directly instead of inheriting it from another target/compiler.
target_compile_features(prebuilt_extension_source PRIVATE cxx_std_17)
target_include_directories(prebuilt_extension_source PRIVATE ${PROJECT_ROOT_DIR}/src)
add_library(prebuilt_extension STATIC IMPORTED)
# Use a real imported archive with an explicitly referenced C++ initializer.
set_target_properties(prebuilt_extension PROPERTIES IMPORTED_LOCATION
  "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}prebuilt_extension_source${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_dependencies(prebuilt_extension prebuilt_extension_source)
add_compatibility_test(prebuilt_extension_test zvec_core zvec_turbo zvec_ailego)
zvec_link_static_extension(TARGET prebuilt_extension_test LIBRARY prebuilt_extension
  INITIALIZER RegisterCompatibilityExtension)

# Let CMake handle transitive/conditional links itself, including a registration
# initializer propagated through an interface target. Reuse the behavior check.
add_library(prebuilt_extension_interface INTERFACE)
zvec_link_static_extension(TARGET prebuilt_extension_interface
  LIBRARY prebuilt_extension INITIALIZER RegisterCompatibilityExtension)
add_executable(conditional_extension_test EXCLUDE_FROM_ALL prebuilt_extension_test.cc)
target_compile_features(conditional_extension_test PRIVATE cxx_std_17)
target_include_directories(conditional_extension_test PRIVATE ${PROJECT_ROOT_DIR}/src)
target_link_libraries(conditional_extension_test PRIVATE zvec_core zvec_turbo
  zvec_ailego "$<$<BOOL:1>:prebuilt_extension_interface>")
add_dependencies(unittest conditional_extension_test)
add_test(NAME conditional_extension_test COMMAND conditional_extension_test)

cc_library(NAME legacy_initializer STATIC ALWAYS_LINK
  SRCS_NO_GLOB SRCS legacy_initializer.cc)
add_compatibility_test(legacy_initializer_test legacy_initializer)

cc_test(NAME quantizer_compatibility_test STRICT SRCS quantizer_compatibility_test.cc
  LIBS zvec_core zvec_turbo zvec_ailego gtest gtest_main INCS ${PROJECT_ROOT_DIR}/src)
set_tests_properties(factory_abi_test component_factory_test turbo_factory_test
  prebuilt_extension_test legacy_initializer_test quantizer_compatibility_test
  component_registry_test legacy_builtin_test conditional_extension_test
  PROPERTIES TIMEOUT 90)
