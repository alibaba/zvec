# Exercise the link helper without requiring the database dependencies.
cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(link_dependencies_test LANGUAGES CXX)
  enable_testing()
  get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
  include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
endif()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/factory.cc
  "int hidden_factory() { return 42; }\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/runtime.cc
  "int hidden_factory();\nint runtime_value() { return hidden_factory(); }\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/api.cc [=[
int runtime_value();
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
int public_value() { return runtime_value(); }
]=])
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/consumer.cc [=[
#ifdef _WIN32
__declspec(dllimport)
#endif
int public_value();
int main() { return public_value() == 42 ? 0 : 1; }
]=])
add_library(link_hidden_factory STATIC ${CMAKE_CURRENT_BINARY_DIR}/factory.cc)
add_library(link_private_runtime STATIC ${CMAKE_CURRENT_BINARY_DIR}/runtime.cc)
set_target_properties(link_private_runtime PROPERTIES ALWAYS_LINK TRUE)
add_library(link_shared_api SHARED ${CMAKE_CURRENT_BINARY_DIR}/api.cc)
set_target_properties(link_hidden_factory link_private_runtime link_shared_api
  PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_link_libraries(link_shared_api PRIVATE link_private_runtime link_hidden_factory)
add_executable(link_dependencies_test ${CMAKE_CURRENT_BINARY_DIR}/consumer.cc)
_target_link_libraries(link_dependencies_test link_shared_api)
get_target_property(CONSUMER_LIBS link_dependencies_test LINK_LIBRARIES)
if("link_private_runtime" IN_LIST CONSUMER_LIBS OR
   "link_hidden_factory" IN_LIST CONSUMER_LIBS)
  message(FATAL_ERROR "Shared-library private archives leaked into consumer: ${CONSUMER_LIBS}")
endif()

# A public dependency must still be force-loaded for factory registration.
add_library(link_public_api INTERFACE)
target_link_libraries(link_public_api INTERFACE link_private_runtime link_hidden_factory)
add_library(link_public_consumer STATIC EXCLUDE_FROM_ALL ${CMAKE_CURRENT_BINARY_DIR}/factory.cc)
_target_link_libraries(link_public_consumer link_public_api)
get_target_property(PUBLIC_LIBS link_public_consumer LINK_LIBRARIES)
if(MSVC)
  get_target_property(PUBLIC_OPTS link_public_consumer LINK_OPTIONS)
  if(NOT PUBLIC_OPTS MATCHES "WHOLEARCHIVE.*link_private_runtime")
    message(FATAL_ERROR "Public ALWAYS_LINK dependency did not propagate")
  endif()
elseif(NOT "link_private_runtime" IN_LIST PUBLIC_LIBS)
  message(FATAL_ERROR "Public ALWAYS_LINK dependency did not propagate")
endif()
add_test(NAME link_dependencies_test COMMAND link_dependencies_test)
add_dependencies(unittest link_dependencies_test)
