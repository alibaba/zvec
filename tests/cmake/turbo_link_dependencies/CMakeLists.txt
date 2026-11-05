# Exercise Turbo's actual CMake rules without building its third-party runtime.
cmake_minimum_required(VERSION 3.18)
project(turbo_link_dependencies_test LANGUAGES C CXX)
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# Reproduce the Windows shared-build target layout even on non-Windows hosts.
set(WIN32 TRUE)
set(BUILD_ZVEC_SHARED ON)
set(BUILD_ZVEC_CORE_SHARED ON)
set(RABITQ_SUPPORTED ON)
set(AUTO_DETECT_ARCH OFF)

add_library(zvec_ailego INTERFACE)
add_library(rabitq_fastscan INTERFACE)
add_library(rabitqlib::fastscan ALIAS rabitq_fastscan)
target_compile_definitions(rabitq_fastscan INTERFACE TURBO_RABITQ_DEPENDENCY=1)

# Copy the build rules, using a small source to check dependency propagation
# independently of platform-specific SIMD compilation and downloaded headers.
set(TURBO_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/turbo")
file(MAKE_DIRECTORY "${TURBO_FIXTURE_DIR}")
configure_file("${PROJECT_ROOT_DIR}/src/turbo/CMakeLists.txt"
               "${TURBO_FIXTURE_DIR}/CMakeLists.txt" COPYONLY)
file(WRITE "${TURBO_FIXTURE_DIR}/dependency.cc" [=[
#ifndef TURBO_RABITQ_DEPENDENCY
#error "RaBitQ usage requirements did not reach Turbo"
#endif
int turbo_dependency() { return TURBO_RABITQ_DEPENDENCY; }
]=])
add_subdirectory("${TURBO_FIXTURE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/turbo-build")

foreach(_target zvec_turbo zvec_turbo_objects zvec_turbo_export_objects)
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "Missing Turbo variant: ${_target}")
  endif()
  get_target_property(_libs ${_target} LINK_LIBRARIES)
  if(NOT "rabitqlib::fastscan" IN_LIST _libs)
    message(FATAL_ERROR "RaBitQ dependency missing from ${_target}: ${_libs}")
  endif()
endforeach()
