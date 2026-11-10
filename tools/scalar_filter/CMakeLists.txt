cmake_minimum_required(VERSION 3.20)
project(zvec_scalar_filter_tools LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(ZVEC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(ZVEC_BUILD_DIR "${ZVEC_SOURCE_DIR}/build" CACHE PATH "Directory containing lib/libzvec")
find_library(SCALAR_FILTER_ZVEC_LIBRARY NAMES zvec
  PATHS "${ZVEC_BUILD_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
find_package(Threads REQUIRED)
add_library(scalar_filter_zvec SHARED IMPORTED)
set_target_properties(scalar_filter_zvec PROPERTIES
  IMPORTED_LOCATION "${SCALAR_FILTER_ZVEC_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ZVEC_SOURCE_DIR}/src/include")

add_executable(scalar_filter_benchmark benchmark.cc)
add_executable(scalar_filter_smoke smoke.cc)
set(SCALAR_FILTER_TARGETS scalar_filter_benchmark scalar_filter_smoke)
option(SCALAR_FILTER_BUILD_INTERNAL_STATS "Build an ON-library-only driver with JIT activation stats" OFF)
if(SCALAR_FILTER_BUILD_INTERNAL_STATS)
  add_executable(scalar_filter_jit_benchmark benchmark.cc)
  target_compile_definitions(scalar_filter_jit_benchmark PRIVATE
    SCALAR_FILTER_INTERNAL_STATS=1 ZVEC_ENABLE_SCALAR_JIT=1)
  target_include_directories(scalar_filter_jit_benchmark PRIVATE
    "${ZVEC_SOURCE_DIR}/src"
    "${ZVEC_SOURCE_DIR}/thirdparty/gandiva_scalar/include"
    "${ZVEC_BUILD_DIR}/external/usr/local/include")
  list(APPEND SCALAR_FILTER_TARGETS scalar_filter_jit_benchmark)
endif()
foreach(tool ${SCALAR_FILTER_TARGETS})
  target_link_libraries(${tool} PRIVATE scalar_filter_zvec Threads::Threads)
  target_compile_definitions(${tool} PRIVATE ZVEC_DB_USE_SHARED)
  set_property(TARGET ${tool} PROPERTY BUILD_RPATH "${ZVEC_BUILD_DIR}/lib")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${tool} PRIVATE -Wall -Wextra)
  endif()
endforeach()
