cmake_minimum_required(VERSION 3.26)
project(zvec_matrix_regression C CXX)
set(PROJECT_ROOT_DIR "" CACHE PATH "Version checkout")
set(VALIDATION_DEPS "${PROJECT_ROOT_DIR}" CACHE PATH "Checkout with submodules")
option(VALIDATION_FIXED "Build all related correctness tests" OFF)
option(VALIDATION_LEGACY "Use pre-refactor matrix API" OFF)
if(NOT EXISTS "${PROJECT_ROOT_DIR}/src/turbo/CMakeLists.txt")
  message(FATAL_ERROR "Pass an absolute PROJECT_ROOT_DIR")
endif()
set(CC_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ZVEC_VERSION_STRING "0.2.0")
set(PROXIMA_ZVEC_VERSION "0.2.0")
# Match the previous focused regression scope, not the full product build.
add_definitions(-DRABITQ_SUPPORTED=0 -DDISKANN_SUPPORTED=0)
# Match the repository root CMakeLists: Ubuntu GCC defaults to --as-needed,
# which drops factory-registering shared libraries from DT_NEEDED.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()
include_directories(${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/include
                    ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR})
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
add_subdirectory(${VALIDATION_DEPS}/thirdparty/googletest/googletest-1.10.0 gtest EXCLUDE_FROM_ALL)
add_subdirectory(${PROJECT_ROOT_DIR}/src/ailego ailego)
add_subdirectory(${PROJECT_ROOT_DIR}/src/turbo turbo)
target_include_directories(zvec_turbo SYSTEM PRIVATE
  ${VALIDATION_DEPS}/thirdparty/RaBitQ-Library/RaBitQ-Library-0.7.0/include)
foreach(part framework metric quantizer utility)
  add_subdirectory(${PROJECT_ROOT_DIR}/src/core/${part} ${part})
endforeach()
foreach(part cluster flat ivf)
  add_subdirectory(${PROJECT_ROOT_DIR}/src/core/algorithm/${part} ${part})
endforeach()
if(VALIDATION_FIXED)
  add_subdirectory(${PROJECT_ROOT_DIR}/tests/core/metric tests_metric)
  add_subdirectory(${PROJECT_ROOT_DIR}/tests/turbo tests_turbo)
  foreach(part flat cluster ivf)
    add_subdirectory(${PROJECT_ROOT_DIR}/tests/core/algorithm/${part} tests_${part})
  endforeach()
endif()
set(VALIDATION_LIBS zvec_ailego core_framework core_utility core_metric
    core_quantizer core_knn_cluster core_knn_flat core_knn_ivf zvec_turbo)
cc_gtest(NAME legacy_ivf_comparison SRCS legacy_compatibility.cc
  INCS ${PROJECT_ROOT_DIR}/src/core/algorithm/ivf LIBS ${VALIDATION_LIBS})
target_compile_options(legacy_ivf_comparison PRIVATE
  -include ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_clock.h)
cc_gtest(NAME ivf_interleaved_bench SRCS bench.cc
  INCS ${PROJECT_ROOT_DIR}/src/core/algorithm/ivf LIBS ${VALIDATION_LIBS})
cc_binary(NAME interleaved_dispatch_probe SRCS dispatch_probe.cc
  LIBS ${VALIDATION_LIBS})
