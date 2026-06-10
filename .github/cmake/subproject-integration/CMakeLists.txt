cmake_minimum_required(VERSION 3.13)
project(zvec_subproject_integration LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ZVEC_SOURCE_DIR "" CACHE PATH "Path to the zvec source tree")
if(NOT ZVEC_SOURCE_DIR)
  message(FATAL_ERROR "ZVEC_SOURCE_DIR is required")
endif()

option(ENABLE_ZVEC_CORE "Validate zvec-core-only subproject integration" OFF)

if(ENABLE_ZVEC_CORE)
  set(BUILD_ZVEC_CORE_ONLY ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${ZVEC_SOURCE_DIR}" zvec)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(ENABLE_ZVEC_CORE)
  add_executable(zvec_embed_core_only_example
    "${ZVEC_SOURCE_DIR}/examples/c++/core/main.cc")
  target_link_libraries(zvec_embed_core_only_example PRIVATE zvec_core_shared)
else()
  add_executable(zvec_embed_db_example
    "${ZVEC_SOURCE_DIR}/examples/c++/db/main.cc")
  target_link_libraries(zvec_embed_db_example PRIVATE zvec_shared)

  add_executable(zvec_embed_core_example
    "${ZVEC_SOURCE_DIR}/examples/c++/core/main.cc")
  target_link_libraries(zvec_embed_core_example PRIVATE zvec_shared)

  add_executable(zvec_embed_ailego_example
    "${ZVEC_SOURCE_DIR}/examples/c++/ailego/main.cc")
  target_link_libraries(zvec_embed_ailego_example PRIVATE zvec_shared)
endif()
