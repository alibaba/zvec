# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec-example-c)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to Zvec and dependencies ---
# Allow custom host build directory, default to "build"
if(NOT DEFINED HOST_BUILD_DIR)
    set(HOST_BUILD_DIR "build")
endif()

set(ZVEC_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../src/include)
set(ZVEC_GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/src/generated)
set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/lib)

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR} ${ZVEC_GENERATED_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR})

# Support multi-config builds (MSVC puts libs in Debug/Release subdirectories)
if(CMAKE_BUILD_TYPE)
    link_directories(${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE})
endif()

# Find required packages
find_package(Threads REQUIRED)

# Create INTERFACE target for zvec_c_api (fat shared library)
# No whole-archive flags needed — all symbols are already resolved in the .so/.dylib
add_library(zvec-c-api INTERFACE)
target_link_libraries(zvec-c-api INTERFACE
    zvec_c_api
    Threads::Threads
)
if(NOT WIN32)
    target_link_libraries(zvec-c-api INTERFACE ${CMAKE_DL_LIBS})
endif()
if(APPLE)
    target_link_options(zvec-c-api INTERFACE -Wl,-rpath,${ZVEC_LIB_DIR})
endif()

# Basic example
add_executable(c_api_basic_example basic_example.c)
target_link_libraries(c_api_basic_example PRIVATE
    zvec-c-api
)

# Schema example
add_executable(c_api_collection_schema_example collection_schema_example.c)
target_link_libraries(c_api_collection_schema_example PRIVATE
    zvec-c-api
)

# Struct document example
add_executable(c_api_doc_example doc_example.c)
target_link_libraries(c_api_doc_example PRIVATE
    zvec-c-api
)

# Index example
add_executable(c_api_index_example index_example.c)
target_link_libraries(c_api_index_example PRIVATE
    zvec-c-api
)

# Field schema example
add_executable(c_api_field_schema_example field_schema_example.c)
target_link_libraries(c_api_field_schema_example PRIVATE
    zvec-c-api
)

# Optimized example
add_executable(c_api_optimized_example optimized_example.c)
target_link_libraries(c_api_optimized_example PRIVATE
    zvec-c-api
)

# Strip symbols to reduce executable size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND (ANDROID OR (CMAKE_SYSTEM_NAME STREQUAL "Linux")))
    add_custom_command(TARGET c_api_basic_example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_basic_example>"
        COMMENT "Stripping symbols from c_api_basic_example")
    add_custom_command(TARGET c_api_collection_schema_example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_collection_schema_example>"
        COMMENT "Stripping symbols from c_api_collection_schema_example")
    add_custom_command(TARGET c_api_doc_example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_doc_example>"
        COMMENT "Stripping symbols from c_api_doc_example")
    add_custom_command(TARGET c_api_index_example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_index_example>"
        COMMENT "Stripping symbols from c_api_index_example")
    add_custom_command(TARGET c_api_field_schema_example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_field_schema_example>"
        COMMENT "Stripping symbols from c_api_field_schema_example")
    add_custom_command(TARGET c_api_optimized_example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_optimized_example>"
        COMMENT "Stripping symbols from c_api_optimized_example")
endif()

# Optimize for size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ANDROID)
    set_property(TARGET c_api_basic_example c_api_collection_schema_example c_api_doc_example
                 c_api_index_example c_api_field_schema_example c_api_optimized_example
                 PROPERTY COMPILE_FLAGS "-Os")
    set_property(TARGET c_api_basic_example c_api_collection_schema_example c_api_doc_example
                 c_api_index_example c_api_field_schema_example c_api_optimized_example
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()