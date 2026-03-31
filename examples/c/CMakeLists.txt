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

# On Windows, libraries are in Debug/Release subdirectories
if(WIN32)
    set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/lib/$<CONFIG>)
    set(ZVEC_DEPENDENCY_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/external/usr/local/lib/$<CONFIG>)
else()
    set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/lib)
    set(ZVEC_DEPENDENCY_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/external/usr/local/lib)
endif()

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR} ${ZVEC_GENERATED_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR} ${ZVEC_DEPENDENCY_LIB_DIR})

# Find required packages
find_package(Threads REQUIRED)

# --- Determine debug/release library names ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLOG_LIB glogd)
    set(GFLAGS_LIB gflags_nothreads_debug)
    set(PROTOBUF_LIB protobufd)
else()
    set(GLOG_LIB glog)
    set(GFLAGS_LIB gflags_nothreads)
    set(PROTOBUF_LIB protobuf)
endif()

# --- Dependency groups ---
if(NOT WIN32)
    set(zvec_c_api_deps
        roaring
        rocksdb
        arrow
        arrow_acero
        arrow_bundled_dependencies
        arrow_compute
        arrow_dataset
        parquet
        antlr4-runtime
        ${GLOG_LIB}
        ${GFLAGS_LIB}
        ${PROTOBUF_LIB}
        lz4
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
    )
else()
    # Windows static libraries use different naming conventions
    set(PROTOBUF_LIB libprotobuf)
    set(zvec_c_api_deps
        roaring
        rocksdb
        arrow_static
        arrow_acero_static
        arrow_bundled_dependencies
        arrow_compute_static
        arrow_dataset_static
        parquet_static
        antlr4-runtime-static
        ${GLOG_LIB}
        ${GFLAGS_LIB}
        ${PROTOBUF_LIB}
        lz4
        ${CMAKE_THREAD_LIBS_INIT}
        rpcrt4
        shlwapi
    )
endif()

# Create INTERFACE target for zvec_c_api with platform-specific linking
add_library(zvec-c-api INTERFACE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(zvec-c-api INTERFACE
        -Wl,--whole-archive
        -lzvec_c_api
        -Wl,--no-whole-archive
        -Wl,--start-group
        ${zvec_c_api_deps}
        -Wl,--end-group
    )
elseif(APPLE)
    # On macOS, use -Wl,-force_load with full path to ensure all symbols are loaded
    target_link_libraries(zvec-c-api INTERFACE
        -Wl,-force_load
        ${ZVEC_LIB_DIR}/libzvec_c_api.dylib
        ${zvec_c_api_deps}
    )
    # Set rpath so the executable can find the dylib at runtime
    target_link_options(zvec-c-api INTERFACE
        -Wl,-rpath,${ZVEC_LIB_DIR}
    )
elseif(ANDROID)
    target_link_libraries(zvec-c-api INTERFACE
        -Wl,--whole-archive
        -lzvec_c_api
        -Wl,--no-whole-archive
        -Wl,--start-group
        ${zvec_c_api_deps}
        -Wl,--end-group
    )
elseif(WIN32)
    target_link_libraries(zvec-c-api INTERFACE
        zvec_c_api
        ${zvec_c_api_deps}
    )
    target_link_options(zvec-c-api INTERFACE "/WHOLEARCHIVE:zvec_c_api")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
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