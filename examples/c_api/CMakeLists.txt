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
project(zvec-example-c-api)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to Zvec and dependencies ---
if(NOT DEFINED HOST_BUILD_DIR)
    set(HOST_BUILD_DIR "build")
endif()

set(ZVEC_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../src/include)
set(ZVEC_GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/src/generated)
set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/lib)

include_directories(${ZVEC_INCLUDE_DIR} ${ZVEC_GENERATED_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR})

if(CMAKE_BUILD_TYPE)
    link_directories(${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE})
endif()

find_package(Threads REQUIRED)

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

# HNSW INT8 build example (no rotation)
add_executable(c_api_hnsw_int8_build hnsw/int8_build.c)
target_link_libraries(c_api_hnsw_int8_build PRIVATE
    zvec-c-api
)

# HNSW INT8 query example (no rotation)
add_executable(c_api_hnsw_int8_query hnsw/int8_query.c)
target_link_libraries(c_api_hnsw_int8_query PRIVATE
    zvec-c-api
)

# HNSW INT8 + Rotate build example
add_executable(c_api_hnsw_int8_rotate_build hnsw/int8_rotate_build.c)
target_link_libraries(c_api_hnsw_int8_rotate_build PRIVATE
    zvec-c-api
)

# HNSW INT8 + Rotate query example
add_executable(c_api_hnsw_int8_rotate_query hnsw/int8_rotate_query.c)
target_link_libraries(c_api_hnsw_int8_rotate_query PRIVATE
    zvec-c-api
)

# Strip symbols in Release
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND (ANDROID OR (CMAKE_SYSTEM_NAME STREQUAL "Linux")))
    add_custom_command(TARGET c_api_hnsw_int8_build POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_hnsw_int8_build>"
        COMMENT "Stripping symbols from c_api_hnsw_int8_build")
    add_custom_command(TARGET c_api_hnsw_int8_query POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_hnsw_int8_query>"
        COMMENT "Stripping symbols from c_api_hnsw_int8_query")
    add_custom_command(TARGET c_api_hnsw_int8_rotate_build POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_hnsw_int8_rotate_build>"
        COMMENT "Stripping symbols from c_api_hnsw_int8_rotate_build")
    add_custom_command(TARGET c_api_hnsw_int8_rotate_query POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:c_api_hnsw_int8_rotate_query>"
        COMMENT "Stripping symbols from c_api_hnsw_int8_rotate_query")
endif()
