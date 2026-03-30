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

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
include(GNUInstallDirs)

# Retrieve version from git repository and generate version header
git_version(ZVEC_VERSION ${CMAKE_CURRENT_SOURCE_DIR})

# Debug: print version variables
message(STATUS "ZVEC_VERSION: ${ZVEC_VERSION}")

# Parse version string to extract major.minor.patch
# Format: vX.Y.Z-commit-hash or vX.Y.Z
if(ZVEC_VERSION MATCHES "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(ZVEC_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(ZVEC_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(ZVEC_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(ZVEC_VERSION_STRING "${ZVEC_VERSION}")
else()
    # Default version if parsing fails
    set(ZVEC_VERSION_MAJOR 0)
    set(ZVEC_VERSION_MINOR 2)
    set(ZVEC_VERSION_PATCH 1)
    set(ZVEC_VERSION_STRING "${ZVEC_VERSION_MAJOR}.${ZVEC_VERSION_MINOR}.${ZVEC_VERSION_PATCH}")
endif()

message(STATUS "Parsed version: ${ZVEC_VERSION_MAJOR}.${ZVEC_VERSION_MINOR}.${ZVEC_VERSION_PATCH} (${ZVEC_VERSION_STRING})")

# Configure version header file
configure_file(
    ${PROJECT_SOURCE_DIR}/src/include/zvec/version.h.in
    ${PROJECT_BINARY_DIR}/src/generated/zvec_version.h
)

# Configure c_api.h with version information embedded
configure_file(
    ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
    ${PROJECT_BINARY_DIR}/src/generated/zvec/c_api.h
    @ONLY
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C API library source files
set(ZVEC_C_API_SOURCES
    c_api.cc
)

# C API library header files
set(ZVEC_C_API_HEADERS
    ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
)

# =============================================================================
# Build FAT Shared Library (zvec_c_api.so)
# =============================================================================
# BUILD_RELEASE_FAT_LIBS=ON:  Fully self-contained, zero external dependencies
#                             Users only need -lzvec_c_api
# BUILD_RELEASE_FAT_LIBS=OFF: Development mode, third-party libs linked normally
#                             Allows parallel test execution without symbol conflicts
#
# Implementation:
# - Always embeds zvec_db, zvec_core, zvec_ailego via --whole-archive
# - For release: also embeds all third-party libs (rocksdb, glog, protobuf, etc.)
# - Uses --exclude-libs,ALL to hide third-party symbols from export
# =============================================================================
add_library(zvec_c_api SHARED
    ${ZVEC_C_API_SOURCES}
    ${ZVEC_C_API_HEADERS}
)

# Set library properties
set_target_properties(zvec_c_api PROPERTIES
    OUTPUT_NAME "zvec_c_api"
    POSITION_INDEPENDENT_CODE ON
    # Hide all symbols by default, only export C API
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

find_package(Threads REQUIRED)

# FAT mode: embed ALL libraries (including third-party) statically
# This creates a truly self-contained library with zero external dependencies
# Users only need to link libzvec_c_api.so without installing any dependencies
if(APPLE)
    # Third-party libraries that are dependencies of zvec_db and zvec_ailego
    # These must be linked before the internal libraries that depend on them
    target_link_libraries(zvec_c_api
        PRIVATE
            Arrow::arrow_static
            Arrow::parquet_static
            Arrow::arrow_compute
            Arrow::arrow_dataset
            Arrow::arrow_acero
            rocksdb
            glog
            libprotobuf
            antlr4
    )
    
    # Link zvec internal libraries with -force_load to extract all objects
    target_link_libraries(zvec_c_api
        PRIVATE
            # zvec static libraries (force-loaded to extract all objects)
            "-Wl,-force_load,$<TARGET_FILE:zvec_db>"
            "-Wl,-force_load,$<TARGET_FILE:zvec_core>"
            "-Wl,-force_load,$<TARGET_FILE:zvec_ailego>"
            # zvec_turbo is required by zvec_core (metric module)
            "-Wl,-force_load,$<TARGET_FILE:zvec_turbo>"
            # Only roaring needs explicit force_load (C library with no transitive deps)
            "-Wl,-force_load,$<TARGET_FILE:roaring>"
            # Explicitly link C++ standard library for cross-platform compatibility
            stdc++
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )

else()
    target_link_libraries(zvec_c_api
        PRIVATE
            # Force load all zvec internal libraries (extract all objects)
            "-Wl,--whole-archive"
                zvec_db
                zvec_core
                zvec_ailego
                zvec_turbo  # Required by zvec_core (metric module)
            "-Wl,--no-whole-archive"
            # Most third-party libs are linked via zvec_db
            # Only roaring needs explicit whole-archive (C library)
            "-Wl,--whole-archive"
                roaring
            "-Wl,--no-whole-archive"
            # Explicitly link C++ standard library for cross-platform compatibility
            stdc++
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
endif()

# Include directories
target_include_directories(zvec_c_api
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/generated>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Compile options
target_compile_options(zvec_c_api PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# =============================================================================
# Installation Rules
# =============================================================================

# Install shared library
install(TARGETS zvec_c_api
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zvec
)
