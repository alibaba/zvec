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
    # Combine all libraries in a single target_link_libraries call
    target_link_libraries(zvec_c_api
        PRIVATE
            # zvec static libraries
            zvec_db
            zvec_core
            zvec_ailego
            # Third-party libraries
            roaring
            Arrow::arrow_static
            Arrow::parquet_static
            Arrow::arrow_compute
            Arrow::arrow_dataset
            Arrow::arrow_acero
            rocksdb
            glog
            libprotobuf
            antlr4
            sparsehash
            magic_enum
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
    
    # Then use target_link_libraries with -force_load on macOS
    # This ensures all symbols from static libraries are included
    # Note: sparsehash and magic_enum are header-only, skip them
    target_link_libraries(zvec_c_api PRIVATE
        -Wl,-force_load,$<TARGET_FILE:zvec_db>
        -Wl,-force_load,$<TARGET_FILE:zvec_core>
        -Wl,-force_load,$<TARGET_FILE:zvec_ailego>
        -Wl,-force_load,$<TARGET_FILE:roaring>
        -Wl,-force_load,$<TARGET_FILE:Arrow::arrow_static>
        -Wl,-force_load,$<TARGET_FILE:Arrow::parquet_static>
        -Wl,-force_load,$<TARGET_FILE:Arrow::arrow_compute>
        -Wl,-force_load,$<TARGET_FILE:Arrow::arrow_dataset>
        -Wl,-force_load,$<TARGET_FILE:Arrow::arrow_acero>
        -Wl,-force_load,$<TARGET_FILE:rocksdb>
        -Wl,-force_load,$<TARGET_FILE:glog>
        -Wl,-force_load,$<TARGET_FILE:libprotobuf>
        -Wl,-force_load,$<TARGET_FILE:antlr4>
    )

else()
    target_link_libraries(zvec_c_api
        PRIVATE
            # Force load all zvec static libraries (extract all objects)
            "-Wl,--whole-archive"
                zvec_db
                zvec_core
                zvec_ailego
            "-Wl,--no-whole-archive"
            # Force load ALL third-party libraries for zero-dependency deployment
            "-Wl,--whole-archive"
                roaring
                Arrow::arrow_static
                Arrow::parquet_static
                Arrow::arrow_compute
                Arrow::arrow_dataset
                Arrow::arrow_acero
                rocksdb
                glog
                libprotobuf
                antlr4
                sparsehash
                magic_enum
            "-Wl,--no-whole-archive"
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
endif()

# Include directories
target_include_directories(zvec_c_api
    PUBLIC
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
