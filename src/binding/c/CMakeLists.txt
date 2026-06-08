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

# On Windows, define ZVEC_BUILD_SHARED so that c_api.h uses __declspec(dllexport)
if(MSVC OR WIN32)
    target_compile_definitions(zvec_c_api PRIVATE ZVEC_BUILD_SHARED)
endif()

find_package(Threads REQUIRED)

# Static linking of C++ standard library is handled in platform-specific sections

# Determine whole-archive flag for current platform
if(MSVC)
    # MSVC uses /WHOLEARCHIVE:filename syntax
    set(ZVEC_WHOLE_ARCHIVE_PREFIX "/WHOLEARCHIVE:")
    set(ZVEC_NO_WHOLE_ARCHIVE_SUFFIX "")
elseif(APPLE)
    # macOS uses -Wl,-force_load,filename
    set(ZVEC_WHOLE_ARCHIVE_PREFIX "-Wl,-force_load,")
    set(ZVEC_NO_WHOLE_ARCHIVE_SUFFIX "")
else()
    # Linux/Unix uses -Wl,--whole-archive and -Wl,--no-whole-archive
    set(ZVEC_WHOLE_ARCHIVE_PREFIX "-Wl,--whole-archive")
    set(ZVEC_NO_WHOLE_ARCHIVE_SUFFIX "-Wl,--no-whole-archive")
endif()

# FAT mode: embed ALL libraries (including third-party) statically
# This creates a truly self-contained library with zero external dependencies
# Users only need to link libzvec_c_api.so without installing any dependencies
if(MSVC)
    target_link_options(zvec_c_api PRIVATE
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_db>
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_core>
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_ailego>
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_turbo>
    )

    target_link_libraries(zvec_c_api
        PRIVATE
            zvec_db
            zvec_core
            zvec_ailego
            zvec_turbo
            ${ZVEC_CXX_STD_LIB}
            Threads::Threads
    )
elseif(APPLE)
    target_link_options(zvec_c_api PRIVATE
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_db>
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_core>
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_ailego>
        ${ZVEC_WHOLE_ARCHIVE_PREFIX}$<TARGET_FILE:zvec_turbo>
    )
    
    # Link the libraries themselves (without force_load flags)
    target_link_libraries(zvec_c_api
        PRIVATE
            zvec_db
            zvec_core
            zvec_ailego
            zvec_turbo
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
    
    # Optional: Static link libc++ for better portability (uncomment if needed)
    # target_link_options(zvec_c_api PRIVATE -static-libstdc++)

else()
    # Linux and other Unix-like systems
    target_link_libraries(zvec_c_api
        PRIVATE
            # Force load all zvec internal libraries (extract all objects)
            ${ZVEC_WHOLE_ARCHIVE_PREFIX}
                zvec_db
                zvec_core
                zvec_ailego
                zvec_turbo  # Required by zvec_core (metric module)
            ${ZVEC_NO_WHOLE_ARCHIVE_SUFFIX}
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
    
    # Link C++ standard library appropriately for the platform
    # On Linux, we need to explicitly link libstdc++
    # Try static linking first, fall back to dynamic
    set(STATIC_LIBS_FOUND FALSE)
    
    # Try to find static libraries using compiler's built-in search
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.a
        OUTPUT_VARIABLE LIBSTDCPP_A_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcc.a
        OUTPUT_VARIABLE LIBGCC_A_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Check if both static libraries exist (not just filename returned)
    # If library not found, -print-file-name returns just the filename without path
    if(EXISTS "${LIBSTDCPP_A_PATH}" AND EXISTS "${LIBGCC_A_PATH}")
        # Verify we got actual paths, not just filenames
        if(IS_ABSOLUTE "${LIBSTDCPP_A_PATH}" AND IS_ABSOLUTE "${LIBGCC_A_PATH}")
            set(STATIC_LIBS_FOUND TRUE)
            message(STATUS "Found static libraries: ${LIBSTDCPP_A_PATH}")
        endif()
    endif()
    
    if(STATIC_LIBS_FOUND)
        # Both static libraries found - use them
        target_link_options(zvec_c_api PRIVATE
            -static-libstdc++
            -static-libgcc
        )
        message(STATUS "Using static libstdc++ and libgcc")
    else()
        # Fall back to dynamic linking
        # Note: We still need to link the libraries, just not statically
        message(STATUS "Static libraries not found, using dynamic linking for libstdc++ and libgcc")
    endif()
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

# Strip symbols in release builds to reduce library size
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(UNIX AND NOT APPLE)
        # Linux: Use strip command to remove symbols
        add_custom_command(TARGET zvec_c_api POST_BUILD
            COMMAND ${CMAKE_STRIP} $<TARGET_FILE:zvec_c_api>
            COMMENT "Stripping symbols from libzvec_c_api.so to reduce size"
        )
    elseif(APPLE)
        # macOS: Use strip with platform-specific flags
        add_custom_command(TARGET zvec_c_api POST_BUILD
            COMMAND /usr/bin/strip -x $<TARGET_FILE:zvec_c_api>
            COMMENT "Stripping symbols from libzvec_c_api.dylib to reduce size"
        )
    endif()
endif()

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
