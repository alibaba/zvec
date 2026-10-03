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

# SQLite loadable extension (zvec.so / zvec.dylib / zvec.dll).
#
# The extension is written in C against the public C API, so it links the
# existing "fat" shared library zvec_c_api instead of re-creating its
# whole-archive recipe. It needs only SQLite's headers (sqlite3.h and
# sqlite3ext.h): at runtime the host process provides every sqlite3_* function
# through the extension function table, so the module never links libsqlite3.

if(NOT TARGET zvec_c_api)
    message(FATAL_ERROR
        "BUILD_SQLITE_BINDINGS requires BUILD_C_BINDINGS=ON "
        "(the extension links the zvec_c_api target).")
endif()

if(IOS)
    # iOS cannot dlopen arbitrary modules. A static variant registered with
    # sqlite3_auto_extension() is needed there and is not implemented yet.
    message(WARNING "SQLite loadable extension is not supported on iOS; skipping.")
    return()
endif()

find_path(ZVEC_SQLITE_INCLUDE_DIR
    NAMES sqlite3ext.h
    DOC "Directory containing sqlite3.h and sqlite3ext.h")
if(NOT ZVEC_SQLITE_INCLUDE_DIR OR NOT EXISTS "${ZVEC_SQLITE_INCLUDE_DIR}/sqlite3.h")
    message(FATAL_ERROR
        "sqlite3.h and sqlite3ext.h not found. Install the SQLite development "
        "package (e.g. sqlite-devel / libsqlite3-dev) or pass "
        "-DZVEC_SQLITE_INCLUDE_DIR=<dir with both headers>.")
endif()
message(STATUS "SQLite headers: ${ZVEC_SQLITE_INCLUDE_DIR}")

add_library(zvec_sqlite MODULE zvec_sqlite.c)

set_target_properties(zvec_sqlite PROPERTIES
    # File name must be zvec.<ext> (no "lib" prefix) so SQLite's default
    # entry point resolves to sqlite3_zvec_init.
    OUTPUT_NAME "zvec"
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    # Own directory: on Windows zvec.dll would otherwise collide with the
    # all-in-one C++ shared library (zvec_shared, OUTPUT_NAME zvec) in bin/.
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/sqlite"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/sqlite"
)

target_include_directories(zvec_sqlite PRIVATE "${ZVEC_SQLITE_INCLUDE_DIR}")
target_link_libraries(zvec_sqlite PRIVATE zvec_c_api)

# The repo builds with CMAKE_BUILD_WITH_INSTALL_RPATH (cmake/bazel.cmake), where
# the default "$ORIGIN/../lib" assumes the binary sits in bin/. The module lives
# in lib/sqlite/, one level below libzvec_c_api, so it needs "$ORIGIN/.." (build
# tree) and "$ORIGIN" (installed next to the shared library).
if(APPLE)
    set_target_properties(zvec_sqlite PROPERTIES
        INSTALL_RPATH "@loader_path/..;@loader_path")
elseif(UNIX)
    set_target_properties(zvec_sqlite PROPERTIES
        INSTALL_RPATH "$ORIGIN/..;$ORIGIN")
endif()

target_compile_options(zvec_sqlite PRIVATE
    $<$<C_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<C_COMPILER_ID:Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# Export only sqlite3_zvec_init and drop unreachable code.
if(APPLE)
    set(_zvec_sqlite_exports "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols.txt")
    target_link_options(zvec_sqlite PRIVATE
        -Wl,-dead_strip
        "-Wl,-exported_symbols_list,${_zvec_sqlite_exports}")
    set_property(TARGET zvec_sqlite APPEND PROPERTY
        LINK_DEPENDS "${_zvec_sqlite_exports}")
elseif(UNIX)
    set(_zvec_sqlite_exports "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols.lds")
    target_link_options(zvec_sqlite PRIVATE
        -Wl,--gc-sections
        "-Wl,--version-script=${_zvec_sqlite_exports}")
    set_property(TARGET zvec_sqlite APPEND PROPERTY
        LINK_DEPENDS "${_zvec_sqlite_exports}")
endif()
