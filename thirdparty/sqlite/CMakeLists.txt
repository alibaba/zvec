cmake_minimum_required(VERSION 3.12)
project(sqlite3 C)

set(SQLITE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sqlite")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GEN_DIR})

# Determine platform-specific commands to generate sqlite3.c
if(CMAKE_HOST_WIN32)
    # Windows uses the official batch script to compile generation tools and emit files
    set(GEN_COMMAND ${SQLITE_SRC_DIR}/make.bat sqlite3.c)
else()
    # Unix-like platforms run configure and make
    set(GEN_COMMAND cd "${SQLITE_SRC_DIR}" && ./configure --all && make sqlite3.c)
endif()

# Automatically trigger the generation sequence right before compilation
add_custom_command(
    OUTPUT "${GEN_DIR}/sqlite3.c" "${GEN_DIR}/sqlite3.h"
    COMMAND ${GEN_COMMAND}
    COMMAND ${CMAKE_COMMAND} -E copy "${SQLITE_SRC_DIR}/sqlite3.c" "${GEN_DIR}/sqlite3.c"
    COMMAND ${CMAKE_COMMAND} -E copy "${SQLITE_SRC_DIR}/sqlite3.h" "${GEN_DIR}/sqlite3.h"
    COMMENT "Generating official SQLite3 amalgamation file cross-platform..."
)

# Define the library from the single generated file
add_library(sqlite3 STATIC "${GEN_DIR}/sqlite3.c")
target_include_directories(sqlite3 PUBLIC ${GEN_DIR})

# Thread safety configurations
target_compile_definitions(sqlite3 PRIVATE SQLITE_THREADSAFE=1)

if(UNIX AND NOT APPLE)
    target_link_libraries(sqlite3 PRIVATE m pthread dl)
endif()

