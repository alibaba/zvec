cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Paths to pre-built zvec and dependencies ---
# These assume zvec was built in <zvec_root>/build/ via the standard CMake flow.
# Override with -DZVEC_ROOT_DIR=... if needed.
if(NOT DEFINED ZVEC_ROOT_DIR)
    set(ZVEC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Root of zvec source tree")
endif()

set(ZVEC_INCLUDE_DIR "${ZVEC_ROOT_DIR}/src/include")
set(ZVEC_SRC_DIR "${ZVEC_ROOT_DIR}/src")
set(ZVEC_LIB_DIR "${ZVEC_ROOT_DIR}/build/lib")
set(ZVEC_DEPENDENCY_LIB_DIR "${ZVEC_ROOT_DIR}/build/external/usr/local/lib")

# --- Extension source files ---
set(EXTENSION_SOURCES
    src/zvec_extension.cpp
    src/collection_registry.cpp
    src/type_conversion.cpp
    src/schema_parser.cpp
    src/functions/zvec_create.cpp
    src/functions/zvec_insert.cpp
    src/functions/zvec_search.cpp
    src/functions/zvec_fetch.cpp
)

# Build the loadable extension
build_static_extension(zvec ${EXTENSION_SOURCES})
build_loadable_extension(zvec "" ${EXTENSION_SOURCES})

# --- Include paths ---
target_include_directories(zvec_loadable_extension PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${ZVEC_INCLUDE_DIR}"
    "${ZVEC_SRC_DIR}"
)
target_include_directories(zvec_extension PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${ZVEC_INCLUDE_DIR}"
    "${ZVEC_SRC_DIR}"
)

# --- Library search paths ---
link_directories(${ZVEC_LIB_DIR} ${ZVEC_DEPENDENCY_LIB_DIR})

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

find_package(Threads REQUIRED)

# --- Dependency lists (mirrors examples/c++/CMakeLists.txt) ---
set(ZVEC_ALL_DEPS
    ${ZVEC_LIB_DIR}/libzvec_db.a
    ${ZVEC_LIB_DIR}/libzvec_core.a
    ${ZVEC_LIB_DIR}/libzvec_ailego.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libroaring.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/librocksdb.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libarrow.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libarrow_acero.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libarrow_bundled_dependencies.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libarrow_compute.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libarrow_dataset.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libparquet.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/libantlr4-runtime.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/lib${GLOG_LIB}.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/lib${GFLAGS_LIB}.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/lib${PROTOBUF_LIB}.a
    ${ZVEC_DEPENDENCY_LIB_DIR}/liblz4.a
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

# --- Link zvec libraries with whole-archive for self-registering components ---
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ZVEC_LINK_FLAGS
        -Wl,--whole-archive
        ${ZVEC_LIB_DIR}/libzvec_core.a
        -Wl,--no-whole-archive
        -Wl,--start-group
        ${ZVEC_ALL_DEPS}
        -Wl,--end-group
    )
elseif(APPLE)
    set(ZVEC_LINK_FLAGS
        -Wl,-force_load,${ZVEC_LIB_DIR}/libzvec_core.a
        ${ZVEC_ALL_DEPS}
    )
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

target_link_libraries(zvec_loadable_extension ${ZVEC_LINK_FLAGS})
target_link_libraries(zvec_extension ${ZVEC_LINK_FLAGS})
