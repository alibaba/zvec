set(ROCKSDB_BUILD_SHARED OFF CACHE BOOL "Disable install in rocksdb" FORCE)

set(WITH_TESTS OFF CACHE BOOL "Disable testing in rocksdb" FORCE)
set(WITH_ALL_TESTS OFF CACHE BOOL "Build all test, rather than a small subset" FORCE)
set(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "Disable benchmarks in rocksdb" FORCE)
set(WITH_CORE_TOOLS OFF CACHE BOOL "build with ldb and sst_dump" FORCE)
set(WITH_TOOLS OFF CACHE BOOL "build with tools" FORCE)
set(WITH_LZ4 ON CACHE BOOL "build with lz4" FORCE)
set(USE_RTTI ON CACHE BOOL "build with RTTI" FORCE)
set(FAIL_ON_WARNINGS OFF CACHE BOOL "build with no Werror" FORCE)

set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})
add_subdirectory(rocksdb-8.1.1)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

get_target_property(rocksdb_SOURCE_DIR rocksdb SOURCE_DIR)
set(ROCKSDB_INCLUDE_DIR ${rocksdb_SOURCE_DIR}/include)

target_include_directories(rocksdb PUBLIC $<BUILD_INTERFACE:${ROCKSDB_INCLUDE_DIR}>)
add_dependencies(rocksdb Lz4.BUILD)
