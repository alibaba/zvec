set(ROCKSDB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb-8.1.1)
if (ANDROID)
    set(ROCKSDB_ANDROID_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb.android.patch)
    apply_patch_once("rocksdb_android_fix" "${ROCKSDB_SRC_DIR}" "${ROCKSDB_ANDROID_PATCH}")
endif()

set(ROCKSDB_BUILD_SHARED OFF CACHE BOOL "Disable install in rocksdb" FORCE)

if(MSVC)
    if(ZVEC_USE_STATIC_CRT)
        set(WITH_MD_LIBRARY OFF CACHE BOOL "build rocksdb with MT to match project CRT" FORCE)
    else()
        set(WITH_MD_LIBRARY ON CACHE BOOL "build rocksdb with MD to match project CRT" FORCE)
    endif()
endif()

set(WITH_TESTS OFF CACHE BOOL "Disable testing in rocksdb" FORCE)
set(WITH_ALL_TESTS OFF CACHE BOOL "Build all test, rather than a small subset" FORCE)
set(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "Disable benchmarks in rocksdb" FORCE)
set(WITH_CORE_TOOLS OFF CACHE BOOL "build with ldb and sst_dump" FORCE)
set(WITH_TOOLS OFF CACHE BOOL "build with tools" FORCE)
set(WITH_LZ4 ON CACHE BOOL "build with lz4" FORCE)
set(USE_RTTI ON CACHE BOOL "build with RTTI" FORCE)
# TODO(windows): verify
set(ROCKSDB_SKIP_THIRDPARTY ON CACHE BOOL "skip thirdparty.inc" FORCE)
set(FAIL_ON_WARNINGS OFF CACHE BOOL "build with no Werror" FORCE)
set(PORTABLE ON CACHE BOOL "build a portable lib" FORCE)

set(lz4_INCLUDE_DIRS "${lz4_INCLUDE_DIR}" CACHE PATH "lz4 include directory" FORCE)
set(lz4_LIBRARIES "${lz4_LIBRARY}" CACHE FILEPATH "lz4 library" FORCE)
set(lz4_FOUND TRUE CACHE BOOL "lz4 found" FORCE)

# TODO(windows): may remove
if(NOT TARGET lz4::lz4)
    add_library(lz4::lz4 UNKNOWN IMPORTED GLOBAL)
    set_target_properties(lz4::lz4 PROPERTIES
        IMPORTED_LOCATION "${lz4_LIBRARY}"
        IMPORTED_LOCATION_DEBUG "${lz4_LIBRARY}"
        IMPORTED_LOCATION_RELEASE "${lz4_LIBRARY}"
        IMPORTED_LOCATION_MINSIZEREL "${lz4_LIBRARY}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${lz4_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${lz4_INCLUDE_DIR}"
    )
    add_dependencies(lz4::lz4 Lz4.BUILD)
endif()

set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})
add_subdirectory(rocksdb-8.1.1)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

get_target_property(rocksdb_SOURCE_DIR rocksdb SOURCE_DIR)
set(ROCKSDB_INCLUDE_DIR ${rocksdb_SOURCE_DIR}/include)

target_include_directories(rocksdb PUBLIC $<BUILD_INTERFACE:${ROCKSDB_INCLUDE_DIR}>)
target_include_directories(rocksdb PRIVATE ${lz4_INCLUDE_DIR})
add_dependencies(rocksdb Lz4.BUILD)
