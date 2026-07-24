cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
project(thirdparty)

include(${PROJECT_ROOT_DIR}/cmake/utils.cmake)

# Optimize third-party libraries for size instead of speed to reduce the final
# binary size. zvec's own code (built under src/) keeps its default speed
# optimization since these flags are directory-scoped to thirdparty/.
# - GCC/Clang: use -Os in place of -O3
# - MSVC: favor small code (/O1, /Ob1) in place of /O2, /Ob2, while preserving
#   the runtime library and other default Release flags
if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
elseif(MSVC)
    string(REPLACE "/O2" "/O1" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/O2" "/O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "/Ob1" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "/Ob1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(EXTERNAL_BINARY_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INC_DIR ${CMAKE_BINARY_DIR}/external/usr/local/include)
#TODO(windows): fix EXTERNAL_LIB_DIR on Windows
set(EXTERNAL_LIB_DIR ${CMAKE_BINARY_DIR}/external/usr/local/lib)
file(MAKE_DIRECTORY ${EXTERNAL_INC_DIR})
file(MAKE_DIRECTORY ${EXTERNAL_LIB_DIR})

add_subdirectory(sparsehash sparsehash EXCLUDE_FROM_ALL)
add_subdirectory(magic_enum magic_enum EXCLUDE_FROM_ALL)
add_subdirectory(RaBitQ-Library RaBitQ-Library EXCLUDE_FROM_ALL)
add_subdirectory(googletest googletest EXCLUDE_FROM_ALL)

add_subdirectory(arrow arrow EXCLUDE_FROM_ALL)
add_subdirectory(gflags gflags EXCLUDE_FROM_ALL)
add_subdirectory(glog glog EXCLUDE_FROM_ALL)
add_subdirectory(yaml-cpp yaml-cpp EXCLUDE_FROM_ALL)
add_subdirectory(protobuf protobuf EXCLUDE_FROM_ALL)
add_subdirectory(antlr antlr EXCLUDE_FROM_ALL)
add_subdirectory(lz4 lz4 EXCLUDE_FROM_ALL)
add_subdirectory(rocksdb rocksdb EXCLUDE_FROM_ALL)
add_subdirectory(CRoaring CRoaring EXCLUDE_FROM_ALL)
add_subdirectory(FastPFOR FastPFOR EXCLUDE_FROM_ALL)
add_subdirectory(limonp limonp EXCLUDE_FROM_ALL)
add_subdirectory(cppjieba cppjieba EXCLUDE_FROM_ALL)
add_subdirectory(utf8proc utf8proc EXCLUDE_FROM_ALL)
add_subdirectory(snowball snowball EXCLUDE_FROM_ALL)
