cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec)
set(CC_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror=return-type")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(NOT DEFINED PROJECT_ROOT_DIR OR NOT PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

include_directories(${PROJECT_ROOT_DIR}/src/include)
include_directories(${PROJECT_ROOT_DIR}/src)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")

option(BUILD_C_BINDINGS "Build C bindings (libzvec_c) for Rust/FFI consumers" OFF)
message(STATUS "BUILD_C_BINDINGS:${BUILD_C_BINDINGS}")

option(BUILD_TOOLS "Build tools" ON)
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

option(USE_OSS_MIRROR "Use OSS mirror for faster third-party downloads" OFF)
if(DEFINED ENV{USE_OSS_MIRROR} AND NOT "$ENV{USE_OSS_MIRROR}" STREQUAL "")
  set(USE_OSS_MIRROR "$ENV{USE_OSS_MIRROR}" CACHE BOOL "Use OSS mirror for faster third-party downloads" FORCE)
endif()
message(STATUS "USE_OSS_MIRROR:${USE_OSS_MIRROR}")

cc_directory(thirdparty)
cc_directories(src)
cc_directories(tests)

if(BUILD_TOOLS)
    cc_directories(tools)
endif()

git_version(GIT_SRCS_VER ${PROJECT_ROOT_DIR})
set(CPACK_PACKAGE_VERSION ${GIT_SRCS_VER})
set(CPACK_PACKAGE_NAME zvec)
include(CPack)

if(BUILD_PYTHON_BINDINGS)
    if(APPLE)
        set(CMAKE_STRIP "")
        message(STATUS "Disabled strip on macOS to preserve code signature")
    endif()

    include(GNUInstallDirs)
    if(DEFINED SKBUILD_PLATLIB_DIR)
        set(ZVEC_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
    elseif(DEFINED Python_SITEARCH)
        set(ZVEC_PY_INSTALL_DIR "${Python_SITEARCH}")
    else()
        set(ZVEC_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    message(STATUS "Zvec install path: ${ZVEC_PY_INSTALL_DIR}")
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR})
endif()