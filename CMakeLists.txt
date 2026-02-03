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
    set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

include_directories(${PROJECT_ROOT_DIR}/src/include)
include_directories(${PROJECT_ROOT_DIR}/src)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")

option(BUILD_TOOLS "Build tools" ON)
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

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

    MESSAGE(STATUS "Zvec install path: ${ZVEC_PY_INSTALL_DIR}")
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR})
endif()
