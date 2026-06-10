cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec-example-c++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to Zvec ---
# Allow custom host build directory, default to "build"
if(NOT DEFINED HOST_BUILD_DIR)
    set(HOST_BUILD_DIR "build")
endif()

get_filename_component(ZVEC_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(ZVEC_INCLUDE_DIR ${ZVEC_ROOT_DIR}/src/include)
set(ZVEC_LIB_DIR ${ZVEC_ROOT_DIR}/${HOST_BUILD_DIR}/lib)
set(ZVEC_LINK_VIA_CMAKE_TARGET OFF)
if(TARGET zvec_shared OR TARGET zvec_core_shared)
    set(ZVEC_LINK_VIA_CMAKE_TARGET ON)
endif()

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR})
set(ZVEC_LIB_SEARCH_DIRS ${ZVEC_LIB_DIR})

# Support multi-config builds (MSVC puts libs in Debug/Release subdirectories)
if(CMAKE_BUILD_TYPE)
    set(ZVEC_CONFIG_LIB_DIR ${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE})
    if(EXISTS "${ZVEC_CONFIG_LIB_DIR}")
        list(APPEND ZVEC_LIB_SEARCH_DIRS ${ZVEC_CONFIG_LIB_DIR})
    endif()
endif()
if(NOT ZVEC_LINK_VIA_CMAKE_TARGET)
    link_directories(${ZVEC_LIB_SEARCH_DIRS})
    set(CMAKE_BUILD_RPATH ${ZVEC_LIB_SEARCH_DIRS})
endif()
if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

function(zvec_find_shared_library OUT_VAR LIB_NAME)
    unset(${OUT_VAR} CACHE)
    if(WIN32)
        find_library(${OUT_VAR}
            NAMES ${LIB_NAME}_shared ${LIB_NAME}
            PATHS ${ZVEC_LIB_SEARCH_DIRS}
            NO_DEFAULT_PATH
        )
    else()
        set(ZVEC_ORIGINAL_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
        if(APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib")
        else()
            set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
        endif()
        find_library(${OUT_VAR}
            NAMES ${LIB_NAME}
            PATHS ${ZVEC_LIB_SEARCH_DIRS}
            NO_DEFAULT_PATH
        )
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${ZVEC_ORIGINAL_LIBRARY_SUFFIXES}")
    endif()
    set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
endfunction()

set(ZVEC_CORE_EXAMPLE_DEFAULT ON)
if(NOT TARGET zvec_core_shared)
    zvec_find_shared_library(ZVEC_CORE_SHARED_LIBRARY zvec_core)
    if(NOT ZVEC_CORE_SHARED_LIBRARY)
        set(ZVEC_CORE_EXAMPLE_DEFAULT OFF)
    endif()
endif()

set(ZVEC_FULL_EXAMPLES_DEFAULT ON)
if(NOT TARGET zvec_shared)
    zvec_find_shared_library(ZVEC_SHARED_LIBRARY zvec)
    if(NOT ZVEC_SHARED_LIBRARY)
        set(ZVEC_FULL_EXAMPLES_DEFAULT OFF)
    endif()
endif()

option(BUILD_ZVEC_CORE_EXAMPLE "Build zvec-core example linked only with libzvec_core" ${ZVEC_CORE_EXAMPLE_DEFAULT})
option(BUILD_ZVEC_FULL_EXAMPLES "Build examples linked with libzvec" ${ZVEC_FULL_EXAMPLES_DEFAULT})

# --- Create INTERFACE target for libzvec (all-in-one C++ shared library) ---
# libzvec.so/.dylib/.dll already bundles all zvec internal components
# (zvec, zvec_core, zvec_ailego, zvec_turbo), so no individual dependency
# libraries need to be specified by the consumer.
if(BUILD_ZVEC_FULL_EXAMPLES)
    add_library(zvec-lib INTERFACE)
    if(TARGET zvec_shared)
        target_link_libraries(zvec-lib INTERFACE zvec_shared)
    else()
        if(ZVEC_SHARED_LIBRARY)
            target_link_libraries(zvec-lib INTERFACE "${ZVEC_SHARED_LIBRARY}")
        else()
            message(FATAL_ERROR
                "libzvec shared library was not found in ${ZVEC_LIB_SEARCH_DIRS}. "
                "Build zvec with -DBUILD_ZVEC_SHARED=ON or disable BUILD_ZVEC_FULL_EXAMPLES.")
        endif()
    endif()
endif()

# --- Create INTERFACE target for libzvec_core (core-only all-in-one library) ---
# The core example intentionally depends only on libzvec_core.
if(BUILD_ZVEC_CORE_EXAMPLE)
    add_library(zvec-core-lib INTERFACE)
    if(TARGET zvec_core_shared)
        target_link_libraries(zvec-core-lib INTERFACE zvec_core_shared)
    else()
        zvec_find_shared_library(ZVEC_CORE_SHARED_LIBRARY zvec_core)
        if(ZVEC_CORE_SHARED_LIBRARY)
            target_link_libraries(zvec-core-lib INTERFACE "${ZVEC_CORE_SHARED_LIBRARY}")
        else()
            message(FATAL_ERROR
                "libzvec_core shared library was not found in ${ZVEC_LIB_SEARCH_DIRS}. "
                "Build zvec with -DBUILD_ZVEC_CORE_SHARED=ON or "
                "-DBUILD_ZVEC_CORE_ONLY=ON, or disable BUILD_ZVEC_CORE_EXAMPLE.")
        endif()
    endif()
endif()

# --- Executables ---
set(ZVEC_EXAMPLE_TARGETS)

if(BUILD_ZVEC_FULL_EXAMPLES)
    add_executable(db-example db/main.cc)
    target_link_libraries(db-example PRIVATE zvec-lib)
    if(ANDROID)
        target_link_libraries(db-example PRIVATE log)
    endif()
    list(APPEND ZVEC_EXAMPLE_TARGETS db-example)

    add_executable(ailego-example ailego/main.cc)
    target_link_libraries(ailego-example PRIVATE zvec-lib)
    list(APPEND ZVEC_EXAMPLE_TARGETS ailego-example)
endif()

if(BUILD_ZVEC_CORE_EXAMPLE)
    add_executable(core-example core/main.cc)
    target_link_libraries(core-example PRIVATE zvec-core-lib)
    list(APPEND ZVEC_EXAMPLE_TARGETS core-example)
endif()

# Strip symbols to reduce executable size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ANDROID)
    foreach(ZVEC_EXAMPLE_TARGET ${ZVEC_EXAMPLE_TARGETS})
        add_custom_command(TARGET ${ZVEC_EXAMPLE_TARGET} POST_BUILD
            COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:${ZVEC_EXAMPLE_TARGET}>"
            COMMENT "Stripping symbols from ${ZVEC_EXAMPLE_TARGET}")
    endforeach()
endif()

# Optimize for size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ANDROID AND ZVEC_EXAMPLE_TARGETS)
    set_property(TARGET ${ZVEC_EXAMPLE_TARGETS} PROPERTY COMPILE_FLAGS "-Os")
    set_property(TARGET ${ZVEC_EXAMPLE_TARGETS} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
