cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

option(ZVEC_USE_STATIC_CRT "Use static CRT (/MT) instead of dynamic CRT (/MD), default=ON" ON)
if(ZVEC_USE_STATIC_CRT)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

project(zvec-example-c++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(
        _ZVEC_EXAMPLE_COMPILER_FLAGS
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_MINSIZEREL
    )
    if(ZVEC_USE_STATIC_CRT)
        foreach(_ZVEC_EXAMPLE_COMPILER_FLAG ${_ZVEC_EXAMPLE_COMPILER_FLAGS})
            string(REPLACE "/MD" "/MT" ${_ZVEC_EXAMPLE_COMPILER_FLAG} "${${_ZVEC_EXAMPLE_COMPILER_FLAG}}")
        endforeach()
    else()
        foreach(_ZVEC_EXAMPLE_COMPILER_FLAG ${_ZVEC_EXAMPLE_COMPILER_FLAGS})
            string(REPLACE "/MT" "/MD" ${_ZVEC_EXAMPLE_COMPILER_FLAG} "${${_ZVEC_EXAMPLE_COMPILER_FLAG}}")
        endforeach()
    endif()
    unset(_ZVEC_EXAMPLE_COMPILER_FLAGS)
endif()

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

if(ANDROID)
    option(ZVEC_ENABLE_OMEGA "Link examples against OMEGA support from the host build" OFF)
else()
    option(ZVEC_ENABLE_OMEGA "Link examples against OMEGA support from the host build" ON)
endif()

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR})
set(ZVEC_LIB_SEARCH_DIRS
    ${ZVEC_LIB_DIR}
    ${ZVEC_ROOT_DIR}/${HOST_BUILD_DIR}/external/usr/local/lib
)

# Support multi-config builds (MSVC puts libs in Debug/Release subdirectories)
if(CMAKE_BUILD_TYPE)
    set(ZVEC_CONFIG_LIB_DIR ${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE})
    if(EXISTS "${ZVEC_CONFIG_LIB_DIR}")
        list(APPEND ZVEC_LIB_SEARCH_DIRS ${ZVEC_CONFIG_LIB_DIR})
    endif()
endif()
function(zvec_find_shared_library OUT_VAR LIB_NAME)
    unset(${OUT_VAR} CACHE)
    if(WIN32)
        find_library(${OUT_VAR}
            NAMES ${LIB_NAME}_shared ${LIB_NAME}
            PATHS ${ZVEC_LIB_SEARCH_DIRS}
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH
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
            NO_CMAKE_FIND_ROOT_PATH
        )
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${ZVEC_ORIGINAL_LIBRARY_SUFFIXES}")
    endif()
    set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
endfunction()

function(zvec_require_shared_library OUT_VAR LIB_NAME)
    zvec_find_shared_library(${OUT_VAR} ${LIB_NAME})
    if(NOT ${OUT_VAR})
        message(FATAL_ERROR
            "lib${LIB_NAME} shared library was not found in ${ZVEC_LIB_SEARCH_DIRS}. "
            "Build zvec first, or pass -DHOST_BUILD_DIR=<build-dir>.")
    endif()
    set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
endfunction()

if(WIN32)
    set(ZVEC_WINDOWS_LIBRARIES)
    foreach(ZVEC_LIB_SEARCH_DIR ${ZVEC_LIB_SEARCH_DIRS})
        file(GLOB ZVEC_WINDOWS_LIBS_IN_DIR "${ZVEC_LIB_SEARCH_DIR}/*.lib")
        list(APPEND ZVEC_WINDOWS_LIBRARIES ${ZVEC_WINDOWS_LIBS_IN_DIR})
    endforeach()
    list(FILTER ZVEC_WINDOWS_LIBRARIES EXCLUDE REGEX "(_shared|zvec_c_api)\\.lib$")
    list(REMOVE_DUPLICATES ZVEC_WINDOWS_LIBRARIES)
    if(NOT ZVEC_WINDOWS_LIBRARIES)
        message(FATAL_ERROR
            "No zvec libraries were found in ${ZVEC_LIB_SEARCH_DIRS}. "
            "Build zvec first, or pass -DHOST_BUILD_DIR=<build-dir>.")
    endif()

    add_library(zvec-lib INTERFACE)
    target_link_libraries(zvec-lib INTERFACE ${ZVEC_WINDOWS_LIBRARIES} Rpcrt4.lib)

    add_library(zvec-ailego-lib INTERFACE)
    target_link_libraries(zvec-ailego-lib INTERFACE ${ZVEC_WINDOWS_LIBRARIES})

    add_library(zvec-core-lib INTERFACE)
    target_link_libraries(zvec-core-lib INTERFACE ${ZVEC_WINDOWS_LIBRARIES})
else()
    zvec_require_shared_library(ZVEC_SHARED_LIBRARY zvec)
    zvec_require_shared_library(ZVEC_AILEGO_SHARED_LIBRARY zvec_ailego)
    zvec_require_shared_library(ZVEC_CORE_SHARED_LIBRARY zvec_core)

    # --- Create INTERFACE target for libzvec (all-in-one C++ shared library) ---
    # libzvec.so/.dylib already bundles all zvec internal components.
    add_library(zvec-lib INTERFACE)
    target_link_libraries(zvec-lib INTERFACE "${ZVEC_SHARED_LIBRARY}")

    # --- Create INTERFACE target for libzvec_ailego (ailego-only all-in-one library) ---
    add_library(zvec-ailego-lib INTERFACE)
    target_link_libraries(zvec-ailego-lib INTERFACE "${ZVEC_AILEGO_SHARED_LIBRARY}")

    # --- Create INTERFACE target for libzvec_core (core-only all-in-one library) ---
    add_library(zvec-core-lib INTERFACE)
    target_link_libraries(zvec-core-lib INTERFACE "${ZVEC_CORE_SHARED_LIBRARY}")
endif()

# --- Executables ---
set(ZVEC_EXAMPLE_TARGETS)

add_executable(db-example db/main.cc)
target_link_libraries(db-example PRIVATE zvec-lib)
if(ANDROID)
    target_link_libraries(db-example PRIVATE log)
endif()
list(APPEND ZVEC_EXAMPLE_TARGETS db-example)

add_executable(ailego-example ailego/main.cc)
target_link_libraries(ailego-example PRIVATE zvec-ailego-lib)
list(APPEND ZVEC_EXAMPLE_TARGETS ailego-example)

add_executable(core-example core/main.cc)
target_link_libraries(core-example PRIVATE zvec-core-lib)
list(APPEND ZVEC_EXAMPLE_TARGETS core-example)

add_executable(external-vector-example core/external_vector_example.cc)
target_link_libraries(external-vector-example PRIVATE zvec-core-lib)
list(APPEND ZVEC_EXAMPLE_TARGETS external-vector-example)

if(ZVEC_ENABLE_OMEGA)
    add_executable(omega-example omega/main.cc)
    target_link_libraries(omega-example PRIVATE zvec-lib)
    list(APPEND ZVEC_EXAMPLE_TARGETS omega-example)
endif()
# This is also a link-integration check: DiskANN must work through the
# all-in-one core library without linking core_knn_diskann separately.
add_executable(diskann-core-example core/diskann_main.cc)
target_link_libraries(diskann-core-example PRIVATE zvec-core-lib)
list(APPEND ZVEC_EXAMPLE_TARGETS diskann-core-example)

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
