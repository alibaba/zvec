cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(zvec-example-c++)
endif()
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
include_directories(${ZVEC_INCLUDE_DIR})

if(ANDROID OR IOS)
    if(NOT TARGET zvec::static OR
       NOT TARGET zvec::core_static OR
       NOT TARGET zvec::ailego_static)
        message(FATAL_ERROR
            "Mobile C++ examples must be built from the zvec root with "
            "-DBUILD_CPP_EXAMPLES=ON so they use the static SDK targets.")
    endif()

    add_library(zvec-lib INTERFACE)
    target_link_libraries(zvec-lib INTERFACE zvec::static)

    add_library(zvec-ailego-lib INTERFACE)
    target_link_libraries(zvec-ailego-lib INTERFACE zvec::ailego_static)

    add_library(zvec-core-lib INTERFACE)
    target_link_libraries(zvec-core-lib INTERFACE zvec::core_static)
elseif(TARGET zvec_shared AND
       TARGET zvec_core_shared AND
       TARGET zvec_ailego_shared)
    # An in-tree desktop build can link targets directly; the shared-library
    # files do not need to exist yet during CMake configuration.
    add_library(zvec-lib INTERFACE)
    target_link_libraries(zvec-lib INTERFACE zvec_shared)

    add_library(zvec-ailego-lib INTERFACE)
    target_link_libraries(zvec-ailego-lib INTERFACE zvec_ailego_shared)

    add_library(zvec-core-lib INTERFACE)
    target_link_libraries(zvec-core-lib INTERFACE zvec_core_shared)
else()
    set(ZVEC_LIB_DIR ${ZVEC_ROOT_DIR}/${HOST_BUILD_DIR}/lib)
    set(ZVEC_LIB_SEARCH_DIRS ${ZVEC_LIB_DIR})

    # Support multi-config builds (MSVC puts libs in Debug/Release subdirectories)
    if(CMAKE_BUILD_TYPE)
        set(ZVEC_CONFIG_LIB_DIR ${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE})
        if(EXISTS "${ZVEC_CONFIG_LIB_DIR}")
            list(APPEND ZVEC_LIB_SEARCH_DIRS ${ZVEC_CONFIG_LIB_DIR})
        endif()
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

    zvec_require_shared_library(ZVEC_SHARED_LIBRARY zvec)
    zvec_require_shared_library(ZVEC_AILEGO_SHARED_LIBRARY zvec_ailego)
    zvec_require_shared_library(ZVEC_CORE_SHARED_LIBRARY zvec_core)

    # Desktop examples keep using the public all-in-one shared libraries.
    add_library(zvec-lib INTERFACE)
    target_link_libraries(zvec-lib INTERFACE "${ZVEC_SHARED_LIBRARY}")

    add_library(zvec-ailego-lib INTERFACE)
    target_link_libraries(zvec-ailego-lib INTERFACE "${ZVEC_AILEGO_SHARED_LIBRARY}")

    add_library(zvec-core-lib INTERFACE)
    target_link_libraries(zvec-core-lib INTERFACE "${ZVEC_CORE_SHARED_LIBRARY}")
endif()

# --- Executables ---
set(ZVEC_EXAMPLE_TARGETS)

add_executable(db-example db/main.cc)
target_link_libraries(db-example PRIVATE zvec-lib)
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

# This is also a link-integration check: DiskANN must work through the
# all-in-one core library without linking core_knn_diskann separately.
add_executable(diskann-core-example core/diskann_main.cc)
target_link_libraries(diskann-core-example PRIVATE zvec-core-lib)
list(APPEND ZVEC_EXAMPLE_TARGETS diskann-core-example)

add_custom_target(zvec_cpp_examples DEPENDS ${ZVEC_EXAMPLE_TARGETS})

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
