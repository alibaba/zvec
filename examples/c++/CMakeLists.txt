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

set(ZVEC_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../src/include)
set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/lib)
set(ZVEC_DEPENDENCY_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/external/usr/local/lib)

if(ANDROID)
    option(ZVEC_ENABLE_OMEGA "Link examples against OMEGA support from the host build" OFF)
else()
    option(ZVEC_ENABLE_OMEGA "Link examples against OMEGA support from the host build" ON)
endif()

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR})

# Support multi-config builds (MSVC puts libs in Debug/Release subdirectories)
if(CMAKE_BUILD_TYPE)
    link_directories(${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE})
endif()
if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(ZVEC_ENABLE_OMEGA)
    # OMEGA support is compiled into libzvec.so; no extra libraries needed
    # at link time for the examples.  We only verify the host build produced
    # the OMEGA static libs so we know the feature was actually compiled in.
    set(_zvec_omega_search_paths
        ${ZVEC_DEPENDENCY_LIB_DIR}
        ${ZVEC_DEPENDENCY_LIB_DIR}/Debug
        ${ZVEC_DEPENDENCY_LIB_DIR}/Release
        ${ZVEC_DEPENDENCY_LIB_DIR}/RelWithDebInfo
        ${ZVEC_DEPENDENCY_LIB_DIR}/MinSizeRel
    )
    find_library(ZVEC_OMEGA_LIB
        NAMES omega
        PATHS ${_zvec_omega_search_paths}
        NO_DEFAULT_PATH
    )
    find_library(ZVEC_LIGHTGBM_LIB
        NAMES lib_lightgbm _lightgbm
        PATHS ${_zvec_omega_search_paths}
        NO_DEFAULT_PATH
    )
    if(NOT ZVEC_OMEGA_LIB OR NOT ZVEC_LIGHTGBM_LIB)
        message(WARNING
            "ZVEC_ENABLE_OMEGA=ON but could not locate OMEGA host libraries under "
            "${ZVEC_DEPENDENCY_LIB_DIR}. "
            "omega-example will still link against libzvec (which should bundle OMEGA)."
        )
    endif()
endif()

# --- Dependency groups ---
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

# --- Create INTERFACE target for libzvec (all-in-one C++ shared library) ---
# libzvec.so/.dylib/.dll already bundles all zvec internal components
# (zvec_db, zvec_core, zvec_ailego, zvec_turbo), so no individual dependency
# libraries need to be specified by the consumer.
add_library(zvec-lib INTERFACE)
target_link_libraries(zvec-lib INTERFACE
    zvec
    Threads::Threads
)
if(NOT WIN32)
    target_link_libraries(zvec-lib INTERFACE ${CMAKE_DL_LIBS})
endif()

# --- Executables ---
add_executable(db-example db/main.cc)
target_link_libraries(db-example PRIVATE zvec-lib)
if(ANDROID)
    target_link_libraries(db-example PRIVATE log)
endif()

add_executable(core-example core/main.cc)
target_link_libraries(core-example PRIVATE zvec-lib)

add_executable(ailego-example ailego/main.cc)
target_link_libraries(ailego-example PRIVATE zvec-lib)

if(ZVEC_ENABLE_OMEGA)
    add_executable(omega-example omega/main.cc)
    target_link_libraries(omega-example PRIVATE zvec-lib)
endif()

# Strip symbols to reduce executable size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ANDROID)
    add_custom_command(TARGET db-example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:db-example>"
        COMMENT "Stripping symbols from db-example")
    add_custom_command(TARGET core-example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:core-example>"
        COMMENT "Stripping symbols from core-example")
    add_custom_command(TARGET ailego-example POST_BUILD
        COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:ailego-example>"
        COMMENT "Stripping symbols from ailego-example")
    if(ZVEC_ENABLE_OMEGA)
        add_custom_command(TARGET omega-example POST_BUILD
            COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:omega-example>"
            COMMENT "Stripping symbols from omega-example")
    endif()
endif()

# Optimize for size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ANDROID)
    set_property(TARGET db-example core-example ailego-example
                 PROPERTY COMPILE_FLAGS "-Os")
    set_property(TARGET db-example core-example ailego-example
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    if(ZVEC_ENABLE_OMEGA)
        set_property(TARGET omega-example
                     PROPERTY COMPILE_FLAGS "-Os")
        set_property(TARGET omega-example
                     PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()
