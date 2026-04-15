cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0077 NEW)
project(zvec-example-c++)
set(CMAKE_CXX_STANDARD 17)

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to Zvec and dependencies ---
# Allow custom host build directory, default to "build"
if(NOT DEFINED HOST_BUILD_DIR)
    set(HOST_BUILD_DIR "build")
endif()

set(ZVEC_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../src/include)
set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/lib)
set(ZVEC_DEPENDENCY_LIB_DIR ${CMAKE_BINARY_DIR}/../../../${HOST_BUILD_DIR}/external/usr/local/lib)

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR} ${ZVEC_DEPENDENCY_LIB_DIR})

# Support multi-config builds (libs in Release/ or Debug/ subdirectories)
if (CMAKE_BUILD_TYPE)
    link_directories(
            ${ZVEC_LIB_DIR}/${CMAKE_BUILD_TYPE}
            ${ZVEC_DEPENDENCY_LIB_DIR}/${CMAKE_BUILD_TYPE}
    )
endif ()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# --- Determine debug/release library names ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLOG_LIB glogd)
    if (WIN32)
        set(GFLAGS_LIB gflags_nothreads_static_debug)
    else ()
        set(GFLAGS_LIB gflags_nothreads_debug)
    endif ()
    set(PROTOBUF_LIB protobufd)
else()
    set(GLOG_LIB glog)
    if (WIN32)
        set(GFLAGS_LIB gflags_nothreads_static)
    else ()
        set(GFLAGS_LIB gflags_nothreads)
    endif ()
    set(PROTOBUF_LIB protobuf)
endif()

# --- Dependency groups ---
find_package(Threads REQUIRED)


set(zvec_core_deps
    zvec_turbo
)

if (NOT WIN32)
    set(zvec_ailego_deps
            arrow
            parquet
            arrow_bundled_dependencies
            ${CMAKE_THREAD_LIBS_INIT}
            ${CMAKE_DL_LIBS}
    )

    set(zvec_db_deps
            roaring
            rocksdb
            arrow
            arrow_acero
            arrow_bundled_dependencies
            arrow_compute
            arrow_dataset
            parquet
            antlr4-runtime
            ${GLOG_LIB}
            ${GFLAGS_LIB}
            ${PROTOBUF_LIB}
            lz4
    )
else ()
    # Windows static libraries use different naming conventions
    set(PROTOBUF_LIB libprotobuf)
    set(zvec_ailego_deps
            arrow_static
            parquet_static
            arrow_bundled_dependencies
            ${CMAKE_THREAD_LIBS_INIT}
            ${CMAKE_DL_LIBS}
    )
    set(zvec_db_deps
            roaring
            rocksdb
            arrow_static
            arrow_acero_static
            arrow_bundled_dependencies
            arrow_compute_static
            arrow_dataset_static
            parquet_static
            antlr4-runtime-static
            ${GLOG_LIB}
            ${GFLAGS_LIB}
            ${PROTOBUF_LIB}
            lz4
            rpcrt4
            shlwapi
    )
endif ()

# --- Create INTERFACE targets for Zvec components ---

# zvec_ailego: links libzvec_ailego.a + its deps
add_library(zvec-ailego INTERFACE)
if (WIN32)
    target_link_libraries(zvec-ailego INTERFACE
            zvec_ailego
            ${zvec_ailego_deps}
    )
else ()
    target_link_libraries(zvec-ailego INTERFACE
            -lzvec_ailego
            ${zvec_ailego_deps}
    )
endif ()

# zvec_core: links libzvec_core.a via special flags (handled externally), but declare logical deps
add_library(zvec-core INTERFACE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(zvec-core INTERFACE
        -Wl,--whole-archive
        zvec_core
        -Wl,--no-whole-archive
        -Wl,--start-group
        zvec-ailego
        ${zvec_core_deps}
        -Wl,--end-group
    )
elseif(APPLE)
    target_link_libraries(zvec-core INTERFACE
        -Wl,-force_load ${ZVEC_LIB_DIR}/libzvec_core.a
        zvec-ailego
        ${zvec_core_deps}
    )
elseif(ANDROID)
    target_link_libraries(zvec-core INTERFACE
        -Wl,--whole-archive
        zvec_core
        -Wl,--no-whole-archive
        -Wl,--start-group
        zvec-ailego
        ${zvec_core_deps}
        -Wl,--end-group
    )
elseif (WIN32)
    target_link_libraries(zvec-core INTERFACE
            zvec_core
            zvec-ailego
            ${zvec_core_deps}
    )
    target_link_options(zvec-core INTERFACE "/WHOLEARCHIVE:zvec_core")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# zvec_db: links libzvec_db.a + all deps
add_library(zvec-db INTERFACE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(zvec-db INTERFACE
        zvec_db
        zvec-core
        zvec-ailego
        -Wl,--start-group
        ${zvec_db_deps}
        -Wl,--end-group
    )
elseif(APPLE)
    target_link_libraries(zvec-db INTERFACE
        zvec_db
        zvec-core
        zvec-ailego
        ${zvec_db_deps}
    )
elseif(ANDROID)
    target_link_libraries(zvec-db INTERFACE
        zvec_db
        zvec-core
        zvec-ailego
        -Wl,--start-group
        ${zvec_db_deps}
        -Wl,--end-group
    )
elseif (WIN32)
    target_link_libraries(zvec-db INTERFACE
            zvec_db
            zvec-core
            zvec-ailego
            ${zvec_db_deps}
    )
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()


# --- Main executable ---
add_executable(db-example db/main.cc)
target_link_libraries(db-example PRIVATE
    zvec-db
)
if(ANDROID)
    target_link_libraries(db-example PRIVATE
        log
    )
endif()

add_executable(core-example core/main.cc)
target_link_libraries(core-example PRIVATE
    zvec-core
)

add_executable(ailego-example ailego/main.cc)
target_link_libraries(ailego-example PRIVATE
    zvec-ailego
)

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
endif()

# Optimize for size
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ANDROID)
    set_property(TARGET db-example core-example ailego-example
                 PROPERTY COMPILE_FLAGS "-Os")
    set_property(TARGET db-example core-example ailego-example
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
