cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec-example-c++)
set(CMAKE_CXX_STANDARD 17)

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to Zvec and dependencies ---
set(ZVEC_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../../../src/include)
set(ZVEC_LIB_DIR ${CMAKE_BINARY_DIR}/../../../build/lib)
set(ZVEC_DEPENDENCY_LIB_DIR ${CMAKE_BINARY_DIR}/../../../build/external/usr/local/lib)

# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR} ${ZVEC_DEPENDENCY_LIB_DIR})

# --- Determine debug/release library names ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLOG_LIB glogd)
    set(GFLAGS_LIB gflags_nothreads_debug)
    set(PROTOBUF_LIB protobufd)
else()
    set(GLOG_LIB glog)
    set(GFLAGS_LIB gflags_nothreads)
    set(PROTOBUF_LIB protobuf)
endif()

# --- Dependency groups ---
find_package(Threads REQUIRED)

set(zvec_ailego_deps
    arrow
    parquet
    arrow_bundled_dependencies
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

set(zvec_core_deps
    # empty
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

# --- Create INTERFACE targets for Zvec components ---

# zvec_ailego: links libzvec_ailego.a + its deps
add_library(zvec-ailego INTERFACE)
target_link_libraries(zvec-ailego INTERFACE
    -lzvec_ailego
    ${zvec_ailego_deps}
)

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
endif()


# --- Main executable ---
add_executable(db-example db/main.cc)
target_link_libraries(db-example PRIVATE
    zvec-db
)

add_executable(core-example core/main.cc)
target_link_libraries(core-example PRIVATE
    zvec-core
)

add_executable(ailego-example ailego/main.cc)
target_link_libraries(ailego-example PRIVATE
    zvec-ailego
)
