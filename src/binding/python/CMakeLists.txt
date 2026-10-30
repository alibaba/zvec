include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable LTO on musl: Alpine's fortify-headers use always_inline wrappers
# around external symbols (e.g. __orig_vsnprintf), which conflicts with GCC's
# LTO ("function body can be overwritten at link time"). glibc uses compiler
# builtins (__builtin___vsnprintf_chk) and is unaffected. This must run before
# find_package(pybind11), which checks CMAKE_INTERPROCEDURAL_OPTIMIZATION at
# load time to decide whether to inject -flto.
# ZVEC_ON_MUSL is detected globally in the root CMakeLists.txt via ldd output.
if(ZVEC_ON_MUSL AND NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
    message(STATUS "Disabled LTO on musl (conflicts with fortify-headers)")
endif()

find_package(pybind11 REQUIRED)

set(SRC_LISTS
        binding.cc
        model/python_collection.cc
        model/python_doc.cc
        model/python_reranker.cc
        model/param/python_param.cc
        model/schema/python_schema.cc
        model/common/python_config.cc
        typing/python_type.cc
)

pybind11_add_module(_zvec ${SRC_LISTS})

# Ensure any change to the linker version script (exports.map) triggers a
# re-link of _zvec.so. target_link_options() alone is a command-line flag
# and does not register the script file as a build dependency, so stale
# wheels would silently keep the old export set otherwise.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET _zvec APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
elseif (APPLE)
    set_property(TARGET _zvec APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/exports.mac")
endif()

# Use the static SDK's explicit built-in registry on every platform. Target
# names preserve dependency ordering and build dependencies for the linker.
target_link_libraries(_zvec PRIVATE
        zvec
        zvec_core
        zvec_ailego
        zvec_turbo
        ${CMAKE_DL_LIBS}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_zvec PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
    # On musl, dynamic linking would force auditwheel to rewrite the ELF via
    # patchelf so it can vendor libstdc++/libgcc. The patchelf versions used
    # in current build environments (0.17.x/0.18.x) have corrupted binaries
    # on musl 1.2.5 x86_64, causing init crashes. Statically linking
    # libstdc++ and libgcc avoids patchelf entirely and keeps _zvec
    # self-contained.
    if(ZVEC_ON_MUSL)
        target_link_options(_zvec PRIVATE -static-libstdc++ -static-libgcc)
        message(STATUS "_zvec: statically linking libstdc++ and libgcc on musl")
    endif()
    # DiskAnn is statically linked into _zvec.so, so the wheel does not ship
    # a separate runtime library. Its I/O backends retain their existing
    # io_uring, dynamically loaded libaio, and synchronous pread fallbacks.
elseif (APPLE)
    target_link_options(_zvec PRIVATE
            "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.mac"
    )
endif ()

target_include_directories(_zvec PRIVATE ${PYBIND11_INCLUDE_DIR} ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/binding/python/include)
