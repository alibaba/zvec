include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Retrieve version from git repository
git_version(ZVEC_VERSION ${CMAKE_CURRENT_SOURCE_DIR})

# Add repository
cc_directory(ailego)
cc_directory(turbo)
cc_directory(core)
cc_directory(db)
cc_directory(binding)

# =============================================================================
# Build ALL-IN-ONE C++ Shared Library (libzvec.so / libzvec.dylib)
# =============================================================================
# Merges all zvec internal static libraries (zvec_db, zvec_core, zvec_ailego,
# zvec_turbo) into a single shared library, exposing the full C++ interface.
# Unlike zvec_c_api which only exposes C symbols, this library preserves all
# C++ symbols for direct C++ linking.

include(GNUInstallDirs)
find_package(Threads REQUIRED)

# Generate a minimal stub source (add_library SHARED requires at least one source)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/zvec_allin_stub.cc
    "// Auto-generated stub for zvec all-in-one C++ shared library\n"
)

add_library(zvec_shared SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/zvec_allin_stub.cc
)

set_target_properties(zvec_shared PROPERTIES
    OUTPUT_NAME "zvec"
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Determine whole-archive flag for current platform
if(MSVC)
    set(ZVEC_ALLIN_WA_PREFIX "/WHOLEARCHIVE:")
    set(ZVEC_ALLIN_WA_SUFFIX "")
elseif(APPLE)
    set(ZVEC_ALLIN_WA_PREFIX "-Wl,-force_load,")
    set(ZVEC_ALLIN_WA_SUFFIX "")
else()
    set(ZVEC_ALLIN_WA_PREFIX "-Wl,--whole-archive")
    set(ZVEC_ALLIN_WA_SUFFIX "-Wl,--no-whole-archive")
endif()

# Platform-specific whole-archive linking (mirrors binding/c/CMakeLists.txt)
if(MSVC)
    # On MSVC, WINDOWS_EXPORT_ALL_SYMBOLS generates a .def file only from the
    # target's own object files.  /WHOLEARCHIVE: pulls objects into the DLL but
    # they are NOT scanned for the .def, leaving the import library empty.
    # Fix: include static-library objects directly via TARGET_OBJECTS (CMake 3.21+)
    # so the def-file generator sees every symbol.
    target_sources(zvec_shared PRIVATE
        $<TARGET_OBJECTS:zvec_db>
        $<TARGET_OBJECTS:zvec_core>
        $<TARGET_OBJECTS:zvec_ailego>
        $<TARGET_OBJECTS:zvec_turbo>
    )
    target_link_libraries(zvec_shared
        PRIVATE
            zvec_db
            zvec_core
            zvec_ailego
            zvec_turbo
            Threads::Threads
    )
elseif(APPLE)
    target_link_options(zvec_shared PRIVATE
        ${ZVEC_ALLIN_WA_PREFIX}$<TARGET_FILE:zvec_db>
        ${ZVEC_ALLIN_WA_PREFIX}$<TARGET_FILE:zvec_core>
        ${ZVEC_ALLIN_WA_PREFIX}$<TARGET_FILE:zvec_ailego>
        ${ZVEC_ALLIN_WA_PREFIX}$<TARGET_FILE:zvec_turbo>
    )
    target_link_libraries(zvec_shared
        PRIVATE
            zvec_db
            zvec_core
            zvec_ailego
            zvec_turbo
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
else()
    # Linux and other Unix-like systems
    target_link_libraries(zvec_shared
        PRIVATE
            ${ZVEC_ALLIN_WA_PREFIX}
                zvec_db
                zvec_core
                zvec_ailego
                zvec_turbo
            ${ZVEC_ALLIN_WA_SUFFIX}
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )

    # Try static linking of libstdc++ and libgcc for better portability
    set(ALLIN_STATIC_LIBS_FOUND FALSE)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.a
        OUTPUT_VARIABLE ALLIN_LIBSTDCPP_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcc.a
        OUTPUT_VARIABLE ALLIN_LIBGCC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS "${ALLIN_LIBSTDCPP_PATH}" AND EXISTS "${ALLIN_LIBGCC_PATH}")
        if(IS_ABSOLUTE "${ALLIN_LIBSTDCPP_PATH}" AND IS_ABSOLUTE "${ALLIN_LIBGCC_PATH}")
            set(ALLIN_STATIC_LIBS_FOUND TRUE)
        endif()
    endif()
    if(ALLIN_STATIC_LIBS_FOUND)
        target_link_options(zvec_shared PRIVATE -static-libstdc++ -static-libgcc)
        message(STATUS "zvec_shared: Using static libstdc++ and libgcc")
    else()
        message(STATUS "zvec_shared: Using dynamic linking for libstdc++ and libgcc")
    endif()
endif()

# Include directories
target_include_directories(zvec_shared
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/generated>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Strip symbols in release builds to reduce library size
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(UNIX AND NOT APPLE)
        add_custom_command(TARGET zvec_shared POST_BUILD
            COMMAND ${CMAKE_STRIP} $<TARGET_FILE:zvec_shared>
            COMMENT "Stripping symbols from libzvec.so"
        )
    elseif(APPLE)
        add_custom_command(TARGET zvec_shared POST_BUILD
            COMMAND /usr/bin/strip -x $<TARGET_FILE:zvec_shared>
            COMMENT "Stripping symbols from libzvec.dylib"
        )
    endif()
endif()

# Installation
install(TARGETS zvec_shared
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

