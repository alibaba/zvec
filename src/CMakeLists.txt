include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# Add repository
cc_directory(ailego)
cc_directory(turbo)
cc_directory(core)
cc_directory(db)
cc_directory(binding)

# =============================================================================
# Build ALL-IN-ONE C++ Shared Libraries
# =============================================================================
# Merges zvec internal static libraries into shared libraries while preserving
# C++ symbols for direct C++ linking. Windows uses component-owned object
# variants so DLL export definitions never leak into static or plugin targets.

include(GNUInstallDirs)
find_package(Threads REQUIRED)

# =============================================================================
# Public header installation
# =============================================================================
# src/include is a *build* directory, not the published API surface: it also
# holds the engine internals (zvec/core/framework/**, most of zvec/ailego/**).
# Installing it wholesale ships 95 headers where the stable API only needs 18,
# inviting users to depend on an unstable surface.  The stable closure is
# therefore installed from an explicit manifest -- the whitelist approach LevelDB
# uses, and the opposite of Arrow's ARROW_INSTALL_ALL_HEADERS -- while the
# advanced engine headers stay behind ZVEC_INSTALL_ENGINE_HEADERS.
#
# cmake/zvec_public_headers.cmake mirrors the real include graph and is kept
# honest by scripts/check_public_headers.py (rules 6 and 7).
include(${PROJECT_ROOT_DIR}/cmake/zvec_public_headers.cmake)

set(ZVEC_INCLUDE_ROOT ${PROJECT_ROOT_DIR}/src/include)
foreach(ZVEC_PUBLIC_HEADER ${ZVEC_STABLE_PUBLIC_HEADERS})
  if(NOT EXISTS ${ZVEC_INCLUDE_ROOT}/${ZVEC_PUBLIC_HEADER})
    message(FATAL_ERROR
        "ZVEC_STABLE_PUBLIC_HEADERS lists a missing header: ${ZVEC_PUBLIC_HEADER}")
  endif()
  get_filename_component(ZVEC_PUBLIC_HEADER_DIR ${ZVEC_PUBLIC_HEADER} DIRECTORY)
  install(
      FILES ${ZVEC_INCLUDE_ROOT}/${ZVEC_PUBLIC_HEADER}
      DESTINATION "${CMAKE_INSTALL_INCDIR}/${ZVEC_PUBLIC_HEADER_DIR}"
  )
endforeach()

install(
    FILES ${PROJECT_ROOT_DIR}/src/include/zvec/README.md
    DESTINATION "${CMAKE_INSTALL_INCDIR}/zvec"
)

# Advanced engine headers.  The manifest above already installs the few zvec/core
# and zvec/ailego headers that zvec/db/** transitively needs, so this only *adds*
# to the package.  It must stay all-or-nothing: zvec/core/** reaches 31 of the 35
# zvec/ailego headers, so publishing core without ailego would ship headers that
# do not compile.
if(ZVEC_INSTALL_ENGINE_HEADERS)
  install(
      DIRECTORY ${ZVEC_INCLUDE_ROOT}/zvec/core
                ${ZVEC_INCLUDE_ROOT}/zvec/ailego
                ${ZVEC_INCLUDE_ROOT}/zvec/turbo
      DESTINATION "${CMAKE_INSTALL_INCDIR}/zvec"
      # Same suffixes cc_library(PACKED) used to install, so turning the
      # switch on yields exactly the header set the SDK shipped before.
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hxx"
  )
endif()

function(zvec_add_all_in_one_shared TARGET_NAME OUTPUT_NAME)
    cmake_parse_arguments(ZVEC_ALLIN "SKIP_INSTALL" "EXPORT_SYMBOLS"
        "LIBS;EXPORT_COMPONENTS" ${ARGN})
    if(NOT ZVEC_ALLIN_LIBS)
        message(FATAL_ERROR "zvec_add_all_in_one_shared requires LIBS")
    endif()
    # Base path (without .txt / .lds) of the link-time export whitelist.
    if(NOT ZVEC_ALLIN_EXPORT_SYMBOLS)
        set(ZVEC_ALLIN_EXPORT_SYMBOLS
            ${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_cpp)
    endif()

    foreach(ZVEC_ALLIN_LIB ${ZVEC_ALLIN_LIBS})
        if(NOT TARGET ${ZVEC_ALLIN_LIB})
            message(FATAL_ERROR "Target ${ZVEC_ALLIN_LIB} is required by ${TARGET_NAME}")
        endif()
    endforeach()
    foreach(ZVEC_ALLIN_EXPORT_COMPONENT ${ZVEC_ALLIN_EXPORT_COMPONENTS})
        if(NOT ZVEC_ALLIN_EXPORT_COMPONENT IN_LIST ZVEC_ALLIN_LIBS)
            message(FATAL_ERROR
                "Export component ${ZVEC_ALLIN_EXPORT_COMPONENT} is not part of ${TARGET_NAME}")
        endif()
    endforeach()

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_stub.cc
        "// Auto-generated stub for ${TARGET_NAME}\n"
    )

    add_library(${TARGET_NAME} SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_stub.cc
    )

    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME "${OUTPUT_NAME}"
        POSITION_INDEPENDENT_CODE ON
    )
    if(WIN32)
        set_target_properties(${TARGET_NAME} PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS OFF
        )
    endif()
    if(WIN32)
        set_target_properties(${TARGET_NAME} PROPERTIES
            ARCHIVE_OUTPUT_NAME "${TARGET_NAME}"
        )
    endif()
    target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

    if(WIN32)
        foreach(ZVEC_ALLIN_LIB ${ZVEC_ALLIN_LIBS})
            if(ZVEC_ALLIN_LIB IN_LIST ZVEC_ALLIN_EXPORT_COMPONENTS)
                set(ZVEC_ALLIN_OBJECT_TARGET ${ZVEC_ALLIN_LIB}_export_objects)
            else()
                set(ZVEC_ALLIN_OBJECT_TARGET ${ZVEC_ALLIN_LIB}_objects)
            endif()
            if(NOT TARGET ${ZVEC_ALLIN_OBJECT_TARGET})
                message(FATAL_ERROR
                    "Object target ${ZVEC_ALLIN_OBJECT_TARGET} is required by ${TARGET_NAME}")
            endif()
            target_sources(${TARGET_NAME} PRIVATE
                $<TARGET_OBJECTS:${ZVEC_ALLIN_OBJECT_TARGET}>
            )
        endforeach()

        set(ZVEC_ALLIN_LINK_LIBS)
        foreach(ZVEC_ALLIN_LIB ${ZVEC_ALLIN_LIBS})
            get_target_property(ZVEC_ALLIN_LIB_DEPS ${ZVEC_ALLIN_LIB} LINK_LIBRARIES)
            if(NOT ZVEC_ALLIN_LIB_DEPS)
                continue()
            endif()
            foreach(ZVEC_ALLIN_LIB_DEP ${ZVEC_ALLIN_LIB_DEPS})
                if(NOT ZVEC_ALLIN_LIB_DEP IN_LIST ZVEC_ALLIN_LIBS)
                    list(APPEND ZVEC_ALLIN_LINK_LIBS ${ZVEC_ALLIN_LIB_DEP})
                endif()
            endforeach()
        endforeach()
        list(APPEND ZVEC_ALLIN_LINK_LIBS Threads::Threads)
        list(REMOVE_DUPLICATES ZVEC_ALLIN_LINK_LIBS)
        target_link_libraries(${TARGET_NAME}
            PRIVATE
                ${ZVEC_ALLIN_LINK_LIBS}
        )
    elseif(APPLE)
        foreach(ZVEC_ALLIN_LIB ${ZVEC_ALLIN_LIBS})
            list(APPEND ZVEC_ALLIN_WA_OPTIONS
                -Wl,-force_load,$<TARGET_FILE:${ZVEC_ALLIN_LIB}>
            )
        endforeach()

        target_link_options(${TARGET_NAME} PRIVATE ${ZVEC_ALLIN_WA_OPTIONS})
        target_link_libraries(${TARGET_NAME}
            PRIVATE
                ${ZVEC_ALLIN_LIBS}
                Threads::Threads
                ${CMAKE_DL_LIBS}
        )
    else()
        target_link_libraries(${TARGET_NAME}
            PRIVATE
                -Wl,--whole-archive
                    ${ZVEC_ALLIN_LIBS}
                -Wl,--no-whole-archive
                Threads::Threads
                ${CMAKE_DL_LIBS}
        )

        # Try static linking of libstdc++ and libgcc for better portability.
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
            target_link_options(${TARGET_NAME} PRIVATE -static-libstdc++ -static-libgcc)
            message(STATUS "${TARGET_NAME}: Using static libstdc++ and libgcc")
        else()
            message(STATUS "${TARGET_NAME}: Using dynamic linking for libstdc++ and libgcc")
        endif()
    endif()

    # Restrict the exported symbols to zvec's own public C++ API.  All
    # third-party symbols (arrow, rocksdb, parquet, antlr4) are hidden, and
    # unreferenced code is dead-stripped to reduce the shared library size.
    # The public C++ headers never expose third-party types, so hiding them is
    # safe for external C++ consumers.
    #
    # Two granularities, selected per target through EXPORT_SYMBOLS:
    #   exported_symbols_cpp.*  namespace-wide ("__ZN4zvec*").  Used by the
    #                           standalone engine libraries, whose whole
    #                           zvec::core / zvec::ailego / zvec::turbo surface
    #                           is their API.
    #   exported_symbols_zvec.* generated from the published headers, so libzvec
    #                           exports exactly what a consumer can name --
    #                           measured 10784 -> 2602 symbols.  Regenerate with
    #                           scripts/generate_export_whitelist.py --write.
    #
    # Note the lists are only applied to non-Debug builds: Debug keeps the full
    # symbol table so that in-tree debugging and crash symbolication are not
    # affected.  A consequence is that Debug exports a superset of Release.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(APPLE)
            set(_ZVEC_ALLIN_EXPORTS ${ZVEC_ALLIN_EXPORT_SYMBOLS}.txt)
            target_link_options(${TARGET_NAME} PRIVATE
                "-Wl,-exported_symbols_list,${_ZVEC_ALLIN_EXPORTS}"
                "-Wl,-dead_strip")
            set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
                LINK_DEPENDS ${_ZVEC_ALLIN_EXPORTS})
        elseif(UNIX)
            set(_ZVEC_ALLIN_EXPORTS ${ZVEC_ALLIN_EXPORT_SYMBOLS}.lds)
            target_link_options(${TARGET_NAME} PRIVATE
                "-Wl,--version-script,${_ZVEC_ALLIN_EXPORTS}"
                "-Wl,--gc-sections")
            set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
                LINK_DEPENDS ${_ZVEC_ALLIN_EXPORTS})
        endif()
    endif()

    target_include_directories(${TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/generated>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )

    set(ZVEC_ALLIN_USE_DEFS)
    foreach(ZVEC_ALLIN_LIB ${ZVEC_ALLIN_EXPORT_COMPONENTS})
        if(ZVEC_ALLIN_LIB STREQUAL zvec)
            list(APPEND ZVEC_ALLIN_USE_DEFS ZVEC_DB_USE_SHARED)
        elseif(ZVEC_ALLIN_LIB STREQUAL zvec_core)
            list(APPEND ZVEC_ALLIN_USE_DEFS ZVEC_CORE_USE_SHARED)
        elseif(ZVEC_ALLIN_LIB STREQUAL zvec_ailego)
            list(APPEND ZVEC_ALLIN_USE_DEFS ZVEC_AILEGO_USE_SHARED)
        elseif(ZVEC_ALLIN_LIB STREQUAL zvec_turbo)
            list(APPEND ZVEC_ALLIN_USE_DEFS ZVEC_TURBO_USE_SHARED)
        endif()
    endforeach()
    if(ZVEC_ALLIN_USE_DEFS)
        list(REMOVE_DUPLICATES ZVEC_ALLIN_USE_DEFS)
        target_compile_definitions(${TARGET_NAME} INTERFACE ${ZVEC_ALLIN_USE_DEFS})
    endif()

    # Strip symbols in release builds to reduce library size.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(UNIX AND NOT APPLE)
            # add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            #     COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${TARGET_NAME}>
            #     COMMENT "Stripping symbols from $<TARGET_FILE_NAME:${TARGET_NAME}>"
            # )
        elseif(APPLE)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND /usr/bin/strip -x $<TARGET_FILE:${TARGET_NAME}>
                COMMENT "Stripping symbols from $<TARGET_FILE_NAME:${TARGET_NAME}>"
            )
        endif()
    endif()

    if(NOT ZVEC_ALLIN_SKIP_INSTALL)
        install(TARGETS ${TARGET_NAME}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endfunction()

# Mobile public C++ API.  These build-tree targets keep the application and
# zvec in one C++ runtime when the NDK/iOS toolchain uses a static libc++.
# Whole-archive is required because module registration is performed by static
# initializers that otherwise have no referenced symbol at link time.
function(zvec_add_mobile_static_sdk TARGET_NAME)
    cmake_parse_arguments(ZVEC_STATIC_SDK "" "" "LIBS" ${ARGN})
    if(NOT ZVEC_STATIC_SDK_LIBS)
        message(FATAL_ERROR "zvec_add_mobile_static_sdk requires LIBS")
    endif()

    foreach(ZVEC_STATIC_SDK_LIB ${ZVEC_STATIC_SDK_LIBS})
        if(NOT TARGET ${ZVEC_STATIC_SDK_LIB})
            message(FATAL_ERROR
                "Target ${ZVEC_STATIC_SDK_LIB} is required by ${TARGET_NAME}")
        endif()
    endforeach()

    add_library(${TARGET_NAME} INTERFACE)
    target_compile_features(${TARGET_NAME} INTERFACE cxx_std_17)
    target_include_directories(${TARGET_NAME}
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/generated>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
    )

    if(IOS)
        set(ZVEC_STATIC_SDK_LINK_OPTIONS)
        foreach(ZVEC_STATIC_SDK_LIB ${ZVEC_STATIC_SDK_LIBS})
            list(APPEND ZVEC_STATIC_SDK_LINK_OPTIONS
                -Wl,-force_load,$<TARGET_FILE:${ZVEC_STATIC_SDK_LIB}>
            )
        endforeach()
        target_link_options(${TARGET_NAME}
            INTERFACE ${ZVEC_STATIC_SDK_LINK_OPTIONS}
        )
        target_link_libraries(${TARGET_NAME}
            INTERFACE
                ${ZVEC_STATIC_SDK_LIBS}
                Threads::Threads
                ${CMAKE_DL_LIBS}
                # Arrow's iOS timezone implementation uses CoreFoundation.
                "-framework CoreFoundation"
        )
    else()
        # Keep whole-archive scoped to the SDK archives themselves.  Putting
        # these flags in target_link_libraries() also encloses transitive
        # dependencies inserted by CMake, which forces both Arrow's bundled
        # utf8proc and zvec's standalone utf8proc into the executable.
        set(ZVEC_STATIC_SDK_LINK_OPTIONS)
        foreach(ZVEC_STATIC_SDK_LIB ${ZVEC_STATIC_SDK_LIBS})
            list(APPEND ZVEC_STATIC_SDK_LINK_OPTIONS
                -Wl,--whole-archive,$<TARGET_FILE:${ZVEC_STATIC_SDK_LIB}>,--no-whole-archive
            )
        endforeach()
        target_link_options(${TARGET_NAME}
            INTERFACE ${ZVEC_STATIC_SDK_LINK_OPTIONS}
        )
        target_link_libraries(${TARGET_NAME}
            INTERFACE
                ${ZVEC_STATIC_SDK_LIBS}
                Threads::Threads
                ${CMAKE_DL_LIBS}
        )
        if(ANDROID)
            target_link_libraries(${TARGET_NAME} INTERFACE log)
        endif()
    endif()
endfunction()

if(ANDROID OR IOS)
    zvec_add_mobile_static_sdk(zvec_static_sdk
        LIBS zvec zvec_core zvec_ailego zvec_turbo
    )
    zvec_add_mobile_static_sdk(zvec_core_static_sdk
        LIBS zvec_core zvec_ailego zvec_turbo
    )
    zvec_add_mobile_static_sdk(zvec_ailego_static_sdk
        LIBS zvec_ailego
    )

    add_library(zvec::static ALIAS zvec_static_sdk)
    add_library(zvec::core_static ALIAS zvec_core_static_sdk)
    add_library(zvec::ailego_static ALIAS zvec_ailego_static_sdk)
endif()

# The standalone engine libraries are optional in the package.  They used to be
# tied to ZVEC_INSTALL_ENGINE_HEADERS because publishing libzvec_core without
# zvec/core/** ships a library that cannot be called -- but that coupling only
# ever protected one direction.  With the generated export whitelist libzvec
# exports every symbol the published headers declare, engine ones included, so
# the headers can ship while the standalone libraries do not: a consumer links
# -lzvec and gets the same API.  That is the default, and it is what makes the
# released package two shared libraries wide instead of four.
#
# Both are still *built* either way, since libzvec and libzvec_c_api
# whole-archive them and in-tree tests, examples and subproject-integration link
# the targets directly.  The static archives keep installing through
# cc_library(PACKED) for static and mobile-SDK consumers.
set(ZVEC_ENGINE_SHARED_INSTALL_OPTIONS)
if(NOT ZVEC_INSTALL_ENGINE_SHARED_LIBRARIES)
  set(ZVEC_ENGINE_SHARED_INSTALL_OPTIONS SKIP_INSTALL)
endif()

if(BUILD_ZVEC_AILEGO_SHARED)
    zvec_add_all_in_one_shared(zvec_ailego_shared zvec_ailego
        LIBS
            zvec_ailego
        EXPORT_COMPONENTS
            zvec_ailego
        ${ZVEC_ENGINE_SHARED_INSTALL_OPTIONS}
    )
endif()

if(BUILD_ZVEC_CORE_SHARED)
    zvec_add_all_in_one_shared(zvec_core_shared zvec_core
        LIBS
            zvec_core
            zvec_ailego
            zvec_turbo
        EXPORT_COMPONENTS
            zvec_core
            zvec_ailego
            zvec_turbo
        ${ZVEC_ENGINE_SHARED_INSTALL_OPTIONS}
    )
endif()

if(BUILD_ZVEC_SHARED)
    zvec_add_all_in_one_shared(zvec_shared zvec
        LIBS
            zvec
            zvec_core
            zvec_ailego
            zvec_turbo
        EXPORT_COMPONENTS
            zvec
            zvec_core
            zvec_ailego
            zvec_turbo
        EXPORT_SYMBOLS
            ${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_zvec
    )
endif()
