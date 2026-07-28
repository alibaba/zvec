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
# Build ALL-IN-ONE C++ Shared Libraries
# =============================================================================
# Merges zvec internal static libraries into shared libraries while preserving
# C++ symbols for direct C++ linking. Windows uses component-owned object
# variants so DLL export definitions never leak into static or plugin targets.

include(GNUInstallDirs)
find_package(Threads REQUIRED)

function(zvec_add_all_in_one_shared TARGET_NAME OUTPUT_NAME)
    cmake_parse_arguments(ZVEC_ALLIN "" "" "LIBS;EXPORT_COMPONENTS" ${ARGN})
    if(NOT ZVEC_ALLIN_LIBS)
        message(FATAL_ERROR "zvec_add_all_in_one_shared requires LIBS")
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

    install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endfunction()

if(BUILD_ZVEC_AILEGO_SHARED)
    zvec_add_all_in_one_shared(zvec_ailego_shared zvec_ailego
        LIBS
            zvec_ailego
        EXPORT_COMPONENTS
            zvec_ailego
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
            zvec_ailego
    )
endif()
