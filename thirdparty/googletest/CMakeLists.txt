if(NOT MSVC)
add_compile_options(-Wno-deprecated-copy)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-maybe-uninitialized)
endif()

if (MSVC)
    if(NOT ZVEC_USE_STATIC_CRT)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    else()
        set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
    endif()
endif ()


set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})
add_subdirectory(googletest-1.10.0)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

# googletest hardcodes /Zi via COMPILE_FLAGS (old-style string property) in internal_utils.cmake.
# Replace -Zi with /Z7 for sccache compatibility (/Z7 embeds debug info in .obj, no .pdb).
# Also clear COMPILE_PDB_NAME to prevent CMake from adding /Fd<path> — sccache would try
# to collect the nonexistent .pdb and fail.
if(MSVC)
    foreach(_gt_target gtest gtest_main gmock gmock_main)
        get_target_property(_gt_flags ${_gt_target} COMPILE_FLAGS)
        if(_gt_flags)
            string(REPLACE "-Zi" "/Z7" _gt_flags "${_gt_flags}")
            set_target_properties(${_gt_target} PROPERTIES COMPILE_FLAGS "${_gt_flags}")
        endif()
        set_target_properties(${_gt_target} PROPERTIES COMPILE_PDB_NAME "")
    endforeach()
endif()

mark_target_includes_system(gtest gtest_main gmock gmock_main)

get_target_property(GTEST_INCLUDE_DIRS gtest INTERFACE_INCLUDE_DIRECTORIES)

set(GTEST_FOUND TRUE PARENT_SCOPE)
set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS} PARENT_SCOPE)
set(GTEST_INCLUDE_DIR ${GTEST_INCLUDE_DIRS} PARENT_SCOPE)
set(GTEST_LIBRARIES $<TARGET_FILE:gtest> PARENT_SCOPE)
set(GTEST_LIBRARY ${GTEST_LIBRARIES} PARENT_SCOPE)
set(GTEST_MAIN_LIBRARIES $<TARGET_FILE:gtest_main> PARENT_SCOPE)
set(GTEST_MAIN_LIBRARY ${GTEST_MAIN_LIBRARIES} PARENT_SCOPE)
set(GTEST_BOTH_LIBRARIES $<TARGET_FILE:gtest> $<TARGET_FILE:gtest_main> PARENT_SCOPE)
