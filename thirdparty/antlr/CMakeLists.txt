set(ANTLR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/antlr4)
set(ANTLR_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/antlr4.patch)
apply_patch_once("antlr4_fix" "${ANTLR_SRC_DIR}" "${ANTLR_PATCH}")
if(MSVC)
    # Replace /Zi with /Z7 for sccache compatibility (/Z7 embeds debug info in .obj, no .pdb).
    set(ANTLR_WINDOWS_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/antlr4.windows.patch)
    apply_patch_once("antlr4_windows_fix" "${ANTLR_SRC_DIR}" "${ANTLR_WINDOWS_PATCH}")
endif()

if(MSVC)
    set(WITH_STATIC_CRT ${ZVEC_USE_STATIC_CRT} CACHE BOOL "" FORCE)
endif()
add_subdirectory(antlr4/runtime/Cpp/)

target_include_directories(antlr4_static SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/antlr4/runtime/Cpp/runtime/src/"
)
mark_target_includes_system(antlr4_static)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(antlr4_static PRIVATE -Wno-unknown-pragmas -Wno-unqualified-std-cast-call)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(antlr4_static PRIVATE -Wno-unknown-pragmas -Wno-unqualified-std-cast-call -Wno-attributes -Wno-implicit-fallthrough)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_definitions(antlr4_static INTERFACE ANTLR4CPP_STATIC)
endif ()

add_library(antlr4 ALIAS antlr4_static)
