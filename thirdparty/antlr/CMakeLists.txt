set(ANTLR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/antlr4)
set(ANTLR_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/antlr4.patch)
apply_patch_once("antlr4_fix" "${ANTLR_SRC_DIR}" "${ANTLR_PATCH}")

add_subdirectory(antlr4/runtime/Cpp/)

add_library(antlr4 UNKNOWN IMPORTED GLOBAL)
set_target_properties(
  antlr4 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/antlr4/runtime/Cpp/runtime/src/"
  IMPORTED_LOCATION "${EXTERNAL_LIB_DIR}/libantlr4-runtime.a"
  )
add_dependencies(antlr4 antlr4_static)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(antlr4_static PRIVATE -Wno-unknown-pragmas -Wno-unqualified-std-cast-call)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(antlr4_static PRIVATE -Wno-unknown-pragmas -Wno-unqualified-std-cast-call -Wno-attributes -Wno-implicit-fallthrough)
endif()