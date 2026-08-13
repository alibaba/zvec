set(SNOWBALL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/snowball-3.1.1")
set(SNOWBALL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/snowball-codegen")

# ---------------------------------------------------------------------------
# Build the Snowball compiler as a native tool. When cross-compiling, it must
# be built directly with a host compiler instead of the target toolchain.
# ---------------------------------------------------------------------------
file(GLOB _snowball_compiler_srcs CONFIGURE_DEPENDS
  "${SNOWBALL_SOURCE_DIR}/compiler/*.c"
)

if(CMAKE_CROSSCOMPILING)
  if(NOT SNOWBALL_HOST_CC)
    find_program(SNOWBALL_HOST_CC
      NAMES cc gcc clang
      DOC "Host C compiler used to build the Snowball code generator"
    )
  endif()
  if(NOT SNOWBALL_HOST_CC)
    message(FATAL_ERROR
      "Cross-compiling Snowball requires a host C compiler. "
      "Set SNOWBALL_HOST_CC to a host cc/gcc/clang executable.")
  endif()

  set(_snowball_compiler_exe
      "${SNOWBALL_BUILD_DIR}/host/snowball_compiler")
  if(CMAKE_HOST_WIN32)
    string(APPEND _snowball_compiler_exe ".exe")
  endif()

  add_custom_command(
    OUTPUT "${_snowball_compiler_exe}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${SNOWBALL_BUILD_DIR}/host"
    COMMAND "${SNOWBALL_HOST_CC}"
            -O2
            -I "${SNOWBALL_SOURCE_DIR}/compiler"
            -o "${_snowball_compiler_exe}"
            ${_snowball_compiler_srcs}
    DEPENDS ${_snowball_compiler_srcs}
            "${SNOWBALL_SOURCE_DIR}/compiler/header.h"
            "${SNOWBALL_SOURCE_DIR}/compiler/tokens.h"
    COMMENT "Building Snowball compiler with ${SNOWBALL_HOST_CC}"
    VERBATIM
  )

  set(_snowball_compiler "${_snowball_compiler_exe}")
  set(_snowball_compiler_dependency "${_snowball_compiler_exe}")
else()
  if(NOT TARGET snowball_compiler)
    add_executable(snowball_compiler EXCLUDE_FROM_ALL
      ${_snowball_compiler_srcs}
    )
    target_include_directories(snowball_compiler PRIVATE
      "${SNOWBALL_SOURCE_DIR}/compiler"
    )
    if(MSVC)
      target_compile_definitions(snowball_compiler PRIVATE
        _CRT_SECURE_NO_WARNINGS
      )
    endif()
    set_target_properties(snowball_compiler PROPERTIES C_STANDARD 99)
  endif()

  set(_snowball_compiler "$<TARGET_FILE:snowball_compiler>")
  set(_snowball_compiler_dependency snowball_compiler)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/snowball_codegen.cmake")
snowball_generate_utf8(
  SOURCE_DIR "${SNOWBALL_SOURCE_DIR}"
  BUILD_DIR "${SNOWBALL_BUILD_DIR}"
  COMPILER "${_snowball_compiler}"
  COMPILER_DEPENDENCY "${_snowball_compiler_dependency}"
)

set_source_files_properties(${SNOWBALL_GENERATED_SOURCES}
  PROPERTIES GENERATED TRUE)

if(NOT TARGET snowball)
  add_library(snowball STATIC ${SNOWBALL_GENERATED_SOURCES})
  add_dependencies(snowball ${SNOWBALL_CODEGEN_TARGET})
  # Public include points to the SOURCE directory — libstemmer.h exists at
  # configure time and does not depend on the codegen step.
  target_include_directories(snowball SYSTEM PUBLIC
    "${SNOWBALL_SOURCE_DIR}/include"
  )
  # Private includes for generated headers (modules_utf8.h, stem_*.h).
  target_include_directories(snowball PRIVATE
    "${SNOWBALL_BUILD_DIR}"
    "${SNOWBALL_BUILD_DIR}/libstemmer"
    "${SNOWBALL_BUILD_DIR}/src_c"
  )
  set_target_properties(snowball PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
  )
endif()

set(snowball_FOUND TRUE PARENT_SCOPE)
