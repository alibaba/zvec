include(ExternalProject)

set(SNOWBALL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/snowball-3.1.1")
set(SNOWBALL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/snowball-codegen")
set(SNOWBALL_HOST_CC "" CACHE STRING
    "Optional host C compiler for building the Snowball code generator")
find_program(_SNOWBALL_MAKE NAMES make gmake REQUIRED)

# ---------------------------------------------------------------------------
# Parse modules.txt → UTF-8 algorithm list
# ---------------------------------------------------------------------------
set(_snowball_gen_srcs)
set(_snowball_gen_hdrs)
set(_snowball_make_targets)
file(STRINGS "${SNOWBALL_SOURCE_DIR}/libstemmer/modules.txt" _lines)
foreach(_line IN LISTS _lines)
  if(_line MATCHES "^#" OR _line MATCHES "^[ \t]*$")
    continue()
  endif()
  if(_line MATCHES "^([a-z_]+)[ \t]+([A-Z_0-9,]+)")
    set(_alg "${CMAKE_MATCH_1}")
    list(APPEND _snowball_gen_srcs
      "${SNOWBALL_BUILD_DIR}/src_c/stem_UTF_8_${_alg}.c")
    list(APPEND _snowball_gen_hdrs
      "${SNOWBALL_BUILD_DIR}/src_c/stem_UTF_8_${_alg}.h")
    list(APPEND _snowball_make_targets
      "src_c/stem_UTF_8_${_alg}.c")
  endif()
endforeach()

set(_snowball_make_args "CFLAGS=-O2")
if(NOT SNOWBALL_HOST_CC STREQUAL "")
  list(APPEND _snowball_make_args "CC=${SNOWBALL_HOST_CC}")
endif()

# ---------------------------------------------------------------------------
# Phase 1 (host): build snowball compiler & generate UTF-8 sources only
# ---------------------------------------------------------------------------
# Copy source tree into the build directory so the original stays clean.
# Request only the UTF-8 stemmer sources, the utf8 libstemmer entry point,
# and the utf8 modules header — no ISO-8859/KOI8 stemmers, no host .a.
# Each src_c/stem_UTF_8_*.c target implicitly builds the snowball compiler
# (host executable) as a dependency.
# By default make uses system `cc`; set SNOWBALL_HOST_CC to override when
# the environment CC points to a cross-compiler.
ExternalProject_Add(snowball_codegen
  DOWNLOAD_COMMAND  ${CMAKE_COMMAND} -E copy_directory
                    ${SNOWBALL_SOURCE_DIR} ${SNOWBALL_BUILD_DIR}
  SOURCE_DIR        ${SNOWBALL_BUILD_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ${_SNOWBALL_MAKE}
                    libstemmer/libstemmer_utf8.c
                    libstemmer/modules_utf8.h
                    ${_snowball_make_targets}
                    ${_snowball_make_args}
  BUILD_IN_SOURCE   TRUE
  INSTALL_COMMAND   ""
  BUILD_BYPRODUCTS
    ${SNOWBALL_BUILD_DIR}/runtime/api.c
    ${SNOWBALL_BUILD_DIR}/runtime/utilities.c
    ${SNOWBALL_BUILD_DIR}/libstemmer/libstemmer_utf8.c
    ${SNOWBALL_BUILD_DIR}/libstemmer/modules_utf8.h
    ${_snowball_gen_srcs}
    ${_snowball_gen_hdrs}
)

# ---------------------------------------------------------------------------
# Phase 2 (target): compile generated sources with the project toolchain
# ---------------------------------------------------------------------------
set(_snowball_target_srcs
  ${SNOWBALL_BUILD_DIR}/runtime/api.c
  ${SNOWBALL_BUILD_DIR}/runtime/utilities.c
  ${SNOWBALL_BUILD_DIR}/libstemmer/libstemmer_utf8.c
  ${_snowball_gen_srcs}
)

set_source_files_properties(${_snowball_target_srcs}
  PROPERTIES GENERATED TRUE)

if(NOT TARGET snowball)
  add_library(snowball STATIC ${_snowball_target_srcs})
  add_dependencies(snowball snowball_codegen)
  # Public include points to the SOURCE directory — libstemmer.h exists at
  # configure time and does not depend on the codegen step.
  target_include_directories(snowball SYSTEM PUBLIC
    ${SNOWBALL_SOURCE_DIR}/include
  )
  # Private includes for generated headers (modules_utf8.h, stem_*.h).
  target_include_directories(snowball PRIVATE
    ${SNOWBALL_BUILD_DIR}
    ${SNOWBALL_BUILD_DIR}/libstemmer
    ${SNOWBALL_BUILD_DIR}/src_c
  )
  set_target_properties(snowball PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
  )
endif()

set(snowball_FOUND TRUE PARENT_SCOPE)
