cmake_minimum_required(VERSION 3.26)
project(snowball_probe C)
message(STATUS "Compiler architecture: ${CMAKE_C_COMPILER_ARCHITECTURE_ID}")

get_filename_component(ZVEC_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(MSVC)
  set(CMAKE_C_FLAGS_RELEASE "/O1 /Ob1 /DNDEBUG")
  add_compile_options(/utf-8 /EHsc /Zc:preprocessor /Zc:__cplusplus /we4716)
  if(PROBE_MODE STREQUAL "Od")
    set(CMAKE_C_FLAGS_RELEASE "/Od /DNDEBUG")
  elseif(PROBE_MODE STREQUAL "O2")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
  endif()
endif()

add_subdirectory("${ZVEC_SOURCE_DIR}/thirdparty/snowball" snowball)
if(MSVC AND NOT PROBE_MODE STREQUAL "patched")
  get_target_property(_options snowball_compiler COMPILE_OPTIONS)
  list(REMOVE_ITEM _options /Od /Ob0)
  set_property(TARGET snowball_compiler PROPERTY COMPILE_OPTIONS "${_options}")
endif()
if(PROBE_MODE MATCHES "^(analyser|tokeniser|space)-Od$")
  set_property(SOURCE
    "${ZVEC_SOURCE_DIR}/thirdparty/snowball/snowball-3.1.1/compiler/${CMAKE_MATCH_1}.c"
    DIRECTORY "${ZVEC_SOURCE_DIR}/thirdparty/snowball"
    APPEND PROPERTY COMPILE_OPTIONS /Od)
endif()
if(PROBE_MODE STREQUAL "analyser-tokeniser-Od")
  foreach(_source analyser tokeniser)
    set_property(SOURCE
      "${ZVEC_SOURCE_DIR}/thirdparty/snowball/snowball-3.1.1/compiler/${_source}.c"
      DIRECTORY "${ZVEC_SOURCE_DIR}/thirdparty/snowball"
      APPEND PROPERTY COMPILE_OPTIONS /Od /Ob0)
  endforeach()
endif()

add_executable(initialization_probe
  initialization.c
  "${ZVEC_SOURCE_DIR}/thirdparty/snowball/snowball-3.1.1/compiler/tokeniser.c"
  "${ZVEC_SOURCE_DIR}/thirdparty/snowball/snowball-3.1.1/compiler/space.c")
target_include_directories(initialization_probe PRIVATE
  "${ZVEC_SOURCE_DIR}/thirdparty/snowball/snowball-3.1.1/compiler")
set_target_properties(initialization_probe PROPERTIES C_STANDARD 99)
if(MSVC)
  target_compile_definitions(initialization_probe PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(initialization_probe PRIVATE /FAs)
  if(PROBE_MODE STREQUAL "patched")
    target_compile_options(initialization_probe PRIVATE /Od /Ob0)
  endif()
endif()
