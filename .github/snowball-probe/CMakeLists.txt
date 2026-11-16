cmake_minimum_required(VERSION 3.26)
project(snowball_probe C)

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
if(PROBE_MODE MATCHES "^(analyser|tokeniser|space)-Od$")
  set_property(SOURCE
    "${ZVEC_SOURCE_DIR}/thirdparty/snowball/snowball-3.1.1/compiler/${CMAKE_MATCH_1}.c"
    DIRECTORY "${ZVEC_SOURCE_DIR}/thirdparty/snowball"
    APPEND PROPERTY COMPILE_OPTIONS /Od)
endif()
