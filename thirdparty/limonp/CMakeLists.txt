set(limonp_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/limonp-v1.0.2")

if(NOT TARGET limonp)
  add_library(limonp INTERFACE)
  target_include_directories(limonp SYSTEM INTERFACE
    ${limonp_SOURCE_DIR}/include
  )
endif()

set(limonp_FOUND TRUE PARENT_SCOPE)
set(limonp_INCLUDE_DIR ${limonp_SOURCE_DIR}/include PARENT_SCOPE)
set(limonp_INCLUDE_DIRS ${limonp_SOURCE_DIR}/include PARENT_SCOPE)
