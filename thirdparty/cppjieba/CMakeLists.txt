set(cppjieba_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cppjieba-5.6.7")

if(NOT TARGET cppjieba)
  add_library(cppjieba INTERFACE)
  target_include_directories(cppjieba SYSTEM INTERFACE
    ${cppjieba_SOURCE_DIR}/include
  )
  target_link_libraries(cppjieba INTERFACE limonp)
endif()

set(cppjieba_FOUND TRUE PARENT_SCOPE)
set(cppjieba_INCLUDE_DIR ${cppjieba_SOURCE_DIR}/include PARENT_SCOPE)
set(cppjieba_INCLUDE_DIRS
    ${cppjieba_SOURCE_DIR}/include
    ${limonp_INCLUDE_DIR}
    PARENT_SCOPE)
set(cppjieba_DICT_DIR ${cppjieba_SOURCE_DIR}/dict PARENT_SCOPE)
