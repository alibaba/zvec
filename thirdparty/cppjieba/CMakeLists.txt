set(cppjieba_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cppjieba-5.6.7")
set(cppjieba_MEMORY_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/cppjieba.memory.patch")
set(cppjieba_PATCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/cppjieba-patch")
set(cppjieba_BINARY_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/cppjieba/include")

file(GLOB_RECURSE cppjieba_HEADERS CONFIGURE_DEPENDS
    RELATIVE "${cppjieba_SOURCE_DIR}/include"
    "${cppjieba_SOURCE_DIR}/include/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${cppjieba_MEMORY_PATCH}")
foreach(cppjieba_HEADER IN LISTS cppjieba_HEADERS)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${cppjieba_SOURCE_DIR}/include/${cppjieba_HEADER}")
endforeach()

# Patch a fresh build-tree copy so quoted includes also use the patched headers.
file(REMOVE_RECURSE "${cppjieba_PATCH_DIR}")
file(COPY "${cppjieba_SOURCE_DIR}/include" DESTINATION "${cppjieba_PATCH_DIR}")
# Keep in-tree builds from discovering the source repository above the copy.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            git -C "${cppjieba_PATCH_DIR}" apply
            --ignore-space-change --ignore-whitespace "${cppjieba_MEMORY_PATCH}"
    RESULT_VARIABLE cppjieba_PATCH_RESULT
    ERROR_VARIABLE cppjieba_PATCH_ERROR)
if(NOT cppjieba_PATCH_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to apply cppjieba memory patch: ${cppjieba_PATCH_ERROR}")
endif()

# Preserve output timestamps when a CMake rerun produces identical headers.
foreach(cppjieba_HEADER IN LISTS cppjieba_HEADERS)
  get_filename_component(cppjieba_HEADER_DIR "${cppjieba_HEADER}" DIRECTORY)
  file(MAKE_DIRECTORY "${cppjieba_BINARY_INCLUDE_DIR}/${cppjieba_HEADER_DIR}")
  file(COPY_FILE
      "${cppjieba_PATCH_DIR}/include/${cppjieba_HEADER}"
      "${cppjieba_BINARY_INCLUDE_DIR}/${cppjieba_HEADER}" ONLY_IF_DIFFERENT)
endforeach()
file(REMOVE_RECURSE "${cppjieba_PATCH_DIR}")

if(NOT TARGET cppjieba)
  add_library(cppjieba INTERFACE)
  target_include_directories(cppjieba SYSTEM INTERFACE
    ${cppjieba_BINARY_INCLUDE_DIR}
  )
  target_link_libraries(cppjieba INTERFACE limonp)
endif()

set(cppjieba_FOUND TRUE PARENT_SCOPE)
set(cppjieba_INCLUDE_DIR ${cppjieba_BINARY_INCLUDE_DIR} PARENT_SCOPE)
set(cppjieba_INCLUDE_DIRS
    ${cppjieba_BINARY_INCLUDE_DIR}
    ${limonp_INCLUDE_DIR}
    PARENT_SCOPE)
set(cppjieba_DICT_DIR ${cppjieba_SOURCE_DIR}/dict PARENT_SCOPE)
