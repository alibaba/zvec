# Optional benchmark-only hook, passed through pip's CMAKE_PROJECT_INCLUDE.
# It adds a compile-only view of the existing unittest dependencies, leaving
# normal test registration and the unittest target unchanged.
if(NOT PROJECT_NAME STREQUAL "zvec" OR
   NOT PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  return()
endif()
function(zvec_ci_add_test_build_target)
  get_target_property(_test_targets unittest MANUALLY_ADDED_DEPENDENCIES)
  if(NOT _test_targets)
    message(FATAL_ERROR "No unittest build dependencies were registered")
  endif()
  add_custom_target(zvec_ci_test_build DEPENDS ${_test_targets})
endfunction()
cmake_language(DEFER CALL zvec_ci_add_test_build_target)
