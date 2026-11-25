include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # A dylib can interpose calls bound to libSystem in core_utility.dylib.
  cc_library(
      NAME buffer_storage_directory_pwrite
      SHARED EXCLUDE
      SRCS buffer_storage_directory_pwrite.cc
    )
endif()

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(TEST_LIBS zvec_ailego core_framework core_utility core_metric core_quantizer core_knn_hnsw core_knn_flat)
  if(TARGET buffer_storage_directory_pwrite AND
     CC_TARGET STREQUAL "buffer_storage_directory_split_test")
    list(APPEND TEST_LIBS buffer_storage_directory_pwrite)
  endif()
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS ${TEST_LIBS}
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm/hnsw
    )
endforeach()
