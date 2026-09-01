include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

if(WIN32)
  list(FILTER ALL_TEST_SRCS EXCLUDE REGEX "/diskann_file_reader_test\\.cc$")
else()
  list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
       "/diskann_file_reader_windows_test\\.cc$")
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
       "/diskann_file_reader_aio_test\\.cc$")
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_ailego core_framework core_utility core_metric core_quantizer core_knn_cluster core_knn_diskann
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm/diskann
    )
endforeach()
