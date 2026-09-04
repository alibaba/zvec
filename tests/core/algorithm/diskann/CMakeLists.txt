include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

# The full DiskAnn suite repeatedly builds 10k-vector indexes and is intended
# for desktop CI. Mobile CI runs a focused compatibility test that covers the
# portable I/O path, failure recovery, concurrency, and an end-to-end
# build/dump/load/search cycle.
if(ANDROID OR IOS)
  # Mobile CI runs the focused portable-I/O and end-to-end compatibility test.
  list(FILTER ALL_TEST_SRCS INCLUDE REGEX "diskann_mobile_compat_test\\.cc$")
else()
  if(WIN32)
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
         "/diskann_file_reader_test\\.cc$|/diskann_mobile_compat_test\\.cc$")
  else()
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
         "/diskann_file_reader_windows_test\\.cc$")
  endif()

  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
         "/diskann_file_reader_aio_test\\.cc$")
  endif()
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
