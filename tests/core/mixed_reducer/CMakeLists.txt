include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

set(MERGED_HOLDER_TEST_LIBS core_mix_reducer core_quantizer)
if(DISKANN_SUPPORTED)
  list(APPEND MERGED_HOLDER_TEST_LIBS
       core_utility core_metric core_knn_cluster core_knn_diskann)
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_ailego core_framework ${MERGED_HOLDER_TEST_LIBS}
      SRCS ${CC_SRCS}
      INCS ${PROJECT_ROOT_DIR}/src/core
    )
endforeach()
