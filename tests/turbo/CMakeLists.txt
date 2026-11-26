include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(TEST_EXTRA_LIBS)
  if(CC_TARGET STREQUAL "turbo_rabitq_quantizer_test")
    if(NOT RABITQ_SUPPORTED)
      continue()
    endif()
    # Same registered centroid trainer as the dedicated RaBitQ index.
    list(APPEND TEST_EXTRA_LIBS core_knn_cluster rabitqlib)
  endif()
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_ailego core_framework core_metric core_quantizer zvec_turbo ${TEST_EXTRA_LIBS}
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core/ ${PROJECT_ROOT_DIR}/src/turbo/
    )
endforeach()
