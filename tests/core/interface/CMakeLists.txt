include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

set(ZVEC_TEST_CORE_INTERFACE_LIBS
    zvec_ailego
    core_framework
    core_metric
    core_interface
    core_knn_flat
    core_utility
    core_quantizer
    sparsehash
    core_knn_hnsw
    core_mix_reducer
    core_knn_flat_sparse
    core_knn_hnsw_sparse
    core_knn_ivf
    core_knn_hnsw_rabitq
)

if(ZVEC_ENABLE_OMEGA)
  list(APPEND ZVEC_TEST_CORE_INTERFACE_LIBS core_knn_omega omega)
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS ${ZVEC_TEST_CORE_INTERFACE_LIBS}
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
           ${PROJECT_ROOT_DIR}/thirdparty/omega/OMEGALib/include
    )
endforeach()
