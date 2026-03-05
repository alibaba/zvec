
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(APPLE)
  set(APPLE_FRAMEWORK_LIBS
    -framework CoreFoundation
    -framework CoreGraphics
    -framework CoreData
    -framework CoreText
    -framework Security
    -framework Foundation
    -Wl,-U,_MallocExtension_ReleaseFreeMemory
    -Wl,-U,_ProfilerStart
    -Wl,-U,_ProfilerStop
    -Wl,-U,_RegisterThriftProtocol
  )
endif()

file(GLOB ALL_TEST_SRCS *_test.cc)
foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gmock(
    NAME ${CC_TARGET} STRICT
    LIBS zvec_common
    zvec_proto
    zvec_sqlengine
    zvec_ailego
    core_metric
    core_utility
    core_quantizer
    core_knn_hnsw core_knn_hnsw_sparse sparsehash
    core_knn_flat core_knn_flat_sparse core_knn_ivf
    core_mix_reducer
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    SRCS ${CC_SRCS}
    INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/db ${PROJECT_ROOT_DIR}/src/db/common
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  cc_test_suite(zvec_sqlengine ${CC_TARGET})
endforeach()
