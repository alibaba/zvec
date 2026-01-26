
include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/option.cmake)

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

file(GLOB_RECURSE ALL_TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_test.cc)
foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gmock(
    NAME ${CC_TARGET} STRICT
    LIBS zvec_db
    zvec_proto
    core_metric_static
    core_utility_static
    core_quantizer_static
    core_knn_hnsw core_knn_hnsw_sparse sparsehash
    core_knn_flat core_knn_flat_sparse core_knn_ivf
    core_mix_reducer
    Arrow::arrow_dataset
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    SRCS ${CC_SRCS} utils/utils.cc
    INCS . .. ../../src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  cc_test_suite(zvec_index ${CC_TARGET})
endforeach()
