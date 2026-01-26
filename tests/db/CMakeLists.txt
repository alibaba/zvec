include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/option.cmake)

cc_directory(common)
cc_directory(sqlengine)
cc_directories(index)

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
    LIBS zvec_db
    zvec_proto
    core_knn_flat
    core_knn_flat_sparse
    core_knn_hnsw
    core_knn_hnsw_sparse
    core_knn_ivf
    core_mix_reducer
    core_metric
    core_utility
    core_quantizer
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    SRCS ${CC_SRCS} index/utils/utils.cc
    INCS . .. ../../src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  cc_test_suite(zvec_db ${CC_TARGET})
endforeach()
