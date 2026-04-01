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


# Common libraries
set(CRASH_RECOVERY_COMMON_LIBS
    zvec_db
    zvec_proto
    core_knn_flat
    core_knn_flat_sparse
    core_knn_hnsw
    core_knn_hnsw_sparse
    core_knn_ivf
    core_knn_hnsw_rabitq
    core_mix_reducer
    core_metric
    core_utility
    core_quantizer
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)


# Build data_generator executable
cc_binary(
    NAME data_generator
    LIBS ${CRASH_RECOVERY_COMMON_LIBS}
    SRCS data_generator.cc
    INCS .. ../../src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)


# Build collection_optimizer executable
cc_binary(
    NAME collection_optimizer
    LIBS ${CRASH_RECOVERY_COMMON_LIBS}
    SRCS collection_optimizer.cc
    INCS .. ../../src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)


# Build test executables
file(GLOB ALL_TEST_SRCS *_test.cc)
foreach(CC_SRCS ${ALL_TEST_SRCS})
    get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
    cc_gmock(
        NAME ${CC_TARGET} STRICT
        LIBS ${CRASH_RECOVERY_COMMON_LIBS}
        SRCS ${CC_SRCS}
        INCS .. ../../src
        LDFLAGS ${APPLE_FRAMEWORK_LIBS}
    )
    add_dependencies(${CC_TARGET} data_generator)
    add_dependencies(${CC_TARGET} collection_optimizer)
    cc_test_suite(zvec_crash_recovery ${CC_TARGET})
endforeach()
