
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
    NAME ${CC_TARGET} STRICT
    LIBS zvec_ailego core_framework core_utility
    DEFS ${ZVEC_BATCH_TOPK_DEFINITIONS}
    SRCS ${CC_SRCS}
    INCS . ${PROJECT_ROOT_DIR}/src/core/
  )
  cc_test_suite(zvec_ailego ${CC_TARGET})
endforeach()

# Exercise the fallback on SIMD-capable CI hosts as well.
set(BATCH_TOPK_SCALAR_SRCS
    ${PROJECT_ROOT_DIR}/src/core/utility/batch_topk.cc
    ${PROJECT_ROOT_DIR}/src/core/utility/batch_topk_avx2.cc
    ${PROJECT_ROOT_DIR}/src/core/utility/batch_topk_avx512.cc)
cc_gtest(
    NAME batch_topk_scalar_test STRICT
    SRCS batch_topk_test.cc ${BATCH_TOPK_SCALAR_SRCS}
    DEFS ZVEC_BATCH_TOPK_FORCE_SCALAR=1
    LIBS zvec_ailego core_framework
    INCS . ${PROJECT_ROOT_DIR}/src/core/
  )
cc_test_suite(zvec_ailego batch_topk_scalar_test)
