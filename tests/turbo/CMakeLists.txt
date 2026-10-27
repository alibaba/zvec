include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_ailego core_framework core_metric core_quantizer zvec_turbo
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core/ ${PROJECT_ROOT_DIR}/src/turbo/
    )
  target_compile_definitions(${CC_TARGET} PRIVATE
    ZVEC_TURBO_HAVE_SME2=$<BOOL:${ZVEC_TURBO_HAVE_SME2}>
    ZVEC_TURBO_BUILD_AVX2=$<BOOL:${ZVEC_TURBO_COMPILER_AVX2}>
    ZVEC_TURBO_BUILD_AVX512=$<BOOL:${ZVEC_TURBO_COMPILER_AVX512}>
    ZVEC_TURBO_BUILD_VNNI=$<BOOL:${ZVEC_TURBO_COMPILER_VNNI}>)
endforeach()
