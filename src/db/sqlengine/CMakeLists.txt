include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(SCALAR_JIT_SRCS)
set(SCALAR_JIT_LIBS)
if(ZVEC_ENABLE_SCALAR_JIT)
    set(SCALAR_JIT_SRCS scalar_jit/scalar_predicate.cc scalar_jit/scalar_predicate_cache.cc
        scalar_jit/predicate_helpers.cc)
    set(SCALAR_JIT_LIBS Gandiva::gandiva_static)
endif()

cc_library(
    NAME zvec_sqlengine STATIC STRICT
    SRCS *.cc common/*.cc antlr/gen/*.cc parser/*.cc analyzer/*.cc planner/*.cc planner/ops/*.cc planner/physical_rules/*.cc ${SCALAR_JIT_SRCS}
    LIBS zvec_index
         zvec_common
         antlr4
         Arrow::arrow_acero
         ${SCALAR_JIT_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )

if(ZVEC_ENABLE_SCALAR_JIT)
    target_compile_definitions(zvec_sqlengine PUBLIC ZVEC_ENABLE_SCALAR_JIT=1)
endif()
