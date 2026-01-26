include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME zvec_sqlengine STATIC STRICT
    SRCS *.cc common/*.cc antlr/gen/*.cc parser/*.cc analyzer/*.cc planner/*.cc planner/ops/*.cc planner/physical_rules/*.cc
    LIBS zvec_index
         zvec_common
         antlr4
         Arrow::arrow_acero
    INCS . ${PROJECT_ROOT_DIR}/src
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
