include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB ZVEC_INDEX_SRCS
     *.cc
     segment/*.cc
     column/vector_column/*.cc
     column/inverted_column/*.cc
     storage/*.cc
     storage/wal/*.cc
     common/*.cc
     ../training/*.cc)

if(NOT ZVEC_ENABLE_OMEGA)
  list(FILTER ZVEC_INDEX_SRCS EXCLUDE REGEX ".*/training/omega_model_trainer\\.cc$")
  list(FILTER ZVEC_INDEX_SRCS EXCLUDE REGEX ".*/training/omega_training_coordinator\\.cc$")
endif()

set(ZVEC_INDEX_LIBS
         zvec_common
         zvec_proto
         rocksdb
         core_interface
         Arrow::arrow_static
         Arrow::arrow_compute
         Arrow::arrow_dataset)
if(ZVEC_ENABLE_OMEGA)
  list(APPEND ZVEC_INDEX_LIBS omega)
endif()

cc_library(
    NAME zvec_index STATIC STRICT
    SRCS ${ZVEC_INDEX_SRCS}
    LIBS ${ZVEC_INDEX_LIBS}
    INCS .  ${PROJECT_ROOT_DIR}/src
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
