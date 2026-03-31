include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME zvec_index STATIC STRICT
    SRCS *.cc segment/*.cc column/vector_column/*.cc column/inverted_column/*.cc storage/*.cc storage/wal/*.cc common/*.cc
    LIBS zvec_common
         zvec_proto
         rocksdb
         core_interface
         Arrow::arrow_static
         Arrow::arrow_compute
         Arrow::arrow_dataset
    INCS .  ${PROJECT_ROOT_DIR}/src
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
