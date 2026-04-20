include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(common)
cc_directory(index)
cc_directory(sqlengine)

file(GLOB_RECURSE ALL_DB_SRCS *.cc *.c *.h)

cc_library(
  NAME zvec_db STATIC STRICT SRCS_NO_GLOB PACKED
  SRCS ${ALL_DB_SRCS}
  INCS .
  PUBINCS ${PROJECT_ROOT_DIR}/src/include
  LIBS
    zvec_ailego
    zvec_core
    glog
    roaring
    rocksdb
    antlr4
    Arrow::arrow_static
    Arrow::arrow_compute
    Arrow::arrow_dataset
    Arrow::arrow_acero
  VERSION "${PROXIMA_ZVEC_VERSION}"
)