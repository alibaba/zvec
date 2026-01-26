include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_proto_library(
  NAME zvec_proto STATIC
  SRCS proto/*.proto
  PROTOROOT ./
)

cc_directory(common)
cc_directory(index)
cc_directory(sqlengine)

file(GLOB_RECURSE ALL_DB_SRCS *.cc *.c *.h)

cc_library(
  NAME zvec_db STATIC STRICT SRCS_NO_GLOB
  SRCS ${ALL_DB_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/proto/zvec.pb.cc
  INCS . ${CMAKE_CURRENT_BINARY_DIR}
  LIBS 
    zvec_ailego
    zvec_core
    glog
    roaring
    rocksdb
    antlr4
    libprotobuf
    Arrow::arrow_static
    Arrow::arrow_compute
    Arrow::arrow_dataset
    Arrow::arrow_acero
  DEPS zvec_proto
  VERSION "${PROXIMA_ZVEC_VERSION}"
)