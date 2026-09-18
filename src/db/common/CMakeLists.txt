include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# proxima_zvec_store library
cc_library(
    NAME zvec_common STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS    glog
            zvec_ailego
            roaring
            rocksdb
            utf8proc
    INCS .
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
