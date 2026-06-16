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

# Ensure bitpacked_simd_sse41.cc is compiled with SSE4.1 flag and
# bitpacked_simd_avx2.cc with AVX2 flag in the packed zvec target as well
# (they are also compiled separately in zvec_index).
if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(HOST_ARCH MATCHES "^(x86|x64)$")
        setup_compiler_march_for_x86(_DB_MARCH_SSE _DB_MARCH_AVX2 _DB_MARCH_AVX512 _DB_MARCH_AVX512FP16)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/index/column/fts_column/posting/bitpacked_simd_sse41.cc
            PROPERTIES COMPILE_FLAGS "${_DB_MARCH_SSE}"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/index/column/fts_column/posting/bitpacked_simd_avx2.cc
            PROPERTIES COMPILE_FLAGS "${_DB_MARCH_AVX2}"
        )
    endif()
endif()

cc_library(
  NAME zvec STATIC STRICT SRCS_NO_GLOB PACKED
  SRCS ${ALL_DB_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/proto/zvec.pb.cc
  INCS . ${CMAKE_CURRENT_BINARY_DIR}
  PUBINCS ${PROJECT_ROOT_DIR}/src/include
  LIBS
    zvec_ailego
    zvec_core
    glog
    roaring
    rocksdb
    antlr4
    libprotobuf
    FastPFOR
    cppjieba
    Arrow::arrow_static
    Arrow::parquet_static
    Arrow::arrow_compute
    Arrow::arrow_dataset
    Arrow::arrow_acero
  DEPS zvec_proto
  VERSION "${PROXIMA_ZVEC_VERSION}"
)
