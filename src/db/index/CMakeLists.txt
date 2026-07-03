include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        setup_compiler_march_for_x86(INDEX_MARCH_FLAG_SSE INDEX_MARCH_FLAG_AVX2 INDEX_MARCH_FLAG_AVX512 INDEX_MARCH_FLAG_AVX512FP16)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/column/fts_column/posting/bitpacked_simd_sse41.cc
            PROPERTIES
            COMPILE_FLAGS "${INDEX_MARCH_FLAG_SSE}"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/column/fts_column/posting/bitpacked_simd_avx2.cc
            PROPERTIES
            COMPILE_FLAGS "${INDEX_MARCH_FLAG_AVX2}"
        )
    endif()
endif()

cc_library(
    NAME zvec_index STATIC STRICT
    SRCS *.cc segment/*.cc column/vector_column/*.cc column/inverted_column/*.cc column/fts_column/*.cc column/fts_column/tokenizer/*.cc column/fts_column/posting/*.cc column/fts_column/iterator/*.cc storage/*.cc storage/wal/*.cc common/*.cc
    LIBS zvec_common
         zvec_proto
         rocksdb
         core_interface
         Arrow::arrow_static
         Arrow::parquet_static
         Arrow::arrow_compute
         Arrow::arrow_dataset
         cppjieba
         FastPFOR
         utf8proc
    INCS .  ${PROJECT_ROOT_DIR}/src
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
