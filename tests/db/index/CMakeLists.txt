
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(APPLE)
  set(APPLE_FRAMEWORK_LIBS
    -framework CoreFoundation
    -framework CoreGraphics
    -framework CoreData
    -framework CoreText
    -framework Security
    -framework Foundation
    -Wl,-U,_MallocExtension_ReleaseFreeMemory
    -Wl,-U,_ProfilerStart
    -Wl,-U,_ProfilerStop
    -Wl,-U,_RegisterThriftProtocol
  )
endif()

file(GLOB_RECURSE ALL_TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_test.cc)

if(MSVC)
  #TODO(windows): should include these UTs
  set(_MSVC_EXCLUDE_TESTS
    "storage/mmap_store_test.cc"
    "storage/mem_store_test.cc"
    "column/inverted_column/inverted_indexer_util_test.cc"
    "segment/segment_test.cc"
  )
  foreach(_excl ${_MSVC_EXCLUDE_TESTS})
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX "${_excl}")
  endforeach()
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gmock(
    NAME ${CC_TARGET} STRICT
    LIBS zvec
    zvec_ailego
    zvec_core
    sparsehash
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    SRCS ${CC_SRCS} utils/utils.cc
    INCS . .. ../../src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  db_test_force_load_core(${CC_TARGET})
  cc_test_suite(zvec_index ${CC_TARGET})
endforeach()

# Android runners deploy the original resources beside each test workdir.
# Allow relocated test runners to override the fixture path independently of
# the embedded defaults used by production code.
if(ANDROID)
  set(JIEBA_TEST_DICT_DEFAULT_DIR "../jieba_dict")
else()
  set(JIEBA_TEST_DICT_DEFAULT_DIR
      "${PROJECT_SOURCE_DIR}/thirdparty/cppjieba/cppjieba-5.6.7/dict")
endif()
set(JIEBA_TEST_DICT_DIR "${JIEBA_TEST_DICT_DEFAULT_DIR}" CACHE STRING
    "Path to the original Jieba dictionary and model at test runtime")

# Inject TEST_SOURCE_DIR for fts_column_indexer_test so it can locate testdata/
if(TARGET fts_column_indexer_test)
  target_compile_definitions(fts_column_indexer_test PRIVATE
    TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/column/fts_column"
    JIEBA_DICT_DIR="${JIEBA_TEST_DICT_DIR}")
endif()

# Compare embedded Jieba resources with the original dictionary and model.
if(TARGET jieba_tokenizer_test)
  target_compile_definitions(jieba_tokenizer_test PRIVATE
    JIEBA_DICT_DIR="${JIEBA_TEST_DICT_DIR}")
endif()
