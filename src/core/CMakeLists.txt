include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(RABITQ_SUPPORTED AND AUTO_DETECT_ARCH)
  set(HNSW_RABITQ_FILES
      hnsw_rabitq_query_algorithm.cc
      hnsw_rabitq_streamer.cc
      hnsw_rabitq_searcher.cc
      hnsw_rabitq_entity.cc
      rabitq_reformer.cc
      rabitq_converter.cc
  )
  set(HNSW_RABITQ_FILES_FULL ${HNSW_RABITQ_FILES})
  list(TRANSFORM HNSW_RABITQ_FILES_FULL PREPEND "algorithm/hnsw_rabitq/")
  foreach(FILE ${HNSW_RABITQ_FILES_FULL})
      set_source_files_properties(
          ${FILE}
          PROPERTIES
          COMPILE_FLAGS "${RABITQ_ARCH_FLAG}"
      )
  endforeach()
endif()

cc_directory(framework)
cc_directory(algorithm)
cc_directory(metric)
cc_directory(quantizer)
cc_directory(utility)
cc_directory(interface)
cc_directory(mixed_reducer)

git_version(GIT_SRCS_VER ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ALL_CORE_SRCS *.cc *.c *.h)

# Remove algorithm/hnsw_rabitq implementation files if not supported.
# interface/indexes/hnsw_rabitq_index.cc is kept because it provides the vtable
# for HNSWRabitqIndex and guards rabitqlib usage with #if RABITQ_SUPPORTED.
if(NOT RABITQ_SUPPORTED)
  list(FILTER ALL_CORE_SRCS EXCLUDE REGEX ".*/algorithm/hnsw_rabitq/.*")
endif()

cc_library(
    NAME zvec_core STATIC STRICT PACKED
    SRCS ${ALL_CORE_SRCS}
    LIBS zvec_ailego zvec_turbo sparsehash magic_enum rabitqlib
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${GIT_SRCS_VER}"
)
