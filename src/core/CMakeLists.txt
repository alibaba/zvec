include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(framework)
cc_directory(algorithm)
cc_directory(metric)
cc_directory(quantizer)
cc_directory(utility)
cc_directory(interface)
cc_directory(mixed_reducer)

file(GLOB_RECURSE ALL_CORE_SRCS *.cc *.c *.h)

# Remove algorithm/hnsw_rabitq implementation files if not supported.
# interface/indexes/hnsw_rabitq_index.cc is kept because it provides the vtable
# for HNSWRabitqIndex and guards rabitqlib usage with #if RABITQ_SUPPORTED.
if(NOT RABITQ_SUPPORTED)
  list(FILTER ALL_CORE_SRCS EXCLUDE REGEX ".*/algorithm/hnsw_rabitq/.*")
  list(FILTER ALL_CORE_SRCS EXCLUDE REGEX ".*/algorithm/ivf_rabitq/.*")
endif()

# Keep the guarded DiskAnn interface available everywhere, but pack its
# implementation into zvec_core only on supported platforms.
if(NOT DISKANN_SUPPORTED)
  list(FILTER ALL_CORE_SRCS EXCLUDE REGEX ".*/algorithm/diskann/.*")
endif()

set(ZVEC_CORE_LIBS zvec_ailego zvec_turbo sparsehash magic_enum rabitqlib)
# The DiskAnn runtime loader uses dlopen/dlsym, so link libdl on Linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ZVEC_CORE_LIBS ${CMAKE_DL_LIBS})
endif()

set(ZVEC_CORE_LIBRARY_OPTIONS)
if(WIN32 AND (BUILD_ZVEC_CORE_SHARED OR BUILD_ZVEC_SHARED))
  list(APPEND ZVEC_CORE_LIBRARY_OPTIONS OBJECTS)
  if(BUILD_ZVEC_CORE_SHARED)
    list(APPEND ZVEC_CORE_LIBRARY_OPTIONS
      EXPORT_DEF ZVEC_CORE_BUILD_SHARED
    )
  endif()
endif()

cc_library(
    NAME zvec_core STATIC STRICT PACKED ${ZVEC_CORE_LIBRARY_OPTIONS}
    SRCS ${ALL_CORE_SRCS}
    LIBS ${ZVEC_CORE_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${ZVEC_VERSION_STRING}"
)
