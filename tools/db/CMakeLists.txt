include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_binary(
    NAME fts_bench PACKED
    SRCS fts_bench_main.cc
    LIBS
         zvec
         gflags
         roaring
         rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)

# Compare upstream-style per-candidate locked fetch directly with owning views.
# The locked control supplies DocFilter's translation unit directly, replacing
# the archive member. Both targets retain the same Arrow evaluator and query
# lifecycle plumbing; the control is not a pristine upstream binary.
foreach(BENCH_TARGET filter_lock_bench_locked filter_view_bench)
  set(CONTROL_SOURCE)
  set(CONTROL_DEFS)
  if(BENCH_TARGET STREQUAL "filter_lock_bench_locked")
    set(CONTROL_SOURCE ${PROJECT_SOURCE_DIR}/src/db/sqlengine/planner/doc_filter.cc)
    set(CONTROL_DEFS ZVEC_FORWARD_FILTER_DISABLE_VIEW)
  endif()
  cc_binary(
    NAME ${BENCH_TARGET} PACKED
    SRCS filter_lock_bench_main.cc ${CONTROL_SOURCE}
    DEFS ${CONTROL_DEFS}
    LIBS zvec zvec_core gflags
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  # Include core registries (e.g. MMapFileStorage) in the executable.
  if(MSVC)
    target_link_options(${BENCH_TARGET} PRIVATE
      "/WHOLEARCHIVE:$<TARGET_FILE:zvec_core>")
  elseif(APPLE)
    target_link_options(${BENCH_TARGET} PRIVATE
      "-Wl,-force_load,$<TARGET_FILE:zvec_core>")
  else()
    target_link_options(${BENCH_TARGET} PRIVATE
      "-Wl,--whole-archive,$<TARGET_FILE:zvec_core>,--no-whole-archive")
  endif()
endforeach()
