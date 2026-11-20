include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

cc_binary(
  NAME candidate_filter_bench PACKED
  SRCS candidate_filter_bench.cc
  LIBS zvec zvec_core
  INCS . ${PROJECT_ROOT_DIR}/src
  LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)
if(MSVC)
  target_link_options(candidate_filter_bench PRIVATE
    "/WHOLEARCHIVE:$<TARGET_FILE:zvec_core>")
elseif(APPLE)
  target_link_options(candidate_filter_bench PRIVATE
    "-Wl,-force_load,$<TARGET_FILE:zvec_core>")
else()
  target_link_options(candidate_filter_bench PRIVATE
    "-Wl,--whole-archive,$<TARGET_FILE:zvec_core>,--no-whole-archive")
endif()
