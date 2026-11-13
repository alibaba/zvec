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

cc_binary(
    NAME bsi_bench PACKED
    SRCS bsi_bench_main.cc
    LIBS zvec gflags roaring rocksdb
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
)

# Preserve the vector/storage factory registrations used by Collection.
if(MSVC)
    target_link_options(bsi_bench PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:zvec_core>")
elseif(APPLE)
    target_link_options(bsi_bench PRIVATE "-Wl,-force_load,$<TARGET_FILE:zvec_core>")
else()
    target_link_options(bsi_bench PRIVATE
        "-Wl,--whole-archive,$<TARGET_FILE:zvec_core>,--no-whole-archive")
endif()
