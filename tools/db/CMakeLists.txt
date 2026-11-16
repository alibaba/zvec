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

if(APPLE)
    set(NONVECTOR_FRAMEWORK_LIBS
        -framework CoreFoundation -framework CoreGraphics -framework CoreData
        -framework CoreText -framework Security -framework Foundation
        -Wl,-U,_MallocExtension_ReleaseFreeMemory
        -Wl,-U,_ProfilerStart -Wl,-U,_ProfilerStop
        -Wl,-U,_RegisterThriftProtocol)
endif()

cc_binary(
    NAME nonvector_benchmark PACKED
    SRCS nonvector_benchmark.cc
    LIBS zvec zvec_core
    INCS . ${PROJECT_SOURCE_DIR}/src
    LDFLAGS ${NONVECTOR_FRAMEWORK_LIBS}
)
if(MSVC)
    target_link_libraries(nonvector_benchmark psapi)
    target_link_options(nonvector_benchmark PRIVATE
        "/WHOLEARCHIVE:$<TARGET_FILE:zvec_core>")
elseif(APPLE)
    target_link_options(nonvector_benchmark PRIVATE
        "-Wl,-force_load,$<TARGET_FILE:zvec_core>")
else()
    target_link_options(nonvector_benchmark PRIVATE
        "-Wl,--whole-archive,$<TARGET_FILE:zvec_core>,--no-whole-archive")
endif()
