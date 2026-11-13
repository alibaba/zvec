include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

function(db_test_force_load_core TARGET)
  # Include core registries and keep their RTTI in the test executable.
  if(MSVC)
    target_link_options(${TARGET} PRIVATE
      "/WHOLEARCHIVE:$<TARGET_FILE:zvec_core>")
  elseif(APPLE)
    target_link_options(${TARGET} PRIVATE
      "-Wl,-force_load,$<TARGET_FILE:zvec_core>")
  else()
    target_link_options(${TARGET} PRIVATE
      "-Wl,--whole-archive,$<TARGET_FILE:zvec_core>,--no-whole-archive")
  endif()
endfunction()

cc_directory(common)
cc_directories(crash_recovery)
cc_directory(sqlengine)
cc_directories(index)
cc_directory(utf8_acp)

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

file(GLOB ALL_TEST_SRCS *_test.cc)

option(ZVEC_FORWARD_FILTER_SANITIZERS
  "Instrument forward-filter query/storage code and its focused tests" OFF)
if(ZVEC_FORWARD_FILTER_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Forward-filter sanitizers require Clang or GCC")
  endif()
  # Link instrumented definitions directly into the focused tests, replacing
  # the corresponding static archive members. Dependencies remain unchanged;
  # this is targeted coverage, not a fully instrumented dependency build.
  set(FORWARD_SANITIZED_SOURCES
    collection.cc
    index/segment/segment.cc
    index/storage/memory_forward_store.cc
    index/storage/mmap_forward_store.cc
    index/storage/bufferpool_forward_store.cc
    sqlengine/sqlengine_impl.cc
    sqlengine/planner/doc_filter.cc
    sqlengine/planner/query_planner.cc
    sqlengine/planner/plan_info.cc
    sqlengine/planner/segment_node.cc)
  list(TRANSFORM FORWARD_SANITIZED_SOURCES PREPEND "${PROJECT_SOURCE_DIR}/src/db/")
  add_library(forward_filter_sanitized OBJECT ${FORWARD_SANITIZED_SOURCES})
  target_include_directories(forward_filter_sanitized PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(forward_filter_sanitized PRIVATE zvec)
  target_compile_options(forward_filter_sanitized PRIVATE
    -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  if(APPLE)
    # libc++ vector annotations cannot be mixed with uninstrumented archive
    # instantiations. Keep allocation bounds/UAF checks, without capacity
    # annotations, consistently across the instrumented objects and tests.
    target_compile_definitions(forward_filter_sanitized PRIVATE
      __SANITIZER_DISABLE_CONTAINER_OVERFLOW__)
  endif()
endif()

# The collection DiskAnn stress cases repeatedly rebuild large indexes and are
# intended for desktop CI. Mobile CI exercises DiskAnn through the focused
# diskann_mobile_collection_test target and the core compatibility suite.
if(ANDROID OR IOS)
  set(DISKANN_STRESS_TESTS 0)
else()
  set(DISKANN_STRESS_TESTS 1)
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gmock(
    NAME ${CC_TARGET} STRICT
    DEFS DISKANN_STRESS_TESTS=${DISKANN_STRESS_TESTS}
    LIBS zvec zvec_core
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    SRCS ${CC_SRCS} index/utils/utils.cc
    INCS . .. ../../src
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  db_test_force_load_core(${CC_TARGET})
  if(ZVEC_FORWARD_FILTER_SANITIZERS AND
     CC_TARGET MATCHES "^(forward_filter_.*|concurrent_read)_test$")
    target_sources(${CC_TARGET} PRIVATE $<TARGET_OBJECTS:forward_filter_sanitized>)
    target_compile_options(${CC_TARGET} PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(${CC_TARGET} PRIVATE -fsanitize=address,undefined)
    if(APPLE)
      target_compile_definitions(${CC_TARGET} PRIVATE
        __SANITIZER_DISABLE_CONTAINER_OVERFLOW__)
    endif()
  endif()
  cc_test_suite(zvec ${CC_TARGET})
endforeach()
