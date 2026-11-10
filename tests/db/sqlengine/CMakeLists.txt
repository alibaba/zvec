
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

file(GLOB ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gmock(
    NAME ${CC_TARGET} STRICT
    LIBS zvec zvec_core zvec_ailego sparsehash
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    SRCS ${CC_SRCS}
    INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/db ${PROJECT_ROOT_DIR}/src/db/common
    LDFLAGS ${APPLE_FRAMEWORK_LIBS}
  )
  db_test_force_load_core(${CC_TARGET})
  cc_test_suite(zvec_sqlengine ${CC_TARGET})
endforeach()

if(ZVEC_ENABLE_SCALAR_JIT)
  # Internal binding benchmark needs Arrow's private/static symbols. Keep it
  # opt-in as a build target, separate from the public shared-library tools.
  add_executable(scalar_filter_binding_benchmark EXCLUDE_FROM_ALL
    ${PROJECT_ROOT_DIR}/tools/scalar_filter/binding_benchmark.cc)
  target_compile_features(scalar_filter_binding_benchmark PRIVATE cxx_std_17)
  target_include_directories(scalar_filter_binding_benchmark PRIVATE
    ${PROJECT_ROOT_DIR}/src)
  target_link_libraries(scalar_filter_binding_benchmark PRIVATE zvec)
endif()
