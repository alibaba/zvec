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
  cc_test_suite(zvec ${CC_TARGET})
endforeach()
