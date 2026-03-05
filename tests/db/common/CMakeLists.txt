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
        LIBS zvec_common
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        SRCS ${CC_SRCS}
        INCS .. ../../src
        LDFLAGS ${APPLE_FRAMEWORK_LIBS}
    )
    cc_test_suite(zvec_common ${CC_TARGET})
endforeach()
