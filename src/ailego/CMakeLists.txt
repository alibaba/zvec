include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE AND NOT ANDROID)
    find_library(LIB_RT NAMES rt)
else()
    set(LIB_RT "")
endif()

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
    list(APPEND EXTRA_LIBS ${LIB_RT})
endif()

# The archive is built but never installed.  The SDK publishes the stable API
# only, so zvec/ailego/** is not in the package, and an installed
# libzvec_ailego.a whose declaring headers are absent is a library that cannot be
# called.  It still has to exist in the build tree: libzvec and libzvec_c_api
# whole-archive it, and in-tree tests, examples and subproject-integration link
# the target directly.
set(ZVEC_AILEGO_LIBRARY_OPTIONS SKIP_ARCHIVE_INSTALL)

if(WIN32 AND
   (BUILD_ZVEC_AILEGO_SHARED OR BUILD_ZVEC_CORE_SHARED OR BUILD_ZVEC_SHARED))
    list(APPEND ZVEC_AILEGO_LIBRARY_OPTIONS
        OBJECTS
        EXPORT_DEF ZVEC_AILEGO_BUILD_SHARED
    )
endif()

set(ZVEC_AILEGO_DEFS)
if(ZVEC_ON_MUSL)
    list(APPEND ZVEC_AILEGO_DEFS ZVEC_ON_MUSL)
endif()

cc_library(
    NAME zvec_ailego STATIC STRICT PACKED ${ZVEC_AILEGO_LIBRARY_OPTIONS}
    SRCS    ${ALL_SRCS}
    DEFS    ${ZVEC_AILEGO_DEFS}
    LIBS    ${EXTRA_LIBS}
    VERSION "${ZVEC_VERSION_STRING}"
)
