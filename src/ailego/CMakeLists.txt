include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE AND NOT ANDROID)
    find_library(LIB_RT NAMES rt)
else()
    set(LIB_RT "")
endif()

git_version(GIT_SRCS_VER ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
    list(APPEND EXTRA_LIBS ${LIB_RT})
endif()

cc_library(
    NAME zvec_ailego STATIC STRICT PACKED
    SRCS    ${ALL_SRCS}
    LIBS    ${EXTRA_LIBS}
            Arrow::arrow_static
            Arrow::parquet_static
    VERSION "${GIT_SRCS_VER}"
)