include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CORE_INTERFACE_LIBS zvec_ailego core_framework sparsehash magic_enum rabitqlib)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CORE_INTERFACE_LIBS ${CMAKE_DL_LIBS})
endif()

cc_library(
        NAME core_interface STATIC STRICT ALWAYS_LINK
        SRCS *.cc indexes/*.cc
        INCS . ${PROJECT_ROOT_DIR}/src/ ${PROJECT_ROOT_DIR}/src/core
        LIBS ${CORE_INTERFACE_LIBS}
        VERSION "${PROXIMA_ZVEC_VERSION}"
)
