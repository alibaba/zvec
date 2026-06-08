##
##  Copyright (C) The Software Authors. All rights reserved.
##
##  \file     CMakeLists.txt
##  \brief    Build script for the zvec internal plugin-loading glue library.
##            Lives inside the main zvec_core artifact and provides the
##            implicit DiskAnn runtime bring-up used by DiskAnnIndex on first
##            use. These APIs are NOT part of the public user surface: users
##            simply instantiate ``DiskAnnIndexParam`` / ``DiskAnnIndex`` and
##            the runtime (``libzvec_diskann_plugin.so``) is loaded behind
##            the scenes.
##

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CORE_PLUGIN_LIBS zvec_ailego core_framework)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CORE_PLUGIN_LIBS ${CMAKE_DL_LIBS})
endif()

cc_library(
    NAME core_plugin
    STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS ${CORE_PLUGIN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
