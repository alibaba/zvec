include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_SRCS *.cc *.c)

# The DiskAnn plugin is loaded at runtime via zvec::LoadDiskAnnPlugin() with
# RTLD_GLOBAL, so its undefined references to internal zvec symbols
# (core_framework, core_knn_cluster, zvec_ailego, ...) are resolved at load
# time against the hosting binary (_zvec.so for the Python extension, the
# test executable for gtest, or libzvec_core for tools).
#
# As a consequence the plugin .so must NOT carry NEEDED entries for those
# libs: otherwise the dynamic loader would try to resolve them from disk
# before invoking dlopen's RTLD_GLOBAL symbol sharing, and the Python wheel
# (which only ships _zvec.so + the plugin) would fail to load.
#
# We therefore link the plugin only against its system-level dependency
# (libaio) and rely on the global include dirs plus -Wl,--unresolved-symbols
# =ignore-all to let the linker build a shared library with unresolved
# references to internal APIs.
set(CORE_KNN_DISKANN_LIBS "")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386")
    list(APPEND CORE_KNN_DISKANN_LIBS aio)
endif()

cc_library(
    NAME core_knn_diskann
    SHARED STRICT
    SRCS *.cc
    LIBS ${CORE_KNN_DISKANN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
)

# Internal zvec libs are referenced only for header availability; the actual
# symbol resolution happens at dlopen time. Expose their public include dirs
# without adding them as link dependencies.
foreach(_dep zvec_ailego core_framework core_knn_cluster)
    if(TARGET ${_dep})
        target_include_directories(core_knn_diskann PRIVATE
            $<TARGET_PROPERTY:${_dep},INTERFACE_INCLUDE_DIRECTORIES>)
        add_dependencies(core_knn_diskann ${_dep})
    endif()
endforeach()

# Allow the plugin to have unresolved symbols that will be satisfied at
# dlopen(RTLD_NOW | RTLD_GLOBAL) time by the hosting binary.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(core_knn_diskann PRIVATE
        "LINKER:--unresolved-symbols=ignore-all")
endif()

# Rename the artifact to libzvec_diskann_plugin.so so it is clearly identified
# as an optional plugin that users load at runtime via zvec::LoadDiskAnnPlugin().
set_target_properties(core_knn_diskann PROPERTIES
    OUTPUT_NAME zvec_diskann_plugin
)