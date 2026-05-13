# CRoaring v2.0.4 - integrated as a single-file amalgamation build.
#
# We deliberately do NOT pull the upstream repository as a git submodule
# anymore: it ships ~292M of benchmarks/test data that we never build, which
# made `git submodule update --init --recursive` painfully slow and bloated
# the local checkout.
#
# Instead, the official amalgamation produced by CRoaring's release pipeline
# (https://github.com/RoaringBitmap/CRoaring/releases/tag/v2.0.4) is checked
# into CRoaring-2.0.4/:
#
#   CRoaring-2.0.4/roaring.c          - single C TU containing the entire library
#   CRoaring-2.0.4/roaring.h          - public C API (from official release)
#   CRoaring-2.0.4/roaring.hh         - public C++ wrapper (from official release)
#   CRoaring-2.0.4/roaring64map.hh    - 64-bit C++ wrapper (from cpp/ in tarball)
#   CRoaring-2.0.4/include/roaring/roaring.h
#                                   - identical to roaring.h, exposed under the
#                                     historical <roaring/roaring.h> path so
#                                     existing callers keep working.
#
# To upgrade, replace these files with the assets from a newer release tag
# (and grab cpp/roaring64map.hh from the source tarball if it is still split).
set(_ROARING_AMAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CRoaring-2.0.4)

add_library(roaring STATIC ${_ROARING_AMAL_DIR}/roaring.c)
set_target_properties(roaring PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR}
)

# Two include roots:
#   - CRoaring-2.0.4/         -> <roaring.h>, <roaring.hh>, <roaring64map.hh>
#   - CRoaring-2.0.4/include/ -> <roaring/roaring.h>  (legacy path used in code)
target_include_directories(roaring PUBLIC
    $<BUILD_INTERFACE:${_ROARING_AMAL_DIR}>
    $<BUILD_INTERFACE:${_ROARING_AMAL_DIR}/include>
)

# The amalgamation is plain C11; the C++ wrappers in roaring.hh require C++11+.
target_compile_features(roaring PUBLIC c_std_11)

# Silence warnings that come from the third-party amalgamation itself; we do
# not want them polluting our build.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang"
   OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(roaring PRIVATE
        -Wno-unused-parameter
        -Wno-unused-but-set-variable
        -Wno-unused-function
    )
endif()

# Preserve the legacy ROARING_* variables so any consumer that still reads
# them (instead of linking the `roaring` target directly) keeps working.
set(ROARING_FOUND        TRUE                          PARENT_SCOPE)
set(ROARING_INCLUDE_DIR  ${_ROARING_AMAL_DIR}          PARENT_SCOPE)
set(ROARING_INCLUDE_DIRS ${_ROARING_AMAL_DIR}
                         ${_ROARING_AMAL_DIR}/include  PARENT_SCOPE)
set(ROARING_LIBRARY      $<TARGET_FILE:roaring>        PARENT_SCOPE)
set(ROARING_LIBRARIES    $<TARGET_FILE:roaring>        PARENT_SCOPE)

unset(_ROARING_AMAL_DIR)
