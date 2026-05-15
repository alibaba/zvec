##
##  \file     CMakeLists.txt
##  \brief    Build script for FastPFOR SIMD bitpacking library (thirdparty)
##

include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)

cc_library(
    NAME FastPFOR STATIC
    SRCS FastPFOR-0.4.0/src/simdbitpacking.cpp
         FastPFOR-0.4.0/src/bitpacking.cpp
         FastPFOR-0.4.0/src/bitpackingaligned.cpp
         FastPFOR-0.4.0/src/bitpackingunaligned.cpp
         FastPFOR-0.4.0/src/simdunalignedbitpacking.cpp
    INCS FastPFOR-0.4.0/headers
    PUBINCS FastPFOR-0.4.0/headers
    CXXFLAGS -msse4.1
)
