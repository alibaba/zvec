include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)

# On ARM platforms, FastPFOR uses SIMDe to emulate SSE intrinsics.
# Detection covers native ARM builds AND cross-compilation (e.g. iOS/Android).
set(_FASTPFOR_IS_ARM FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|arm64")
    set(_FASTPFOR_IS_ARM TRUE)
elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(_FASTPFOR_IS_ARM TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_FASTPFOR_IS_ARM TRUE)
endif()

if(_FASTPFOR_IS_ARM)
    include(FetchContent)
    FetchContent_Declare(
        simde
        GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
        GIT_TAG v0.8.2
    )
    FetchContent_MakeAvailable(simde)
    set(FASTPFOR_EXTRA_INCS ${simde_SOURCE_DIR})
    set(FASTPFOR_EXTRA_CXXFLAGS "")
    set(FASTPFOR_EXTRA_DEFS SIMDE_ENABLE_NATIVE_ALIASES)
elseif(MSVC)
    set(FASTPFOR_EXTRA_INCS "")
    set(FASTPFOR_EXTRA_CXXFLAGS "")
    set(FASTPFOR_EXTRA_DEFS "")
else()
    set(FASTPFOR_EXTRA_INCS "")
    set(FASTPFOR_EXTRA_CXXFLAGS -msse4.1)
    set(FASTPFOR_EXTRA_DEFS "")
endif()

cc_library(
    NAME FastPFOR STATIC
    SRCS FastPFOR-0.4.0/src/simdbitpacking.cpp
         FastPFOR-0.4.0/src/bitpacking.cpp
         FastPFOR-0.4.0/src/bitpackingaligned.cpp
         FastPFOR-0.4.0/src/bitpackingunaligned.cpp
         FastPFOR-0.4.0/src/simdunalignedbitpacking.cpp
    INCS FastPFOR-0.4.0/headers ${FASTPFOR_EXTRA_INCS}
    PUBINCS FastPFOR-0.4.0/headers ${FASTPFOR_EXTRA_INCS}
    DEFS ${FASTPFOR_EXTRA_DEFS}
    CXXFLAGS ${FASTPFOR_EXTRA_CXXFLAGS}
)
