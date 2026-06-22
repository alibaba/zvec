include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

# On non-x86 platforms, FastPFOR uses SIMDe to emulate SSE intrinsics.
# Detection covers native ARM/RISC-V builds AND cross-compilation (e.g. iOS/Android).
set(_FASTPFOR_NEEDS_SIMDE FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|arm64")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv32")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
endif()

# RISC-V: patch FastPFOR common.h to include SIMDe header for __riscv
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv32")
    set(FASTPFOR_RISCV_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/fastpfor.riscv.patch)
    apply_patch_once("fastpfor_riscv_fix" "${CMAKE_CURRENT_SOURCE_DIR}/FastPFOR-0.4.0" "${FASTPFOR_RISCV_PATCH}")
endif()

if(_FASTPFOR_NEEDS_SIMDE)
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
