include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

option(ZVEC_FASTPFOR_PORTABLE
    "Use SIMDe for FastPFOR without requiring SSE4.1" OFF)

# On non-x86 platforms and in portable mode, FastPFOR uses SIMDe for SSE
# intrinsics. Detection also covers cross-compilation (e.g. iOS/Android).
set(_FASTPFOR_NEEDS_SIMDE FALSE)
if(ZVEC_FASTPFOR_PORTABLE)
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|arm64")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv32")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_FASTPFOR_NEEDS_SIMDE TRUE)
endif()

# RISC-V: preserve the existing patch and marker for previously configured trees.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv32")
    set(FASTPFOR_RISCV_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/fastpfor.riscv.patch)
    apply_patch_once("fastpfor_riscv_fix" "${CMAKE_CURRENT_SOURCE_DIR}/FastPFOR-0.4.0" "${FASTPFOR_RISCV_PATCH}")
endif()

if(ZVEC_FASTPFOR_PORTABLE)
    set(FASTPFOR_PORTABLE_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/fastpfor.portable.patch)
    apply_patch_once("fastpfor_portable_fix" "${CMAKE_CURRENT_SOURCE_DIR}/FastPFOR-0.4.0" "${FASTPFOR_PORTABLE_PATCH}")
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

if(ZVEC_FASTPFOR_PORTABLE)
    list(APPEND FASTPFOR_EXTRA_DEFS ZVEC_FASTPFOR_PORTABLE)
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

# cc_library's DEFS are private, but consumers instantiate FastPFOR's
# header-defined codecs and must use the same portable intrinsics.
if(ZVEC_FASTPFOR_PORTABLE)
    target_compile_definitions(FastPFOR INTERFACE ${FASTPFOR_EXTRA_DEFS})
endif()

# Mark SIMDe's x86 emulation headers as a SYSTEM include so consumers of
# FastPFOR (e.g. src/db's bitpacked posting-list translation units) do not
# inherit SIMDe's warnings under strict /W4 /WX builds. Without this MSVC
# ARM64 fails on C4189 / C4127 / C4324 from simde/x86/mmx.h.
if(_FASTPFOR_NEEDS_SIMDE)
    set_property(TARGET FastPFOR APPEND PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${simde_SOURCE_DIR}")
endif()
