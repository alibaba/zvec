cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0077 NEW)
# Some thirdparty subdirectories declare cmake_minimum_required < 3.25, which would
# leave CMP0141 unset for them and make them ignore CMAKE_MSVC_DEBUG_INFORMATION_FORMAT.
# Force NEW as the default so /Z7 (see below) is applied uniformly across the whole tree.
set(CMAKE_POLICY_DEFAULT_CMP0141 NEW)
project(zvec)
set(CC_CXX_STANDARD 17)

if(MSVC)
    set(INTTYPES_FORMAT VC7)
    add_compile_options(/utf-8)
    # Use /Z7 (embed debug info in .obj) instead of /Zi (separate .pdb) to enable sccache/ccache.
    # Since CMake 3.25 (CMP0141=NEW) the debug info format is no longer part of
    # CMAKE_*_FLAGS_<CFG> but is driven by this abstract variable and appended at the end of
    # the compile command. Patching /Zi->/Z7 in the flag strings is therefore a no-op and the
    # abstract default (/Zi) would clash with thirdparty /Z7, producing warning D9025.
    # CACHE FORCE so it propagates into thirdparty subdirectories that call project().
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>" CACHE STRING "" FORCE)
    add_compile_options(/EHsc) # def c++ exception behavior
    add_compile_options(/Zc:preprocessor /Zc:__cplusplus)
    add_compile_options(/we4716) # -Werror=return-type

    ###### reduce output length to make vibe coding work better :) should be removed or solved later
    # TODO(windows): fix&cleanup
    add_compile_options(/wd4267 /wd4244 /wd4305 /wd4242) # conversion/truncation from 'size_t' to 'type', possible loss of data
    add_compile_options(/wd4310) # cast truncates constant value
    add_compile_options(/wd4146) # unary minus operator applied to unsigned type #usage: uint32_t seg_id_{-1U};
    add_compile_options(/wd4245) # conversion from 'type1' to 'type2', signed/unsigned mismatch
    add_compile_options(/wd4251) # class 'type' needs to have dll-interface to be used by clients of class 'type2'
    add_compile_options(/wd4099) # type name first seen using 'object_type1' now seen using 'object_type2'
    add_compile_options(/wd4200) # nonstandard extension used : zero-sized array in struct/union
    add_compile_options(/wd4324) # structure was padded due to alignment specifier
    add_compile_options(/wd4702) # unreachable code
    add_compile_options(/wd4996) # wstring_convert are deprecated in C++17 # usage: in antlr4
    add_compile_options(/wd4101) # unreferenced local variable
    add_compile_options(/wd4996) # The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name: _lseek
    # result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?) # usage: keeps |= (1 << k);
    add_compile_options(/wd4334)
    ######

else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror=return-type")
    # Fine-grained dead code elimination: place each function/data in its own section
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT IOS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

# Detect musl libc. Some code paths need to avoid large stack buffers on musl
# because its default thread stack is much smaller than glibc's (128KB vs 8MB).
# Detection: musl's ldd prints "musl libc" to stderr.
set(ZVEC_ON_MUSL FALSE CACHE BOOL "Building on musl libc" FORCE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND ldd --version
        OUTPUT_VARIABLE _zvec_ldd_stdout
        ERROR_VARIABLE _zvec_ldd_stderr
    )
    if("${_zvec_ldd_stdout}${_zvec_ldd_stderr}" MATCHES "musl")
        set(ZVEC_ON_MUSL TRUE CACHE BOOL "Building on musl libc" FORCE)
        message(STATUS "musl libc detected")
    endif()
endif()

if(NOT DEFINED PROJECT_ROOT_DIR OR NOT PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

# LTO (Link-Time Optimization) — helps on Linux with --gc-sections but may
# increase size on macOS due to aggressive inlining.  Default OFF; enable
# explicitly with -DZVEC_ENABLE_LTO=ON when targeting Linux.
option(ZVEC_ENABLE_LTO "Enable Link-Time Optimization for Release builds" OFF)
if(ZVEC_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_OUTPUT)
    if(IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO enabled for Release build")
    else()
        message(STATUS "LTO not supported: ${IPO_OUTPUT}")
    endif()
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
include(${PROJECT_ROOT_DIR}/cmake/version.cmake)

# iOS platform detection
if(NOT ANDROID AND NOT IOS AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(IOS TRUE)
endif()

# iOS bundle properties for test executables
if(IOS)
  set(MACOSX_BUNDLE_BUNDLE_VERSION "1")
  set(MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0")
  set(CMAKE_MACOSX_BUNDLE_INFO_PLIST "${PROJECT_ROOT_DIR}/cmake/iOSBundleInfo.plist.in")
endif()

if (NOT ANDROID AND NOT IOS AND AUTO_DETECT_ARCH AND HOST_ARCH MATCHES "^(x86|x64)$")
  setup_compiler_march_for_x86(MATH_MARCH_FLAG_SSE MATH_MARCH_FLAG_AVX2 MATH_MARCH_FLAG_AVX512 MATH_MARCH_FLAG_AVX512FP16)
  message(STATUS "best compiler march, sse: " ${MATH_MARCH_FLAG_SSE} ", avx2: " ${MATH_MARCH_FLAG_AVX2} ", avx512: " ${MATH_MARCH_FLAG_AVX512} ", avx512fp16: " ${MATH_MARCH_FLAG_AVX512FP16})
endif()

include_directories(${PROJECT_ROOT_DIR}/src/include)
include_directories(${PROJECT_ROOT_DIR}/src)

option(BUILD_ZVEC_SHARED "Build all-in-one C++ shared library libzvec" ON)
option(BUILD_ZVEC_AILEGO_SHARED "Build all-in-one zvec-ailego shared library libzvec_ailego" ON)
option(BUILD_ZVEC_CORE_SHARED "Build all-in-one zvec-core shared library libzvec_core" ON)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
option(BUILD_C_BINDINGS "Build C bindings" ON)
option(BUILD_TOOLS "Build tools" ON)

message(STATUS "BUILD_ZVEC_SHARED:${BUILD_ZVEC_SHARED}")
message(STATUS "BUILD_ZVEC_AILEGO_SHARED:${BUILD_ZVEC_AILEGO_SHARED}")
message(STATUS "BUILD_ZVEC_CORE_SHARED:${BUILD_ZVEC_CORE_SHARED}")
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")
message(STATUS "BUILD_C_BINDINGS:${BUILD_C_BINDINGS}")
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64" AND NOT ANDROID AND NOT IOS)
  include(CheckCXXCompilerFlag)

  check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_RABITQ_AVX2)
  check_cxx_compiler_flag(
      "-mavx512f -mavx512bw -mavx512dq -mfma"
      COMPILER_SUPPORTS_RABITQ_AVX512)
  check_cxx_compiler_flag(
      "-mavx512f -mavx512bw -mavx512dq -mavx512vpopcntdq -mfma"
      COMPILER_SUPPORTS_RABITQ_AVX512_POPCNT)

  if(COMPILER_SUPPORTS_RABITQ_AVX2 AND
     COMPILER_SUPPORTS_RABITQ_AVX512 AND
     COMPILER_SUPPORTS_RABITQ_AVX512_POPCNT)
    set(RABITQ_SUPPORTED ON)
    add_definitions(-DRABITQ_SUPPORTED=1)
  else()
    set(RABITQ_SUPPORTED OFF)
    add_definitions(-DRABITQ_SUPPORTED=0)
    message(STATUS "RaBitQ support disabled - compiler cannot build all runtime dispatch variants")
  endif()
elseif(IOS)
  set(RABITQ_SUPPORTED OFF)
  add_definitions(-DRABITQ_SUPPORTED=0)
  message(STATUS "RaBitQ support disabled - not supported on iOS")
else()
  set(RABITQ_SUPPORTED OFF)
  add_definitions(-DRABITQ_SUPPORTED=0)
  message(STATUS "RaBitQ support disabled - only supported on Linux x86_64")
endif()
message(STATUS "RABITQ_SUPPORTED: ${RABITQ_SUPPORTED}")

# DiskAnn support:
#   - Linux x86_64 and ARM64 with io_uring, libaio, or pread
#   - macOS ARM64 (Apple Silicon) with synchronous pread
#   - Windows x86_64 with overlapped I/O
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|aarch64|arm64)$" AND NOT ANDROID AND NOT IOS)
    OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" AND NOT IOS)
    OR (WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"))
  set(DISKANN_SUPPORTED ON)
  add_definitions(-DDISKANN_SUPPORTED=1)
else()
  set(DISKANN_SUPPORTED OFF)
  add_definitions(-DDISKANN_SUPPORTED=0)
  message(STATUS "DiskAnn support disabled for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()
message(STATUS "DISKANN_SUPPORTED: ${DISKANN_SUPPORTED}")

option(USE_OSS_MIRROR "Use OSS mirror for faster third-party downloads" OFF)
if(DEFINED ENV{USE_OSS_MIRROR} AND NOT "$ENV{USE_OSS_MIRROR}" STREQUAL "")
  set(USE_OSS_MIRROR "$ENV{USE_OSS_MIRROR}" CACHE BOOL "Use OSS mirror for faster third-party downloads" FORCE)
endif()
message(STATUS "USE_OSS_MIRROR:${USE_OSS_MIRROR}")

cc_directory(thirdparty)
cc_directories(src)

cc_directories(tests)

add_custom_target(clang_tidy_deps DEPENDS ARROW.BUILD glog gflags Lz4.BUILD)

if(BUILD_TOOLS)
    cc_directories(tools)
endif()

set(CPACK_PACKAGE_VERSION ${ZVEC_VERSION_NUMBER})
set(CPACK_PACKAGE_VERSION_MAJOR ${ZVEC_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${ZVEC_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${ZVEC_VERSION_PATCH})
set(CPACK_PACKAGE_NAME zvec)
include(CPack)

# cppjieba dict files bundled with installed artifacts so the `jieba` FTS
# tokenizer works out of the box (source of truth for both the Python wheel
# and the native SDK packages).
set(ZVEC_JIEBA_DICT_SRC
    "${PROJECT_SOURCE_DIR}/thirdparty/cppjieba/cppjieba-5.6.7/dict")

if(BUILD_PYTHON_BINDINGS)
    if(APPLE)
        set(CMAKE_STRIP "")
        message(STATUS "Disabled strip on macOS to preserve code signature")
    endif()

    include(GNUInstallDirs)
    if(DEFINED SKBUILD_PLATLIB_DIR)
        set(ZVEC_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
    elseif(DEFINED Python_SITEARCH)
        set(ZVEC_PY_INSTALL_DIR "${Python_SITEARCH}")
    else()
        set(ZVEC_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    message(STATUS "Zvec install path: ${ZVEC_PY_INSTALL_DIR}")
    # COMPONENT python: only these runtime artifacts are pulled into the wheel
    # (see install.components in pyproject.toml). The cc_library(PACKED) SDK
    # install rules use the default component and are excluded from the wheel.
    # Install the extension inside the zvec package (zvec/_zvec*.so) rather than
    # at the site-packages root, so it does not pollute the top-level namespace.
    # The Python code imports it as `zvec._zvec` accordingly.
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR}/zvec
            COMPONENT python)

    # DiskAnn is now statically linked into _zvec.so via --whole-archive
    # (see src/binding/python/CMakeLists.txt -> core_knn_diskann_static),
    # so no separate runtime .so needs to be installed in the wheel.
    # Bundle cppjieba's dictionary files so the `jieba` FTS tokenizer works
    # out of the box. python/zvec/__init__.py resolves this directory via
    # importlib.resources and registers it with set_default_jieba_dict_dir().
    install(FILES
        "${ZVEC_JIEBA_DICT_SRC}/jieba.dict.utf8"
        "${ZVEC_JIEBA_DICT_SRC}/hmm_model.utf8"
        DESTINATION ${ZVEC_PY_INSTALL_DIR}/zvec/data/jieba_dict
        COMPONENT python)
endif()

# Ship the same dict files with the native SDK install (prebuilt packages)
# as well, under <prefix>/data/jieba_dict (mirrors the wheel's
# zvec/data/jieba_dict layout). Unlike Python there is no
# import-time registration: consumers register the directory explicitly via
# zvec_set_default_jieba_dict_dir() / zvec_config_data_set_jieba_dict_dir(),
# the ZVEC_JIEBA_DICT_DIR env var, or per-field extra_params.jieba_dict_dir.
# Default component, so the wheel build (COMPONENT python) is unaffected.
install(FILES
    "${ZVEC_JIEBA_DICT_SRC}/jieba.dict.utf8"
    "${ZVEC_JIEBA_DICT_SRC}/hmm_model.utf8"
    DESTINATION data/jieba_dict)
