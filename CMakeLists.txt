cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0077 NEW)
project(zvec)
set(CC_CXX_STANDARD 17)

if(MSVC)
    set(INTTYPES_FORMAT VC7)
    # Use /Z7 (embed debug info in .obj) instead of /Zi (separate .pdb) to enable sccache/ccache.
    # Use CACHE FORCE so it propagates into thirdparty subdirectories that call project().
    foreach(_cfg DEBUG RELWITHDEBINFO)
        foreach(_lang C CXX)
            string(REPLACE "/Zi" "/Z7" _flags "${CMAKE_${_lang}_FLAGS_${_cfg}}")
            set(CMAKE_${_lang}_FLAGS_${_cfg} "${_flags}" CACHE STRING "" FORCE)
        endforeach()
    endforeach()
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
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT IOS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(NOT DEFINED PROJECT_ROOT_DIR OR NOT PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

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

option(RABITQ_ENABLE_AVX512 "Compile RaBitQ with AVX-512 support" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64" AND NOT ANDROID AND NOT IOS)
  include(CheckCCompilerFlag)

  check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
  check_c_compiler_flag("-mavx512f -mavx512bw -mavx512vl" COMPILER_SUPPORTS_AVX512)

  if(COMPILER_SUPPORTS_AVX2 OR COMPILER_SUPPORTS_AVX512)
    set(RABITQ_SUPPORTED ON)
    add_definitions(-DRABITQ_SUPPORTED=1)
    if(RABITQ_ENABLE_AVX512 AND COMPILER_SUPPORTS_AVX512)
      add_definitions(-DRABITQ_COMPILED_AVX512=1)
      set(RABITQ_ARCH_FLAG "${MATH_MARCH_FLAG_AVX512}")
    else()
      set(RABITQ_ARCH_FLAG "${MATH_MARCH_FLAG_AVX2}")
    endif()
  else()
    set(RABITQ_SUPPORTED OFF)
    add_definitions(-DRABITQ_SUPPORTED=0)
    message(STATUS "RaBitQ support disabled - compiler does not support AVX2 or AVX-512")
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
message(STATUS "RABITQ_ARCH_FLAG: ${RABITQ_ARCH_FLAG}")

# DiskAnn support (Linux x86_64 only, requires libaio)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386" AND NOT ANDROID AND NOT IOS)
  set(DISKANN_SUPPORTED ON)
  add_definitions(-DDISKANN_SUPPORTED=1)
else()
  set(DISKANN_SUPPORTED OFF)
  add_definitions(-DDISKANN_SUPPORTED=0)
  message(STATUS "DiskAnn support disabled - only supported on Linux x86_64")
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

add_custom_target(clang_tidy_deps DEPENDS zvec_proto ARROW.BUILD glog gflags Lz4.BUILD)

if(BUILD_TOOLS)
    cc_directories(tools)
endif()

git_version(GIT_SRCS_VER ${PROJECT_ROOT_DIR})
set(CPACK_PACKAGE_VERSION ${GIT_SRCS_VER})
set(CPACK_PACKAGE_NAME zvec)
include(CPack)

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
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR}
            COMPONENT python)

    # DiskAnn ships as a runtime-loaded shared module
    # (libzvec_diskann_plugin.so) that is brought online implicitly the
    # first time a DiskAnn index is created — users never call any load
    # function. The Python extension resolves the module next to _zvec.so
    # (see the $ORIGIN rpath in src/binding/python/CMakeLists.txt); the
    # module must therefore be installed alongside _zvec.so in the wheel.
    #
    # Gate on DISKANN_SUPPORTED, not on the target's existence: on unsupported
    # platforms (e.g. macOS / ARM64) the core_knn_diskann target is still
    # defined, but built from an empty stub (src/core/algorithm/CMakeLists.txt)
    # with zero exported symbols and a runtime load path compiled out
    # (#if DISKANN_SUPPORTED). Shipping that stub is pure dead weight, so it is
    # only packaged where DiskAnn is real — currently Linux x86_64 with libaio.
    if(DISKANN_SUPPORTED)
        install(TARGETS core_knn_diskann LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR}
                COMPONENT python)
    endif()
    # Bundle cppjieba's dictionary files so the `jieba` FTS tokenizer works
    # out of the box. python/zvec/__init__.py resolves this directory via
    # importlib.resources and registers it with set_default_jieba_dict_dir().
    set(ZVEC_JIEBA_DICT_SRC
        "${PROJECT_SOURCE_DIR}/thirdparty/cppjieba/cppjieba-5.6.7/dict")
    install(FILES
        "${ZVEC_JIEBA_DICT_SRC}/jieba.dict.utf8"
        "${ZVEC_JIEBA_DICT_SRC}/hmm_model.utf8"
        DESTINATION ${ZVEC_PY_INSTALL_DIR}/zvec/data/jieba_dict
        COMPONENT python)
endif()
