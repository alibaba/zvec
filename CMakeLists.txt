cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec)
set(CC_CXX_STANDARD 17)

if(MSVC)
    set(INTTYPES_FORMAT VC7)
    add_compile_options(/FS) # handle .pdb
    add_compile_options(/EHsc) # def c++ exception behavior
    add_compile_options(/Zc:preprocessor)
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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(NOT DEFINED PROJECT_ROOT_DIR OR NOT PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if (NOT ANDROID AND AUTO_DETECT_ARCH AND HOST_ARCH MATCHES "^(x86|x64)$")
  setup_compiler_march_for_x86(MATH_MARCH_FLAG_SSE MATH_MARCH_FLAG_AVX2 MATH_MARCH_FLAG_AVX512 MATH_MARCH_FLAG_AVX512FP16)
  message(STATUS "best compiler march, sse: " ${MATH_MARCH_FLAG_SSE} ", avx2: " ${MATH_MARCH_FLAG_AVX2} ", avx512: " ${MATH_MARCH_FLAG_AVX512} ", avx512fp16: " ${MATH_MARCH_FLAG_AVX512FP16})
endif()

include_directories(${PROJECT_ROOT_DIR}/src/include)
include_directories(${PROJECT_ROOT_DIR}/src)
include_directories(${PROJECT_ROOT_DIR})

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")

option(BUILD_TOOLS "Build tools" ON)
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

option(RABITQ_ENABLE_AVX512 "Compile RaBitQ with AVX-512 support" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64" AND NOT ANDROID)
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
else()
  set(RABITQ_SUPPORTED OFF)
  add_definitions(-DRABITQ_SUPPORTED=0)
  message(STATUS "RaBitQ support disabled - only supported on Linux x86_64")
endif()
message(STATUS "RABITQ_ARCH_FLAG: ${RABITQ_ARCH_FLAG}")

option(USE_OSS_MIRROR "Use OSS mirror for faster third-party downloads" OFF)
if(DEFINED ENV{USE_OSS_MIRROR} AND NOT "$ENV{USE_OSS_MIRROR}" STREQUAL "")
  set(USE_OSS_MIRROR "$ENV{USE_OSS_MIRROR}" CACHE BOOL "Use OSS mirror for faster third-party downloads" FORCE)
endif()
message(STATUS "USE_OSS_MIRROR:${USE_OSS_MIRROR}")

cc_directory(thirdparty)
cc_directories(src)
cc_directories(tests)

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
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR})

    if(MSVC)
        install(FILES
            $<TARGET_FILE:glog>
            $<TARGET_FILE:gflags>
            $<TARGET_FILE:roaring>
            $<TARGET_FILE:libprotobuf>
            DESTINATION ${ZVEC_PY_INSTALL_DIR}
        )
    endif()
endif()
