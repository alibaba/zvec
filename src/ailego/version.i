// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ailego/internal/platform.h"

#ifndef AILEGO_VERSION_TO_STRING_
#define AILEGO_VERSION_TO_STRING_(x) #x
#endif

#ifndef AILEGO_VERSION_TO_STRING
#define AILEGO_VERSION_TO_STRING(x) AILEGO_VERSION_TO_STRING_(x)
#endif

/*! http://nadeausoftware.com/articles/2012/01/
 *  c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
 */
#if defined(__linux) || defined(__linux__)
#define AILEGO_VERSION_PLATFORM "Linux"
#elif defined(__FreeBSD__)
#define AILEGO_VERSION_PLATFORM "FreeBSD"
#elif defined(__NetBSD__)
#define AILEGO_VERSION_PLATFORM "NetBSD"
#elif defined(__OpenBSD__)
#define AILEGO_VERSION_PLATFORM "OpenBSD"
#elif defined(__APPLE__) || defined(__MACH__)
#define AILEGO_VERSION_PLATFORM "Darwin"
#elif defined(__CYGWIN__) && !defined(_WIN32)
#define AILEGO_VERSION_PLATFORM "Cygwin"
#elif defined(_WIN64)
#define AILEGO_VERSION_PLATFORM "Microsoft Windows (64-bit)"
#elif defined(_WIN32)
#define AILEGO_VERSION_PLATFORM "Microsoft Windows (32-bit)"
#elif defined(__sun) && defined(__SVR4)
#define AILEGO_VERSION_PLATFORM "Solaris"
#elif defined(_AIX)
#define AILEGO_VERSION_PLATFORM "AIX"
#elif defined(__hpux)
#define AILEGO_VERSION_PLATFORM "HP-UX"
#elif defined(__unix) || defined(__unix__)
#define AILEGO_VERSION_PLATFORM "Unix"
#else
#define AILEGO_VERSION_PLATFORM "Unknown Platform"
#endif

/*! http://nadeausoftware.com/articles/2012/10/
 *  c_c_tip_how_detect_compiler_name_and_version_using_compiler_predefined_macros
 */
#if defined(__NVCC__)
#define AILEGO_VERSION_COMPILER_NAME "Nvidia CUDA Compiler"
#elif defined(__clang__)
#define AILEGO_VERSION_COMPILER_NAME "Clang/LLVM"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#define AILEGO_VERSION_COMPILER_NAME "Intel ICC/ICPC"
#elif defined(__GNUC__) || defined(__GNUG__)
#define AILEGO_VERSION_COMPILER_NAME "GNU GCC/G++"
#elif defined(__HP_cc) || defined(__HP_aCC)
#define AILEGO_VERSION_COMPILER_NAME "Hewlett-Packard C/aC++"
#elif defined(__IBMC__) || defined(__IBMCPP__)
#define AILEGO_VERSION_COMPILER_NAME "IBM XL C/C++"
#elif defined(_MSC_VER)
#define AILEGO_VERSION_COMPILER_NAME "Microsoft Visual C++"
#elif defined(__PGI)
#define AILEGO_VERSION_COMPILER_NAME "Portland Group PGCC/PGCPP"
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define AILEGO_VERSION_COMPILER_NAME "Oracle Solaris Studio"
#else
#define AILEGO_VERSION_COMPILER_NAME "Unknown Compiler"
#endif

#if defined(__CUDACC_VER_MAJOR__)
#define AILEGO_VERSION_COMPILER \
  AILEGO_VERSION_COMPILER_NAME  \
  " (" AILEGO_VERSION_TO_STRING(__CUDACC_VER_MAJOR__) \
  "." AILEGO_VERSION_TO_STRING(__CUDACC_VER_MINOR__)  \
  "." AILEGO_VERSION_TO_STRING(__CUDACC_VER_BUILD__) ")"
#elif defined(__VERSION__)
#define AILEGO_VERSION_COMPILER \
  AILEGO_VERSION_COMPILER_NAME " (" __VERSION__ ")"
#elif defined(_MSC_FULL_VER)
#define AILEGO_VERSION_COMPILER \
  AILEGO_VERSION_COMPILER_NAME " (" AILEGO_VERSION_TO_STRING(_MSC_FULL_VER) ")"
#elif defined(_MSC_VER)
#define AILEGO_VERSION_COMPILER \
  AILEGO_VERSION_COMPILER_NAME " (" AILEGO_VERSION_TO_STRING(_MSC_VER) ")"
#elif defined(__PGIC__)
#define AILEGO_VERSION_COMPILER                                         \
  AILEGO_VERSION_COMPILER_NAME                                          \
  " (" AILEGO_VERSION_TO_STRING(__PGIC__) "." AILEGO_VERSION_TO_STRING( \
      __PGIC_MINOR__) "." AILEGO_VERSION_TO_STRING(__PGIC_PATCHLEVEL__) ")"
#elif defined(__xlc__)
#define AILEGO_VERSION_COMPILER AILEGO_VERSION_COMPILER_NAME " (" __xlc__ ")"
#elif defined(__SUNPRO_C)
#define AILEGO_VERSION_COMPILER \
  AILEGO_VERSION_COMPILER_NAME " (" AILEGO_VERSION_TO_STRING(__SUNPRO_C) ")"
#elif defined(__HP_cc)
#define AILEGO_VERSION_COMPILER \
  AILEGO_VERSION_COMPILER_NAME " (" AILEGO_VERSION_TO_STRING(__HP_cc) ")"
#else
#define AILEGO_VERSION_COMPILER AILEGO_VERSION_COMPILER_NAME
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define AILEGO_VERSION_PROCESSOR "x86 64-bit Processor"
#elif defined(__i386) || defined(_M_IX86)
#define AILEGO_VERSION_PROCESSOR "x86 32-bit Processor"
#elif defined(__ARM_ARCH)
#if defined(__ARM_64BIT_STATE)
#define AILEGO_VERSION_PROCESSOR "ARM 64-bit Processor"
#else
#define AILEGO_VERSION_PROCESSOR "ARM 32-bit Processor"
#endif
#elif defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
#define AILEGO_VERSION_PROCESSOR "Itanium Processor"
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
#define AILEGO_VERSION_PROCESSOR "PowerPC 64-bit Processor"
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#define AILEGO_VERSION_PROCESSOR "PowerPC 32-bit Processor"
#elif defined(__sparc)
#define AILEGO_VERSION_PROCESSOR "SPARC Processor"
#else
#define AILEGO_VERSION_PROCESSOR "Unknown Processor"
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define AILEGO_VERSION_BYTE_ORDER "  Little-endian Byte Order\n"
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define AILEGO_VERSION_BYTE_ORDER "  Big-endian Byte Order\n"
#elif __BYTE_ORDER__ == __ORDER_PDP_ENDIAN__
#define AILEGO_VERSION_BYTE_ORDER "  PDP-endian Byte Order\n"
#else
#define AILEGO_VERSION_BYTE_ORDER ""
#endif

#if defined(_DEBUG) || (!defined(__OPTIMIZE__) && !defined(NDEBUG))
#define AILEGO_VERSION_DEBUG_INFO "  Debug Information\n"
#else
#define AILEGO_VERSION_DEBUG_INFO ""
#endif

#if defined(__SANITIZE_ADDRESS__)
#define AILEGO_VERSION_ASAN "  Address Sanitizer\n"
#else
#define AILEGO_VERSION_ASAN ""
#endif

#if defined(__STDC_VERSION__)
#define AILEGO_VERSION_STDC \
  "  C Standard " AILEGO_VERSION_TO_STRING(__STDC_VERSION__) "\n"
#else
#define AILEGO_VERSION_STDC ""
#endif

#if defined(__cplusplus)
#define AILEGO_VERSION_CPLUSPLUS \
  "  C++ Standard " AILEGO_VERSION_TO_STRING(__cplusplus) "\n"
#else
#define AILEGO_VERSION_CPLUSPLUS ""
#endif

#if defined(__GXX_ABI_VERSION)
#define AILEGO_VERSION_GXX_ABI \
  "  GNU C++ ABI " AILEGO_VERSION_TO_STRING(__GXX_ABI_VERSION) "\n"
#else
#define AILEGO_VERSION_GXX_ABI ""
#endif

#if defined(__GLIBC__)
#define AILEGO_VERSION_GLIBC               \
  "  GNU glibc " AILEGO_VERSION_TO_STRING( \
      __GLIBC__) "." AILEGO_VERSION_TO_STRING(__GLIBC_MINOR__) "\n"
#else
#define AILEGO_VERSION_GLIBC ""
#endif

#if defined(WINVER)
#define AILEGO_VERSION_WINSDK \
  "  Microsoft Windows SDK " AILEGO_VERSION_TO_STRING(WINVER) "\n"
#else
#define AILEGO_VERSION_WINSDK ""
#endif

#if defined(__CLR_VER)
#define AILEGO_VERSION_CLR \
  "  Microsoft CLR " AILEGO_VERSION_TO_STRING(__CLR_VER) "\n"
#else
#define AILEGO_VERSION_CLR ""
#endif

#if defined(__LSB_VERSION__)
#define AILEGO_VERSION_LSB \
  "  Linux Standards Base " AILEGO_VERSION_TO_STRING(__LSB_VERSION__) "\n"
#else
#define AILEGO_VERSION_LSB ""
#endif

#if defined(_POSIX_VERSION)
#define AILEGO_VERSION_POSIX \
  "  POSIX Specification " AILEGO_VERSION_TO_STRING(_POSIX_VERSION) "\n"
#else
#define AILEGO_VERSION_POSIX ""
#endif

#if defined(_XOPEN_VERSION)
#define AILEGO_VERSION_XOPEN \
  "  X/Open Specification " AILEGO_VERSION_TO_STRING(_XOPEN_VERSION) "\n"
#else
#define AILEGO_VERSION_XOPEN ""
#endif

#if defined(_OPENMP)
#define AILEGO_VERSION_OPENMP \
  "  OpenMP API " AILEGO_VERSION_TO_STRING(_OPENMP) "\n"
#else
#define AILEGO_VERSION_OPENMP ""
#endif

#if defined(__ARM_NEON)
#define AILEGO_VERSION_SIMD "  Arm Neon Instruction Set\n"
#elif defined(__AVX512FP16__)
#define AILEGO_VERSION_SIMD "  AVX-512FP16 Instruction Set\n"
#elif defined(__AVX512F__)
#define AILEGO_VERSION_SIMD "  AVX-512F Instruction Set\n"
#elif defined(__AVX2__)
#define AILEGO_VERSION_SIMD "  AVX-2 Instruction Set\n"
#elif defined(__AVX__)
#define AILEGO_VERSION_SIMD "  AVX Instruction Set\n"
#elif defined(__SSE4_2__)
#define AILEGO_VERSION_SIMD "  SSE-4.2 Instruction Set\n"
#elif defined(__SSE4_1__)
#define AILEGO_VERSION_SIMD "  SSE-4.1 Instruction Set\n"
#elif defined(__SSSE3__)
#define AILEGO_VERSION_SIMD "  SSSE-3 Instruction Set\n"
#elif defined(__SSE3__)
#define AILEGO_VERSION_SIMD "  SSE-3 Instruction Set\n"
#elif defined(__SSE2__)
#define AILEGO_VERSION_SIMD "  SSE-2 Instruction Set\n"
#elif defined(__SSE__)
#define AILEGO_VERSION_SIMD "  SSE Instruction Set\n"
#elif defined(__MMX__)
#define AILEGO_VERSION_SIMD "  MMX Instruction Set\n"
#else
#define AILEGO_VERSION_SIMD ""
#endif

#if defined(PY_VERSION)
#if PY_RELEASE_LEVEL == PY_RELEASE_LEVEL_ALPHA
#define AILEGO_VERSION_PYTHON \
  "  Python API " PY_VERSION  \
  " Alpha " AILEGO_VERSION_TO_STRING(PY_RELEASE_SERIAL) "\n"
#elif PY_RELEASE_LEVEL == PY_RELEASE_LEVEL_BETA
#define AILEGO_VERSION_PYTHON \
  "  Python API " PY_VERSION  \
  " Beta " AILEGO_VERSION_TO_STRING(PY_RELEASE_SERIAL) "\n"
#elif PY_RELEASE_LEVEL == PY_RELEASE_LEVEL_GAMMA
#define AILEGO_VERSION_PYTHON \
  "  Python API " PY_VERSION  \
  " Release Candidate " AILEGO_VERSION_TO_STRING(PY_RELEASE_SERIAL) "\n"
#elif PY_RELEASE_LEVEL == PY_RELEASE_LEVEL_FINAL
#define AILEGO_VERSION_PYTHON "  Python API " PY_VERSION " Final\n"
#else
#define AILEGO_VERSION_PYTHON "  Python API " PY_VERSION "\n"
#endif
#else
#define AILEGO_VERSION_PYTHON ""
#endif

//! Gather information of compiling
#define AILEGO_VERSION_COMPILE_DETAILS(__PREFIX_INFO__)                      \
  __PREFIX_INFO__                                                            \
  "Compiled by " AILEGO_VERSION_COMPILER                                     \
  ".\n"                                                                      \
  "Compiled for " AILEGO_VERSION_PROCESSOR                                   \
  ".\n"                                                                      \
  "Compiled on " AILEGO_VERSION_PLATFORM " on " __DATE__ " " __TIME__        \
  ".\n"                                                                      \
  "Compiled with: \n"                                                        \
  "" AILEGO_VERSION_BYTE_ORDER "" AILEGO_VERSION_SIMD                        \
  "" AILEGO_VERSION_DEBUG_INFO "" AILEGO_VERSION_ASAN "" AILEGO_VERSION_STDC \
  "" AILEGO_VERSION_CPLUSPLUS "" AILEGO_VERSION_GXX_ABI                      \
  "" AILEGO_VERSION_POSIX "" AILEGO_VERSION_XOPEN "" AILEGO_VERSION_LSB      \
  "" AILEGO_VERSION_GLIBC "" AILEGO_VERSION_WINSDK "" AILEGO_VERSION_CLR     \
  "" AILEGO_VERSION_OPENMP "" AILEGO_VERSION_PYTHON "\n"
