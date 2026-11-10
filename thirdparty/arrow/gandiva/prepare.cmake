# SPDX-License-Identifier: Apache-2.0
# Included by thirdparty/arrow after its normal Arrow 21 patches are applied.
# This is an explicit source overlay, not symbol interposition or a separate
# Arrow build. configure_file also tracks changes for incremental rebuilding.
if(ANDROID OR IOS OR WIN32 OR CMAKE_CROSSCOMPILING OR
   NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux)$")
    message(FATAL_ERROR
        "ZVEC_ENABLE_SCALAR_JIT currently requires a native 64-bit Linux/macOS build")
endif()
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _scalar_jit_arch_count)
    if(_scalar_jit_arch_count GREATER 1)
        message(FATAL_ERROR "Scalar JIT does not support universal/multi-architecture builds")
    endif()
endif()

set(LLVM_ROOT "" CACHE PATH "LLVM 20 installation used by the scalar JIT backend")
if(NOT LLVM_ROOT AND DEFINED ENV{LLVM_ROOT})
    set(LLVM_ROOT "$ENV{LLVM_ROOT}" CACHE PATH "LLVM 20 installation" FORCE)
endif()
find_package(LLVM CONFIG REQUIRED GLOBAL
    HINTS "${LLVM_ROOT}/lib/cmake/llvm" "${LLVM_ROOT}")
if(NOT LLVM_VERSION_MAJOR EQUAL 20)
    message(FATAL_ERROR "Scalar JIT is pinned to LLVM 20.x; found ${LLVM_PACKAGE_VERSION}")
endif()
if(NOT TARGET LLVM)
    message(FATAL_ERROR "Scalar JIT requires LLVM's shared library target (LLVM_BUILD_LLVM_DYLIB=ON)")
endif()
find_program(ZVEC_SCALAR_JIT_CLANG NAMES clang-${LLVM_VERSION_MAJOR} clang
    HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(ZVEC_SCALAR_JIT_LLVM_LINK NAMES llvm-link
    HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_package(OpenSSL REQUIRED GLOBAL)
find_package(Threads REQUIRED GLOBAL)

file(READ "${ARROW_SRC_DIR}/cpp/CMakeLists.txt" _scalar_jit_arrow_cmake)
if(NOT _scalar_jit_arrow_cmake MATCHES "set\\(ARROW_VERSION \"21\\.0\\.0\"\\)")
    message(FATAL_ERROR "Scalar JIT source overrides require Apache Arrow 21.0.0")
endif()
foreach(_scalar_jit_override annotator.h llvm_generator.h llvm_generator.cc
                            scalar_predicate_abi.h)
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/overrides/${_scalar_jit_override}"
        "${ARROW_SRC_DIR}/cpp/src/gandiva/${_scalar_jit_override}" COPYONLY)
endforeach()

set(_ARROW_SCALAR_JIT_ARGS
    -DARROW_GANDIVA=ON
    -DARROW_BUILD_STATIC=ON
    -DARROW_LLVM_USE_SHARED=ON
    -DARROW_WITH_RE2=ON
    -DARROW_WITH_UTF8PROC=ON
    "-DLLVM_ROOT=${LLVM_ROOT}"
    "-DLLVM_DIR=${LLVM_DIR}"
    "-DCLANG_EXECUTABLE=${ZVEC_SCALAR_JIT_CLANG}"
    "-DLLVM_LINK_EXECUTABLE=${ZVEC_SCALAR_JIT_LLVM_LINK}"
    "-DOPENSSL_INCLUDE_DIR=${OPENSSL_INCLUDE_DIR}"
    "-DOPENSSL_CRYPTO_LIBRARY=${OPENSSL_CRYPTO_LIBRARY}"
    "-DOPENSSL_SSL_LIBRARY=${OPENSSL_SSL_LIBRARY}")
set(LIB_GANDIVA "${EXTERNAL_LIB_DIR}/libgandiva.a")
message(STATUS "Scalar JIT: Arrow 21 static + Gandiva static, LLVM ${LLVM_PACKAGE_VERSION} shared")
