# SPDX-License-Identifier: Apache-2.0
# Included by thirdparty/arrow after its normal Arrow 21 patches are applied.
# A small patch connects Gandiva's existing expression visitor to the separate
# scalar extension. Existing upstream files are never overwritten.
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
set(_SCALAR_JIT_ARROW_LINK_ARGS "-DCMAKE_PROJECT_arrow_INCLUDE=")
set(_SCALAR_JIT_LLVM_HOOK)
if(ZVEC_SCALAR_JIT_LLVM_SHARED)
    if(NOT TARGET LLVM)
        message(FATAL_ERROR "Shared scalar JIT requires LLVM's shared library target (LLVM_BUILD_LLVM_DYLIB=ON)")
    endif()
    set(_SCALAR_JIT_LLVM_LIBS LLVM)
    set(_SCALAR_JIT_LLVM_LINKAGE shared)
else()
    # Match Arrow 21's FindLLVMAlt.cmake. Keep the optimizer and ORC JIT, but
    # only pull in the native code generator instead of every LLVM backend.
    llvm_map_components_to_libnames(_SCALAR_JIT_LLVM_LIBS
        analysis bitreader core debuginfodwarf ipo linker native orcjit target passes)
    set(_scalar_jit_llvm_link_only)
    foreach(_scalar_jit_llvm_lib IN LISTS _SCALAR_JIT_LLVM_LIBS)
        if(NOT TARGET ${_scalar_jit_llvm_lib})
            message(FATAL_ERROR "Scalar JIT requires LLVM static component ${_scalar_jit_llvm_lib}; install the static libraries or set ZVEC_SCALAR_JIT_LLVM_SHARED=ON")
        endif()
        get_target_property(_scalar_jit_llvm_type ${_scalar_jit_llvm_lib} TYPE)
        if(NOT _scalar_jit_llvm_type STREQUAL "STATIC_LIBRARY")
            message(FATAL_ERROR "Scalar JIT component ${_scalar_jit_llvm_lib} is not static; set ZVEC_SCALAR_JIT_LLVM_SHARED=ON to use shared LLVM")
        endif()
        list(APPEND _scalar_jit_llvm_link_only "$<LINK_ONLY:${_scalar_jit_llvm_lib}>")
    endforeach()
    set(_SCALAR_JIT_LLVM_LIBS ${_scalar_jit_llvm_link_only})
    set(_SCALAR_JIT_LLVM_HOOK "${CMAKE_CURRENT_LIST_DIR}/llvm_link_only.cmake")
    set(_SCALAR_JIT_ARROW_LINK_ARGS
        "-DCMAKE_PROJECT_arrow_INCLUDE=${_SCALAR_JIT_LLVM_HOOK}")
    set(_SCALAR_JIT_LLVM_LINKAGE "static native components")
endif()
find_program(ZVEC_SCALAR_JIT_CLANG NAMES clang-${LLVM_VERSION_MAJOR} clang
    HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(ZVEC_SCALAR_JIT_LLVM_LINK NAMES llvm-link
    HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_package(OpenSSL REQUIRED GLOBAL)
find_package(Threads REQUIRED GLOBAL)

file(READ "${ARROW_SRC_DIR}/cpp/CMakeLists.txt" _scalar_jit_arrow_cmake)
if(NOT _scalar_jit_arrow_cmake MATCHES "set\\(ARROW_VERSION \"21\\.0\\.0\"\\)")
    message(FATAL_ERROR "Scalar JIT patch is validated against Apache Arrow 21.0.0")
endif()

set(_SCALAR_JIT_PATCH "${CMAKE_CURRENT_LIST_DIR}/scalar.patch")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_SCALAR_JIT_PATCH}")
apply_patch_once("arrow_gandiva_scalar" "${ARROW_SRC_DIR}" "${_SCALAR_JIT_PATCH}")

set(_SCALAR_JIT_INPUTS "${CMAKE_CURRENT_LIST_FILE}" "${_SCALAR_JIT_PATCH}"
    ${_SCALAR_JIT_LLVM_HOOK})
foreach(_scalar_jit_file scalar_generator.h scalar_generator.cc
                        scalar_codegen_context.h scalar_codegen_context.cc
                        scalar_predicate_abi.h)
    set(_scalar_jit_source "${CMAKE_CURRENT_LIST_DIR}/extension/${_scalar_jit_file}")
    configure_file(
        "${_scalar_jit_source}"
        "${ARROW_SRC_DIR}/cpp/src/gandiva/${_scalar_jit_file}" COPYONLY)
    list(APPEND _SCALAR_JIT_INPUTS "${_scalar_jit_source}")
endforeach()

set(_ARROW_SCALAR_JIT_ARGS
    -DARROW_GANDIVA=ON
    -DARROW_BUILD_STATIC=ON
    -DARROW_LLVM_USE_SHARED=${ZVEC_SCALAR_JIT_LLVM_SHARED}
    -DARROW_WITH_RE2=ON
    -DARROW_WITH_UTF8PROC=ON
    ${_SCALAR_JIT_ARROW_LINK_ARGS}
    "-DLLVM_ROOT=${LLVM_ROOT}"
    "-DLLVM_DIR=${LLVM_DIR}"
    "-DCLANG_EXECUTABLE=${ZVEC_SCALAR_JIT_CLANG}"
    "-DLLVM_LINK_EXECUTABLE=${ZVEC_SCALAR_JIT_LLVM_LINK}"
    "-DOPENSSL_INCLUDE_DIR=${OPENSSL_INCLUDE_DIR}"
    "-DOPENSSL_CRYPTO_LIBRARY=${OPENSSL_CRYPTO_LIBRARY}"
    "-DOPENSSL_SSL_LIBRARY=${OPENSSL_SSL_LIBRARY}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    # Helper bitcode is shipped in the artifact. Its function attributes must
    # not inherit the build machine's ISA; only runtime JIT compilation may
    # select the execution machine's CPU. Gandiva uses a separate clang
    # invocation for bitcode, independent of Arrow's CMAKE_CXX_FLAGS.
    list(APPEND _ARROW_SCALAR_JIT_ARGS
        "-DARROW_GANDIVA_PC_CXX_FLAGS=-march=x86-64")
endif()
set(LIB_GANDIVA "${EXTERNAL_LIB_DIR}/libgandiva.a")
message(STATUS "Scalar JIT: Arrow 21 static + Gandiva static, LLVM ${LLVM_PACKAGE_VERSION} ${_SCALAR_JIT_LLVM_LINKAGE}")
