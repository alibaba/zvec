# SPDX-License-Identifier: Apache-2.0
# Included after Arrow's project() only for static LLVM builds. Arrow supplies
# LLVM's headers/definitions through LLVM::LLVM_HEADERS. Its static libraries
# need their system dependencies at link time, not those dependencies' headers:
# a broad prefix such as /opt/homebrew/include can otherwise override bundled
# RE2 headers with an ABI-incompatible system version.
if(NOT ARROW_GANDIVA OR ARROW_LLVM_USE_SHARED)
    return()
endif()
function(zvec_scalar_jit_link_only_llvm)
    if(NOT TARGET gandiva_objlib OR NOT TARGET gandiva_static)
        message(FATAL_ERROR "Arrow's Gandiva static targets are unavailable")
    endif()
    # LLVM::LLVM_LIBS is a directory-scoped imported target. Adjust the
    # consumers in that directory, keeping their explicit LLVM headers intact.
    foreach(_scalar_jit_target gandiva_objlib gandiva_static)
        foreach(_scalar_jit_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_scalar_jit_libs ${_scalar_jit_target}
                ${_scalar_jit_property})
            if(_scalar_jit_libs)
                list(TRANSFORM _scalar_jit_libs REPLACE "^LLVM::LLVM_LIBS$"
                    "$<LINK_ONLY:LLVM::LLVM_LIBS>")
                set_target_properties(${_scalar_jit_target} PROPERTIES
                    ${_scalar_jit_property} "${_scalar_jit_libs}")
            endif()
        endforeach()
    endforeach()
endfunction()
cmake_language(DEFER CALL zvec_scalar_jit_link_only_llvm)
