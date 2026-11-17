# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
