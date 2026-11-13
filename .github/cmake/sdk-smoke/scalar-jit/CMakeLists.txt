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

# Only public installed headers and the packaged shared library are used.
find_package(Threads REQUIRED)
add_executable(scalar_jit_smoke smoke.cc)
target_include_directories(scalar_jit_smoke PRIVATE "${ZVEC_SDK_DIR}/include")
target_link_libraries(scalar_jit_smoke PRIVATE "${ZVEC_SDK_LIBRARY}" Threads::Threads)
# Keep the executable alongside the shared SDK smoke tests for existing callers.
set_target_properties(scalar_jit_smoke PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Match the package's baseline ISA and avoid requiring the build image's
  # newer libstdc++ / libgcc when the executable runs in a clean container.
  target_link_options(scalar_jit_smoke PRIVATE -static-libstdc++ -static-libgcc)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    target_compile_options(scalar_jit_smoke PRIVATE -march=x86-64)
  endif()
endif()
if(NOT WIN32)
  set_target_properties(scalar_jit_smoke PROPERTIES
    BUILD_RPATH "${ZVEC_SDK_DIR}/lib")
endif()
