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

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.c)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_c_api
      SRCS ${CC_SRCS} utils.c
      INCS . .. ../../src
    )

  # iOS: embed the shared library inside the .app bundle so the simulator can
  # find it at runtime, and patch the executable's rpath accordingly.
  if(IOS AND TARGET ${CC_TARGET})
    add_custom_command(TARGET ${CC_TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
              "$<TARGET_BUNDLE_DIR:${CC_TARGET}>/Frameworks"
      COMMAND ${CMAKE_COMMAND} -E copy
              "$<TARGET_FILE:zvec_c_api>"
              "$<TARGET_BUNDLE_DIR:${CC_TARGET}>/Frameworks/"
      COMMAND install_name_tool -add_rpath "@executable_path/Frameworks"
              "$<TARGET_BUNDLE_DIR:${CC_TARGET}>/${CC_TARGET}" 2>/dev/null || true
    )
  endif()
endforeach()
