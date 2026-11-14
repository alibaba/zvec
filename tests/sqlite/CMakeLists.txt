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

# The test host links a real libsqlite3 and dlopen()s the extension into it.
# (The extension itself needs only headers.) Override with
# -DZVEC_SQLITE_LIBRARY=<path> for a non-system SQLite, e.g. Homebrew on macOS,
# since Apple's system SQLite does not support loading extensions.
find_library(ZVEC_SQLITE_LIBRARY NAMES sqlite3 DOC "SQLite library for the test host")
if(NOT ZVEC_SQLITE_LIBRARY)
    message(WARNING "libsqlite3 not found; skipping SQLite extension tests. "
                    "Pass -DZVEC_SQLITE_LIBRARY=<path> to enable them.")
    return()
endif()

file(GLOB_RECURSE ALL_TEST_SRCS *_test.c)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_test(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_c_api
      SRCS ${CC_SRCS}
      INCS ${ZVEC_SQLITE_INCLUDE_DIR}
      DEPS zvec_sqlite
      # Absolute path of the built extension, resolved at test time.
      ARGS $<TARGET_FILE:zvec_sqlite>
    )
  # cc_test() uses the plain target_link_libraries() signature, so this call
  # must not use the PRIVATE/PUBLIC keyword form.
  target_link_libraries(${CC_TARGET} ${ZVEC_SQLITE_LIBRARY} ${CMAKE_DL_LIBS})
  # Exit code 77 = host SQLite cannot load extensions; report as skipped.
  set_tests_properties(${CC_TARGET} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
