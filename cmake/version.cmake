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

##
## Single source of truth for the zvec version.
##
## Resolves the version once per configure run and exposes:
##   ZVEC_VERSION_MAJOR / ZVEC_VERSION_MINOR / ZVEC_VERSION_PATCH
##   ZVEC_VERSION_STRING  - full descriptive string (e.g. v0.5.1-3-gabc1234)
##   ZVEC_VERSION_NUMBER  - MAJOR.MINOR.PATCH only, for CPack/SOVERSION
##
## Resolution order:
##   1. -DOVERRIDE_GIT_DESCRIBE=vX.Y.Z  (explicit, for packagers and CI)
##   2. git describe --tags on a real git checkout
##   3. dummy v0.0.0 with a loud warning
##
## The dummy fallback is deliberately obvious: a plausible-looking version
## would make consumers trust a wrong ABI identity (see issue #621). Builds
## from source tarballs/zips (no .git) or without git installed still work,
## they just report v0.0.0 unless OVERRIDE_GIT_DESCRIBE is provided.
##

# Guard against re-resolving in every subdirectory that includes this file.
# Intentionally a normal variable, so it resets on each configure run and
# picks up changes to OVERRIDE_GIT_DESCRIBE.
if(NOT DEFINED ZVEC_VERSION_RESOLVED)
  set(ZVEC_VERSION_RESOLVED TRUE)

  set(OVERRIDE_GIT_DESCRIBE "" CACHE STRING
      "Explicit version string (vX.Y.Z[-N-g<sha>]) used instead of 'git describe'")

  set(_zvec_describe "")

  if(OVERRIDE_GIT_DESCRIBE)
    if(NOT OVERRIDE_GIT_DESCRIBE MATCHES "^v[0-9]+\\.[0-9]+\\.[0-9]+")
      message(FATAL_ERROR
          "Provided OVERRIDE_GIT_DESCRIBE '${OVERRIDE_GIT_DESCRIBE}' does not match "
          "the expected format 'vX.Y.Z' or 'vX.Y.Z-N-g<sha>'")
    endif()
    set(_zvec_describe "${OVERRIDE_GIT_DESCRIBE}")
    message(STATUS "zvec version from OVERRIDE_GIT_DESCRIBE: ${_zvec_describe}")
  else()
    # Git is optional here: tarball builds must not fail just because the
    # version cannot be determined.
    find_package(Git QUIET)

    # Require a real repository at the project root. Without this check a
    # source tree extracted inside an unrelated git repository would silently
    # inherit that repository's tags.
    if(Git_FOUND AND EXISTS "${PROJECT_ROOT_DIR}/.git")
      execute_process(
          COMMAND "${GIT_EXECUTABLE}" describe --tags --match "v*.*.*"
          WORKING_DIRECTORY "${PROJECT_ROOT_DIR}"
          RESULT_VARIABLE _zvec_git_result
          OUTPUT_VARIABLE _zvec_git_output
          ERROR_VARIABLE _zvec_git_error
          OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(_zvec_git_result EQUAL 0)
        set(_zvec_describe "${_zvec_git_output}")
      else()
        # No tags reachable (shallow clone). Keep the commit id so the build
        # stays traceable even though the version is unknown.
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
            WORKING_DIRECTORY "${PROJECT_ROOT_DIR}"
            RESULT_VARIABLE _zvec_git_result
            OUTPUT_VARIABLE _zvec_git_output
            ERROR_VARIABLE _zvec_git_error
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_zvec_git_result EQUAL 0)
          set(_zvec_describe "g${_zvec_git_output}")
        endif()
      endif()
    endif()
  endif()

  if(_zvec_describe MATCHES "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(ZVEC_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(ZVEC_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(ZVEC_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(ZVEC_VERSION_STRING "${_zvec_describe}")
  else()
    message(WARNING
        "Could not determine the zvec version from git (got '${_zvec_describe}'), "
        "likely a shallow clone, a source tarball without .git, or git is not installed. "
        "Continuing with dummy version v0.0.0. Fetch tags (git fetch --tags) or pass "
        "-DOVERRIDE_GIT_DESCRIBE=vX.Y.Z to build with a correct version.")
    set(ZVEC_VERSION_MAJOR 0)
    set(ZVEC_VERSION_MINOR 0)
    set(ZVEC_VERSION_PATCH 0)
    if(_zvec_describe)
      set(ZVEC_VERSION_STRING "v0.0.0-${_zvec_describe}")
    else()
      set(ZVEC_VERSION_STRING "v0.0.0")
    endif()
  endif()

  set(ZVEC_VERSION_NUMBER
      "${ZVEC_VERSION_MAJOR}.${ZVEC_VERSION_MINOR}.${ZVEC_VERSION_PATCH}")

  message(STATUS "zvec version: ${ZVEC_VERSION_STRING} (${ZVEC_VERSION_NUMBER})")

  # Cache as INTERNAL so every subdirectory sees the same values regardless of
  # include order, while FORCE keeps them in sync across re-configures.
  foreach(_zvec_var
      ZVEC_VERSION_MAJOR ZVEC_VERSION_MINOR ZVEC_VERSION_PATCH
      ZVEC_VERSION_STRING ZVEC_VERSION_NUMBER)
    set(${_zvec_var} "${${_zvec_var}}" CACHE INTERNAL "zvec version component" FORCE)
  endforeach()

  unset(_zvec_describe)
  unset(_zvec_git_result)
  unset(_zvec_git_output)
  unset(_zvec_git_error)
  unset(_zvec_var)
endif()
