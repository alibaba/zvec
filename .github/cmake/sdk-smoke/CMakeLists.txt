cmake_minimum_required(VERSION 3.16)
project(zvec_sdk_smoke LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ZVEC_SDK_DIR)
  message(FATAL_ERROR "ZVEC_SDK_DIR must point to the staged zvec-sdk package contents")
endif()

# Fail fast if the package is incomplete. The C++ header check also guards
# against install-layout regressions (headers must live at include/zvec/,
# not in a nested include/include/).
if(NOT EXISTS "${ZVEC_SDK_DIR}/include/zvec/c_api.h")
  message(FATAL_ERROR "SDK package is missing include/zvec/c_api.h")
endif()
file(GLOB _smoke_cpp_headers "${ZVEC_SDK_DIR}/include/zvec/db/*.h")
if(NOT _smoke_cpp_headers)
  message(FATAL_ERROR "SDK package is missing C++ headers under include/zvec/db/")
endif()

# CMAKE_FIND_ROOT_PATH_BOTH keeps the explicit PATHS usable when
# cross-compiling (the NDK toolchain re-roots find_library searches).
find_library(ZVEC_SDK_C_API_LIBRARY
  NAMES zvec_c_api
  PATHS "${ZVEC_SDK_DIR}/lib"
  NO_DEFAULT_PATH
  CMAKE_FIND_ROOT_PATH_BOTH
  REQUIRED)
# The all-in-one library installs as libzvec.so / libzvec.dylib on Unix and
# as zvec.dll + zvec_shared.lib on Windows.
find_library(ZVEC_SDK_LIBRARY
  NAMES zvec_shared zvec
  PATHS "${ZVEC_SDK_DIR}/lib"
  NO_DEFAULT_PATH
  CMAKE_FIND_ROOT_PATH_BOTH
  REQUIRED)

add_executable(c_smoke smoke.c)
target_include_directories(c_smoke PRIVATE "${ZVEC_SDK_DIR}/include")
target_link_libraries(c_smoke PRIVATE "${ZVEC_SDK_C_API_LIBRARY}")

add_executable(cpp_smoke smoke.cc)
target_include_directories(cpp_smoke PRIVATE "${ZVEC_SDK_DIR}/include")
target_link_libraries(cpp_smoke PRIVATE "${ZVEC_SDK_LIBRARY}")

# Let the executables find the packaged shared libraries at runtime
# (Windows locates the DLLs through PATH instead).
if(NOT WIN32)
  set_target_properties(c_smoke cpp_smoke PROPERTIES
    BUILD_RPATH "${ZVEC_SDK_DIR}/lib")
endif()

enable_testing()
add_test(NAME c_smoke COMMAND c_smoke)
add_test(NAME cpp_smoke COMMAND cpp_smoke)
# Clear external overrides so Chinese FTS exercises the embedded defaults.
set_tests_properties(cpp_smoke PROPERTIES ENVIRONMENT "ZVEC_JIEBA_DICT_DIR=")

# ---------------------------------------------------------------------------
# Package shape: the stable API only, always
# ---------------------------------------------------------------------------
# There is exactly one package shape -- 18 stable headers, libzvec and
# libzvec_c_api, libzvec.a -- so these assertions run unconditionally on every
# staged tree instead of hiding behind opt-in flags.  They check the *install
# result*, not the install rules, which is the only place a stray copy, a new
# install(DIRECTORY) or a staging script can widen the published surface without
# touching anything the header checker sees.
#
# Assert on the directories, not on a probe header: the stable closure reaches
# no engine header at all (scripts/check_public_headers.py rule 6), so the
# package must not contain an engine header *directory* either -- not even a
# partially populated one, which a single-file probe would walk straight past.
foreach(_engine_dir core ailego turbo)
  if(EXISTS "${ZVEC_SDK_DIR}/include/zvec/${_engine_dir}")
    message(FATAL_ERROR
      "the package carries an engine header directory: "
      "include/zvec/${_engine_dir}")
  endif()
endforeach()

# No engine library in any form.  libzvec whole-archives all three components and
# exports the engine API, so packaging them separately would ship two ways to
# call one implementation and twice the ABI to keep compatible; the archives are
# equally unc callable, because the headers that declare them are not installed.
# Shared *and* static suffixes, since neither may appear.  The names are specific
# enough not to match the installed libzvec.a / zvec.lib.
set(_engine_libraries)
foreach(_engine_name zvec_core zvec_ailego zvec_turbo)
  file(GLOB _found_engine_libraries
    "${ZVEC_SDK_DIR}/lib/*${_engine_name}*.so"
    "${ZVEC_SDK_DIR}/lib/*${_engine_name}*.so.*"
    "${ZVEC_SDK_DIR}/lib/*${_engine_name}*.dylib"
    "${ZVEC_SDK_DIR}/lib/*${_engine_name}*.dll"
    "${ZVEC_SDK_DIR}/lib/*${_engine_name}*.a"
    "${ZVEC_SDK_DIR}/lib/*${_engine_name}*.lib")
  list(APPEND _engine_libraries ${_found_engine_libraries})
endforeach()
if(_engine_libraries)
  message(FATAL_ERROR
    "the package carries engine libraries, which are built but never installed: "
    "${_engine_libraries}")
endif()
