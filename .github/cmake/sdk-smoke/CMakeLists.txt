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
# Advanced engine headers (optional in the package)
# ---------------------------------------------------------------------------
# libzvec_core is published as its own library, so when the package carries the
# engine headers we must prove they still compile and link.  ZVEC_REQUIRE_CORE
# turns "absent" into a hard error so CI can pin the expected package shape.
set(_core_header "${ZVEC_SDK_DIR}/include/zvec/core/interface/index_factory.h")
if(EXISTS "${_core_header}")
  find_library(ZVEC_SDK_CORE_LIBRARY
    NAMES zvec_core
    PATHS "${ZVEC_SDK_DIR}/lib"
    NO_DEFAULT_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
    REQUIRED)

  add_executable(core_smoke core_smoke.cc)
  target_include_directories(core_smoke PRIVATE "${ZVEC_SDK_DIR}/include")
  target_link_libraries(core_smoke PRIVATE "${ZVEC_SDK_CORE_LIBRARY}")
  if(NOT WIN32)
    set_target_properties(core_smoke PROPERTIES
      BUILD_RPATH "${ZVEC_SDK_DIR}/lib")
  endif()

  add_test(NAME core_smoke COMMAND core_smoke)
elseif(ZVEC_REQUIRE_CORE)
  message(FATAL_ERROR
    "ZVEC_REQUIRE_CORE is set but the package has no engine headers "
    "(expected ${_core_header})")
else()
  message(STATUS "engine headers absent from the package; skipping core_smoke")
endif()

# The shared libzvec_ailego is built but deliberately not installed: ailego is
# an internal foundation library whose symbols are whole-archived into libzvec,
# and a trimmed package (ZVEC_INSTALL_ENGINE_HEADERS=OFF) ships only the headers
# the stable API needs.  Guard that decision so it cannot silently come back.
# The *static* libzvec_ailego.a is intentionally kept for static and mobile-SDK
# consumers, so match shared-library suffixes only.
file(GLOB _ailego_shared_libraries
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.so"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.so.*"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.dylib"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.dll"
)
if(_ailego_shared_libraries AND ZVEC_FORBID_AILEGO_LIBRARY)
  message(FATAL_ERROR
    "shared libzvec_ailego must not be packaged, found: "
    "${_ailego_shared_libraries}")
endif()
