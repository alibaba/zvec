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
# The engine headers have to compile and link against *something* in the
# package.  By default that is libzvec, which whole-archives every component and
# exports exactly what the published headers declare; a package configured with
# -DZVEC_INSTALL_ENGINE_SHARED_LIBRARIES=ON also carries the standalone
# libzvec_core, which is preferred here so that shape stays exercised.
# ZVEC_REQUIRE_CORE turns "engine headers absent" into a hard error so CI can
# pin the expected package shape.
set(_core_header "${ZVEC_SDK_DIR}/include/zvec/core/interface/index_factory.h")
if(EXISTS "${_core_header}")
  # Shared libraries only.  The package also keeps the static archives for
  # static and mobile consumers, and those are not self-contained -- rocksdb,
  # arrow and protobuf are not installed -- so letting find_library() pick up
  # libzvec_core.a would produce a link that cannot resolve.  On Windows the
  # DLL's import library is zvec_core_shared.lib and the static zvec_core.lib is
  # dropped during staging, which is why zvec_core_shared comes first.
  set(_saved_library_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib;.so")
  endif()
  find_library(ZVEC_SDK_CORE_LIBRARY
    NAMES zvec_core_shared zvec_core
    PATHS "${ZVEC_SDK_DIR}/lib"
    NO_DEFAULT_PATH
    CMAKE_FIND_ROOT_PATH_BOTH)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_saved_library_suffixes}")

  if(ZVEC_SDK_CORE_LIBRARY)
    set(_core_smoke_library "${ZVEC_SDK_CORE_LIBRARY}")
    message(STATUS "core_smoke: linking the standalone libzvec_core")
  else()
    set(_core_smoke_library "${ZVEC_SDK_LIBRARY}")
    message(STATUS "core_smoke: linking libzvec (no standalone libzvec_core)")
  endif()

  add_executable(core_smoke core_smoke.cc)
  target_include_directories(core_smoke PRIVATE "${ZVEC_SDK_DIR}/include")
  target_link_libraries(core_smoke PRIVATE "${_core_smoke_library}")
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

# The mirror image of ZVEC_REQUIRE_CORE, and the one the *default* package must
# satisfy: ZVEC_INSTALL_ENGINE_HEADERS is OFF by default, so a release that
# started shipping zvec/core/framework/** again would be widening the published
# surface silently.  Pin the trimmed shape the same way ZVEC_REQUIRE_CORE pins
# the full one.
if(ZVEC_FORBID_ENGINE_HEADERS AND EXISTS "${_core_header}")
  message(FATAL_ERROR
    "ZVEC_FORBID_ENGINE_HEADERS is set but the package carries engine headers "
    "(found ${_core_header})")
endif()

# The standalone engine shared libraries are built but, by default, not
# installed: libzvec whole-archives both and exports their public API, so a
# package carrying them as well would ship two ways to call the same code and
# twice the ABI to keep compatible.  Guard that decision so it cannot silently
# come back.  The *static* archives are intentionally kept for static and
# mobile-SDK consumers, so match shared-library suffixes only.
file(GLOB _engine_shared_libraries
  "${ZVEC_SDK_DIR}/lib/*zvec_core*.so"
  "${ZVEC_SDK_DIR}/lib/*zvec_core*.so.*"
  "${ZVEC_SDK_DIR}/lib/*zvec_core*.dylib"
  "${ZVEC_SDK_DIR}/lib/*zvec_core*.dll"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.so"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.so.*"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.dylib"
  "${ZVEC_SDK_DIR}/lib/*zvec_ailego*.dll"
)
if(_engine_shared_libraries AND ZVEC_FORBID_ENGINE_LIBRARIES)
  message(FATAL_ERROR
    "standalone engine shared libraries must not be packaged, found: "
    "${_engine_shared_libraries}")
endif()
