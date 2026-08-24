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
# The jieba FTS dict must ship with the SDK (mirrors the Python wheel's
# zvec/data/jieba_dict bundle). Consumers register the directory via
# zvec_set_default_jieba_dict_dir() or ZVEC_JIEBA_DICT_DIR.
foreach(_smoke_dict_file jieba.dict.utf8 hmm_model.utf8)
  if(NOT EXISTS "${ZVEC_SDK_DIR}/data/jieba_dict/${_smoke_dict_file}")
    message(FATAL_ERROR "SDK package is missing data/jieba_dict/${_smoke_dict_file}")
  endif()
endforeach()

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
