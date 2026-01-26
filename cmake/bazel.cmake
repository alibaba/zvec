##
##  The following functions used by user's CMakeLists.txt:
##

##  1. Functions for C/C++
##
##  1.1. Add a subdirectory to the build
##    cc_directory(<source_dir> [binary_dir])
##
##  1.2. Add subdirectories to the build
##    cc_directories(<source_dir1> [source_dir2 ...])
##
##  1.3. Build a C/C++ static or shared library
##    cc_library(
##        NAME <name>
##        [STATIC] [SHARED] [STRICT] [ALWAYS_LINK] [EXCLUDE] [PACKED] [SRCS_NO_GLOB]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [PUBINCS public_dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [PACKED_EXCLUDES pattern1 ...]
##        [VERSION <version>]
##      )
##
##  1.4. Build a C/C++ executable program
##    cc_binary(
##        NAME <name>
##        [STRICT] [PACKED]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [VERSION <version>]
##      )
##
##  1.5. Build a C/C++ executable test program
##    cc_test(
##        NAME <name>
##        [STRICT]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [ARGS args1 ...]
##        [VERSION <version>]
##      )
##
##  1.6. Add existing test cases to a test suite
##    cc_test_suite(<suite_name> [test_name ...])
##
##  1.7. Import a C/C++ static or shared library
##    cc_import(
##        NAME <name>
##        [STATIC | SHARED] [PACKED]
##        PATH <file>
##        [INCS dir1 ...]
##        [PUBINCS public_dir1 ...]
##        [DEPS target1 ...]
##        [IMPLIB <file>]
##        [PACKED_EXCLUDES pattern1 ...]
##      )
##
##  1.8. Import a C/C++ interface library
##    cc_interface(
##        NAME <name>
##        [PACKED]
##        [INCS dir1 ...]
##        [PUBINCS public_dir1 ...]
##        [DEPS target1 ...]
##        [PACKED_EXCLUDES pattern1 ...]
##      )
##
##  1.9. Build a C/C++ executable google test program
##    cc_gtest(
##        NAME <name>
##        [STRICT]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [ARGS args1 ...]
##        [VERSION <version>]
##      )
##
##  1.10. Build a C/C++ executable google mock program
##    cc_gmock(
##        NAME <name>
##        [STRICT]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [ARGS args1 ...]
##        [VERSION <version>]
##      )
##
##  1.11. Build a C++ protobuf static or shared library
##    cc_proto_library(
##        NAME <name>
##        [STATIC] [SHARED] [STRICT] [EXCLUDE] [PACKED]
##        SRCS <file1.proto> [file2.proto ...]
##        [PROTOROOT path]
##        [CXXFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [VERSION <version>]
##        [PROTOBUF_VERSION <Protobuf version>]
##      )
##

##  2. Functions for CUDA
##
##  2.1. Add a subdirectory to the build
##    cuda_directory(<source_dir> [binary_dir])
##
##  2.2. Add subdirectories to the build
##    cuda_directories(<source_dir1> [source_dir2 ...])
##
##  2.3. Build a CUDA static or shared library
##    cuda_library(
##        NAME <name>
##        [STATIC] [SHARED] [STRICT] [ALWAYS_LINK] [EXCLUDE] [PACKED]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [PUBINCS public_dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [CUDAFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [PACKED_EXCLUDES pattern1 ...]
##        [VERSION <version>]
##      )
##
##  2.4. Build a CUDA executable program
##    cuda_binary(
##        NAME <name>
##        [STRICT] [PACKED]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [CUDAFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [VERSION <version>]
##      )
##
##  2.5. Build a CUDA executable test program
##    cuda_test(
##        NAME <name>
##        [STRICT]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [CUDAFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [ARGS args1 ...]
##        [VERSION <version>]
##      )
##
##  2.6. Add existing test cases to a test suite
##    cuda_test_suite(<suite_name> [test_name ...])
##
##  2.7. Import a C/C++/CUDA static or shared library
##    cuda_import(
##        NAME <name>
##        [STATIC | SHARED] [PACKED]
##        PATH <file>
##        [INCS dir1 ...]
##        [PUBINCS public_dir1 ...]
##        [DEPS target1 ...]
##        [IMPLIB <file>]
##        [PACKED_EXCLUDES pattern1 ...]
##      )
##
##  2.8. Import a C/C++/CUDA interface library
##    cuda_interface(
##        NAME <name>
##        [PACKED]
##        [INCS dir1 ...]
##        [PUBINCS public_dir1 ...]
##        [DEPS target1 ...]
##        [PACKED_EXCLUDES pattern1 ...]
##      )
##
##  2.9. Build a CUDA executable google test program
##    cuda_gtest(
##        NAME <name>
##        [STRICT]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [CUDAFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [ARGS args1 ...]
##        [VERSION <version>]
##      )
##
##  2.10. Build a CUDA executable google mock program
##    cuda_gmock(
##        NAME <name>
##        [STRICT]
##        SRCS <file1> [file2 ...]
##        [INCS dir1 ...]
##        [DEFS DEF1=1 ...]
##        [LIBS lib1 ...]
##        [CFLAGS flag1 ...]
##        [CXXFLAGS flag1 ...]
##        [CUDAFLAGS flag1 ...]
##        [LDFLAGS flag1 ...]
##        [DEPS target1 ...]
##        [ARGS args1 ...]
##        [VERSION <version>]
##      )
##

##  3. Utility functions
##
##  3.1. Download a git repository
##    git_repository(
##        NAME <name>
##        URL <url>
##        [TAG <tag>]
##        [PATH <local path>]
##      )
##
##  3.2. Download a hg repository
##    hg_repository(
##        NAME <name>
##        URL <url>
##        [TAG <tag>]
##        [PATH <local path>]
##      )
##
##  3.3. Download a svn repository
##    svn_repository(
##        NAME <name>
##        URL <url>
##        [REV <rev>]
##        [PATH <local path>]
##      )
##
##  3.4. Download a http archive
##    http_archive(
##        NAME <name>
##        URL <url>
##        [SHA256 <sha256 value> | SHA1 <sha1 value> | MD5 <md5 value>]
##        [PATH <local path>]
##      )
##
##  3.5. Retrieve a version string from GIT
##    git_version(
##        <result variable>
##        <repository path>
##      )
##
##  3.6. Retrieve a version string from HG
##    hg_version(
##        <result variable>
##        <repository path>
##      )
##
##  3.7. Retrieve a version string from SVN
##    svn_version(
##        <result variable>
##        <repository path>
##      )
##

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
include(CMakeParseArguments)

# Using AppleClang instead of Clang (Compiler id)
if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

# Enable unit testing
enable_testing()

# Add unittest target
if(NOT TARGET unittest)
  add_custom_target(
      unittest
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      --build-config $<CONFIGURATION>
    )
endif()

# Directories of target output
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif()

# RPATH settings
set(CMAKE_MACOSX_RPATH ON)
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_SKIP_BUILD_RPATH ON)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib64:$ORIGIN/../lib:$ORIGIN")
  else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN")
  endif()
else()
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib:@loader_path")
endif()

# Define standard installation directories
if(NOT CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()
if(NOT CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR bin)
endif()
if(NOT CMAKE_INSTALL_INCDIR)
  set(CMAKE_INSTALL_INCDIR include)
endif()
if(NOT CMAKE_INSTALL_ETCDIR)
  set(CMAKE_INSTALL_ETCDIR etc)
endif()

# Generates a compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(NOT MSVC)
  # Use color in diagnostics
  set(
      _COMPILER_FLAGS
      "$<$<C_COMPILER_ID:Clang>:-fcolor-diagnostics;-stdlib=libc++>"
      "$<$<C_COMPILER_ID:AppleClang>:-fcolor-diagnostics>"
      "$<$<C_COMPILER_ID:GNU>:-fdiagnostics-color=always>"
    )
  add_compile_options(
      "$<$<COMPILE_LANGUAGE:C>:${_COMPILER_FLAGS}>"
      "$<$<COMPILE_LANGUAGE:CXX>:${_COMPILER_FLAGS}>"
    )
  unset(_COMPILER_FLAGS)
else()
  # Replace the default compiling flags
  set(
      _COMPILER_FLAGS
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS_MINSIZEREL
    )
  foreach(COMPILER_FLAG ${_COMPILER_FLAGS})
    string(REPLACE "/MT" "/MD" ${COMPILER_FLAG} "${${COMPILER_FLAG}}")
    string(REGEX REPLACE "/W[0-9]" "" ${COMPILER_FLAG} "${${COMPILER_FLAG}}")
  endforeach()
  unset(_COMPILER_FLAGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  # Build shared library as default
  set(BUILD_SHARED_LIBS ON)
endif()

set(CMAKE_C_FLAGS_ASAN ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_ASAN ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_EXE_LINKER_FLAGS_ASAN ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
set(CMAKE_SHARED_LINKER_FLAGS_ASAN ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
set(CMAKE_STATIC_LINKER_FLAGS_ASAN ${CMAKE_STATIC_LINKER_FLAGS_DEBUG})
set(CMAKE_MODULE_LINKER_FLAGS_ASAN ${CMAKE_MODULE_LINKER_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_COVERAGE ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_COVERAGE ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
set(CMAKE_STATIC_LINKER_FLAGS_COVERAGE ${CMAKE_STATIC_LINKER_FLAGS_DEBUG})
set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE ${CMAKE_MODULE_LINKER_FLAGS_DEBUG})

# C/C++ ASAN compile flags
set(
    BAZEL_CC_ASAN_COMPILE_FLAGS
    "$<$<CONFIG:ASAN>:$<$<CXX_COMPILER_ID:Clang>:-fsanitize=address>>"
    "$<$<CONFIG:ASAN>:$<$<CXX_COMPILER_ID:AppleClang>:-fsanitize=address>>"
    "$<$<CONFIG:ASAN>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=address>>"
    "$<$<CONFIG:ASAN>:$<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>>"
  )

# C/C++ COVERAGE compile flags
set(
    BAZEL_CC_COVERAGE_COMPILE_FLAGS
    "$<$<CONFIG:COVERAGE>:$<$<CXX_COMPILER_ID:Clang>:--coverage>>"
    "$<$<CONFIG:COVERAGE>:$<$<CXX_COMPILER_ID:AppleClang>:--coverage>>"
    "$<$<CONFIG:COVERAGE>:$<$<CXX_COMPILER_ID:GNU>:--coverage>>"
  )

# C/C++ strict compile flags
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(
      BAZEL_CC_STRICT_COMPILE_FLAGS
      "$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wshadow>"
      "$<$<CXX_COMPILER_ID:AppleClang>:-Wall;-Wextra;-Wshadow>"
      "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wshadow-local;-Wno-misleading-indentation>"
      "$<$<CXX_COMPILER_ID:MSVC>:/W4>"
      ${BAZEL_CC_ASAN_COMPILE_FLAGS}
      ${BAZEL_CC_COVERAGE_COMPILE_FLAGS}
    )
else()
  set(
      BAZEL_CC_STRICT_COMPILE_FLAGS
      "$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wshadow>"
      "$<$<CXX_COMPILER_ID:AppleClang>:-Wall;-Wextra;-Wshadow>"
      "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wshadow;-Wno-misleading-indentation>"
      "$<$<CXX_COMPILER_ID:MSVC>:/W4>"
      ${BAZEL_CC_ASAN_COMPILE_FLAGS}
      ${BAZEL_CC_COVERAGE_COMPILE_FLAGS}
    )
endif()


# C/C++ strict link flags
set(
    BAZEL_CC_STRICT_LINK_FLAGS
    "$<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>"
    ${BAZEL_CC_ASAN_COMPILE_FLAGS}
    ${BAZEL_CC_COVERAGE_COMPILE_FLAGS}
  )

# C/C++ unstrict compile flags
set(
    BAZEL_CC_UNSTRICT_COMPILE_FLAGS
    "$<$<CXX_COMPILER_ID:Clang>:-Wall>"
    "$<$<CXX_COMPILER_ID:AppleClang>:-Wall>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wall>"
    "$<$<CXX_COMPILER_ID:MSVC>:/W3>"
    ${BAZEL_CC_ASAN_COMPILE_FLAGS}
    ${BAZEL_CC_COVERAGE_COMPILE_FLAGS}
  )

# C/C++ unstrict link flags
set(
    BAZEL_CC_UNSTRICT_LINK_FLAGS
    "$<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>"
    ${BAZEL_CC_ASAN_COMPILE_FLAGS}
    ${BAZEL_CC_COVERAGE_COMPILE_FLAGS}
  )

# CUDA strict compile flags
set(
    BAZEL_CUDA_STRICT_COMPILE_FLAGS
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:Clang>:-Wall;-Wextra;-Wshadow>>"
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:AppleClang>:-Wall;-Wextra;-Wshadow>>"
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-Wall;-Wextra;-Wshadow>>"
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:MSVC>:/W4>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wshadow>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:AppleClang>:-Wall;-Wextra;-Wshadow>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wshadow>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4>>"
    "$<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>"
  )

# CUDA strict link flags
set(BAZEL_CUDA_STRICT_LINK_FLAGS "")

# CUDA unstrict compile flags
set(
    BAZEL_CUDA_UNSTRICT_COMPILE_FLAGS
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:Clang>:-Wall>>"
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:AppleClang>:-Wall>>"
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-Wall>>"
    "$<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:MSVC>:/W3>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-Wall>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:AppleClang>:-Wall>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wall>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W3>>"
    "$<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>"
  )

# CUDA unstrict link flags
set(BAZEL_CUDA_UNSTRICT_LINK_FLAGS "")

## Find workspace directory
function(_find_workspace_directory _RESULT)
  # Find Workspace.cmake folder
  set(_CURRENT_WORKSPACE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  get_filename_component(
      _PARENT_WORKSPACE_DIR ${_CURRENT_WORKSPACE_DIR} DIRECTORY
    )
  while(NOT ("${_CURRENT_WORKSPACE_DIR}" STREQUAL "${_PARENT_WORKSPACE_DIR}"))
    if(EXISTS "${_CURRENT_WORKSPACE_DIR}/Workspace.cmake")
      set(${_RESULT} ${_CURRENT_WORKSPACE_DIR} PARENT_SCOPE)
      message(STATUS "Found workspace at ${${_RESULT}}")
      break()
    endif()

    # Find next parent folder
    set(_CURRENT_WORKSPACE_DIR ${_PARENT_WORKSPACE_DIR})
    get_filename_component(
        _PARENT_WORKSPACE_DIR ${_CURRENT_WORKSPACE_DIR} DIRECTORY
      )
  endwhile()
endfunction()

## Retrieve absolute paths
function(_absolute_paths _RESULT)
  foreach(FILEPATH ${ARGN})
    if(NOT IS_ABSOLUTE ${FILEPATH})
      get_filename_component(FILEPATH ${FILEPATH} ABSOLUTE)
    endif()
    list(APPEND FILEPATHS ${FILEPATH})
  endforeach()
  set(${_RESULT} "${FILEPATHS}" PARENT_SCOPE)
endfunction()

## Add both shared and static library
macro(_add_library _NAME _OPTION)
  add_library(${_NAME}_objects OBJECT ${_OPTION} ${ARGN})
  add_library(
      ${_NAME}_static STATIC ${_OPTION} $<TARGET_OBJECTS:${_NAME}_objects>
    )
  add_library(
      ${_NAME} SHARED ${_OPTION} $<TARGET_OBJECTS:${_NAME}_objects>
    )
  add_dependencies(${_NAME} ${_NAME}_static)
  if(NOT MSVC)
    set_property(TARGET ${_NAME}_static PROPERTY OUTPUT_NAME ${_NAME})
  endif()
endmacro()

## Link dependencies
function(_targets_link_dependencies _NAME)
  foreach(LIB ${ARGN})
    if(TARGET ${LIB})
      list(APPEND LIBS_DEPS ${LIB})
      list(
          APPEND LIBS_INCS 
          "$<TARGET_PROPERTY:${LIB},INTERFACE_INCLUDE_DIRECTORIES>"
        )
    endif()
  endforeach()

  if(LIBS_DEPS)
    add_dependencies(${_NAME} ${LIBS_DEPS})
    target_include_directories(${_NAME} PRIVATE "${LIBS_INCS}")
  endif()
endfunction()

## Link libraries
function(_target_link_libraries _NAME)
  function(_collect_always_link_libs LIB_LIST RESULT_VAR)
    if(NOT _COLLECT_ALWAYS_LINK_VISITED)
      set(_COLLECT_ALWAYS_LINK_VISITED "" PARENT_SCOPE)
    endif()
    
    set(LOCAL_RESULT "")
    foreach(LIB ${LIB_LIST})
      if(NOT TARGET ${LIB})
        continue()
      endif()
      
      list(FIND _COLLECT_ALWAYS_LINK_VISITED ${LIB} ALREADY_VISITED)
      if(NOT ALREADY_VISITED EQUAL -1)
        continue()
      endif()
      
      list(APPEND _COLLECT_ALWAYS_LINK_VISITED ${LIB})
      set(_COLLECT_ALWAYS_LINK_VISITED "${_COLLECT_ALWAYS_LINK_VISITED}" PARENT_SCOPE)
      
      get_target_property(ALWAYS_LINK ${LIB} ALWAYS_LINK)
      if(ALWAYS_LINK)
        list(APPEND LOCAL_RESULT ${LIB})
      endif()
      
      get_target_property(DEP_LIBS ${LIB} INTERFACE_LINK_LIBRARIES)
      if(DEP_LIBS)
        _collect_always_link_libs("${DEP_LIBS}" DEP_ALWAYS_LINK_LIBS)
        list(APPEND LOCAL_RESULT ${DEP_ALWAYS_LINK_LIBS})
      endif()
      
      get_target_property(LINK_LIBS ${LIB} LINK_LIBRARIES)
      if(LINK_LIBS)
        _collect_always_link_libs("${LINK_LIBS}" LINK_ALWAYS_LINK_LIBS)
        list(APPEND LOCAL_RESULT ${LINK_ALWAYS_LINK_LIBS})
      endif()
    endforeach()
    
    list(REMOVE_DUPLICATES LOCAL_RESULT)
    set(${RESULT_VAR} "${LOCAL_RESULT}" PARENT_SCOPE)
  endfunction()
  
  _collect_always_link_libs("${ARGN}" ALL_ALWAYS_LINK_LIBS)
  
  set(ALL_LIBS_TO_PROCESS ${ARGN})
  foreach(ALWAYS_LIB ${ALL_ALWAYS_LINK_LIBS})
    list(FIND ARGN ${ALWAYS_LIB} FOUND_INDEX)
    if(FOUND_INDEX EQUAL -1)
      list(APPEND ALL_LIBS_TO_PROCESS ${ALWAYS_LIB})
    endif()
  endforeach()
  
  list(REMOVE_DUPLICATES ALL_LIBS_TO_PROCESS)
  
  foreach(LIB ${ALL_LIBS_TO_PROCESS})
    if(NOT TARGET ${LIB})
      list(APPEND LINK_LIBS ${LIB})
      continue()
    endif()

    list(FIND ALL_ALWAYS_LINK_LIBS ${LIB} IS_ALWAYS_LINK)
    if(IS_ALWAYS_LINK EQUAL -1)
      list(APPEND LINK_LIBS ${LIB})
      continue()
    endif()

    if(NOT MSVC)
      if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        list(APPEND LINK_LIBS -Wl,--whole-archive ${LIB} -Wl,--no-whole-archive)
      else()
        list(APPEND LINK_LIBS -Wl,-force_load ${LIB})
      endif()
    else()
      # Microsoft Visual C++
      list(APPEND LINK_LIBS /WHOLEARCHIVE:$<TARGET_FILE:${LIB}>)
      get_target_property(OTHER_LINK_LIBS ${LIB} INTERFACE_LINK_LIBRARIES)
      if(OTHER_LINK_LIBS)
        foreach(OTHER_LIB ${OTHER_LINK_LIBS})
          list(FIND ALL_LIBS_TO_PROCESS ${OTHER_LIB} FOUND_INDEX)
          if(FOUND_INDEX EQUAL -1)
            list(APPEND LINK_LIBS ${OTHER_LIB})
          endif()
        endforeach()
      endif()
      list(APPEND LIBS_DEPS ${LIB})
      list(
          APPEND LIBS_INCS
          "$<TARGET_PROPERTY:${LIB},INTERFACE_INCLUDE_DIRECTORIES>"
        )
    endif()
  endforeach()

  target_link_libraries(${_NAME} ${LINK_LIBS})
  if(LIBS_DEPS)
    add_dependencies(${_NAME} ${LIBS_DEPS})
    target_include_directories(${_NAME} PRIVATE "${LIBS_INCS}")
  endif()
endfunction()

## Add a subdirectory to the build
function(cc_directory)
  add_subdirectory(${ARGN})
endfunction()

## Add subdirectories to the build
function(cc_directories)
  foreach(SRC_DIR ${ARGN})
    add_subdirectory(${SRC_DIR})
  endforeach()
endfunction()

## Set the properties of target
function(_cc_target_properties)
  cmake_parse_arguments(
      CC_ARGS "STRICT;ALWAYS_LINK" "NAME;VERSION;C_STANDARD;CXX_STANDARD"
      "INCS;PUBINCS;DEFS;LIBS;CFLAGS;CXXFLAGS;LDFLAGS;DEPS" ${ARGN}
    )

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  get_target_property(TARGET_TYPE ${CC_ARGS_NAME} TYPE)
  if(("${TARGET_TYPE}" STREQUAL "SHARED_LIBRARY") OR
      ("${TARGET_TYPE}" STREQUAL "STATIC_LIBRARY") OR
      ("${TARGET_TYPE}" STREQUAL "EXECUTABLE"))
    set(TARGET_LINKABLE TRUE)
  endif()

  if(CC_ARGS_ALWAYS_LINK)
    if(("${TARGET_TYPE}" STREQUAL "STATIC_LIBRARY") OR
        ("${TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY"))
      set_property(TARGET ${CC_ARGS_NAME} PROPERTY ALWAYS_LINK TRUE)
    endif()
  endif()

  # Set the warning level of compiling
  if(CC_ARGS_STRICT)
    target_compile_options(
        ${CC_ARGS_NAME} PRIVATE "${BAZEL_CC_STRICT_COMPILE_FLAGS}"
      )
    if(TARGET_LINKABLE)
      target_link_libraries(${CC_ARGS_NAME} "${BAZEL_CC_STRICT_LINK_FLAGS}")
    endif()
  else()
    target_compile_options(
        ${CC_ARGS_NAME} PRIVATE "${BAZEL_CC_UNSTRICT_COMPILE_FLAGS}"
      )
    if(TARGET_LINKABLE)
      target_link_libraries(${CC_ARGS_NAME} "${BAZEL_CC_UNSTRICT_LINK_FLAGS}")
    endif()
  endif()

  if(CC_ARGS_DEFS)
    target_compile_definitions(${CC_ARGS_NAME} PRIVATE "${CC_ARGS_DEFS}")
  endif()

  if(CC_ARGS_CFLAGS OR CC_ARGS_CXXFLAGS)
    target_compile_options(
        ${CC_ARGS_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:${CC_ARGS_CFLAGS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${CC_ARGS_CXXFLAGS}>"
      )
  endif()

  if(CC_ARGS_LDFLAGS)
    string(REPLACE ";" " " CC_ARGS_LDFLAGS "${CC_ARGS_LDFLAGS}")
    set_property(
        TARGET ${CC_ARGS_NAME} PROPERTY LINK_FLAGS "${CC_ARGS_LDFLAGS}"
      )
  endif()

  if(CC_ARGS_INCS)
    _absolute_paths(INC_DIRS ${CC_ARGS_INCS})
    target_include_directories(${CC_ARGS_NAME} PRIVATE "${INC_DIRS}")
  endif()

  if(BAZEL_WORKSPACE_DIR)
    target_include_directories(${CC_ARGS_NAME} PRIVATE "${BAZEL_WORKSPACE_DIR}")
  endif()

  if(CC_ARGS_PUBINCS)
    _absolute_paths(INC_DIRS ${CC_ARGS_PUBINCS})
    target_include_directories(${CC_ARGS_NAME} PUBLIC "${INC_DIRS}")
  endif()

  if(CC_ARGS_LIBS)
    if(NOT TARGET_LINKABLE)
      _targets_link_dependencies(${CC_ARGS_NAME} ${CC_ARGS_LIBS})
    else()
      if ("${TARGET_TYPE}" STREQUAL "EXECUTABLE")
        _target_link_libraries(${CC_ARGS_NAME} "${CC_ARGS_LIBS}")
      else()
        target_link_libraries(${CC_ARGS_NAME} "${CC_ARGS_LIBS}")
      endif()
    endif()
  endif()

  if(CC_ARGS_DEPS)
    add_dependencies(${CC_ARGS_NAME} "${CC_ARGS_DEPS}")
  endif()

  if(CC_ARGS_VERSION)
    set_property(
        TARGET ${CC_ARGS_NAME} PROPERTY VERSION "${CC_ARGS_VERSION}"
      )
  endif()

  if(NOT CC_C_STANDARD)
    set(CC_C_STANDARD 99)
  endif()

  if(NOT CC_CXX_STANDARD)
    set(CC_CXX_STANDARD 11)
  endif()

  set_target_properties(
      ${CC_ARGS_NAME} PROPERTIES DEFINE_SYMBOL ""
      C_STANDARD ${CC_C_STANDARD} CXX_STANDARD ${CC_CXX_STANDARD}
      C_STANDARD_REQUIRED ON C_EXTENSIONS ON
      CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endfunction()

## Build a C/C++ static or shared library
function(cc_library)
  cmake_parse_arguments(
      CC_ARGS
      "STATIC;SHARED;EXCLUDE;PACKED;SRCS_NO_GLOB"
      "NAME;VERSION"
      "SRCS;INCS;PUBINCS;DEFS;LIBS;CFLAGS;CXXFLAGS;LDFLAGS;DEPS;PACKED_EXCLUDES"
      ${ARGN}
  )

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name provided.")
  endif()

  if(CC_ARGS_SRCS_NO_GLOB)
    set(SOURCE_FILES ${CC_ARGS_SRCS})
    if(NOT SOURCE_FILES)
      message(FATAL_ERROR "No source files provided for ${CC_ARGS_NAME} (SRCS_NO_GLOB mode).")
    endif()
  else()
    set(SOURCE_FILES "")
    foreach(_src IN LISTS CC_ARGS_SRCS)
      if(IS_ABSOLUTE "${_src}" OR NOT "${_src}" MATCHES "[*?]")
        list(APPEND SOURCE_FILES "${_src}")
      else()
        file(GLOB _globbed_srcs ${_src})
        list(APPEND SOURCE_FILES ${_globbed_srcs})
      endif()
    endforeach()
    if(NOT SOURCE_FILES)
      message(FATAL_ERROR "No source files found for ${CC_ARGS_NAME} after globbing.")
    endif()
  endif()

  if(CC_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CC_ARGS_NAME}")
    list(APPEND CC_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CC_ARGS_VERSION}")
  endif()

  if(CC_ARGS_EXCLUDE)
    set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
  endif()

  if(CC_ARGS_SHARED AND CC_ARGS_STATIC)
    _add_library(${CC_ARGS_NAME} "${EXCLUDE_OPTION}" ${SOURCE_FILES})
  elseif(CC_ARGS_SHARED)
    add_library(${CC_ARGS_NAME} SHARED ${EXCLUDE_OPTION} ${SOURCE_FILES})
  elseif(CC_ARGS_STATIC)
    add_library(${CC_ARGS_NAME} STATIC ${EXCLUDE_OPTION} ${SOURCE_FILES})
  else()
    add_library(${CC_ARGS_NAME} ${EXCLUDE_OPTION} ${SOURCE_FILES})
  endif()

  if(TARGET ${CC_ARGS_NAME}_objects)
    _cc_target_properties(
        NAME "${CC_ARGS_NAME}_objects"
        INCS "${CC_ARGS_INCS};${CC_ARGS_PUBINCS}"
        DEFS "${CC_ARGS_DEFS}"
        LIBS "${CC_ARGS_LIBS}"
        CFLAGS "${CC_ARGS_CFLAGS}"
        CXXFLAGS "${CC_ARGS_CXXFLAGS}"
        LDFLAGS "${CC_ARGS_LDFLAGS}"
        DEPS "${CC_ARGS_DEPS}"
        "${CC_ARGS_UNPARSED_ARGUMENTS}"
    )
  endif()

  if(TARGET ${CC_ARGS_NAME}_static)
    _cc_target_properties(
        NAME "${CC_ARGS_NAME}_static"
        INCS "${CC_ARGS_INCS}"
        PUBINCS "${CC_ARGS_PUBINCS}"
        DEFS "${CC_ARGS_DEFS}"
        LIBS "${CC_ARGS_LIBS}"
        CFLAGS "${CC_ARGS_CFLAGS}"
        CXXFLAGS "${CC_ARGS_CXXFLAGS}"
        LDFLAGS "${CC_ARGS_LDFLAGS}"
        DEPS "${CC_ARGS_DEPS}"
        "${CC_ARGS_UNPARSED_ARGUMENTS}"
    )
    if(CC_ARGS_PACKED)
      install(
        TARGETS ${CC_ARGS_NAME}_static
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      )
    endif()
  endif()

  _cc_target_properties(
      NAME "${CC_ARGS_NAME}"
      INCS "${CC_ARGS_INCS}"
      PUBINCS "${CC_ARGS_PUBINCS}"
      DEFS "${CC_ARGS_DEFS}"
      LIBS "${CC_ARGS_LIBS}"
      CFLAGS "${CC_ARGS_CFLAGS}"
      CXXFLAGS "${CC_ARGS_CXXFLAGS}"
      LDFLAGS "${CC_ARGS_LDFLAGS}"
      DEPS "${CC_ARGS_DEPS}"
      VERSION "${CC_ARGS_VERSION}"
      "${CC_ARGS_UNPARSED_ARGUMENTS}"
  )
  if(CC_ARGS_PACKED)
    install(
        TARGETS ${CC_ARGS_NAME}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
    if(CC_ARGS_PUBINCS)
      foreach(PACKED_EXCLUDE ${CC_ARGS_PACKED_EXCLUDES})
        list(APPEND PATTERN_EXCLUDES "PATTERN;${PACKED_EXCLUDE};EXCLUDE")
      endforeach()
      install(
          DIRECTORY ${CC_ARGS_PUBINCS} DESTINATION ${CMAKE_INSTALL_INCDIR}
          FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hxx"
          ${PATTERN_EXCLUDES}
      )
    endif()
  endif()
endfunction()

## Build a C/C++ executable program
function(cc_binary)
  cmake_parse_arguments(
      CC_ARGS "PACKED" "NAME;VERSION"
     "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;LDFLAGS;DEPS" ${ARGN}
    )

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CC_ARGS_SRCS ${CC_ARGS_SRCS})
  if(NOT CC_ARGS_SRCS)
    message(FATAL_ERROR "No source files found of ${CC_ARGS_NAME}.")
  endif()

  if(CC_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CC_ARGS_NAME}")
    list(APPEND CC_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CC_ARGS_VERSION}")
  endif()
  add_executable(${CC_ARGS_NAME} ${CC_ARGS_SRCS})

  if(CC_ARGS_PACKED)
    install(
        TARGETS ${CC_ARGS_NAME} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      )
  endif()

  _cc_target_properties(
      NAME "${CC_ARGS_NAME}"
      INCS "${CC_ARGS_INCS}"
      DEFS "${CC_ARGS_DEFS}"
      LIBS "${CC_ARGS_LIBS}"
      CFLAGS "${CC_ARGS_CFLAGS}"
      CXXFLAGS "${CC_ARGS_CXXFLAGS}"
      LDFLAGS "${CC_ARGS_LDFLAGS}"
      DEPS "${CC_ARGS_DEPS}"
      VERSION "${CC_ARGS_VERSION}"
      "${CC_ARGS_UNPARSED_ARGUMENTS}"
    )
endfunction()

## Build a C/C++ executable test program
function(cc_test)
  cmake_parse_arguments(
      CC_ARGS "" "NAME;VERSION"
      "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;LDFLAGS;DEPS;ARGS" ${ARGN}
    )

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CC_ARGS_SRCS ${CC_ARGS_SRCS})
  if(NOT CC_ARGS_SRCS)
    message(FATAL_ERROR "No source files found of ${CC_ARGS_NAME}.")
  endif()

  if(CC_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CC_ARGS_NAME}")
    list(APPEND CC_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CC_ARGS_VERSION}")
  endif()
  add_executable(${CC_ARGS_NAME} EXCLUDE_FROM_ALL ${CC_ARGS_SRCS})

  _cc_target_properties(
      NAME "${CC_ARGS_NAME}"
      INCS "${CC_ARGS_INCS}"
      DEFS "${CC_ARGS_DEFS}"
      LIBS "${CC_ARGS_LIBS}"
      CFLAGS "${CC_ARGS_CFLAGS}"
      CXXFLAGS "${CC_ARGS_CXXFLAGS}"
      LDFLAGS "${CC_ARGS_LDFLAGS}"
      DEPS "${CC_ARGS_DEPS}"
      "${CC_ARGS_UNPARSED_ARGUMENTS}"
    )
  add_dependencies(unittest ${CC_ARGS_NAME})
  add_custom_target(
      unittest.${CC_ARGS_NAME}
      COMMAND $<TARGET_FILE:${CC_ARGS_NAME}> "${CC_ARGS_ARGS}"
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      DEPENDS ${CC_ARGS_NAME}
    )
  add_test(
      NAME ${CC_ARGS_NAME}
      COMMAND $<TARGET_FILE:${CC_ARGS_NAME}> "${CC_ARGS_ARGS}"
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
endfunction()

## Add existing test cases to a test suite
function(cc_test_suite _NAME)
  if(NOT TARGET unittest.${_NAME})
    add_custom_target(unittest.${_NAME} COMMAND "")
  endif()
  foreach(TEST_TARGET ${ARGN})
    list(APPEND TEST_TARGETS unittest.${TEST_TARGET})
  endforeach()
  if(TEST_TARGETS)
    add_dependencies(unittest.${_NAME} ${TEST_TARGETS})
  endif()
endfunction()

## Import a C/C++ static or shared library
function(cc_import)
  cmake_parse_arguments(
      CC_ARGS "STATIC;SHARED;PACKED"
      "NAME;PATH;IMPLIB" "INCS;PUBINCS;DEPS;PACKED_EXCLUDES" ${ARGN}
    )

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CC_ARGS_PATH ${CC_ARGS_PATH})
  if(NOT CC_ARGS_PATH)
    message(FATAL_ERROR "No imported target file found of ${CC_ARGS_NAME}.")
  endif()
  if(MSVC AND CC_ARGS_SHARED AND NOT CC_ARGS_IMPLIB)
    string(REGEX REPLACE
        ".[Dd][Ll][Ll]$" ".lib" CC_ARGS_IMPLIB ${CC_ARGS_PATH}
      )
  endif()

  if(CC_ARGS_SHARED)
    add_library(${CC_ARGS_NAME} SHARED IMPORTED GLOBAL)
  elseif(CC_ARGS_STATIC)
    add_library(${CC_ARGS_NAME} STATIC IMPORTED GLOBAL)
  else()
    add_library(${CC_ARGS_NAME} UNKNOWN IMPORTED GLOBAL)
  endif()

  set_property(
      TARGET ${CC_ARGS_NAME} PROPERTY IMPORTED_LOCATION ${CC_ARGS_PATH}
    )
  if(MSVC AND CC_ARGS_SHARED)
    set_property(
        TARGET ${CC_ARGS_NAME} PROPERTY IMPORTED_IMPLIB ${CC_ARGS_IMPLIB}
      )
  endif()

  if(CC_ARGS_INCS)
    _absolute_paths(INC_DIRS ${CC_ARGS_INCS})
    foreach(INC_DIR ${INC_DIRS})
      set_property(
          TARGET ${CC_ARGS_NAME} APPEND PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "${INC_DIR}"
        )
    endforeach()
  endif()

  if(CC_ARGS_PUBINCS)
    _absolute_paths(INC_DIRS ${CC_ARGS_PUBINCS})
    foreach(INC_DIR ${INC_DIRS})
      set_property(
          TARGET ${CC_ARGS_NAME} APPEND PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "${INC_DIR}"
        )
    endforeach()
  endif()

  if(CC_ARGS_DEPS)
    add_dependencies(${CC_ARGS_NAME} "${CC_ARGS_DEPS}")
  endif()

  if(CC_ARGS_PACKED)
    install(
        TARGETS ${CC_ARGS_NAME}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      )
    if(CC_ARGS_PUBINCS)
      foreach(PACKED_EXCLUDE ${CC_ARGS_PACKED_EXCLUDES})
        list(APPEND PATTERN_EXCLUDES "PATTERN;${PACKED_EXCLUDE};EXCLUDE")
      endforeach()
      install(
          DIRECTORY ${CC_ARGS_PUBINCS} DESTINATION ${CMAKE_INSTALL_INCDIR}
          FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hxx"
          ${PATTERN_EXCLUDES}
        )
    endif()
  endif()
endfunction()

## Import a C/C++ interface library
function(cc_interface)
  cmake_parse_arguments(
      CC_ARGS "PACKED" "NAME" "INCS;PUBINCS;DEPS;PACKED_EXCLUDES" ${ARGN}
    )

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  add_library(${CC_ARGS_NAME} INTERFACE GLOBAL)
  if(CC_ARGS_INCS)
    _absolute_paths(INC_DIRS ${CC_ARGS_INCS})
    target_include_directories(${CC_ARGS_NAME} INTERFACE "${INC_DIRS}")
  endif()

  if(CC_ARGS_PUBINCS)
    _absolute_paths(INC_DIRS ${CC_ARGS_PUBINCS})
    target_include_directories(${CC_ARGS_NAME} INTERFACE "${INC_DIRS}")
  endif()

  if(CC_ARGS_DEPS)
    add_dependencies(${CC_ARGS_NAME} "${CC_ARGS_DEPS}")
  endif()

  if(CC_ARGS_PACKED AND CC_ARGS_PUBINCS)
    foreach(PACKED_EXCLUDE ${CC_ARGS_PACKED_EXCLUDES})
      list(APPEND PATTERN_EXCLUDES "PATTERN;${PACKED_EXCLUDE};EXCLUDE")
    endforeach()
    install(
        DIRECTORY ${CC_ARGS_PUBINCS} DESTINATION ${CMAKE_INSTALL_INCDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hxx"
        ${PATTERN_EXCLUDES}
      )
  endif()
endfunction()

## Find gtest library
function(_find_gtest)
  if(DEFINED FIND_GTEST_LIBS AND DEFINED FIND_GTEST_INCS)
    return()
  endif()

  if(NOT TARGET gtest OR NOT TARGET gtest_main)
    # Find gtest using 'find_package'
    find_package(GTest REQUIRED)
    set(
        FIND_GTEST_INCS "${GTEST_INCLUDE_DIRS}"
        CACHE STRING "GTest includes"
      )
    set(
        FIND_GTEST_LIBS "${GTEST_BOTH_LIBRARIES}"
        CACHE STRING "GTest libraries"
      )
  else()
    # Find gtest using target names
    set(FIND_GTEST_INCS "" CACHE STRING "GTest includes")
    set(FIND_GTEST_LIBS "gtest;gtest_main" CACHE STRING "GTest libraries")
  endif()
endfunction()

## Build a C/C++ executable google test program
function(cc_gtest)
  cmake_parse_arguments(
    CC_ARGS "" "NAME;VERSION"
    "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;LDFLAGS;DEPS;ARGS" ${ARGN}
  )
  _find_gtest()
  cc_test(
      NAME "${CC_ARGS_NAME}"
      VERSION "${CC_ARGS_VERSION}"
      SRCS "${CC_ARGS_SRCS}"
      INCS "${CC_ARGS_INCS};${FIND_GTEST_INCS}"
      DEFS "${CC_ARGS_DEFS}"
      LIBS "${CC_ARGS_LIBS};${FIND_GTEST_LIBS}"
      CFLAGS "${CC_ARGS_CFLAGS}"
      CXXFLAGS "${CC_ARGS_CXXFLAGS}"
      LDFLAGS "${CC_ARGS_LDFLAGS}"
      DEPS "${CC_ARGS_DEPS}"
      ARGS "${CC_ARGS_ARGS}"
    )
endfunction()

## Find gmock library
function(_find_gmock)
  if(DEFINED FIND_GMOCK_LIBS AND DEFINED FIND_GMOCK_INCS)
    return()
  endif()

  if(NOT TARGET gmock OR NOT TARGET gmock_main)
    # Find gmock/gtest using 'find_package'
    find_package(GMock REQUIRED)
    find_package(GTest REQUIRED)
    set(
        FIND_GMOCK_INCS "${GMOCK_INCLUDE_DIRS};${GTEST_INCLUDE_DIRS}"
        CACHE STRING "GMock includes"
      )
    set(
        FIND_GMOCK_LIBS "${GMOCK_BOTH_LIBRARIES};${GTEST_LIBRARIES}"
        CACHE STRING "GMock libraries"
      )
  else()
    # Find gmock using target names
    set(FIND_GMOCK_INCS "" CACHE STRING "GMock includes")
    set(FIND_GMOCK_LIBS "gmock;gmock_main" CACHE STRING "GMock libraries")
  endif()
endfunction()

## Build a C/C++ executable google mock program
function(cc_gmock)
  cmake_parse_arguments(
    CC_ARGS "" "NAME;VERSION"
    "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;LDFLAGS;DEPS;ARGS" ${ARGN}
  )
  _find_gmock()
  cc_test(
      NAME "${CC_ARGS_NAME}"
      VERSION "${CC_ARGS_VERSION}"
      SRCS "${CC_ARGS_SRCS}"
      INCS "${CC_ARGS_INCS};${FIND_GMOCK_INCS}"
      DEFS "${CC_ARGS_DEFS}"
      LIBS "${CC_ARGS_LIBS};${FIND_GMOCK_LIBS}"
      CFLAGS "${CC_ARGS_CFLAGS}"
      CXXFLAGS "${CC_ARGS_CXXFLAGS}"
      LDFLAGS "${CC_ARGS_LDFLAGS}"
      DEPS "${CC_ARGS_DEPS}"
      ARGS "${CC_ARGS_ARGS}"
    )
endfunction()

## Find protobuf library
function(_find_protobuf _VERSION)
  if(DEFINED CC_PROTOBUF_PROTOC_${_VERSION})
    return()
  endif()

  # Find protobuf using 'find_package'
  if(NOT TARGET protoc OR NOT TARGET libprotobuf)
    find_package(Protobuf ${_VERSION} REQUIRED)
    set(
        CC_PROTOBUF_PROTOC_${_VERSION}
        "${PROTOBUF_PROTOC_EXECUTABLE}" CACHE PATH "Protobuf compiler"
      )
    set(
        CC_PROTOBUF_INCS_${_VERSION}
        "${PROTOBUF_INCLUDE_DIRS}" CACHE STRING "Protobuf includes"
      )
    set(
        CC_PROTOBUF_LIBS_${_VERSION}
        "${PROTOBUF_LIBRARIES}" CACHE STRING "Protobuf libraries"
      )
    return()
  endif()

  # Find protobuf using target names
  get_target_property(protoc_VERSION protoc VERSION)
  get_target_property(libprotobuf_VERSION libprotobuf VERSION)
  if(_VERSION)
    if(${protoc_VERSION} VERSION_LESS ${_VERSION})
      message(
          FATAL_ERROR
          "The 'protoc' version is ${protoc_VERSION}, less than ${_VERSION}."
        )
    endif()
    if(${libprotobuf_VERSION} VERSION_LESS ${_VERSION})
      message(
          FATAL_ERROR
          "The 'libprotobuf' version is ${libprotobuf_VERSION}, "
          "less than ${_VERSION}."
        )
    endif()
  endif()

  message(STATUS "Found binary 'protoc ${protoc_VERSION}'")
  message(STATUS "Found library 'libprotobuf ${libprotobuf_VERSION}'")
  set(
      CC_PROTOBUF_PROTOC_${_VERSION}
      "$<TARGET_FILE:protoc>" CACHE PATH "Protobuf compiler"
    )
  get_target_property(protoc_SOURCE_DIR protoc SOURCE_DIR)
  get_filename_component(protoc_INCLUDE_DIR ${protoc_SOURCE_DIR}/../src ABSOLUTE)
  set(
      CC_PROTOBUF_INCS_${_VERSION}
      "${protoc_INCLUDE_DIR}" CACHE STRING "Protobuf includes"
    )
  set(
      CC_PROTOBUF_LIBS_${_VERSION} libprotobuf CACHE STRING "Protobuf libraries"
    )
endfunction()

## Build a C++ protobuf static or shared library
function(cc_proto_library)
  cmake_parse_arguments(
      CC_ARGS "STATIC;SHARED;EXCLUDE;PACKED"
      "NAME;VERSION;PROTOROOT;PROTOBUF_VERSION"
      "SRCS;CXXFLAGS;LDFLAGS;DEPS" ${ARGN}
    )

  _find_protobuf("${CC_ARGS_PROTOBUF_VERSION}")
  set(CC_PROTOBUF_PROTOC ${CC_PROTOBUF_PROTOC_${CC_ARGS_PROTOBUF_VERSION}})
  set(CC_PROTOBUF_INCS ${CC_PROTOBUF_INCS_${CC_ARGS_PROTOBUF_VERSION}})
  set(CC_PROTOBUF_LIBS ${CC_PROTOBUF_LIBS_${CC_ARGS_PROTOBUF_VERSION}})

  if(NOT CC_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CC_ARGS_SRCS ${CC_ARGS_SRCS})
  if(NOT CC_ARGS_SRCS)
    message(FATAL_ERROR "No source files found of ${CC_ARGS_NAME}.")
  endif()

  if(CC_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CC_ARGS_NAME}")
    list(APPEND CC_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CC_ARGS_VERSION}")
  endif()

  if(CC_ARGS_EXCLUDE)
    set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
  endif()

  set(PROTO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
  if(CC_ARGS_PROTOROOT)
    get_filename_component(PROTO_ROOT ${CC_ARGS_PROTOROOT} ABSOLUTE)
  endif()

  # Compile proto files to C++ sources
  set(CPP_OUTPATH "${CMAKE_CURRENT_BINARY_DIR}")
  foreach(PROTO_FILE ${CC_ARGS_SRCS})
    get_filename_component(PROTO_FILE ${PROTO_FILE} ABSOLUTE)

    if(NOT ${PROTO_FILE} MATCHES "\\.proto$$")
      message(FATAL_ERROR "Unrecognized proto file ${PROTOFILE}")
    endif()
    if(NOT ${PROTO_FILE} MATCHES "^${PROTO_ROOT}")
      message(FATAL_ERROR "'${PROTO_FILE}' NOT IN '${PROTO_ROOT}'")
    endif()

    string(
        REGEX REPLACE "^${PROTO_ROOT}(/?)" "" ROOT_CLEANED_FILE ${PROTO_FILE}
      )
    string(REGEX REPLACE "\\.proto$$" "" EXT_CLEANED_FILE ${ROOT_CLEANED_FILE})
    set(CPP_FILE "${CPP_OUTPATH}/${EXT_CLEANED_FILE}.pb.cc")
    set(HDR_FILE "${CPP_OUTPATH}/${EXT_CLEANED_FILE}.pb.h")
    set(INJ_FILE "${CPP_OUTPATH}/${EXT_CLEANED_FILE}.pb.cmake")
    file(RELATIVE_PATH REL_CPP_FILE ${CMAKE_BINARY_DIR} ${CPP_FILE})

    set(INJECTED_SCRIPT
        "foreach(SRC ${EXT_CLEANED_FILE}.pb.cc ${EXT_CLEANED_FILE}.pb.h)\n"
        "  file(READ \$\{SRC\} SRC_CODE)\n"
        "  file(REMOVE \$\{SRC\})\n"
        "  file(APPEND \$\{SRC\} \"#ifdef __GNUC__\\n\")\n"
        "  file(APPEND \$\{SRC\} \"#pragma GCC diagnostic push\\n\")\n"
        "  file(APPEND \$\{SRC\} \"#pragma GCC diagnostic ignored \\\"-Wshadow\\\"\\n\")\n"
        "  file(APPEND \$\{SRC\} \"#pragma GCC diagnostic ignored \\\"-Wunused-parameter\\\"\\n\")\n"
        "  file(APPEND \$\{SRC\} \"#endif\\n\\n\")\n"
        "  file(APPEND \$\{SRC\} \"\$\{SRC_CODE\}\")\n"
        "  file(APPEND \$\{SRC\} \"\\n#ifdef __GNUC__\\n\")\n"
        "  file(APPEND \$\{SRC\} \"#pragma GCC diagnostic pop\\n\")\n"
        "  file(APPEND \$\{SRC\} \"#endif\\n\")\n"
        "endforeach()\n"
      )
    file(WRITE "${INJ_FILE}" ${INJECTED_SCRIPT})

    add_custom_command(
        OUTPUT "${CPP_FILE}" "${HDR_FILE}"
        # COMMAND ${CMAKE_COMMAND} -E make_directory ${CPP_OUTPATH}
        COMMAND ${CC_PROTOBUF_PROTOC}
        --cpp_out "${CPP_OUTPATH}" --python_out "${CPP_OUTPATH}"
        --proto_path "${PROTO_ROOT}" --proto_path "${CC_PROTOBUF_INCS}" "${PROTO_FILE}"

        COMMAND ${CMAKE_COMMAND} -P "${INJ_FILE}"
        DEPENDS "${PROTO_FILE}"
        COMMENT "Generating CXX source ${REL_CPP_FILE}"
        VERBATIM
      )
    list(APPEND CC_SRCS "${CPP_FILE}" "${HDR_FILE}")
  endforeach()

  # Compile C++ sources
  if(CC_ARGS_SHARED AND CC_ARGS_STATIC)
    _add_library(${CC_ARGS_NAME} "${EXCLUDE_OPTION}" "${CC_SRCS}")
  elseif(CC_ARGS_SHARED)
    add_library(${CC_ARGS_NAME} SHARED ${EXCLUDE_OPTION} ${CC_SRCS})
  elseif(CC_ARGS_STATIC)
    add_library(${CC_ARGS_NAME} STATIC ${EXCLUDE_OPTION} ${CC_SRCS})
  else()
    add_library(${CC_ARGS_NAME} ${EXCLUDE_OPTION} ${CC_SRCS})
  endif()

  if(TARGET ${CC_ARGS_NAME}_objects)
    _cc_target_properties(
        NAME "${CC_ARGS_NAME}_objects"
        INCS "${CPP_OUTPATH};${CC_PROTOBUF_INCS}"
        LIBS "${CC_PROTOBUF_LIBS}"
        CXXFLAGS "${CC_ARGS_CXXFLAGS}"
        LDFLAGS "${CC_ARGS_LDFLAGS}"
        DEPS "${CC_ARGS_DEPS}"
        "${CC_ARGS_UNPARSED_ARGUMENTS}"
      )
  endif()

  if(TARGET ${CC_ARGS_NAME}_static)
    _cc_target_properties(
        NAME "${CC_ARGS_NAME}_static"
        PUBINCS "${CPP_OUTPATH};${CC_PROTOBUF_INCS}"
        LIBS "${CC_PROTOBUF_LIBS}"
        CXXFLAGS "${CC_ARGS_CXXFLAGS}"
        LDFLAGS "${CC_ARGS_LDFLAGS}"
        DEPS "${CC_ARGS_DEPS}"
        "${CC_ARGS_UNPARSED_ARGUMENTS}"
      )
    if(CC_ARGS_PACKED)
      install(
          TARGETS ${CC_ARGS_NAME}_static
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        )
    endif()
  endif()

  _cc_target_properties(
      NAME "${CC_ARGS_NAME}"
      PUBINCS "${CPP_OUTPATH};${CC_PROTOBUF_INCS}"
      LIBS "${CC_PROTOBUF_LIBS}"
      CXXFLAGS "${CC_ARGS_CXXFLAGS}"
      LDFLAGS "${CC_ARGS_LDFLAGS}"
      DEPS "${CC_ARGS_DEPS}"
      VERSION "${CC_ARGS_VERSION}"
      "${CC_ARGS_UNPARSED_ARGUMENTS}"
    )
  if(CC_ARGS_PACKED)
    install(
        TARGETS ${CC_ARGS_NAME}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      )
  endif()
endfunction()

## Add a subdirectory to the build
function(cuda_directory)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()
  cc_directory(${ARGN})
endfunction()

## Add subdirectories to the build
function(cuda_directories)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()
  cc_directories(${ARGN})
endfunction()

## Set the properties of cuda target
function(_cuda_target_properties)
  cmake_parse_arguments(
      CUDA_ARGS "STRICT;ALWAYS_LINK" "NAME;VERSION;C_STANDARD;CXX_STANDARD"
      "INCS;PUBINCS;DEFS;LIBS;CFLAGS;CXXFLAGS;CUDAFLAGS;LDFLAGS;DEPS" ${ARGN}
    )

  if(NOT CUDA_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  get_target_property(TARGET_TYPE ${CUDA_ARGS_NAME} TYPE)
  if(("${TARGET_TYPE}" STREQUAL "SHARED_LIBRARY") OR
      ("${TARGET_TYPE}" STREQUAL "STATIC_LIBRARY") OR
      ("${TARGET_TYPE}" STREQUAL "EXECUTABLE"))
    set(TARGET_LINKABLE TRUE)
  endif()

  if(CUDA_ARGS_ALWAYS_LINK)
    if(("${TARGET_TYPE}" STREQUAL "STATIC_LIBRARY") OR
        ("${TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY"))
      set_property(TARGET ${CUDA_ARGS_NAME} PROPERTY ALWAYS_LINK TRUE)
    endif()
  endif()

  # Set the warning level of compiling
  if(CUDA_ARGS_STRICT)
    target_compile_options(
        ${CUDA_ARGS_NAME} PRIVATE "${BAZEL_CUDA_STRICT_COMPILE_FLAGS}"
      )
    if(TARGET_LINKABLE)
      target_link_libraries(
          ${CUDA_ARGS_NAME} "${BAZEL_CUDA_STRICT_LINK_FLAGS}"
        )
    endif()
  else()
    target_compile_options(
        ${CUDA_ARGS_NAME} PRIVATE "${BAZEL_CUDA_UNSTRICT_COMPILE_FLAGS}"
      )
    if(TARGET_LINKABLE)
      target_link_libraries(
          ${CUDA_ARGS_NAME} "${BAZEL_CUDA_UNSTRICT_LINK_FLAGS}"
        )
    endif()
  endif()

  target_compile_options(
      ${CUDA_ARGS_NAME} PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-ccbin=${CMAKE_CXX_COMPILER}>"
    )

  if(CUDA_ARGS_DEFS)
    target_compile_definitions(${CUDA_ARGS_NAME} PRIVATE "${CUDA_ARGS_DEFS}")
  endif()

  if(CUDA_ARGS_CFLAGS OR CUDA_ARGS_CXXFLAGS OR CUDA_ARGS_CUDAFLAGS)
    target_compile_options(
        ${CUDA_ARGS_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:${CUDA_ARGS_CFLAGS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${CUDA_ARGS_CXXFLAGS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_ARGS_CUDAFLAGS}>"
      )
  endif()

  if(CUDA_ARGS_LDFLAGS)
    string(REPLACE ";" " " CUDA_ARGS_LDFLAGS "${CUDA_ARGS_LDFLAGS}")
    set_property(
        TARGET ${CUDA_ARGS_NAME} PROPERTY LINK_FLAGS "${CUDA_ARGS_LDFLAGS}"
      )
  endif()

  if(CUDA_ARGS_INCS)
    _absolute_paths(INC_DIRS ${CUDA_ARGS_INCS})
    target_include_directories(${CUDA_ARGS_NAME} PRIVATE "${INC_DIRS}")
  endif()

  target_include_directories(
      ${CUDA_ARGS_NAME} PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )

  if(BAZEL_WORKSPACE_DIR)
    target_include_directories(
        ${CUDA_ARGS_NAME} PRIVATE "${BAZEL_WORKSPACE_DIR}"
      )
  endif()

  if(CUDA_ARGS_PUBINCS)
    _absolute_paths(INC_DIRS ${CUDA_ARGS_PUBINCS})
    target_include_directories(${CUDA_ARGS_NAME} PUBLIC "${INC_DIRS}")
  endif()

  if(CUDA_ARGS_LIBS)
    if(NOT TARGET_LINKABLE)
      _targets_link_dependencies(${CUDA_ARGS_NAME} ${CUDA_ARGS_LIBS})
    else()
      if ("${TARGET_TYPE}" STREQUAL "EXECUTABLE")
        _target_link_libraries(${CUDA_ARGS_NAME} "${CUDA_ARGS_LIBS}")
      else()
        target_link_libraries(${CUDA_ARGS_NAME} "${CUDA_ARGS_LIBS}")
      endif()
    endif()
  endif()

  if(CUDA_ARGS_DEPS)
    add_dependencies(${CUDA_ARGS_NAME} "${CUDA_ARGS_DEPS}")
  endif()

  if(CUDA_ARGS_VERSION)
    set_property(
        TARGET ${CUDA_ARGS_NAME} PROPERTY VERSION "${CUDA_ARGS_VERSION}"
      )
  endif()

  if(NOT CUDA_C_STANDARD)
    set(CUDA_C_STANDARD 99)
  endif()

  if(NOT CUDA_CXX_STANDARD)
    set(CUDA_CXX_STANDARD 11)
  endif()

  set_target_properties(
      ${CUDA_ARGS_NAME} PROPERTIES DEFINE_SYMBOL ""
      C_STANDARD ${CUDA_C_STANDARD} CXX_STANDARD ${CUDA_CXX_STANDARD}
      C_STANDARD_REQUIRED ON C_EXTENSIONS ON
      CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      CUDA_STANDARD 11 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
      WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endfunction()

## Build a CUDA static or shared library
function(cuda_library)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()

  cmake_parse_arguments(
      CUDA_ARGS "STATIC;SHARED;EXCLUDE;PACKED" "NAME;VERSION"
      "SRCS;INCS;PUBINCS;DEFS;LIBS;CFLAGS;CXXFLAGS;CUDAFLAGS;LDFLAGS;DEPS;PACKED_EXCS"
      ${ARGN}
    )

  if(NOT CUDA_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CUDA_ARGS_SRCS ${CUDA_ARGS_SRCS})
  if(NOT CUDA_ARGS_SRCS)
    message(FATAL_ERROR "No source files found of ${CUDA_ARGS_NAME}.")
  endif()

  if(CUDA_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CUDA_ARGS_NAME}")
    list(APPEND CUDA_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CUDA_ARGS_VERSION}")
  endif()

  if(CUDA_ARGS_EXCLUDE)
    set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
  endif()

  if(CUDA_ARGS_SHARED AND CUDA_ARGS_STATIC)
    _add_library(${CUDA_ARGS_NAME} "${EXCLUDE_OPTION}" "${CUDA_ARGS_SRCS}")
  elseif(CUDA_ARGS_SHARED)
    add_library(${CUDA_ARGS_NAME} SHARED ${EXCLUDE_OPTION} ${CUDA_ARGS_SRCS})
  elseif(CUDA_ARGS_STATIC)
    add_library(${CUDA_ARGS_NAME} STATIC ${EXCLUDE_OPTION} ${CUDA_ARGS_SRCS})
  else()
    add_library(${CUDA_ARGS_NAME} ${EXCLUDE_OPTION} ${CUDA_ARGS_SRCS})
  endif()

  if(TARGET ${CUDA_ARGS_NAME}_objects)
    _cuda_target_properties(
        NAME "${CUDA_ARGS_NAME}_objects"
        INCS "${CUDA_ARGS_INCS};${CUDA_ARGS_PUBINCS}"
        DEFS "${CUDA_ARGS_DEFS}"
        LIBS "${CUDA_ARGS_LIBS}"
        CFLAGS "${CUDA_ARGS_CFLAGS}"
        CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
        CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
        LDFLAGS "${CUDA_ARGS_LDFLAGS}"
        DEPS "${CUDA_ARGS_DEPS}"
        "${CUDA_ARGS_UNPARSED_ARGUMENTS}"
      )
  endif()

  if(TARGET ${CUDA_ARGS_NAME}_static)
    _cuda_target_properties(
        NAME "${CUDA_ARGS_NAME}_static"
        INCS "${CUDA_ARGS_INCS}"
        PUBINCS "${CUDA_ARGS_PUBINCS}"
        DEFS "${CUDA_ARGS_DEFS}"
        LIBS "${CUDA_ARGS_LIBS}"
        CFLAGS "${CUDA_ARGS_CFLAGS}"
        CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
        CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
        LDFLAGS "${CUDA_ARGS_LDFLAGS}"
        DEPS "${CUDA_ARGS_DEPS}"
        "${CUDA_ARGS_UNPARSED_ARGUMENTS}"
      )
    if(CUDA_ARGS_PACKED)
      install(
          TARGETS ${CUDA_ARGS_NAME}_static
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        )
    endif()
  endif()

  _cuda_target_properties(
      NAME "${CUDA_ARGS_NAME}"
      INCS "${CUDA_ARGS_INCS}"
      PUBINCS "${CUDA_ARGS_PUBINCS}"
      DEFS "${CUDA_ARGS_DEFS}"
      LIBS "${CUDA_ARGS_LIBS}"
      CFLAGS "${CUDA_ARGS_CFLAGS}"
      CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
      CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
      LDFLAGS "${CUDA_ARGS_LDFLAGS}"
      DEPS "${CUDA_ARGS_DEPS}"
      VERSION "${CUDA_ARGS_VERSION}"
      "${CUDA_ARGS_UNPARSED_ARGUMENTS}"
    )
  if(CUDA_ARGS_PACKED)
    install(
        TARGETS ${CUDA_ARGS_NAME}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      )
    if(CUDA_ARGS_PUBINCS)
      foreach(PACKED_EXCLUDE ${CUDA_ARGS_PACKED_IGORNES})
        list(APPEND PATTERN_EXCLUDES "PATTERN;${PACKED_EXCLUDE};EXCLUDE")
      endforeach()
      install(
          DIRECTORY ${CUDA_ARGS_PUBINCS} DESTINATION ${CMAKE_INSTALL_INCDIR}
          FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
          PATTERN "*.hxx" PATTERN "*.cuh"
          ${PATTERN_EXCLUDES}
        )
    endif()
  endif()
endfunction()

## Build a CUDA executable program
function(cuda_binary)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()

  cmake_parse_arguments(
      CUDA_ARGS "PACKED" "NAME;VERSION"
     "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;CUDAFLAGS;LDFLAGS;DEPS" ${ARGN}
    )

  if(NOT CUDA_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CUDA_ARGS_SRCS ${CUDA_ARGS_SRCS})
  if(NOT CUDA_ARGS_SRCS)
    message(FATAL_ERROR "No source files found of ${CUDA_ARGS_NAME}.")
  endif()

  if(CUDA_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CUDA_ARGS_NAME}")
    list(APPEND CUDA_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CUDA_ARGS_VERSION}")
  endif()
  add_executable(${CUDA_ARGS_NAME} ${CUDA_ARGS_SRCS})

  if(CUDA_ARGS_PACKED)
    install(
        TARGETS ${CUDA_ARGS_NAME} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      )
  endif()

  _cuda_target_properties(
      NAME "${CUDA_ARGS_NAME}"
      INCS "${CUDA_ARGS_INCS}"
      DEFS "${CUDA_ARGS_DEFS}"
      LIBS "${CUDA_ARGS_LIBS}"
      CFLAGS "${CUDA_ARGS_CFLAGS}"
      CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
      CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
      LDFLAGS "${CUDA_ARGS_LDFLAGS}"
      DEPS "${CUDA_ARGS_DEPS}"
      VERSION "${CUDA_ARGS_VERSION}"
      "${CUDA_ARGS_UNPARSED_ARGUMENTS}"
    )
endfunction()

## Build a CUDA executable test program
function(cuda_test)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()

  cmake_parse_arguments(
      CUDA_ARGS "" "NAME;VERSION"
      "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;CUDAFLAGS;LDFLAGS;DEPS;ARGS" ${ARGN}
    )

  if(NOT CUDA_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CUDA_ARGS_SRCS ${CUDA_ARGS_SRCS})
  if(NOT CUDA_ARGS_SRCS)
    message(FATAL_ERROR "No source files found of ${CUDA_ARGS_NAME}.")
  endif()

  if(CUDA_ARGS_VERSION)
    string(REPLACE "-" "_" MACRO_PREFIX "${CUDA_ARGS_NAME}")
    list(APPEND CUDA_ARGS_DEFS ${MACRO_PREFIX}_VERSION="${CUDA_ARGS_VERSION}")
  endif()
  add_executable(${CUDA_ARGS_NAME} EXCLUDE_FROM_ALL ${CUDA_ARGS_SRCS})

  _cuda_target_properties(
      NAME "${CUDA_ARGS_NAME}"
      INCS "${CUDA_ARGS_INCS}"
      DEFS "${CUDA_ARGS_DEFS}"
      LIBS "${CUDA_ARGS_LIBS}"
      CFLAGS "${CUDA_ARGS_CFLAGS}"
      CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
      CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
      LDFLAGS "${CUDA_ARGS_LDFLAGS}"
      DEPS "${CUDA_ARGS_DEPS}"
      "${CUDA_ARGS_UNPARSED_ARGUMENTS}"
    )
  add_dependencies(unittest ${CUDA_ARGS_NAME})
  add_custom_target(
      unittest.${CUDA_ARGS_NAME}
      COMMAND $<TARGET_FILE:${CUDA_ARGS_NAME}> "${CUDA_ARGS_ARGS}"
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      DEPENDS ${CUDA_ARGS_NAME}
    )
  add_test(
      NAME ${CUDA_ARGS_NAME}
      COMMAND $<TARGET_FILE:${CUDA_ARGS_NAME}> "${CUDA_ARGS_ARGS}"
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
endfunction()

## Add existing test cases to a test suite
function(cuda_test_suite)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()
  cc_test_suite(${ARGN})
endfunction()

## Import a C/C++/CUDA static or shared library
function(cuda_import)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()

  cmake_parse_arguments(
      CUDA_ARGS "STATIC;SHARED;PACKED"
      "NAME;PATH;IMPLIB" "INCS;PUBINCS;DEPS;PACKED_EXCLUDES" ${ARGN}
    )

  if(NOT CUDA_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB CUDA_ARGS_PATH ${CUDA_ARGS_PATH})
  if(NOT CUDA_ARGS_PATH)
    message(FATAL_ERROR "No imported target file found of ${CUDA_ARGS_NAME}.")
  endif()
  if(MSVC AND CUDA_ARGS_SHARED AND NOT CUDA_ARGS_IMPLIB)
    string(REGEX REPLACE
        ".[Dd][Ll][Ll]$" ".lib" CUDA_ARGS_IMPLIB ${CUDA_ARGS_PATH}
      )
  endif()

  if(CUDA_ARGS_SHARED)
    add_library(${CUDA_ARGS_NAME} SHARED IMPORTED GLOBAL)
  elseif(CUDA_ARGS_STATIC)
    add_library(${CUDA_ARGS_NAME} STATIC IMPORTED GLOBAL)
  else()
    add_library(${CUDA_ARGS_NAME} UNKNOWN IMPORTED GLOBAL)
  endif()

  set_property(
      TARGET ${CUDA_ARGS_NAME} PROPERTY IMPORTED_LOCATION ${CUDA_ARGS_PATH}
    )
  if(MSVC AND CUDA_ARGS_SHARED)
    set_property(
        TARGET ${CUDA_ARGS_NAME} PROPERTY IMPORTED_IMPLIB ${CUDA_ARGS_IMPLIB}
      )
  endif()

  if(CUDA_ARGS_INCS)
    _absolute_paths(INC_DIRS ${CUDA_ARGS_INCS})
    foreach(INC_DIR ${INC_DIRS})
      set_property(
          TARGET ${CUDA_ARGS_NAME} APPEND PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "${INC_DIR}"
        )
    endforeach()
  endif()

  if(CUDA_ARGS_PUBINCS)
    _absolute_paths(INC_DIRS ${CUDA_ARGS_PUBINCS})
    foreach(INC_DIR ${INC_DIRS})
      set_property(
          TARGET ${CUDA_ARGS_NAME} APPEND PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "${INC_DIR}"
        )
    endforeach()
  endif()

  if(CUDA_ARGS_DEPS)
    add_dependencies(${CUDA_ARGS_NAME} "${CUDA_ARGS_DEPS}")
  endif()

  if(CUDA_ARGS_PACKED)
    install(
        TARGETS ${CUDA_ARGS_NAME}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      )
    if(CUDA_ARGS_PUBINCS)
      foreach(PACKED_EXCLUDE ${CUDA_ARGS_PACKED_EXCLUDES})
        list(APPEND PATTERN_EXCLUDES "PATTERN;${PACKED_EXCLUDE};EXCLUDE")
      endforeach()
      install(
          DIRECTORY ${CUDA_ARGS_PUBINCS} DESTINATION ${CMAKE_INSTALL_INCDIR}
          FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
          PATTERN "*.hxx" PATTERN "*.cuh"
          ${PATTERN_EXCLUDES}
        )
    endif()
  endif()
endfunction()

## Import a C/C++/CUDA interface library
function(cuda_interface)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA language supported.")
  endif()

  cmake_parse_arguments(
      CUDA_ARGS "PACKED" "NAME" "INCS;PUBINCS;DEPS;PACKED_EXCLUDES" ${ARGN}
    )

  if(NOT CUDA_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  add_library(${CUDA_ARGS_NAME} INTERFACE GLOBAL)
  if(CUDA_ARGS_INCS)
    _absolute_paths(INC_DIRS ${CUDA_ARGS_INCS})
    target_include_directories(${CUDA_ARGS_NAME} INTERFACE "${INC_DIRS}")
  endif()

  if(CUDA_ARGS_PUBINCS)
    _absolute_paths(INC_DIRS ${CUDA_ARGS_PUBINCS})
    target_include_directories(${CUDA_ARGS_NAME} INTERFACE "${INC_DIRS}")
  endif()

  if(CUDA_ARGS_DEPS)
    add_dependencies(${CUDA_ARGS_NAME} "${CUDA_ARGS_DEPS}")
  endif()

  if(CUDA_ARGS_PACKED AND CUDA_ARGS_PUBINCS)
    foreach(PACKED_EXCLUDE ${CUDA_ARGS_PACKED_EXCLUDES})
      list(APPEND PATTERN_EXCLUDES "PATTERN;${PACKED_EXCLUDE};EXCLUDE")
    endforeach()
    install(
        DIRECTORY ${CUDA_ARGS_PUBINCS} DESTINATION ${CMAKE_INSTALL_INCDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        PATTERN "*.hxx" PATTERN "*.cuh"
        ${PATTERN_EXCLUDES}
      )
  endif()
endfunction()

## Build a C/C++/CUDA executable google test program
function(cuda_gtest)
  cmake_parse_arguments(
      CUDA_ARGS "" "NAME;VERSION"
      "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;CUDAFLAGS;LDFLAGS;DEPS;ARGS" ${ARGN}
    )
  _find_gtest()
  cuda_test(
      NAME "${CUDA_ARGS_NAME}"
      VERSION "${CUDA_ARGS_VERSION}"
      SRCS "${CUDA_ARGS_SRCS}"
      INCS "${CUDA_ARGS_INCS};${FIND_GTEST_INCS}"
      DEFS "${CUDA_ARGS_DEFS}"
      LIBS "${CUDA_ARGS_LIBS};${FIND_GTEST_LIBS}"
      CFLAGS "${CUDA_ARGS_CFLAGS}"
      CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
      CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
      LDFLAGS "${CUDA_ARGS_LDFLAGS}"
      DEPS "${CUDA_ARGS_DEPS}"
      ARGS "${CUDA_ARGS_ARGS}"
    )
endfunction()

## Build a C/C++/CUDA executable google mock program
function(cuda_gmock)
  cmake_parse_arguments(
      CUDA_ARGS "" "NAME;VERSION"
      "SRCS;INCS;DEFS;LIBS;CFLAGS;CXXFLAGS;CUDAFLAGS;LDFLAGS;DEPS;ARGS" ${ARGN}
    )
  _find_gmock()
  cuda_test(
      NAME "${CUDA_ARGS_NAME}"
      VERSION "${CUDA_ARGS_VERSION}"
      SRCS "${CUDA_ARGS_SRCS}"
      INCS "${CUDA_ARGS_INCS};${FIND_GMOCK_INCS}"
      DEFS "${CUDA_ARGS_DEFS}"
      LIBS "${CUDA_ARGS_LIBS};${FIND_GMOCK_LIBS}"
      CFLAGS "${CUDA_ARGS_CFLAGS}"
      CXXFLAGS "${CUDA_ARGS_CXXFLAGS}"
      CUDAFLAGS "${CUDA_ARGS_CUDAFLAGS}"
      LDFLAGS "${CUDA_ARGS_LDFLAGS}"
      DEPS "${CUDA_ARGS_DEPS}"
      ARGS "${CUDA_ARGS_ARGS}"
    )
endfunction()

## Add a subdirectory to the build
function(go_directory)
  add_subdirectory(${ARGN})
endfunction()

## Add subdirectories to the build
function(go_directories)
  foreach(SRC_DIR ${ARGN})
    add_subdirectory(${SRC_DIR})
  endforeach()
endfunction()

## Build a go executable program
function(go_binary)
  find_program(
      GO_EXECUTABLE go PATHS $ENV{HOME}/go ENV GOROOT GOPATH PATH_SUFFIXES bin
    )
  if(NOT GO_EXECUTABLE)
    message(FATAL_ERROR "No go language compiler found.")
  endif()

  cmake_parse_arguments(
      GO_ARGS "PACKED" "NAME"
      "GOPATH;SRCS;ASMFLAGS;GCFLAGS;LDFLAGS;DEPS" ${ARGN}
    )
  if(NOT GO_ARGS_NAME)
    message(FATAL_ERROR "No target name privated.")
  endif()

  file(GLOB GO_ARGS_SRCS ${GO_ARGS_SRCS})
  if(NOT GO_ARGS_SRCS)
    message(FATAL_ERROR "No source files/directories found of ${GO_ARGS_NAME}.")
  endif()

  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    string(REPLACE ";" "\;" GO_ARGS_GOPATH "${GO_ARGS_GOPATH}")
  else()
    string(REPLACE ";" ":" GO_ARGS_GOPATH "${GO_ARGS_GOPATH}")
  endif()

  set(
      GO_OUTPUT_FILE
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GO_ARGS_NAME}${CMAKE_EXECUTABLE_SUFFIX}
    )
  file(RELATIVE_PATH GO_OUTPUT_REL_FILE ${CMAKE_BINARY_DIR} ${GO_OUTPUT_FILE})
  add_custom_target(
      ${GO_ARGS_NAME}
      COMMAND ${CMAKE_COMMAND} -E env GOPATH="${GO_ARGS_GOPATH}"
      "${GO_EXECUTABLE}" build -v -buildmode=exe
      -compiler=gc -gcflags="${GO_ARGS_GCFLAGS}" -asmflags="${GO_ARGS_ASMFLAGS}"
      -ldflags="${GO_ARGS_LDFLAGS}"
      -o "${GO_OUTPUT_FILE}" "${GO_ARGS_SRCS}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      DEPENDS "${GO_ARGS_DEPS}"
      COMMENT "Building GO executable ${GO_OUTPUT_REL_FILE}"
    )
  if(GO_ARGS_PACKED)
    install(PROGRAMS ${GO_OUTPUT_FILE} DESTINATION "${CMAKE_INSTALL_BINDIR}")
  endif()
endfunction()

## Fetch content
function(_fetch_content)
  cmake_parse_arguments(
      DL_ARGS ""
      "NAME;PATH;GIT_URL;GIT_TAG;HG_URL;HG_TAG;SVN_URL;SVN_REV;URL;URL_HASH"
      "" ${ARGN}
    )

  if(NOT DL_ARGS_NAME)
    message(FATAL_ERROR "No fetch name privated.")
  endif()

  if(NOT DL_ARGS_PATH)
    # Download to current source directory
    set(DL_ARGS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${DL_ARGS_NAME}")
  endif()

  set(
      CMAKELISTS_CONTENT
      "cmake_minimum_required(VERSION 3.1)\n"
      "project(${DL_ARGS_NAME})\n"
      "include(ExternalProject)\n"
      "ExternalProject_Add(\n"
      "    ${DL_ARGS_NAME}\n"
      "    PREFIX \"external\"\n"
      "    GIT_REPOSITORY \"${DL_ARGS_GIT_URL}\"\n"
      "    GIT_TAG \"${DL_ARGS_GIT_TAG}\"\n"
      "    HG_REPOSITORY \"${DL_ARGS_HG_URL}\"\n"
      "    HG_TAG \"${DL_ARGS_HG_TAG}\"\n"
      "    SVN_REPOSITORY \"${DL_ARGS_SVN_URL}\"\n"
      "    SVN_REVISION \"${DL_ARGS_SVN_REV}\"\n"
      "    URL \"${DL_ARGS_URL}\"\n"
      "    URL_HASH \"${DL_ARGS_URL_HASH}\"\n"
      "    SOURCE_DIR \"${DL_ARGS_PATH}\"\n"
      "    BINARY_DIR \"\"\n"
      "    CONFIGURE_COMMAND \"\"\n"
      "    BUILD_COMMAND \"\"\n"
      "    INSTALL_COMMAND \"\"\n"
      "    TEST_COMMAND \"\"\n"
      "    LOG_DOWNLOAD ON\n"
      "  )\n"
    )
  set(
      CMAKELISTS_DIRECTORY
      "${PROJECT_BINARY_DIR}/downloads/${DL_ARGS_NAME}"
    )
  add_custom_target(
      external.${DL_ARGS_NAME}
      COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . &&
              "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKELISTS_DIRECTORY}"
    )

  # Write a cmake script into folder
  file(WRITE "${CMAKELISTS_DIRECTORY}/CMakeLists.txt" ${CMAKELISTS_CONTENT})

  execute_process(
      COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKELISTS_DIRECTORY}"
    )
  execute_process(
      COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKELISTS_DIRECTORY}"
    )
endfunction()

## Download a git repository
function(git_repository)
  cmake_parse_arguments(GIT_ARGS "" "NAME;PATH;URL;TAG" "" ${ARGN})

  if(NOT GIT_ARGS_NAME)
    message(FATAL_ERROR "No repository name privated.")
  endif()
  if(NOT GIT_ARGS_URL)
    message(FATAL_ERROR "No repository URL privated.")
  endif()

  if(GIT_ARGS_PATH AND NOT IS_ABSOLUTE ${GIT_ARGS_PATH})
    get_filename_component(GIT_ARGS_PATH ${GIT_ARGS_PATH} ABSOLUTE)
  endif()

  _fetch_content(
      NAME "${GIT_ARGS_NAME}"
      PATH "${GIT_ARGS_PATH}"
      GIT_URL "${GIT_ARGS_URL}"
      GIT_TAG "${GIT_ARGS_TAG}"
    )
endfunction()

## Download a hg repository
function(hg_repository)
  cmake_parse_arguments(HG_ARGS "" "NAME;PATH;URL;TAG" "" ${ARGN})

  if(NOT HG_ARGS_NAME)
    message(FATAL_ERROR "No repository name privated.")
  endif()
  if(NOT HG_ARGS_URL)
    message(FATAL_ERROR "No repository URL privated.")
  endif()

  if(HG_ARGS_PATH AND NOT IS_ABSOLUTE ${HG_ARGS_PATH})
    get_filename_component(HG_ARGS_PATH ${HG_ARGS_PATH} ABSOLUTE)
  endif()

  _fetch_content(
      NAME "${HG_ARGS_NAME}"
      PATH "${HG_ARGS_PATH}"
      HG_URL "${HG_ARGS_URL}"
      HG_TAG "${HG_ARGS_TAG}"
    )
endfunction()

## Download a svn repository
function(svn_repository)
  cmake_parse_arguments(SVN_ARGS "" "NAME;PATH;URL;REV" "" ${ARGN})

  if(NOT SVN_ARGS_NAME)
    message(FATAL_ERROR "No repository name privated.")
  endif()
  if(NOT SVN_ARGS_URL)
    message(FATAL_ERROR "No repository URL privated.")
  endif()

  if(SVN_ARGS_PATH AND NOT IS_ABSOLUTE ${SVN_ARGS_PATH})
    get_filename_component(SVN_ARGS_PATH ${SVN_ARGS_PATH} ABSOLUTE)
  endif()

  _fetch_content(
      NAME "${SVN_ARGS_NAME}"
      PATH "${SVN_ARGS_PATH}"
      SVN_URL "${SVN_ARGS_URL}"
      SVN_REV "${SVN_ARGS_REV}"
    )
endfunction()

## Download a http archive
function(http_archive)
  cmake_parse_arguments(HTTP_ARGS "" "NAME;PATH;URL;SHA256;SHA1;MD5" "" ${ARGN})

  if(NOT HTTP_ARGS_NAME)
    message(FATAL_ERROR "No archive name privated.")
  endif()
  if(NOT HTTP_ARGS_URL)
    message(FATAL_ERROR "No archive URL privated.")
  endif()

  if(HTTP_ARGS_PATH AND NOT IS_ABSOLUTE ${HTTP_ARGS_PATH})
    get_filename_component(HTTP_ARGS_PATH ${HTTP_ARGS_PATH} ABSOLUTE)
  endif()

  if(HTTP_ARGS_SHA256)
    set(HTTP_URL_HASH "SHA256=${HTTP_ARGS_SHA256}")
  elseif(HTTP_ARGS_SHA1)
    set(HTTP_URL_HASH "SHA1=${HTTP_ARGS_SHA1}")
  elseif(HTTP_ARGS_MD5)
    set(HTTP_URL_HASH "MD5=${HTTP_ARGS_MD5}")
  else()
    set(HTTP_URL_HASH "")
  endif()

  _fetch_content(
      NAME "${HTTP_ARGS_NAME}"
      PATH "${HTTP_ARGS_PATH}"
      URL "${HTTP_ARGS_URL}"
      URL_HASH "${HTTP_URL_HASH}"
    )
endfunction()

## Retrieve a version string from GIT
function(git_version _RESULT _SOURCES_DIR)
  find_package(Git REQUIRED)

  if(NOT IS_ABSOLUTE ${_SOURCES_DIR})
    get_filename_component(_SOURCES_DIR ${_SOURCES_DIR} ABSOLUTE)
  endif()

  # git describe --tags
  execute_process(
      COMMAND "${GIT_EXECUTABLE}" describe --tags
      WORKING_DIRECTORY "${_SOURCES_DIR}"
      RESULT_VARIABLE GIT_VER_RESULT
      OUTPUT_VARIABLE GIT_VER_OUTPUT
      ERROR_VARIABLE GIT_VER_ERROR
    )
  if(GIT_VER_RESULT EQUAL 0)
    string(STRIP ${GIT_VER_OUTPUT} GIT_VER_OUTPUT)
    set(${_RESULT} "${GIT_VER_OUTPUT}" PARENT_SCOPE)
    return()
  endif()

  # git rev-parse --short HEAD
  execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
      WORKING_DIRECTORY "${_SOURCES_DIR}"
      RESULT_VARIABLE GIT_VER_RESULT
      OUTPUT_VARIABLE GIT_VER_OUTPUT
      ERROR_VARIABLE GIT_VER_ERROR
    )
  if(GIT_VER_RESULT EQUAL 0)
    string(STRIP ${GIT_VER_OUTPUT} GIT_VER_OUTPUT)
    set(${_RESULT} "g${GIT_VER_OUTPUT}" PARENT_SCOPE)
    return()
  endif()

  set(${_RESULT} "" PARENT_SCOPE)
endfunction()

## Retrieve a version string from HG
function(hg_version _RESULT _SOURCES_DIR)
  find_package(Hg REQUIRED)

  if(NOT IS_ABSOLUTE ${_SOURCES_DIR})
    get_filename_component(_SOURCES_DIR ${_SOURCES_DIR} ABSOLUTE)
  endif()

  # hg log -T "{latesttagdistance}" -r .
  execute_process(
      COMMAND "${HG_EXECUTABLE}" log -T "{latesttagdistance}" -r .
      WORKING_DIRECTORY "${_SOURCES_DIR}"
      RESULT_VARIABLE HG_VER_RESULT
      OUTPUT_VARIABLE HG_VER_OUTPUT
      ERROR_VARIABLE HG_VER_ERROR
    )
  if(HG_VER_RESULT EQUAL 0)
    string(STRIP ${HG_VER_OUTPUT} HG_VER_OUTPUT)
    if(HG_VER_OUTPUT STREQUAL "0")
      # hg log -T "{latesttag}" -r .
      execute_process(
          COMMAND "${HG_EXECUTABLE}" log -T "{latesttag}" -r .
          WORKING_DIRECTORY "${_SOURCES_DIR}"
          RESULT_VARIABLE HG_VER_RESULT
          OUTPUT_VARIABLE HG_VER_OUTPUT
          ERROR_VARIABLE HG_VER_ERROR
        )
    else()
      # hg log -T "{latesttag}-{latesttagdistance}-h{node|short}" -r .
      execute_process(
          COMMAND "${HG_EXECUTABLE}" log
          -T "{latesttag}-{latesttagdistance}-h{node|short}" -r .
          WORKING_DIRECTORY "${_SOURCES_DIR}"
          RESULT_VARIABLE HG_VER_RESULT
          OUTPUT_VARIABLE HG_VER_OUTPUT
          ERROR_VARIABLE HG_VER_ERROR
        )
    endif()

    if(HG_VER_RESULT EQUAL 0)
      string(STRIP ${HG_VER_OUTPUT} HG_VER_OUTPUT)
      if(NOT HG_VER_OUTPUT MATCHES "^null.*")
        set(${_RESULT} "${HG_VER_OUTPUT}" PARENT_SCOPE)
        return()
      endif()
    endif()
  endif()

  # hg log -T "h{node|short}" -r .
  execute_process(
      COMMAND "${HG_EXECUTABLE}" log -T "h{node|short}" -r .
      WORKING_DIRECTORY "${_SOURCES_DIR}"
      RESULT_VARIABLE HG_VER_RESULT
      OUTPUT_VARIABLE HG_VER_OUTPUT
      ERROR_VARIABLE HG_VER_ERROR
    )
  if(HG_VER_RESULT EQUAL 0)
    string(STRIP ${HG_VER_OUTPUT} HG_VER_OUTPUT)
    set(${_RESULT} "${HG_VER_OUTPUT}" PARENT_SCOPE)
    return()
  endif()

  set(${_RESULT} "" PARENT_SCOPE)
endfunction()

## Retrieve a version string from SVN
function(svn_version _RESULT _SOURCES_DIR)
  find_package(Subversion REQUIRED)

  if(NOT IS_ABSOLUTE ${_SOURCES_DIR})
    get_filename_component(_SOURCES_DIR ${_SOURCES_DIR} ABSOLUTE)
  endif()

  # svn info --show-item revision
  execute_process(
      COMMAND "${Subversion_SVN_EXECUTABLE}" info --show-item revision
      WORKING_DIRECTORY "${_SOURCES_DIR}"
      RESULT_VARIABLE SVN_VER_RESULT
      OUTPUT_VARIABLE SVN_VER_OUTPUT
      ERROR_VARIABLE SVN_VER_ERROR
    )
  if(SVN_VER_RESULT EQUAL 0)
    string(STRIP ${SVN_VER_OUTPUT} SVN_VER_OUTPUT)
    set(${_RESULT} "r${SVN_VER_OUTPUT}" PARENT_SCOPE)
    return()
  endif()

  set(${_RESULT} "" PARENT_SCOPE)
endfunction()

_find_workspace_directory(BAZEL_WORKSPACE_DIR)
if(BAZEL_WORKSPACE_DIR)
  include("${BAZEL_WORKSPACE_DIR}/Workspace.cmake")
endif()
