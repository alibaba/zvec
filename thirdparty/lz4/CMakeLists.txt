set(lz4_INCLUDE_DIR "${EXTERNAL_BINARY_DIR}/usr/local/include")
set(lz4_LIBRARY_DIR "${EXTERNAL_BINARY_DIR}/usr/local/lib/")
file(MAKE_DIRECTORY ${lz4_INCLUDE_DIR})
file(MAKE_DIRECTORY ${lz4_LIBRARY_DIR})

if(MSVC)
    set(LZ4_LIB_NAME lz4.lib)
else()
    set(LZ4_LIB_NAME liblz4.a)
endif()

include(ExternalProject)

set(_lz4_env "")
if(ANDROID)
  string(REGEX REPLACE "^android-([0-9]+)$" "\\1" ANDROID_API_LEVEL "${ANDROID_PLATFORM}")

  if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(TARGET_TRIPLE "aarch64-linux-android")
  elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(TARGET_TRIPLE "armv7a-linux-androideabi")
  elseif(ANDROID_ABI STREQUAL "x86")
    set(TARGET_TRIPLE "i686-linux-android")
  elseif(ANDROID_ABI STREQUAL "x86_64")
    set(TARGET_TRIPLE "x86_64-linux-android")
  else()
    message(FATAL_ERROR "Unsupported ANDROID_ABI: ${ANDROID_ABI}")
  endif()

  set(SYSROOT "${ANDROID_NDK}/toolchains/llvm/prebuilt/${ANDROID_HOST_TAG}/sysroot")
  set(COMMON_FLAGS
    "--sysroot=${SYSROOT}"
    "-target ${TARGET_TRIPLE}${ANDROID_API_LEVEL}"
    "-fPIC"
    "-D__ANDROID_API__=${ANDROID_API_LEVEL}"
  )

  list(APPEND COMMON_FLAGS ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}})

  string(JOIN " " _lz4_cflags ${COMMON_FLAGS})

  list(APPEND _lz4_env
    "CC=${CMAKE_C_COMPILER}"
    "AR=${CMAKE_AR}"
    "RANLIB=${CMAKE_RANLIB}"
    "STRIP=${ANDROID_NDK}/toolchains/llvm/prebuilt/${ANDROID_HOST_TAG}/bin/llvm-strip"
    "CFLAGS=${_lz4_cflags}"
  )

else()
  list(APPEND _lz4_env "CFLAGS=-fPIC")
endif()

if(MSVC)
    ExternalProject_Add(
            Lz4.BUILD
            PREFIX lz4
            URL "${CMAKE_CURRENT_SOURCE_DIR}/lz4-1.9.4"
            SOURCE_SUBDIR build/cmake
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BINARY_DIR}/usr/local
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_STATIC_LIBS=ON
            -DLZ4_BUILD_CLI=OFF
            -DLZ4_BUILD_LEGACY_LZ4C=OFF
            BUILD_COMMAND "${CMAKE_COMMAND}" --build . --config $<CONFIG> -j
            INSTALL_COMMAND "${CMAKE_COMMAND}" --install . --config $<CONFIG>
            BUILD_IN_SOURCE OFF
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            BUILD_BYPRODUCTS ${lz4_LIBRARY_DIR}/${LZ4_LIB_NAME}
    )
else()
    ExternalProject_Add(
      Lz4.BUILD
      PREFIX lz4
      URL "${CMAKE_CURRENT_SOURCE_DIR}/lz4-1.9.4"
      CONFIGURE_COMMAND ""
        BUILD_COMMAND env ${_lz4_env} BUILD_SHARED=no make -j
      INSTALL_COMMAND make DESTDIR=${EXTERNAL_BINARY_DIR} BUILD_SHARED=no install
      BUILD_IN_SOURCE ON
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON
      LOG_INSTALL ON
      BUILD_BYPRODUCTS ${lz4_LIBRARY_DIR}/liblz4.a
    )
endif()
add_library(lz4 STATIC IMPORTED GLOBAL)
set_target_properties(
    lz4 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${lz4_INCLUDE_DIR}"
    IMPORTED_LOCATION "${lz4_LIBRARY_DIR}/${LZ4_LIB_NAME}"
)
add_dependencies(lz4 Lz4.BUILD)

set(lz4_FOUND TRUE PARENT_SCOPE)
set(lz4_LIBRARY ${lz4_LIBRARY_DIR}/${LZ4_LIB_NAME} PARENT_SCOPE)
set(lz4_LIBRARIES ${lz4_LIBRARY_DIR}/${LZ4_LIB_NAME} PARENT_SCOPE)
set(lz4_INCLUDE_DIR "${EXTERNAL_BINARY_DIR}/usr/local/include" PARENT_SCOPE)
set(lz4_INCLUDE_DIRS "${EXTERNAL_BINARY_DIR}/usr/local/include" PARENT_SCOPE)
set(lz4_VERSION 1.9.4 PARENT_SCOPE)
