include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE AND NOT ANDROID)
    find_library(LIB_RT NAMES rt)
else()
    set(LIB_RT "")
endif()

git_version(GIT_SRCS_VER ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
    list(APPEND EXTRA_LIBS ${LIB_RT})
endif()

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
        setup_compiler_march_for_x86(MATH_MARCH_FLAG_SSE MATH_MARCH_FLAG_AVX2 MATH_MARCH_FLAG_AVX512 MATH_MARCH_FLAG_AVX512FP16)
        message(STATUS "best compiler march, sse: " ${MATH_MARCH_FLAG_SSE} ", avx2: " ${MATH_MARCH_FLAG_AVX2} ", avx512: " ${MATH_MARCH_FLAG_AVX512} ", avx512fp16: " ${MATH_MARCH_FLAG_AVX512FP16})

        file(GLOB_RECURSE MATH_FILES_SSE
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_sse.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_sse.c
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_sse.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_sse.c
            )

        file(GLOB_RECURSE MATH_FILES_AVX2
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx2.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx2.c
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx2.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx2.c
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx.c
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx.c
        )

        file(GLOB_RECURSE MATH_FILES_AVX512
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx512.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx512.c
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx512.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx512.c
        )

        file(GLOB_RECURSE MATH_FILES_AVX512FP16
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_dispatch.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_dispatch.c
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx512fp16.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx512fp16.c
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_dispatch.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_dispatch.c
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx512fp16.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx512fp16.c
        )

        foreach(MATH_FILE ${MATH_FILES_SSE})
            set_source_files_properties(
                ${MATH_FILE}
                PROPERTIES
                COMPILE_FLAGS "${MATH_MARCH_FLAG_SSE}"
            )
        endforeach()

        foreach(MATH_FILE ${MATH_FILES_AVX2})
            set_source_files_properties(
                ${MATH_FILE}
                PROPERTIES
                COMPILE_FLAGS "${MATH_MARCH_FLAG_AVX2}"
            )
        endforeach()

        foreach(MATH_FILE ${MATH_FILES_AVX512})
            set_source_files_properties(
                ${MATH_FILE}
                PROPERTIES
                COMPILE_FLAGS "${MATH_MARCH_FLAG_AVX512}"
            )
        endforeach()

        foreach(MATH_FILE ${MATH_FILES_AVX512FP16})
        set_source_files_properties(
            ${MATH_FILE}
            PROPERTIES
            COMPILE_FLAGS "${MATH_MARCH_FLAG_AVX512FP16}"
        )
    endforeach()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      # set(CMAKE_CXX_FLAGS "-march=armv8-a")
      # set(CMAKE_C_FLAGS "-march=armv8-a")
      set(MATH_MARCH_FLAG_NEON "-march=armv8-a")

      file(GLOB_RECURSE MATH_FILES_NEON
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_dispatch.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_dispatch.c
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_dispatch.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_dispatch.c
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_neon.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math/*_neon.c
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_neon.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_neon.c
      )

      foreach(MATH_FILE ${MATH_FILES_NEON})
          set_source_files_properties(
              ${MATH_FILE}
              PROPERTIES
              COMPILE_FLAGS "${MATH_MARCH_FLAG_NEON}"
          )
      endforeach()
    endif()
endif()

cc_library(
    NAME zvec_ailego STATIC STRICT PACKED
    SRCS    ${ALL_SRCS}
    LIBS    ${EXTRA_LIBS}
            Arrow::arrow_static
            Arrow::parquet_static
    VERSION "${GIT_SRCS_VER}"
)
