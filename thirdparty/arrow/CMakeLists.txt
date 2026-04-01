set(ARROW_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apache-arrow-21.0.0)
if(ANDROID)
        set(ARROW_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/arrow.android.patch)
        apply_patch_once("arrow_android_fix" "${ARROW_SRC_DIR}" "${ARROW_PATCH}")
else()
        set(ARROW_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/arrow.patch)
        apply_patch_once("arrow_fix" "${ARROW_SRC_DIR}" "${ARROW_PATCH}")
endif()
if(MSVC)
        set(ARROW_WIN_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/arrow.windows.patch)
        apply_patch_once("arrow_windows_crt_fix" "${ARROW_SRC_DIR}" "${ARROW_WIN_PATCH}")
endif()

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(NPROC)

if(MSVC)
    set(LIB_PARQUET ${EXTERNAL_LIB_DIR}/parquet_static.lib)
    set(LIB_ARROW ${EXTERNAL_LIB_DIR}/arrow_static.lib)
    set(LIB_COMPUTE ${EXTERNAL_LIB_DIR}/arrow_compute_static.lib)
    set(LIB_ACERO ${EXTERNAL_LIB_DIR}/arrow_acero_static.lib)
    set(LIB_ARROW_DEPENDS ${EXTERNAL_LIB_DIR}/arrow_bundled_dependencies.lib)
    set(LIB_ARROW_DATASET ${EXTERNAL_LIB_DIR}/arrow_dataset_static.lib)
else()
    set(LIB_PARQUET ${EXTERNAL_LIB_DIR}/libparquet.a)
    set(LIB_ARROW ${EXTERNAL_LIB_DIR}/libarrow.a)
    set(LIB_COMPUTE ${EXTERNAL_LIB_DIR}/libarrow_compute.a)
    set(LIB_ACERO ${EXTERNAL_LIB_DIR}/libarrow_acero.a)
    set(LIB_ARROW_DEPENDS ${EXTERNAL_LIB_DIR}/libarrow_bundled_dependencies.a)
    set(LIB_ARROW_DATASET ${EXTERNAL_LIB_DIR}/libarrow_dataset.a)
endif()

set(CONFIGURE_ENV_LIST "")
if(USE_OSS_MIRROR)
        list(APPEND CONFIGURE_ENV_LIST
                "ARROW_BOOST_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/boost-1.88.0-cmake.tar.gz"
                "ARROW_RAPIDJSON_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/rapidjson-232389d4f1012dddec4ef84861face2d2ba85709.tar.gz"
                "ARROW_RE2_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/re2-2022-06-01.tar.gz"
                "ARROW_THRIFT_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/thrift-0.22.0.tar.gz"
                "ARROW_UTF8PROC_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/utf8proc-2.10.0.tar.gz"
                "ARROW_XSIMD_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/xsimd-13.0.0.tar.gz"
                "ARROW_ZLIB_URL=https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/zlib-1.3.1.tar.gz"
        )
        message(STATUS "Using OSS mirror for third-party downloads")
endif()

if(ANDROID)
        ExternalProject_Add(
                ARROW.BUILD PREFIX arrow
                SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apache-arrow-21.0.0
                DOWNLOAD_COMMAND ""
                BUILD_IN_SOURCE false
                CONFIGURE_COMMAND env ${CONFIGURE_ENV_LIST} "${CMAKE_COMMAND}" ${CMAKE_CACHE_ARGS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_DEBUG_POSTFIX= -DARROW_BUILD_SHARED=OFF -DARROW_ACERO=ON -DARROW_FILESYSTEM=ON -DARROW_DATASET=ON -DARROW_PARQUET=ON -DARROW_COMPUTE=ON -DARROW_WITH_ZLIB=OFF -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_MIMALLOC=OFF -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DANDROID_ABI=${ANDROID_ABI} -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} -DARROW_WITH_MUSL=OFF "<SOURCE_DIR>/cpp"
                BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target all -- -j ${NPROC}
                INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix=${EXTERNAL_BINARY_DIR}/usr/local
                BYPRODUCTS ${LIB_PARQUET} ${LIB_ARROW} ${LIB_COMPUTE} ${LIB_ACERO} ${LIB_ARROW_DEPENDS} ${LIB_ARROW_DATASET}
                LOG_DOWNLOAD ON
                LOG_CONFIGURE ON
                LOG_BUILD ON
                LOG_INSTALL ON
        )
elseif (MSVC)
    if(ZVEC_USE_STATIC_CRT)
        set(_ARROW_CRT_FLAG "/MT")
        set(_ARROW_MSVC_RUNTIME "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(_ARROW_CRT_FLAG "/MD")
        set(_ARROW_MSVC_RUNTIME "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
    endif()
    set(ARROW_EXTRA_CMAKE_ARGS
        "-DCMAKE_C_FLAGS=/FS ${_ARROW_CRT_FLAG}"
        "-DCMAKE_CXX_FLAGS=/FS ${_ARROW_CRT_FLAG}"
        -DARROW_USE_STATIC_CRT=${ZVEC_USE_STATIC_CRT}
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=${_ARROW_MSVC_RUNTIME}"
    )
        ExternalProject_Add(
                ARROW.BUILD PREFIX arrow
                SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apache-arrow-21.0.0
                DOWNLOAD_COMMAND ""
                BUILD_IN_SOURCE false
                CONFIGURE_COMMAND "${CMAKE_COMMAND}" -E env ${CONFIGURE_ENV_LIST} "${CMAKE_COMMAND}" ${CMAKE_CACHE_ARGS} ${ARROW_EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=$<CONFIG> -DCMAKE_DEBUG_POSTFIX= -DARROW_BUILD_SHARED=OFF -DARROW_ACERO=ON -DARROW_FILESYSTEM=ON -DARROW_DATASET=ON -DARROW_PARQUET=ON -DARROW_COMPUTE=ON -DARROW_WITH_ZLIB=OFF -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_MIMALLOC=OFF -DCMAKE_INSTALL_LIBDIR=lib "<SOURCE_DIR>/cpp"
                BUILD_COMMAND "${CMAKE_COMMAND}" --build . -j ${NPROC} --config $<CONFIG>
                INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix=${EXTERNAL_BINARY_DIR}/usr/local --config $<CONFIG>
                BYPRODUCTS ${LIB_PARQUET} ${LIB_ARROW} ${LIB_COMPUTE} ${LIB_ACERO} ${LIB_ARROW_DEPENDS} ${LIB_ARROW_DATASET}
                LOG_DOWNLOAD ON
                LOG_CONFIGURE ON
                LOG_BUILD ON
                LOG_INSTALL ON
        )
else ()
    ExternalProject_Add(
            ARROW.BUILD PREFIX arrow
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apache-arrow-21.0.0
            DOWNLOAD_COMMAND ""
            BUILD_IN_SOURCE false
            CONFIGURE_COMMAND env ${CONFIGURE_ENV_LIST} "${CMAKE_COMMAND}" ${CMAKE_CACHE_ARGS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_DEBUG_POSTFIX= -DARROW_BUILD_SHARED=OFF -DARROW_ACERO=ON -DARROW_FILESYSTEM=ON -DARROW_DATASET=ON -DARROW_PARQUET=ON -DARROW_COMPUTE=ON -DARROW_WITH_ZLIB=OFF -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_MIMALLOC=OFF -DCMAKE_INSTALL_LIBDIR=lib "<SOURCE_DIR>/cpp"
            BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target all -- -j ${NPROC}
            INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix=${EXTERNAL_BINARY_DIR}/usr/local
            BYPRODUCTS ${LIB_PARQUET} ${LIB_ARROW} ${LIB_COMPUTE} ${LIB_ACERO} ${LIB_ARROW_DEPENDS} ${LIB_ARROW_DATASET}
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
    )
endif()

add_library(arrow UNKNOWN IMPORTED GLOBAL)
add_dependencies(arrow ARROW.BUILD)

set(Arrow_FOUND TRUE PARENT_SCOPE)
set(Arrow_INCLUDE_DIR ${EXTERNAL_INC_DIR} PARENT_SCOPE)
set(Arrow_LIBRARIES ${LIB_ARROW} PARENT_SCOPE)
set(Arrow_DIR ${EXTERNAL_BINARY_DIR} PARENT_SCOPE)
set(Arrow_LIBRARY_DIR ${EXTERNAL_BINARY_DIR} PARENT_SCOPE)

add_library(Arrow::arrow_depends UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_depends PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_ARROW_DEPENDS}"
)
add_dependencies(Arrow::arrow_depends ARROW.BUILD)


add_library(Arrow::arrow_static UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_ARROW}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends"
        INTERFACE_COMPILE_DEFINITIONS "ARROW_STATIC" #TODO(windows): see if this can be removed
)
add_dependencies(Arrow::arrow_static ARROW.BUILD)

add_library(Arrow::parquet_static UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::parquet_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_PARQUET}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static"
        INTERFACE_COMPILE_DEFINITIONS "PARQUET_STATIC"
)
add_dependencies(Arrow::parquet_static ARROW.BUILD)

add_library(Arrow::arrow_compute UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_compute PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_COMPUTE}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static"
        INTERFACE_COMPILE_DEFINITIONS "ARROW_COMPUTE_STATIC"
)
add_dependencies(Arrow::arrow_compute ARROW.BUILD)

add_library(Arrow::arrow_acero UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_acero PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_ACERO}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static;Arrow::arrow_compute"
        INTERFACE_COMPILE_DEFINITIONS "ARROW_ACERO_STATIC"
)
add_dependencies(Arrow::arrow_acero ARROW.BUILD)

add_library(Arrow::arrow_dataset UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_dataset PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_ARROW_DATASET}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static;Arrow::arrow_compute;Arrow::arrow_acero"
        INTERFACE_COMPILE_DEFINITIONS "ARROW_DS_STATIC"
)
add_dependencies(Arrow::arrow_dataset ARROW.BUILD)
