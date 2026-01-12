set(ARROW_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apache-arrow-21.0.0)
set(ARROW_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/arrow.patch)
apply_patch_once("arrow_fix" "${ARROW_SRC_DIR}" "${ARROW_PATCH}")

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(NPROC)

set(LIB_PARQUET ${EXTERNAL_LIB_DIR}/libparquet.a)
set(LIB_ARROW ${EXTERNAL_LIB_DIR}/libarrow.a)
set(LIB_COMPUTE ${EXTERNAL_LIB_DIR}/libarrow_compute.a)
set(LIB_ACERO ${EXTERNAL_LIB_DIR}/libarrow_acero.a)
set(LIB_ARROW_DEPENDS ${EXTERNAL_LIB_DIR}/libarrow_bundled_dependencies.a)
set(LIB_ARROW_DATASET ${EXTERNAL_LIB_DIR}/libarrow_dataset.a)

ExternalProject_Add(
        ARROW.BUILD PREFIX arrow
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apache-arrow-21.0.0
        DOWNLOAD_COMMAND ""
        BUILD_IN_SOURCE false
        CONFIGURE_COMMAND "${CMAKE_COMMAND}" ${CMAKE_CACHE_ARGS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_DEBUG_POSTFIX= -DARROW_BUILD_SHARED=OFF -DARROW_ACERO=ON -DARROW_FILESYSTEM=ON -DARROW_DATASET=ON -DARROW_PARQUET=ON -DARROW_COMPUTE=ON -DARROW_WITH_ZLIB=OFF -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_MIMALLOC=OFF -DCMAKE_INSTALL_LIBDIR=lib "<SOURCE_DIR>/cpp"
        BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target all -- -j ${NPROC}
        INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix=${EXTERNAL_BINARY_DIR}/usr/local
        BYPRODUCTS ${LIB_PARQUET} ${LIB_ARROW} ${LIB_COMPUTE} ${LIB_ACERO} ${LIB_ARROW_DEPENDS} ${LIB_ARROW_DATASET}
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_INSTALL ON
)
add_library(arrow UNKNOWN IMPORTED GLOBAL)
add_dependencies(arrow ARROW.BUILD)

set(Arrow_FOUND TRUE PARENT_SCOPE)
set(Arrow_INCLUDE_DIR ${EXTERNAL_INC_DIR} PARENT_SCOPE)
set(Arrow_LIBRARIES ${EXTERNAL_LIB_DIR}/libarrow.a PARENT_SCOPE)
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
)
add_dependencies(Arrow::arrow_static ARROW.BUILD)

add_library(Arrow::parquet_static UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::parquet_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_PARQUET}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static"
)
add_dependencies(Arrow::parquet_static ARROW.BUILD)

add_library(Arrow::arrow_compute UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_compute PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_COMPUTE}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static"
)
add_dependencies(Arrow::arrow_compute ARROW.BUILD)

add_library(Arrow::arrow_acero UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_acero PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_ACERO}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static;Arrow::arrow_compute"
)
add_dependencies(Arrow::arrow_acero ARROW.BUILD)

add_library(Arrow::arrow_dataset UNKNOWN IMPORTED GLOBAL)
set_target_properties(
        Arrow::arrow_dataset PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INC_DIR}
        IMPORTED_LOCATION "${LIB_ARROW_DATASET}"
        INTERFACE_LINK_LIBRARIES "Arrow::arrow_depends;Arrow::arrow_static;Arrow::arrow_compute;Arrow::arrow_acero"
)
add_dependencies(Arrow::arrow_dataset ARROW.BUILD)
