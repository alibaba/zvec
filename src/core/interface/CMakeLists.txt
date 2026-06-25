include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB CORE_INTERFACE_SRCS *.cc indexes/*.cc)
if(NOT ZVEC_ENABLE_OMEGA)
  list(FILTER CORE_INTERFACE_SRCS EXCLUDE REGEX ".*/indexes/omega_index\\.cc$")
  list(FILTER CORE_INTERFACE_SRCS EXCLUDE REGEX ".*/indexes/omega_training_session\\.cc$")
endif()

set(CORE_INTERFACE_INCS
    .
    ${PROJECT_ROOT_DIR}/src/include
    ${PROJECT_ROOT_DIR}/src/
    ${PROJECT_ROOT_DIR}/src/core)
if(ZVEC_ENABLE_OMEGA)
  list(APPEND CORE_INTERFACE_INCS ${PROJECT_ROOT_DIR}/thirdparty/omega/OMEGALib/include)
endif()

set(CORE_INTERFACE_LIBS
    zvec_ailego
    core_framework
    core_mix_reducer
    core_knn_hnsw_rabitq
    core_plugin
    sparsehash
    magic_enum
    rabitqlib)
if(ZVEC_ENABLE_OMEGA)
  list(APPEND CORE_INTERFACE_LIBS core_knn_omega)
endif()

cc_library(
        NAME core_interface STATIC STRICT ALWAYS_LINK
        SRCS ${CORE_INTERFACE_SRCS}
        INCS ${CORE_INTERFACE_INCS}
        LIBS ${CORE_INTERFACE_LIBS}
        VERSION "${PROXIMA_ZVEC_VERSION}"
)

if(ZVEC_ENABLE_OMEGA AND TARGET core_interface)
  target_compile_options(core_interface PRIVATE
      "$<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>"
      "$<$<CXX_COMPILER_ID:AppleClang>:-Wno-unknown-pragmas>"
      "$<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>"
    )
endif()
