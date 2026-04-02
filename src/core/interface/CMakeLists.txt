include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
        NAME core_interface STATIC STRICT ALWAYS_LINK
        SRCS *.cc indexes/*.cc
        INCS . ${PROJECT_ROOT_DIR}/src/include ${PROJECT_ROOT_DIR}/src/ ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/thirdparty/omega/OMEGALib/include
        LIBS zvec_ailego core_framework core_mix_reducer core_knn_omega core_knn_hnsw_rabitq sparsehash magic_enum rabitqlib
        VERSION "${PROXIMA_ZVEC_VERSION}"
)
