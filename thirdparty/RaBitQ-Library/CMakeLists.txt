set(RABITQ_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RaBitQ-Library-0.1)
set(RABITQ_LUT_REUSE_PATCH
    ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_lut_reuse.patch)
apply_patch_once("rabitq_lut_reuse" "${RABITQ_SRC_DIR}"
                 "${RABITQ_LUT_REUSE_PATCH}" --unidiff-zero)

add_library(rabitqlib INTERFACE)
target_include_directories(
    rabitqlib SYSTEM INTERFACE "${RABITQ_SRC_DIR}/include"
  )
