set(RABITQ_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RaBitQ-Library-0.1)

add_library(rabitqlib INTERFACE)
target_include_directories(
    rabitqlib SYSTEM INTERFACE "${RABITQ_SOURCE_DIR}/include"
  )

if(RABITQ_SUPPORTED)
  set(RABITQ_LUT_REUSE_PATCH
      "${CMAKE_CURRENT_SOURCE_DIR}/rabitq_lut_reuse.patch")
  apply_patch_once(
      "rabitq_lut_reuse"
      "${RABITQ_SOURCE_DIR}"
      "${RABITQ_LUT_REUSE_PATCH}"
      --unidiff-zero)

  set(RABITQ_DISPATCH_PATCH
      "${CMAKE_CURRENT_SOURCE_DIR}/rabitq.defer_simd_errors.patch")
  apply_patch_once(
      "rabitq_defer_simd_errors"
      "${RABITQ_SOURCE_DIR}"
      "${RABITQ_DISPATCH_PATCH}")

  set(RABITQ_COMMON_SOURCES
      "${RABITQ_SOURCE_DIR}/src/utils/cpu_features.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/dispatch.cpp"
  )
  set(RABITQ_AVX2_SOURCES
      "${RABITQ_SOURCE_DIR}/src/simd/pack_excode_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_excode_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/fastscan_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/warmup_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/rotator_avx2.cpp"
  )
  set(RABITQ_AVX512_SOURCES
      "${RABITQ_SOURCE_DIR}/src/simd/pack_excode_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_excode_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/fastscan_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/rotator_avx512.cpp"
  )
  set(RABITQ_AVX512_POPCNT_SOURCES
      "${RABITQ_SOURCE_DIR}/src/simd/warmup_avx512.cpp"
  )

  add_library(
      rabitqlib_runtime STATIC
      ${RABITQ_COMMON_SOURCES}
      ${RABITQ_AVX2_SOURCES}
      ${RABITQ_AVX512_SOURCES}
      ${RABITQ_AVX512_POPCNT_SOURCES}
  )
  target_include_directories(
      rabitqlib_runtime SYSTEM PUBLIC "${RABITQ_SOURCE_DIR}/include"
  )
  set_target_properties(
      rabitqlib_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

  set_source_files_properties(
      ${RABITQ_AVX2_SOURCES}
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
  )
  set_source_files_properties(
      ${RABITQ_AVX512_SOURCES}
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mfma"
  )
  set_source_files_properties(
      ${RABITQ_AVX512_POPCNT_SOURCES}
      PROPERTIES
      COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mfma"
  )

  target_link_libraries(rabitqlib INTERFACE rabitqlib_runtime)
endif()
