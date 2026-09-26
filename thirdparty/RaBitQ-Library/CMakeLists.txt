set(RABITQ_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RaBitQ-Library-0.3.8)

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

  # src/index/ivf_search.cpp and src/index/qg.cpp back rabitqlib's own IVF and
  # SymphonyQG indexes, which zvec reimplements; qg.cpp would also drag in an
  # OpenMP dependency. dispatch.cpp does odr-use the HNSW search kernels while
  # resolving its function pointers, so those three are required.
  set(RABITQ_COMMON_SOURCES
      "${RABITQ_SOURCE_DIR}/src/utils/cpu_features.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/dispatch.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_generic.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/fastscan_generic.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/quantization_generic.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/estimator_generic.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/matrix_generic.cpp"
  )
  set(RABITQ_AVX2_SOURCES
      "${RABITQ_SOURCE_DIR}/src/simd/quantization_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/pack_excode_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_excode_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/fastscan_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/estimator_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/matrix_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/warmup_avx2.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/rotator_avx2.cpp"
  )
  set(RABITQ_AVX512_SOURCES
      "${RABITQ_SOURCE_DIR}/src/simd/quantization_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/pack_excode_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_excode_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/space_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/fastscan_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/estimator_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/matrix_avx512.cpp"
      "${RABITQ_SOURCE_DIR}/src/simd/rotator_avx512.cpp"
  )
  set(RABITQ_AVX512_POPCNT_SOURCES
      "${RABITQ_SOURCE_DIR}/src/simd/warmup_avx512.cpp"
  )
  set(RABITQ_HNSW_AVX2_SOURCES
      "${RABITQ_SOURCE_DIR}/src/index/hnsw_search_avx2.cpp"
  )
  set(RABITQ_HNSW_AVX512_CORE_SOURCES
      "${RABITQ_SOURCE_DIR}/src/index/hnsw_search_avx512_core.cpp"
  )
  set(RABITQ_HNSW_AVX512_POPCNT_SOURCES
      "${RABITQ_SOURCE_DIR}/src/index/hnsw_search_avx512_popcnt.cpp"
  )

  add_library(
      rabitqlib_runtime STATIC
      ${RABITQ_COMMON_SOURCES}
      ${RABITQ_AVX2_SOURCES}
      ${RABITQ_AVX512_SOURCES}
      ${RABITQ_AVX512_POPCNT_SOURCES}
      ${RABITQ_HNSW_AVX2_SOURCES}
      ${RABITQ_HNSW_AVX512_CORE_SOURCES}
      ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
  )
  target_include_directories(
      rabitqlib_runtime SYSTEM PUBLIC "${RABITQ_SOURCE_DIR}/include"
  )
  set_target_properties(
      rabitqlib_runtime PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
  )

  if(MSVC)
    # MSVC spells the GNU restrict extension differently, and rabitqlib's public
    # headers use __restrict__. PUBLIC so consumers see the same spelling.
    target_compile_definitions(rabitqlib_runtime PUBLIC __restrict__=__restrict)
    # Eigen's inline code must have one ISA-independent definition across the
    # generic, AVX2 and AVX-512 translation units.
    target_compile_definitions(rabitqlib_runtime PUBLIC EIGEN_DONT_VECTORIZE)

    # MSVC has no inline asm, so the AVX2 rotator calls the FHT kernels that
    # this generated MASM source provides.
    target_sources(
        rabitqlib_runtime PRIVATE
        "${RABITQ_SOURCE_DIR}/src/simd/fht_avx2_masm.asm"
    )
    set_source_files_properties(
        "${RABITQ_SOURCE_DIR}/src/simd/fht_avx2_masm.asm"
        PROPERTIES LANGUAGE ASM_MASM
    )
    # /arch:AVX512 enables the whole F/CD/VL/BW/DQ group; the optional popcount
    # backend stays guarded by the runtime CPUID check in cpu_features.cpp.
    set_source_files_properties(
        ${RABITQ_AVX2_SOURCES}
        ${RABITQ_HNSW_AVX2_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX2_FLAGS}"
    )
    set_source_files_properties(
        ${RABITQ_AVX512_SOURCES}
        ${RABITQ_AVX512_POPCNT_SOURCES}
        ${RABITQ_HNSW_AVX512_CORE_SOURCES}
        ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX512_FLAGS}"
    )
  else()
    set_source_files_properties(
        ${RABITQ_AVX2_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX2_FLAGS}"
    )
    set_source_files_properties(
        ${RABITQ_AVX512_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX512_FLAGS}"
    )
    set_source_files_properties(
        ${RABITQ_AVX512_POPCNT_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX512_POPCNT_FLAGS}"
    )
    set_source_files_properties(
        ${RABITQ_HNSW_AVX2_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX2_FLAGS}"
    )
    set_source_files_properties(
        ${RABITQ_HNSW_AVX512_CORE_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_HNSW_AVX512_CORE_FLAGS}"
    )
    set_source_files_properties(
        ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
        PROPERTIES COMPILE_OPTIONS "${RABITQ_AVX512_POPCNT_FLAGS}"
    )
  endif()

  target_link_libraries(rabitqlib INTERFACE rabitqlib_runtime)
endif()
