include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)


set(ZVEC_DB_TEST_LIBS
  zvec_db
  # Vector index implementations register themselves through static factory
  # initializers.  Keep these targets explicit so collection/segment tests can
  # create vector indexes through the normal db path.
  core_metric
  core_utility
  core_quantizer
  core_knn_flat
  core_knn_flat_sparse
  core_knn_hnsw
  core_knn_hnsw_sparse
  core_knn_ivf
  core_knn_hnsw_rabitq
  core_knn_vamana
  core_mix_reducer
)


function(db_gtests DB_TEST_DIR)
  file(GLOB DB_TEST_SRCS *_test.cc)

  foreach(DB_TEST_SRC ${DB_TEST_SRCS})
    get_filename_component(DB_TEST_NAME ${DB_TEST_SRC} NAME_WE)
    set(DB_TEST_TARGET db_${DB_TEST_DIR}_${DB_TEST_NAME})

    set(DB_TEST_COST_SUITE zvec_db_default)
    set(DB_TEST_COST_LABEL default)
    if(DB_TEST_SRC MATCHES "_extended_test\\.cc$")
      set(DB_TEST_COST_SUITE zvec_db_extended)
      set(DB_TEST_COST_LABEL extended)
    endif()

    cc_gtest(
      NAME ${DB_TEST_TARGET} STRICT
      LIBS ${ZVEC_DB_TEST_LIBS}
      SRCS ${DB_TEST_SRC}
      INCS .
           ${PROJECT_ROOT_DIR}/tests/db_new
           ${PROJECT_ROOT_DIR}/src
           ${PROJECT_ROOT_DIR}/src/include
    )

    cc_test_suite(zvec_db ${DB_TEST_TARGET})
    cc_test_suite(zvec_db_${DB_TEST_DIR} ${DB_TEST_TARGET})
    cc_test_suite(${DB_TEST_COST_SUITE} ${DB_TEST_TARGET})
    set_tests_properties(${DB_TEST_TARGET} PROPERTIES
      LABELS
      "zvec_db;zvec_db_${DB_TEST_DIR};${DB_TEST_COST_SUITE};${DB_TEST_COST_LABEL}")
  endforeach()
endfunction()
