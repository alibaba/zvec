// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define CHECK(expr)                                         \
  do {                                                      \
    if (!(expr)) {                                          \
      char *error = NULL;                                   \
      zvec_get_last_error(&error);                          \
      fprintf(stderr, "line %d: %s: %s\n", __LINE__, #expr, \
              error ? error : "");                          \
      zvec_free(error);                                     \
      exit(1);                                              \
    }                                                       \
  } while (0)
#define OK(expr) CHECK((expr) == ZVEC_OK)

static void check_settings(const zvec_index_params_t *params, int bits) {
  int b, c, s, n;
  OK(zvec_index_params_get_rabitq_quantizer_params(params, &b, &c, &s, &n));
  CHECK(b == bits && c == 4 && s == 128 && n == 6);
}

static void test_params(void) {
  const zvec_index_type_t types[] = {ZVEC_INDEX_TYPE_HNSW, ZVEC_INDEX_TYPE_IVF,
                                     ZVEC_INDEX_TYPE_DISKANN};
  for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
    zvec_index_params_t *p = zvec_index_params_create(types[i]);
    CHECK(p != NULL);
    CHECK(zvec_index_params_set_rabitq_quantizer_params(p, 3, 4, 128, 6) ==
          ZVEC_ERROR_INVALID_ARGUMENT);
    OK(zvec_index_params_set_quantize_type(p, ZVEC_QUANTIZE_TYPE_RABITQ));
    OK(zvec_index_params_set_rabitq_quantizer_params(p, 3, 4, 128, 6));
    check_settings(p, 3);
    const int invalid[][4] = {{0, 4, 128, 6},
                              {10, 4, 128, 6},
                              {3, 0, 128, 6},
                              {3, 4, -1, 6},
                              {3, 4, 128, 0}};
    for (size_t j = 0; j < sizeof(invalid) / sizeof(invalid[0]); ++j) {
      CHECK(zvec_index_params_set_rabitq_quantizer_params(
                p, invalid[j][0], invalid[j][1], invalid[j][2],
                invalid[j][3]) == ZVEC_ERROR_INVALID_ARGUMENT);
      check_settings(p, 3);
    }
    OK(zvec_index_params_get_rabitq_quantizer_params(p, NULL, NULL, NULL,
                                                     NULL));
    if (types[i] == ZVEC_INDEX_TYPE_IVF) {
      int nlist, bits, samples;
      OK(zvec_index_params_get_ivf_rabitq_params(p, &nlist, &bits, &samples));
      CHECK(bits == 3 && samples == 128);
      OK(zvec_index_params_set_ivf_rabitq_params(p, 8, 9, 64));
      OK(zvec_index_params_get_rabitq_quantizer_params(p, &bits, NULL, &samples,
                                                       NULL));
      CHECK(bits == 9 && samples == 64);
    }
    zvec_index_params_destroy(p);
  }
  CHECK(zvec_index_params_get_rabitq_quantizer_params(
            NULL, NULL, NULL, NULL, NULL) == ZVEC_ERROR_INVALID_ARGUMENT);
  CHECK(zvec_index_params_set_rabitq_quantizer_params(NULL, 3, 4, 128, 6) ==
        ZVEC_ERROR_INVALID_ARGUMENT);
  zvec_index_params_t *native =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW_RABITQ);
  CHECK(native != NULL &&
        zvec_index_params_get_type(native) == ZVEC_INDEX_TYPE_HNSW_RABITQ);
  zvec_index_params_destroy(native);
}

#if RABITQ_SUPPORTED
static void test_collection(zvec_index_type_t type, zvec_metric_type_t metric,
                            int bits) {
  char path[128];
  snprintf(path, sizeof(path), "./c_api_turbo_rabitq_%d_%d_%d", (int)type,
           (int)metric, bits);
  zvec_test_delete_dir(path);
  zvec_index_params_t *p = zvec_index_params_create(type);
  CHECK(p != NULL);
  OK(zvec_index_params_set_metric_type(p, metric));
  OK(zvec_index_params_set_quantize_type(p, ZVEC_QUANTIZE_TYPE_RABITQ));
  OK(zvec_index_params_set_rabitq_quantizer_params(p, bits, 4, 128, 6));
  if (type == ZVEC_INDEX_TYPE_IVF)
    OK(zvec_index_params_set_ivf_params(p, 8, 10, false));
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("turbo_rabitq");
  zvec_field_schema_t *field =
      zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  CHECK(schema != NULL && field != NULL);
  OK(zvec_field_schema_set_index_params(field, p));
  OK(zvec_collection_schema_add_field(schema, field));
  zvec_collection_t *coll = NULL;
  OK(zvec_collection_create_and_open(path, schema, NULL, &coll));
  CHECK(coll != NULL);
  float query_vector[128];
  uint32_t state = 8121;
  for (int i = 0; i < 256; ++i) {
    float v[128];
    double norm = 0;
    for (int d = 0; d < 128; ++d) {
      state = state * 1664525u + 1013904223u;
      v[d] = ((float)(state >> 8) / 16777216.0f) - 0.5f;
      norm += (double)v[d] * v[d];
    }
    for (int d = 0; d < 128; ++d) v[d] = (float)(v[d] / sqrt(norm));
    if (i == 17) memcpy(query_vector, v, sizeof(v));
    char pk[32];
    snprintf(pk, sizeof(pk), "%d", i);
    zvec_doc_t *doc = zvec_doc_create();
    CHECK(doc != NULL);
    zvec_doc_set_pk(doc, pk);
    OK(zvec_doc_add_field_by_value(doc, "vec", ZVEC_DATA_TYPE_VECTOR_FP32, v,
                                   sizeof(v)));
    const zvec_doc_t *docs[] = {doc};
    size_t success = 0, errors = 0;
    OK(zvec_collection_insert(coll, docs, 1, &success, &errors));
    CHECK(success == 1 && errors == 0);
    zvec_doc_destroy(doc);
  }
  OK(zvec_collection_optimize(coll));
  zvec_collection_stats_t *stats = NULL;
  OK(zvec_collection_get_stats(coll, &stats));
  CHECK(stats != NULL && zvec_collection_stats_get_doc_count(stats) == 256);
  CHECK(zvec_collection_stats_get_index_count(stats) == 1);
  CHECK(zvec_collection_stats_get_index_completeness(stats, 0) == 1.0f);
  zvec_collection_stats_destroy(stats);
  zvec_vector_query_t *query = zvec_vector_query_create();
  CHECK(query != NULL);
  OK(zvec_vector_query_set_topk(query, 5));
  OK(zvec_vector_query_set_field_name(query, "vec"));
  OK(zvec_vector_query_set_query_vector(query, query_vector,
                                        sizeof(query_vector)));
  if (type == ZVEC_INDEX_TYPE_HNSW) {
    zvec_hnsw_query_params_t *q =
        zvec_query_params_hnsw_create(256, 0, false, true);
    CHECK(q != NULL);
    OK(zvec_vector_query_set_hnsw_params(query, q));
    // The query owns q after a successful setter.
  } else if (type == ZVEC_INDEX_TYPE_DISKANN) {
    zvec_diskann_query_params_t *q = zvec_query_params_diskann_create(256);
    CHECK(q != NULL);
    OK(zvec_vector_query_set_diskann_params(query, q));
  } else {
    zvec_ivf_query_params_t *q = zvec_query_params_ivf_create(8, true, 10);
    CHECK(q != NULL);
    OK(zvec_vector_query_set_ivf_params(query, q));
    // The query owns q after a successful setter.
  }
  char ids[5][32];
  for (int reopen = 0; reopen < 2; ++reopen) {
    zvec_collection_schema_t *loaded = NULL;
    OK(zvec_collection_get_schema(coll, &loaded));
    CHECK(loaded != NULL);
    const zvec_field_schema_t *f =
        zvec_collection_schema_get_vector_field(loaded, "vec");
    CHECK(f != NULL && zvec_field_schema_get_index_type(f) == type);
    check_settings(zvec_field_schema_get_index_params(f), bits);
    zvec_collection_schema_destroy(loaded);
    zvec_doc_t **results = NULL;
    size_t count = 0;
    OK(zvec_collection_query(coll, query, &results, &count));
    CHECK(count == 5);
    for (size_t i = 0; i < count; ++i) {
      const char *pk = zvec_doc_get_pk_copy(results[i]);
      CHECK(pk != NULL);
      if (i == 0) CHECK(strcmp(pk, "17") == 0);
      if (reopen == 0)
        snprintf(ids[i], sizeof(ids[i]), "%s", pk);
      else
        CHECK(strcmp(ids[i], pk) == 0);
      zvec_free((void *)pk);
      zvec_doc_destroy(results[i]);
    }
    zvec_free(results);
    if (!reopen) {
      OK(zvec_collection_close(coll));
      coll = NULL;
      OK(zvec_collection_open(path, NULL, &coll));
    }
  }
  zvec_vector_query_destroy(query);
  OK(zvec_collection_destroy(coll));
  OK(zvec_collection_close(coll));
  zvec_field_schema_destroy(field);
  zvec_collection_schema_destroy(schema);
  zvec_index_params_destroy(p);
}
#endif

int main(void) {
  zvec_config_data_t *config = zvec_config_data_create();
  CHECK(config != NULL);
  OK(zvec_initialize(config));
  zvec_config_data_destroy(config);
  test_params();
#if RABITQ_SUPPORTED
  const zvec_metric_type_t metrics[] = {
      ZVEC_METRIC_TYPE_L2, ZVEC_METRIC_TYPE_IP, ZVEC_METRIC_TYPE_COSINE};
  for (size_t i = 0; i < 3; ++i)
    for (int bits = 3; bits <= 9; bits += 6) {
      test_collection(ZVEC_INDEX_TYPE_HNSW, metrics[i], bits);
      test_collection(ZVEC_INDEX_TYPE_IVF, metrics[i], bits);
      test_collection(ZVEC_INDEX_TYPE_DISKANN, metrics[i], bits);
    }
  puts("RaBitQ C API parameters and 18 collection cases passed");
#else
  puts(
      "RaBitQ C API parameter tests passed; collection cases skipped on "
      "unsupported build");
#endif
  return 0;
}
