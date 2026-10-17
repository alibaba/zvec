// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zvec/c_api.h"

static zvec_error_code_t check(zvec_error_code_t error, const char *context) {
  if (error != ZVEC_OK) {
    char *error_msg = NULL;
    zvec_get_last_error(&error_msg);
    fprintf(stderr, "%s failed: %s\n", context,
            error_msg ? error_msg : "unknown error");
    zvec_free(error_msg);
  }
  return error;
}

static zvec_error_code_t create_omega_collection(zvec_collection_t **collection) {
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("omega_collection");
  zvec_collection_options_t *options = NULL;
  zvec_index_params_t *invert_params = NULL;
  zvec_index_params_t *omega_params = NULL;
  zvec_field_schema_t *id_field = NULL;
  zvec_field_schema_t *embedding_field = NULL;
  zvec_error_code_t error = ZVEC_OK;

  if (!schema) {
    return ZVEC_ERROR_INTERNAL_ERROR;
  }

  invert_params = zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  omega_params = zvec_index_params_create(ZVEC_INDEX_TYPE_OMEGA);
  options = zvec_collection_options_create();
  if (!invert_params || !omega_params || !options) {
    error = ZVEC_ERROR_RESOURCE_EXHAUSTED;
    goto cleanup;
  }

  zvec_index_params_set_invert_params(invert_params, true, false);
  zvec_index_params_set_metric_type(omega_params, ZVEC_METRIC_TYPE_IP);
  zvec_index_params_set_hnsw_params(omega_params, 8, 64);

  id_field = zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  embedding_field = zvec_field_schema_create(
      "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  if (!id_field || !embedding_field) {
    error = ZVEC_ERROR_RESOURCE_EXHAUSTED;
    goto cleanup;
  }

  zvec_field_schema_set_index_params(id_field, invert_params);
  zvec_field_schema_set_index_params(embedding_field, omega_params);

  error = zvec_collection_schema_add_field(schema, id_field);
  if (error != ZVEC_OK) goto cleanup;
  error = zvec_collection_schema_add_field(schema, embedding_field);
  if (error != ZVEC_OK) goto cleanup;

  error = zvec_collection_create_and_open("./omega_collection", schema, options,
                                          collection);

cleanup:
  zvec_index_params_destroy(invert_params);
  zvec_index_params_destroy(omega_params);
  zvec_collection_options_destroy(options);
  zvec_collection_schema_destroy(schema);
  return error;
}

int main(void) {
  zvec_collection_t *collection = NULL;
  zvec_vector_query_t *query = NULL;
  zvec_omega_query_params_t *omega_query = NULL;
  zvec_doc_t *docs[2] = {NULL, NULL};
  zvec_doc_t **results = NULL;
  size_t result_count = 0;
  int exit_code = 1;
  float vector1[] = {1.0f, 0.1f, 0.1f, 0.1f};
  float vector2[] = {2.0f, 0.2f, 0.2f, 0.2f};

  remove("./omega_collection");

  if (check(create_omega_collection(&collection), "create_omega_collection") !=
      ZVEC_OK) {
    goto cleanup;
  }

  for (int i = 0; i < 2; ++i) {
    docs[i] = zvec_doc_create();
    if (!docs[i]) {
      fprintf(stderr, "failed to create document %d\n", i);
      goto cleanup;
    }
  }

  zvec_doc_set_pk(docs[0], "doc1");
  zvec_doc_add_field_by_value(docs[0], "id", ZVEC_DATA_TYPE_STRING, "doc1", 4);
  zvec_doc_add_field_by_value(docs[0], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                              vector1, sizeof(vector1));

  zvec_doc_set_pk(docs[1], "doc2");
  zvec_doc_add_field_by_value(docs[1], "id", ZVEC_DATA_TYPE_STRING, "doc2", 4);
  zvec_doc_add_field_by_value(docs[1], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                              vector2, sizeof(vector2));

  {
    size_t success_count = 0;
    size_t error_count = 0;
    if (check(zvec_collection_insert(collection, (const zvec_doc_t **)docs, 2,
                                     &success_count, &error_count),
              "zvec_collection_insert") != ZVEC_OK) {
      goto cleanup;
    }
  }

  if (check(zvec_collection_flush(collection), "zvec_collection_flush") !=
      ZVEC_OK) {
    goto cleanup;
  }

  query = zvec_vector_query_create();
  omega_query = zvec_query_params_omega_create(32, 0.95f, 0.0f, false, false);
  if (!query || !omega_query) {
    fprintf(stderr, "failed to create omega query\n");
    goto cleanup;
  }

  zvec_vector_query_set_field_name(query, "embedding");
  zvec_vector_query_set_query_vector(query, vector1, sizeof(vector1));
  zvec_vector_query_set_topk(query, 2);
  zvec_vector_query_set_include_doc_id(query, true);
  zvec_vector_query_set_omega_params(query, omega_query);

  if (check(zvec_collection_query(collection, query, &results, &result_count),
            "zvec_collection_query") != ZVEC_OK) {
    goto cleanup;
  }

  printf("omega c example results: %zu\n", result_count);
  if (result_count == 0) {
    fprintf(stderr, "omega c example returned no results\n");
    goto cleanup;
  }
  printf("top result score: %.4f\n", zvec_doc_get_score(results[0]));

  exit_code = 0;

cleanup:
  zvec_docs_free(results, result_count);
  zvec_vector_query_destroy(query);
  if (collection) {
    zvec_collection_destroy(collection);
  }
  zvec_doc_destroy(docs[0]);
  zvec_doc_destroy(docs[1]);
  return exit_code;
}
