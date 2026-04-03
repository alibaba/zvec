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

/**
 * @brief Print error message and return error code
 */
static zvec_error_code_t handle_error(zvec_error_code_t error,
                                      const char *context) {
  if (error != ZVEC_OK) {
    char *error_msg = NULL;
    zvec_get_last_error(&error_msg);
    fprintf(stderr, "Error in %s: %d - %s\n", context, error,
            error_msg ? error_msg : "Unknown error");
    zvec_free(error_msg);
  }
  return error;
}

/**
 * @brief Index creation and management example
 */
int main() {
  printf("=== ZVec Index Example ===\n\n");

  zvec_error_code_t error;

  // 1. Create collection schema
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("index_example_collection");
  if (!schema) {
    fprintf(stderr, "Failed to create collection schema\n");
    return -1;
  }
  printf("✓ Collection schema created successfully\n");

  // 2. Create different index parameter configurations
  printf("Creating index parameters...\n");

  // Inverted index parameters
  zvec_index_params_t *invert_params_standard =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  if (!invert_params_standard) {
    fprintf(stderr, "Failed to create invert index parameters (standard)\n");
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_invert_params(invert_params_standard, true, false);

  zvec_index_params_t *invert_params_extended =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  if (!invert_params_extended) {
    fprintf(stderr, "Failed to create invert index parameters (extended)\n");
    zvec_index_params_destroy(invert_params_standard);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_invert_params(invert_params_extended, true, true);

  // HNSW index parameters with different configurations
  zvec_index_params_t *hnsw_params_fast =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  if (!hnsw_params_fast) {
    fprintf(stderr, "Failed to create HNSW index parameters (fast)\n");
    zvec_index_params_destroy(invert_params_standard);
    zvec_index_params_destroy(invert_params_extended);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(hnsw_params_fast, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(hnsw_params_fast, 16, 100);

  zvec_index_params_t *hnsw_params_balanced =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  if (!hnsw_params_balanced) {
    fprintf(stderr, "Failed to create HNSW index parameters (balanced)\n");
    zvec_index_params_destroy(invert_params_standard);
    zvec_index_params_destroy(invert_params_extended);
    zvec_index_params_destroy(hnsw_params_fast);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(hnsw_params_balanced,
                                    ZVEC_METRIC_TYPE_COSINE);
  zvec_index_params_set_hnsw_params(hnsw_params_balanced, 32, 200);

  zvec_index_params_t *hnsw_params_accurate =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  if (!hnsw_params_accurate) {
    fprintf(stderr, "Failed to create HNSW index parameters (accurate)\n");
    zvec_index_params_destroy(invert_params_standard);
    zvec_index_params_destroy(invert_params_extended);
    zvec_index_params_destroy(hnsw_params_fast);
    zvec_index_params_destroy(hnsw_params_balanced);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(hnsw_params_accurate, ZVEC_METRIC_TYPE_IP);
  zvec_index_params_set_hnsw_params(hnsw_params_accurate, 64, 400);

  // Flat index parameters
  zvec_index_params_t *flat_params_l2 =
      zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
  if (!flat_params_l2) {
    fprintf(stderr, "Failed to create Flat index parameters (L2)\n");
    zvec_index_params_destroy(invert_params_standard);
    zvec_index_params_destroy(invert_params_extended);
    zvec_index_params_destroy(hnsw_params_fast);
    zvec_index_params_destroy(hnsw_params_balanced);
    zvec_index_params_destroy(hnsw_params_accurate);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(flat_params_l2, ZVEC_METRIC_TYPE_L2);

  zvec_index_params_t *flat_params_cosine =
      zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
  if (!flat_params_cosine) {
    fprintf(stderr, "Failed to create Flat index parameters (cosine)\n");
    zvec_index_params_destroy(invert_params_standard);
    zvec_index_params_destroy(invert_params_extended);
    zvec_index_params_destroy(hnsw_params_fast);
    zvec_index_params_destroy(hnsw_params_balanced);
    zvec_index_params_destroy(hnsw_params_accurate);
    zvec_index_params_destroy(flat_params_l2);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(flat_params_cosine,
                                    ZVEC_METRIC_TYPE_COSINE);

  // 3. Create fields with different index types
  printf("Creating fields with various index types...\n");

  // Fields with inverted indexes
  zvec_field_schema_t *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  if (id_field) {
    zvec_field_schema_set_index_params(id_field, invert_params_standard);
    error = zvec_collection_schema_add_field(schema, id_field);
    if (handle_error(error, "adding ID field") == ZVEC_OK) {
      printf("✓ ID field with standard inverted index added\n");
    }
  }

  zvec_field_schema_t *category_field =
      zvec_field_schema_create("category", ZVEC_DATA_TYPE_STRING, true, 0);
  if (category_field) {
    zvec_field_schema_set_index_params(category_field, invert_params_extended);
    error = zvec_collection_schema_add_field(schema, category_field);
    if (handle_error(error, "adding category field") == ZVEC_OK) {
      printf("✓ Category field with extended inverted index added\n");
    }
  }

  // Vector fields with HNSW indexes (different configurations)
  zvec_field_schema_t *fast_search_field = zvec_field_schema_create(
      "fast_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 64);
  if (fast_search_field) {
    zvec_field_schema_set_index_params(fast_search_field, hnsw_params_fast);
    error = zvec_collection_schema_add_field(schema, fast_search_field);
    if (handle_error(error, "adding fast search field") == ZVEC_OK) {
      printf("✓ Fast search vector field (64D) with HNSW index added\n");
    }
  }

  zvec_field_schema_t *balanced_field = zvec_field_schema_create(
      "balanced_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  if (balanced_field) {
    zvec_field_schema_set_index_params(balanced_field, hnsw_params_balanced);
    error = zvec_collection_schema_add_field(schema, balanced_field);
    if (handle_error(error, "adding balanced field") == ZVEC_OK) {
      printf("✓ Balanced vector field (128D) with HNSW index added\n");
    }
  }

  zvec_field_schema_t *accurate_field = zvec_field_schema_create(
      "accurate_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 256);
  if (accurate_field) {
    zvec_field_schema_set_index_params(accurate_field, hnsw_params_accurate);
    error = zvec_collection_schema_add_field(schema, accurate_field);
    if (handle_error(error, "adding accurate field") == ZVEC_OK) {
      printf("✓ Accurate vector field (256D) with HNSW index added\n");
    }
  }

  // Vector field with Flat index
  zvec_field_schema_t *exact_field = zvec_field_schema_create(
      "exact_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 32);
  if (exact_field) {
    zvec_field_schema_set_index_params(exact_field, flat_params_l2);
    error = zvec_collection_schema_add_field(schema, exact_field);
    if (handle_error(error, "adding exact field") == ZVEC_OK) {
      printf("✓ Exact search vector field (32D) with Flat index added\n");
    }
  }

  // 4. Create collection
  zvec_collection_options_t *options = zvec_collection_options_create();
  if (!options) {
    fprintf(stderr, "Failed to create collection options\n");
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_collection_t *collection = NULL;

  error = zvec_collection_create_and_open("./index_example_collection", schema,
                                          options, &collection);
  zvec_collection_options_destroy(options);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  printf("✓ Collection created successfully\n");

  // 5. Create test data
  printf("Creating test documents...\n");

  zvec_doc_t *docs[3];
  for (int i = 0; i < 3; i++) {
    docs[i] = zvec_doc_create();
    if (!docs[i]) {
      fprintf(stderr, "Failed to create document %d\n", i);
      // Cleanup
      for (int j = 0; j < i; j++) {
        zvec_doc_destroy(docs[j]);
      }
      goto cleanup;
    }
  }

  // Prepare vector data
  float fast_vec[3][64];
  float balanced_vec[3][128];
  float accurate_vec[3][256];
  float exact_vec[3][32];

  // Generate different vector patterns for testing
  for (int doc_idx = 0; doc_idx < 3; doc_idx++) {
    for (int i = 0; i < 64; i++) {
      fast_vec[doc_idx][i] = (float)(doc_idx * 64 + i) / (64.0f * 3.0f);
    }
    for (int i = 0; i < 128; i++) {
      balanced_vec[doc_idx][i] = (float)(doc_idx * 128 + i) / (128.0f * 3.0f);
    }
    for (int i = 0; i < 256; i++) {
      accurate_vec[doc_idx][i] = (float)(doc_idx * 256 + i) / (256.0f * 3.0f);
    }
    for (int i = 0; i < 32; i++) {
      exact_vec[doc_idx][i] = (float)(doc_idx * 32 + i) / (32.0f * 3.0f);
    }
  }

  // Populate documents
  for (int i = 0; i < 3; i++) {
    char pk[16];
    snprintf(pk, sizeof(pk), "doc%d", i + 1);
    zvec_doc_set_pk(docs[i], pk);

    char id_val[16];
    snprintf(id_val, sizeof(id_val), "ID_%d", i + 1);
    zvec_doc_add_field_by_value(docs[i], "id", ZVEC_DATA_TYPE_STRING, id_val,
                                strlen(id_val));

    char category_val[16];
    snprintf(category_val, sizeof(category_val), "cat_%d", (i % 2) + 1);
    zvec_doc_add_field_by_value(docs[i], "category", ZVEC_DATA_TYPE_STRING,
                                category_val, strlen(category_val));

    zvec_doc_add_field_by_value(docs[i], "fast_vector",
                                ZVEC_DATA_TYPE_VECTOR_FP32, fast_vec[i],
                                64 * sizeof(float));
    zvec_doc_add_field_by_value(docs[i], "balanced_vector",
                                ZVEC_DATA_TYPE_VECTOR_FP32, balanced_vec[i],
                                128 * sizeof(float));
    zvec_doc_add_field_by_value(docs[i], "accurate_vector",
                                ZVEC_DATA_TYPE_VECTOR_FP32, accurate_vec[i],
                                256 * sizeof(float));
    zvec_doc_add_field_by_value(docs[i], "exact_vector",
                                ZVEC_DATA_TYPE_VECTOR_FP32, exact_vec[i],
                                32 * sizeof(float));
  }

  // 6. Insert documents
  size_t success_count = 0, error_count = 0;
  error = zvec_collection_insert(collection, (const zvec_doc_t **)docs, 3,
                                 &success_count, &error_count);
  if (handle_error(error, "inserting documents") == ZVEC_OK) {
    printf("✓ Documents inserted - Success: %zu, Failed: %zu\n", success_count,
           error_count);
  }

  // Cleanup documents
  for (int i = 0; i < 3; i++) {
    zvec_doc_destroy(docs[i]);
  }

  // 7. Flush collection to build indexes
  error = zvec_collection_flush(collection);
  if (handle_error(error, "flushing collection") == ZVEC_OK) {
    printf("✓ Collection flushed - indexes built\n");
  }

  // 8. Test different query types
  printf("Testing various index queries...\n");

  // Test HNSW query (balanced)
  zvec_vector_query_t *hnsw_query = zvec_vector_query_create();
  if (!hnsw_query) {
    fprintf(stderr, "Failed to create HNSW query\n");
    goto cleanup;
  }
  zvec_vector_query_set_field_name(hnsw_query, "balanced_vector");
  zvec_vector_query_set_query_vector(hnsw_query, balanced_vec[0],
                                     128 * sizeof(float));
  zvec_vector_query_set_topk(hnsw_query, 2);
  zvec_vector_query_set_filter(hnsw_query, "");
  zvec_vector_query_set_include_vector(hnsw_query, false);
  zvec_vector_query_set_include_doc_id(hnsw_query, true);

  zvec_doc_t **hnsw_results = NULL;
  size_t hnsw_result_count = 0;
  error =
      zvec_collection_query(collection, (const zvec_vector_query_t *)hnsw_query,
                            &hnsw_results, &hnsw_result_count);
  if (error == ZVEC_OK) {
    printf("✓ HNSW query successful - Found %zu results\n", hnsw_result_count);
    zvec_docs_free(hnsw_results, hnsw_result_count);
  }
  zvec_vector_query_destroy(hnsw_query);

  // Test Flat query (exact)
  zvec_vector_query_t *flat_query = zvec_vector_query_create();
  if (!flat_query) {
    fprintf(stderr, "Failed to create Flat query\n");
    goto cleanup;
  }
  zvec_vector_query_set_field_name(flat_query, "exact_vector");
  zvec_vector_query_set_query_vector(flat_query, exact_vec[0],
                                     32 * sizeof(float));
  zvec_vector_query_set_topk(flat_query, 2);
  zvec_vector_query_set_filter(flat_query, "");
  zvec_vector_query_set_include_vector(flat_query, false);
  zvec_vector_query_set_include_doc_id(flat_query, true);

  zvec_doc_t **flat_results = NULL;
  size_t flat_result_count = 0;
  error =
      zvec_collection_query(collection, (const zvec_vector_query_t *)flat_query,
                            &flat_results, &flat_result_count);
  if (error == ZVEC_OK) {
    printf("✓ Flat (exact) query successful - Found %zu results\n",
           flat_result_count);
    zvec_docs_free(flat_results, flat_result_count);
  }
  zvec_vector_query_destroy(flat_query);

  // 9. Performance comparison information
  printf("\nIndex Performance Characteristics:\n");
  printf("- Inverted Index: Fast text search, supports filtering\n");
  printf(
      "- HNSW Index: Approximate nearest neighbor search, good balance of "
      "speed/accuracy\n");
  printf("- Flat Index: Exact search, slower but 100%% accurate\n");
  printf(
      "- Trade-off: Speed vs Accuracy - choose based on your requirements\n");

  // 10. Cleanup
cleanup:
  zvec_collection_destroy(collection);
  zvec_collection_schema_destroy(schema);

  // Cleanup index parameters

  printf("✓ Index example completed\n");
  return 0;
}