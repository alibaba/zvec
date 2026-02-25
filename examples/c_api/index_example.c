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
static ZVecErrorCode handle_error(ZVecErrorCode error, const char *context) {
  if (error != ZVEC_OK) {
    char *error_msg = NULL;
    zvec_get_last_error(&error_msg);
    fprintf(stderr, "Error in %s: %d - %s\n", context, error,
            error_msg ? error_msg : "Unknown error");
    free(error_msg);
  }
  return error;
}

/**
 * @brief Index creation and management example
 */
int main() {
  printf("=== ZVec Index Example ===\n\n");

  ZVecErrorCode error;

  // 1. Create collection schema
  ZVecCollectionSchema *schema =
      zvec_collection_schema_create("index_example_collection");
  if (!schema) {
    fprintf(stderr, "Failed to create collection schema\n");
    return -1;
  }
  printf("✓ Collection schema created successfully\n");

  // 2. Create different index parameter configurations
  printf("Creating index parameters...\n");

  // Inverted index parameters
  ZVecInvertIndexParams *invert_params_standard =
      zvec_index_params_invert_create(true, false);
  ZVecInvertIndexParams *invert_params_extended =
      zvec_index_params_invert_create(true, true);

  // HNSW index parameters with different configurations
  ZVecHnswIndexParams *hnsw_params_fast = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
  ZVecHnswIndexParams *hnsw_params_balanced = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED, 32, 200, 100);
  ZVecHnswIndexParams *hnsw_params_accurate = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_IP, ZVEC_QUANTIZE_TYPE_UNDEFINED, 64, 400, 200);

  // Flat index parameters
  ZVecFlatIndexParams *flat_params_l2 = zvec_index_params_flat_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  ZVecFlatIndexParams *flat_params_cosine = zvec_index_params_flat_create(
      ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED);

  if (!invert_params_standard || !invert_params_extended || !hnsw_params_fast ||
      !hnsw_params_balanced || !hnsw_params_accurate || !flat_params_l2 ||
      !flat_params_cosine) {
    fprintf(stderr, "Failed to create index parameters\n");
    zvec_collection_schema_destroy(schema);
    return -1;
  }

  // 3. Create fields with different index types
  printf("Creating fields with various index types...\n");

  // Fields with inverted indexes
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  if (id_field) {
    zvec_field_schema_set_invert_index(id_field, invert_params_standard);
    error = zvec_collection_schema_add_field(schema, id_field);
    if (handle_error(error, "adding ID field") == ZVEC_OK) {
      printf("✓ ID field with standard inverted index added\n");
    }
  }

  ZVecFieldSchema *category_field =
      zvec_field_schema_create("category", ZVEC_DATA_TYPE_STRING, true, 0);
  if (category_field) {
    zvec_field_schema_set_invert_index(category_field, invert_params_extended);
    error = zvec_collection_schema_add_field(schema, category_field);
    if (handle_error(error, "adding category field") == ZVEC_OK) {
      printf("✓ Category field with extended inverted index added\n");
    }
  }

  // Vector fields with HNSW indexes (different configurations)
  ZVecFieldSchema *fast_search_field = zvec_field_schema_create(
      "fast_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 64);
  if (fast_search_field) {
    zvec_field_schema_set_hnsw_index(fast_search_field, hnsw_params_fast);
    error = zvec_collection_schema_add_field(schema, fast_search_field);
    if (handle_error(error, "adding fast search field") == ZVEC_OK) {
      printf("✓ Fast search vector field (64D) with HNSW index added\n");
    }
  }

  ZVecFieldSchema *balanced_field = zvec_field_schema_create(
      "balanced_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  if (balanced_field) {
    zvec_field_schema_set_hnsw_index(balanced_field, hnsw_params_balanced);
    error = zvec_collection_schema_add_field(schema, balanced_field);
    if (handle_error(error, "adding balanced field") == ZVEC_OK) {
      printf("✓ Balanced vector field (128D) with HNSW index added\n");
    }
  }

  ZVecFieldSchema *accurate_field = zvec_field_schema_create(
      "accurate_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 256);
  if (accurate_field) {
    zvec_field_schema_set_hnsw_index(accurate_field, hnsw_params_accurate);
    error = zvec_collection_schema_add_field(schema, accurate_field);
    if (handle_error(error, "adding accurate field") == ZVEC_OK) {
      printf("✓ Accurate vector field (256D) with HNSW index added\n");
    }
  }

  // Vector field with Flat index
  ZVecFieldSchema *exact_field = zvec_field_schema_create(
      "exact_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 32);
  if (exact_field) {
    zvec_field_schema_set_flat_index(exact_field, flat_params_l2);
    error = zvec_collection_schema_add_field(schema, exact_field);
    if (handle_error(error, "adding exact field") == ZVEC_OK) {
      printf("✓ Exact search vector field (32D) with Flat index added\n");
    }
  }

  // 4. Create collection
  ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();
  ZVecCollection *collection = NULL;

  error = zvec_collection_create_and_open("./index_example_collection", schema,
                                          &options, &collection);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    // Cleanup index parameters
    zvec_index_params_invert_destroy(invert_params_standard);
    zvec_index_params_invert_destroy(invert_params_extended);
    zvec_index_params_hnsw_destroy(hnsw_params_fast);
    zvec_index_params_hnsw_destroy(hnsw_params_balanced);
    zvec_index_params_hnsw_destroy(hnsw_params_accurate);
    zvec_index_params_flat_destroy(flat_params_l2);
    zvec_index_params_flat_destroy(flat_params_cosine);
    return -1;
  }
  printf("✓ Collection created successfully\n");

  // 5. Create test data
  printf("Creating test documents...\n");

  ZVecDoc *docs[3];
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
  error = zvec_collection_insert(collection, (const ZVecDoc **)docs, 3,
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
  ZVecVectorQuery hnsw_query = {0};
  hnsw_query.field_name = (ZVecString){.data = "balanced_vector",
                                       .length = strlen("balanced_vector")};
  hnsw_query.query_vector = (ZVecByteArray){.data = (uint8_t *)balanced_vec[0],
                                            .length = 128 * sizeof(float)};
  hnsw_query.topk = 2;
  hnsw_query.filter = (ZVecString){.data = "", .length = 0};
  hnsw_query.include_vector = false;
  hnsw_query.include_doc_id = true;
  hnsw_query.output_fields.strings = NULL;
  hnsw_query.output_fields.count = 0;

  ZVecDoc **hnsw_results = NULL;
  size_t hnsw_result_count = 0;
  error = zvec_collection_query(collection, &hnsw_query, &hnsw_results,
                                &hnsw_result_count);
  if (error == ZVEC_OK) {
    printf("✓ HNSW query successful - Found %zu results\n", hnsw_result_count);
    zvec_docs_free(hnsw_results, hnsw_result_count);
  }

  // Test Flat query (exact)
  ZVecVectorQuery flat_query = {0};
  flat_query.field_name =
      (ZVecString){.data = "exact_vector", .length = strlen("exact_vector")};
  flat_query.query_vector = (ZVecByteArray){.data = (uint8_t *)exact_vec[0],
                                            .length = 32 * sizeof(float)};
  flat_query.topk = 2;
  flat_query.filter = (ZVecString){.data = "", .length = 0};
  flat_query.include_vector = false;
  flat_query.include_doc_id = true;
  flat_query.output_fields.strings = NULL;
  flat_query.output_fields.count = 0;

  ZVecDoc **flat_results = NULL;
  size_t flat_result_count = 0;
  error = zvec_collection_query(collection, &flat_query, &flat_results,
                                &flat_result_count);
  if (error == ZVEC_OK) {
    printf("✓ Flat (exact) query successful - Found %zu results\n",
           flat_result_count);
    zvec_docs_free(flat_results, flat_result_count);
  }

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
  zvec_index_params_invert_destroy(invert_params_standard);
  zvec_index_params_invert_destroy(invert_params_extended);
  zvec_index_params_hnsw_destroy(hnsw_params_fast);
  zvec_index_params_hnsw_destroy(hnsw_params_balanced);
  zvec_index_params_hnsw_destroy(hnsw_params_accurate);
  zvec_index_params_flat_destroy(flat_params_l2);
  zvec_index_params_flat_destroy(flat_params_cosine);

  printf("✓ Index example completed\n");
  return 0;
}