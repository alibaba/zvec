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
 * @brief Field schema creation and management example
 */
int main() {
  printf("=== ZVec Field Schema Example ===\n\n");

  zvec_error_code_t error;

  // 1. Create collection schema
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("field_example_collection");
  if (!schema) {
    fprintf(stderr, "Failed to create collection schema\n");
    return -1;
  }
  printf("✓ Collection schema created successfully\n");

  // 2. Create different types of index parameters
  zvec_index_params_t *invert_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  if (!invert_params) {
    fprintf(stderr, "Failed to create invert index parameters\n");
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_invert_params(invert_params, true, false);

  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  if (!hnsw_params) {
    fprintf(stderr, "Failed to create HNSW index parameters\n");
    zvec_index_params_destroy(invert_params);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_COSINE);
  zvec_index_params_set_hnsw_params(hnsw_params, 16, 200);

  zvec_index_params_t *flat_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
  if (!flat_params) {
    fprintf(stderr, "Failed to create Flat index parameters\n");
    zvec_index_params_destroy(invert_params);
    zvec_index_params_destroy(hnsw_params);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_index_params_set_metric_type(flat_params, ZVEC_METRIC_TYPE_L2);

  if (!invert_params || !hnsw_params || !flat_params) {
    fprintf(stderr, "Failed to create index parameters\n");
    zvec_index_params_destroy(invert_params);
    zvec_index_params_destroy(hnsw_params);
    zvec_index_params_destroy(flat_params);
    zvec_collection_schema_destroy(schema);
    return -1;
  }

  // 3. Create scalar fields with different data types
  printf("Creating scalar fields...\n");

  // String field with inverted index
  zvec_field_schema_t *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  if (name_field) {
    zvec_field_schema_set_index_params(name_field, invert_params);
    error = zvec_collection_schema_add_field(schema, name_field);
    if (handle_error(error, "adding name field") == ZVEC_OK) {
      printf("✓ String field 'name' with inverted index added\n");
    }
  }

  // Integer field
  zvec_field_schema_t *age_field =
      zvec_field_schema_create("age", ZVEC_DATA_TYPE_INT32, true, 0);
  if (age_field) {
    error = zvec_collection_schema_add_field(schema, age_field);
    if (handle_error(error, "adding age field") == ZVEC_OK) {
      printf("✓ Integer field 'age' added\n");
    }
  }

  // Float field
  zvec_field_schema_t *score_field =
      zvec_field_schema_create("score", ZVEC_DATA_TYPE_FLOAT, true, 0);
  if (score_field) {
    error = zvec_collection_schema_add_field(schema, score_field);
    if (handle_error(error, "adding score field") == ZVEC_OK) {
      printf("✓ Float field 'score' added\n");
    }
  }

  // Boolean field
  zvec_field_schema_t *active_field =
      zvec_field_schema_create("active", ZVEC_DATA_TYPE_BOOL, false, 0);
  if (active_field) {
    error = zvec_collection_schema_add_field(schema, active_field);
    if (handle_error(error, "adding active field") == ZVEC_OK) {
      printf("✓ Boolean field 'active' added\n");
    }
  }

  // 4. Create vector fields with different dimensions and indexes
  printf("Creating vector fields...\n");

  // Small dimension vector with HNSW index
  zvec_field_schema_t *small_vector_field = zvec_field_schema_create(
      "small_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 32);
  if (small_vector_field) {
    zvec_field_schema_set_index_params(small_vector_field, hnsw_params);
    error = zvec_collection_schema_add_field(schema, small_vector_field);
    if (handle_error(error, "adding small vector field") == ZVEC_OK) {
      printf(
          "✓ Small vector field 'small_vector' (32D) with HNSW index added\n");
    }
  }

  // Medium dimension vector with Flat index
  zvec_field_schema_t *medium_vector_field = zvec_field_schema_create(
      "medium_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  if (medium_vector_field) {
    zvec_field_schema_set_index_params(medium_vector_field, flat_params);
    error = zvec_collection_schema_add_field(schema, medium_vector_field);
    if (handle_error(error, "adding medium vector field") == ZVEC_OK) {
      printf(
          "✓ Medium vector field 'medium_vector' (128D) with Flat index "
          "added\n");
    }
  }

  // Large dimension vector with HNSW index
  zvec_field_schema_t *large_vector_field = zvec_field_schema_create(
      "large_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 512);
  if (large_vector_field) {
    zvec_field_schema_set_index_params(large_vector_field, hnsw_params);
    error = zvec_collection_schema_add_field(schema, large_vector_field);
    if (handle_error(error, "adding large vector field") == ZVEC_OK) {
      printf(
          "✓ Large vector field 'large_vector' (512D) with HNSW index added\n");
    }
  }

  // 5. Create collection with the schema
  zvec_collection_options_t *options = zvec_collection_options_create();
  if (!options) {
    fprintf(stderr, "Failed to create collection options\n");
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_collection_t *collection = NULL;

  error = zvec_collection_create_and_open("./field_example_collection", schema,
                                          options, &collection);
  zvec_collection_options_destroy(options);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  printf("✓ Collection created successfully\n");

  // 6. Create test documents with various field types
  printf("Creating test documents...\n");

  zvec_doc_t *doc1 = zvec_doc_create();
  zvec_doc_t *doc2 = zvec_doc_create();

  if (!doc1 || !doc2) {
    fprintf(stderr, "Failed to create documents\n");
    goto cleanup;
  }

  // Document 1
  zvec_doc_set_pk(doc1, "user1");
  zvec_doc_add_field_by_value(doc1, "name", ZVEC_DATA_TYPE_STRING,
                              "Alice Johnson", strlen("Alice Johnson"));
  int32_t age1 = 28;
  zvec_doc_add_field_by_value(doc1, "age", ZVEC_DATA_TYPE_INT32, &age1,
                              sizeof(age1));
  float score1 = 87.5f;
  zvec_doc_add_field_by_value(doc1, "score", ZVEC_DATA_TYPE_FLOAT, &score1,
                              sizeof(score1));
  bool active1 = true;
  zvec_doc_add_field_by_value(doc1, "active", ZVEC_DATA_TYPE_BOOL, &active1,
                              sizeof(active1));

  // Add vector data
  float small_vec1[32];
  float medium_vec1[128];
  float large_vec1[512];

  for (int i = 0; i < 32; i++) small_vec1[i] = (float)i / 32.0f;
  for (int i = 0; i < 128; i++) medium_vec1[i] = (float)i / 128.0f;
  for (int i = 0; i < 512; i++) large_vec1[i] = (float)i / 512.0f;

  zvec_doc_add_field_by_value(doc1, "small_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              small_vec1, 32 * sizeof(float));
  zvec_doc_add_field_by_value(doc1, "medium_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              medium_vec1, 128 * sizeof(float));
  zvec_doc_add_field_by_value(doc1, "large_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              large_vec1, 512 * sizeof(float));

  // Document 2
  zvec_doc_set_pk(doc2, "user2");
  zvec_doc_add_field_by_value(doc2, "name", ZVEC_DATA_TYPE_STRING, "Bob Smith",
                              strlen("Bob Smith"));
  int32_t age2 = 35;
  zvec_doc_add_field_by_value(doc2, "age", ZVEC_DATA_TYPE_INT32, &age2,
                              sizeof(age2));
  float score2 = 92.0f;
  zvec_doc_add_field_by_value(doc2, "score", ZVEC_DATA_TYPE_FLOAT, &score2,
                              sizeof(score2));
  bool active2 = false;
  zvec_doc_add_field_by_value(doc2, "active", ZVEC_DATA_TYPE_BOOL, &active2,
                              sizeof(active2));

  // Add vector data
  float small_vec2[32];
  float medium_vec2[128];
  float large_vec2[512];

  for (int i = 0; i < 32; i++) small_vec2[i] = (float)(32 - i) / 32.0f;
  for (int i = 0; i < 128; i++) medium_vec2[i] = (float)(128 - i) / 128.0f;
  for (int i = 0; i < 512; i++) large_vec2[i] = (float)(512 - i) / 512.0f;

  zvec_doc_add_field_by_value(doc2, "small_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              small_vec2, 32 * sizeof(float));
  zvec_doc_add_field_by_value(doc2, "medium_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              medium_vec2, 128 * sizeof(float));
  zvec_doc_add_field_by_value(doc2, "large_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              large_vec2, 512 * sizeof(float));

  // 7. Insert documents
  zvec_doc_t *docs[] = {doc1, doc2};
  size_t success_count = 0, error_count = 0;
  error = zvec_collection_insert(collection, (const zvec_doc_t **)docs, 2,
                                 &success_count, &error_count);
  if (handle_error(error, "inserting documents") == ZVEC_OK) {
    printf("✓ Documents inserted - Success: %zu, Failed: %zu\n", success_count,
           error_count);
  }

  // 8. Flush and test queries
  zvec_collection_flush(collection);
  printf("✓ Collection flushed\n");

  // Test vector query on medium vector field
  zvec_vector_query_t *query = zvec_vector_query_create();
  if (!query) {
    fprintf(stderr, "Failed to create vector query\n");
    goto cleanup;
  }
  zvec_vector_query_set_field_name(query, "medium_vector");
  zvec_vector_query_set_query_vector(query, medium_vec1, 128 * sizeof(float));
  zvec_vector_query_set_topk(query, 2);
  zvec_vector_query_set_filter(query, "");
  zvec_vector_query_set_include_vector(query, false);
  zvec_vector_query_set_include_doc_id(query, true);

  zvec_doc_t **results = NULL;
  size_t result_count = 0;
  error = zvec_collection_query(collection, (const zvec_vector_query_t *)query,
                                &results, &result_count);
  if (error == ZVEC_OK) {
    printf("✓ Vector query successful - Found %zu results\n", result_count);
    zvec_docs_free(results, result_count);
  }
  zvec_vector_query_destroy(query);

  // 9. Cleanup
cleanup:
  if (doc1) zvec_doc_destroy(doc1);
  if (doc2) zvec_doc_destroy(doc2);
  zvec_collection_destroy(collection);
  zvec_collection_schema_destroy(schema);

  printf("✓ Field schema example completed\n");
  return 0;
}