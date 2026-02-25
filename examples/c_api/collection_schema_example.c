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
 * @brief Collection schema creation and management example
 */
int main() {
  printf("=== ZVec Collection Schema Example ===\n\n");

  ZVecErrorCode error;

  // 1. Create collection schema
  ZVecCollectionSchema *schema =
      zvec_collection_schema_create("schema_example_collection");
  if (!schema) {
    fprintf(stderr, "Failed to create collection schema\n");
    return 1;
  }
  printf("✓ Collection schema created successfully\n");

  // 2. Set schema properties
  schema->max_doc_count_per_segment = 1000000;
  printf("✓ Set max documents per segment: %llu\n",
         (unsigned long long)schema->max_doc_count_per_segment);

  // 3. Create index parameters
  ZVecInvertIndexParams *invert_params =
      zvec_index_params_invert_create(true, false);
  ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 200, 50);

  if (!invert_params || !hnsw_params) {
    fprintf(stderr, "Failed to create index parameters\n");
    zvec_collection_schema_destroy(schema);
    return 1;
  }

  // 4. Create and add ID field (primary key)
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  if (!id_field) {
    fprintf(stderr, "Failed to create ID field\n");
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }

  error = zvec_collection_schema_add_field(schema, id_field);
  if (handle_error(error, "adding ID field") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }
  printf("✓ ID field added successfully\n");

  // 5. Create and add text field with inverted index
  ZVecFieldSchema *text_field =
      zvec_field_schema_create("content", ZVEC_DATA_TYPE_STRING, true, 0);
  if (!text_field) {
    fprintf(stderr, "Failed to create text field\n");
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }

  zvec_field_schema_set_invert_index(text_field, invert_params);
  error = zvec_collection_schema_add_field(schema, text_field);
  if (handle_error(error, "adding text field") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }
  printf("✓ Text field with inverted index added successfully\n");

  // 6. Create and add vector field with HNSW index
  ZVecFieldSchema *vector_field = zvec_field_schema_create(
      "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  if (!vector_field) {
    fprintf(stderr, "Failed to create vector field\n");
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }

  zvec_field_schema_set_hnsw_index(vector_field, hnsw_params);
  error = zvec_collection_schema_add_field(schema, vector_field);
  if (handle_error(error, "adding vector field") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }
  printf("✓ Vector field with HNSW index added successfully\n");

  // 7. Check field count
  // Note: This function may not exist in current API, commenting out for now
  // size_t field_count = zvec_collection_schema_get_field_count(schema);
  // printf("✓ Total field count: %zu\n", field_count);

  // 8. Create collection with schema
  ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();
  ZVecCollection *collection = NULL;

  error = zvec_collection_create_and_open("./schema_example_collection", schema,
                                          &options, &collection);
  if (handle_error(error, "creating collection with schema") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }
  printf("✓ Collection created successfully with schema\n");

  // 9. Prepare test data
  float vector1[128];
  float vector2[128];
  for (int i = 0; i < 128; i++) {
    vector1[i] = (float)(i + 1) / 128.0f;
    vector2[i] = (float)(i + 2) / 128.0f;
  }

  // 10. Create documents
  ZVecDoc *docs[2];
  for (int i = 0; i < 2; i++) {
    docs[i] = zvec_doc_create();
    if (!docs[i]) {
      fprintf(stderr, "Failed to create document %d\n", i);
      // Cleanup
      for (int j = 0; j < i; j++) {
        zvec_doc_destroy(docs[j]);
      }
      zvec_collection_destroy(collection);
      zvec_collection_schema_destroy(schema);
      zvec_index_params_invert_destroy(invert_params);
      zvec_index_params_hnsw_destroy(hnsw_params);
      return 1;
    }
  }

  // Add fields to document 1
  zvec_doc_set_pk(docs[0], "doc1");
  zvec_doc_add_field_by_value(docs[0], "id", ZVEC_DATA_TYPE_STRING, "doc1",
                              strlen("doc1"));
  zvec_doc_add_field_by_value(docs[0], "content", ZVEC_DATA_TYPE_STRING,
                              "First test document",
                              strlen("First test document"));
  zvec_doc_add_field_by_value(docs[0], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                              vector1, 128 * sizeof(float));

  // Add fields to document 2
  zvec_doc_set_pk(docs[1], "doc2");
  zvec_doc_add_field_by_value(docs[1], "id", ZVEC_DATA_TYPE_STRING, "doc2",
                              strlen("doc2"));
  zvec_doc_add_field_by_value(docs[1], "content", ZVEC_DATA_TYPE_STRING,
                              "Second test document",
                              strlen("Second test document"));
  zvec_doc_add_field_by_value(docs[1], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                              vector2, 128 * sizeof(float));

  // 11. Insert documents
  size_t success_count = 0, error_count = 0;
  error = zvec_collection_insert(collection, (const ZVecDoc **)docs, 2,
                                 &success_count, &error_count);
  if (handle_error(error, "inserting documents") != ZVEC_OK) {
    // Cleanup
    for (int i = 0; i < 2; i++) {
      zvec_doc_destroy(docs[i]);
    }
    zvec_collection_destroy(collection);
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return 1;
  }
  printf("✓ Documents inserted - Success: %zu, Failed: %zu\n", success_count,
         error_count);

  // Cleanup documents
  for (int i = 0; i < 2; i++) {
    zvec_doc_destroy(docs[i]);
  }

  // 12. Flush collection
  error = zvec_collection_flush(collection);
  if (handle_error(error, "flushing collection") == ZVEC_OK) {
    printf("✓ Collection flushed successfully\n");
  }

  // 13. Query test
  ZVecVectorQuery query = {0};
  query.field_name =
      (ZVecString){.data = "embedding", .length = strlen("embedding")};
  query.query_vector = (ZVecByteArray){.data = (uint8_t *)vector1,
                                       .length = 128 * sizeof(float)};
  query.topk = 5;
  query.filter = (ZVecString){.data = "", .length = 0};
  query.include_vector = true;
  query.include_doc_id = true;
  query.output_fields.strings = NULL;
  query.output_fields.count = 0;

  ZVecDoc **results = NULL;
  size_t result_count = 0;
  error = zvec_collection_query(collection, &query, &results, &result_count);
  if (error == ZVEC_OK) {
    printf("✓ Vector query successful - Returned %zu results\n", result_count);
    zvec_docs_free(results, result_count);
  }

  // 14. Cleanup resources
  zvec_collection_destroy(collection);
  zvec_collection_schema_destroy(schema);
  zvec_index_params_invert_destroy(invert_params);
  zvec_index_params_hnsw_destroy(hnsw_params);
  printf("✓ Schema example completed\n");

  return 0;
}