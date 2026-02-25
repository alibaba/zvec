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
 * @brief Create a simple test collection using CollectionSchema
 */
static ZVecErrorCode create_simple_test_collection(
    ZVecCollection **collection) {
  // Create collection schema using C API
  ZVecCollectionSchema *schema =
      zvec_collection_schema_create("test_collection");
  if (!schema) {
    return ZVEC_ERROR_INTERNAL_ERROR;
  }

  ZVecErrorCode error = ZVEC_OK;

  // Create index parameters
  ZVecInvertIndexParams *invert_params =
      zvec_index_params_invert_create(true, false);
  ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 200, 50);

  // Create and add ID field (primary key)
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_field_schema_set_invert_index(id_field, invert_params);
  error = zvec_collection_schema_add_field(schema, id_field);
  if (error != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return error;
  }

  // Create text field (inverted index)
  ZVecFieldSchema *text_field =
      zvec_field_schema_create("text", ZVEC_DATA_TYPE_STRING, true, 0);
  zvec_field_schema_set_invert_index(text_field, invert_params);
  error = zvec_collection_schema_add_field(schema, text_field);
  if (error != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return error;
  }

  // Create embedding field (HNSW index)
  ZVecFieldSchema *embedding_field = zvec_field_schema_create(
      "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 3);
  zvec_field_schema_set_hnsw_index(embedding_field, hnsw_params);
  error = zvec_collection_schema_add_field(schema, embedding_field);
  if (error != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    zvec_index_params_invert_destroy(invert_params);
    zvec_index_params_hnsw_destroy(hnsw_params);
    return error;
  }

  // Use default options
  ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();

  // Create collection using the new API
  error = zvec_collection_create_and_open("./test_collection", schema, &options,
                                          collection);

  // Cleanup resources
  zvec_collection_schema_destroy(schema);
  zvec_index_params_invert_destroy(invert_params);
  zvec_index_params_hnsw_destroy(hnsw_params);

  return error;
}

/**
 * @brief Basic C API usage example
 */
int main() {
  printf("=== ZVec C API Basic Example ===\n\n");

  ZVecErrorCode error;

  // Create collection using simplified function
  ZVecCollection *collection = NULL;
  error = create_simple_test_collection(&collection);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    return 1;
  }
  printf("✓ Collection created successfully\n");

  // Prepare test data
  float vector1[] = {0.1f, 0.2f, 0.3f};
  float vector2[] = {0.4f, 0.5f, 0.6f};

  ZVecDoc *docs[2];
  for (int i = 0; i < 2; ++i) {
    docs[i] = zvec_doc_create();
    if (!docs[i]) {
      fprintf(stderr, "Failed to create document %d\n", i);
      // Cleanup allocated resources
      for (int j = 0; j < i; ++j) {
        zvec_doc_destroy(docs[j]);
      }
      return ZVEC_ERROR_INTERNAL_ERROR;
    }
  }

  // Manually add fields to document 1
  zvec_doc_set_pk(docs[0], "doc1");
  zvec_doc_add_field_by_value(docs[0], "id", ZVEC_DATA_TYPE_STRING, "doc1",
                              strlen("doc1"));
  zvec_doc_add_field_by_value(docs[0], "text", ZVEC_DATA_TYPE_STRING,
                              "First document", strlen("First document"));
  zvec_doc_add_field_by_value(docs[0], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                              vector1, 3 * sizeof(float));

  // Manually add fields to document 2
  zvec_doc_set_pk(docs[1], "doc2");
  zvec_doc_add_field_by_value(docs[1], "id", ZVEC_DATA_TYPE_STRING, "doc2",
                              strlen("doc2"));
  zvec_doc_add_field_by_value(docs[1], "text", ZVEC_DATA_TYPE_STRING,
                              "Second document", strlen("Second document"));
  zvec_doc_add_field_by_value(docs[1], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                              vector2, 3 * sizeof(float));

  // Insert documents
  size_t success_count = 0;
  size_t error_count = 0;
  error = zvec_collection_insert(collection, (const ZVecDoc **)docs, 2,
                                 &success_count, &error_count);
  if (handle_error(error, "inserting documents") != ZVEC_OK) {
    zvec_collection_destroy(collection);
    return 1;
  }
  printf("✓ Documents inserted - Success: %zu, Failed: %zu\n", success_count,
         error_count);
  for (int i = 0; i < 2; ++i) {
    zvec_doc_destroy(docs[i]);
  }

  // Flush collection
  error = zvec_collection_flush(collection);
  if (handle_error(error, "flushing collection") != ZVEC_OK) {
    printf("Collection flush failed\n");
  } else {
    printf("✓ Collection flushed successfully\n");
  }

  // Get collection statistics
  ZVecCollectionStats *stats = NULL;
  error = zvec_collection_get_stats(collection, &stats);
  if (handle_error(error, "getting collection stats") == ZVEC_OK) {
    printf("✓ Collection stats - Document count: %llu\n",
           (unsigned long long)stats->doc_count);
    // Free statistics memory
    zvec_collection_stats_destroy(stats);
  }

  printf("Testing vector query...\n");
  // Query documents
  ZVecVectorQuery query = {0};
  query.field_name =
      (ZVecString){.data = "embedding", .length = strlen("embedding")};
  query.query_vector =
      (ZVecByteArray){.data = (uint8_t *)vector1, .length = 3 * sizeof(float)};
  query.topk = 10;
  query.filter = (ZVecString){.data = "", .length = 0};
  query.include_vector = true;
  query.include_doc_id = true;
  query.output_fields.strings = NULL;
  query.output_fields.count = 0;

  ZVecDoc **results = NULL;
  size_t result_count = 0;
  error = zvec_collection_query(collection, &query, &results, &result_count);

  if (error != ZVEC_OK) {
    char *error_msg = NULL;
    zvec_get_last_error(&error_msg);
    printf("[ERROR] Query failed: %s\n",
           error_msg ? error_msg : "Unknown error");
    free(error_msg);
    goto cleanup;
  }

  printf("✓ Query successful - Returned %zu results\n", result_count);

  // Process query results
  for (size_t i = 0; i < result_count && i < 5; ++i) {
    const ZVecDoc *doc = results[i];
    const char *pk = zvec_doc_get_pk_copy(doc);

    printf("  Result %zu: PK=%s, DocID=%llu, Score=%.4f\n", i + 1,
           pk ? pk : "NULL", (unsigned long long)zvec_doc_get_doc_id(doc),
           zvec_doc_get_score(doc));

    if (pk) {
      free((void *)pk);
    }
  }

  // Free query results memory
  zvec_docs_free(results, result_count);

cleanup:
  // Cleanup resources
  zvec_collection_destroy(collection);
  printf("✓ Example completed\n");
  return 0;
}