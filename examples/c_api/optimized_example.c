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
#include <time.h>
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
 * @brief Create test vector data
 */
static float *create_test_vector(size_t dimension) {
  float *vector = malloc(dimension * sizeof(float));
  if (!vector) {
    return NULL;
  }

  for (size_t i = 0; i < dimension; i++) {
    vector[i] = (float)rand() / RAND_MAX;
  }

  return vector;
}

/**
 * @brief Optimized C API usage example with performance considerations
 */
int main() {
  printf("=== ZVec Optimized C API Example ===\n\n");

  // Get version information
  const char *version = zvec_get_version();
  printf("ZVec Version: %s\n\n", version ? version : "Unknown");

  ZVecErrorCode error;

  // 1. Create optimized collection schema
  ZVecCollectionSchema *schema =
      zvec_collection_schema_create("optimized_example_collection");
  if (!schema) {
    fprintf(stderr, "Failed to create collection schema\n");
    return -1;
  }
  printf("✓ Collection schema created\n");

  // 2. Create optimized index parameters
  ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED,
      32,   // Higher M for better connectivity
      200,  // Construction ef for quality
      50    // Search ef for performance
  );

  if (!hnsw_params) {
    fprintf(stderr, "Failed to create HNSW parameters\n");
    zvec_collection_schema_destroy(schema);
    return -1;
  }

  // 3. Create fields with optimized configuration
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  ZVecFieldSchema *text_field =
      zvec_field_schema_create("text", ZVEC_DATA_TYPE_STRING, true, 0);
  ZVecFieldSchema *embedding_field = zvec_field_schema_create(
      "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);

  if (!id_field || !text_field || !embedding_field) {
    fprintf(stderr, "Failed to create field schemas\n");
    goto cleanup_params;
  }

  // Set indexes
  zvec_field_schema_set_hnsw_index(embedding_field, hnsw_params);

  // Add fields to schema
  error = zvec_collection_schema_add_field(schema, id_field);
  if (handle_error(error, "adding ID field") != ZVEC_OK) goto cleanup_fields;

  error = zvec_collection_schema_add_field(schema, text_field);
  if (handle_error(error, "adding text field") != ZVEC_OK) goto cleanup_fields;

  error = zvec_collection_schema_add_field(schema, embedding_field);
  if (handle_error(error, "adding embedding field") != ZVEC_OK)
    goto cleanup_fields;

  printf("✓ Fields configured with indexes\n");

  // 4. Create collection with optimized options
  ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();
  options.enable_mmap = true;  // Enable memory mapping for better performance

  ZVecCollection *collection = NULL;
  error = zvec_collection_create_and_open("./optimized_example_collection",
                                          schema, &options, &collection);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    goto cleanup_fields;
  }
  printf("✓ Collection created with optimized settings\n");

  // 5. Bulk insert test data
  const size_t DOC_COUNT = 1000;
  const size_t BATCH_SIZE = 100;

  printf("Inserting %zu documents in batches of %zu...\n", DOC_COUNT,
         BATCH_SIZE);

  clock_t start_time = clock();

  for (size_t batch_start = 0; batch_start < DOC_COUNT;
       batch_start += BATCH_SIZE) {
    size_t current_batch_size = (batch_start + BATCH_SIZE > DOC_COUNT)
                                    ? DOC_COUNT - batch_start
                                    : BATCH_SIZE;

    ZVecDoc **batch_docs = malloc(current_batch_size * sizeof(ZVecDoc *));
    if (!batch_docs) {
      fprintf(stderr, "Failed to allocate batch documents\n");
      break;
    }

    // Create batch documents
    for (size_t i = 0; i < current_batch_size; i++) {
      batch_docs[i] = zvec_doc_create();
      if (!batch_docs[i]) {
        fprintf(stderr, "Failed to create document\n");
        // Cleanup previous documents in batch
        for (size_t j = 0; j < i; j++) {
          zvec_doc_destroy(batch_docs[j]);
        }
        free(batch_docs);
        goto cleanup_collection;
      }

      size_t doc_id = batch_start + i;
      char pk[32];
      snprintf(pk, sizeof(pk), "doc_%zu", doc_id);
      zvec_doc_set_pk(batch_docs[i], pk);

      // Add ID field
      char id_str[32];
      snprintf(id_str, sizeof(id_str), "ID_%zu", doc_id);
      zvec_doc_add_field_by_value(batch_docs[i], "id", ZVEC_DATA_TYPE_STRING,
                                  id_str, strlen(id_str));

      // Add text field
      char text_str[64];
      snprintf(text_str, sizeof(text_str),
               "Document number %zu with sample text", doc_id);
      zvec_doc_add_field_by_value(batch_docs[i], "text", ZVEC_DATA_TYPE_STRING,
                                  text_str, strlen(text_str));

      // Add vector field
      float *vector = create_test_vector(128);
      if (vector) {
        zvec_doc_add_field_by_value(batch_docs[i], "embedding",
                                    ZVEC_DATA_TYPE_VECTOR_FP32, vector,
                                    128 * sizeof(float));
        free(vector);
      }
    }

    // Insert batch
    size_t success_count, error_count;
    error = zvec_collection_insert(collection, (const ZVecDoc **)batch_docs,
                                   current_batch_size, &success_count,
                                   &error_count);
    if (handle_error(error, "inserting batch") != ZVEC_OK) {
      // Cleanup batch documents
      for (size_t i = 0; i < current_batch_size; i++) {
        zvec_doc_destroy(batch_docs[i]);
      }
      free(batch_docs);
      goto cleanup_collection;
    }

    printf("  Batch %zu-%zu: %zu successful, %zu failed\n", batch_start,
           batch_start + current_batch_size - 1, success_count, error_count);

    // Cleanup batch documents
    for (size_t i = 0; i < current_batch_size; i++) {
      zvec_doc_destroy(batch_docs[i]);
    }
    free(batch_docs);
  }

  clock_t insert_end_time = clock();
  double insert_time =
      ((double)(insert_end_time - start_time)) / CLOCKS_PER_SEC;
  printf("✓ Bulk insertion completed in %.3f seconds (%.0f docs/sec)\n",
         insert_time, DOC_COUNT / insert_time);

  // 6. Flush and optimize collection
  printf("Flushing and optimizing collection...\n");
  zvec_collection_flush(collection);
  zvec_collection_optimize(collection);
  printf("✓ Collection optimized\n");

  // 7. Performance query test
  printf("Testing query performance...\n");

  float *query_vector = create_test_vector(128);
  if (!query_vector) {
    fprintf(stderr, "Failed to create query vector\n");
    goto cleanup_collection;
  }

  ZVecVectorQuery query = {0};
  query.field_name =
      (ZVecString){.data = "embedding", .length = strlen("embedding")};
  query.query_vector = (ZVecByteArray){.data = (uint8_t *)query_vector,
                                       .length = 128 * sizeof(float)};
  query.topk = 10;
  query.filter = (ZVecString){.data = "", .length = 0};
  query.include_vector = false;
  query.include_doc_id = true;
  query.output_fields.strings = NULL;
  query.output_fields.count = 0;

  const int QUERY_COUNT = 100;
  start_time = clock();

  for (int q = 0; q < QUERY_COUNT; q++) {
    ZVecDoc **results = NULL;
    size_t result_count = 0;

    error = zvec_collection_query(collection, &query, &results, &result_count);
    if (error != ZVEC_OK) {
      char *error_msg = NULL;
      zvec_get_last_error(&error_msg);
      printf("Query %d failed: %s\n", q,
             error_msg ? error_msg : "Unknown error");
      free(error_msg);
      continue;
    }

    if (results) {
      zvec_docs_free(results, result_count);
    }
  }

  clock_t query_end_time = clock();
  double query_time = ((double)(query_end_time - start_time)) / CLOCKS_PER_SEC;
  double avg_query_time = (query_time * 1000) / QUERY_COUNT;

  printf("✓ Performance test completed\n");
  printf("  Average query time: %.2f ms\n", avg_query_time);
  printf("  Queries per second: %.0f\n", 1000.0 / avg_query_time);

  free(query_vector);

  // 8. Memory usage information
  ZVecCollectionStats *stats = NULL;
  error = zvec_collection_get_stats(collection, &stats);
  if (error == ZVEC_OK && stats) {
    printf("Collection Statistics:\n");
    printf("  Document count: %llu\n", (unsigned long long)stats->doc_count);
    zvec_collection_stats_destroy(stats);
  }

  // 9. Cleanup
cleanup_collection:
  zvec_collection_destroy(collection);

cleanup_fields:
  // Field schemas are managed by the collection schema, no need to destroy
  // individually

cleanup_params:
  zvec_collection_schema_destroy(schema);
  zvec_index_params_hnsw_destroy(hnsw_params);

  printf("✓ Optimized example completed\n");

  return 0;
}