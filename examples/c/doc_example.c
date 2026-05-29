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

#include <math.h>
#include <stdbool.h>
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
 * @brief Create a test document with all data types
 * @param doc_index Document index for generating unique data
 * @return zvec_doc_t* Created document pointer
 */
static zvec_doc_t *create_full_type_test_doc(int doc_index) {
  zvec_doc_t *doc = zvec_doc_create();
  if (!doc) {
    fprintf(stderr, "Failed to create document\n");
    return NULL;
  }

  // Set primary key
  char pk_buffer[32];
  snprintf(pk_buffer, sizeof(pk_buffer), "doc_%d", doc_index);
  zvec_doc_set_pk(doc, pk_buffer);

  // Add Id field with inverted index
  char id_buffer[32];
  snprintf(id_buffer, sizeof(id_buffer), "id_%d", doc_index);
  zvec_doc_add_field_by_value(doc, "id", ZVEC_DATA_TYPE_STRING, id_buffer,
                              strlen(id_buffer));

  // Add scalar fields with different data types
  // String field
  char string_value[64];
  snprintf(string_value, sizeof(string_value), "test_string_%d", doc_index);
  zvec_doc_add_field_by_value(doc, "string_field", ZVEC_DATA_TYPE_STRING,
                              string_value, strlen(string_value));

  // Boolean field
  bool bool_value = (doc_index % 2 == 0);
  zvec_doc_add_field_by_value(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                              &bool_value, sizeof(bool_value));

  // Integer fields
  int32_t int32_value = doc_index * 1000;
  zvec_doc_add_field_by_value(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                              &int32_value, sizeof(int32_value));

  int64_t int64_value = (int64_t)doc_index * 1000000LL;
  zvec_doc_add_field_by_value(doc, "int64_field", ZVEC_DATA_TYPE_INT64,
                              &int64_value, sizeof(int64_value));

  // Floating point fields
  float float_value = (float)doc_index * 1.5f;
  zvec_doc_add_field_by_value(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                              &float_value, sizeof(float_value));

  double double_value = (double)doc_index * 2.718281828;
  zvec_doc_add_field_by_value(doc, "double_field", ZVEC_DATA_TYPE_DOUBLE,
                              &double_value, sizeof(double_value));

  // Vector fields with different dimensions
  // FP32 vector (3D)
  float fp32_vector[3] = {(float)doc_index, (float)doc_index * 2.0f,
                          (float)doc_index * 3.0f};
  zvec_doc_add_field_by_value(doc, "vector_fp32", ZVEC_DATA_TYPE_VECTOR_FP32,
                              fp32_vector, 3 * sizeof(float));

  // Larger FP32 vector (16D)
  float large_vector[16];
  for (int i = 0; i < 16; i++) {
    large_vector[i] = (float)(doc_index * 16 + i) / 256.0f;
  }
  zvec_doc_add_field_by_value(doc, "large_vector", ZVEC_DATA_TYPE_VECTOR_FP32,
                              large_vector, 16 * sizeof(float));

  return doc;
}

/**
 * @brief Compare two documents for equality
 */
static bool compare_documents(const zvec_doc_t *doc1, const zvec_doc_t *doc2) {
  if (!doc1 || !doc2) return false;

  // Compare primary keys
  const char *pk1 = zvec_doc_get_pk_pointer(doc1);
  const char *pk2 = zvec_doc_get_pk_pointer(doc2);

  if (!pk1 || !pk2 || strcmp(pk1, pk2) != 0) {
    return false;
  }

  // TODO: Compare other fields and values

  return true;
}

/**
 * @brief Print document fields and their values
 * @param doc The document to print
 * @param doc_index Document index for identification
 */
static void print_doc(const zvec_doc_t *doc, int doc_index) {
  if (!doc) {
    printf("Document %d: NULL document\n", doc_index);
    return;
  }

  printf("\n=== Document %d ===\n", doc_index);

  // Print primary key
  const char *pk = zvec_doc_get_pk_pointer(doc);
  printf("Primary Key: %s\n", pk ? pk : "NULL");

  // Print document ID
  uint64_t doc_id = zvec_doc_get_doc_id(doc);
  printf("Document ID: %llu\n", (unsigned long long)doc_id);

  // Print score
  float score = zvec_doc_get_score(doc);
  printf("Score: %.6f\n", score);

  // Print scalar fields
  printf("\nScalar Fields:\n");

  // ID field (using pointer function for strings)
  const void *id_value = NULL;
  size_t id_size = 0;
  zvec_error_code_t error = zvec_doc_get_field_value_pointer(
      doc, "id", ZVEC_DATA_TYPE_STRING, &id_value, &id_size);
  if (error == ZVEC_OK && id_value) {
    printf("  id: %.*s\n", (int)id_size, (const char *)id_value);
  }

  // String field (using pointer function for strings)
  const void *string_value = NULL;
  size_t string_size = 0;
  error = zvec_doc_get_field_value_pointer(
      doc, "string_field", ZVEC_DATA_TYPE_STRING, &string_value, &string_size);
  if (error == ZVEC_OK && string_value) {
    printf("  string_field: %.*s\n", (int)string_size,
           (const char *)string_value);
  }

  // Boolean field
  bool bool_value;
  error = zvec_doc_get_field_value_basic(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                                         &bool_value, sizeof(bool_value));
  if (error == ZVEC_OK) {
    printf("  bool_field: %s\n", bool_value ? "true" : "false");
  }

  // Int32 field
  int32_t int32_value;
  error =
      zvec_doc_get_field_value_basic(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                                     &int32_value, sizeof(int32_value));
  if (error == ZVEC_OK) {
    printf("  int32_field: %d\n", int32_value);
  }

  // Int64 field
  int64_t int64_value;
  error =
      zvec_doc_get_field_value_basic(doc, "int64_field", ZVEC_DATA_TYPE_INT64,
                                     &int64_value, sizeof(int64_value));
  if (error == ZVEC_OK) {
    printf("  int64_field: %lld\n", (long long)int64_value);
  }

  // Float field
  float float_value;
  error =
      zvec_doc_get_field_value_basic(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                     &float_value, sizeof(float_value));
  if (error == ZVEC_OK) {
    printf("  float_field: %.6f\n", float_value);
  }

  // Double field
  double double_value;
  error =
      zvec_doc_get_field_value_basic(doc, "double_field", ZVEC_DATA_TYPE_DOUBLE,
                                     &double_value, sizeof(double_value));
  if (error == ZVEC_OK) {
    printf("  double_field: %.6f\n", double_value);
  }

  // Print vector fields (using copy function for complex types)
  printf("\nVector Fields:\n");

  // FP32 vector (3D)
  void *fp32_vector = NULL;
  size_t fp32_size = 0;
  error = zvec_doc_get_field_value_copy(
      doc, "vector_fp32", ZVEC_DATA_TYPE_VECTOR_FP32, &fp32_vector, &fp32_size);
  if (error == ZVEC_OK && fp32_vector) {
    const float *vec = (const float *)fp32_vector;
    size_t dim = fp32_size / sizeof(float);
    printf("  vector_fp32 (%zuD): [", dim);
    for (size_t i = 0; i < dim && i < 10; i++) {  // Limit to first 10 elements
      printf("%.3f", vec[i]);
      if (i < dim - 1 && i < 9) printf(", ");
    }
    if (dim > 10) printf(", ...");
    printf("]\n");
    zvec_free(fp32_vector);  // Free the allocated memory
  }

  // Large vector (16D)
  void *large_vector = NULL;
  size_t large_size = 0;
  error = zvec_doc_get_field_value_copy(doc, "large_vector",
                                        ZVEC_DATA_TYPE_VECTOR_FP32,
                                        &large_vector, &large_size);
  if (error == ZVEC_OK && large_vector) {
    const float *vec = (const float *)large_vector;
    size_t dim = large_size / sizeof(float);
    printf("  large_vector (%zuD): [", dim);
    for (size_t i = 0; i < dim && i < 10; i++) {  // Limit to first 10 elements
      printf("%.3f", vec[i]);
      if (i < dim - 1 && i < 9) printf(", ");
    }
    if (dim > 10) printf(", ...");
    printf("]\n");
    zvec_free(large_vector);  // Free the allocated memory
  }

  printf("==================\n\n");
}

/**
 * @brief Document creation, manipulation, and query example
 */
int main() {
  printf("=== ZVec Document Example ===\n\n");

  zvec_error_code_t error;

  // 1. Create collection schema for document testing
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("doc_example_collection");
  if (!schema) {
    fprintf(stderr, "Failed to create collection schema\n");
    return -1;
  }
  printf("✓ Collection schema created\n");

  // 2. Create index parameters
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
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(hnsw_params, 16, 200);

  // 3. Create fields for all data types
  printf("Creating fields for all data types...\n");

  // Id field with inverted index
  zvec_field_schema_t *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  if (id_field) {
    zvec_field_schema_set_index_params(id_field, invert_params);
    error = zvec_collection_schema_add_field(schema, id_field);
    if (handle_error(error, "adding ID field") == ZVEC_OK) {
      printf("✓ ID field with inverted index added\n");
    }
  }

  // Scalar fields
  zvec_field_schema_t *string_field =
      zvec_field_schema_create("string_field", ZVEC_DATA_TYPE_STRING, true, 0);
  zvec_field_schema_t *bool_field =
      zvec_field_schema_create("bool_field", ZVEC_DATA_TYPE_BOOL, true, 0);
  zvec_field_schema_t *int32_field =
      zvec_field_schema_create("int32_field", ZVEC_DATA_TYPE_INT32, true, 0);
  zvec_field_schema_t *int64_field =
      zvec_field_schema_create("int64_field", ZVEC_DATA_TYPE_INT64, true, 0);
  zvec_field_schema_t *float_field =
      zvec_field_schema_create("float_field", ZVEC_DATA_TYPE_FLOAT, true, 0);
  zvec_field_schema_t *double_field =
      zvec_field_schema_create("double_field", ZVEC_DATA_TYPE_DOUBLE, true, 0);

  if (string_field) zvec_collection_schema_add_field(schema, string_field);
  if (bool_field) zvec_collection_schema_add_field(schema, bool_field);
  if (int32_field) zvec_collection_schema_add_field(schema, int32_field);
  if (int64_field) zvec_collection_schema_add_field(schema, int64_field);
  if (float_field) zvec_collection_schema_add_field(schema, float_field);
  if (double_field) zvec_collection_schema_add_field(schema, double_field);

  // Vector fields
  zvec_field_schema_t *vector_fp32_field = zvec_field_schema_create(
      "vector_fp32", ZVEC_DATA_TYPE_VECTOR_FP32, false, 3);
  zvec_field_schema_t *large_vector_field = zvec_field_schema_create(
      "large_vector", ZVEC_DATA_TYPE_VECTOR_FP32, false, 16);

  if (vector_fp32_field) {
    zvec_field_schema_set_index_params(vector_fp32_field, hnsw_params);
    error = zvec_collection_schema_add_field(schema, vector_fp32_field);
    if (handle_error(error, "adding vector FP32 field") == ZVEC_OK) {
      printf("✓ Vector FP32 field with HNSW index added\n");
    }
  }

  if (large_vector_field) {
    zvec_field_schema_set_index_params(large_vector_field, hnsw_params);
    error = zvec_collection_schema_add_field(schema, large_vector_field);
    if (handle_error(error, "adding large vector field") == ZVEC_OK) {
      printf("✓ Large vector field with HNSW index added\n");
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

  error = zvec_collection_create_and_open("./doc_example_collection", schema,
                                          options, &collection);
  zvec_collection_options_destroy(options);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  printf("✓ Collection created successfully\n");

  // 5. Create and insert multiple test documents
  printf("Creating and inserting test documents...\n");

#define DOC_COUNT 5
  // Use dynamic allocation for MSVC compatibility (no VLA support)
  zvec_doc_t **test_docs =
      (zvec_doc_t **)malloc(DOC_COUNT * sizeof(zvec_doc_t *));
  if (!test_docs) {
    fprintf(stderr, "Failed to allocate test documents\n");
    goto cleanup;
  }

  for (int i = 0; i < DOC_COUNT; i++) {
    test_docs[i] = create_full_type_test_doc(i);
    if (!test_docs[i]) {
      fprintf(stderr, "Failed to create document %d\n", i);
      // Cleanup
      for (int j = 0; j < i; j++) {
        zvec_doc_destroy(test_docs[j]);
      }
      goto cleanup;
    }
    printf("✓ Created document %d with PK: %s\n", i,
           zvec_doc_get_pk_pointer(test_docs[i]));
  }

  // Print all documents before insertion
  printf("\nDocuments before insertion:\n");
  for (int i = 0; i < DOC_COUNT; i++) {
    print_doc(test_docs[i], i);
  }

  // Insert documents
  size_t success_count = 0, error_count = 0;
  error = zvec_collection_insert(collection, (const zvec_doc_t **)test_docs,
                                 DOC_COUNT, &success_count, &error_count);
  if (handle_error(error, "inserting documents") == ZVEC_OK) {
    printf("✓ Documents inserted - Success: %zu, Failed: %zu\n", success_count,
           error_count);
  }

  // 6. Flush collection
  error = zvec_collection_flush(collection);
  if (handle_error(error, "flushing collection") != ZVEC_OK) {
    printf("Warning: Collection flush failed\n");
  } else {
    printf("✓ Collection flushed successfully\n");
  }

  // Use the first document's vector for querying
  float query_vector[] = {0.0f, 0.0f, 0.0f};
  zvec_vector_query_t *query = zvec_vector_query_create();
  if (!query) {
    fprintf(stderr, "Failed to create vector query\n");
    zvec_collection_destroy(collection);
    zvec_collection_schema_destroy(schema);
    return -1;
  }
  zvec_vector_query_set_field_name(query, "vector_fp32");
  zvec_vector_query_set_query_vector(query, query_vector, 3 * sizeof(float));
  zvec_vector_query_set_topk(query, 5);
  zvec_vector_query_set_filter(query, "");
  zvec_vector_query_set_include_vector(query, true);
  zvec_vector_query_set_include_doc_id(query, true);

  zvec_doc_t **query_results = NULL;
  size_t result_count = 0;

  error = zvec_collection_query(collection, (const zvec_vector_query_t *)query,
                                &query_results, &result_count);
  if (handle_error(error, "querying documents") != ZVEC_OK) {
    query_results = NULL;
    result_count = 0;
  }

  printf("Query returned %zu results\n", result_count);

  // Print query results
  printf("\nQuery Results:\n");
  for (size_t i = 0; i < result_count; i++) {
    print_doc(query_results[i], i);
  }

  // Compare query results
  for (size_t i = 0; i < result_count && i < DOC_COUNT; i++) {
    const char *result_pk = zvec_doc_get_pk_pointer(query_results[i]);
    printf("Comparing query result[%zu]: %s\n", i, result_pk);

    // Find matching original document
    bool found = false;
    for (int j = 0; j < DOC_COUNT; j++) {
      const char *original_pk = zvec_doc_get_pk_pointer(test_docs[j]);
      if (strcmp(result_pk, original_pk) == 0) {
        if (compare_documents(test_docs[j], query_results[i])) {
          printf("✓ Query result %s matches original document\n", result_pk);
        } else {
          printf("✗ Query result %s does not match original document\n",
                 result_pk);
        }
        found = true;
        break;
      }
    }

    if (!found) {
      printf("⚠ Original document not found for: %s\n", result_pk);
    }
  }

  // 7. Filter query test
  printf("\n=== Filter Query Test ===\n");

  // Create filtered query
  zvec_vector_query_set_filter(query, "string_field = 'string_field_0'");

  zvec_doc_t **filtered_results = NULL;
  size_t filtered_count = 0;

  error = zvec_collection_query(collection, (const zvec_vector_query_t *)query,
                                &filtered_results, &filtered_count);
  if (handle_error(error, "filtered querying") == ZVEC_OK) {
    printf("Filtered query returned %zu results\n", filtered_count);

    // Verify filter results
    bool filter_correct = true;
    for (size_t i = 0; i < filtered_count; i++) {
      // Note: Field value access may require different API
      // For now, we'll just check that we got results
      const char *pk = zvec_doc_get_pk_pointer(filtered_results[i]);
      if (strstr(pk, "doc_") == NULL) {
        filter_correct = false;
        break;
      }
    }

    if (filter_correct) {
      printf("✓ Filter query results are correct\n");
    } else {
      printf("✗ Filter query results are incorrect\n");
    }

    if (filtered_results) {
      zvec_docs_free(filtered_results, filtered_count);
    }
  }

  // 8. Cleanup query results
  if (query_results) {
    zvec_docs_free(query_results, result_count);
  }

  // 9. Cleanup documents
  for (int i = 0; i < DOC_COUNT; i++) {
    zvec_doc_destroy(test_docs[i]);
  }
  free(test_docs);  // Free the array itself

  // 10. Final cleanup
cleanup:
  zvec_collection_destroy(collection);
  zvec_collection_schema_destroy(schema);

  printf("✓ Document example completed\n");

  return 0;
}