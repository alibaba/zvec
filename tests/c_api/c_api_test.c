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

#include "zvec/c_api.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _POSIX_C_SOURCE
#include <sys/time.h>
#endif
#include <unistd.h>
#include "utils.h"

// =============================================================================
// Test helper macro definitions
// =============================================================================

static int test_count = 0;
static int passed_count = 0;
static int current_test_passed = 1;  // Track if current test function passes

#define TEST_START()                        \
  do {                                      \
    printf("Running test: %s\n", __func__); \
    test_count++;                           \
    current_test_passed = 1;                \
  } while (0)

#define TEST_ASSERT(condition)                   \
  do {                                           \
    if (condition) {                             \
      printf("  ✓ PASS\n");                      \
    } else {                                     \
      printf("  ✗ FAIL at line %d\n", __LINE__); \
      current_test_passed = 0;                   \
    }                                            \
  } while (0)

#define TEST_END()             \
  do {                         \
    if (current_test_passed) { \
      passed_count++;          \
    }                          \
  } while (0)

// =============================================================================
// Helper functions tests
// =============================================================================

void test_version_functions(void) {
  TEST_START();

  // Test version retrieval functions
  const char *version = zvec_get_version();
  TEST_ASSERT(version != NULL);

  // Test version component retrieval
  int major = zvec_get_version_major();
  int minor = zvec_get_version_minor();
  int patch = zvec_get_version_patch();

  TEST_ASSERT(major >= 0);
  TEST_ASSERT(minor >= 0);
  TEST_ASSERT(patch >= 0);

  TEST_ASSERT(zvec_check_version(major, minor, patch));

  // Test version checking functions
  bool compatible = zvec_check_version(0, 3, 0);
  TEST_ASSERT(compatible == true);

  bool not_compatible = zvec_check_version(99, 99, 99);
  TEST_ASSERT(not_compatible == false);

  TEST_END();
}

void test_error_handling_functions(void) {
  TEST_START();

  char *error_msg = NULL;
  ZVecErrorCode err = zvec_get_last_error(&error_msg);
  TEST_ASSERT(err == ZVEC_OK);

  if (error_msg) {
    free(error_msg);
  }

  // Test error clearing
  zvec_clear_error();

  // Test error details retrieval
  ZVecErrorDetails error_details = {0};
  err = zvec_get_last_error_details(&error_details);
  TEST_ASSERT(err == ZVEC_OK);

  TEST_END();
}

void test_zvec_config() {
  TEST_START();

  // Test 1: Console log config creation and destruction
  ZVecConsoleLogConfig *console_config =
      zvec_config_console_log_create(ZVEC_LOG_LEVEL_INFO);
  TEST_ASSERT(console_config != NULL);
  if (console_config) {
    TEST_ASSERT(console_config->level == ZVEC_LOG_LEVEL_INFO);
    zvec_config_console_log_destroy(console_config);
  }

  // Test 2: File log config creation and destruction
  ZVecFileLogConfig *file_config = zvec_config_file_log_create(
      ZVEC_LOG_LEVEL_WARN, "./logs", "test_log", 100, 7);
  TEST_ASSERT(file_config != NULL);
  if (file_config) {
    TEST_ASSERT(file_config->level == ZVEC_LOG_LEVEL_WARN);
    TEST_ASSERT(strcmp(file_config->dir.data, "./logs") == 0);
    TEST_ASSERT(strcmp(file_config->basename.data, "test_log") == 0);
    TEST_ASSERT(file_config->file_size == 100);
    TEST_ASSERT(file_config->overdue_days == 7);
    zvec_config_file_log_destroy(file_config);
  }

  // Test 3: File log config edge cases
  ZVecFileLogConfig *empty_file_config =
      zvec_config_file_log_create(ZVEC_LOG_LEVEL_INFO, "", "", 0, 0);
  TEST_ASSERT(empty_file_config != NULL);
  if (empty_file_config) {
    TEST_ASSERT(empty_file_config->level == ZVEC_LOG_LEVEL_INFO);
    TEST_ASSERT(strcmp(empty_file_config->dir.data, "") == 0);
    TEST_ASSERT(strcmp(empty_file_config->basename.data, "") == 0);
    TEST_ASSERT(empty_file_config->file_size == 0);
    TEST_ASSERT(empty_file_config->overdue_days == 0);
    zvec_config_file_log_destroy(empty_file_config);
  }

  // Test 4: Log config creation with console type
  ZVecConsoleLogConfig *temp_console =
      zvec_config_console_log_create(ZVEC_LOG_LEVEL_ERROR);
  TEST_ASSERT(temp_console != NULL);
  if (temp_console) {
    zvec_config_console_log_destroy(temp_console);
  }

  // Test 5: Log config creation with file type
  ZVecFileLogConfig *temp_file = zvec_config_file_log_create(
      ZVEC_LOG_LEVEL_DEBUG, "./logs", "app", 50, 30);
  TEST_ASSERT(temp_file != NULL);
  TEST_ASSERT(temp_file->level == ZVEC_LOG_LEVEL_DEBUG);
  TEST_ASSERT(strcmp(temp_file->dir.data, "./logs") == 0);
  TEST_ASSERT(strcmp(temp_file->basename.data, "app") == 0);
  TEST_ASSERT(temp_file->file_size == 50);
  TEST_ASSERT(temp_file->overdue_days == 30);

  zvec_config_file_log_destroy(temp_file);

  // Test 6: Config data creation and basic operations
  ZVecConfigData *config_data = zvec_config_data_create();
  TEST_ASSERT(config_data != NULL);
  if (config_data) {
    // Test initial values
    TEST_ASSERT(config_data->log_config != NULL);
    TEST_ASSERT(config_data->log_type == ZVEC_LOG_TYPE_CONSOLE);

    // Test memory limit setting
    ZVecErrorCode err =
        zvec_config_data_set_memory_limit(config_data, 1024 * 1024 * 1024);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(config_data->memory_limit_bytes == 1024 * 1024 * 1024);

    // Test thread count settings
    err = zvec_config_data_set_query_thread_count(config_data, 8);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(config_data->query_thread_count == 8);

    err = zvec_config_data_set_optimize_thread_count(config_data, 4);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(config_data->optimize_thread_count == 4);

    // Test log config replacement
    TEST_ASSERT(config_data->log_type == ZVEC_LOG_TYPE_CONSOLE);
    TEST_ASSERT(config_data->log_config != NULL);

    ZVecFileLogConfig *new_file = zvec_config_file_log_create(
        ZVEC_LOG_LEVEL_DEBUG, "./logs", "app", 50, 30);
    TEST_ASSERT(new_file != NULL);
    zvec_config_data_set_log_config(config_data, ZVEC_LOG_TYPE_FILE, new_file);
    TEST_ASSERT(config_data->log_type == ZVEC_LOG_TYPE_FILE);
    TEST_ASSERT(config_data->log_config != NULL);

    zvec_config_data_destroy(config_data);
  }

  // Test 7: Edge cases and error conditions
  // Test NULL pointer handling
  ZVecErrorCode err = zvec_config_data_set_memory_limit(NULL, 1024);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_log_config(NULL, ZVEC_LOG_TYPE_CONSOLE, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_query_thread_count(NULL, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_optimize_thread_count(NULL, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test boundary values
  ZVecConfigData *boundary_config = zvec_config_data_create();
  if (boundary_config) {
    // Test zero values
    err = zvec_config_data_set_memory_limit(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->memory_limit_bytes == 0);

    // Test maximum values
    err = zvec_config_data_set_memory_limit(boundary_config, UINT64_MAX);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->memory_limit_bytes == UINT64_MAX);

    // Test zero thread counts
    err = zvec_config_data_set_query_thread_count(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->query_thread_count == 0);

    err = zvec_config_data_set_optimize_thread_count(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->optimize_thread_count == 0);

    zvec_config_data_destroy(boundary_config);
  }

  // Test 8: Memory leak prevention - double destroy safety
  ZVecConfigData *double_destroy_test = zvec_config_data_create();
  if (double_destroy_test) {
    zvec_config_data_destroy(double_destroy_test);
  }

  TEST_END();
}

void test_zvec_initialize() {
  TEST_START();

  ZVecConfigData *config = zvec_config_data_create();
  TEST_ASSERT(config != NULL);
  if (config) {
    TEST_ASSERT(config->log_config != NULL);
    TEST_ASSERT(config->log_type == ZVEC_LOG_TYPE_CONSOLE);
  }
  ZVecErrorCode err = zvec_initialize(config);
  TEST_ASSERT(err == ZVEC_OK);
  bool is_initialized = false;
  zvec_is_initialized(&is_initialized);
  TEST_ASSERT(is_initialized);

  TEST_END();
}

// =============================================================================
// Schema-related tests
// =============================================================================

void test_schema_basic_operations(void) {
  TEST_START();

  // Test 1: Basic Schema creation and destruction
  ZVecCollectionSchema *schema = zvec_collection_schema_create("demo");
  TEST_ASSERT(schema != NULL);
  TEST_ASSERT(schema->name != NULL);
  TEST_ASSERT(strcmp(schema->name->data, "demo") == 0);
  TEST_ASSERT(schema->field_count == 0);
  TEST_ASSERT(schema->fields == NULL);
  TEST_ASSERT(schema->max_doc_count_per_segment > 0);

  // Test 2: Schema field count operations
  size_t initial_count = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(initial_count == 0);

  // Test 3: Adding fields to schema
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
  ZVecErrorCode err = zvec_collection_schema_add_field(schema, id_field);
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_add = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(count_after_add == 1);

  // Test 4: Finding fields in schema
  const ZVecFieldSchema *found_field =
      zvec_collection_schema_find_field(schema, "id");
  TEST_ASSERT(found_field != NULL);
  TEST_ASSERT(strcmp(found_field->name->data, "id") == 0);
  TEST_ASSERT(found_field->data_type == ZVEC_DATA_TYPE_INT64);

  // Test 5: Getting field by index
  ZVecFieldSchema *indexed_field = zvec_collection_schema_get_field(schema, 0);
  TEST_ASSERT(indexed_field != NULL);
  TEST_ASSERT(strcmp(indexed_field->name->data, "id") == 0);

  // Test 6: Adding multiple fields
  ZVecFieldSchema fields_to_add[2];
  ZVecFieldSchema *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  ZVecFieldSchema *age_field =
      zvec_field_schema_create("age", ZVEC_DATA_TYPE_INT32, true, 0);

  fields_to_add[0] = *name_field;
  fields_to_add[1] = *age_field;

  err = zvec_collection_schema_add_fields(schema, fields_to_add, 2);
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_multi_add = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(count_after_multi_add == 3);

  // Test 7: Finding newly added fields
  const ZVecFieldSchema *name_found =
      zvec_collection_schema_find_field(schema, "name");
  TEST_ASSERT(name_found != NULL);
  TEST_ASSERT(strcmp(name_found->name->data, "name") == 0);

  const ZVecFieldSchema *age_found =
      zvec_collection_schema_find_field(schema, "age");
  TEST_ASSERT(age_found != NULL);
  TEST_ASSERT(strcmp(age_found->name->data, "age") == 0);

  // Test 8: Setting and getting max doc count
  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 10000);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 10000);

  // Test 9: Schema validation
  ZVecString *validation_error = NULL;
  err = zvec_collection_schema_validate(schema, &validation_error);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(validation_error == NULL);

  // Test 10: Removing single field
  err = zvec_collection_schema_remove_field(schema, "age");
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_remove = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(count_after_remove == 2);

  const ZVecFieldSchema *removed_field =
      zvec_collection_schema_find_field(schema, "age");
  TEST_ASSERT(removed_field == NULL);

  // Test 11: Removing multiple fields
  const char *fields_to_remove[] = {"name", "id"};
  err = zvec_collection_schema_remove_fields(schema, fields_to_remove, 2);
  TEST_ASSERT(err == ZVEC_OK);

  size_t final_count = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(final_count == 0);

  // Test 12: Schema cleanup
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

void test_schema_edge_cases(void) {
  TEST_START();

  // Test 1: NULL parameter handling for schema creation
  ZVecCollectionSchema *null_schema = zvec_collection_schema_create(NULL);
  TEST_ASSERT(null_schema == NULL);

  // Test 2: Empty string schema name
  ZVecCollectionSchema *empty_schema = zvec_collection_schema_create("");
  TEST_ASSERT(empty_schema != NULL);
  TEST_ASSERT(empty_schema->name != NULL);
  TEST_ASSERT(strcmp(empty_schema->name->data, "") == 0);
  zvec_collection_schema_destroy(empty_schema);

  // Test 3: Very long schema name
  char long_name[1024];
  memset(long_name, 'a', 1023);
  long_name[1023] = '\0';
  ZVecCollectionSchema *long_schema = zvec_collection_schema_create(long_name);
  TEST_ASSERT(long_schema != NULL);
  TEST_ASSERT(long_schema->name != NULL);
  TEST_ASSERT(strlen(long_schema->name->data) == 1023);
  zvec_collection_schema_destroy(long_schema);

  // Test 4: NULL schema parameter handling for all functions
  ZVecErrorCode err;
  size_t count = zvec_collection_schema_get_field_count(NULL);
  TEST_ASSERT(count == 0);

  const ZVecFieldSchema *null_field =
      zvec_collection_schema_find_field(NULL, "test");
  TEST_ASSERT(null_field == NULL);

  ZVecFieldSchema *null_indexed_field =
      zvec_collection_schema_get_field(NULL, 0);
  TEST_ASSERT(null_indexed_field == NULL);

  uint64_t null_max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(NULL);
  TEST_ASSERT(null_max_doc_count == 0);

  err = zvec_collection_schema_set_max_doc_count_per_segment(NULL, 1000);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  ZVecString *null_validation_error = NULL;
  err = zvec_collection_schema_validate(NULL, &null_validation_error);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  TEST_ASSERT(null_validation_error == NULL);

  err = zvec_collection_schema_add_field(NULL, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_schema_add_fields(NULL, NULL, 0);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_schema_remove_field(NULL, "test");
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  const char *null_field_names[] = {NULL};
  err = zvec_collection_schema_remove_fields(NULL, null_field_names, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 5: Working with valid schema for edge cases
  ZVecCollectionSchema *schema = zvec_collection_schema_create("edge_test");
  TEST_ASSERT(schema != NULL);

  // Test 6: Adding NULL field to schema
  err = zvec_collection_schema_add_field(schema, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 7: Adding fields with NULL array
  err = zvec_collection_schema_add_fields(schema, NULL, 5);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 8: Adding zero fields
  err = zvec_collection_schema_add_fields(schema, NULL, 0);
  TEST_ASSERT(err == ZVEC_OK);

  // Test 9: Finding field with NULL name
  const ZVecFieldSchema *null_name_field =
      zvec_collection_schema_find_field(schema, NULL);
  TEST_ASSERT(null_name_field == NULL);

  // Test 10: Finding non-existent field
  const ZVecFieldSchema *nonexistent_field =
      zvec_collection_schema_find_field(schema, "nonexistent");
  TEST_ASSERT(nonexistent_field == NULL);

  // Test 11: Getting field with invalid index
  ZVecFieldSchema *invalid_index_field =
      zvec_collection_schema_get_field(schema, 1000);
  TEST_ASSERT(invalid_index_field == NULL);

  // Test 12: Getting field from empty schema with index 0
  ZVecFieldSchema *zero_index_field =
      zvec_collection_schema_get_field(schema, 0);
  TEST_ASSERT(zero_index_field == NULL);

  // Test 13: Removing field with NULL name
  err = zvec_collection_schema_remove_field(schema, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 14: Removing non-existent field
  err = zvec_collection_schema_remove_field(schema, "nonexistent");
  TEST_ASSERT(err == ZVEC_ERROR_NOT_FOUND);

  // Test 15: Removing fields with NULL array
  err = zvec_collection_schema_remove_fields(schema, NULL, 5);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 16: Removing zero fields
  err = zvec_collection_schema_remove_fields(schema, NULL, 0);
  TEST_ASSERT(err == ZVEC_OK);

  // Test 17: Setting extremely large max doc count
  err =
      zvec_collection_schema_set_max_doc_count_per_segment(schema, UINT64_MAX);
  TEST_ASSERT(err == ZVEC_OK);
  uint64_t retrieved_max_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(retrieved_max_count == UINT64_MAX);

  // Test 18: Setting zero max doc count
  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 0);
  TEST_ASSERT(err == ZVEC_OK);
  uint64_t zero_max_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(zero_max_count == 0);

  // Test 19: Schema validation with empty schema
  ZVecString *empty_validation_error = NULL;
  err = zvec_collection_schema_validate(schema, &empty_validation_error);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 20: Add duplicate field names
  ZVecFieldSchema *first_id =
      zvec_field_schema_create("duplicate_id", ZVEC_DATA_TYPE_INT64, false, 0);
  ZVecFieldSchema *second_id =
      zvec_field_schema_create("duplicate_id", ZVEC_DATA_TYPE_STRING, false, 0);

  err = zvec_collection_schema_add_field(schema, first_id);
  TEST_ASSERT(err == ZVEC_OK);

  err = zvec_collection_schema_add_field(schema, second_id);
  TEST_ASSERT(err == ZVEC_ERROR_ALREADY_EXISTS);
  zvec_field_schema_destroy(second_id);

  // Verify fields
  size_t field_count = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(field_count == 1);

  // Test 21: Cleanup
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

void test_schema_field_operations(void) {
  TEST_START();

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Test field count
    size_t initial_count = zvec_collection_schema_get_field_count(schema);
    TEST_ASSERT(initial_count == 5);

    // Test finding non-existent field
    const ZVecFieldSchema *nonexistent =
        zvec_collection_schema_find_field(schema, "nonexistent");
    TEST_ASSERT(nonexistent == NULL);

    // Test finding existing field
    const ZVecFieldSchema *id_field =
        zvec_collection_schema_find_field(schema, "id");
    TEST_ASSERT(id_field != NULL);
    if (id_field) {
      TEST_ASSERT(strcmp(id_field->name->data, "id") == 0);
      TEST_ASSERT(id_field->data_type == ZVEC_DATA_TYPE_INT64);
    }

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_normal_schema_creation(void) {
  TEST_START();

  ZVecCollectionSchema *schema =
      zvec_test_create_normal_schema(false, "test_normal", NULL, NULL, 1000);
  TEST_ASSERT(schema != NULL);

  if (schema) {
    TEST_ASSERT(strcmp(schema->name->data, "test_normal") == 0);

    // Verify field count
    size_t field_count = zvec_collection_schema_get_field_count(schema);
    TEST_ASSERT(field_count > 0);

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_schema_with_indexes(void) {
  TEST_START();

  // Test Schema with scalar index
  ZVecCollectionSchema *scalar_index_schema =
      zvec_test_create_schema_with_scalar_index(true, true,
                                                "scalar_index_test");
  TEST_ASSERT(scalar_index_schema != NULL);
  if (scalar_index_schema) {
    zvec_collection_schema_destroy(scalar_index_schema);
  }

  // Test Schema with vector index
  ZVecCollectionSchema *vector_index_schema =
      zvec_test_create_schema_with_vector_index(false, "vector_index_test",
                                                NULL);
  TEST_ASSERT(vector_index_schema != NULL);
  if (vector_index_schema) {
    zvec_collection_schema_destroy(vector_index_schema);
  }

  TEST_END();
}

void test_schema_max_doc_count(void) {
  TEST_START();

  // Test 1: Setting max doc count to a valid value
  ZVecCollectionSchema *schema = zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  ZVecErrorCode err =
      zvec_collection_schema_set_max_doc_count_per_segment(schema, 1000);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 1000);

  zvec_collection_schema_destroy(schema);

  // Test 2: Setting max doc count to zero
  schema = zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 0);
  TEST_ASSERT(err == ZVEC_OK);

  max_doc_count = zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 0);

  zvec_collection_schema_destroy(schema);

  // Test 3: Setting max doc count to maximum value
  schema = zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  err =
      zvec_collection_schema_set_max_doc_count_per_segment(schema, UINT64_MAX);
  TEST_ASSERT(err == ZVEC_OK);

  max_doc_count = zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == UINT64_MAX);

  zvec_collection_schema_destroy(schema);

  TEST_END();
}

// =============================================================================
// Collection-related tests
// =============================================================================

void test_collection_basic_operations(void) {
  TEST_START();

  // Create temporary directory
  char temp_dir[] = "/tmp/zvec_test_collection_basic_operations";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test collection operations
      ZVecDoc *doc1 = zvec_test_create_doc(1, schema, NULL);
      ZVecDoc *doc2 = zvec_test_create_doc(2, schema, NULL);
      ZVecDoc *doc3 = zvec_test_create_doc(3, schema, NULL);

      TEST_ASSERT(doc1 != NULL);
      TEST_ASSERT(doc2 != NULL);
      TEST_ASSERT(doc3 != NULL);

      if (doc1 && doc2 && doc3) {
        ZVecDoc *docs[] = {doc1, doc2, doc3};
        size_t success_count, error_count;

        // Test insert operation
        err = zvec_collection_insert(collection, (const ZVecDoc **)docs, 3,
                                     &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 3);
        TEST_ASSERT(error_count == 0);

        // Test update operation
        zvec_doc_set_score(doc1, 0.95f);
        ZVecDoc *update_docs[] = {doc1};
        err = zvec_collection_update(collection, (const ZVecDoc **)update_docs,
                                     1, &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 1);
        TEST_ASSERT(error_count == 0);

        // Test upsert operation
        zvec_doc_set_pk(doc3, "pk_3_modified");
        ZVecDoc *upsert_docs[] = {doc3};
        err = zvec_collection_upsert(collection, (const ZVecDoc **)upsert_docs,
                                     1, &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 1);
        TEST_ASSERT(error_count == 0);

        // Test delete operation by primary keys
        const char *pks[] = {"pk_1", "pk_2"};
        err = zvec_collection_delete(collection, pks, 2, &success_count,
                                     &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 2);
        TEST_ASSERT(error_count == 0);

        // Test delete by filter
        err = zvec_collection_delete_by_filter(collection, "id > 0");
        TEST_ASSERT(err == ZVEC_OK);

        // Clean up documents
        zvec_doc_destroy(doc1);
        zvec_doc_destroy(doc2);
        zvec_doc_destroy(doc3);
      }

      // Test collection flush
      err = zvec_collection_flush(collection);
      TEST_ASSERT(err == ZVEC_OK);

      // Test collection optimization
      err = zvec_collection_optimize(collection);
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_edge_cases(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_edge_cases";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;

    // Test empty name collection
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    if (collection) {
      zvec_collection_destroy(collection);
      collection = NULL;
    }

    // Test long name collection
    char long_name[256];
    memset(long_name, 'a', 255);
    long_name[255] = '\0';

    char long_path[512];
    snprintf(long_path, sizeof(long_path), "%s/%s", temp_dir,
             "very_long_collection_name_that_tests_path_limits");

    err = zvec_collection_create_and_open(long_path, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    if (collection) {
      zvec_collection_destroy(collection);
      collection = NULL;
    }

    // Test NULL name集合
    err = zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err != ZVEC_OK);

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_delete_by_filter(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_delete_by_filter";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      // Test normal deletion filtering
      err = zvec_collection_delete_by_filter(collection, "id > 1");
      TEST_ASSERT(err == ZVEC_OK);

      // Test NULL filter
      err = zvec_collection_delete_by_filter(collection, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      // Test empty string filter
      err = zvec_collection_delete_by_filter(collection, "");
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_stats(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_stats";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      ZVecCollectionStats *stats = NULL;
      err = zvec_collection_get_stats(collection, &stats);
      TEST_ASSERT(err == ZVEC_OK);

      if (stats) {
        // Basic validation of statistics
        TEST_ASSERT(stats->doc_count ==
                    0);  // New collection should have no documents
        zvec_collection_stats_destroy(stats);
      }

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

// =============================================================================
// Field-related tests
// =============================================================================

void test_field_schema_functions(void) {
  TEST_START();

  // Test scalar field creation
  ZVecFieldSchema scalar_field = {0};
  ZVecString name1 = {0};
  name1.data = "test_field";
  name1.length = 10;
  scalar_field.name = &name1;
  scalar_field.data_type = ZVEC_DATA_TYPE_STRING;
  scalar_field.nullable = true;
  scalar_field.dimension = 0;

  TEST_ASSERT(strcmp(scalar_field.name->data, "test_field") == 0);
  TEST_ASSERT(scalar_field.data_type == ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(scalar_field.nullable == true);

  // Test vector field creation
  ZVecFieldSchema vector_field = {0};
  ZVecString name2 = {0};
  name2.data = "vec_field";
  name2.length = 9;
  vector_field.name = &name2;
  vector_field.data_type = ZVEC_DATA_TYPE_VECTOR_FP32;
  vector_field.nullable = false;
  vector_field.dimension = 128;

  TEST_ASSERT(strcmp(vector_field.name->data, "vec_field") == 0);
  TEST_ASSERT(vector_field.data_type == ZVEC_DATA_TYPE_VECTOR_FP32);
  TEST_ASSERT(vector_field.dimension == 128);

  // Test sparse vector field creation
  ZVecFieldSchema sparse_field = {0};
  ZVecString name3 = {0};
  name3.data = "sparse_field";
  name3.length = 12;
  sparse_field.name = &name3;
  sparse_field.data_type = ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32;
  sparse_field.nullable = false;
  sparse_field.dimension = 0;

  TEST_ASSERT(strcmp(sparse_field.name->data, "sparse_field") == 0);
  TEST_ASSERT(sparse_field.data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32);

  TEST_END();
}

void test_field_helper_functions(void) {
  TEST_START();

  // Test scalar field helper functions
  ZVecInvertIndexParams *invert_params =
      zvec_test_create_default_invert_params(true);
  ZVecFieldSchema *scalar_field = zvec_test_create_scalar_field(
      "test_scalar", ZVEC_DATA_TYPE_INT32, true, invert_params);
  TEST_ASSERT(scalar_field != NULL);
  if (scalar_field) {
    TEST_ASSERT(strcmp(scalar_field->name->data, "test_scalar") == 0);
    TEST_ASSERT(scalar_field->data_type == ZVEC_DATA_TYPE_INT32);
    free(scalar_field);
  }
  if (invert_params) {
    free(invert_params);
  }

  // Test vector field helper functions
  ZVecHnswIndexParams *hnsw_params = zvec_test_create_default_hnsw_params();
  ZVecFieldSchema *vector_field = zvec_test_create_vector_field(
      "test_vector", ZVEC_DATA_TYPE_VECTOR_FP32, 128, false, hnsw_params);
  TEST_ASSERT(vector_field != NULL);
  if (vector_field) {
    TEST_ASSERT(strcmp(vector_field->name->data, "test_vector") == 0);
    TEST_ASSERT(vector_field->data_type == ZVEC_DATA_TYPE_VECTOR_FP32);
    TEST_ASSERT(vector_field->dimension == 128);
    free(vector_field);
  }
  if (hnsw_params) {
    free(hnsw_params);
  }

  TEST_END();
}

// =============================================================================
// Document-related tests
// =============================================================================

void test_doc_creation(void) {
  TEST_START();

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Test complete document creation
    ZVecDoc *doc = zvec_test_create_doc(1, schema, NULL);
    TEST_ASSERT(doc != NULL);
    if (doc) {
      zvec_doc_destroy(doc);
    }

    // Test null value document creation
    ZVecDoc *null_doc = zvec_test_create_doc_null(2, schema, NULL);
    TEST_ASSERT(null_doc != NULL);
    if (null_doc) {
      zvec_doc_destroy(null_doc);
    }

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_doc_primary_key(void) {
  TEST_START();

  // Test primary key generation
  char *pk = zvec_test_make_pk(12345);
  TEST_ASSERT(pk != NULL);
  if (pk) {
    TEST_ASSERT(strcmp(pk, "pk_12345") == 0);
    free(pk);
  }

  TEST_END();
}

// Test for zvec_doc_add_field_by_value - covers all data types
void test_doc_add_field_by_value(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  if (!doc) {
    TEST_END();
    return;
  }

  // Scalar types
  // BINARY
  const char *binary_data = "binary";
  ZVecErrorCode err =
      zvec_doc_add_field_by_value(doc, "binary_field", ZVEC_DATA_TYPE_BINARY,
                                  binary_data, strlen(binary_data));
  TEST_ASSERT(err == ZVEC_OK);

  // STRING
  const char *string_data = "hello";
  err = zvec_doc_add_field_by_value(doc, "string_field", ZVEC_DATA_TYPE_STRING,
                                    string_data, strlen(string_data));
  TEST_ASSERT(err == ZVEC_OK);

  // BOOL
  bool bool_val = true;
  err = zvec_doc_add_field_by_value(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                                    &bool_val, sizeof(bool_val));
  TEST_ASSERT(err == ZVEC_OK);

  // INT32
  int32_t int32_val = -12345;
  err = zvec_doc_add_field_by_value(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                                    &int32_val, sizeof(int32_val));
  TEST_ASSERT(err == ZVEC_OK);

  // INT64
  int64_t int64_val = -9876543210LL;
  err = zvec_doc_add_field_by_value(doc, "int64_field", ZVEC_DATA_TYPE_INT64,
                                    &int64_val, sizeof(int64_val));
  TEST_ASSERT(err == ZVEC_OK);

  // UINT32
  uint32_t uint32_val = 4294967295U;
  err = zvec_doc_add_field_by_value(doc, "uint32_field", ZVEC_DATA_TYPE_UINT32,
                                    &uint32_val, sizeof(uint32_val));
  TEST_ASSERT(err == ZVEC_OK);

  // UINT64
  uint64_t uint64_val = 18446744073709551615ULL;
  err = zvec_doc_add_field_by_value(doc, "uint64_field", ZVEC_DATA_TYPE_UINT64,
                                    &uint64_val, sizeof(uint64_val));
  TEST_ASSERT(err == ZVEC_OK);

  // FLOAT
  float float_val = 3.14159f;
  err = zvec_doc_add_field_by_value(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                    &float_val, sizeof(float_val));
  TEST_ASSERT(err == ZVEC_OK);

  // DOUBLE
  double double_val = 3.14159265358979;
  err = zvec_doc_add_field_by_value(doc, "double_field", ZVEC_DATA_TYPE_DOUBLE,
                                    &double_val, sizeof(double_val));
  TEST_ASSERT(err == ZVEC_OK);

  // Vector types
  // VECTOR_BINARY32
  uint32_t binary32_vec[] = {0xFFFFFFFF, 0x00000000, 0xAAAAAAAA, 0x55555555};
  err = zvec_doc_add_field_by_value(doc, "binary32_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_BINARY32,
                                    binary32_vec, sizeof(binary32_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_BINARY64
  uint64_t binary64_vec[] = {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL};
  err = zvec_doc_add_field_by_value(doc, "binary64_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_BINARY64,
                                    binary64_vec, sizeof(binary64_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP16
  uint16_t fp16_vec[] = {0x3C00, 0x4000, 0xC000, 0x8000};
  err = zvec_doc_add_field_by_value(doc, "fp16_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_FP16, fp16_vec,
                                    sizeof(fp16_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP32
  float fp32_vec[] = {1.0f, -2.0f, 3.5f, -4.5f};
  err = zvec_doc_add_field_by_value(doc, "fp32_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_FP32, fp32_vec,
                                    sizeof(fp32_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP64
  double fp64_vec[] = {1.1, -2.2, 3.3, -4.4};
  err = zvec_doc_add_field_by_value(doc, "fp64_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_FP64, fp64_vec,
                                    sizeof(fp64_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT4 (packed - each byte contains 2 values)
  int8_t int4_vec[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
  err = zvec_doc_add_field_by_value(doc, "int4_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_INT4, int4_vec,
                                    sizeof(int4_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT8
  int8_t int8_vec[] = {-128, -1, 0, 1, 127};
  err = zvec_doc_add_field_by_value(doc, "int8_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_INT8, int8_vec,
                                    sizeof(int8_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT16
  int16_t int16_vec[] = {-32768, -1, 0, 1, 32767};
  err = zvec_doc_add_field_by_value(doc, "int16_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_INT16, int16_vec,
                                    sizeof(int16_vec));
  TEST_ASSERT(err == ZVEC_OK);

  // Sparse vector types
  // SPARSE_VECTOR_FP16 - format: [nnz(size_t)][indices...][values...]
  size_t sparse_fp16_nnz = 3;
  uint32_t sparse_fp16_indices[] = {0, 5, 10};
  uint16_t sparse_fp16_values[] = {0x3C00, 0x4000, 0xC000};
  size_t sparse_fp16_size = sizeof(sparse_fp16_nnz) +
                            sizeof(sparse_fp16_indices) +
                            sizeof(sparse_fp16_values);
  char *sparse_fp16_buffer = (char *)malloc(sparse_fp16_size);
  memcpy(sparse_fp16_buffer, &sparse_fp16_nnz, sizeof(sparse_fp16_nnz));
  memcpy(sparse_fp16_buffer + sizeof(sparse_fp16_nnz), sparse_fp16_indices,
         sizeof(sparse_fp16_indices));
  memcpy(sparse_fp16_buffer + sizeof(sparse_fp16_nnz) +
             sizeof(sparse_fp16_indices),
         sparse_fp16_values, sizeof(sparse_fp16_values));
  err = zvec_doc_add_field_by_value(doc, "sparse_fp16_field",
                                    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16,
                                    sparse_fp16_buffer, sparse_fp16_size);
  TEST_ASSERT(err == ZVEC_OK);
  free(sparse_fp16_buffer);

  // SPARSE_VECTOR_FP32
  size_t sparse_fp32_nnz = 3;
  uint32_t sparse_fp32_indices[] = {2, 7, 15};
  float sparse_fp32_values[] = {1.5f, -2.5f, 3.5f};
  size_t sparse_fp32_size = sizeof(sparse_fp32_nnz) +
                            sizeof(sparse_fp32_indices) +
                            sizeof(sparse_fp32_values);
  char *sparse_fp32_buffer = (char *)malloc(sparse_fp32_size);
  memcpy(sparse_fp32_buffer, &sparse_fp32_nnz, sizeof(sparse_fp32_nnz));
  memcpy(sparse_fp32_buffer + sizeof(sparse_fp32_nnz), sparse_fp32_indices,
         sizeof(sparse_fp32_indices));
  memcpy(sparse_fp32_buffer + sizeof(sparse_fp32_nnz) +
             sizeof(sparse_fp32_indices),
         sparse_fp32_values, sizeof(sparse_fp32_values));
  err = zvec_doc_add_field_by_value(doc, "sparse_fp32_field",
                                    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32,
                                    sparse_fp32_buffer, sparse_fp32_size);
  TEST_ASSERT(err == ZVEC_OK);
  free(sparse_fp32_buffer);

  // Array types
  // ARRAY_BINARY - format: [length(uint32_t)][data][length][data]...
  uint8_t array_bin_data[] = {
      1, 0, 0, 0, 0x01,        // length=1, data=0x01
      2, 0, 0, 0, 0x02, 0x03,  // length=2, data=0x02,0x03
      2, 0, 0, 0, 0x04, 0x05   // length=2, data=0x04,0x05
  };
  err = zvec_doc_add_field_by_value(doc, "array_binary_field",
                                    ZVEC_DATA_TYPE_ARRAY_BINARY, array_bin_data,
                                    sizeof(array_bin_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_STRING - null-terminated strings
  const char *array_str_data[] = {"str1", "str2", "str3"};
  ZVecString *array_zvec_str[3];
  for (int i = 0; i < 3; i++) {
    array_zvec_str[i] = zvec_string_create(array_str_data[i]);
  }
  err = zvec_doc_add_field_by_value(doc, "array_string_field",
                                    ZVEC_DATA_TYPE_ARRAY_STRING, array_zvec_str,
                                    sizeof(array_zvec_str));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_BOOL
  bool array_bool_data[] = {true, false, true, false};
  err = zvec_doc_add_field_by_value(doc, "array_bool_field",
                                    ZVEC_DATA_TYPE_ARRAY_BOOL, array_bool_data,
                                    sizeof(array_bool_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_INT32
  int32_t array_int32_data[] = {-100, -50, 0, 50, 100};
  err = zvec_doc_add_field_by_value(doc, "array_int32_field",
                                    ZVEC_DATA_TYPE_ARRAY_INT32,
                                    array_int32_data, sizeof(array_int32_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_INT64
  int64_t array_int64_data[] = {-1000000, -500000, 0, 500000, 1000000};
  err = zvec_doc_add_field_by_value(doc, "array_int64_field",
                                    ZVEC_DATA_TYPE_ARRAY_INT64,
                                    array_int64_data, sizeof(array_int64_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_UINT32
  uint32_t array_uint32_data[] = {0, 100, 1000, 10000, 4294967295U};
  err = zvec_doc_add_field_by_value(
      doc, "array_uint32_field", ZVEC_DATA_TYPE_ARRAY_UINT32, array_uint32_data,
      sizeof(array_uint32_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_UINT64
  uint64_t array_uint64_data[] = {0, 100, 1000, 10000, 18446744073709551615ULL};
  err = zvec_doc_add_field_by_value(
      doc, "array_uint64_field", ZVEC_DATA_TYPE_ARRAY_UINT64, array_uint64_data,
      sizeof(array_uint64_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_FLOAT
  float array_float_data[] = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f};
  err = zvec_doc_add_field_by_value(doc, "array_float_field",
                                    ZVEC_DATA_TYPE_ARRAY_FLOAT,
                                    array_float_data, sizeof(array_float_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_DOUBLE
  double array_double_data[] = {-1.1, -0.1, 0.0, 0.1, 1.1};
  err = zvec_doc_add_field_by_value(
      doc, "array_double_field", ZVEC_DATA_TYPE_ARRAY_DOUBLE, array_double_data,
      sizeof(array_double_data));
  TEST_ASSERT(err == ZVEC_OK);

  // Verify we can retrieve some of the values
  void *result = NULL;
  size_t result_size = 0;
  err = zvec_doc_get_field_value_copy(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                                      &result, &result_size);
  TEST_ASSERT(err == ZVEC_OK && result_size == sizeof(int32_t));
  if (result) {
    TEST_ASSERT(*(int32_t *)result == -12345);
    free(result);
  }

  err = zvec_doc_get_field_value_copy(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                      &result, &result_size);
  TEST_ASSERT(err == ZVEC_OK && result_size == sizeof(float));
  if (result) {
    TEST_ASSERT(fabs(*(float *)result - 3.14159f) < 0.0001f);
    free(result);
  }

  zvec_doc_destroy(doc);
  TEST_END();
}

// Test for zvec_doc_add_field_by_struct - covers all data types
void test_doc_add_field_by_struct(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  if (!doc) {
    TEST_END();
    return;
  }

  ZVecErrorCode err;
  ZVecDocField field;

  // Scalar types
  // BINARY
  memset(&field, 0, sizeof(field));
  field.name.data = "binary_field";
  field.name.length = strlen("binary_field");
  field.data_type = ZVEC_DATA_TYPE_BINARY;
  uint8_t binary_data[] = {0x01, 0x02, 0x03, 0x04};
  field.value.binary_value.data = binary_data;
  field.value.binary_value.length = sizeof(binary_data);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // STRING
  memset(&field, 0, sizeof(field));
  field.name.data = "string_field";
  field.name.length = strlen("string_field");
  field.data_type = ZVEC_DATA_TYPE_STRING;
  const char *string_data = "hello world";
  field.value.string_value.data = (char *)string_data;
  field.value.string_value.length = strlen(string_data);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // BOOL
  memset(&field, 0, sizeof(field));
  field.name.data = "bool_field";
  field.name.length = strlen("bool_field");
  field.data_type = ZVEC_DATA_TYPE_BOOL;
  field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // INT32
  memset(&field, 0, sizeof(field));
  field.name.data = "int32_field";
  field.name.length = strlen("int32_field");
  field.data_type = ZVEC_DATA_TYPE_INT32;
  field.value.int32_value = -12345;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // INT64
  memset(&field, 0, sizeof(field));
  field.name.data = "int64_field";
  field.name.length = strlen("int64_field");
  field.data_type = ZVEC_DATA_TYPE_INT64;
  field.value.int64_value = -9876543210LL;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // UINT32
  memset(&field, 0, sizeof(field));
  field.name.data = "uint32_field";
  field.name.length = strlen("uint32_field");
  field.data_type = ZVEC_DATA_TYPE_UINT32;
  field.value.uint32_value = 4294967295U;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // UINT64
  memset(&field, 0, sizeof(field));
  field.name.data = "uint64_field";
  field.name.length = strlen("uint64_field");
  field.data_type = ZVEC_DATA_TYPE_UINT64;
  field.value.uint64_value = 18446744073709551615ULL;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // FLOAT
  memset(&field, 0, sizeof(field));
  field.name.data = "float_field";
  field.name.length = strlen("float_field");
  field.data_type = ZVEC_DATA_TYPE_FLOAT;
  field.value.float_value = 3.14159f;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // DOUBLE
  memset(&field, 0, sizeof(field));
  field.name.data = "double_field";
  field.name.length = strlen("double_field");
  field.data_type = ZVEC_DATA_TYPE_DOUBLE;
  field.value.double_value = 3.14159265358979;
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_BINARY32
  memset(&field, 0, sizeof(field));
  field.name.data = "binary32_vec_field";
  field.name.length = strlen("binary32_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_BINARY32;
  uint32_t binary32_vec[] = {0xFFFFFFFF, 0x00000000, 0xAAAAAAAA, 0x55555555};
  field.value.vector_value.data = (const float *)binary32_vec;
  field.value.vector_value.length = sizeof(binary32_vec) / sizeof(uint32_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_BINARY64
  memset(&field, 0, sizeof(field));
  field.name.data = "binary64_vec_field";
  field.name.length = strlen("binary64_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_BINARY64;
  uint64_t binary64_vec[] = {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL};
  field.value.vector_value.data = (const float *)binary64_vec;
  field.value.vector_value.length = sizeof(binary64_vec) / sizeof(uint64_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP16
  memset(&field, 0, sizeof(field));
  field.name.data = "fp16_vec_field";
  field.name.length = strlen("fp16_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_FP16;
  uint16_t fp16_vec[] = {0x3C00, 0x4000, 0xC000, 0x8000};
  field.value.vector_value.data = (const float *)fp16_vec;
  field.value.vector_value.length = sizeof(fp16_vec) / sizeof(uint16_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP32
  memset(&field, 0, sizeof(field));
  field.name.data = "fp32_vec_field";
  field.name.length = strlen("fp32_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_FP32;
  float fp32_vec[] = {1.0f, -2.0f, 3.5f, -4.5f};
  field.value.vector_value.data = fp32_vec;
  field.value.vector_value.length = sizeof(fp32_vec) / sizeof(float);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP64
  memset(&field, 0, sizeof(field));
  field.name.data = "fp64_vec_field";
  field.name.length = strlen("fp64_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_FP64;
  double fp64_vec[] = {1.1, -2.2, 3.3, -4.4};
  field.value.vector_value.data = (const float *)fp64_vec;
  field.value.vector_value.length = sizeof(fp64_vec) / sizeof(double);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT4
  memset(&field, 0, sizeof(field));
  field.name.data = "int4_vec_field";
  field.name.length = strlen("int4_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_INT4;
  int8_t int4_vec[] = {0x12, 0x34, 0x56, 0x78};
  field.value.vector_value.data = (const float *)int4_vec;
  field.value.vector_value.length =
      sizeof(int4_vec) * 2;  // Each byte contains 2 values
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT8
  memset(&field, 0, sizeof(field));
  field.name.data = "int8_vec_field";
  field.name.length = strlen("int8_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_INT8;
  int8_t int8_vec[] = {-128, -1, 0, 1, 127};
  field.value.vector_value.data = (const float *)int8_vec;
  field.value.vector_value.length = sizeof(int8_vec) / sizeof(int8_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT16
  memset(&field, 0, sizeof(field));
  field.name.data = "int16_vec_field";
  field.name.length = strlen("int16_vec_field");
  field.data_type = ZVEC_DATA_TYPE_VECTOR_INT16;
  int16_t int16_vec[] = {-32768, -1, 0, 1, 32767};
  field.value.vector_value.data = (const float *)int16_vec;
  field.value.vector_value.length = sizeof(int16_vec) / sizeof(int16_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // Sparse vector types
  // SPARSE_VECTOR_FP16
  memset(&field, 0, sizeof(field));
  field.name.data = "sparse_fp16_field";
  field.name.length = strlen("sparse_fp16_field");
  field.data_type = ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16;
  uint16_t sparse_fp16_values[] = {0x3C00, 0x4000, 0xC000};
  field.value.vector_value.data = (const float *)sparse_fp16_values;
  field.value.vector_value.length =
      sizeof(sparse_fp16_values) / sizeof(uint16_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // SPARSE_VECTOR_FP32
  memset(&field, 0, sizeof(field));
  field.name.data = "sparse_fp32_field";
  field.name.length = strlen("sparse_fp32_field");
  field.data_type = ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32;
  float sparse_fp32_values[] = {1.5f, -2.5f, 3.5f};
  field.value.vector_value.data = sparse_fp32_values;
  field.value.vector_value.length = sizeof(sparse_fp32_values) / sizeof(float);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // Array types
  // ARRAY_BINARY
  memset(&field, 0, sizeof(field));
  field.name.data = "array_binary_field";
  field.name.length = strlen("array_binary_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_BINARY;
  uint8_t array_bin_data[] = {
      1, 0, 0, 0, 0x01,        // length=1, data=0x01
      2, 0, 0, 0, 0x02, 0x03,  // length=2, data=0x02,0x03
      2, 0, 0, 0, 0x04, 0x05   // length=2, data=0x04,0x05
  };
  field.value.binary_value.data = array_bin_data;
  field.value.binary_value.length = sizeof(array_bin_data);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_STRING
  memset(&field, 0, sizeof(field));
  field.name.data = "array_string_field";
  field.name.length = strlen("array_string_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_STRING;
  const char array_string_data[] = "str1\0str2\0str3\0";
  field.value.string_value.data = (char *)array_string_data;
  field.value.string_value.length = sizeof(array_string_data);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_BOOL
  memset(&field, 0, sizeof(field));
  field.name.data = "array_bool_field";
  field.name.length = strlen("array_bool_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_BOOL;
  bool array_bool_data[] = {true, false, true, false};
  field.value.binary_value.data = (const uint8_t *)array_bool_data;
  field.value.binary_value.length = sizeof(array_bool_data);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_INT32
  memset(&field, 0, sizeof(field));
  field.name.data = "array_int32_field";
  field.name.length = strlen("array_int32_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_INT32;
  int32_t array_int32_data[] = {-100, -50, 0, 50, 100};
  field.value.vector_value.data = (const float *)array_int32_data;
  field.value.vector_value.length = sizeof(array_int32_data) / sizeof(int32_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_INT64
  memset(&field, 0, sizeof(field));
  field.name.data = "array_int64_field";
  field.name.length = strlen("array_int64_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_INT64;
  int64_t array_int64_data[] = {-1000000, -500000, 0, 500000, 1000000};
  field.value.vector_value.data = (const float *)array_int64_data;
  field.value.vector_value.length = sizeof(array_int64_data) / sizeof(int64_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_UINT32
  memset(&field, 0, sizeof(field));
  field.name.data = "array_uint32_field";
  field.name.length = strlen("array_uint32_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_UINT32;
  uint32_t array_uint32_data[] = {0, 100, 1000, 10000, 4294967295U};
  field.value.vector_value.data = (const float *)array_uint32_data;
  field.value.vector_value.length =
      sizeof(array_uint32_data) / sizeof(uint32_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_UINT64
  memset(&field, 0, sizeof(field));
  field.name.data = "array_uint64_field";
  field.name.length = strlen("array_uint64_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_UINT64;
  uint64_t array_uint64_data[] = {0, 100, 1000, 10000, 18446744073709551615ULL};
  field.value.vector_value.data = (const float *)array_uint64_data;
  field.value.vector_value.length =
      sizeof(array_uint64_data) / sizeof(uint64_t);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_FLOAT
  memset(&field, 0, sizeof(field));
  field.name.data = "array_float_field";
  field.name.length = strlen("array_float_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_FLOAT;
  float array_float_data[] = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f};
  field.value.vector_value.data = array_float_data;
  field.value.vector_value.length = sizeof(array_float_data) / sizeof(float);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_DOUBLE
  memset(&field, 0, sizeof(field));
  field.name.data = "array_double_field";
  field.name.length = strlen("array_double_field");
  field.data_type = ZVEC_DATA_TYPE_ARRAY_DOUBLE;
  double array_double_data[] = {-1.1, -0.1, 0.0, 0.1, 1.1};
  field.value.vector_value.data = (const float *)array_double_data;
  field.value.vector_value.length = sizeof(array_double_data) / sizeof(double);
  err = zvec_doc_add_field_by_struct(doc, &field);
  TEST_ASSERT(err == ZVEC_OK);

  // Verify we can retrieve some of the values
  void *result = NULL;
  size_t result_size = 0;

  err = zvec_doc_get_field_value_copy(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                                      &result, &result_size);
  TEST_ASSERT(err == ZVEC_OK && result_size == sizeof(int32_t));
  if (result) {
    TEST_ASSERT(*(int32_t *)result == -12345);
    free(result);
  }

  err = zvec_doc_get_field_value_copy(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                      &result, &result_size);
  TEST_ASSERT(err == ZVEC_OK && result_size == sizeof(float));
  if (result) {
    TEST_ASSERT(fabs(*(float *)result - 3.14159f) < 0.0001f);
    free(result);
  }

  zvec_doc_destroy(doc);
  TEST_END();
}

void test_doc_basic_operations(void);
void test_doc_get_field_value_basic(void);
void test_doc_get_field_value_copy(void);
void test_doc_get_field_value_pointer(void);
void test_doc_field_operations(void);
void test_doc_error_conditions(void);
void test_doc_serialization(void);
void test_doc_add_field_by_value(void);
void test_doc_add_field_by_struct(void);

void test_doc_functions(void) {
  test_doc_basic_operations();
  test_doc_get_field_value_basic();
  test_doc_get_field_value_copy();
  test_doc_get_field_value_pointer();
  test_doc_field_operations();
  test_doc_error_conditions();
  test_doc_serialization();
}

void test_doc_basic_operations(void) {
  TEST_START();

  // Create test document
  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  // Test primary key operations
  zvec_doc_set_pk(doc, "test_doc_complete");
  const char *pk = zvec_doc_get_pk_pointer(doc);
  TEST_ASSERT(pk != NULL);
  TEST_ASSERT(strcmp(pk, "test_doc_complete") == 0);

  // Test document ID and score operations
  zvec_doc_set_doc_id(doc, 99999);
  uint64_t doc_id = zvec_doc_get_doc_id(doc);
  TEST_ASSERT(doc_id == 99999);

  zvec_doc_set_score(doc, 0.95f);
  float score = zvec_doc_get_score(doc);
  TEST_ASSERT(score == 0.95f);

  // Test operator operations
  zvec_doc_set_operator(doc, ZVEC_DOC_OP_INSERT);
  ZVecDocOperator op = zvec_doc_get_operator(doc);
  TEST_ASSERT(op == ZVEC_DOC_OP_INSERT);

  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_get_field_value_basic(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  ZVecErrorCode err;

  printf(
      "=== Testing zvec_doc_get_field_value_basic with all supported types "
      "===\n");

  // BOOL type
  ZVecDocField bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &bool_field);
  TEST_ASSERT(err == ZVEC_OK);

  bool bool_result;
  err = zvec_doc_get_field_value_basic(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                                       &bool_result, sizeof(bool_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bool_result == true);

  // INT32 type
  ZVecDocField int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;  // Min int32
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  int32_t int32_result;
  err = zvec_doc_get_field_value_basic(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                                       &int32_result, sizeof(int32_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int32_result == -2147483648);

  // INT64 type
  ZVecDocField int64_field;
  int64_field.name.data = "int64_field";
  int64_field.name.length = strlen("int64_field");
  int64_field.data_type = ZVEC_DATA_TYPE_INT64;
  int64_field.value.int64_value = 9223372036854775807LL;  // Max int64
  err = zvec_doc_add_field_by_struct(doc, &int64_field);
  TEST_ASSERT(err == ZVEC_OK);

  int64_t int64_result;
  err = zvec_doc_get_field_value_basic(doc, "int64_field", ZVEC_DATA_TYPE_INT64,
                                       &int64_result, sizeof(int64_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int64_result == 9223372036854775807LL);

  // UINT32 type
  ZVecDocField uint32_field;
  uint32_field.name.data = "uint32_field";
  uint32_field.name.length = strlen("uint32_field");
  uint32_field.data_type = ZVEC_DATA_TYPE_UINT32;
  uint32_field.value.uint32_value = 4294967295U;  // Max uint32
  err = zvec_doc_add_field_by_struct(doc, &uint32_field);
  TEST_ASSERT(err == ZVEC_OK);

  uint32_t uint32_result;
  err =
      zvec_doc_get_field_value_basic(doc, "uint32_field", ZVEC_DATA_TYPE_UINT32,
                                     &uint32_result, sizeof(uint32_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint32_result == 4294967295U);

  // UINT64 type
  ZVecDocField uint64_field;
  uint64_field.name.data = "uint64_field";
  uint64_field.name.length = strlen("uint64_field");
  uint64_field.data_type = ZVEC_DATA_TYPE_UINT64;
  uint64_field.value.uint64_value = 18446744073709551615ULL;  // Max uint64
  err = zvec_doc_add_field_by_struct(doc, &uint64_field);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t uint64_result;
  err =
      zvec_doc_get_field_value_basic(doc, "uint64_field", ZVEC_DATA_TYPE_UINT64,
                                     &uint64_result, sizeof(uint64_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint64_result == 18446744073709551615ULL);

  // FLOAT type
  ZVecDocField float_field;
  float_field.name.data = "float_field";
  float_field.name.length = strlen("float_field");
  float_field.data_type = ZVEC_DATA_TYPE_FLOAT;
  float_field.value.float_value = 3.14159265359f;
  err = zvec_doc_add_field_by_struct(doc, &float_field);
  TEST_ASSERT(err == ZVEC_OK);

  float float_result;
  err = zvec_doc_get_field_value_basic(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                       &float_result, sizeof(float_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fabsf(float_result - 3.14159265359f) < 1e-6f);

  // DOUBLE type
  ZVecDocField double_field;
  double_field.name.data = "double_field";
  double_field.name.length = strlen("double_field");
  double_field.data_type = ZVEC_DATA_TYPE_DOUBLE;
  double_field.value.double_value = 2.71828182845904523536;
  err = zvec_doc_add_field_by_struct(doc, &double_field);
  TEST_ASSERT(err == ZVEC_OK);

  double double_result;
  err =
      zvec_doc_get_field_value_basic(doc, "double_field", ZVEC_DATA_TYPE_DOUBLE,
                                     &double_result, sizeof(double_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fabs(double_result - 2.71828182845904523536) < 1e-15);

  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_get_field_value_copy(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  ZVecErrorCode err;

  printf(
      "=== Testing zvec_doc_get_field_value_copy with all supported types "
      "===\n");

  // Basic scalar types first
  bool bool_val = true;
  err = zvec_doc_add_field_by_value(doc, "bool_field2", ZVEC_DATA_TYPE_BOOL,
                                    &bool_val, sizeof(bool_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *bool_copy_result;
  size_t bool_copy_size;
  err = zvec_doc_get_field_value_copy(doc, "bool_field2", ZVEC_DATA_TYPE_BOOL,
                                      &bool_copy_result, &bool_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bool_copy_result != NULL);
  TEST_ASSERT(bool_copy_size == sizeof(bool));
  TEST_ASSERT(*(bool *)bool_copy_result == true);
  free(bool_copy_result);

  int32_t int32_val = -12345;
  err = zvec_doc_add_field_by_value(doc, "int32_field2", ZVEC_DATA_TYPE_INT32,
                                    &int32_val, sizeof(int32_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *int32_copy_result;
  size_t int32_copy_size;
  err = zvec_doc_get_field_value_copy(doc, "int32_field2", ZVEC_DATA_TYPE_INT32,
                                      &int32_copy_result, &int32_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int32_copy_result != NULL);
  TEST_ASSERT(int32_copy_size == sizeof(int32_t));
  TEST_ASSERT(*(int32_t *)int32_copy_result == -12345);
  free(int32_copy_result);

  int64_t int64_val = -9223372036854775807LL;
  err = zvec_doc_add_field_by_value(doc, "int64_field2", ZVEC_DATA_TYPE_INT64,
                                    &int64_val, sizeof(int64_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *int64_copy_result;
  size_t int64_copy_size;
  err = zvec_doc_get_field_value_copy(doc, "int64_field2", ZVEC_DATA_TYPE_INT64,
                                      &int64_copy_result, &int64_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int64_copy_result != NULL);
  TEST_ASSERT(int64_copy_size == sizeof(int64_t));
  TEST_ASSERT(*(int64_t *)int64_copy_result == -9223372036854775807LL);
  free(int64_copy_result);

  uint32_t uint32_val = 4000000000U;
  err = zvec_doc_add_field_by_value(doc, "uint32_field2", ZVEC_DATA_TYPE_UINT32,
                                    &uint32_val, sizeof(uint32_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *uint32_copy_result;
  size_t uint32_copy_size;
  err =
      zvec_doc_get_field_value_copy(doc, "uint32_field2", ZVEC_DATA_TYPE_UINT32,
                                    &uint32_copy_result, &uint32_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint32_copy_result != NULL);
  TEST_ASSERT(uint32_copy_size == sizeof(uint32_t));
  TEST_ASSERT(*(uint32_t *)uint32_copy_result == 4000000000U);
  free(uint32_copy_result);

  uint64_t uint64_val = 18000000000000000000ULL;
  err = zvec_doc_add_field_by_value(doc, "uint64_field2", ZVEC_DATA_TYPE_UINT64,
                                    &uint64_val, sizeof(uint64_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *uint64_copy_result;
  size_t uint64_copy_size;
  err =
      zvec_doc_get_field_value_copy(doc, "uint64_field2", ZVEC_DATA_TYPE_UINT64,
                                    &uint64_copy_result, &uint64_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint64_copy_result != NULL);
  TEST_ASSERT(uint64_copy_size == sizeof(uint64_t));
  TEST_ASSERT(*(uint64_t *)uint64_copy_result == 18000000000000000000ULL);
  free(uint64_copy_result);

  float float_val = 3.14159265f;
  err = zvec_doc_add_field_by_value(doc, "float_field2", ZVEC_DATA_TYPE_FLOAT,
                                    &float_val, sizeof(float_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *float_copy_result;
  size_t float_copy_size;
  err = zvec_doc_get_field_value_copy(doc, "float_field2", ZVEC_DATA_TYPE_FLOAT,
                                      &float_copy_result, &float_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(float_copy_result != NULL);
  TEST_ASSERT(float_copy_size == sizeof(float));
  TEST_ASSERT(fabs(*(float *)float_copy_result - 3.14159265f) < 1e-6f);
  free(float_copy_result);

  double double_val = 2.718281828459045;
  err = zvec_doc_add_field_by_value(doc, "double_field2", ZVEC_DATA_TYPE_DOUBLE,
                                    &double_val, sizeof(double_val));
  TEST_ASSERT(err == ZVEC_OK);

  void *double_copy_result;
  size_t double_copy_size;
  err =
      zvec_doc_get_field_value_copy(doc, "double_field2", ZVEC_DATA_TYPE_DOUBLE,
                                    &double_copy_result, &double_copy_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(double_copy_result != NULL);
  TEST_ASSERT(double_copy_size == sizeof(double));
  TEST_ASSERT(fabs(*(double *)double_copy_result - 2.718281828459045) < 1e-15);
  free(double_copy_result);

  // String and binary types
  ZVecDocField string_field;
  string_field.name.data = "string_field";
  string_field.name.length = strlen("string_field");
  string_field.data_type = ZVEC_DATA_TYPE_STRING;
  string_field.value.string_value = *zvec_string_create("Hello, 世界!");
  err = zvec_doc_add_field_by_struct(doc, &string_field);
  TEST_ASSERT(err == ZVEC_OK);

  void *string_result;
  size_t string_size;
  err = zvec_doc_get_field_value_copy(
      doc, "string_field", ZVEC_DATA_TYPE_STRING, &string_result, &string_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(string_result != NULL);
  TEST_ASSERT(string_size == strlen("Hello, 世界!"));
  TEST_ASSERT(memcmp(string_result, "Hello, 世界!", string_size) == 0);
  free(string_result);

  ZVecDocField binary_field;
  binary_field.name.data = "binary_field";
  binary_field.name.length = strlen("binary_field");
  binary_field.data_type = ZVEC_DATA_TYPE_BINARY;
  uint8_t binary_data[] = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD};
  binary_field.value.string_value =
      *zvec_bin_create(binary_data, sizeof(binary_data));
  err = zvec_doc_add_field_by_struct(doc, &binary_field);
  TEST_ASSERT(err == ZVEC_OK);

  void *binary_result;
  size_t binary_size;
  err = zvec_doc_get_field_value_copy(
      doc, "binary_field", ZVEC_DATA_TYPE_BINARY, &binary_result, &binary_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(binary_result != NULL);
  TEST_ASSERT(binary_size == 6);
  TEST_ASSERT(memcmp(binary_result, "\x00\x01\x02\xFF\xFE\xFD", binary_size) ==
              0);
  free(binary_result);

  // VECTOR_FP32 type
  float test_vector[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  ZVecDocField fp32_vec_field;
  fp32_vec_field.name.data = "fp32_vec_field";
  fp32_vec_field.name.length = strlen("fp32_vec_field");
  fp32_vec_field.data_type = ZVEC_DATA_TYPE_VECTOR_FP32;
  fp32_vec_field.value.vector_value.data = test_vector;
  fp32_vec_field.value.vector_value.length = 5;
  err = zvec_doc_add_field_by_struct(doc, &fp32_vec_field);
  TEST_ASSERT(err == ZVEC_OK);

  void *fp32_vec_result;
  size_t fp32_vec_size;
  err = zvec_doc_get_field_value_copy(doc, "fp32_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_FP32,
                                      &fp32_vec_result, &fp32_vec_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp32_vec_result != NULL);
  TEST_ASSERT(fp32_vec_size == 5 * sizeof(float));
  TEST_ASSERT(memcmp(fp32_vec_result, test_vector, fp32_vec_size) == 0);
  free(fp32_vec_result);

  // VECTOR_FP16 type (16-bit float vector)
  uint16_t fp16_data[] = {0x3C00, 0x4000, 0x4200,
                          0x4400};  // FP16: 1.0, 2.0, 3.0, 4.0
  err = zvec_doc_add_field_by_value(doc, "fp16_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_FP16, fp16_data,
                                    sizeof(fp16_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *fp16_result;
  size_t fp16_size;
  err = zvec_doc_get_field_value_copy(doc, "fp16_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_FP16, &fp16_result,
                                      &fp16_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp16_result != NULL);
  TEST_ASSERT(fp16_size == sizeof(fp16_data));
  TEST_ASSERT(memcmp(fp16_result, fp16_data, fp16_size) == 0);
  free(fp16_result);

  // VECTOR_INT8 type
  int8_t int8_data[] = {-128, -1, 0, 1, 127};
  err = zvec_doc_add_field_by_value(doc, "int8_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_INT8, int8_data,
                                    sizeof(int8_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *int8_result;
  size_t int8_size;
  err = zvec_doc_get_field_value_copy(doc, "int8_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_INT8, &int8_result,
                                      &int8_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int8_result != NULL);
  TEST_ASSERT(int8_size == sizeof(int8_data));
  TEST_ASSERT(memcmp(int8_result, int8_data, int8_size) == 0);
  free(int8_result);

  // VECTOR_BINARY32 type (32-bit aligned binary vector)
  uint8_t bin32_data[] = {0xAA, 0x55, 0xAA, 0x55};
  err = zvec_doc_add_field_by_value(doc, "bin32_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_BINARY32, bin32_data,
                                    sizeof(bin32_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *bin32_result;
  size_t bin32_size;
  err = zvec_doc_get_field_value_copy(doc, "bin32_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_BINARY32,
                                      &bin32_result, &bin32_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bin32_result != NULL);
  TEST_ASSERT(bin32_size == sizeof(bin32_data));
  TEST_ASSERT(memcmp(bin32_result, bin32_data, bin32_size) == 0);
  free(bin32_result);

  // VECTOR_BINARY64 type (64-bit aligned binary vector)
  uint64_t bin64_data[] = {0xAA55AA55AA55AA55ULL, 0x55AA55AA55AA55AAULL};
  err = zvec_doc_add_field_by_value(doc, "bin64_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_BINARY64, bin64_data,
                                    sizeof(bin64_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *bin64_result;
  size_t bin64_size;
  err = zvec_doc_get_field_value_copy(doc, "bin64_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_BINARY64,
                                      &bin64_result, &bin64_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bin64_result != NULL);
  TEST_ASSERT(bin64_size == sizeof(bin64_data));
  TEST_ASSERT(memcmp(bin64_result, bin64_data, bin64_size) == 0);
  free(bin64_result);

  // VECTOR_FP64 type (double precision vector)
  double fp64_data[] = {1.1, 2.2, 3.3, 4.4};
  err = zvec_doc_add_field_by_value(doc, "fp64_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_FP64, fp64_data,
                                    sizeof(fp64_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *fp64_result;
  size_t fp64_size;
  err = zvec_doc_get_field_value_copy(doc, "fp64_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_FP64, &fp64_result,
                                      &fp64_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp64_result != NULL);
  TEST_ASSERT(fp64_size == sizeof(fp64_data));
  TEST_ASSERT(memcmp(fp64_result, fp64_data, fp64_size) == 0);
  free(fp64_result);

  // VECTOR_INT16 type
  int16_t int16_data[] = {-32768, -1, 0, 1, 32767};
  err = zvec_doc_add_field_by_value(doc, "int16_vec_field",
                                    ZVEC_DATA_TYPE_VECTOR_INT16, int16_data,
                                    sizeof(int16_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *int16_result;
  size_t int16_size;
  err = zvec_doc_get_field_value_copy(doc, "int16_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_INT16,
                                      &int16_result, &int16_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int16_result != NULL);
  TEST_ASSERT(int16_size == sizeof(int16_data));
  TEST_ASSERT(memcmp(int16_result, int16_data, int16_size) == 0);
  free(int16_result);

  // SPARSE_VECTOR_FP16 type - format: [nnz(uint32_t)][indices...][values...]
  uint32_t sparse_fp16_nnz = 3;
  size_t sparse_fp16_size_input =
      sizeof(uint32_t) +
      sparse_fp16_nnz * (sizeof(uint32_t) + sizeof(uint16_t));
  void *sparse_fp16_input = malloc(sparse_fp16_size_input);
  uint32_t *fp16_nnz_ptr = (uint32_t *)sparse_fp16_input;
  *fp16_nnz_ptr = sparse_fp16_nnz;
  uint32_t *fp16_indices =
      (uint32_t *)((char *)sparse_fp16_input + sizeof(uint32_t));
  uint16_t *fp16_values =
      (uint16_t *)((char *)sparse_fp16_input + sizeof(uint32_t) +
                   sparse_fp16_nnz * sizeof(uint32_t));
  fp16_indices[0] = 0;
  fp16_indices[1] = 5;
  fp16_indices[2] = 10;
  fp16_values[0] = 0x3C00;
  fp16_values[1] = 0x4000;
  fp16_values[2] = 0x4200;  // FP16: 1.0, 2.0, 3.0
  err = zvec_doc_add_field_by_value(doc, "sparse_fp16_field",
                                    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16,
                                    sparse_fp16_input, sparse_fp16_size_input);
  TEST_ASSERT(err == ZVEC_OK);
  free(sparse_fp16_input);

  void *sparse_fp16_result;
  size_t sparse_fp16_result_size;
  err = zvec_doc_get_field_value_copy(
      doc, "sparse_fp16_field", ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16,
      &sparse_fp16_result, &sparse_fp16_result_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(sparse_fp16_result != NULL);
  // Sparse vector format: [nnz(size_t)][indices...][values...]
  size_t retrieved_nnz = *(size_t *)sparse_fp16_result;
  TEST_ASSERT(retrieved_nnz == 3);
  uint32_t *retrieved_fp16_indices =
      (uint32_t *)((char *)sparse_fp16_result + sizeof(size_t));
  uint16_t *retrieved_fp16_vals =
      (uint16_t *)((char *)sparse_fp16_result + sizeof(size_t) +
                   retrieved_nnz * sizeof(uint32_t));
  TEST_ASSERT(retrieved_fp16_indices[0] == 0);
  TEST_ASSERT(retrieved_fp16_indices[1] == 5);
  TEST_ASSERT(retrieved_fp16_indices[2] == 10);
  TEST_ASSERT(retrieved_fp16_vals[0] == 0x3C00);
  TEST_ASSERT(retrieved_fp16_vals[1] == 0x4000);
  TEST_ASSERT(retrieved_fp16_vals[2] == 0x4200);
  free(sparse_fp16_result);

  // SPARSE_VECTOR_FP32 type - format: [nnz(uint32_t)][indices...][values...]
  uint32_t sparse_fp32_nnz = 4;
  size_t sparse_fp32_size_input =
      sizeof(uint32_t) + sparse_fp32_nnz * (sizeof(uint32_t) + sizeof(float));
  void *sparse_fp32_input = malloc(sparse_fp32_size_input);
  uint32_t *fp32_nnz_ptr = (uint32_t *)sparse_fp32_input;
  *fp32_nnz_ptr = sparse_fp32_nnz;
  uint32_t *fp32_indices =
      (uint32_t *)((char *)sparse_fp32_input + sizeof(uint32_t));
  float *fp32_values = (float *)((char *)sparse_fp32_input + sizeof(uint32_t) +
                                 sparse_fp32_nnz * sizeof(uint32_t));
  fp32_indices[0] = 2;
  fp32_indices[1] = 7;
  fp32_indices[2] = 15;
  fp32_indices[3] = 20;
  fp32_values[0] = 1.5f;
  fp32_values[1] = 2.5f;
  fp32_values[2] = 3.5f;
  fp32_values[3] = 4.5f;
  err = zvec_doc_add_field_by_value(doc, "sparse_fp32_field",
                                    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32,
                                    sparse_fp32_input, sparse_fp32_size_input);
  TEST_ASSERT(err == ZVEC_OK);
  free(sparse_fp32_input);

  void *sparse_fp32_result;
  size_t sparse_fp32_result_size;
  err = zvec_doc_get_field_value_copy(
      doc, "sparse_fp32_field", ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32,
      &sparse_fp32_result, &sparse_fp32_result_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(sparse_fp32_result != NULL);
  retrieved_nnz = *(size_t *)sparse_fp32_result;
  TEST_ASSERT(retrieved_nnz == 4);
  uint32_t *retrieved_fp32_indices =
      (uint32_t *)((char *)sparse_fp32_result + sizeof(size_t));
  float *retrieved_fp32_vals =
      (float *)((char *)sparse_fp32_result + sizeof(size_t) +
                retrieved_nnz * sizeof(uint32_t));
  TEST_ASSERT(retrieved_fp32_indices[0] == 2);
  TEST_ASSERT(retrieved_fp32_indices[1] == 7);
  TEST_ASSERT(retrieved_fp32_indices[2] == 15);
  TEST_ASSERT(retrieved_fp32_indices[3] == 20);
  TEST_ASSERT(fabs(retrieved_fp32_vals[0] - 1.5f) < 1e-5f);
  TEST_ASSERT(fabs(retrieved_fp32_vals[1] - 2.5f) < 1e-5f);
  TEST_ASSERT(fabs(retrieved_fp32_vals[2] - 3.5f) < 1e-5f);
  TEST_ASSERT(fabs(retrieved_fp32_vals[3] - 4.5f) < 1e-5f);
  free(sparse_fp32_result);

  // ARRAY_BINARY type
  // Format: [length(uint32_t)][data][length][data]...
  uint8_t array_bin_data[] = {
      1, 0, 0, 0, 0x01,        // length=1, data=0x01
      2, 0, 0, 0, 0x02, 0x03,  // length=2, data=0x02,0x03
      2, 0, 0, 0, 0x04, 0x05   // length=2, data=0x04,0x05
  };
  err = zvec_doc_add_field_by_value(doc, "array_binary_field",
                                    ZVEC_DATA_TYPE_ARRAY_BINARY, array_bin_data,
                                    sizeof(array_bin_data));
  TEST_ASSERT(err == ZVEC_OK);
  void *array_binary_result;
  size_t array_binary_size;
  err = zvec_doc_get_field_value_copy(doc, "array_binary_field",
                                      ZVEC_DATA_TYPE_ARRAY_BINARY,
                                      &array_binary_result, &array_binary_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_binary_result != NULL);
  // The result is a contiguous buffer of binary data without length prefixes
  TEST_ASSERT(array_binary_size == 5);  // 1 + 2 + 2 bytes
  const uint8_t *result_bytes = (const uint8_t *)array_binary_result;
  TEST_ASSERT(result_bytes[0] == 0x01);
  TEST_ASSERT(result_bytes[1] == 0x02);
  TEST_ASSERT(result_bytes[2] == 0x03);
  TEST_ASSERT(result_bytes[3] == 0x04);
  TEST_ASSERT(result_bytes[4] == 0x05);
  free(array_binary_result);


  // ARRAY_STRING type
  const char *array_str_data[] = {"str1", "str2", "str3"};
  ZVecString *array_zvec_str[3];
  for (int i = 0; i < 3; i++) {
    array_zvec_str[i] = zvec_string_create(array_str_data[i]);
  }
  err = zvec_doc_add_field_by_value(doc, "array_string_field",
                                    ZVEC_DATA_TYPE_ARRAY_STRING, array_zvec_str,
                                    sizeof(array_zvec_str));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_string_result;
  size_t array_string_size;
  err = zvec_doc_get_field_value_copy(doc, "array_string_field",
                                      ZVEC_DATA_TYPE_ARRAY_STRING,
                                      &array_string_result, &array_string_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_string_result != NULL);
  free(array_string_result);
  for (int i = 0; i < 3; i++) {
    zvec_free_string(array_zvec_str[i]);
  }

  free(string_field.value.string_value.data);

  // ARRAY_BOOL type
  bool array_bool_data[] = {true, false, true, false, true};
  err = zvec_doc_add_field_by_value(doc, "array_bool_field",
                                    ZVEC_DATA_TYPE_ARRAY_BOOL, array_bool_data,
                                    sizeof(array_bool_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_bool_result;
  size_t array_bool_size;
  err = zvec_doc_get_field_value_copy(doc, "array_bool_field",
                                      ZVEC_DATA_TYPE_ARRAY_BOOL,
                                      &array_bool_result, &array_bool_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_bool_result != NULL);
  // Verify the bit-packed bool array
  uint8_t *bool_bytes = (uint8_t *)array_bool_result;
  TEST_ASSERT((bool_bytes[0] & 0x01) != 0);  // index 0: true
  TEST_ASSERT((bool_bytes[0] & 0x02) == 0);  // index 1: false
  TEST_ASSERT((bool_bytes[0] & 0x04) != 0);  // index 2: true
  TEST_ASSERT((bool_bytes[0] & 0x08) == 0);  // index 3: false
  TEST_ASSERT((bool_bytes[0] & 0x10) != 0);  // index 4: true
  free(array_bool_result);

  // ARRAY_INT32 type
  int32_t array_int32_data[] = {100, 200, 300};
  err = zvec_doc_add_field_by_value(doc, "array_int32_field",
                                    ZVEC_DATA_TYPE_ARRAY_INT32,
                                    array_int32_data, sizeof(array_int32_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_int32_result;
  size_t array_int32_size;
  err = zvec_doc_get_field_value_copy(doc, "array_int32_field",
                                      ZVEC_DATA_TYPE_ARRAY_INT32,
                                      &array_int32_result, &array_int32_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_int32_result != NULL);
  TEST_ASSERT(array_int32_size == sizeof(array_int32_data));
  TEST_ASSERT(((int32_t *)array_int32_result)[0] == 100);
  TEST_ASSERT(((int32_t *)array_int32_result)[1] == 200);
  TEST_ASSERT(((int32_t *)array_int32_result)[2] == 300);
  free(array_int32_result);

  // ARRAY_INT64 type
  int64_t array_int64_data[] = {-9223372036854775807LL, 0,
                                9223372036854775807LL};
  err = zvec_doc_add_field_by_value(doc, "array_int64_field",
                                    ZVEC_DATA_TYPE_ARRAY_INT64,
                                    array_int64_data, sizeof(array_int64_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_int64_result;
  size_t array_int64_size;
  err = zvec_doc_get_field_value_copy(doc, "array_int64_field",
                                      ZVEC_DATA_TYPE_ARRAY_INT64,
                                      &array_int64_result, &array_int64_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_int64_result != NULL);
  TEST_ASSERT(array_int64_size == sizeof(array_int64_data));
  TEST_ASSERT(((int64_t *)array_int64_result)[0] == -9223372036854775807LL);
  TEST_ASSERT(((int64_t *)array_int64_result)[1] == 0);
  TEST_ASSERT(((int64_t *)array_int64_result)[2] == 9223372036854775807LL);
  free(array_int64_result);

  // ARRAY_UINT32 type
  uint32_t array_uint32_data[] = {0U, 1000000U, 4000000000U};
  err = zvec_doc_add_field_by_value(
      doc, "array_uint32_field", ZVEC_DATA_TYPE_ARRAY_UINT32, array_uint32_data,
      sizeof(array_uint32_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_uint32_result;
  size_t array_uint32_size;
  err = zvec_doc_get_field_value_copy(doc, "array_uint32_field",
                                      ZVEC_DATA_TYPE_ARRAY_UINT32,
                                      &array_uint32_result, &array_uint32_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_uint32_result != NULL);
  TEST_ASSERT(array_uint32_size == sizeof(array_uint32_data));
  TEST_ASSERT(((uint32_t *)array_uint32_result)[0] == 0U);
  TEST_ASSERT(((uint32_t *)array_uint32_result)[1] == 1000000U);
  TEST_ASSERT(((uint32_t *)array_uint32_result)[2] == 4000000000U);
  free(array_uint32_result);

  // ARRAY_UINT64 type
  uint64_t array_uint64_data[] = {0ULL, 1000000000000ULL,
                                  18000000000000000000ULL};
  err = zvec_doc_add_field_by_value(
      doc, "array_uint64_field", ZVEC_DATA_TYPE_ARRAY_UINT64, array_uint64_data,
      sizeof(array_uint64_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_uint64_result;
  size_t array_uint64_size;
  err = zvec_doc_get_field_value_copy(doc, "array_uint64_field",
                                      ZVEC_DATA_TYPE_ARRAY_UINT64,
                                      &array_uint64_result, &array_uint64_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_uint64_result != NULL);
  TEST_ASSERT(array_uint64_size == sizeof(array_uint64_data));
  TEST_ASSERT(((uint64_t *)array_uint64_result)[0] == 0ULL);
  TEST_ASSERT(((uint64_t *)array_uint64_result)[1] == 1000000000000ULL);
  TEST_ASSERT(((uint64_t *)array_uint64_result)[2] == 18000000000000000000ULL);
  free(array_uint64_result);

  // ARRAY_FLOAT type
  float array_float_data[] = {1.5f, 2.5f, 3.5f};
  err = zvec_doc_add_field_by_value(doc, "array_float_field",
                                    ZVEC_DATA_TYPE_ARRAY_FLOAT,
                                    array_float_data, sizeof(array_float_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_float_result;
  size_t array_float_size;
  err = zvec_doc_get_field_value_copy(doc, "array_float_field",
                                      ZVEC_DATA_TYPE_ARRAY_FLOAT,
                                      &array_float_result, &array_float_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_float_result != NULL);
  TEST_ASSERT(array_float_size == sizeof(array_float_data));
  TEST_ASSERT(((float *)array_float_result)[0] == 1.5f);
  TEST_ASSERT(((float *)array_float_result)[1] == 2.5f);
  TEST_ASSERT(((float *)array_float_result)[2] == 3.5f);
  free(array_float_result);

  // ARRAY_DOUBLE type
  double array_double_data[] = {1.111111, 2.222222, 3.333333};
  err = zvec_doc_add_field_by_value(
      doc, "array_double_field", ZVEC_DATA_TYPE_ARRAY_DOUBLE, array_double_data,
      sizeof(array_double_data));
  TEST_ASSERT(err == ZVEC_OK);

  void *array_double_result;
  size_t array_double_size;
  err = zvec_doc_get_field_value_copy(doc, "array_double_field",
                                      ZVEC_DATA_TYPE_ARRAY_DOUBLE,
                                      &array_double_result, &array_double_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_double_result != NULL);
  TEST_ASSERT(array_double_size == sizeof(array_double_data));
  TEST_ASSERT(fabs(((double *)array_double_result)[0] - 1.111111) < 1e-10);
  TEST_ASSERT(fabs(((double *)array_double_result)[1] - 2.222222) < 1e-10);
  TEST_ASSERT(fabs(((double *)array_double_result)[2] - 3.333333) < 1e-10);
  free(array_double_result);


  free(binary_field.value.string_value.data);
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_get_field_value_pointer(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  ZVecErrorCode err;

  // Add fields for pointer testing
  ZVecDocField bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &bool_field);
  TEST_ASSERT(err == ZVEC_OK);

  ZVecDocField int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  ZVecDocField string_field;
  string_field.name.data = "string_field";
  string_field.name.length = strlen("string_field");
  string_field.data_type = ZVEC_DATA_TYPE_STRING;
  string_field.value.string_value = *zvec_string_create("Hello, 世界!");
  err = zvec_doc_add_field_by_struct(doc, &string_field);
  TEST_ASSERT(err == ZVEC_OK);

  ZVecDocField binary_field;
  binary_field.name.data = "binary_field";
  binary_field.name.length = strlen("binary_field");
  binary_field.data_type = ZVEC_DATA_TYPE_BINARY;
  uint8_t binary_data[] = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD};
  binary_field.value.string_value =
      *zvec_bin_create(binary_data, sizeof(binary_data));
  err = zvec_doc_add_field_by_struct(doc, &binary_field);
  TEST_ASSERT(err == ZVEC_OK);

  float test_vector[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  ZVecDocField fp32_vec_field;
  fp32_vec_field.name.data = "fp32_vec_field";
  fp32_vec_field.name.length = strlen("fp32_vec_field");
  fp32_vec_field.data_type = ZVEC_DATA_TYPE_VECTOR_FP32;
  fp32_vec_field.value.vector_value.data = test_vector;
  fp32_vec_field.value.vector_value.length = 5;
  err = zvec_doc_add_field_by_struct(doc, &fp32_vec_field);
  TEST_ASSERT(err == ZVEC_OK);

  // Add more fields for comprehensive pointer testing
  int64_t int64_val = -9223372036854775807LL;
  err =
      zvec_doc_add_field_by_value(doc, "int64_field_ptr", ZVEC_DATA_TYPE_INT64,
                                  &int64_val, sizeof(int64_val));
  TEST_ASSERT(err == ZVEC_OK);

  uint32_t uint32_val = 4000000000U;
  err = zvec_doc_add_field_by_value(doc, "uint32_field_ptr",
                                    ZVEC_DATA_TYPE_UINT32, &uint32_val,
                                    sizeof(uint32_val));
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t uint64_val = 18000000000000000000ULL;
  err = zvec_doc_add_field_by_value(doc, "uint64_field_ptr",
                                    ZVEC_DATA_TYPE_UINT64, &uint64_val,
                                    sizeof(uint64_val));
  TEST_ASSERT(err == ZVEC_OK);

  float float_val = 3.14159265f;
  err =
      zvec_doc_add_field_by_value(doc, "float_field_ptr", ZVEC_DATA_TYPE_FLOAT,
                                  &float_val, sizeof(float_val));
  TEST_ASSERT(err == ZVEC_OK);

  double double_val = 2.718281828459045;
  err = zvec_doc_add_field_by_value(doc, "double_field_ptr",
                                    ZVEC_DATA_TYPE_DOUBLE, &double_val,
                                    sizeof(double_val));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_BINARY64
  uint64_t bin64_vec_data[] = {0xAA55AA55AA55AA55ULL, 0x55AA55AA55AA55AAULL};
  err = zvec_doc_add_field_by_value(doc, "bin64_vec_field_ptr",
                                    ZVEC_DATA_TYPE_VECTOR_BINARY64,
                                    bin64_vec_data, sizeof(bin64_vec_data));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP16
  uint16_t fp16_vec_data[] = {0x3C00, 0x4000, 0x4200, 0x4400};
  err = zvec_doc_add_field_by_value(doc, "fp16_vec_field_ptr",
                                    ZVEC_DATA_TYPE_VECTOR_FP16, fp16_vec_data,
                                    sizeof(fp16_vec_data));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_FP64
  double fp64_vec_data[] = {1.1, 2.2, 3.3, 4.4};
  err = zvec_doc_add_field_by_value(doc, "fp64_vec_field_ptr",
                                    ZVEC_DATA_TYPE_VECTOR_FP64, fp64_vec_data,
                                    sizeof(fp64_vec_data));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT8
  int8_t int8_vec_data[] = {-128, -1, 0, 1, 127};
  err = zvec_doc_add_field_by_value(doc, "int8_vec_field_ptr",
                                    ZVEC_DATA_TYPE_VECTOR_INT8, int8_vec_data,
                                    sizeof(int8_vec_data));
  TEST_ASSERT(err == ZVEC_OK);

  // VECTOR_INT16
  int16_t int16_vec_data[] = {-32768, -1, 0, 1, 32767};
  err = zvec_doc_add_field_by_value(doc, "int16_vec_field_ptr",
                                    ZVEC_DATA_TYPE_VECTOR_INT16, int16_vec_data,
                                    sizeof(int16_vec_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_INT32
  int32_t array_int32_data[] = {100, 200, 300};
  err = zvec_doc_add_field_by_value(doc, "array_int32_field_ptr",
                                    ZVEC_DATA_TYPE_ARRAY_INT32,
                                    array_int32_data, sizeof(array_int32_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_INT64
  int64_t array_int64_data[] = {-9223372036854775807LL, 0,
                                9223372036854775807LL};
  err = zvec_doc_add_field_by_value(doc, "array_int64_field_ptr",
                                    ZVEC_DATA_TYPE_ARRAY_INT64,
                                    array_int64_data, sizeof(array_int64_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_UINT32
  uint32_t array_uint32_data[] = {0U, 1000000U, 4000000000U};
  err = zvec_doc_add_field_by_value(
      doc, "array_uint32_field_ptr", ZVEC_DATA_TYPE_ARRAY_UINT32,
      array_uint32_data, sizeof(array_uint32_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_UINT64
  uint64_t array_uint64_data[] = {0ULL, 1000000000000ULL,
                                  18000000000000000000ULL};
  err = zvec_doc_add_field_by_value(
      doc, "array_uint64_field_ptr", ZVEC_DATA_TYPE_ARRAY_UINT64,
      array_uint64_data, sizeof(array_uint64_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_FLOAT
  float array_float_data[] = {1.5f, 2.5f, 3.5f};
  err = zvec_doc_add_field_by_value(doc, "array_float_field_ptr",
                                    ZVEC_DATA_TYPE_ARRAY_FLOAT,
                                    array_float_data, sizeof(array_float_data));
  TEST_ASSERT(err == ZVEC_OK);

  // ARRAY_DOUBLE
  double array_double_data[] = {1.111111, 2.222222, 3.333333};
  err = zvec_doc_add_field_by_value(
      doc, "array_double_field_ptr", ZVEC_DATA_TYPE_ARRAY_DOUBLE,
      array_double_data, sizeof(array_double_data));
  TEST_ASSERT(err == ZVEC_OK);

  printf(
      "=== Testing zvec_doc_get_field_value_pointer with all supported types "
      "===\n");

  // Test pointer access to BOOL
  const void *bool_ptr;
  size_t bool_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                                         &bool_ptr, &bool_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bool_ptr != NULL);
  TEST_ASSERT(bool_ptr_size == sizeof(bool));
  TEST_ASSERT(*(const bool *)bool_ptr == true);

  // Test pointer access to INT32
  const void *int32_ptr;
  size_t int32_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "int32_field", ZVEC_DATA_TYPE_INT32, &int32_ptr, &int32_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int32_ptr != NULL);
  TEST_ASSERT(int32_ptr_size == sizeof(int32_t));
  TEST_ASSERT(*(const int32_t *)int32_ptr == -2147483648);

  // Test pointer access to STRING
  const void *string_ptr;
  size_t string_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "string_field",
                                         ZVEC_DATA_TYPE_STRING, &string_ptr,
                                         &string_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(string_ptr != NULL);
  TEST_ASSERT(string_ptr_size == strlen("Hello, 世界!"));
  TEST_ASSERT(memcmp(string_ptr, "Hello, 世界!", string_ptr_size) == 0);

  // Test pointer access to BINARY
  const void *binary_ptr;
  size_t binary_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "binary_field",
                                         ZVEC_DATA_TYPE_BINARY, &binary_ptr,
                                         &binary_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(binary_ptr != NULL);
  TEST_ASSERT(binary_ptr_size == 6);
  TEST_ASSERT(memcmp(binary_ptr, "\x00\x01\x02\xFF\xFE\xFD", binary_ptr_size) ==
              0);

  // Test pointer access to VECTOR_FP32
  const void *fp32_vec_ptr;
  size_t fp32_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "fp32_vec_field",
                                         ZVEC_DATA_TYPE_VECTOR_FP32,
                                         &fp32_vec_ptr, &fp32_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp32_vec_ptr != NULL);
  TEST_ASSERT(fp32_vec_ptr_size == 5 * sizeof(float));
  TEST_ASSERT(memcmp(fp32_vec_ptr, test_vector, fp32_vec_ptr_size) == 0);

  // Test pointer access to INT64
  const void *int64_ptr;
  size_t int64_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "int64_field_ptr",
                                         ZVEC_DATA_TYPE_INT64, &int64_ptr,
                                         &int64_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int64_ptr != NULL);
  TEST_ASSERT(int64_ptr_size == sizeof(int64_t));
  TEST_ASSERT(*(const int64_t *)int64_ptr == -9223372036854775807LL);

  // Test pointer access to UINT32
  const void *uint32_ptr;
  size_t uint32_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "uint32_field_ptr",
                                         ZVEC_DATA_TYPE_UINT32, &uint32_ptr,
                                         &uint32_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint32_ptr != NULL);
  TEST_ASSERT(uint32_ptr_size == sizeof(uint32_t));
  TEST_ASSERT(*(const uint32_t *)uint32_ptr == 4000000000U);

  // Test pointer access to UINT64
  const void *uint64_ptr;
  size_t uint64_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "uint64_field_ptr",
                                         ZVEC_DATA_TYPE_UINT64, &uint64_ptr,
                                         &uint64_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint64_ptr != NULL);
  TEST_ASSERT(uint64_ptr_size == sizeof(uint64_t));
  TEST_ASSERT(*(const uint64_t *)uint64_ptr == 18000000000000000000ULL);

  // Test pointer access to FLOAT
  const void *float_ptr;
  size_t float_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "float_field_ptr",
                                         ZVEC_DATA_TYPE_FLOAT, &float_ptr,
                                         &float_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(float_ptr != NULL);
  TEST_ASSERT(float_ptr_size == sizeof(float));
  TEST_ASSERT(fabs(*(const float *)float_ptr - 3.14159265f) < 1e-6f);

  // Test pointer access to DOUBLE
  const void *double_ptr;
  size_t double_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "double_field_ptr",
                                         ZVEC_DATA_TYPE_DOUBLE, &double_ptr,
                                         &double_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(double_ptr != NULL);
  TEST_ASSERT(double_ptr_size == sizeof(double));
  TEST_ASSERT(fabs(*(const double *)double_ptr - 2.718281828459045) < 1e-15);

  // Test pointer access to VECTOR_BINARY64
  const void *bin64_vec_ptr;
  size_t bin64_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "bin64_vec_field_ptr",
                                         ZVEC_DATA_TYPE_VECTOR_BINARY64,
                                         &bin64_vec_ptr, &bin64_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bin64_vec_ptr != NULL);
  TEST_ASSERT(bin64_vec_ptr_size == sizeof(bin64_vec_data));
  TEST_ASSERT(memcmp(bin64_vec_ptr, bin64_vec_data, bin64_vec_ptr_size) == 0);

  // Test pointer access to VECTOR_FP16
  const void *fp16_vec_ptr;
  size_t fp16_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "fp16_vec_field_ptr",
                                         ZVEC_DATA_TYPE_VECTOR_FP16,
                                         &fp16_vec_ptr, &fp16_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp16_vec_ptr != NULL);
  TEST_ASSERT(fp16_vec_ptr_size == sizeof(fp16_vec_data));
  TEST_ASSERT(memcmp(fp16_vec_ptr, fp16_vec_data, fp16_vec_ptr_size) == 0);

  // Test pointer access to VECTOR_FP64
  const void *fp64_vec_ptr;
  size_t fp64_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "fp64_vec_field_ptr",
                                         ZVEC_DATA_TYPE_VECTOR_FP64,
                                         &fp64_vec_ptr, &fp64_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp64_vec_ptr != NULL);
  TEST_ASSERT(fp64_vec_ptr_size == sizeof(fp64_vec_data));
  TEST_ASSERT(memcmp(fp64_vec_ptr, fp64_vec_data, fp64_vec_ptr_size) == 0);

  // Test pointer access to VECTOR_INT8
  const void *int8_vec_ptr;
  size_t int8_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "int8_vec_field_ptr",
                                         ZVEC_DATA_TYPE_VECTOR_INT8,
                                         &int8_vec_ptr, &int8_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int8_vec_ptr != NULL);
  TEST_ASSERT(int8_vec_ptr_size == sizeof(int8_vec_data));
  TEST_ASSERT(memcmp(int8_vec_ptr, int8_vec_data, int8_vec_ptr_size) == 0);

  // Test pointer access to VECTOR_INT16
  const void *int16_vec_ptr;
  size_t int16_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "int16_vec_field_ptr",
                                         ZVEC_DATA_TYPE_VECTOR_INT16,
                                         &int16_vec_ptr, &int16_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int16_vec_ptr != NULL);
  TEST_ASSERT(int16_vec_ptr_size == sizeof(int16_vec_data));
  TEST_ASSERT(memcmp(int16_vec_ptr, int16_vec_data, int16_vec_ptr_size) == 0);

  // Test pointer access to ARRAY_INT32
  const void *array_int32_ptr;
  size_t array_int32_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "array_int32_field_ptr", ZVEC_DATA_TYPE_ARRAY_INT32,
      &array_int32_ptr, &array_int32_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_int32_ptr != NULL);
  TEST_ASSERT(array_int32_ptr_size == sizeof(array_int32_data));
  TEST_ASSERT(((const int32_t *)array_int32_ptr)[0] == 100);
  TEST_ASSERT(((const int32_t *)array_int32_ptr)[1] == 200);
  TEST_ASSERT(((const int32_t *)array_int32_ptr)[2] == 300);

  // Test pointer access to ARRAY_INT64
  const void *array_int64_ptr;
  size_t array_int64_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "array_int64_field_ptr", ZVEC_DATA_TYPE_ARRAY_INT64,
      &array_int64_ptr, &array_int64_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_int64_ptr != NULL);
  TEST_ASSERT(array_int64_ptr_size == sizeof(array_int64_data));
  TEST_ASSERT(((const int64_t *)array_int64_ptr)[0] == -9223372036854775807LL);
  TEST_ASSERT(((const int64_t *)array_int64_ptr)[1] == 0);
  TEST_ASSERT(((const int64_t *)array_int64_ptr)[2] == 9223372036854775807LL);

  // Test pointer access to ARRAY_UINT32
  const void *array_uint32_ptr;
  size_t array_uint32_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "array_uint32_field_ptr", ZVEC_DATA_TYPE_ARRAY_UINT32,
      &array_uint32_ptr, &array_uint32_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_uint32_ptr != NULL);
  TEST_ASSERT(array_uint32_ptr_size == sizeof(array_uint32_data));
  TEST_ASSERT(((const uint32_t *)array_uint32_ptr)[0] == 0U);
  TEST_ASSERT(((const uint32_t *)array_uint32_ptr)[1] == 1000000U);
  TEST_ASSERT(((const uint32_t *)array_uint32_ptr)[2] == 4000000000U);

  // Test pointer access to ARRAY_UINT64
  const void *array_uint64_ptr;
  size_t array_uint64_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "array_uint64_field_ptr", ZVEC_DATA_TYPE_ARRAY_UINT64,
      &array_uint64_ptr, &array_uint64_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_uint64_ptr != NULL);
  TEST_ASSERT(array_uint64_ptr_size == sizeof(array_uint64_data));
  TEST_ASSERT(((const uint64_t *)array_uint64_ptr)[0] == 0ULL);
  TEST_ASSERT(((const uint64_t *)array_uint64_ptr)[1] == 1000000000000ULL);
  TEST_ASSERT(((const uint64_t *)array_uint64_ptr)[2] ==
              18000000000000000000ULL);

  // Test pointer access to ARRAY_FLOAT
  const void *array_float_ptr;
  size_t array_float_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "array_float_field_ptr", ZVEC_DATA_TYPE_ARRAY_FLOAT,
      &array_float_ptr, &array_float_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_float_ptr != NULL);
  TEST_ASSERT(array_float_ptr_size == sizeof(array_float_data));
  TEST_ASSERT(((const float *)array_float_ptr)[0] == 1.5f);
  TEST_ASSERT(((const float *)array_float_ptr)[1] == 2.5f);
  TEST_ASSERT(((const float *)array_float_ptr)[2] == 3.5f);

  // Test pointer access to ARRAY_DOUBLE
  const void *array_double_ptr;
  size_t array_double_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "array_double_field_ptr", ZVEC_DATA_TYPE_ARRAY_DOUBLE,
      &array_double_ptr, &array_double_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(array_double_ptr != NULL);
  TEST_ASSERT(array_double_ptr_size == sizeof(array_double_data));
  TEST_ASSERT(fabs(((const double *)array_double_ptr)[0] - 1.111111) < 1e-10);
  TEST_ASSERT(fabs(((const double *)array_double_ptr)[1] - 2.222222) < 1e-10);
  TEST_ASSERT(fabs(((const double *)array_double_ptr)[2] - 3.333333) < 1e-10);

  free(string_field.value.string_value.data);
  free(binary_field.value.string_value.data);
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_field_operations(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  ZVecErrorCode err;

  // Add some fields
  ZVecDocField bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &bool_field);
  TEST_ASSERT(err == ZVEC_OK);

  ZVecDocField int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  ZVecDocField string_field;
  string_field.name.data = "string_field";
  string_field.name.length = strlen("string_field");
  string_field.data_type = ZVEC_DATA_TYPE_STRING;
  string_field.value.string_value = *zvec_string_create("Hello");
  err = zvec_doc_add_field_by_struct(doc, &string_field);
  TEST_ASSERT(err == ZVEC_OK);

  // Test field count
  size_t field_count = zvec_doc_get_field_count(doc);
  TEST_ASSERT(field_count >= 3);

  // Test field existence checks
  TEST_ASSERT(zvec_doc_has_field(doc, "bool_field") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "int32_field") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "string_field") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "nonexistent") == false);

  TEST_ASSERT(zvec_doc_has_field_value(doc, "bool_field") == true);
  TEST_ASSERT(zvec_doc_is_field_null(doc, "bool_field") == false);
  TEST_ASSERT(zvec_doc_is_field_null(doc, "nonexistent") == false);

  // Test field names retrieval
  char **field_names;
  size_t name_count;
  err = zvec_doc_get_field_names(doc, &field_names, &name_count);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(name_count >= 3);
  TEST_ASSERT(field_names != NULL);

  // Verify some expected fields are present
  bool found_key_fields = false;
  for (size_t i = 0; i < name_count; i++) {
    if (strcmp(field_names[i], "bool_field") == 0 ||
        strcmp(field_names[i], "int32_field") == 0 ||
        strcmp(field_names[i], "string_field") == 0) {
      found_key_fields = true;
      break;
    }
  }
  TEST_ASSERT(found_key_fields == true);

  zvec_free_str_array(field_names, name_count);
  free(string_field.value.string_value.data);
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_error_conditions(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  // Add a field for error testing
  ZVecDocField bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  zvec_doc_add_field_by_struct(doc, &bool_field);

  ZVecErrorCode err;
  const void *dummy_ptr;
  size_t dummy_ptr_size;
  int32_t int32_result;
  void *string_result;
  size_t string_size;

  printf("=== Testing error conditions ===\n");

  // Test non-existent field
  err =
      zvec_doc_get_field_value_basic(doc, "missing_field", ZVEC_DATA_TYPE_INT32,
                                     &int32_result, sizeof(int32_result));
  TEST_ASSERT(err != ZVEC_OK);

  err =
      zvec_doc_get_field_value_copy(doc, "missing_field", ZVEC_DATA_TYPE_STRING,
                                    &string_result, &string_size);
  TEST_ASSERT(err != ZVEC_OK);

  err = zvec_doc_get_field_value_pointer(
      doc, "missing_field", ZVEC_DATA_TYPE_FLOAT, &dummy_ptr, &dummy_ptr_size);
  TEST_ASSERT(err != ZVEC_OK);

  // Test wrong data type access
  err = zvec_doc_get_field_value_basic(doc, "bool_field", ZVEC_DATA_TYPE_INT32,
                                       &int32_result, sizeof(int32_result));
  TEST_ASSERT(err != ZVEC_OK);

  err = zvec_doc_get_field_value_copy(doc, "bool_field", ZVEC_DATA_TYPE_STRING,
                                      &string_result, &string_size);
  TEST_ASSERT(err != ZVEC_OK);

  err = zvec_doc_get_field_value_pointer(
      doc, "bool_field", ZVEC_DATA_TYPE_FLOAT, &dummy_ptr, &dummy_ptr_size);
  TEST_ASSERT(err != ZVEC_OK);

  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_serialization(void) {
  TEST_START();

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  ZVecErrorCode err;

  // Add fields for serialization testing
  ZVecDocField int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  ZVecDocField string_field;
  string_field.name.data = "string_field";
  string_field.name.length = strlen("string_field");
  string_field.data_type = ZVEC_DATA_TYPE_STRING;
  string_field.value.string_value = *zvec_string_create("Serialization Test");
  err = zvec_doc_add_field_by_struct(doc, &string_field);
  TEST_ASSERT(err == ZVEC_OK);

  printf("=== Testing document serialization ===\n");

  uint8_t *serialized_data;
  size_t data_size;
  err = zvec_doc_serialize(doc, &serialized_data, &data_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(serialized_data != NULL);
  TEST_ASSERT(data_size > 0);

  ZVecDoc *deserialized_doc;
  err = zvec_doc_deserialize(serialized_data, data_size, &deserialized_doc);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(deserialized_doc != NULL);

  // Verify deserialized document has same field count
  size_t field_count = zvec_doc_get_field_count(doc);
  size_t deserialized_field_count = zvec_doc_get_field_count(deserialized_doc);
  TEST_ASSERT(deserialized_field_count == field_count);

  // Test a field from deserialized document
  int32_t deserialized_int32;
  err = zvec_doc_get_field_value_basic(
      deserialized_doc, "int32_field", ZVEC_DATA_TYPE_INT32,
      &deserialized_int32, sizeof(deserialized_int32));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(deserialized_int32 == -2147483648);

  zvec_free_uint8_array(serialized_data);
  free(string_field.value.string_value.data);
  zvec_doc_destroy(deserialized_doc);
  zvec_doc_destroy(doc);

  TEST_END();
}

// =============================================================================
// Index parameter tests
// =============================================================================

void test_index_params(void) {
  TEST_START();

  // Test HNSW parameter creation
  ZVecHnswIndexParams *hnsw_params = zvec_test_create_default_hnsw_params();
  TEST_ASSERT(hnsw_params != NULL);
  if (hnsw_params) {
    free(hnsw_params);
  }

  // Test Flat parameter creation
  ZVecFlatIndexParams *flat_params = zvec_test_create_default_flat_params();
  TEST_ASSERT(flat_params != NULL);
  if (flat_params) {
    free(flat_params);
  }

  // Test scalar index parameter creation
  ZVecInvertIndexParams *invert_params =
      zvec_test_create_default_invert_params(true);
  TEST_ASSERT(invert_params != NULL);
  if (invert_params) {
    free(invert_params);
  }

  TEST_END();
}

// =============================================================================
// Memory management tests
// =============================================================================
void test_zvec_string_functions(void) {
  TEST_START();

  // Test string creation and basic operations
  ZVecString *str1 = zvec_string_create("Hello World");
  TEST_ASSERT(str1 != NULL);
  TEST_ASSERT(zvec_string_length(str1) == 11);
  TEST_ASSERT(strcmp(zvec_string_c_str(str1), "Hello World") == 0);

  // Test string copy
  ZVecString *str2 = zvec_string_copy(str1);
  TEST_ASSERT(str2 != NULL);
  TEST_ASSERT(zvec_string_length(str2) == 11);
  TEST_ASSERT(strcmp(zvec_string_c_str(str2), "Hello World") == 0);

  // Test string comparison
  int cmp_result = zvec_string_compare(str1, str2);
  TEST_ASSERT(cmp_result == 0);

  ZVecString *str3 = zvec_string_create("Hello");
  TEST_ASSERT(zvec_string_compare(str1, str3) > 0);

  // Test string creation from view
  ZVecStringView view = {"Hello View", 10};
  ZVecString *str4 = zvec_string_create_from_view(&view);
  TEST_ASSERT(str4 != NULL);
  TEST_ASSERT(zvec_string_length(str4) == 10);
  TEST_ASSERT(strcmp(zvec_string_c_str(str4), "Hello View") == 0);

  // Test string view with embedded null bytes
  char binary_data[] = {'H', 'e', 'l', 'l', 'o', '\0', 'W', 'o', 'r', 'l', 'd'};
  ZVecStringView binary_view = {binary_data, 11};
  ZVecString *str5 = zvec_string_create_from_view(&binary_view);
  TEST_ASSERT(str5 != NULL);
  TEST_ASSERT(zvec_string_length(str5) == 11);
  // Note: strcmp will stop at first null byte, so we need to compare manually
  TEST_ASSERT(memcmp(zvec_string_c_str(str5), binary_data, 11) == 0);

  // Cleanup
  zvec_free_string(str1);
  zvec_free_string(str2);
  zvec_free_string(str3);
  zvec_free_string(str4);
  zvec_free_string(str5);

  TEST_END();
}

void test_index_params_functions(void) {
  TEST_START();

  // Test base index params
  ZVecBaseIndexParams base_params;
  zvec_index_params_base_init(&base_params, ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(base_params.index_type == ZVEC_INDEX_TYPE_HNSW);

  // Test invert index params
  ZVecInvertIndexParams invert_params;
  zvec_index_params_invert_init(&invert_params, true, false);
  TEST_ASSERT(invert_params.base.index_type == ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(invert_params.enable_range_optimization == true);
  TEST_ASSERT(invert_params.enable_extended_wildcard == false);

  // Test vector index params
  ZVecVectorIndexParams vector_params;
  zvec_index_params_vector_init(&vector_params, ZVEC_INDEX_TYPE_HNSW,
                                ZVEC_METRIC_TYPE_L2,
                                ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(vector_params.base.index_type == ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(vector_params.metric_type == ZVEC_METRIC_TYPE_L2);
  TEST_ASSERT(vector_params.quantize_type == ZVEC_QUANTIZE_TYPE_UNDEFINED);

  // Test HNSW index params
  ZVecHnswIndexParams hnsw_params;
  zvec_index_params_hnsw_init(&hnsw_params, ZVEC_METRIC_TYPE_COSINE, 16, 200,
                              50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(hnsw_params.base.base.index_type == ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params.base.metric_type == ZVEC_METRIC_TYPE_COSINE);
  TEST_ASSERT(hnsw_params.m == 16);
  TEST_ASSERT(hnsw_params.ef_construction == 200);
  TEST_ASSERT(hnsw_params.ef_search == 50);

  // Test Flat index params
  ZVecFlatIndexParams flat_params;
  zvec_index_params_flat_init(&flat_params, ZVEC_METRIC_TYPE_IP,
                              ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(flat_params.base.base.index_type == ZVEC_INDEX_TYPE_FLAT);
  TEST_ASSERT(flat_params.base.metric_type == ZVEC_METRIC_TYPE_IP);

  // Test IVF index params
  ZVecIVFIndexParams ivf_params;
  zvec_index_params_ivf_init(&ivf_params, ZVEC_METRIC_TYPE_L2, 100, 10, true, 5,
                             ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(ivf_params.base.base.index_type == ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(ivf_params.n_list == 100);
  TEST_ASSERT(ivf_params.n_iters == 10);
  TEST_ASSERT(ivf_params.use_soar == true);
  TEST_ASSERT(ivf_params.n_probe == 5);

  TEST_END();
}

void test_utility_functions(void) {
  TEST_START();

  // Test error code to string conversion
  const char *error_str = zvec_error_code_to_string(ZVEC_OK);
  TEST_ASSERT(error_str != NULL);
  TEST_ASSERT(strlen(error_str) > 0);

  error_str = zvec_error_code_to_string(ZVEC_ERROR_INVALID_ARGUMENT);
  TEST_ASSERT(error_str != NULL);

  // Test data type to string conversion
  const char *data_type_str = zvec_data_type_to_string(ZVEC_DATA_TYPE_INT32);
  TEST_ASSERT(data_type_str != NULL);
  TEST_ASSERT(strlen(data_type_str) > 0);

  data_type_str = zvec_data_type_to_string(ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(data_type_str != NULL);

  // Test index type to string conversion
  const char *index_type_str = zvec_index_type_to_string(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(index_type_str != NULL);
  TEST_ASSERT(strlen(index_type_str) > 0);

  index_type_str = zvec_index_type_to_string(ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(index_type_str != NULL);

  TEST_END();
}

void test_memory_management_functions(void) {
  TEST_START();

  // Test string allocation and deallocation
  ZVecString *str = zvec_string_create("Test String");
  TEST_ASSERT(str != NULL);
  zvec_free_string(str);

  TEST_END();
}

void test_query_params_functions(void) {
  TEST_START();

  // Test basic query parameters creation and destruction
  ZVecQueryParams *base_params = zvec_query_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(base_params != NULL);

  // Test union query parameters
  ZVecQueryParamsUnion *union_params =
      zvec_query_params_union_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(union_params != NULL);

  // Test HNSW query parameters
  ZVecHnswQueryParams *hnsw_params = zvec_query_params_hnsw_create(
      ZVEC_INDEX_TYPE_HNSW, 50, 0.5f, false, true);
  TEST_ASSERT(hnsw_params != NULL);

  // Test IVF query parameters
  ZVecIVFQueryParams *ivf_params =
      zvec_query_params_ivf_create(ZVEC_INDEX_TYPE_IVF, 10, true, 1.5f);
  TEST_ASSERT(ivf_params != NULL);

  // Test Flat query parameters
  ZVecFlatQueryParams *flat_params =
      zvec_query_params_flat_create(ZVEC_INDEX_TYPE_FLAT, false, 2.0f);
  TEST_ASSERT(flat_params != NULL);

  // Test setting various parameters on base query params
  ZVecErrorCode err;

  // Test index type setting
  err = zvec_query_params_set_index_type(base_params, ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(err == ZVEC_OK);

  // Test radius setting
  err = zvec_query_params_set_radius(base_params, 0.8f);
  TEST_ASSERT(err == ZVEC_OK);

  // Test linear search setting
  err = zvec_query_params_set_is_linear(base_params, false);
  TEST_ASSERT(err == ZVEC_OK);

  // Test refiner setting
  err = zvec_query_params_set_is_using_refiner(base_params, true);
  TEST_ASSERT(err == ZVEC_OK);

  // Test HNSW-specific parameters
  err = zvec_query_params_hnsw_set_ef(hnsw_params, 75);
  TEST_ASSERT(err == ZVEC_OK);

  // Test IVF-specific parameters
  err = zvec_query_params_ivf_set_nprobe(ivf_params, 15);
  TEST_ASSERT(err == ZVEC_OK);

  // Test IVF scale factor setting
  err = zvec_query_params_ivf_set_scale_factor(ivf_params, 2.5f);
  TEST_ASSERT(err == ZVEC_OK);

  // Test destruction of valid parameters
  zvec_query_params_destroy(base_params);
  zvec_query_params_hnsw_destroy(hnsw_params);
  zvec_query_params_ivf_destroy(ivf_params);
  zvec_query_params_flat_destroy(flat_params);
  zvec_query_params_union_destroy(union_params);


  // Test boundary cases - null pointer handling
  zvec_query_params_hnsw_destroy(NULL);
  zvec_query_params_ivf_destroy(NULL);
  zvec_query_params_flat_destroy(NULL);
  zvec_query_params_union_destroy(NULL);


  TEST_END();
}

void test_collection_stats_functions(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_stats_functions";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      ZVecCollectionStats *stats = NULL;

      // Test normal statistics retrieval
      err = zvec_collection_get_stats(collection, &stats);
      TEST_ASSERT(err == ZVEC_OK);

      if (stats) {
        TEST_ASSERT(stats->doc_count == 0);
        zvec_collection_stats_destroy(stats);
      }

      // Test NULL parameters
      err = zvec_collection_get_stats(NULL, &stats);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_get_stats(collection, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      // Test statistics destruction boundary cases
      zvec_collection_stats_destroy(NULL);
      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_dml_functions(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_dml";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test insertion function boundary cases
      size_t success_count, error_count;

      // Test NULL collection
      err = zvec_collection_insert(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test NULL document array
      err = zvec_collection_insert(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test zero document count
      ZVecDoc *empty_docs[1];
      err = zvec_collection_insert(collection, (const ZVecDoc **)empty_docs, 0,
                                   &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test NULL count pointer
      err = zvec_collection_insert(collection, (const ZVecDoc **)empty_docs, 1,
                                   NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test update function boundary cases
      err = zvec_collection_update(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_update(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_update(collection, (const ZVecDoc **)empty_docs, 0,
                                   NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test upsert function boundary cases
      err = zvec_collection_upsert(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_upsert(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_upsert(collection, (const ZVecDoc **)empty_docs, 0,
                                   NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test deletion function boundary cases
      const char *pks[1];
      err = zvec_collection_delete(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_delete(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_delete(collection, pks, 0, NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test deletion by filter boundary cases
      err = zvec_collection_delete_by_filter(NULL, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_delete_by_filter(collection, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

// =============================================================================
// Actual Query Execution Tests
// =============================================================================

void test_actual_vector_queries(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_actual_queries";

  // Create schema with vector field
  ZVecCollectionSchema *schema = zvec_collection_schema_create("query_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Add ID field
    ZVecFieldSchema *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    // Add vector field with HNSW index
    ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
        ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
    ZVecFieldSchema *vec_field = zvec_field_schema_create(
        "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
    zvec_field_schema_set_hnsw_index(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);

    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Insert test documents
      float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
      float vec2[] = {0.0f, 1.0f, 0.0f, 0.0f};
      float vec3[] = {0.0f, 0.0f, 1.0f, 0.0f};
      float vec4[] = {0.7f, 0.7f, 0.0f, 0.0f};  // Similar to vec1 and vec2

      ZVecDoc *docs[4];
      for (int i = 0; i < 4; i++) {
        docs[i] = zvec_doc_create();
        zvec_doc_set_pk(docs[i], zvec_test_make_pk(i + 1));
        zvec_doc_add_field_by_value(docs[i], "id", ZVEC_DATA_TYPE_INT64,
                                    &(int64_t){i + 1}, sizeof(int64_t));
      }

      zvec_doc_add_field_by_value(
          docs[0], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec1, sizeof(vec1));
      zvec_doc_add_field_by_value(
          docs[1], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec2, sizeof(vec2));
      zvec_doc_add_field_by_value(
          docs[2], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec3, sizeof(vec3));
      zvec_doc_add_field_by_value(
          docs[3], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec4, sizeof(vec4));

      size_t success_count, error_count;
      err = zvec_collection_insert(collection, (const ZVecDoc **)docs, 4,
                                   &success_count, &error_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(success_count == 4);
      TEST_ASSERT(error_count == 0);

      // Flush collection to build index
      zvec_collection_flush(collection);

      // Test 1: Basic vector search
      ZVecVectorQuery query1 = {0};
      query1.field_name = (ZVecString){.data = "embedding", .length = 9};
      query1.query_vector =
          (ZVecByteArray){.data = (uint8_t *)vec1, .length = sizeof(vec1)};
      query1.topk = 3;
      query1.include_vector = true;
      query1.include_doc_id = true;

      ZVecDoc **results = NULL;
      size_t result_count = 0;
      err = zvec_collection_query(collection, &query1, &results, &result_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(result_count > 0);
      TEST_ASSERT(results != NULL);

      // First result should be vec1 itself (distance ~0)
      if (result_count > 0) {
        float score = zvec_doc_get_score(results[0]);
        TEST_ASSERT(score < 0.001f);  // Very small distance
      }

      zvec_docs_free(results, result_count);

      // Test 2: Search with filter
      ZVecVectorQuery query2 = query1;
      query2.filter = (ZVecString){.data = "id > 2", .length = 6};

      err = zvec_collection_query(collection, &query2, &results, &result_count);
      TEST_ASSERT(err == ZVEC_OK);

      // Should only return documents with id > 2
      for (size_t i = 0; i < result_count; i++) {
        int64_t id;
        zvec_doc_get_field_value_basic(results[i], "id", ZVEC_DATA_TYPE_INT64,
                                       &id, sizeof(id));
        TEST_ASSERT(id > 2);
      }

      zvec_docs_free(results, result_count);

      // Cleanup documents
      for (int i = 0; i < 4; i++) {
        zvec_doc_destroy(docs[i]);
      }

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
    zvec_index_params_hnsw_destroy(hnsw_params);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_index_creation_and_management(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_index_management";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test 1: Create HNSW index
      ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
          ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
      TEST_ASSERT(hnsw_params != NULL);

      err = zvec_collection_create_hnsw_index(collection, "dense", hnsw_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Test 2: Create scalar index
      ZVecInvertIndexParams *invert_params =
          zvec_index_params_invert_create(true, false);
      TEST_ASSERT(invert_params != NULL);

      err = zvec_collection_create_invert_index(collection, "name",
                                                invert_params);
      TEST_ASSERT(err == ZVEC_OK);

      err = zvec_collection_drop_index(collection, "name");
      TEST_ASSERT(err == ZVEC_OK);

      // Test 3: Optimize collection
      err = zvec_collection_optimize(collection);
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
      zvec_index_params_hnsw_destroy(hnsw_params);
      zvec_index_params_invert_destroy(invert_params);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_ddl_operations(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_ddl";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  size_t field_count = zvec_collection_schema_get_field_count(schema);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test 1: Add new column
      ZVecFieldSchema *new_field =
          zvec_field_schema_create("new_int32", ZVEC_DATA_TYPE_INT32, true, 0);
      TEST_ASSERT(new_field != NULL);

      err = zvec_collection_add_column(collection, new_field, NULL);
      TEST_ASSERT(err == ZVEC_OK);

      // Test 2: Get collection schema and verify field count
      ZVecCollectionSchema *retrieved_schema = NULL;
      err = zvec_collection_get_schema(collection, &retrieved_schema);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(retrieved_schema != NULL);

      size_t new_field_count =
          zvec_collection_schema_get_field_count(retrieved_schema);
      TEST_ASSERT((field_count + 1) == new_field_count);

      // Test 3: Alter column
      ZVecFieldSchema *alter_field =
          zvec_field_schema_create("new_float", ZVEC_DATA_TYPE_FLOAT, true, 0);
      TEST_ASSERT(alter_field != NULL);

      err = zvec_collection_alter_column(collection, "new_int32", "",
                                         alter_field);
      TEST_ASSERT(err == ZVEC_OK);

      // Test 4: Drop column
      err = zvec_collection_drop_column(collection, "new_float");
      TEST_ASSERT(err == ZVEC_OK);

      // Test 5: Verify field count after drop
      err = zvec_collection_get_schema(collection, &retrieved_schema);
      TEST_ASSERT(err == ZVEC_OK);
      new_field_count =
          zvec_collection_schema_get_field_count(retrieved_schema);
      TEST_ASSERT(new_field_count == field_count);

      zvec_collection_schema_destroy(retrieved_schema);
      zvec_field_schema_destroy(new_field);
      zvec_field_schema_destroy(alter_field);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_field_ddl_operations(void) {
  TEST_START();

  // Test field schema creation with various configurations
  ZVecFieldSchema *field1 =
      zvec_field_schema_create("test_field1", ZVEC_DATA_TYPE_STRING, false, 0);
  TEST_ASSERT(field1 != NULL);
  TEST_ASSERT(strcmp(field1->name->data, "test_field1") == 0);
  TEST_ASSERT(field1->data_type == ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(field1->nullable == false);
  TEST_ASSERT(field1->dimension == 0);

  ZVecFieldSchema *field2 = zvec_field_schema_create(
      "test_field2", ZVEC_DATA_TYPE_VECTOR_FP32, true, 128);
  TEST_ASSERT(field2 != NULL);
  TEST_ASSERT(field2->data_type == ZVEC_DATA_TYPE_VECTOR_FP32);
  TEST_ASSERT(field2->nullable == true);
  TEST_ASSERT(field2->dimension == 128);

  // Test index parameter setting
  ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
  TEST_ASSERT(hnsw_params != NULL);

  ZVecErrorCode err = zvec_field_schema_set_index_params(
      field2, (ZVecIndexParams *)hnsw_params);
  TEST_ASSERT(err == ZVEC_OK);

  // Cleanup
  zvec_field_schema_destroy(field1);
  zvec_field_schema_destroy(field2);
  zvec_index_params_hnsw_destroy(hnsw_params);

  TEST_END();
}

void test_performance_benchmarks(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_performance";

  ZVecCollectionSchema *schema = zvec_collection_schema_create("perf_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Create simple schema for performance testing
    ZVecFieldSchema *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    ZVecFieldSchema *vec_field =
        zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
    ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
        ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
    zvec_field_schema_set_hnsw_index(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);

    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    TEST_ASSERT(collection != NULL);

    if (collection) {
      const size_t BATCH_SIZE = 1000;
      const size_t TOTAL_DOCS = 10000;

      // Test bulk insertion performance
#ifdef _POSIX_C_SOURCE
      struct timeval start_time, end_time;
      gettimeofday(&start_time, NULL);
#else
      clock_t start_clock = clock();
#endif

      for (size_t batch_start = 0; batch_start < TOTAL_DOCS;
           batch_start += BATCH_SIZE) {
        ZVecDoc *batch_docs[BATCH_SIZE];
        size_t current_batch_size = (batch_start + BATCH_SIZE > TOTAL_DOCS)
                                        ? TOTAL_DOCS - batch_start
                                        : BATCH_SIZE;

        // Create batch of documents
        for (size_t i = 0; i < current_batch_size; i++) {
          batch_docs[i] = zvec_doc_create();
          zvec_doc_set_pk(batch_docs[i], zvec_test_make_pk(batch_start + i));

          int64_t id = batch_start + i;
          zvec_doc_add_field_by_value(batch_docs[i], "id", ZVEC_DATA_TYPE_INT64,
                                      &id, sizeof(id));

          // Create random vector
          float vec[128];
          for (int j = 0; j < 128; j++) {
            vec[j] = (float)rand() / RAND_MAX;
          }
          zvec_doc_add_field_by_value(batch_docs[i], "vec",
                                      ZVEC_DATA_TYPE_VECTOR_FP32, vec,
                                      sizeof(vec));
        }

        // Insert batch
        size_t success_count, error_count;
        err = zvec_collection_insert(collection, (const ZVecDoc **)batch_docs,
                                     current_batch_size, &success_count,
                                     &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == current_batch_size);
        TEST_ASSERT(error_count == 0);

        // Cleanup batch documents
        for (size_t i = 0; i < current_batch_size; i++) {
          zvec_doc_destroy(batch_docs[i]);
        }
      }

#ifdef _POSIX_C_SOURCE
      gettimeofday(&end_time, NULL);
      double insert_time = (end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
#else
      clock_t end_clock = clock();
      double insert_time = ((double)(end_clock - start_clock)) / CLOCKS_PER_SEC;
#endif
      printf("  Inserted %zu documents in %.3f seconds (%.0f docs/sec)\n",
             TOTAL_DOCS, insert_time, TOTAL_DOCS / insert_time);

      // Flush and optimize
      zvec_collection_flush(collection);
      zvec_collection_optimize(collection);

      // Test query performance
      float query_vec[128];
      for (int i = 0; i < 128; i++) {
        query_vec[i] = (float)rand() / RAND_MAX;
      }

      ZVecVectorQuery query = {0};
      query.field_name = (ZVecString){.data = "vec", .length = 3};
      query.query_vector = (ZVecByteArray){.data = (uint8_t *)query_vec,
                                           .length = sizeof(query_vec)};
      query.topk = 10;
      query.include_vector = false;
      query.include_doc_id = true;

      const int QUERY_COUNT = 100;
#ifdef _POSIX_C_SOURCE
      struct timeval query_start_time, query_end_time;
      gettimeofday(&query_start_time, NULL);
#else
      clock_t query_start_clock = clock();
#endif

      for (int q = 0; q < QUERY_COUNT; q++) {
        ZVecDoc **results = NULL;
        size_t result_count = 0;

        err =
            zvec_collection_query(collection, &query, &results, &result_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(result_count <= 10);

        zvec_docs_free(results, result_count);
      }

#ifdef _POSIX_C_SOURCE
      gettimeofday(&query_end_time, NULL);
      double query_time =
          (query_end_time.tv_sec - query_start_time.tv_sec) +
          (query_end_time.tv_usec - query_start_time.tv_usec) / 1000000.0;
#else
      clock_t query_end_clock = clock();
      double query_time =
          ((double)(query_end_clock - query_start_clock)) / CLOCKS_PER_SEC;
#endif
      double avg_query_time =
          (query_time * 1000) / QUERY_COUNT;  // ms per query
      printf("  Average query time: %.2f ms\n", avg_query_time);

      zvec_collection_destroy(collection);
      zvec_index_params_hnsw_destroy(hnsw_params);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

// =============================================================================
// Additional tests for uncovered API functions
// =============================================================================

void test_zvec_shutdown(void) {
  TEST_START();

  // Test shutdown
  ZVecErrorCode err = zvec_shutdown();
  TEST_ASSERT(err == ZVEC_OK);

  // Re-initialize for other tests
  ZVecConfigData *config = zvec_config_data_create();
  TEST_ASSERT(config != NULL);
  err = zvec_initialize(config);
  TEST_ASSERT(err == ZVEC_OK);
  zvec_config_data_destroy(config);

  TEST_END();
}

void test_index_params_creation_functions(void) {
  TEST_START();

  // Test zvec_index_params_init_default
  ZVecIndexParams params;
  zvec_index_params_init_default(&params, ZVEC_INDEX_TYPE_HNSW,
                                 ZVEC_METRIC_TYPE_COSINE);
  TEST_ASSERT(params.index_type == ZVEC_INDEX_TYPE_HNSW);

  // Test zvec_index_params_vector_create
  ZVecVectorIndexParams *vector_params = zvec_index_params_vector_create(
      ZVEC_INDEX_TYPE_HNSW, ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_FP16);
  TEST_ASSERT(vector_params != NULL);
  TEST_ASSERT(vector_params->base.index_type == ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(vector_params->metric_type == ZVEC_METRIC_TYPE_L2);
  TEST_ASSERT(vector_params->quantize_type == ZVEC_QUANTIZE_TYPE_FP16);
  if (vector_params) {
    zvec_index_params_vector_destroy(vector_params);
  }

  // Test zvec_index_params_ivf_create
  ZVecIVFIndexParams *ivf_params = zvec_index_params_ivf_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_INT8, 100, 10, true, 5);
  TEST_ASSERT(ivf_params != NULL);
  TEST_ASSERT(ivf_params->base.base.index_type == ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(ivf_params->base.metric_type == ZVEC_METRIC_TYPE_L2);
  TEST_ASSERT(ivf_params->n_list == 100);
  TEST_ASSERT(ivf_params->n_iters == 10);
  TEST_ASSERT(ivf_params->use_soar == true);
  TEST_ASSERT(ivf_params->n_probe == 5);
  if (ivf_params) {
    zvec_index_params_ivf_destroy(ivf_params);
  }

  // Test zvec_index_params_vector_destroy
  ZVecVectorIndexParams *vector_params2 = zvec_index_params_vector_create(
      ZVEC_INDEX_TYPE_FLAT, ZVEC_METRIC_TYPE_IP, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(vector_params2 != NULL);
  zvec_index_params_vector_destroy(vector_params2);

  TEST_END();
}

void test_collection_advanced_index_functions(void) {
  TEST_START();

  const char *temp_dir = "/tmp/zvec_test_advanced_index";
  zvec_test_delete_dir(temp_dir);

  // Create schema
  ZVecCollectionSchema *schema =
      zvec_collection_schema_create("test_collection");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Add fields
    ZVecFieldSchema *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    ZVecFieldSchema *vec_field =
        zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
    zvec_collection_schema_add_field(schema, id_field);
    zvec_collection_schema_add_field(schema, vec_field);

    ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();
    options.max_doc_count_per_segment = 1000;
    ZVecCollection *collection = NULL;

    ZVecErrorCode err = zvec_collection_create_and_open(temp_dir, schema,
                                                        &options, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      // Test zvec_collection_create_flat_index
      ZVecFlatIndexParams *flat_params = zvec_index_params_flat_create(
          ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED);
      TEST_ASSERT(flat_params != NULL);
      err = zvec_collection_create_flat_index(collection, "vec", flat_params);
      TEST_ASSERT(err == ZVEC_OK);
      zvec_index_params_flat_destroy(flat_params);

      // Test zvec_collection_create_ivf_index
      ZVecIVFIndexParams *ivf_params = zvec_index_params_ivf_create(
          ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_INT8, 100, 10, true, 5);
      TEST_ASSERT(ivf_params != NULL);
      err = zvec_collection_drop_index(collection,
                                       "vec");  // Drop previous index first
      TEST_ASSERT(err == ZVEC_OK);
      err = zvec_collection_create_ivf_index(collection, "vec", ivf_params);
      TEST_ASSERT(err == ZVEC_OK);
      zvec_index_params_ivf_destroy(ivf_params);

      // Test zvec_collection_create_index_with_params
      ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
          ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_FP16, 16, 100, 50);
      TEST_ASSERT(hnsw_params != NULL);
      err = zvec_collection_drop_index(collection, "vec");
      TEST_ASSERT(err == ZVEC_OK);
      err = zvec_collection_create_index_with_params(collection, "vec",
                                                     hnsw_params);
      TEST_ASSERT(err == ZVEC_OK);
      zvec_index_params_hnsw_destroy(hnsw_params);

      // Test zvec_field_schema_set_ivf_index
      ZVecFieldSchema *new_vec_field = zvec_field_schema_create(
          "vec2", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
      TEST_ASSERT(new_vec_field != NULL);
      ZVecIVFIndexParams *ivf_params2 = zvec_index_params_ivf_create(
          ZVEC_METRIC_TYPE_IP, ZVEC_QUANTIZE_TYPE_UNDEFINED, 50, 5, false, 3);
      TEST_ASSERT(ivf_params2 != NULL);
      zvec_field_schema_set_ivf_index(new_vec_field, ivf_params2);
      TEST_ASSERT(new_vec_field->index_params != NULL);
      zvec_index_params_ivf_destroy(ivf_params2);
      zvec_field_schema_destroy(new_vec_field);

      zvec_collection_destroy(collection);
    }
    zvec_collection_schema_destroy(schema);
  }

  zvec_test_delete_dir(temp_dir);
  TEST_END();
}

void test_collection_query_functions(void) {
  TEST_START();

  const char *temp_dir = "/tmp/zvec_test_query_funcs";
  zvec_test_delete_dir(temp_dir);

  // Create schema and collection
  ZVecCollectionSchema *schema = zvec_collection_schema_create("query_test");
  ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);

  ZVecFieldSchema *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  ZVecFieldSchema *vec_field =
      zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  zvec_field_schema_set_hnsw_index(vec_field, hnsw_params);

  zvec_collection_schema_add_field(schema, name_field);
  zvec_collection_schema_add_field(schema, vec_field);

  ZVecCollection *collection = NULL;
  ZVecErrorCode err =
      zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
  TEST_ASSERT(err == ZVEC_OK);

  if (collection) {
    // Insert test documents
    ZVecDoc *doc1 = zvec_doc_create();
    zvec_doc_set_pk(doc1, "doc1");
    float vec1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    zvec_doc_add_field_by_value(doc1, "vec", ZVEC_DATA_TYPE_VECTOR_FP32, vec1,
                                sizeof(vec1));
    zvec_doc_add_field_by_value(doc1, "name", ZVEC_DATA_TYPE_STRING,
                                "document1", 9);

    ZVecDoc *doc2 = zvec_doc_create();
    zvec_doc_set_pk(doc2, "doc2");
    float vec2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    zvec_doc_add_field_by_value(doc2, "vec", ZVEC_DATA_TYPE_VECTOR_FP32, vec2,
                                sizeof(vec2));
    zvec_doc_add_field_by_value(doc2, "name", ZVEC_DATA_TYPE_STRING,
                                "document2", 9);

    ZVecDoc *docs[] = {doc1, doc2};
    size_t success_count, error_count;
    err = zvec_collection_insert(collection, (const ZVecDoc **)docs, 2,
                                 &success_count, &error_count);
    TEST_ASSERT(err == ZVEC_OK);

    zvec_collection_flush(collection);
    zvec_collection_optimize(collection);

    // Test zvec_collection_fetch
    const char *pks[] = {"doc1", "doc2"};
    ZVecDoc **results = NULL;
    size_t found_count = 0;
    err = zvec_collection_fetch(collection, pks, 2, &results, &found_count);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(found_count == 2);
    zvec_docs_free(results, found_count);

    // Test zvec_collection_query_by_group
    ZVecGroupByVectorQuery group_query = {0};
    group_query.field_name = ZVEC_STRING("vec");
    float query_vec[4] = {0.5f, 0.5f, 0.0f, 0.0f};
    group_query.query_vector.data = (uint8_t *)query_vec;
    group_query.query_vector.length = sizeof(query_vec);
    group_query.group_by_field_name = ZVEC_STRING("name");
    group_query.group_count = 2;
    group_query.group_topk = 1;
    group_query.include_vector = false;

    ZVecStringArray output_fields = {0};
    output_fields.count = 1;
    output_fields.strings =
        (ZVecString *)malloc(sizeof(ZVecString) * output_fields.count);
    output_fields.strings[0] = ZVEC_STRING("name");
    group_query.output_fields = output_fields;

    ZVecDoc **group_results = NULL;
    ZVecString **group_values = NULL;
    size_t group_result_count = 0;
    err =
        zvec_collection_query_by_group(collection, &group_query, &group_results,
                                       &group_values, &group_result_count);
    TEST_ASSERT(err == ZVEC_OK);
    if (group_results) {
      zvec_docs_free(group_results, group_result_count);
    }
    if (group_values) {
      for (size_t i = 0; i < group_result_count; i++) {
        zvec_free_string(group_values[i]);
      }
      free(group_values);
    }

    free(output_fields.strings);

    // Test zvec_collection_get_options
    ZVecCollectionOptions *options = NULL;
    err = zvec_collection_get_options(collection, &options);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(options != NULL);
    free(options);

    zvec_collection_destroy(collection);
    zvec_doc_destroy(doc1);
    zvec_doc_destroy(doc2);
  }

  zvec_index_params_hnsw_destroy(hnsw_params);
  zvec_collection_schema_destroy(schema);
  zvec_test_delete_dir(temp_dir);

  TEST_END();
}

void test_doc_advanced_functions(void) {
  TEST_START();

  // Test zvec_doc_clear
  ZVecDoc *doc = zvec_doc_create();
  zvec_doc_set_pk(doc, "test_pk");
  zvec_doc_add_field_by_value(doc, "field1", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){100}, sizeof(int32_t));
  TEST_ASSERT(zvec_doc_get_field_count(doc) > 0);
  zvec_doc_clear(doc);
  TEST_ASSERT(zvec_doc_get_field_count(doc) == 0);

  // Test zvec_doc_get_pk_copy
  zvec_doc_set_pk(doc, "test_pk_copy");
  const char *pk_copy = zvec_doc_get_pk_copy(doc);
  TEST_ASSERT(pk_copy != NULL);
  TEST_ASSERT(strcmp(pk_copy, "test_pk_copy") == 0);
  free((void *)pk_copy);

  // Test zvec_doc_is_empty
  ZVecDoc *empty_doc = zvec_doc_create();
  TEST_ASSERT(zvec_doc_is_empty(empty_doc) == true);
  zvec_doc_add_field_by_value(empty_doc, "test", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){1}, sizeof(int32_t));
  TEST_ASSERT(zvec_doc_is_empty(empty_doc) == false);
  zvec_doc_destroy(empty_doc);

  // Test zvec_doc_memory_usage
  ZVecDoc *mem_doc = zvec_doc_create();
  zvec_doc_set_pk(mem_doc, "memory_test");
  char large_data[1024];
  memset(large_data, 'A', sizeof(large_data));
  zvec_doc_add_field_by_value(mem_doc, "large_field", ZVEC_DATA_TYPE_STRING,
                              large_data, sizeof(large_data));
  size_t mem_usage = zvec_doc_memory_usage(mem_doc);
  TEST_ASSERT(mem_usage > 0);
  zvec_doc_destroy(mem_doc);

  // Test zvec_doc_merge
  ZVecDoc *doc1 = zvec_doc_create();
  zvec_doc_set_pk(doc1, "merge_test");
  zvec_doc_add_field_by_value(doc1, "field1", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){100}, sizeof(int32_t));

  ZVecDoc *doc2 = zvec_doc_create();
  zvec_doc_add_field_by_value(doc2, "field2", ZVEC_DATA_TYPE_STRING, "merged",
                              6);

  zvec_doc_merge(doc1, doc2);
  TEST_ASSERT(zvec_doc_has_field(doc1, "field1") == true);
  TEST_ASSERT(zvec_doc_has_field(doc1, "field2") == true);

  zvec_doc_destroy(doc1);
  zvec_doc_destroy(doc2);

  // Test zvec_doc_validate
  ZVecCollectionSchema *schema = zvec_collection_schema_create("validate_test");
  ZVecFieldSchema *val_field =
      zvec_field_schema_create("test_field", ZVEC_DATA_TYPE_INT32, false, 0);
  zvec_collection_schema_add_field(schema, val_field);

  ZVecDoc *val_doc = zvec_doc_create();
  zvec_doc_set_pk(val_doc, "test_pk");
  zvec_doc_add_field_by_value(val_doc, "test_field", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){42}, sizeof(int32_t));

  char *error_msg = NULL;
  ZVecErrorCode err = zvec_doc_validate(val_doc, schema, false, &error_msg);
  TEST_ASSERT(err == ZVEC_OK);
  if (error_msg) {
    free(error_msg);
  }

  zvec_doc_destroy(val_doc);
  zvec_collection_schema_destroy(schema);
  zvec_doc_destroy(doc);

  // Test zvec_doc_to_detail_string
  ZVecDoc *detail_doc = zvec_doc_create();
  zvec_doc_set_pk(detail_doc, "detail_test");
  zvec_doc_add_field_by_value(detail_doc, "int_field", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){12345}, sizeof(int32_t));
  zvec_doc_add_field_by_value(detail_doc, "str_field", ZVEC_DATA_TYPE_STRING,
                              "hello", 5);

  char *detail_str = NULL;
  err = zvec_doc_to_detail_string(detail_doc, &detail_str);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(detail_str != NULL);
  printf("  Document detail: %s\n", detail_str);
  free(detail_str);

  zvec_doc_destroy(detail_doc);

  TEST_END();
}

void test_array_memory_functions(void) {
  TEST_START();

  // Test ZVecStringArray
  ZVecStringArray *str_array = zvec_string_array_create(3);
  TEST_ASSERT(str_array != NULL);
  if (str_array) {
    TEST_ASSERT(str_array->count == 3);
    TEST_ASSERT(str_array->strings != NULL);

    // Add strings at specific indices
    zvec_string_array_add(str_array, 0, "string1");
    zvec_string_array_add(str_array, 1, "string2");
    zvec_string_array_add(str_array, 2, "string3");

    // Verify strings were added
    TEST_ASSERT(strcmp(str_array->strings[0].data, "string1") == 0);
    TEST_ASSERT(strcmp(str_array->strings[1].data, "string2") == 0);
    TEST_ASSERT(strcmp(str_array->strings[2].data, "string3") == 0);
    zvec_string_array_destroy(str_array);
  }

  // Test ZVecMutableByteArray
  ZVecMutableByteArray *byte_array = zvec_byte_array_create(1024);
  TEST_ASSERT(byte_array != NULL);
  if (byte_array) {
    TEST_ASSERT(byte_array->capacity == 1024);
    TEST_ASSERT(byte_array->length == 0);
    TEST_ASSERT(byte_array->data != NULL);

    // Write some data
    byte_array->data[0] = 0x01;
    byte_array->data[1] = 0x02;
    byte_array->data[2] = 0x03;
    byte_array->length = 3;

    TEST_ASSERT(byte_array->length == 3);
    TEST_ASSERT(byte_array->data[0] == 0x01);
    TEST_ASSERT(byte_array->data[1] == 0x02);
    TEST_ASSERT(byte_array->data[2] == 0x03);

    zvec_byte_array_destroy(byte_array);
  }

  // Test ZVecFloatArray
  ZVecFloatArray *float_array = zvec_float_array_create(10);
  TEST_ASSERT(float_array != NULL);
  if (float_array) {
    TEST_ASSERT(float_array->length == 10);
    TEST_ASSERT(float_array->data != NULL);

    // Note: Data is initialized to 0 by zvec_float_array_create
    // The const qualifier indicates this is typically used for read-only access
    // For testing, we verify the allocation succeeded and length is correct
    TEST_ASSERT(float_array->data[0] == 0.0f);
    TEST_ASSERT(float_array->data[9] == 0.0f);

    zvec_float_array_destroy(float_array);
  }

  // Test ZVecInt64Array
  ZVecInt64Array *int64_array = zvec_int64_array_create(5);
  TEST_ASSERT(int64_array != NULL);
  if (int64_array) {
    TEST_ASSERT(int64_array->length == 5);
    TEST_ASSERT(int64_array->data != NULL);

    // Note: Data is initialized to 0 by zvec_int64_array_create
    // The const qualifier indicates this is typically used for read-only access
    TEST_ASSERT(int64_array->data[0] == 0);
    TEST_ASSERT(int64_array->data[4] == 0);

    zvec_int64_array_destroy(int64_array);
  }

  // Test edge case: create with zero size
  ZVecMutableByteArray *zero_array = zvec_byte_array_create(0);
  TEST_ASSERT(zero_array != NULL);
  if (zero_array) {
    zvec_byte_array_destroy(zero_array);
  }

  TEST_END();
}

void test_index_params_destruction(void) {
  TEST_START();

  // Test zvec_index_params_invert_destroy
  ZVecInvertIndexParams *invert_params =
      zvec_index_params_invert_create(true, false);
  TEST_ASSERT(invert_params != NULL);
  zvec_index_params_invert_destroy(invert_params);

  TEST_END();
}

// =============================================================================
// Main function
// =============================================================================

int main(void) {
  printf("Starting comprehensive C API tests...\n\n");

  // Clean up previous test directories
  printf("Cleaning up previous test directories...\n");
  system("rm -rf /tmp/zvec_test_*");
  printf("Cleanup completed.\n\n");

  test_version_functions();
  test_error_handling_functions();
  test_zvec_config();
  test_zvec_initialize();
  test_zvec_string_functions();

  // Schema-related tests
  test_schema_basic_operations();
  test_schema_edge_cases();
  test_schema_field_operations();
  test_normal_schema_creation();
  test_schema_with_indexes();
  test_schema_max_doc_count();

  // Field-related tests
  test_field_schema_functions();
  test_field_helper_functions();
  test_field_ddl_operations();

  // Collection-related tests
  test_collection_basic_operations();
  test_collection_edge_cases();
  test_collection_delete_by_filter();
  test_collection_stats();
  test_collection_stats_functions();
  test_collection_dml_functions();
  test_collection_ddl_operations();

  // Doc-related tests
  test_doc_creation();
  test_doc_primary_key();
  test_doc_basic_operations();
  test_doc_get_field_value_basic();
  test_doc_get_field_value_copy();
  test_doc_get_field_value_pointer();
  test_doc_field_operations();
  test_doc_error_conditions();
  test_doc_serialization();
  test_doc_add_field_by_value();
  test_doc_add_field_by_struct();

  // Index tests
  test_index_params();
  test_index_params_functions();
  test_index_creation_and_management();

  // Query tests
  test_query_params_functions();
  test_actual_vector_queries();

  // Performance tests
  // test_performance_benchmarks();

  // Utility function tests
  test_utility_functions();

  // Memory management tests
  test_memory_management_functions();

  // Additional API coverage tests
  test_zvec_shutdown();
  test_index_params_creation_functions();
  test_collection_advanced_index_functions();
  test_collection_query_functions();
  test_doc_advanced_functions();
  test_array_memory_functions();
  test_index_params_destruction();

  printf("\n=== Comprehensive Test Summary ===\n");
  printf("Total tests: %d\n", test_count);
  printf("Passed: %d\n", passed_count);
  printf("Failed: %d\n", test_count - passed_count);

  return test_count == passed_count ? 0 : 1;
}
