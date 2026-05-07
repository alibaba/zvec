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

// Platform-specific headers
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <glob.h>
#include <unistd.h>
#endif

#include "utils.h"

// =============================================================================
// Test helper macro definitions
// =============================================================================

// Helper function to get field count (replaces GCC-specific statement
// expression)
static size_t get_field_count(zvec_collection_schema_t *schema) {
  const char **names = NULL;
  size_t count = 0;
  zvec_error_code_t err =
      zvec_collection_schema_get_all_field_names(schema, &names, &count);
  if (err == ZVEC_OK && names) {
    size_t i;
    for (i = 0; i < count; i++) {
      zvec_free((char *)names[i]);
    }
    zvec_free(names);
  }
  return count;
}

// Cross-platform helper function to clean up temporary directories
static void cleanup_temp_directory(const char *dir) {
  zvec_test_delete_dir(dir);
}

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
  printf("  Version string: %s\n", version);

  // Test version component retrieval
  int major = zvec_get_version_major();
  int minor = zvec_get_version_minor();
  int patch = zvec_get_version_patch();

  printf("  Version components: %d.%d.%d\n", major, minor, patch);
  TEST_ASSERT(major >= 0);
  TEST_ASSERT(minor >= 0);
  TEST_ASSERT(patch >= 0);

  // Test version compatibility check with current version (should pass)
  TEST_ASSERT(zvec_check_version(major, minor, patch));

  // Test with older version (should pass - current is newer)
  if (minor > 0) {
    TEST_ASSERT(zvec_check_version(major, minor - 1, patch));
  }
  if (major > 0) {
    TEST_ASSERT(zvec_check_version(major - 1, minor, patch));
  }

  // Test with much newer version (should fail - current is older)
  bool not_compatible = zvec_check_version(99, 99, 99);
  TEST_ASSERT(not_compatible == false);

  // Test with invalid negative versions (should fail and set error)
  TEST_ASSERT(zvec_check_version(-1, 0, 0) == false);
  TEST_ASSERT(zvec_check_version(0, -1, 0) == false);
  TEST_ASSERT(zvec_check_version(0, 0, -1) == false);

  TEST_END();
}

void test_error_handling_functions(void) {
  TEST_START();

  char *error_msg = NULL;
  zvec_error_code_t err = zvec_get_last_error(&error_msg);
  TEST_ASSERT(err == ZVEC_OK);

  if (error_msg) {
    zvec_free(error_msg);
  }

  // Test error clearing
  zvec_clear_error();

  // Test error details retrieval
  zvec_error_details_t error_details = {0};
  err = zvec_get_last_error_details(&error_details);
  TEST_ASSERT(err == ZVEC_OK);

  TEST_END();
}

void test_zvec_config() {
  TEST_START();

  // Test 1: Console log config creation and destruction
  zvec_log_config_t *console_config =
      zvec_config_log_create_console(ZVEC_LOG_LEVEL_INFO);
  TEST_ASSERT(console_config != NULL);
  if (console_config) {
    TEST_ASSERT(zvec_config_log_get_level(console_config) ==
                ZVEC_LOG_LEVEL_INFO);
    zvec_config_log_destroy(console_config);
  }

  // Test 2: File log config creation and destruction
  zvec_log_config_t *file_config = zvec_config_log_create_file(
      ZVEC_LOG_LEVEL_WARN, "./logs", "test_log", 100, 7);
  TEST_ASSERT(file_config != NULL);
  if (file_config) {
    TEST_ASSERT(zvec_config_log_get_level(file_config) == ZVEC_LOG_LEVEL_WARN);
    TEST_ASSERT(strcmp(zvec_config_log_get_dir(file_config), "./logs") == 0);
    TEST_ASSERT(strcmp(zvec_config_log_get_basename(file_config), "test_log") ==
                0);
    TEST_ASSERT(zvec_config_log_get_file_size(file_config) == 100);
    TEST_ASSERT(zvec_config_log_get_overdue_days(file_config) == 7);
    zvec_config_log_destroy(file_config);
  }

  // Test 3: File log config edge cases
  zvec_log_config_t *empty_file_config =
      zvec_config_log_create_file(ZVEC_LOG_LEVEL_INFO, "", "", 0, 0);
  TEST_ASSERT(empty_file_config != NULL);
  if (empty_file_config) {
    TEST_ASSERT(zvec_config_log_get_level(empty_file_config) ==
                ZVEC_LOG_LEVEL_INFO);
    TEST_ASSERT(strcmp(zvec_config_log_get_dir(empty_file_config), "") == 0);
    TEST_ASSERT(strcmp(zvec_config_log_get_basename(empty_file_config), "") ==
                0);
    TEST_ASSERT(zvec_config_log_get_file_size(empty_file_config) == 0);
    TEST_ASSERT(zvec_config_log_get_overdue_days(empty_file_config) == 0);
    zvec_config_log_destroy(empty_file_config);
  }

  // Test 4: Log config creation with console type
  zvec_log_config_t *temp_console =
      zvec_config_log_create_console(ZVEC_LOG_LEVEL_ERROR);
  TEST_ASSERT(temp_console != NULL);
  if (temp_console) {
    zvec_config_log_destroy(temp_console);
  }

  // Test 5: Log config creation with file type
  zvec_log_config_t *temp_file = zvec_config_log_create_file(
      ZVEC_LOG_LEVEL_DEBUG, "./logs", "app", 50, 30);
  TEST_ASSERT(temp_file != NULL);
  TEST_ASSERT(zvec_config_log_get_level(temp_file) == ZVEC_LOG_LEVEL_DEBUG);
  TEST_ASSERT(strcmp(zvec_config_log_get_dir(temp_file), "./logs") == 0);
  TEST_ASSERT(strcmp(zvec_config_log_get_basename(temp_file), "app") == 0);
  TEST_ASSERT(zvec_config_log_get_file_size(temp_file) == 50);
  TEST_ASSERT(zvec_config_log_get_overdue_days(temp_file) == 30);

  zvec_config_log_destroy(temp_file);

  // Test 6: Config data creation and basic operations
  zvec_config_data_t *config_data = zvec_config_data_create();
  TEST_ASSERT(config_data != NULL);
  if (config_data) {
    // Test initial values
    TEST_ASSERT(zvec_config_data_get_log_type(config_data) ==
                ZVEC_LOG_TYPE_CONSOLE);

    // Test memory limit setting
    zvec_error_code_t err =
        zvec_config_data_set_memory_limit(config_data, 1024 * 1024 * 1024);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_memory_limit(config_data) ==
                1024 * 1024 * 1024);

    // Test thread count settings
    err = zvec_config_data_set_query_thread_count(config_data, 8);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_query_thread_count(config_data) == 8);

    err = zvec_config_data_set_optimize_thread_count(config_data, 4);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_optimize_thread_count(config_data) == 4);

    // Test log config replacement
    TEST_ASSERT(zvec_config_data_get_log_type(config_data) ==
                ZVEC_LOG_TYPE_CONSOLE);

    zvec_log_config_t *new_file = zvec_config_log_create_file(
        ZVEC_LOG_LEVEL_DEBUG, "./logs", "app", 50, 30);
    TEST_ASSERT(new_file != NULL);
    zvec_config_data_set_log_config(config_data, new_file);
    TEST_ASSERT(zvec_config_data_get_log_type(config_data) ==
                ZVEC_LOG_TYPE_FILE);

    zvec_config_data_destroy(config_data);
  }

  // Test 7: Edge cases and error conditions
  // Test NULL pointer handling
  zvec_error_code_t err = zvec_config_data_set_memory_limit(NULL, 1024);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_log_config(NULL, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_query_thread_count(NULL, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_optimize_thread_count(NULL, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test boundary values
  zvec_config_data_t *boundary_config = zvec_config_data_create();
  if (boundary_config) {
    // Test zero values
    err = zvec_config_data_set_memory_limit(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_memory_limit(boundary_config) == 0);

    // Test maximum values
    err = zvec_config_data_set_memory_limit(boundary_config, UINT64_MAX);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_memory_limit(boundary_config) ==
                UINT64_MAX);

    // Test zero thread counts
    err = zvec_config_data_set_query_thread_count(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_query_thread_count(boundary_config) == 0);

    err = zvec_config_data_set_optimize_thread_count(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_config_data_get_optimize_thread_count(boundary_config) ==
                0);

    zvec_config_data_destroy(boundary_config);
  }

  // Test 8: Memory leak prevention - double destroy safety
  zvec_config_data_t *double_destroy_test = zvec_config_data_create();
  if (double_destroy_test) {
    zvec_config_data_destroy(double_destroy_test);
  }

  TEST_END();
}

void test_zvec_initialize() {
  TEST_START();

  zvec_config_data_t *config = zvec_config_data_create();
  TEST_ASSERT(config != NULL);
  if (config) {
    TEST_ASSERT(zvec_config_data_get_log_type(config) == ZVEC_LOG_TYPE_CONSOLE);
  }
  zvec_error_code_t err = zvec_initialize(config);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(zvec_is_initialized());

  TEST_END();
}

// =============================================================================
// Schema-related tests
// =============================================================================

void test_schema_basic_operations(void) {
  TEST_START();

  // Test 1: Basic Schema creation and destruction
  zvec_collection_schema_t *schema = zvec_collection_schema_create("demo");
  TEST_ASSERT(schema != NULL);
  TEST_ASSERT(zvec_collection_schema_get_name(schema) != NULL);
  TEST_ASSERT(strcmp(zvec_collection_schema_get_name(schema), "demo") == 0);
  TEST_ASSERT(get_field_count(schema) == 0);
  TEST_ASSERT(zvec_collection_schema_get_max_doc_count_per_segment(schema) > 0);

  // Test 2: Schema field count operations
  size_t initial_count = get_field_count(schema);
  TEST_ASSERT(initial_count == 0);

  // Test 3: Adding fields to schema
  zvec_field_schema_t *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
  zvec_error_code_t err = zvec_collection_schema_add_field(schema, id_field);
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_add = get_field_count(schema);
  TEST_ASSERT(count_after_add == 1);

  // Test 4: Finding fields in schema
  const zvec_field_schema_t *found_field =
      zvec_collection_schema_get_field(schema, "id");
  TEST_ASSERT(found_field != NULL);
  TEST_ASSERT(strcmp(zvec_field_schema_get_name(found_field), "id") == 0);
  TEST_ASSERT(zvec_field_schema_get_data_type(found_field) ==
              ZVEC_DATA_TYPE_INT64);

  // Test 5: Getting field by index (using get_all_field_names)
  const char **field_names = NULL;
  size_t field_count = 0;
  err = zvec_collection_schema_get_all_field_names(schema, &field_names,
                                                   &field_count);
  TEST_ASSERT(err == ZVEC_OK && field_count > 0);
  const zvec_field_schema_t *indexed_field = NULL;
  if (field_count > 0) {
    indexed_field = zvec_collection_schema_get_field(schema, field_names[0]);
  }
  TEST_ASSERT(indexed_field != NULL);
  TEST_ASSERT(strcmp(zvec_field_schema_get_name(indexed_field), "id") == 0);
  // Clean up field names
  for (size_t i = 0; i < field_count; i++) zvec_free((char *)field_names[i]);
  zvec_free(field_names);

  // Test 6: Adding multiple fields (use individual add_field calls)
  zvec_field_schema_t *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_field_schema_t *age_field =
      zvec_field_schema_create("age", ZVEC_DATA_TYPE_INT32, true, 0);

  err = zvec_collection_schema_add_field(schema, name_field);
  TEST_ASSERT(err == ZVEC_OK);
  err = zvec_collection_schema_add_field(schema, age_field);
  TEST_ASSERT(err == ZVEC_OK);

  // Add a vector field (required for validation)
  zvec_field_schema_t *vec_field = zvec_field_schema_create(
      "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  err = zvec_collection_schema_add_field(schema, vec_field);
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_multi_add = get_field_count(schema);
  TEST_ASSERT(count_after_multi_add == 4);  // id, name, age, embedding

  // Test 7: Finding newly added fields
  const zvec_field_schema_t *name_found =
      zvec_collection_schema_get_field(schema, "name");
  TEST_ASSERT(name_found != NULL);
  TEST_ASSERT(strcmp(zvec_field_schema_get_name(name_found), "name") == 0);

  const zvec_field_schema_t *age_found =
      zvec_collection_schema_get_field(schema, "age");
  TEST_ASSERT(age_found != NULL);
  TEST_ASSERT(strcmp(zvec_field_schema_get_name(age_found), "age") == 0);

  // Clean up fields we created
  zvec_field_schema_destroy(name_field);
  zvec_field_schema_destroy(age_field);
  zvec_field_schema_destroy(vec_field);

  // Test 8: Setting and getting max doc count
  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 10000);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 10000);

  // Test 9: Schema validation
  zvec_string_t *validation_error = NULL;
  err = zvec_collection_schema_validate(schema, &validation_error);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(validation_error == NULL);

  // Test 10: Removing single field
  err = zvec_collection_schema_drop_field(schema, "age");
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_remove = get_field_count(schema);
  TEST_ASSERT(count_after_remove == 3);  // id, name, embedding

  const zvec_field_schema_t *removed_field =
      zvec_collection_schema_get_field(schema, "age");
  TEST_ASSERT(removed_field == NULL);

  // Test 11: Removing multiple fields (keep embedding for validation)
  // remove_fields has been removed, use drop_field in a loop instead
  err = zvec_collection_schema_drop_field(schema, "name");
  TEST_ASSERT(err == ZVEC_OK);
  err = zvec_collection_schema_drop_field(schema, "id");
  TEST_ASSERT(err == ZVEC_OK);

  size_t final_count = get_field_count(schema);
  TEST_ASSERT(final_count == 1);  // Only embedding remains

  // Test 12: Schema cleanup
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

void test_schema_edge_cases(void) {
  TEST_START();

  // Test 1: NULL parameter handling for schema creation
  zvec_collection_schema_t *null_schema = zvec_collection_schema_create(NULL);
  TEST_ASSERT(null_schema == NULL);

  // Test 2: Empty string schema name
  zvec_collection_schema_t *empty_schema = zvec_collection_schema_create("");
  TEST_ASSERT(empty_schema != NULL);
  TEST_ASSERT(zvec_collection_schema_get_name(empty_schema) != NULL);
  TEST_ASSERT(strcmp(zvec_collection_schema_get_name(empty_schema), "") == 0);
  zvec_collection_schema_destroy(empty_schema);

  // Test 3: Very long schema name
  char long_name[1024];
  memset(long_name, 'a', 1023);
  long_name[1023] = '\0';
  zvec_collection_schema_t *long_schema =
      zvec_collection_schema_create(long_name);
  TEST_ASSERT(long_schema != NULL);
  TEST_ASSERT(zvec_collection_schema_get_name(long_schema) != NULL);
  TEST_ASSERT(strlen(zvec_collection_schema_get_name(long_schema)) == 1023);
  zvec_collection_schema_destroy(long_schema);

  // Test 4: NULL schema parameter handling for all functions
  zvec_error_code_t err;
  const char **test_names = NULL;
  size_t test_count = 0;
  err = zvec_collection_schema_get_all_field_names(NULL, &test_names,
                                                   &test_count);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  TEST_ASSERT(test_count == 0);

  const zvec_field_schema_t *null_field =
      zvec_collection_schema_get_field(NULL, "test");
  TEST_ASSERT(null_field == NULL);

  zvec_field_schema_t *null_indexed_field =
      zvec_collection_schema_get_field(NULL, "test");
  TEST_ASSERT(null_indexed_field == NULL);

  uint64_t null_max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(NULL);
  TEST_ASSERT(null_max_doc_count == 0);

  err = zvec_collection_schema_set_max_doc_count_per_segment(NULL, 1000);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  zvec_string_t *null_validation_error = NULL;
  err = zvec_collection_schema_validate(NULL, &null_validation_error);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  TEST_ASSERT(null_validation_error == NULL);

  err = zvec_collection_schema_add_field(NULL, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_schema_drop_field(NULL, "test");
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  const char *null_field_names[] = {NULL};
  // remove_fields has been removed, use drop_field in a loop instead
  for (int i = 0; i < 1; i++) {
    if (null_field_names[i]) {
      err = zvec_collection_schema_drop_field(NULL, null_field_names[i]);
      TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
    }
  }

  // Test 5: Working with valid schema for edge cases
  zvec_collection_schema_t *schema = zvec_collection_schema_create("edge_test");
  TEST_ASSERT(schema != NULL);

  // Test 6: Adding NULL field to schema
  err = zvec_collection_schema_add_field(schema, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 7: Adding field with NULL name
  zvec_field_schema_t *null_name_field_schema =
      zvec_field_schema_create(NULL, ZVEC_DATA_TYPE_INT32, false, 0);
  err = zvec_collection_schema_add_field(schema, null_name_field_schema);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  zvec_field_schema_destroy(null_name_field_schema);

  // Test 8: Finding field with NULL name
  const zvec_field_schema_t *null_name_field =
      zvec_collection_schema_get_field(schema, NULL);
  TEST_ASSERT(null_name_field == NULL);

  // Test 9: Finding non-existent field
  const zvec_field_schema_t *nonexistent_field =
      zvec_collection_schema_get_field(schema, "nonexistent");
  TEST_ASSERT(nonexistent_field == NULL);

  // Test 10: Getting field from empty schema
  const char **field_names = NULL;
  size_t field_count = 0;
  err = zvec_collection_schema_get_all_field_names(schema, &field_names,
                                                   &field_count);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(field_count == 0);
  if (field_names) zvec_free(field_names);

  // Test 11: Removing field with NULL name
  err = zvec_collection_schema_drop_field(schema, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 12: Removing non-existent field
  err = zvec_collection_schema_drop_field(schema, "nonexistent");
  TEST_ASSERT(err == ZVEC_ERROR_NOT_FOUND);

  // Test 13: Removing fields with NULL array - use drop_field in a loop
  const char *null_test_names[] = {"field1", "field2", "field3", "field4",
                                   "field5"};
  for (int i = 0; i < 5; i++) {
    err = zvec_collection_schema_drop_field(schema, null_test_names[i]);
    // Expected to fail with NOT_FOUND since fields don't exist
    TEST_ASSERT(err == ZVEC_ERROR_NOT_FOUND ||
                err == ZVEC_ERROR_INVALID_ARGUMENT);
  }

  // Test 16: Removing zero fields - nothing to test since remove_fields is
  // removed Just skip this test as it was specific to remove_fields function

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
  zvec_string_t *empty_validation_error = NULL;
  err = zvec_collection_schema_validate(schema, &empty_validation_error);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 20: Add duplicate field names
  zvec_field_schema_t *first_id =
      zvec_field_schema_create("duplicate_id", ZVEC_DATA_TYPE_INT64, false, 0);
  zvec_field_schema_t *second_id =
      zvec_field_schema_create("duplicate_id", ZVEC_DATA_TYPE_STRING, false, 0);

  err = zvec_collection_schema_add_field(schema, first_id);
  TEST_ASSERT(err == ZVEC_OK);

  err = zvec_collection_schema_add_field(schema, second_id);
  TEST_ASSERT(err == ZVEC_ERROR_ALREADY_EXISTS);
  zvec_field_schema_destroy(second_id);

  // Verify fields
  size_t verify_field_count = get_field_count(schema);
  TEST_ASSERT(verify_field_count == 1);

  // Test 21: Cleanup
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

void test_schema_field_operations(void) {
  TEST_START();

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Test field count
    size_t initial_count = get_field_count(schema);
    TEST_ASSERT(initial_count == 5);

    // Test finding non-existent field
    const zvec_field_schema_t *nonexistent =
        zvec_collection_schema_get_field(schema, "nonexistent");
    TEST_ASSERT(nonexistent == NULL);

    // Test finding existing field
    const zvec_field_schema_t *id_field =
        zvec_collection_schema_get_field(schema, "id");
    TEST_ASSERT(id_field != NULL);
    if (id_field) {
      TEST_ASSERT(strcmp(zvec_field_schema_get_name(id_field), "id") == 0);
      TEST_ASSERT(zvec_field_schema_get_data_type(id_field) ==
                  ZVEC_DATA_TYPE_INT64);
    }

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_normal_schema_creation(void) {
  TEST_START();

  zvec_collection_schema_t *schema =
      zvec_test_create_normal_schema(false, "test_normal", NULL, NULL, 1000);
  TEST_ASSERT(schema != NULL);

  if (schema) {
    TEST_ASSERT(
        strcmp(zvec_collection_schema_get_name(schema), "test_normal") == 0);

    // Verify field count
    size_t field_count = get_field_count(schema);
    TEST_ASSERT(field_count > 0);

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_schema_with_indexes(void) {
  TEST_START();

  // Test Schema with scalar index
  zvec_collection_schema_t *scalar_index_schema =
      zvec_test_create_schema_with_scalar_index(true, true,
                                                "scalar_index_test");
  TEST_ASSERT(scalar_index_schema != NULL);
  if (scalar_index_schema) {
    zvec_collection_schema_destroy(scalar_index_schema);
  }

  // Test Schema with vector index
  zvec_collection_schema_t *vector_index_schema =
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
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  zvec_error_code_t err =
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

void test_collection_schema_helpers(void) {
  TEST_START();

  // Create schema with various field types
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("helper_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Add scalar fields
    zvec_field_schema_t *int_field =
        zvec_field_schema_create("int_field", ZVEC_DATA_TYPE_INT32, false, 0);
    zvec_field_schema_t *str_field =
        zvec_field_schema_create("str_field", ZVEC_DATA_TYPE_STRING, true, 0);

    // Add vector field
    zvec_field_schema_t *vec_field = zvec_field_schema_create(
        "vec_field", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);

    zvec_collection_schema_add_field(schema, int_field);
    zvec_collection_schema_add_field(schema, str_field);
    zvec_collection_schema_add_field(schema, vec_field);

    // Test has_field
    TEST_ASSERT(zvec_collection_schema_has_field(schema, "int_field") == true);
    TEST_ASSERT(zvec_collection_schema_has_field(schema, "str_field") == true);
    TEST_ASSERT(zvec_collection_schema_has_field(schema, "vec_field") == true);
    TEST_ASSERT(zvec_collection_schema_has_field(schema, "nonexistent") ==
                false);

    // Test get_forward_field (scalar field)
    zvec_field_schema_t *found_int =
        zvec_collection_schema_get_forward_field(schema, "int_field");
    TEST_ASSERT(found_int != NULL);
    TEST_ASSERT(zvec_field_schema_get_data_type(found_int) ==
                ZVEC_DATA_TYPE_INT32);

    // get_forward_field should return NULL for vector field
    zvec_field_schema_t *vec_as_forward =
        zvec_collection_schema_get_forward_field(schema, "vec_field");
    TEST_ASSERT(vec_as_forward == NULL);

    // Test get_vector_field
    zvec_field_schema_t *found_vec =
        zvec_collection_schema_get_vector_field(schema, "vec_field");
    TEST_ASSERT(found_vec != NULL);
    TEST_ASSERT(zvec_field_schema_is_vector_field(found_vec) == true);

    // get_vector_field should return NULL for scalar field
    zvec_field_schema_t *int_as_vec =
        zvec_collection_schema_get_vector_field(schema, "int_field");
    TEST_ASSERT(int_as_vec == NULL);

    // Test get_all_field_names
    const char **names = NULL;
    size_t name_count = 0;
    zvec_error_code_t err =
        zvec_collection_schema_get_all_field_names(schema, &names, &name_count);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(name_count == 3);
    // Free the strings (caller owns them)
    for (size_t i = 0; i < name_count; i++) {
      zvec_free((char *)names[i]);
    }
    zvec_free(names);

    // Test get_forward_fields
    zvec_field_schema_t **forward_fields = NULL;
    size_t forward_count = 0;
    err = zvec_collection_schema_get_forward_fields(schema, &forward_fields,
                                                    &forward_count);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(forward_count == 2);  // int_field and str_field
    // Note: forward_fields[i] are non-owning pointers, only free the array
    zvec_free(forward_fields);

    // Test get_vector_fields
    zvec_field_schema_t **vector_fields = NULL;
    size_t vector_count = 0;
    err = zvec_collection_schema_get_vector_fields(schema, &vector_fields,
                                                   &vector_count);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(vector_count == 1);  // vec_field
    // Note: vector_fields[i] are non-owning pointers, only free the array
    zvec_free(vector_fields);

    // Test has_index (initially no fields have index)
    TEST_ASSERT(zvec_collection_schema_has_index(schema, "int_field") == false);
    TEST_ASSERT(zvec_collection_schema_has_index(schema, "str_field") == false);
    TEST_ASSERT(zvec_collection_schema_has_index(schema, "vec_field") == false);

    // Test add_index
    zvec_index_params_t *invert_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
    TEST_ASSERT(invert_params != NULL);

    err = zvec_collection_schema_add_index(schema, "int_field", invert_params);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_collection_schema_has_index(schema, "int_field") == true);

    // Test drop_index
    err = zvec_collection_schema_drop_index(schema, "int_field");
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(zvec_collection_schema_has_index(schema, "int_field") == false);

    zvec_index_params_destroy(invert_params);
    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_collection_schema_alter_field(void) {
  TEST_START();

  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("alter_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Create initial field
    zvec_field_schema_t *field =
        zvec_field_schema_create("test_field", ZVEC_DATA_TYPE_INT32, false, 0);
    TEST_ASSERT(field != NULL);

    zvec_error_code_t err = zvec_collection_schema_add_field(schema, field);
    TEST_ASSERT(err == ZVEC_OK);

    // Verify initial state
    const zvec_field_schema_t *found =
        zvec_collection_schema_get_field(schema, "test_field");
    TEST_ASSERT(found != NULL);
    TEST_ASSERT(zvec_field_schema_is_nullable(found) == false);

    // Alter the field to make it nullable
    zvec_field_schema_t *new_field =
        zvec_field_schema_create("test_field", ZVEC_DATA_TYPE_INT32, true, 0);
    TEST_ASSERT(new_field != NULL);

    err = zvec_collection_schema_alter_field(schema, "test_field", new_field);
    TEST_ASSERT(err == ZVEC_OK);

    // Verify the change
    found = zvec_collection_schema_get_field(schema, "test_field");
    TEST_ASSERT(found != NULL);
    TEST_ASSERT(zvec_field_schema_is_nullable(found) == true);

    // Test alter non-existent field
    err = zvec_collection_schema_alter_field(schema, "nonexistent", new_field);
    TEST_ASSERT(err != ZVEC_OK);

    zvec_field_schema_destroy(new_field);
    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

// =============================================================================
// Collection-related tests
// =============================================================================

void test_collection_basic_operations(void) {
  TEST_START();

  // Create temporary directory
  char temp_dir[] = "./zvec_test_collection_basic_operations";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test collection operations
      zvec_doc_t *doc1 = zvec_test_create_doc(1, schema, NULL);
      zvec_doc_t *doc2 = zvec_test_create_doc(2, schema, NULL);
      zvec_doc_t *doc3 = zvec_test_create_doc(3, schema, NULL);

      TEST_ASSERT(doc1 != NULL);
      TEST_ASSERT(doc2 != NULL);
      TEST_ASSERT(doc3 != NULL);

      if (doc1 && doc2 && doc3) {
        zvec_doc_t *docs[] = {doc1, doc2, doc3};
        size_t success_count, error_count;

        // Test insert operation
        err = zvec_collection_insert(collection, (const zvec_doc_t **)docs, 3,
                                     &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 3);
        TEST_ASSERT(error_count == 0);

        // Test update operation
        zvec_doc_set_score(doc1, 0.95f);
        zvec_doc_t *update_docs[] = {doc1};
        err =
            zvec_collection_update(collection, (const zvec_doc_t **)update_docs,
                                   1, &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 1);
        TEST_ASSERT(error_count == 0);

        // Test upsert operation
        zvec_doc_set_pk(doc3, "pk_3_modified");
        zvec_doc_t *upsert_docs[] = {doc3};
        err =
            zvec_collection_upsert(collection, (const zvec_doc_t **)upsert_docs,
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
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_collection_edge_cases(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_edge_cases";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;

    // Test empty name collection
    zvec_error_code_t err =
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
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_collection_delete_by_filter(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_delete_by_filter";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
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
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_collection_stats(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_stats";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      zvec_collection_stats_t *stats = NULL;
      err = zvec_collection_get_stats(collection, &stats);
      TEST_ASSERT(err == ZVEC_OK);

      if (stats) {
        // Basic validation of statistics
        TEST_ASSERT(zvec_collection_stats_get_doc_count(stats) ==
                    0);  // New collection should have no documents
        zvec_collection_stats_destroy(stats);
      }

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

// =============================================================================
// Field-related tests
// =============================================================================

void test_field_schema_functions(void) {
  TEST_START();

  // Test scalar field creation using API
  zvec_field_schema_t *scalar_field =
      zvec_field_schema_create("test_field", ZVEC_DATA_TYPE_STRING, true, 0);
  TEST_ASSERT(scalar_field != NULL);
  if (scalar_field) {
    TEST_ASSERT(
        strcmp(zvec_field_schema_get_name(scalar_field), "test_field") == 0);
    TEST_ASSERT(zvec_field_schema_get_data_type(scalar_field) ==
                ZVEC_DATA_TYPE_STRING);
    TEST_ASSERT(zvec_field_schema_is_nullable(scalar_field) == true);
    TEST_ASSERT(zvec_field_schema_get_dimension(scalar_field) == 0);

    // Test new functions for scalar field
    TEST_ASSERT(zvec_field_schema_is_vector_field(scalar_field) == false);
    TEST_ASSERT(zvec_field_schema_is_dense_vector(scalar_field) == false);
    TEST_ASSERT(zvec_field_schema_is_sparse_vector(scalar_field) == false);
    TEST_ASSERT(zvec_field_schema_is_array_type(scalar_field) == false);
    TEST_ASSERT(zvec_field_schema_get_element_data_type(scalar_field) ==
                ZVEC_DATA_TYPE_STRING);
    TEST_ASSERT(zvec_field_schema_has_invert_index(scalar_field) == false);
    TEST_ASSERT(zvec_field_schema_get_index_type(scalar_field) ==
                ZVEC_INDEX_TYPE_UNDEFINED);

    zvec_field_schema_destroy(scalar_field);
  }

  // Test vector field creation using API
  zvec_field_schema_t *vector_field = zvec_field_schema_create(
      "vec_field", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  TEST_ASSERT(vector_field != NULL);
  if (vector_field) {
    TEST_ASSERT(strcmp(zvec_field_schema_get_name(vector_field), "vec_field") ==
                0);
    TEST_ASSERT(zvec_field_schema_get_data_type(vector_field) ==
                ZVEC_DATA_TYPE_VECTOR_FP32);
    TEST_ASSERT(zvec_field_schema_is_nullable(vector_field) == false);
    TEST_ASSERT(zvec_field_schema_get_dimension(vector_field) == 128);

    // Test new functions for dense vector field
    TEST_ASSERT(zvec_field_schema_is_vector_field(vector_field) == true);
    TEST_ASSERT(zvec_field_schema_is_dense_vector(vector_field) == true);
    TEST_ASSERT(zvec_field_schema_is_sparse_vector(vector_field) == false);
    TEST_ASSERT(zvec_field_schema_is_array_type(vector_field) == false);

    zvec_field_schema_destroy(vector_field);
  }

  // Test sparse vector field creation using API
  zvec_field_schema_t *sparse_field = zvec_field_schema_create(
      "sparse_field", ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32, false, 0);
  TEST_ASSERT(sparse_field != NULL);
  if (sparse_field) {
    TEST_ASSERT(
        strcmp(zvec_field_schema_get_name(sparse_field), "sparse_field") == 0);
    TEST_ASSERT(zvec_field_schema_get_data_type(sparse_field) ==
                ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32);

    // Test new functions for sparse vector field
    TEST_ASSERT(zvec_field_schema_is_vector_field(sparse_field) == true);
    TEST_ASSERT(zvec_field_schema_is_dense_vector(sparse_field) == false);
    TEST_ASSERT(zvec_field_schema_is_sparse_vector(sparse_field) == true);

    zvec_field_schema_destroy(sparse_field);
  }

  // Test array field
  zvec_field_schema_t *array_field = zvec_field_schema_create(
      "array_field", ZVEC_DATA_TYPE_ARRAY_INT32, false, 0);
  TEST_ASSERT(array_field != NULL);
  if (array_field) {
    TEST_ASSERT(zvec_field_schema_is_array_type(array_field) == true);
    TEST_ASSERT(zvec_field_schema_is_vector_field(array_field) == false);
    TEST_ASSERT(zvec_field_schema_get_element_data_type(array_field) ==
                ZVEC_DATA_TYPE_INT32);

    zvec_field_schema_destroy(array_field);
  }

  // Test field with invert index
  zvec_index_params_t *invert_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  zvec_index_params_set_metric_type(invert_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_invert_params(invert_params, true, false);

  zvec_field_schema_t *indexed_field =
      zvec_field_schema_create("indexed_field", ZVEC_DATA_TYPE_INT64, false, 0);
  TEST_ASSERT(indexed_field != NULL);
  if (indexed_field) {
    zvec_field_schema_set_index_params(indexed_field, invert_params);
    TEST_ASSERT(zvec_field_schema_has_index(indexed_field) == true);
    TEST_ASSERT(zvec_field_schema_get_index_type(indexed_field) ==
                ZVEC_INDEX_TYPE_INVERT);
    TEST_ASSERT(zvec_field_schema_has_invert_index(indexed_field) == true);

    zvec_field_schema_destroy(indexed_field);
  }
  zvec_index_params_destroy(invert_params);

  // Test field with HNSW index
  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(hnsw_params, 16, 200);

  zvec_field_schema_t *hnsw_field = zvec_field_schema_create(
      "hnsw_field", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  TEST_ASSERT(hnsw_field != NULL);
  if (hnsw_field) {
    zvec_field_schema_set_index_params(hnsw_field, hnsw_params);
    TEST_ASSERT(zvec_field_schema_has_index(hnsw_field) == true);
    TEST_ASSERT(zvec_field_schema_get_index_type(hnsw_field) ==
                ZVEC_INDEX_TYPE_HNSW);
    TEST_ASSERT(zvec_field_schema_has_invert_index(hnsw_field) ==
                false);  // Vector field, no invert index

    zvec_field_schema_destroy(hnsw_field);
  }
  zvec_index_params_destroy(hnsw_params);

  TEST_END();
}

void test_field_helper_functions(void) {
  TEST_START();

  // Test scalar field helper functions
  zvec_index_params_t *invert_params =
      zvec_test_create_default_invert_params(true);
  zvec_field_schema_t *scalar_field = zvec_test_create_scalar_field(
      "test_scalar", ZVEC_DATA_TYPE_INT32, true, invert_params);
  TEST_ASSERT(scalar_field != NULL);
  if (scalar_field) {
    TEST_ASSERT(
        strcmp(zvec_field_schema_get_name(scalar_field), "test_scalar") == 0);
    TEST_ASSERT(zvec_field_schema_get_data_type(scalar_field) ==
                ZVEC_DATA_TYPE_INT32);
    zvec_field_schema_destroy(scalar_field);
  }
  zvec_index_params_destroy(invert_params);

  // Test vector field helper functions
  zvec_index_params_t *hnsw_params = zvec_test_create_default_hnsw_params();
  zvec_field_schema_t *vector_field = zvec_test_create_vector_field(
      "test_vector", ZVEC_DATA_TYPE_VECTOR_FP32, 128, false, hnsw_params);
  TEST_ASSERT(vector_field != NULL);
  if (vector_field) {
    TEST_ASSERT(
        strcmp(zvec_field_schema_get_name(vector_field), "test_vector") == 0);
    TEST_ASSERT(zvec_field_schema_get_data_type(vector_field) ==
                ZVEC_DATA_TYPE_VECTOR_FP32);
    TEST_ASSERT(zvec_field_schema_get_dimension(vector_field) == 128);
    zvec_field_schema_destroy(vector_field);
  }
  zvec_index_params_destroy(hnsw_params);

  TEST_END();
}

// =============================================================================
// Document-related tests
// =============================================================================

void test_doc_creation(void) {
  TEST_START();

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Test complete document creation
    zvec_doc_t *doc = zvec_test_create_doc(1, schema, NULL);
    TEST_ASSERT(doc != NULL);
    if (doc) {
      zvec_doc_destroy(doc);
    }

    // Test null value document creation
    zvec_doc_t *null_doc = zvec_test_create_doc_null(2, schema, NULL);
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
    zvec_free(pk);
  }

  TEST_END();
}

// Test for zvec_doc_add_field_by_value - covers all data types
void test_doc_add_field_by_value(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  if (!doc) {
    TEST_END();
    return;
  }

  // Scalar types
  // BINARY
  const char *binary_data = "binary";
  zvec_error_code_t err =
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
  zvec_string_t *array_zvec_str[3];
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
    zvec_free(result);
  }

  err = zvec_doc_get_field_value_copy(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                      &result, &result_size);
  TEST_ASSERT(err == ZVEC_OK && result_size == sizeof(float));
  if (result) {
    TEST_ASSERT(fabs(*(float *)result - 3.14159f) < 0.0001f);
    zvec_free(result);
  }

  zvec_doc_destroy(doc);
  TEST_END();
}

// Test for zvec_doc_add_field_by_struct - covers all data types
void test_doc_add_field_by_struct(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  if (!doc) {
    TEST_END();
    return;
  }

  zvec_error_code_t err;
  zvec_doc_field_t field;

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
    zvec_free(result);
  }

  err = zvec_doc_get_field_value_copy(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                      &result, &result_size);
  TEST_ASSERT(err == ZVEC_OK && result_size == sizeof(float));
  if (result) {
    TEST_ASSERT(fabs(*(float *)result - 3.14159f) < 0.0001f);
    zvec_free(result);
  }

  zvec_doc_destroy(doc);
  TEST_END();
}

void test_doc_basic_operations(void);
void test_doc_null_field_api(void);
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
  test_doc_null_field_api();
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
  zvec_doc_t *doc = zvec_doc_create();
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
  zvec_doc_operator_t op = zvec_doc_get_operator(doc);
  TEST_ASSERT(op == ZVEC_DOC_OP_INSERT);

  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_null_field_api(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);
  if (!doc) {
    TEST_END();
    return;
  }

  zvec_error_code_t err = zvec_doc_set_field_null(doc, "nullable_field");
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(zvec_doc_has_field(doc, "nullable_field") == true);
  TEST_ASSERT(zvec_doc_has_field_value(doc, "nullable_field") == false);
  TEST_ASSERT(zvec_doc_is_field_null(doc, "nullable_field") == true);

  err = zvec_doc_set_field_null(NULL, "nullable_field");
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  err = zvec_doc_set_field_null(doc, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  zvec_doc_destroy(doc);
  TEST_END();
}

void test_doc_get_field_value_basic(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  zvec_error_code_t err;

  printf(
      "=== Testing zvec_doc_get_field_value_basic with all supported types "
      "===\n");

  // BOOL type
  zvec_doc_field_t bool_field;
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
  zvec_doc_field_t int32_field;
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
  zvec_doc_field_t int64_field;
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
  zvec_doc_field_t uint32_field;
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
  zvec_doc_field_t uint64_field;
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
  zvec_doc_field_t float_field;
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
  zvec_doc_field_t double_field;
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

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  zvec_error_code_t err;

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
  zvec_free(bool_copy_result);

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
  zvec_free(int32_copy_result);

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
  zvec_free(int64_copy_result);

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
  zvec_free(uint32_copy_result);

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
  zvec_free(uint64_copy_result);

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
  zvec_free(float_copy_result);

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
  zvec_free(double_copy_result);

  // String and binary types
  zvec_doc_field_t string_field;
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
  zvec_free(string_result);

  zvec_doc_field_t binary_field;
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
  zvec_free(binary_result);

  // VECTOR_FP32 type
  float test_vector[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  zvec_doc_field_t fp32_vec_field;
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
  zvec_free(fp32_vec_result);

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
  zvec_free(fp16_result);

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
  zvec_free(int8_result);

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
  zvec_free(bin32_result);

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
  zvec_free(bin64_result);

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
  zvec_free(fp64_result);

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
  zvec_free(int16_result);

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
  zvec_free(sparse_fp16_result);

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
  zvec_free(sparse_fp32_result);

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
  zvec_free(array_binary_result);


  // ARRAY_STRING type
  const char *array_str_data[] = {"str1", "str2", "str3"};
  zvec_string_t *array_zvec_str[3];
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
  zvec_free(array_string_result);
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
  zvec_free(array_bool_result);

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
  zvec_free(array_int32_result);

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
  zvec_free(array_int64_result);

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
  zvec_free(array_uint32_result);

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
  zvec_free(array_uint64_result);

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
  zvec_free(array_float_result);

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
  zvec_free(array_double_result);


  zvec_free(binary_field.value.string_value.data);
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_get_field_value_pointer(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  zvec_error_code_t err;

  // Add fields for pointer testing
  zvec_doc_field_t bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &bool_field);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_doc_field_t int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_doc_field_t string_field;
  string_field.name.data = "string_field";
  string_field.name.length = strlen("string_field");
  string_field.data_type = ZVEC_DATA_TYPE_STRING;
  string_field.value.string_value = *zvec_string_create("Hello, 世界!");
  err = zvec_doc_add_field_by_struct(doc, &string_field);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_doc_field_t binary_field;
  binary_field.name.data = "binary_field";
  binary_field.name.length = strlen("binary_field");
  binary_field.data_type = ZVEC_DATA_TYPE_BINARY;
  uint8_t binary_data[] = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD};
  binary_field.value.string_value =
      *zvec_bin_create(binary_data, sizeof(binary_data));
  err = zvec_doc_add_field_by_struct(doc, &binary_field);
  TEST_ASSERT(err == ZVEC_OK);

  float test_vector[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  zvec_doc_field_t fp32_vec_field;
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

  zvec_free(string_field.value.string_value.data);
  zvec_free(binary_field.value.string_value.data);
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_field_operations(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  zvec_error_code_t err;

  // Add some fields
  zvec_doc_field_t bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &bool_field);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_doc_field_t int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_doc_field_t string_field;
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
  zvec_free(string_field.value.string_value.data);
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_doc_error_conditions(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  // Add a field for error testing
  zvec_doc_field_t bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  zvec_doc_add_field_by_struct(doc, &bool_field);

  zvec_error_code_t err;
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

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  zvec_error_code_t err;

  // Add fields for serialization testing
  zvec_doc_field_t int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_doc_field_t string_field;
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

  zvec_doc_t *deserialized_doc;
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
  zvec_index_params_t *hnsw_params = zvec_test_create_default_hnsw_params();
  TEST_ASSERT(hnsw_params != NULL);
  if (hnsw_params) {
    zvec_free(hnsw_params);
  }

  // Test Flat parameter creation
  zvec_index_params_t *flat_params = zvec_test_create_default_flat_params();
  TEST_ASSERT(flat_params != NULL);
  if (flat_params) {
    zvec_free(flat_params);
  }

  // Test scalar index parameter creation
  zvec_index_params_t *invert_params =
      zvec_test_create_default_invert_params(true);
  TEST_ASSERT(invert_params != NULL);
  if (invert_params) {
    zvec_free(invert_params);
  }

  TEST_END();
}

// =============================================================================
// Memory management tests
// =============================================================================
void test_zvec_string_functions(void) {
  TEST_START();

  // Test string creation and basic operations
  zvec_string_t *str1 = zvec_string_create("Hello World");
  TEST_ASSERT(str1 != NULL);
  TEST_ASSERT(zvec_string_length(str1) == 11);
  TEST_ASSERT(strcmp(zvec_string_c_str(str1), "Hello World") == 0);

  // Test string copy
  zvec_string_t *str2 = zvec_string_copy(str1);
  TEST_ASSERT(str2 != NULL);
  TEST_ASSERT(zvec_string_length(str2) == 11);
  TEST_ASSERT(strcmp(zvec_string_c_str(str2), "Hello World") == 0);

  // Test string comparison
  int cmp_result = zvec_string_compare(str1, str2);
  TEST_ASSERT(cmp_result == 0);

  zvec_string_t *str3 = zvec_string_create("Hello");
  TEST_ASSERT(zvec_string_compare(str1, str3) > 0);

  // Test string creation from view
  zvec_string_view_t view = {"Hello View", 10};
  zvec_string_t *str4 = zvec_string_create_from_view(&view);
  TEST_ASSERT(str4 != NULL);
  TEST_ASSERT(zvec_string_length(str4) == 10);
  TEST_ASSERT(strcmp(zvec_string_c_str(str4), "Hello View") == 0);

  // Test string view with embedded null bytes
  char binary_data[] = {'H', 'e', 'l', 'l', 'o', '\0', 'W', 'o', 'r', 'l', 'd'};
  zvec_string_view_t binary_view = {binary_data, 11};
  zvec_string_t *str5 = zvec_string_create_from_view(&binary_view);
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

  // Test index params with new opaque pointer API
  // Test HNSW params
  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(hnsw_params) == ZVEC_INDEX_TYPE_HNSW);
  // Default metric type is L2, need to set it explicitly
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_COSINE);
  TEST_ASSERT(zvec_index_params_get_metric_type(hnsw_params) ==
              ZVEC_METRIC_TYPE_COSINE);

  int m, ef_construction;
  m = zvec_index_params_get_hnsw_m(hnsw_params);
  ef_construction = zvec_index_params_get_hnsw_ef_construction(hnsw_params);
  TEST_ASSERT(m == 50);
  TEST_ASSERT(ef_construction == 500);

  // Test invert index params
  zvec_index_params_t *invert_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(invert_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(invert_params) ==
              ZVEC_INDEX_TYPE_INVERT);

  bool enable_range_opt, enable_wildcard;
  zvec_index_params_get_invert_params(invert_params, &enable_range_opt,
                                      &enable_wildcard);
  TEST_ASSERT(enable_range_opt == true);  // Default is true
  TEST_ASSERT(enable_wildcard == false);  // Default is false

  // Test flat index params
  zvec_index_params_t *flat_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
  TEST_ASSERT(flat_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(flat_params) == ZVEC_INDEX_TYPE_FLAT);
  // Default metric type is L2, need to set it explicitly
  zvec_index_params_set_metric_type(flat_params, ZVEC_METRIC_TYPE_IP);
  TEST_ASSERT(zvec_index_params_get_metric_type(flat_params) ==
              ZVEC_METRIC_TYPE_IP);

  // Test IVF index params
  zvec_index_params_t *ivf_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(ivf_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(ivf_params) == ZVEC_INDEX_TYPE_IVF);
  // Default metric type is L2
  TEST_ASSERT(zvec_index_params_get_metric_type(ivf_params) ==
              ZVEC_METRIC_TYPE_L2);

  int n_list, n_iters;
  bool use_soar;
  zvec_index_params_get_ivf_params(ivf_params, &n_list, &n_iters, &use_soar);
  TEST_ASSERT(n_list == 1024);
  TEST_ASSERT(n_iters == 10);
  TEST_ASSERT(use_soar == false);  // Default is false

  // Cleanup
  zvec_index_params_destroy(hnsw_params);
  zvec_index_params_destroy(invert_params);
  zvec_index_params_destroy(flat_params);
  zvec_index_params_destroy(ivf_params);

  TEST_END();
}

void test_index_params_api_functions(void) {
  TEST_START();

  // Test zvec_index_params_create for HNSW
  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(hnsw_params) == ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(zvec_index_params_get_metric_type(hnsw_params) ==
              ZVEC_METRIC_TYPE_L2);

  // Test zvec_index_params_set_metric_type
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_COSINE);
  TEST_ASSERT(zvec_index_params_get_metric_type(hnsw_params) ==
              ZVEC_METRIC_TYPE_COSINE);

  // Test zvec_index_params_set_hnsw_params
  zvec_index_params_set_hnsw_params(hnsw_params, 32, 300);
  int m, ef_construction;
  m = zvec_index_params_get_hnsw_m(hnsw_params);
  ef_construction = zvec_index_params_get_hnsw_ef_construction(hnsw_params);
  TEST_ASSERT(m == 32);
  TEST_ASSERT(ef_construction == 300);

  // Test zvec_index_params_create for IVF
  zvec_index_params_t *ivf_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(ivf_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(ivf_params) == ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(zvec_index_params_get_metric_type(ivf_params) ==
              ZVEC_METRIC_TYPE_L2);

  // Test zvec_index_params_set_ivf_params
  zvec_index_params_set_ivf_params(ivf_params, 200, 20, true);
  int n_list, n_iters;
  bool use_soar;
  zvec_index_params_get_ivf_params(ivf_params, &n_list, &n_iters, &use_soar);
  TEST_ASSERT(n_list == 200);
  TEST_ASSERT(n_iters == 20);
  TEST_ASSERT(use_soar == true);

  // Test zvec_index_params_create for INVERT
  zvec_index_params_t *invert_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(invert_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(invert_params) ==
              ZVEC_INDEX_TYPE_INVERT);

  // Test zvec_index_params_set_invert_params
  zvec_index_params_set_invert_params(invert_params, true, true);
  bool enable_range_opt, enable_wildcard;
  zvec_index_params_get_invert_params(invert_params, &enable_range_opt,
                                      &enable_wildcard);
  TEST_ASSERT(enable_range_opt == true);
  TEST_ASSERT(enable_wildcard == true);

  // Test zvec_index_params_create for FLAT
  zvec_index_params_t *flat_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
  TEST_ASSERT(flat_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(flat_params) == ZVEC_INDEX_TYPE_FLAT);
  zvec_index_params_set_metric_type(flat_params, ZVEC_METRIC_TYPE_IP);
  TEST_ASSERT(zvec_index_params_get_metric_type(flat_params) ==
              ZVEC_METRIC_TYPE_IP);

  // Cleanup
  zvec_index_params_destroy(hnsw_params);
  zvec_index_params_destroy(ivf_params);
  zvec_index_params_destroy(invert_params);
  zvec_index_params_destroy(flat_params);

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
  zvec_string_t *str = zvec_string_create("Test String");
  TEST_ASSERT(str != NULL);
  zvec_free_string(str);

  void *buffer = malloc(64);
  TEST_ASSERT(buffer != NULL);
  zvec_free(buffer);

  TEST_END();
}

void test_query_params_functions(void) {
  TEST_START();

  // Test HNSW query parameters
  zvec_hnsw_query_params_t *hnsw_params =
      zvec_query_params_hnsw_create(50, 0.5f, false, true);
  TEST_ASSERT(hnsw_params != NULL);

  // Test IVF query parameters
  zvec_ivf_query_params_t *ivf_params =
      zvec_query_params_ivf_create(10, true, 1.5f);
  TEST_ASSERT(ivf_params != NULL);

  // Test Flat query parameters
  zvec_flat_query_params_t *flat_params =
      zvec_query_params_flat_create(false, 2.0f);
  TEST_ASSERT(flat_params != NULL);

  zvec_error_code_t err;

  // Test HNSW-specific parameters
  err = zvec_query_params_hnsw_set_ef(hnsw_params, 75);
  TEST_ASSERT(err == ZVEC_OK);

  // Test HNSW common parameters (radius, is_linear, is_using_refiner)
  err = zvec_query_params_hnsw_set_radius(hnsw_params, 0.8f);
  TEST_ASSERT(err == ZVEC_OK);
  float radius = zvec_query_params_hnsw_get_radius(hnsw_params);
  TEST_ASSERT(radius == 0.8f);

  err = zvec_query_params_hnsw_set_is_linear(hnsw_params, false);
  TEST_ASSERT(err == ZVEC_OK);
  bool is_linear = zvec_query_params_hnsw_get_is_linear(hnsw_params);
  TEST_ASSERT(is_linear == false);

  err = zvec_query_params_hnsw_set_is_using_refiner(hnsw_params, true);
  TEST_ASSERT(err == ZVEC_OK);
  bool is_using_refiner =
      zvec_query_params_hnsw_get_is_using_refiner(hnsw_params);
  TEST_ASSERT(is_using_refiner == true);

  // Test IVF-specific parameters
  err = zvec_query_params_ivf_set_nprobe(ivf_params, 15);
  TEST_ASSERT(err == ZVEC_OK);

  // Test IVF scale factor setting
  err = zvec_query_params_ivf_set_scale_factor(ivf_params, 2.5f);
  TEST_ASSERT(err == ZVEC_OK);

  // Test IVF common parameters (radius, is_linear, is_using_refiner)
  err = zvec_query_params_ivf_set_radius(ivf_params, 0.9f);
  TEST_ASSERT(err == ZVEC_OK);
  radius = zvec_query_params_ivf_get_radius(ivf_params);
  TEST_ASSERT(radius == 0.9f);

  err = zvec_query_params_ivf_set_is_linear(ivf_params, true);
  TEST_ASSERT(err == ZVEC_OK);
  is_linear = zvec_query_params_ivf_get_is_linear(ivf_params);
  TEST_ASSERT(is_linear == true);

  err = zvec_query_params_ivf_set_is_using_refiner(ivf_params, false);
  TEST_ASSERT(err == ZVEC_OK);
  is_using_refiner = zvec_query_params_ivf_get_is_using_refiner(ivf_params);
  TEST_ASSERT(is_using_refiner == false);

  // Test Flat scale factor setting
  err = zvec_query_params_flat_set_scale_factor(flat_params, 3.0f);
  TEST_ASSERT(err == ZVEC_OK);

  // Test Flat common parameters (radius, is_linear, is_using_refiner)
  err = zvec_query_params_flat_set_radius(flat_params, 0.7f);
  TEST_ASSERT(err == ZVEC_OK);
  radius = zvec_query_params_flat_get_radius(flat_params);
  TEST_ASSERT(radius == 0.7f);

  err = zvec_query_params_flat_set_is_linear(flat_params, true);
  TEST_ASSERT(err == ZVEC_OK);
  is_linear = zvec_query_params_flat_get_is_linear(flat_params);
  TEST_ASSERT(is_linear == true);

  err = zvec_query_params_flat_set_is_using_refiner(flat_params, true);
  TEST_ASSERT(err == ZVEC_OK);
  is_using_refiner = zvec_query_params_flat_get_is_using_refiner(flat_params);
  TEST_ASSERT(is_using_refiner == true);

  // Test destruction of valid parameters
  zvec_query_params_hnsw_destroy(hnsw_params);
  zvec_query_params_ivf_destroy(ivf_params);
  zvec_query_params_flat_destroy(flat_params);


  // Test boundary cases - null pointer handling
  zvec_query_params_hnsw_destroy(NULL);
  zvec_query_params_ivf_destroy(NULL);
  zvec_query_params_flat_destroy(NULL);

  // Test null pointer handling for setters
  err = zvec_query_params_hnsw_set_radius(NULL, 0.5f);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  err = zvec_query_params_ivf_set_radius(NULL, 0.5f);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  err = zvec_query_params_flat_set_radius(NULL, 0.5f);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test default values for getters with NULL
  TEST_ASSERT(zvec_query_params_hnsw_get_radius(NULL) == 0.0f);
  TEST_ASSERT(zvec_query_params_ivf_get_radius(NULL) == 0.0f);
  TEST_ASSERT(zvec_query_params_flat_get_radius(NULL) == 0.0f);
  TEST_ASSERT(zvec_query_params_hnsw_get_is_linear(NULL) == false);
  TEST_ASSERT(zvec_query_params_ivf_get_is_linear(NULL) == false);
  TEST_ASSERT(zvec_query_params_flat_get_is_linear(NULL) == false);
  TEST_ASSERT(zvec_query_params_hnsw_get_is_using_refiner(NULL) == false);
  TEST_ASSERT(zvec_query_params_ivf_get_is_using_refiner(NULL) == false);
  TEST_ASSERT(zvec_query_params_flat_get_is_using_refiner(NULL) == false);

  TEST_END();
}

void test_collection_stats_functions(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_stats_functions";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      zvec_collection_stats_t *stats = NULL;

      // Test normal statistics retrieval
      err = zvec_collection_get_stats(collection, &stats);
      TEST_ASSERT(err == ZVEC_OK);

      if (stats) {
        TEST_ASSERT(zvec_collection_stats_get_doc_count(stats) == 0);
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
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_collection_dml_functions(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_dml";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
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
      zvec_doc_t *empty_docs[1];
      err = zvec_collection_insert(collection, (const zvec_doc_t **)empty_docs,
                                   0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test NULL count pointer
      err = zvec_collection_insert(collection, (const zvec_doc_t **)empty_docs,
                                   1, NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test update function boundary cases
      err = zvec_collection_update(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_update(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_update(collection, (const zvec_doc_t **)empty_docs,
                                   0, NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test upsert function boundary cases
      err = zvec_collection_upsert(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_upsert(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_upsert(collection, (const zvec_doc_t **)empty_docs,
                                   0, NULL, &error_count);
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

      // Test detailed DML result APIs
      zvec_doc_t *result_doc = zvec_test_create_doc(101, schema, NULL);
      TEST_ASSERT(result_doc != NULL);
      if (result_doc) {
        zvec_doc_t *result_docs[] = {result_doc};
        zvec_write_result_t *results = NULL;
        size_t result_count = 0;

        err = zvec_collection_upsert_with_results(
            collection, (const zvec_doc_t **)result_docs, 1, &results,
            &result_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(result_count == 1);
        if (results && result_count == 1) {
          TEST_ASSERT(results[0].code == ZVEC_OK);
          zvec_write_results_free(results, result_count);
        }

        const char *delete_pks[] = {"pk_101"};
        results = NULL;
        result_count = 0;
        err = zvec_collection_delete_with_results(collection, delete_pks, 1,
                                                  &results, &result_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(result_count == 1);
        if (results && result_count == 1) {
          zvec_write_results_free(results, result_count);
        }

        zvec_doc_destroy(result_doc);
      }

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_collection_nullable_roundtrip(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_nullable_roundtrip";
  zvec_test_delete_dir(temp_dir);

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);
  if (!schema) {
    TEST_END();
    return;
  }

  zvec_collection_t *collection = NULL;
  zvec_error_code_t err =
      zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(collection != NULL);

  if (collection) {
    zvec_doc_t *doc = zvec_doc_create();
    TEST_ASSERT(doc != NULL);
    if (doc) {
      zvec_doc_set_pk(doc, "pk_nullable");

      int64_t id = 77;
      err = zvec_doc_add_field_by_value(doc, "id", ZVEC_DATA_TYPE_INT64, &id,
                                        sizeof(id));
      TEST_ASSERT(err == ZVEC_OK);

      const char *name = "nullable";
      err = zvec_doc_add_field_by_value(doc, "name", ZVEC_DATA_TYPE_STRING,
                                        name, strlen(name));
      TEST_ASSERT(err == ZVEC_OK);

      // "weight" in temp schema is nullable.
      err = zvec_doc_set_field_null(doc, "weight");
      TEST_ASSERT(err == ZVEC_OK);

      float dense[128];
      for (size_t i = 0; i < 128; ++i) {
        dense[i] = (float)i / 128.0f;
      }
      err = zvec_doc_add_field_by_value(
          doc, "dense", ZVEC_DATA_TYPE_VECTOR_FP32, dense, sizeof(dense));
      TEST_ASSERT(err == ZVEC_OK);

      uint32_t nnz = 3;
      uint32_t sparse_indices[] = {1, 5, 9};
      float sparse_values[] = {0.2f, 0.5f, 0.9f};
      char sparse_buffer[sizeof(nnz) + sizeof(sparse_indices) +
                         sizeof(sparse_values)];
      memcpy(sparse_buffer, &nnz, sizeof(nnz));
      memcpy(sparse_buffer + sizeof(nnz), sparse_indices,
             sizeof(sparse_indices));
      memcpy(sparse_buffer + sizeof(nnz) + sizeof(sparse_indices),
             sparse_values, sizeof(sparse_values));
      err = zvec_doc_add_field_by_value(doc, "sparse",
                                        ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32,
                                        sparse_buffer, sizeof(sparse_buffer));
      TEST_ASSERT(err == ZVEC_OK);

      zvec_doc_t *docs[] = {doc};
      size_t success_count = 0;
      size_t error_count = 0;
      err = zvec_collection_upsert(collection, (const zvec_doc_t **)docs, 1,
                                   &success_count, &error_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(success_count == 1);
      TEST_ASSERT(error_count == 0);

      const char *pks[] = {"pk_nullable"};
      zvec_doc_t **fetched = NULL;
      size_t fetched_count = 0;
      err = zvec_collection_fetch(collection, pks, 1, &fetched, &fetched_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(fetched_count == 1);
      if (fetched && fetched_count == 1) {
        TEST_ASSERT(zvec_doc_has_field(fetched[0], "weight") == true);
        TEST_ASSERT(zvec_doc_has_field_value(fetched[0], "weight") == false);
        TEST_ASSERT(zvec_doc_is_field_null(fetched[0], "weight") == true);
      }
      zvec_docs_free(fetched, fetched_count);
      zvec_doc_destroy(doc);
    }

    zvec_collection_destroy(collection);
  }

  zvec_collection_schema_destroy(schema);
  zvec_test_delete_dir(temp_dir);

  TEST_END();
}

// =============================================================================
// Actual Query Execution Tests
// =============================================================================

void test_actual_vector_queries(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_actual_queries";

  // Create schema with vector field
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("query_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Add ID field
    zvec_field_schema_t *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    // Add vector field with HNSW index
    zvec_index_params_t *hnsw_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
    TEST_ASSERT(hnsw_params != NULL);
    zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
    zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);
    zvec_field_schema_t *vec_field = zvec_field_schema_create(
        "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
    zvec_field_schema_set_index_params(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);
    zvec_index_params_destroy(hnsw_params);

    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Insert test documents
      float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
      float vec2[] = {0.0f, 1.0f, 0.0f, 0.0f};
      float vec3[] = {0.0f, 0.0f, 1.0f, 0.0f};
      float vec4[] = {0.7f, 0.7f, 0.0f, 0.0f};  // Similar to vec1 and vec2

      zvec_doc_t *docs[4];
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
      err = zvec_collection_insert(collection, (const zvec_doc_t **)docs, 4,
                                   &success_count, &error_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(success_count == 4);
      TEST_ASSERT(error_count == 0);

      // Flush collection to build index
      zvec_collection_flush(collection);

      // Test 1: Basic vector search
      zvec_vector_query_t *query1 = zvec_vector_query_create();
      TEST_ASSERT(query1 != NULL);
      zvec_vector_query_set_field_name(query1, "embedding");
      zvec_vector_query_set_query_vector(query1, vec1, sizeof(vec1));
      zvec_vector_query_set_topk(query1, 3);
      zvec_vector_query_set_include_vector(query1, true);
      zvec_vector_query_set_include_doc_id(query1, true);

      zvec_doc_t **results = NULL;
      size_t result_count = 0;
      err = zvec_collection_query(collection, query1, &results, &result_count);
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
      zvec_vector_query_set_filter(query1, "id > 2");

      err = zvec_collection_query(collection, query1, &results, &result_count);
      TEST_ASSERT(err == ZVEC_OK);

      // Should only return documents with id > 2
      for (size_t i = 0; i < result_count; i++) {
        int64_t id;
        zvec_doc_get_field_value_basic(results[i], "id", ZVEC_DATA_TYPE_INT64,
                                       &id, sizeof(id));
        TEST_ASSERT(id > 2);
      }

      zvec_docs_free(results, result_count);

      // Cleanup documents and query
      for (int i = 0; i < 4; i++) {
        zvec_doc_destroy(docs[i]);
      }

      zvec_vector_query_destroy(query1);
      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_index_creation_and_management(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_index_management";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test 1: Create HNSW index
      zvec_index_params_t *hnsw_params =
          zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
      TEST_ASSERT(hnsw_params != NULL);
      zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_COSINE);
      zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);

      err = zvec_collection_create_index(collection, "dense", hnsw_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Test 2: Create scalar index
      zvec_index_params_t *invert_params =
          zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
      TEST_ASSERT(invert_params != NULL);
      zvec_index_params_set_invert_params(invert_params, true, false);

      err = zvec_collection_create_index(collection, "name", invert_params);
      TEST_ASSERT(err == ZVEC_OK);

      err = zvec_collection_drop_index(collection, "name");
      TEST_ASSERT(err == ZVEC_OK);

      // Test 3: Optimize collection
      err = zvec_collection_optimize(collection);
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
      zvec_index_params_destroy(hnsw_params);
      zvec_index_params_destroy(invert_params);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_collection_ddl_operations(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_collection_ddl";

  zvec_collection_schema_t *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  size_t field_count = get_field_count(schema);

  if (schema) {
    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test 1: Add new column
      zvec_field_schema_t *new_field =
          zvec_field_schema_create("new_int32", ZVEC_DATA_TYPE_INT32, true, 0);
      TEST_ASSERT(new_field != NULL);

      err = zvec_collection_add_column(collection, new_field, NULL);
      TEST_ASSERT(err == ZVEC_OK);

      // Test 2: Get collection schema and verify field count
      zvec_collection_schema_t *retrieved_schema = NULL;
      err = zvec_collection_get_schema(collection, &retrieved_schema);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(retrieved_schema != NULL);

      size_t new_field_count = get_field_count(retrieved_schema);
      TEST_ASSERT((field_count + 1) == new_field_count);

      // Test 3: Alter column
      zvec_field_schema_t *alter_field =
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
      new_field_count = get_field_count(retrieved_schema);
      TEST_ASSERT(new_field_count == field_count);

      zvec_collection_schema_destroy(retrieved_schema);
      zvec_field_schema_destroy(new_field);
      zvec_field_schema_destroy(alter_field);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

void test_field_ddl_operations(void) {
  TEST_START();

  // Test field schema creation with various configurations
  zvec_field_schema_t *field1 =
      zvec_field_schema_create("test_field1", ZVEC_DATA_TYPE_STRING, false, 0);
  TEST_ASSERT(field1 != NULL);
  TEST_ASSERT(strcmp(zvec_field_schema_get_name(field1), "test_field1") == 0);
  TEST_ASSERT(zvec_field_schema_get_data_type(field1) == ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(zvec_field_schema_is_nullable(field1) == false);
  TEST_ASSERT(zvec_field_schema_get_dimension(field1) == 0);

  zvec_field_schema_t *field2 = zvec_field_schema_create(
      "test_field2", ZVEC_DATA_TYPE_VECTOR_FP32, true, 128);
  TEST_ASSERT(field2 != NULL);
  TEST_ASSERT(zvec_field_schema_get_data_type(field2) ==
              ZVEC_DATA_TYPE_VECTOR_FP32);
  TEST_ASSERT(zvec_field_schema_is_nullable(field2) == true);
  TEST_ASSERT(zvec_field_schema_get_dimension(field2) == 128);

  // Test index parameter setting
  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params != NULL);
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);

  zvec_error_code_t err =
      zvec_field_schema_set_index_params(field2, hnsw_params);
  TEST_ASSERT(err == ZVEC_OK);

  // Cleanup
  zvec_field_schema_destroy(field1);
  zvec_field_schema_destroy(field2);
  zvec_index_params_destroy(hnsw_params);

  TEST_END();
}

void test_performance_benchmarks(void) {
  TEST_START();

  char temp_dir[] = "./zvec_test_performance";

  zvec_collection_schema_t *schema = zvec_collection_schema_create("perf_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Create simple schema for performance testing
    zvec_field_schema_t *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    zvec_field_schema_t *vec_field =
        zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
    zvec_index_params_t *hnsw_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
    zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
    zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);
    zvec_field_schema_set_index_params(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);

    zvec_collection_t *collection = NULL;
    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    TEST_ASSERT(collection != NULL);

    if (collection) {
      const size_t BATCH_SIZE = 1000;
      const size_t TOTAL_DOCS = 10000;

      // Test bulk insertion performance
#ifdef _WIN32
      clock_t start_clock = clock();
#else
      struct timeval start_time, end_time;
      gettimeofday(&start_time, NULL);
#endif

      for (size_t batch_start = 0; batch_start < TOTAL_DOCS;
           batch_start += BATCH_SIZE) {
        // Use dynamic allocation for MSVC compatibility (no VLA support)
        zvec_doc_t **batch_docs =
            (zvec_doc_t **)malloc(BATCH_SIZE * sizeof(zvec_doc_t *));
        if (!batch_docs) {
          fprintf(stderr, "Failed to allocate batch documents\n");
          break;
        }
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
        err = zvec_collection_insert(
            collection, (const zvec_doc_t **)batch_docs, current_batch_size,
            &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == current_batch_size);
        TEST_ASSERT(error_count == 0);

        // Cleanup batch documents
        for (size_t i = 0; i < current_batch_size; i++) {
          zvec_doc_destroy(batch_docs[i]);
        }
        free(batch_docs);  // Free the array itself
      }

#ifdef _WIN32
      clock_t end_clock = clock();
      double insert_time = ((double)(end_clock - start_clock)) / CLOCKS_PER_SEC;
#else
      gettimeofday(&end_time, NULL);
      double insert_time = (end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
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

      zvec_vector_query_t *query = zvec_vector_query_create();
      TEST_ASSERT(query != NULL);
      zvec_vector_query_set_field_name(query, "vec");
      zvec_vector_query_set_query_vector(query, query_vec, sizeof(query_vec));
      zvec_vector_query_set_topk(query, 10);
      zvec_vector_query_set_include_vector(query, false);
      zvec_vector_query_set_include_doc_id(query, true);

      const int QUERY_COUNT = 100;
#ifdef _WIN32
      clock_t query_start_clock = clock();
#else
      struct timeval query_start_time, query_end_time;
      gettimeofday(&query_start_time, NULL);
#endif

      for (int q = 0; q < QUERY_COUNT; q++) {
        zvec_doc_t **results = NULL;
        size_t result_count = 0;

        err = zvec_collection_query(collection, query, &results, &result_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(result_count <= 10);

        zvec_docs_free(results, result_count);
      }

#ifdef _WIN32
      clock_t query_end_clock = clock();
      double query_time =
          ((double)(query_end_clock - query_start_clock)) / CLOCKS_PER_SEC;
#else
      gettimeofday(&query_end_time, NULL);
      double query_time =
          (query_end_time.tv_sec - query_start_time.tv_sec) +
          (query_end_time.tv_usec - query_start_time.tv_usec) / 1000000.0;
#endif
      double avg_query_time =
          (query_time * 1000) / QUERY_COUNT;  // ms per query
      printf("  Average query time: %.2f ms\n", avg_query_time);

      zvec_vector_query_destroy(query);
      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  cleanup_temp_directory(temp_dir);

  TEST_END();
}

// =============================================================================
// Additional tests for uncovered API functions
// =============================================================================

void test_zvec_shutdown(void) {
  TEST_START();

  // Test shutdown
  zvec_error_code_t err = zvec_shutdown();
  TEST_ASSERT(err == ZVEC_OK);

  // Re-initialize for other tests
  zvec_config_data_t *config = zvec_config_data_create();
  TEST_ASSERT(config != NULL);
  err = zvec_initialize(config);
  TEST_ASSERT(err == ZVEC_OK);
  zvec_config_data_destroy(config);

  TEST_END();
}

void test_index_params_creation_functions(void) {
  TEST_START();

  // Test HNSW parameters using new API
  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(hnsw_params) == ZVEC_INDEX_TYPE_HNSW);
  // Default metric type is L2
  TEST_ASSERT(zvec_index_params_get_metric_type(hnsw_params) ==
              ZVEC_METRIC_TYPE_L2);

  int m, ef_construction;
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_COSINE);
  zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);
  m = zvec_index_params_get_hnsw_m(hnsw_params);
  ef_construction = zvec_index_params_get_hnsw_ef_construction(hnsw_params);
  TEST_ASSERT(m == 16);
  TEST_ASSERT(ef_construction == 100);

  // Test IVF parameters using new API
  zvec_index_params_t *ivf_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(ivf_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(ivf_params) == ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(zvec_index_params_get_metric_type(ivf_params) ==
              ZVEC_METRIC_TYPE_L2);

  int n_list, n_iters;
  bool use_soar;
  zvec_index_params_set_ivf_params(ivf_params, 100, 10, true);
  zvec_index_params_get_ivf_params(ivf_params, &n_list, &n_iters, &use_soar);
  TEST_ASSERT(n_list == 100);
  TEST_ASSERT(n_iters == 10);
  TEST_ASSERT(use_soar == true);

  // Test Flat parameters using new API
  zvec_index_params_t *flat_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
  TEST_ASSERT(flat_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(flat_params) == ZVEC_INDEX_TYPE_FLAT);
  zvec_index_params_set_metric_type(flat_params, ZVEC_METRIC_TYPE_IP);
  TEST_ASSERT(zvec_index_params_get_metric_type(flat_params) ==
              ZVEC_METRIC_TYPE_IP);

  // Test Invert parameters using new API
  zvec_index_params_t *invert_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(invert_params != NULL);
  TEST_ASSERT(zvec_index_params_get_type(invert_params) ==
              ZVEC_INDEX_TYPE_INVERT);
  bool enable_range_opt, enable_wildcard;
  zvec_index_params_set_invert_params(invert_params, true, false);
  zvec_index_params_get_invert_params(invert_params, &enable_range_opt,
                                      &enable_wildcard);
  TEST_ASSERT(enable_range_opt == true);
  TEST_ASSERT(enable_wildcard == false);

  // Cleanup
  zvec_index_params_destroy(hnsw_params);
  zvec_index_params_destroy(ivf_params);
  zvec_index_params_destroy(flat_params);
  zvec_index_params_destroy(invert_params);

  TEST_END();
}

void test_collection_advanced_index_functions(void) {
  TEST_START();

  const char *temp_dir = "./zvec_test_advanced_index";
  zvec_test_delete_dir(temp_dir);

  // Create schema
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("test_collection");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Add fields
    zvec_field_schema_t *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_field_schema_t *vec_field =
        zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
    zvec_collection_schema_add_field(schema, id_field);
    zvec_collection_schema_add_field(schema, vec_field);

    zvec_collection_options_t *options = zvec_collection_options_create();
    TEST_ASSERT(options != NULL);
    zvec_collection_t *collection = NULL;

    zvec_error_code_t err =
        zvec_collection_create_and_open(temp_dir, schema, options, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      // Test zvec_collection_create_index with FLAT type
      zvec_index_params_t *flat_params =
          zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
      TEST_ASSERT(flat_params != NULL);
      zvec_index_params_set_metric_type(flat_params, ZVEC_METRIC_TYPE_L2);
      err = zvec_collection_create_index(collection, "vec", flat_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Test zvec_collection_create_index with IVF type
      zvec_index_params_t *ivf_params =
          zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
      TEST_ASSERT(ivf_params != NULL);
      zvec_index_params_set_metric_type(ivf_params, ZVEC_METRIC_TYPE_L2);
      zvec_index_params_set_ivf_params(ivf_params, 100, 10, true);
      err = zvec_collection_drop_index(collection,
                                       "vec");  // Drop previous index first
      TEST_ASSERT(err == ZVEC_OK);
      err = zvec_collection_create_index(collection, "vec", ivf_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Test zvec_collection_create_index with HNSW type
      zvec_index_params_t *hnsw_params =
          zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
      TEST_ASSERT(hnsw_params != NULL);
      zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_COSINE);
      zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);
      err = zvec_collection_drop_index(collection,
                                       "vec");  // Drop previous index first
      TEST_ASSERT(err == ZVEC_OK);
      err = zvec_collection_create_index(collection, "vec", hnsw_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Test zvec_field_schema_set_index_params
      zvec_field_schema_t *new_vec_field = zvec_field_schema_create(
          "vec2", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
      TEST_ASSERT(new_vec_field != NULL);
      zvec_index_params_t *ivf_params2 =
          zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
      TEST_ASSERT(ivf_params2 != NULL);
      zvec_index_params_set_metric_type(ivf_params2, ZVEC_METRIC_TYPE_IP);
      zvec_index_params_set_ivf_params(ivf_params2, 50, 5, false);
      zvec_field_schema_set_index_params(new_vec_field, ivf_params2);
      TEST_ASSERT(zvec_field_schema_has_index(new_vec_field) == true);
      zvec_field_schema_destroy(new_vec_field);
      zvec_index_params_destroy(flat_params);
      zvec_index_params_destroy(ivf_params);
      zvec_index_params_destroy(hnsw_params);
      zvec_index_params_destroy(ivf_params2);

      zvec_collection_options_destroy(options);
      zvec_collection_destroy(collection);
    }
    zvec_collection_schema_destroy(schema);
  }

  zvec_test_delete_dir(temp_dir);
  TEST_END();
}

void test_collection_query_functions(void) {
  TEST_START();

  const char *temp_dir = "./zvec_test_query_funcs";
  zvec_test_delete_dir(temp_dir);

  // Create schema and collection
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("query_test");
  zvec_index_params_t *hnsw_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params != NULL);
  zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);

  zvec_field_schema_t *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_field_schema_t *vec_field =
      zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  zvec_field_schema_set_index_params(vec_field, hnsw_params);
  zvec_index_params_destroy(hnsw_params);

  zvec_collection_schema_add_field(schema, name_field);
  zvec_collection_schema_add_field(schema, vec_field);

  zvec_collection_t *collection = NULL;
  zvec_error_code_t err =
      zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
  TEST_ASSERT(err == ZVEC_OK);

  if (collection) {
    // Insert test documents
    zvec_doc_t *doc1 = zvec_doc_create();
    zvec_doc_set_pk(doc1, "doc1");
    float vec1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    zvec_doc_add_field_by_value(doc1, "vec", ZVEC_DATA_TYPE_VECTOR_FP32, vec1,
                                sizeof(vec1));
    zvec_doc_add_field_by_value(doc1, "name", ZVEC_DATA_TYPE_STRING,
                                "document1", 9);

    zvec_doc_t *doc2 = zvec_doc_create();
    zvec_doc_set_pk(doc2, "doc2");
    float vec2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    zvec_doc_add_field_by_value(doc2, "vec", ZVEC_DATA_TYPE_VECTOR_FP32, vec2,
                                sizeof(vec2));
    zvec_doc_add_field_by_value(doc2, "name", ZVEC_DATA_TYPE_STRING,
                                "document2", 9);

    zvec_doc_t *docs[] = {doc1, doc2};
    size_t success_count, error_count;
    err = zvec_collection_insert(collection, (const zvec_doc_t **)docs, 2,
                                 &success_count, &error_count);
    TEST_ASSERT(err == ZVEC_OK);

    zvec_collection_flush(collection);
    zvec_collection_optimize(collection);

    // Test zvec_collection_fetch
    const char *pks[] = {"doc1", "doc2"};
    zvec_doc_t **results = NULL;
    size_t found_count = 0;
    err = zvec_collection_fetch(collection, pks, 2, &results, &found_count);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(found_count == 2);
    zvec_docs_free(results, found_count);

    // Test zvec_collection_get_options
    zvec_collection_options_t *options = NULL;
    err = zvec_collection_get_options(collection, &options);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(options != NULL);
    zvec_collection_options_destroy(options);

    zvec_collection_destroy(collection);
    zvec_doc_destroy(doc1);
    zvec_doc_destroy(doc2);
  }

  zvec_collection_schema_destroy(schema);
  zvec_test_delete_dir(temp_dir);

  TEST_END();
}

void test_doc_advanced_functions(void) {
  TEST_START();

  // Test zvec_doc_clear
  zvec_doc_t *doc = zvec_doc_create();
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
  zvec_free((void *)pk_copy);

  // Test zvec_doc_is_empty
  zvec_doc_t *empty_doc = zvec_doc_create();
  TEST_ASSERT(zvec_doc_is_empty(empty_doc) == true);
  zvec_doc_add_field_by_value(empty_doc, "test", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){1}, sizeof(int32_t));
  TEST_ASSERT(zvec_doc_is_empty(empty_doc) == false);
  zvec_doc_destroy(empty_doc);

  // Test zvec_doc_memory_usage
  zvec_doc_t *mem_doc = zvec_doc_create();
  zvec_doc_set_pk(mem_doc, "memory_test");
  char large_data[1024];
  memset(large_data, 'A', sizeof(large_data));
  zvec_doc_add_field_by_value(mem_doc, "large_field", ZVEC_DATA_TYPE_STRING,
                              large_data, sizeof(large_data));
  size_t mem_usage = zvec_doc_memory_usage(mem_doc);
  TEST_ASSERT(mem_usage > 0);
  zvec_doc_destroy(mem_doc);

  // Test zvec_doc_merge
  zvec_doc_t *doc1 = zvec_doc_create();
  zvec_doc_set_pk(doc1, "merge_test");
  zvec_doc_add_field_by_value(doc1, "field1", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){100}, sizeof(int32_t));

  zvec_doc_t *doc2 = zvec_doc_create();
  zvec_doc_add_field_by_value(doc2, "field2", ZVEC_DATA_TYPE_STRING, "merged",
                              6);

  zvec_doc_merge(doc1, doc2);
  TEST_ASSERT(zvec_doc_has_field(doc1, "field1") == true);
  TEST_ASSERT(zvec_doc_has_field(doc1, "field2") == true);

  zvec_doc_destroy(doc1);
  zvec_doc_destroy(doc2);

  // Test zvec_doc_validate
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("validate_test");
  zvec_field_schema_t *val_field =
      zvec_field_schema_create("test_field", ZVEC_DATA_TYPE_INT32, false, 0);
  zvec_collection_schema_add_field(schema, val_field);

  zvec_doc_t *val_doc = zvec_doc_create();
  zvec_doc_set_pk(val_doc, "test_pk");
  zvec_doc_add_field_by_value(val_doc, "test_field", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){42}, sizeof(int32_t));

  char *error_msg = NULL;
  zvec_error_code_t err =
      zvec_doc_validate_and_sanitize(val_doc, schema, false, &error_msg);
  TEST_ASSERT(err == ZVEC_OK);
  if (error_msg) {
    zvec_free(error_msg);
  }

  zvec_doc_destroy(val_doc);
  zvec_collection_schema_destroy(schema);
  zvec_doc_destroy(doc);

  // Test zvec_doc_to_detail_string
  zvec_doc_t *detail_doc = zvec_doc_create();
  zvec_doc_set_pk(detail_doc, "detail_test");
  zvec_doc_add_field_by_value(detail_doc, "int_field", ZVEC_DATA_TYPE_INT32,
                              &(int32_t){12345}, sizeof(int32_t));
  zvec_doc_add_field_by_value(detail_doc, "str_field", ZVEC_DATA_TYPE_STRING,
                              "hello", 5);

  char *detail_str = NULL;
  err = zvec_doc_to_detail_string(detail_doc, &detail_str);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(detail_str != NULL);
  // printf("  Document detail: %s\n", detail_str);
  zvec_free(detail_str);

  zvec_doc_destroy(detail_doc);

  TEST_END();
}

void test_array_memory_functions(void) {
  TEST_START();

  // Test zvec_string_array_t
  zvec_string_array_t *str_array = zvec_string_array_create(3);
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

  // Test zvec_mutable_byte_array_t
  zvec_mutable_byte_array_t *byte_array = zvec_byte_array_create(1024);
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

  // Test zvec_float_array_t
  zvec_float_array_t *float_array = zvec_float_array_create(10);
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

  // Test zvec_int64_array_t
  zvec_int64_array_t *int64_array = zvec_int64_array_create(5);
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
  zvec_mutable_byte_array_t *zero_array = zvec_byte_array_create(0);
  TEST_ASSERT(zero_array != NULL);
  if (zero_array) {
    zvec_byte_array_destroy(zero_array);
  }

  TEST_END();
}

// =============================================================================
// Missing API coverage tests
// =============================================================================

void test_collection_open_close(void) {
  TEST_START();

  const char *temp_dir = "./zvec_test_open_close";
  zvec_test_delete_dir(temp_dir);

  // First create a collection
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("open_close_test");
  TEST_ASSERT(schema != NULL);

  zvec_field_schema_t *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT32, false, 0);
  zvec_field_schema_t *vec_field =
      zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  zvec_collection_schema_add_field(schema, id_field);
  zvec_collection_schema_add_field(schema, vec_field);

  zvec_collection_options_t *options = zvec_collection_options_create();

  zvec_collection_t *collection = NULL;
  zvec_error_code_t err =
      zvec_collection_create_and_open(temp_dir, schema, options, &collection);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(collection != NULL);

  // Insert some data before closing
  if (collection) {
    zvec_doc_t *doc = zvec_doc_create();
    zvec_doc_set_pk(doc, "doc1");
    int32_t id_val = 1;
    float vec_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    zvec_doc_add_field_by_value(doc, "id", ZVEC_DATA_TYPE_INT32, &id_val,
                                sizeof(int32_t));
    zvec_doc_add_field_by_value(doc, "vec", ZVEC_DATA_TYPE_VECTOR_FP32,
                                vec_data, sizeof(vec_data));
    const zvec_doc_t *docs[] = {doc};
    size_t success_count = 0;
    size_t error_count = 0;
    err = zvec_collection_insert(collection, docs, 1, &success_count,
                                 &error_count);
    if (err != ZVEC_OK || success_count != 1) {
      // Print error for debugging
      char *error_msg = NULL;
      zvec_get_last_error(&error_msg);
      printf("  Insert error: %s (err=%d, success=%zu)\n",
             error_msg ? error_msg : "unknown", err, success_count);
      if (error_msg) zvec_free(error_msg);
    }
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(success_count == 1);
    zvec_doc_destroy(doc);

    // Close the collection
    err = zvec_collection_close(collection);
    TEST_ASSERT(err == ZVEC_OK);

    // Re-open the existing collection using zvec_collection_open
    zvec_collection_t *reopened_collection = NULL;
    err = zvec_collection_open(temp_dir, options, &reopened_collection);
    if (err != ZVEC_OK) {
      char *error_msg = NULL;
      zvec_get_last_error(&error_msg);
      printf("  Open error: %s (err=%d)\n", error_msg ? error_msg : "unknown",
             err);
      if (error_msg) zvec_free(error_msg);
    }
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(reopened_collection != NULL);

    // Verify data is still accessible after re-open
    if (reopened_collection) {
      // Use query API instead of filter
      zvec_vector_query_t *query = zvec_vector_query_create();
      zvec_vector_query_set_topk(query, 10);
      zvec_vector_query_set_field_name(query, "vec");
      float query_vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
      zvec_vector_query_set_query_vector(query, query_vec, sizeof(query_vec));

      zvec_doc_t **results = NULL;
      size_t result_count = 0;
      err = zvec_collection_query(reopened_collection, query, &results,
                                  &result_count);
      if (err != ZVEC_OK || result_count < 1) {
        char *error_msg = NULL;
        zvec_get_last_error(&error_msg);
        printf("  Query error: %s (err=%d, count=%zu)\n",
               error_msg ? error_msg : "unknown", err, result_count);
        if (error_msg) zvec_free(error_msg);
      }
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(result_count >= 1);
      if (result_count > 0) {
        const char *pk = zvec_doc_get_pk_copy(results[0]);
        TEST_ASSERT(pk != NULL);
        TEST_ASSERT(strcmp(pk, "doc1") == 0);
        zvec_free((void *)pk);
        for (size_t i = 0; i < result_count; i++) {
          zvec_doc_destroy(results[i]);
        }
        zvec_free(results);
      }
      zvec_vector_query_destroy(query);
    }

    // Close the reopened collection
    if (reopened_collection) {
      err = zvec_collection_close(reopened_collection);
      TEST_ASSERT(err == ZVEC_OK);
    }
  }

  // Test NULL pointer error handling
  err = zvec_collection_open(NULL, options, &collection);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_open(temp_dir, options, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_close(NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  zvec_collection_options_destroy(options);
  zvec_collection_schema_destroy(schema);
  zvec_test_delete_dir(temp_dir);

  TEST_END();
}

void test_collection_options_getters(void) {
  TEST_START();

  // Test default values
  zvec_collection_options_t *options = zvec_collection_options_create();
  TEST_ASSERT(options != NULL);

  // Test enable_mmap getter
  bool enable_mmap = zvec_collection_options_get_enable_mmap(options);
  TEST_ASSERT(enable_mmap == true || enable_mmap == false);

  // Set and verify
  zvec_error_code_t err =
      zvec_collection_options_set_enable_mmap(options, false);
  TEST_ASSERT(err == ZVEC_OK);
  enable_mmap = zvec_collection_options_get_enable_mmap(options);
  TEST_ASSERT(enable_mmap == false);

  // Test max_buffer_size getter
  size_t max_buffer_size = zvec_collection_options_get_max_buffer_size(options);
  TEST_ASSERT(max_buffer_size > 0);  // Should have a default value

  // Set and verify
  err = zvec_collection_options_set_max_buffer_size(options, 1024 * 1024);
  TEST_ASSERT(err == ZVEC_OK);
  max_buffer_size = zvec_collection_options_get_max_buffer_size(options);
  TEST_ASSERT(max_buffer_size == 1024 * 1024);

  // Test NULL pointer handling - these return defaults, not 0
  TEST_ASSERT(zvec_collection_options_get_enable_mmap(NULL) ==
              true);  // Default is true
  TEST_ASSERT(zvec_collection_options_get_max_buffer_size(NULL) >
              0);  // Default is non-zero

  zvec_collection_options_destroy(options);

  TEST_END();
}

void test_collection_stats_index_info(void) {
  TEST_START();

  const char *temp_dir = "./zvec_test_stats_index";
  zvec_test_delete_dir(temp_dir);

  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("stats_index_test");
  zvec_field_schema_t *vec_field =
      zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  zvec_collection_schema_add_field(schema, vec_field);

  zvec_collection_t *collection = NULL;
  zvec_error_code_t err =
      zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
  TEST_ASSERT(err == ZVEC_OK);

  if (collection) {
    // Create an index first
    zvec_index_params_t *hnsw_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
    zvec_index_params_set_metric_type(hnsw_params, ZVEC_METRIC_TYPE_L2);
    zvec_index_params_set_hnsw_params(hnsw_params, 16, 100);
    err = zvec_collection_create_index(collection, "vec", hnsw_params);
    TEST_ASSERT(err == ZVEC_OK);
    zvec_index_params_destroy(hnsw_params);

    // Get collection stats
    zvec_collection_stats_t *stats = NULL;
    err = zvec_collection_get_stats(collection, &stats);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(stats != NULL);

    // Test index count
    size_t index_count = zvec_collection_stats_get_index_count(stats);
    TEST_ASSERT(index_count >= 1);  // Should have at least one index

    // Test index name getter
    const char *index_name = zvec_collection_stats_get_index_name(stats, 0);
    TEST_ASSERT(index_name != NULL);
    // printf("  Index name at 0: %s\n", index_name);

    // Test index completeness getter
    float completeness = zvec_collection_stats_get_index_completeness(stats, 0);
    TEST_ASSERT(completeness >= 0.0f && completeness <= 1.0f);
    // printf("  Index completeness at 0: %.2f\n", completeness);

    // Test out-of-bounds access
    index_name = zvec_collection_stats_get_index_name(stats, 999);
    TEST_ASSERT(index_name == NULL);

    completeness = zvec_collection_stats_get_index_completeness(stats, 999);
    TEST_ASSERT(completeness == 0.0f);

    // Test NULL pointer handling
    TEST_ASSERT(zvec_collection_stats_get_index_name(NULL, 0) == NULL);
    TEST_ASSERT(zvec_collection_stats_get_index_completeness(NULL, 0) == 0.0f);

    zvec_collection_stats_destroy(stats);
  }

  zvec_collection_destroy(collection);
  zvec_collection_schema_destroy(schema);
  zvec_test_delete_dir(temp_dir);

  TEST_END();
}

void test_field_schema_validate(void) {
  TEST_START();

  // Test valid field schema
  zvec_field_schema_t *valid_field =
      zvec_field_schema_create("valid_field", ZVEC_DATA_TYPE_INT32, false, 0);
  TEST_ASSERT(valid_field != NULL);

  zvec_string_t *error_msg = NULL;
  zvec_error_code_t err = zvec_field_schema_validate(valid_field, &error_msg);
  TEST_ASSERT(err == ZVEC_OK);
  if (error_msg) {
    zvec_free_string(error_msg);
  }

  // Test field with index params
  zvec_field_schema_t *field_with_index = zvec_field_schema_create(
      "vec_field", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  zvec_index_params_t *index_params =
      zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  zvec_index_params_set_metric_type(index_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(index_params, 16, 100);
  zvec_field_schema_set_index_params(field_with_index, index_params);

  err = zvec_field_schema_validate(field_with_index, &error_msg);
  TEST_ASSERT(err == ZVEC_OK);
  if (error_msg) {
    zvec_free_string(error_msg);
  }

  // Test NULL schema pointer
  err = zvec_field_schema_validate(NULL, &error_msg);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  if (error_msg) {
    zvec_free_string(error_msg);
  }

  // Test with NULL error_msg (should not crash)
  err = zvec_field_schema_validate(valid_field, NULL);
  TEST_ASSERT(err == ZVEC_OK);

  zvec_index_params_destroy(index_params);
  zvec_field_schema_destroy(field_with_index);
  zvec_field_schema_destroy(valid_field);

  TEST_END();
}

void test_doc_remove_field(void) {
  TEST_START();

  zvec_doc_t *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  // Add some fields
  zvec_doc_set_pk(doc, "test_pk");
  int32_t val1 = 100;
  int32_t val2 = 200;
  zvec_doc_add_field_by_value(doc, "field1", ZVEC_DATA_TYPE_INT32, &val1,
                              sizeof(int32_t));
  zvec_doc_add_field_by_value(doc, "field2", ZVEC_DATA_TYPE_INT32, &val2,
                              sizeof(int32_t));

  // Verify fields exist
  TEST_ASSERT(zvec_doc_has_field(doc, "field1") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "field2") == true);
  TEST_ASSERT(zvec_doc_get_field_count(doc) == 2);

  // Remove field1
  zvec_error_code_t err = zvec_doc_remove_field(doc, "field1");
  TEST_ASSERT(err == ZVEC_OK);

  // Verify field1 is removed
  TEST_ASSERT(zvec_doc_has_field(doc, "field1") == false);
  TEST_ASSERT(zvec_doc_has_field(doc, "field2") == true);
  TEST_ASSERT(zvec_doc_get_field_count(doc) == 1);

  // Try to remove non-existent field (should not crash)
  err = zvec_doc_remove_field(doc, "non_existent_field");
  TEST_ASSERT(err == ZVEC_OK);  // Or might return error, depending on design

  // Test NULL pointer handling
  err = zvec_doc_remove_field(NULL, "field");
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_doc_remove_field(doc, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Clean up
  zvec_doc_destroy(doc);

  TEST_END();
}

void test_collection_schema_getters(void) {
  TEST_START();

  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("schema_getter_test");
  TEST_ASSERT(schema != NULL);

  // Add multiple fields
  zvec_field_schema_t *field1 =
      zvec_field_schema_create("field1", ZVEC_DATA_TYPE_INT32, false, 0);
  zvec_field_schema_t *field2 =
      zvec_field_schema_create("field2", ZVEC_DATA_TYPE_STRING, true, 0);
  zvec_field_schema_t *field3 = zvec_field_schema_create(
      "field3", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);

  zvec_collection_schema_add_field(schema, field1);
  zvec_collection_schema_add_field(schema, field2);
  zvec_collection_schema_add_field(schema, field3);

  // Test has_field
  TEST_ASSERT(zvec_collection_schema_has_field(schema, "field1") == true);
  TEST_ASSERT(zvec_collection_schema_has_field(schema, "field2") == true);
  TEST_ASSERT(zvec_collection_schema_has_field(schema, "field3") == true);
  TEST_ASSERT(zvec_collection_schema_has_field(schema, "non_existent") ==
              false);

  // Test get_field by name
  zvec_field_schema_t *retrieved_field =
      zvec_collection_schema_get_field(schema, "field2");
  TEST_ASSERT(retrieved_field != NULL);
  TEST_ASSERT(zvec_field_schema_get_data_type(retrieved_field) ==
              ZVEC_DATA_TYPE_STRING);

  // Test non-existent field name
  retrieved_field = zvec_collection_schema_get_field(schema, "non_existent");
  TEST_ASSERT(retrieved_field == NULL);

  // Test get_forward_field (scalar fields only)
  retrieved_field = zvec_collection_schema_get_forward_field(schema, "field1");
  TEST_ASSERT(retrieved_field != NULL);

  retrieved_field = zvec_collection_schema_get_forward_field(schema, "field3");
  TEST_ASSERT(retrieved_field == NULL);  // field3 is vector, not scalar

  // Test get_vector_field
  retrieved_field = zvec_collection_schema_get_vector_field(schema, "field3");
  TEST_ASSERT(retrieved_field != NULL);
  TEST_ASSERT(zvec_field_schema_get_data_type(retrieved_field) ==
              ZVEC_DATA_TYPE_VECTOR_FP32);

  retrieved_field = zvec_collection_schema_get_vector_field(schema, "field1");
  TEST_ASSERT(retrieved_field == NULL);  // field1 is scalar, not vector

  // Test get_all_field_names
  const char **field_names = NULL;
  size_t field_count = 0;
  zvec_error_code_t err = zvec_collection_schema_get_all_field_names(
      schema, &field_names, &field_count);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(field_names != NULL);
  TEST_ASSERT(field_count == 3);
  printf("  Field names: ");
  for (size_t i = 0; i < field_count; i++) {
    printf("%s ", field_names[i]);
  }
  printf("\n");
  zvec_free(field_names);

  // Test get_forward_fields
  zvec_field_schema_t **forward_fields = NULL;
  size_t forward_count = 0;
  err = zvec_collection_schema_get_forward_fields(schema, &forward_fields,
                                                  &forward_count);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(forward_fields != NULL);
  TEST_ASSERT(forward_count == 2);  // field1 and field2 are scalars
  // Note: forward_fields contains pointers to fields owned by schema,
  // do not destroy them individually - just free the array
  zvec_free(forward_fields);

  // Test get_vector_fields
  zvec_field_schema_t **vector_fields = NULL;
  size_t vector_count = 0;
  err = zvec_collection_schema_get_vector_fields(schema, &vector_fields,
                                                 &vector_count);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(vector_fields != NULL);
  TEST_ASSERT(vector_count == 1);  // Only field3 is vector
  // Note: vector_fields contains pointers to fields owned by schema,
  // do not destroy them individually - just free the array
  zvec_free(vector_fields);

  // Test NULL pointer handling
  TEST_ASSERT(zvec_collection_schema_has_field(NULL, "field") == false);
  TEST_ASSERT(zvec_collection_schema_get_field(NULL, "field") == NULL);
  TEST_ASSERT(zvec_collection_schema_get_forward_field(NULL, "field") == NULL);
  TEST_ASSERT(zvec_collection_schema_get_vector_field(NULL, "field") == NULL);

  // Destroy schema - it owns all fields added to it
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

// =============================================================================
// Main function
// =============================================================================

int main(void) {
  printf("Starting comprehensive C API tests...\n\n");

  // Clean up previous test directories
  printf("Cleaning up previous test directories...\n");
#ifdef _WIN32
  system("rmdir /s /q %TEMP%\\zvec_test_* 2>nul");
  system("del /q %TEMP%\\zvec_test_* 2>nul");
#else
  {
    glob_t gl;
    if (glob("/tmp/zvec_test_*", 0, NULL, &gl) == 0) {
      for (size_t gi = 0; gi < gl.gl_pathc; gi++) {
        zvec_test_delete_dir(gl.gl_pathv[gi]);
      }
      globfree(&gl);
    }
  }
#endif
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
  test_collection_schema_helpers();
  test_collection_schema_alter_field();

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
  test_collection_nullable_roundtrip();
  test_collection_ddl_operations();

  // Doc-related tests
  test_doc_creation();
  test_doc_primary_key();
  test_doc_basic_operations();
  test_doc_null_field_api();
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
  test_index_params_api_functions();
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

  // Missing API coverage tests (before shutdown)
  test_collection_open_close();
  test_collection_options_getters();
  test_collection_stats_index_info();
  test_field_schema_validate();
  test_doc_remove_field();
  test_collection_schema_getters();

  // Additional API coverage tests
  test_zvec_shutdown();
  test_index_params_creation_functions();
  test_collection_advanced_index_functions();
  test_collection_query_functions();
  test_doc_advanced_functions();
  test_array_memory_functions();

  printf("\n=== Comprehensive Test Summary ===\n");
  printf("Total tests: %d\n", test_count);
  printf("Passed: %d\n", passed_count);
  printf("Failed: %d\n", test_count - passed_count);

  return test_count == passed_count ? 0 : 1;
}
