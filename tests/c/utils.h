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

#ifndef ZVEC_TESTS_C_API_UTILS_H
#define ZVEC_TESTS_C_API_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "zvec/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Schema Creation Helper Functions
// =============================================================================

/**
 * @brief Create temporary test schema
 * Contains basic scalar fields and vector fields
 *
 * @return zvec_collection_schema_t* Created schema pointer, needs to be
 * released by calling zvec_collection_schema_cleanup
 */
zvec_collection_schema_t *zvec_test_create_temp_schema(void);

/**
 * @brief Create pure scalar schema
 * Contains only scalar fields (int32, string)
 *
 * @return zvec_collection_schema_t* Created schema pointer
 */
zvec_collection_schema_t *zvec_test_create_scalar_schema(void);

/**
 * @brief Create full-featured schema
 * Contains all supported data type fields
 *
 * @param nullable Whether to allow null values
 * @param name Schema name
 * @param scalar_index_params Scalar index parameters (can be NULL)
 * @param vector_index_params Vector index parameters (can be NULL)
 * @param max_doc_count Maximum documents per segment
 * @return zvec_collection_schema_t* Created schema pointer
 */
zvec_collection_schema_t *zvec_test_create_normal_schema(
    bool nullable, const char *name,
    const zvec_index_params_t *scalar_index_params,
    const zvec_index_params_t *vector_index_params, uint64_t max_doc_count);

/**
 * @brief Create schema with scalar index
 *
 * @param nullable Whether to allow null values
 * @param enable_optimize Whether to enable optimization
 * @param name Schema name
 * @return zvec_collection_schema_t* Created schema pointer
 */
zvec_collection_schema_t *zvec_test_create_schema_with_scalar_index(
    bool nullable, bool enable_optimize, const char *name);

/**
 * @brief Create schema with vector index
 *
 * @param nullable Whether to allow null values
 * @param name Schema name
 * @param vector_index_params Vector index parameters (can be NULL, uses default
 * HNSW parameters)
 * @return zvec_collection_schema_t* Created schema pointer
 */
zvec_collection_schema_t *zvec_test_create_schema_with_vector_index(
    bool nullable, const char *name,
    const zvec_index_params_t *vector_index_params);

/**
 * @brief Create schema with specified maximum document count
 *
 * @param doc_count Maximum documents per segment
 * @return zvec_collection_schema_t* Created schema pointer
 */
zvec_collection_schema_t *zvec_test_create_schema_with_max_doc_count(
    uint64_t doc_count);

// =============================================================================
// Document Creation Helper Functions
// =============================================================================

/**
 * @brief Generate primary key based on document ID
 *
 * @param doc_id Document ID
 * @return char* Generated primary key string, needs to be released by calling
 * free()
 */
char *zvec_test_make_pk(uint64_t doc_id);

/**
 * @brief Create complete document
 * Create corresponding test data for each field according to schema
 *
 * @param doc_id Document ID
 * @param schema Schema pointer
 * @param pk Primary key (can be NULL, auto-generated)
 * @return zvec_doc_t* Created document pointer, needs to be released by calling
 * zvec_doc_destroy
 */
zvec_doc_t *zvec_test_create_doc(uint64_t doc_id,
                                 const zvec_collection_schema_t *schema,
                                 const char *pk);

/**
 * @brief Create partial null document
 * Only set values for vector fields, keep scalar fields as null
 *
 * @param doc_id Document ID
 * @param schema Schema pointer
 * @param pk Primary key (can be NULL, auto-generated)
 * @return zvec_doc_t* Created document pointer
 */
zvec_doc_t *zvec_test_create_doc_null(uint64_t doc_id,
                                      const zvec_collection_schema_t *schema,
                                      const char *pk);

/**
 * @brief Create document with specified fields
 * Only create data for specified fields
 *
 * @param doc_id Document ID
 * @param field_names Field name array
 * @param field_types Field type array
 * @param field_count Number of fields
 * @param pk Primary key (can be NULL, auto-generated)
 * @return zvec_doc_t* Created document pointer
 */
zvec_doc_t *zvec_test_create_doc_with_fields(
    uint64_t doc_id, const char **field_names,
    const zvec_data_type_t *field_types, size_t field_count, const char *pk);

// =============================================================================
// Index Parameter Creation Helper Functions
// =============================================================================

/**
 * @brief Create default HNSW index parameters
 *
 * @return zvec_index_params_t* Created parameter pointer
 */
zvec_index_params_t *zvec_test_create_default_hnsw_params(void);

/**
 * @brief Create default Flat index parameters
 *
 * @return zvec_index_params_t* Created parameter pointer
 */
zvec_index_params_t *zvec_test_create_default_flat_params(void);

/**
 * @brief Create default scalar index parameters
 *
 * @param enable_optimize Whether to enable optimization
 * @return zvec_index_params_t* Created parameter pointer
 */
zvec_index_params_t *zvec_test_create_default_invert_params(
    bool enable_optimize);

// =============================================================================
// Field Schema Creation Helper Functions
// =============================================================================

/**
 * @brief Create scalar field schema
 *
 * @param name Field name
 * @param data_type Data type
 * @param nullable Whether to allow null values
 * @param invert_params Scalar index parameters (can be NULL)
 * @return zvec_field_schema_t* Created field schema pointer, needs to be
 * released by calling free()
 */
zvec_field_schema_t *zvec_test_create_scalar_field(
    const char *name, zvec_data_type_t data_type, bool nullable,
    const zvec_index_params_t *invert_params);

/**
 * @brief Create vector field schema
 *
 * @param name Field name
 * @param data_type Data type
 * @param dimension Vector dimension
 * @param nullable Whether to allow null values
 * @param vector_index_params Vector index parameters (can be NULL)
 * @return zvec_field_schema_t* Created field schema pointer
 */
zvec_field_schema_t *zvec_test_create_vector_field(
    const char *name, zvec_data_type_t data_type, uint32_t dimension,
    bool nullable, const zvec_index_params_t *vector_index_params);

/**
 * @brief Create sparse vector field schema
 *
 * @param name Field name
 * @param data_type Data type
 * @param nullable Whether to allow null values
 * @param vector_index_params Vector index parameters (can be NULL)
 * @return zvec_field_schema_t* Created field schema pointer
 */
zvec_field_schema_t *zvec_test_create_sparse_vector_field(
    const char *name, zvec_data_type_t data_type, bool nullable,
    const zvec_index_params_t *vector_index_params);

// =============================================================================
// Memory Management Helper Functions
// =============================================================================

/**
 * @brief Free string array
 *
 * @param strings String array pointer
 * @param count Number of strings
 */
void zvec_test_free_strings(char **strings, size_t count);

/**
 * @brief Delete directory and all its contents
 *
 * @param dir_path Directory path
 * @return int 0 for success, -1 for failure
 */
int zvec_test_delete_dir(const char *dir_path);

#ifdef __cplusplus
}
#endif

#endif  // ZVEC_TESTS_C_API_UTILS_H