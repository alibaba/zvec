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

#ifndef ZVEC_C_API_H
#define ZVEC_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Opaque Handles ---
typedef struct zvec_status zvec_status_t;
typedef struct zvec_collection zvec_collection_t;
typedef struct zvec_schema zvec_schema_t;
typedef struct zvec_field_schema zvec_field_schema_t;
typedef struct zvec_doc zvec_doc_t;
typedef struct zvec_doc_list zvec_doc_list_t;
typedef struct zvec_index_params zvec_index_params_t;
typedef struct zvec_stats zvec_stats_t;

// --- Enums ---
typedef enum {
  ZVEC_OK = 0,
  ZVEC_NOT_FOUND,
  ZVEC_ALREADY_EXISTS,
  ZVEC_INVALID_ARGUMENT,
  ZVEC_PERMISSION_DENIED,
  ZVEC_FAILED_PRECONDITION,
  ZVEC_RESOURCE_EXHAUSTED,
  ZVEC_UNAVAILABLE,
  ZVEC_INTERNAL_ERROR,
  ZVEC_NOT_SUPPORTED,
  ZVEC_UNKNOWN
} zvec_status_code_t;

typedef enum {
  ZVEC_TYPE_UNDEFINED = 0,
  ZVEC_TYPE_BINARY = 1,
  ZVEC_TYPE_STRING = 2,
  ZVEC_TYPE_BOOL = 3,
  ZVEC_TYPE_INT32 = 4,
  ZVEC_TYPE_INT64 = 5,
  ZVEC_TYPE_UINT32 = 6,
  ZVEC_TYPE_UINT64 = 7,
  ZVEC_TYPE_FLOAT = 8,
  ZVEC_TYPE_DOUBLE = 9,
  ZVEC_TYPE_VECTOR_BINARY32 = 20,
  ZVEC_TYPE_VECTOR_FP16 = 22,
  ZVEC_TYPE_VECTOR_FP32 = 23,
  ZVEC_TYPE_VECTOR_FP64 = 24,
  ZVEC_TYPE_VECTOR_INT8 = 26,
  ZVEC_TYPE_SPARSE_VECTOR_FP16 = 30,
  ZVEC_TYPE_SPARSE_VECTOR_FP32 = 31,
  ZVEC_TYPE_ARRAY_STRING = 41,
  ZVEC_TYPE_ARRAY_INT32 = 43,
  ZVEC_TYPE_ARRAY_FLOAT = 47,
} zvec_data_type_t;

// --- Status API ---
zvec_status_code_t zvec_status_code(zvec_status_t *status);
const char *zvec_status_message(zvec_status_t *status);
void zvec_status_destroy(zvec_status_t *status);

// --- Schema API ---
zvec_schema_t *zvec_schema_create(const char *name);
void zvec_schema_destroy(zvec_schema_t *schema);
zvec_status_t *zvec_schema_add_field(zvec_schema_t *schema, const char *name,
                                     zvec_data_type_t type, uint32_t dimension);

// --- Doc API ---
zvec_doc_t *zvec_doc_create();
void zvec_doc_destroy(zvec_doc_t *doc);
void zvec_doc_set_pk(zvec_doc_t *doc, const char *pk);
const char *zvec_doc_pk(zvec_doc_t *doc);
void zvec_doc_set_score(zvec_doc_t *doc, float score);
float zvec_doc_score(zvec_doc_t *doc);

zvec_status_t *zvec_doc_set_string(zvec_doc_t *doc, const char *field,
                                   const char *value);
zvec_status_t *zvec_doc_set_int32(zvec_doc_t *doc, const char *field,
                                  int32_t value);
zvec_status_t *zvec_doc_set_float(zvec_doc_t *doc, const char *field,
                                  float value);
zvec_status_t *zvec_doc_set_float_vector(zvec_doc_t *doc, const char *field,
                                         const float *data, uint32_t count);

// --- Collection API ---
zvec_status_t *zvec_collection_create_and_open(
    const char *path, zvec_schema_t *schema,
    zvec_collection_t **out_collection);
zvec_status_t *zvec_collection_open(const char *path,
                                    zvec_collection_t **out_collection);
void zvec_collection_destroy(zvec_collection_t *collection);

// DDL
zvec_status_t *zvec_collection_flush(zvec_collection_t *collection);
zvec_status_t *zvec_collection_destroy_physical(zvec_collection_t *collection);
zvec_status_t *zvec_collection_get_stats(zvec_collection_t *collection,
                                         zvec_stats_t **out_stats);

// DML
zvec_status_t *zvec_collection_insert(zvec_collection_t *collection,
                                      zvec_doc_t **docs, size_t count);
zvec_status_t *zvec_collection_upsert(zvec_collection_t *collection,
                                      zvec_doc_t **docs, size_t count);
zvec_status_t *zvec_collection_update(zvec_collection_t *collection,
                                      zvec_doc_t **docs, size_t count);
zvec_status_t *zvec_collection_delete(zvec_collection_t *collection,
                                      const char **pks, size_t count);

// DQL
zvec_status_t *zvec_collection_query(zvec_collection_t *collection,
                                     const char *field_name,
                                     const float *vector, uint32_t count,
                                     int topk, zvec_doc_list_t **out_results);
zvec_status_t *zvec_collection_fetch(zvec_collection_t *collection,
                                     const char **pks, size_t count,
                                     zvec_doc_list_t **out_results);

// --- Doc List API ---
size_t zvec_doc_list_size(zvec_doc_list_t *list);
zvec_doc_t *zvec_doc_list_get(zvec_doc_list_t *list, size_t index);
void zvec_doc_list_destroy(zvec_doc_list_t *list);

// --- Stats API ---
uint64_t zvec_stats_total_docs(zvec_stats_t *stats);
void zvec_stats_destroy(zvec_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif  // ZVEC_C_API_H
