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
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** Opaque handle to an open Collection. */
typedef void *zvec_collection_t;

/**
 * Status returned from every API call.
 * When code == 0 (ZVEC_OK), message is NULL.
 * When code != 0, message is a heap-allocated C string.
 * Call zvec_free_string(status.message) after reading the error.
 */
typedef struct {
  int code;
  const char *message;
} zvec_status_t;

/* Status codes (mirror zvec::StatusCode) */
#define ZVEC_OK 0
#define ZVEC_NOT_FOUND 1
#define ZVEC_ALREADY_EXISTS 2
#define ZVEC_INVALID_ARGUMENT 3
#define ZVEC_PERMISSION_DENIED 4
#define ZVEC_FAILED_PRECONDITION 5
#define ZVEC_RESOURCE_EXHAUSTED 6
#define ZVEC_UNAVAILABLE 7
#define ZVEC_INTERNAL_ERROR 8
#define ZVEC_NOT_SUPPORTED 9
#define ZVEC_UNKNOWN 10

/* ── Lifecycle ───────────────────────────────────────────────────────────────
 *
 * schema_json format:
 * {
 *   "name": "my_collection",
 *   "fields": [
 *     { "name": "embedding", "data_type": "VECTOR_FP32", "dimension": 128,
 *       "nullable": false,
 *       "index": { "type": "HNSW", "metric": "L2", "m": 16,
 *                  "ef_construction": 200 } },
 *     { "name": "title", "data_type": "STRING", "nullable": false,
 *       "index": { "type": "INVERT" } },
 *     { "name": "score", "data_type": "FLOAT", "nullable": true }
 *   ]
 * }
 *
 * Supported data_type values: BINARY, STRING, BOOL, INT32, INT64, UINT32,
 *   UINT64, FLOAT, DOUBLE, VECTOR_FP16, VECTOR_FP32, VECTOR_FP64,
 *   VECTOR_INT4, VECTOR_INT8, VECTOR_INT16, SPARSE_VECTOR_FP16,
 *   SPARSE_VECTOR_FP32, ARRAY_BOOL, ARRAY_INT32, ARRAY_INT64, ARRAY_UINT32,
 *   ARRAY_UINT64, ARRAY_FLOAT, ARRAY_DOUBLE, ARRAY_STRING
 *
 * Supported index types: HNSW, IVF, FLAT, INVERT
 * Supported metrics:     L2, IP, COSINE, MIPSL2
 * Supported quantize:    FP16, INT8, INT4  (optional field)
 */
zvec_status_t zvec_create_and_open(const char *path, const char *schema_json,
                                   zvec_collection_t *out);

zvec_status_t zvec_open(const char *path, zvec_collection_t *out);

/**
 * Open an existing collection in read-only mode.
 *
 * Multiple read-only handles can coexist with one active writer.
 * Write APIs invoked through this handle will return read-only errors.
 */
zvec_status_t zvec_open_read_only(const char *path, zvec_collection_t *out);

/** Release the in-memory handle without deleting on-disk data. */
void zvec_collection_free(zvec_collection_t col);

/** Permanently delete the collection from disk, then free the handle. */
zvec_status_t zvec_collection_destroy(zvec_collection_t col);

zvec_status_t zvec_collection_flush(zvec_collection_t col);

/* ── DML ─────────────────────────────────────────────────────────────────────
 *
 * docs_json format (array of doc objects):
 * [
 *   { "pk": "doc_1",
 *     "fields": {
 *       "embedding": [0.1, 0.2, 0.3, 0.4],
 *       "title": "hello",
 *       "score": 9.5,
 *       "sparse_field": { "indices": [0, 5], "values": [0.3, 0.7] }
 *     }
 *   }
 * ]
 *
 * results_json output (must be freed with zvec_free_string):
 * [ { "pk": "doc_1", "code": 0, "message": "" }, ... ]
 */
zvec_status_t zvec_insert(zvec_collection_t col, const char *docs_json,
                          char **results_json);

zvec_status_t zvec_upsert(zvec_collection_t col, const char *docs_json,
                          char **results_json);

zvec_status_t zvec_update(zvec_collection_t col, const char *docs_json,
                          char **results_json);

/**
 * pks_json: JSON array of string PKs, e.g. ["doc_1", "doc_2"]
 * results_json output: [ { "pk": "doc_1", "code": 0, "message": "" }, ... ]
 */
zvec_status_t zvec_delete_by_pks(zvec_collection_t col, const char *pks_json,
                                 char **results_json);

/** filter: SQL-like filter expression, e.g. "category = 'tech'" */
zvec_status_t zvec_delete_by_filter(zvec_collection_t col, const char *filter);

/* ── DQL ─────────────────────────────────────────────────────────────────────
 *
 * query_json format:
 * {
 *   "field_name": "embedding",
 *   "vector": [0.1, 0.2, 0.3, 0.4],          -- float32 values
 *   "topk": 10,
 *   "filter": "category = 'tech'",            -- optional
 *   "include_vector": false,                  -- optional, default false
 *   "output_fields": ["title", "score"]       -- optional, null = all fields
 * }
 *
 * results_json output (must be freed with zvec_free_string):
 * [ { "pk": "doc_1", "score": 0.42, "fields": { "title": "hello" } }, ... ]
 */
zvec_status_t zvec_query(zvec_collection_t col, const char *query_json,
                         char **results_json);

/**
 * pks_json: JSON array of string PKs.
 * results_json output: same format as zvec_query results.
 */
zvec_status_t zvec_fetch(zvec_collection_t col, const char *pks_json,
                         char **results_json);

/**
 * Sparse vector similarity query.
 *
 * query_json format:
 * {
 *   "field_name": "sparse_emb",
 *   "indices": [101, 205, 307],    -- uint32 token indices
 *   "values":  [0.9, 0.6, 0.4],   -- float32 scores (same length as indices)
 *   "topk": 10,
 *   "filter": "kind = 'function'", -- optional scalar pre-filter
 *   "output_fields": ["name"]      -- optional, null = all fields
 * }
 *
 * The field named by "field_name" must be SPARSE_VECTOR_FP32 or
 * SPARSE_VECTOR_FP16 with a FLAT or HNSW_SPARSE index.
 *
 * results_json output: same format as zvec_query.
 */
zvec_status_t zvec_sparse_query(zvec_collection_t col, const char *query_json,
                                char **results_json);

/* ── DDL ─────────────────────────────────────────────────────────────────────
 *
 * index_params_json: same as the "index" object in field schema, e.g.
 *   { "type": "HNSW", "metric": "L2", "m": 16, "ef_construction": 200 }
 *   { "type": "FLAT", "metric": "COSINE" }
 *   { "type": "IVF",  "metric": "L2", "n_list": 1024 }
 *   { "type": "INVERT", "enable_range_optimization": true }
 */
zvec_status_t zvec_create_index(zvec_collection_t col, const char *column_name,
                                const char *index_params_json);

zvec_status_t zvec_drop_index(zvec_collection_t col, const char *column_name);

/**
 * field_schema_json: single field object (same format as entries in
 * schema.fields[]).  expression: default-value expression (may be "").
 */
zvec_status_t zvec_add_column(zvec_collection_t col,
                              const char *field_schema_json,
                              const char *expression);

zvec_status_t zvec_drop_column(zvec_collection_t col, const char *column_name);

/**
 * new_name: new column name, or NULL / "" to keep current name.
 * field_schema_json: updated schema, or NULL / "" to keep current schema.
 */
zvec_status_t zvec_alter_column(zvec_collection_t col, const char *column_name,
                                const char *new_name,
                                const char *field_schema_json);

/* ── Memory management ───────────────────────────────────────────────────── */

/** Free a string returned by any API function. */
void zvec_free_string(char *s);

/** Free the message inside a non-OK status. Safe to call on OK statuses. */
void zvec_status_free(zvec_status_t *status);

#ifdef __cplusplus
}
#endif
