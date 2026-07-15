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

/**
 * @file diskann_example.c
 * @brief End-to-end example demonstrating DiskANN index usage via the C API.
 *
 * DiskANN is a disk-based approximate nearest neighbor search algorithm
 * optimized for large-scale datasets that exceed available memory. It uses
 * a Vamana graph structure combined with product quantization (PQ) to
 * achieve high recall with efficient disk I/O.
 *
 * NOTE: DiskANN requires Linux x86_64 with libaio. On other platforms the
 * example will compile but the runtime plugin will fail to load.
 *
 * Workflow demonstrated:
 *   1. Create collection schema with DiskANN-indexed vector field
 *   2. Insert documents with high-dimensional vectors
 *   3. Flush collection (triggers PQ training + graph build)
 *   4. Search using DiskANN query parameters (list_size controls recall)
 *   5. Clean up all resources
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zvec/c_api.h"

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

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

#define VECTOR_DIM 64
#define NUM_DOCS 100
#define COLLECTION_DIR "./diskann_example_collection"

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */

int main(void) {
  printf("=== ZVec DiskANN Index Example ===\n\n");

  zvec_error_code_t error;
  int i;

  /* ------------------------------------------------------------------
   * Step 1: Create collection schema
   * ------------------------------------------------------------------ */
  printf("[Step 1] Creating collection schema...\n");

  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("diskann_example");
  if (!schema) {
    fprintf(stderr, "Failed to create schema\n");
    return 1;
  }

  /* Index params — declared up-front and NULL-initialized so the
   * cleanup_schema path never touches an uninitialized pointer even if an
   * early field addition fails. */
  zvec_index_params_t *invert_params = NULL;
  zvec_index_params_t *diskann_params = NULL;

  /* Scalar field with inverted index (for primary key / filtering) */
  invert_params = zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  zvec_index_params_set_invert_params(invert_params, true, false);

  zvec_field_schema_t *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_field_schema_set_index_params(id_field, invert_params);
  error = zvec_collection_schema_add_field(schema, id_field);
  if (handle_error(error, "adding id field") != ZVEC_OK) {
    goto cleanup_schema;
  }
  printf("  + id field (STRING, inverted index)\n");

  /* Vector field with DiskANN index */
  diskann_params = zvec_index_params_create(ZVEC_INDEX_TYPE_DISKANN);
  if (!diskann_params) {
    fprintf(stderr, "Failed to create DiskANN index parameters\n");
    goto cleanup_schema;
  }
  zvec_index_params_set_metric_type(diskann_params, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_diskann_params(
      diskann_params, 64, /* max_degree: graph connectivity */
      100,                /* list_size: build-time candidates */
      8);                 /* pq_chunk_num: PQ chunks (0=auto) */

  printf(
      "  DiskANN index params: max_degree=%d, list_size=%d, pq_chunk_num=%d\n",
      zvec_index_params_get_diskann_max_degree(diskann_params),
      zvec_index_params_get_diskann_list_size(diskann_params),
      zvec_index_params_get_diskann_pq_chunk_num(diskann_params));

  zvec_field_schema_t *embedding_field = zvec_field_schema_create(
      "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, VECTOR_DIM);
  zvec_field_schema_set_index_params(embedding_field, diskann_params);
  error = zvec_collection_schema_add_field(schema, embedding_field);
  if (handle_error(error, "adding embedding field") != ZVEC_OK) {
    goto cleanup_schema;
  }
  printf("  + embedding field (VECTOR_FP32, %dD, DiskANN index)\n", VECTOR_DIM);

  /* Index params are copied into field schemas; safe to destroy now */
  zvec_index_params_destroy(invert_params);
  zvec_index_params_destroy(diskann_params);
  invert_params = NULL;
  diskann_params = NULL;

  /* ------------------------------------------------------------------
   * Step 2: Create and open collection
   * ------------------------------------------------------------------ */
  printf("\n[Step 2] Creating collection...\n");

  zvec_collection_options_t *options = zvec_collection_options_create();
  zvec_collection_t *collection = NULL;
  error = zvec_collection_create_and_open(COLLECTION_DIR, schema, options,
                                          &collection);
  zvec_collection_options_destroy(options);
  if (handle_error(error, "creating collection") != ZVEC_OK) {
    goto cleanup_schema;
  }
  printf("  Collection created at %s\n", COLLECTION_DIR);

  /* ------------------------------------------------------------------
   * Step 3: Generate and insert documents
   * ------------------------------------------------------------------ */
  printf("\n[Step 3] Inserting %d documents with %dD vectors...\n", NUM_DOCS,
         VECTOR_DIM);

  /* Allocate vector storage */
  float(*vectors)[VECTOR_DIM] =
      (float(*)[VECTOR_DIM])malloc(NUM_DOCS * VECTOR_DIM * sizeof(float));
  if (!vectors) {
    fprintf(stderr, "Failed to allocate vector storage\n");
    goto cleanup_collection;
  }

  /* Generate deterministic vector data */
  for (i = 0; i < NUM_DOCS; i++) {
    for (int d = 0; d < VECTOR_DIM; d++) {
      vectors[i][d] = (float)((i * VECTOR_DIM + d) % 1000) / 1000.0f;
    }
  }

  /* Insert in batches */
  int batch_size = 20;
  size_t total_success = 0, total_error = 0;

  for (int batch_start = 0; batch_start < NUM_DOCS; batch_start += batch_size) {
    int count = batch_start + batch_size > NUM_DOCS ? NUM_DOCS - batch_start
                                                    : batch_size;

    zvec_doc_t **docs =
        (zvec_doc_t **)malloc((size_t)count * sizeof(zvec_doc_t *));
    for (i = 0; i < count; i++) {
      int idx = batch_start + i;
      docs[i] = zvec_doc_create();

      char pk[32];
      snprintf(pk, sizeof(pk), "doc_%04d", idx);
      zvec_doc_set_pk(docs[i], pk);

      zvec_doc_add_field_by_value(docs[i], "id", ZVEC_DATA_TYPE_STRING, pk,
                                  strlen(pk));
      zvec_doc_add_field_by_value(docs[i], "embedding",
                                  ZVEC_DATA_TYPE_VECTOR_FP32, vectors[idx],
                                  VECTOR_DIM * sizeof(float));
    }

    size_t success_count = 0, error_count = 0;
    error = zvec_collection_insert(collection, (const zvec_doc_t **)docs,
                                   (size_t)count, &success_count, &error_count);
    if (error != ZVEC_OK) {
      handle_error(error, "inserting batch");
    }
    total_success += success_count;
    total_error += error_count;

    for (i = 0; i < count; i++) {
      zvec_doc_destroy(docs[i]);
    }
    free(docs);
  }
  printf("  Inserted: %zu succeeded, %zu failed\n", total_success, total_error);

  /* ------------------------------------------------------------------
   * Step 4: Flush to trigger index build (PQ training + graph construction)
   * ------------------------------------------------------------------ */
  printf("\n[Step 4] Flushing collection (triggers DiskANN index build)...\n");

  error = zvec_collection_flush(collection);
  if (handle_error(error, "flushing collection") != ZVEC_OK) {
    goto cleanup_vectors;
  }

  zvec_collection_stats_t *stats = NULL;
  error = zvec_collection_get_stats(collection, &stats);
  if (error == ZVEC_OK && stats) {
    printf("  Document count after flush: %llu\n",
           (unsigned long long)zvec_collection_stats_get_doc_count(stats));
    zvec_collection_stats_destroy(stats);
  }

  /* ------------------------------------------------------------------
   * Step 5: Search with DiskANN query parameters
   * ------------------------------------------------------------------ */
  printf("\n[Step 5] Searching with DiskANN query parameters...\n");

  /* Create DiskANN query params — list_size controls the search frontier
   * (beam width). Larger values improve recall at the cost of latency. */
  zvec_diskann_query_params_t *da_qp = zvec_query_params_diskann_create(200);
  if (!da_qp) {
    fprintf(stderr, "Failed to create DiskANN query params\n");
    goto cleanup_vectors;
  }
  printf("  DiskANN query params: list_size=%d\n",
         zvec_query_params_diskann_get_list_size(da_qp));

  /* Build the vector query */
  zvec_vector_query_t *query = zvec_vector_query_create();
  zvec_vector_query_set_field_name(query, "embedding");
  zvec_vector_query_set_query_vector(query, vectors[0],
                                     VECTOR_DIM * sizeof(float));
  zvec_vector_query_set_topk(query, 10);
  zvec_vector_query_set_include_vector(query, false);
  zvec_vector_query_set_include_doc_id(query, true);

  /* Attach DiskANN query params (ownership transfers to query) */
  error = zvec_vector_query_set_diskann_params(query, da_qp);
  if (handle_error(error, "setting DiskANN query params") != ZVEC_OK) {
    zvec_vector_query_destroy(query);
    goto cleanup_vectors;
  }
  /* da_qp is now owned by query — do NOT call diskann_destroy on it */

  /* Execute the query */
  zvec_doc_t **results = NULL;
  size_t result_count = 0;
  error = zvec_collection_query(collection, (const zvec_vector_query_t *)query,
                                &results, &result_count);
  if (error != ZVEC_OK) {
    handle_error(error, "executing DiskANN query");
    printf(
        "  (This is expected on non-Linux platforms — DiskANN requires "
        "libaio)\n");
  } else {
    printf("  Query returned %zu results:\n", result_count);
    for (size_t r = 0; r < result_count && r < 5; r++) {
      const char *pk = zvec_doc_get_pk_copy(results[r]);
      printf("    [%zu] pk=%s  doc_id=%llu  score=%.6f\n", r + 1,
             pk ? pk : "NULL",
             (unsigned long long)zvec_doc_get_doc_id(results[r]),
             zvec_doc_get_score(results[r]));
      if (pk) {
        zvec_free((void *)pk);
      }
    }
    if (result_count > 5) {
      printf("    ... and %zu more\n", result_count - 5);
    }
    zvec_docs_free(results, result_count);
  }
  zvec_vector_query_destroy(query);

  /* ------------------------------------------------------------------
   * Step 6: Demonstrate list_size tuning (higher recall vs. lower latency)
   * ------------------------------------------------------------------ */
  printf("\n[Step 6] Tuning list_size for recall/latency trade-off...\n");

  int list_sizes[] = {50, 100, 300};
  for (int li = 0; li < 3; li++) {
    zvec_diskann_query_params_t *tune_qp =
        zvec_query_params_diskann_create(list_sizes[li]);

    zvec_vector_query_t *tune_query = zvec_vector_query_create();
    zvec_vector_query_set_field_name(tune_query, "embedding");
    zvec_vector_query_set_query_vector(tune_query, vectors[0],
                                       VECTOR_DIM * sizeof(float));
    zvec_vector_query_set_topk(tune_query, 10);
    zvec_vector_query_set_include_doc_id(tune_query, true);
    zvec_vector_query_set_diskann_params(tune_query, tune_qp);

    zvec_doc_t **tune_results = NULL;
    size_t tune_count = 0;
    error = zvec_collection_query(collection,
                                  (const zvec_vector_query_t *)tune_query,
                                  &tune_results, &tune_count);
    if (error == ZVEC_OK) {
      printf("  list_size=%3d -> %zu results returned\n", list_sizes[li],
             tune_count);
      zvec_docs_free(tune_results, tune_count);
    } else {
      printf("  list_size=%3d -> query failed (expected on non-Linux)\n",
             list_sizes[li]);
    }
    zvec_vector_query_destroy(tune_query);
  }

  /* ------------------------------------------------------------------
   * Cleanup
   * ------------------------------------------------------------------ */
cleanup_vectors:
  free(vectors);

cleanup_collection:
  zvec_collection_destroy(collection);

cleanup_schema:
  zvec_collection_schema_destroy(schema);
  if (invert_params) {
    zvec_index_params_destroy(invert_params);
  }
  if (diskann_params) {
    zvec_index_params_destroy(diskann_params);
  }

  printf("\n  DiskANN index type string: %s\n",
         zvec_index_type_to_string(ZVEC_INDEX_TYPE_DISKANN));
  printf("=== Example completed ===\n");
  return 0;
}
