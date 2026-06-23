/*
 * Zvec C API — INT8 Build Example (without rotation)
 * =====================================================
 *
 * Builds a zvec Collection with INT8 quantization.
 * Unlike int8_rotate_build.c, this does NOT enable random rotation
 * preprocessing — quantization is applied directly on the raw vectors.
 *
 * Key configuration:
 *   quantize_type = ZVEC_QUANTIZE_TYPE_INT8
 *   enable_rotate = false  (default, not called)
 *
 * Equivalent Python example: examples/python/hnsw/int8_build.py
 *
 * Input : /root/data/cohere/1m/cohere_train_vector_1m.norm.zvec.vecs
 * Output: /root/data/cohere/1m/db/cohere_cosine_int8_c
 *
 * Build:
 *   cd examples/c_api && mkdir build && cd build
 *   cmake .. -DHOST_BUILD_DIR=../../../build
 *   make c_api_hnsw_int8_build
 *
 * Usage:
 *   ./c_api_hnsw_int8_build
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "zvec/c_api.h"

/* ==================== Configuration ==================== */

#define VECS_FILE      "/root/data/cohere/1m/cohere_train_vector_1m.norm.zvec.vecs"
#define COLLECTION_PATH "/root/data/cohere/1m/db/cohere_cosine_int8_c"

#define DIMENSION        768
#define METRIC_TYPE      ZVEC_METRIC_TYPE_COSINE
#define HNSW_M           15
#define EF_CONSTRUCTION  500

#define INSERT_BATCH_SIZE 1000

/* ==================== .zvec.vecs header layout ==================== */
#define VECS_HEADER_SIZE (8 + 2 + 2 + 4 + 8 * 11)  /* 112 bytes */

typedef struct {
    uint64_t num_vecs;
    uint32_t meta_size;
    uint64_t key_offset;
    uint64_t key_size;
    uint64_t dense_offset;
    uint64_t dense_size;
} vecs_header_t;

static int parse_vecs_header(const char *path, vecs_header_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open vecs file: %s\n", path);
        return -1;
    }

    uint8_t buf[VECS_HEADER_SIZE];
    if (fread(buf, 1, VECS_HEADER_SIZE, f) != VECS_HEADER_SIZE) {
        fprintf(stderr, "Failed to read vecs header\n");
        fclose(f);
        return -1;
    }
    fclose(f);

    memcpy(&out->num_vecs, buf + 0, 8);      /* Q: offset 0 */
    memcpy(&out->meta_size, buf + 12, 4);     /* I: offset 8+2+2=12 */
    memcpy(&out->key_offset,   buf + 24, 8);  /* Q: offset 16+8=24 (vals[5]) */
    memcpy(&out->key_size,     buf + 32, 8);  /* Q: offset 24+8=32 (vals[6]) */
    memcpy(&out->dense_offset, buf + 40, 8);  /* Q: offset 32+8=40 (vals[7]) */
    memcpy(&out->dense_size,   buf + 48, 8);  /* Q: offset 40+8=48 (vals[8]) */

    return 0;
}

/* ==================== Helpers ==================== */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int mkdirs(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755);
}

static int remove_dir_recursive(const char *path) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", path);
    return system(cmd);
}

/* ==================== Main ==================== */

int main(void) {
    printf("============================================================\n");
    printf("  Zvec C API — INT8 Build Example (no rotation)\n");
    printf("============================================================\n");

    zvec_error_code_t err;

    /* ---- Step 1: Initialize zvec ---- */
    printf("\n[Step 1] Initializing zvec ...\n");
    zvec_config_data_t *config = zvec_config_data_create();
    if (config) {
        zvec_log_config_t *log_cfg =
            zvec_config_log_create_console(ZVEC_LOG_LEVEL_INFO);
        zvec_config_data_set_log_config(config, log_cfg);
    }
    err = zvec_initialize(config);
    if (config) zvec_config_data_destroy(config);
    if (err != ZVEC_OK) {
        fprintf(stderr, "zvec_initialize failed: %d\n", err);
        return -1;
    }
    printf("  Done.\n");

    /* ---- Step 2: Parse .zvec.vecs header ---- */
    printf("\n[Step 2] Parsing vecs file: %s\n", VECS_FILE);
    vecs_header_t hdr;
    if (parse_vecs_header(VECS_FILE, &hdr) != 0) {
        zvec_shutdown();
        return -1;
    }
    size_t elem_size = hdr.dense_size / hdr.num_vecs;
    size_t vec_dim_floats = elem_size / sizeof(float);
    printf("  num_vecs: %lu, dim: %zu\n",
           (unsigned long)hdr.num_vecs, vec_dim_floats);
    if (vec_dim_floats != DIMENSION) {
        fprintf(stderr, "Dimension mismatch: got %zu, expected %d\n",
                vec_dim_floats, DIMENSION);
        zvec_shutdown();
        return -1;
    }

    /* ---- Step 3: Create collection schema ---- */
    printf("\n[Step 3] Creating collection at %s ...\n", COLLECTION_PATH);
    printf("  quantize_type = INT8\n");
    printf("  enable_rotate = false (default)\n");
    printf("  metric_type   = COSINE\n");

    /* Index params: HNSW + INT8 (no rotation) */
    zvec_index_params_t *hnsw_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
    zvec_index_params_set_metric_type(hnsw_params, METRIC_TYPE);
    zvec_index_params_set_hnsw_params(hnsw_params, HNSW_M, EF_CONSTRUCTION);
    zvec_index_params_set_quantize_type(hnsw_params, ZVEC_QUANTIZE_TYPE_INT8);
    /* NOTE: enable_rotate is NOT set — defaults to false */

    /* Invert index params for ID field */
    zvec_index_params_t *invert_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
    zvec_index_params_set_invert_params(invert_params, true, false);

    /* Schema */
    zvec_collection_schema_t *schema =
        zvec_collection_schema_create("cohere_cosine_int8_c");

    zvec_field_schema_t *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_field_schema_set_index_params(id_field, invert_params);
    zvec_collection_schema_add_field(schema, id_field);
    zvec_free_field_schema(id_field);

    zvec_field_schema_t *vec_field =
        zvec_field_schema_create("embedding", ZVEC_DATA_TYPE_VECTOR_FP32,
                                 false, DIMENSION);
    zvec_field_schema_set_index_params(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);
    zvec_free_field_schema(vec_field);

    zvec_index_params_destroy(hnsw_params);
    zvec_index_params_destroy(invert_params);

    /* Prepare path */
    /* Clean up existing collection if present */
    if (access(COLLECTION_PATH, F_OK) == 0) {
        remove_dir_recursive(COLLECTION_PATH);
    }

    /* Collection options */
    zvec_collection_options_t *options = zvec_collection_options_create();
    zvec_collection_options_set_enable_mmap(options, true);

    zvec_collection_t *collection = NULL;
    err = zvec_collection_create_and_open(COLLECTION_PATH, schema, options,
                                          &collection);
    zvec_collection_options_destroy(options);
    if (err != ZVEC_OK) {
        char *err_msg = NULL;
        zvec_get_last_error(&err_msg);
        fprintf(stderr, "create_and_open failed: %d - %s\n", err,
                err_msg ? err_msg : "unknown");
        zvec_free(err_msg);
        zvec_collection_schema_destroy(schema);
        zvec_shutdown();
        return -1;
    }
    printf("  Collection created.\n");

    /* ---- Step 4: Read vectors and insert ---- */
    printf("\n[Step 4] Inserting %lu vectors (batch_size=%d) ...\n",
           (unsigned long)hdr.num_vecs, INSERT_BATCH_SIZE);

    FILE *vecs_fp = fopen(VECS_FILE, "rb");
    if (!vecs_fp) {
        fprintf(stderr, "Cannot open vecs file for reading\n");
        zvec_collection_close(collection);
        zvec_collection_schema_destroy(schema);
        zvec_shutdown();
        return -1;
    }

    size_t data_start = VECS_HEADER_SIZE + hdr.meta_size;
    size_t dense_abs = data_start + hdr.dense_offset;
    size_t key_abs = data_start + hdr.key_offset;

    double insert_start = now_seconds();
    size_t total_inserted = 0;

    float *vec_buf = (float *)malloc(elem_size);
    zvec_doc_t **batch_docs =
        (zvec_doc_t **)malloc(sizeof(zvec_doc_t *) * INSERT_BATCH_SIZE);
    size_t batch_count = 0;

    for (uint64_t i = 0; i < hdr.num_vecs; i++) {
        uint64_t key_val = 0;
        fseeko(vecs_fp, (off_t)(key_abs + i * 8), SEEK_SET);
        fread(&key_val, 8, 1, vecs_fp);

        fseeko(vecs_fp, (off_t)(dense_abs + i * elem_size), SEEK_SET);
        fread(vec_buf, 1, elem_size, vecs_fp);

        zvec_doc_t *doc = zvec_doc_create();
        char pk[32];
        snprintf(pk, sizeof(pk), "%lu", (unsigned long)key_val);
        zvec_doc_set_pk(doc, pk);

        int64_t id_val = (int64_t)key_val;
        zvec_doc_add_field_by_value(doc, "id", ZVEC_DATA_TYPE_INT64,
                                    &id_val, sizeof(id_val));
        zvec_doc_add_field_by_value(doc, "embedding",
                                    ZVEC_DATA_TYPE_VECTOR_FP32,
                                    vec_buf, elem_size);

        batch_docs[batch_count++] = doc;

        if (batch_count >= INSERT_BATCH_SIZE) {
            size_t success = 0, errors = 0;
            zvec_collection_insert(collection,
                                   (const zvec_doc_t **)batch_docs,
                                   batch_count, &success, &errors);
            total_inserted += success;
            if (total_inserted % 50000 == 0 ||
                total_inserted == hdr.num_vecs) {
                double elapsed = now_seconds() - insert_start;
                double speed = elapsed > 0 ? total_inserted / elapsed : 0;
                printf("  [%8lu / %lu] %.0f docs/s\n",
                       (unsigned long)total_inserted,
                       (unsigned long)hdr.num_vecs, speed);
            }
            for (size_t j = 0; j < batch_count; j++) {
                zvec_doc_destroy(batch_docs[j]);
            }
            batch_count = 0;
        }
    }

    if (batch_count > 0) {
        size_t success = 0, errors = 0;
        zvec_collection_insert(collection,
                               (const zvec_doc_t **)batch_docs,
                               batch_count, &success, &errors);
        total_inserted += success;
        for (size_t j = 0; j < batch_count; j++) {
            zvec_doc_destroy(batch_docs[j]);
        }
    }

    double insert_elapsed = now_seconds() - insert_start;
    printf("\n  Insert complete: %lu docs in %.1fs (%.0f docs/s)\n",
           (unsigned long)total_inserted, insert_elapsed,
           total_inserted / insert_elapsed);

    free(vec_buf);
    free(batch_docs);
    fclose(vecs_fp);

    /* ---- Step 5: Optimize (build HNSW + INT8) ---- */
    printf("\n[Step 5] Optimizing collection (HNSW + INT8) ...\n");
    double opt_start = now_seconds();
    err = zvec_collection_optimize(collection);
    double opt_elapsed = now_seconds() - opt_start;
    if (err != ZVEC_OK) {
        fprintf(stderr, "optimize failed: %d\n", err);
    } else {
        printf("  Optimize done in %.1fs\n", opt_elapsed);
    }

    /* ---- Step 6: Flush ---- */
    printf("\n[Step 6] Flushing collection ...\n");
    zvec_collection_flush(collection);
    printf("  Done.\n");

    printf("\n============================================================\n");
    printf("  Build complete!\n");
    printf("  Collection saved to: %s\n", COLLECTION_PATH);
    printf("  Run c_api_hnsw_int8_query to search.\n");
    printf("============================================================\n");

    zvec_collection_close(collection);
    zvec_collection_schema_destroy(schema);
    zvec_shutdown();
    return 0;
}
