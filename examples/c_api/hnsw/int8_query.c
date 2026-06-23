/*
 * Zvec C API — INT8 Query Example (without rotation)
 * ====================================================
 *
 * Opens an INT8 collection (built by int8_build.c, no rotation),
 * runs vector searches, and evaluates recall against ground truth.
 *
 * Equivalent Python example: examples/python/hnsw/int8_query.py
 *
 * Configuration:
 *   Collection : /root/data/cohere/1m/db/cohere_cosine_int8_c
 *   TopK       : 100
 *   QueryFile  : /root/data/cohere/1m/cohere_test_vector_1m.1000.norm.txt
 *   GroundTruth: /root/data/cohere/1m/neighbors.txt
 *   ef_search  : 180
 *
 * Build:
 *   cd examples/c_api && mkdir build && cd build
 *   cmake .. -DHOST_BUILD_DIR=../../../build
 *   make c_api_hnsw_int8_query
 *
 * Usage:
 *   ./c_api_hnsw_int8_query
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "zvec/c_api.h"

/* ==================== Configuration ==================== */

#define COLLECTION_PATH  "/root/data/cohere/1m/db/cohere_cosine_int8_c"
#define QUERY_FILE       "/root/data/cohere/1m/cohere_test_vector_1m.1000.norm.txt"
#define GROUNDTRUTH_FILE "/root/data/cohere/1m/neighbors.txt"

#define DIMENSION        768
#define TOPK             100
#define EF_SEARCH        180
#define MAX_QUERIES      1000
#define WARMUP_ROUNDS    1
#define MEASURE_ROUNDS   3

/* ==================== Helpers ==================== */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Query: "key;v1 v2 v3 ..." format */
typedef struct {
    char key[64];
    float vec[DIMENSION];
} query_entry_t;

static int parse_query_file(const char *path, query_entry_t **out,
                            size_t *out_count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open query file: %s\n", path);
        return -1;
    }

    size_t capacity = MAX_QUERIES;
    size_t count = 0;
    query_entry_t *entries =
        (query_entry_t *)malloc(capacity * sizeof(query_entry_t));

    char line[65536];
    while (fgets(line, sizeof(line), f)) {
        if (count >= MAX_QUERIES) break;

        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (len == 0) continue;

        char *semi = strchr(line, ';');
        if (!semi) continue;

        *semi = 0;
        strncpy(entries[count].key, line, sizeof(entries[count].key) - 1);
        entries[count].key[sizeof(entries[count].key) - 1] = 0;

        char *p = semi + 1;
        int dim = 0;
        while (*p && dim < DIMENSION) {
            char *end;
            entries[count].vec[dim] = strtof(p, &end);
            if (end == p) break;
            dim++;
            p = end;
        }

        if (dim == DIMENSION) {
            count++;
        } else {
            fprintf(stderr, "  Warning: query '%s' dim=%d, expected %d\n",
                    entries[count].key, dim, DIMENSION);
        }
    }
    fclose(f);

    *out = entries;
    *out_count = count;
    return 0;
}

/* GroundTruth: "key;id1 id2 id3 ..." format */
typedef struct gt_entry {
    char key[64];
    char **ids;
    size_t id_count;
} gt_entry_t;

static int parse_groundtruth_file(const char *path, gt_entry_t **out,
                                  size_t *out_count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Ground truth not found: %s\n", path);
        return -1;
    }

    size_t capacity = MAX_QUERIES;
    size_t count = 0;
    gt_entry_t *entries =
        (gt_entry_t *)malloc(capacity * sizeof(gt_entry_t));

    char line[65536];
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (len == 0) continue;

        char *semi = strchr(line, ';');
        if (!semi) continue;

        *semi = 0;
        strncpy(entries[count].key, line, sizeof(entries[count].key) - 1);
        entries[count].key[sizeof(entries[count].key) - 1] = 0;

        char *p = semi + 1;
        size_t id_cap = TOPK + 10;
        entries[count].ids = (char **)malloc(id_cap * sizeof(char *));
        entries[count].id_count = 0;

        while (*p && entries[count].id_count < id_cap) {
            while (*p == ' ') p++;
            if (!*p) break;
            char *start = p;
            while (*p && *p != ' ') p++;
            size_t idlen = p - start;
            char *id = (char *)malloc(idlen + 1);
            memcpy(id, start, idlen);
            id[idlen] = 0;
            entries[count].ids[entries[count].id_count++] = id;
        }
        count++;
    }
    fclose(f);

    *out = entries;
    *out_count = count;
    return 0;
}

static void free_groundtruth(gt_entry_t *gt, size_t count) {
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < gt[i].id_count; j++) {
            free(gt[i].ids[j]);
        }
        free(gt[i].ids);
    }
    free(gt);
}

static int gt_contains(const gt_entry_t *gt, size_t topk, const char *id) {
    size_t limit = gt->id_count < topk ? gt->id_count : topk;
    for (size_t i = 0; i < limit; i++) {
        if (strcmp(gt->ids[i], id) == 0) return 1;
    }
    return 0;
}

static const gt_entry_t *find_gt(const gt_entry_t *gt, size_t gt_count,
                                 const char *key) {
    for (size_t i = 0; i < gt_count; i++) {
        if (strcmp(gt[i].key, key) == 0) return &gt[i];
    }
    return NULL;
}

/* ==================== Main ==================== */

int main(void) {
    printf("============================================================\n");
    printf("  Zvec C API — INT8 Query Example (no rotation)\n");
    printf("============================================================\n");

    zvec_error_code_t err;

    /* ---- Step 1: Init ---- */
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

    /* ---- Step 2: Open collection ---- */
    printf("\n[Step 2] Opening collection: %s\n", COLLECTION_PATH);
    zvec_collection_options_t *options = zvec_collection_options_create();
    zvec_collection_options_set_read_only(options, true);
    zvec_collection_options_set_enable_mmap(options, true);

    zvec_collection_t *collection = NULL;
    err = zvec_collection_open(COLLECTION_PATH, options, &collection);
    zvec_collection_options_destroy(options);
    if (err != ZVEC_OK) {
        char *err_msg = NULL;
        zvec_get_last_error(&err_msg);
        fprintf(stderr, "collection_open failed: %d - %s\n", err,
                err_msg ? err_msg : "unknown");
        zvec_free(err_msg);
        zvec_shutdown();
        return -1;
    }
    printf("  Collection opened.\n");
    printf("  TopK=%d, ef_search=%d\n", TOPK, EF_SEARCH);

    /* ---- Step 3: Load queries ---- */
    printf("\n[Step 3] Loading queries from: %s\n", QUERY_FILE);
    query_entry_t *queries = NULL;
    size_t num_queries = 0;
    if (parse_query_file(QUERY_FILE, &queries, &num_queries) != 0) {
        fprintf(stderr, "Failed to load queries\n");
        zvec_collection_close(collection);
        zvec_shutdown();
        return -1;
    }
    printf("  Loaded %zu queries.\n", num_queries);

    /* ---- Step 4: Load ground truth ---- */
    gt_entry_t *gt = NULL;
    size_t gt_count = 0;
    int has_gt = 0;
    printf("\n[Step 4] Loading ground truth from: %s\n", GROUNDTRUTH_FILE);
    if (parse_groundtruth_file(GROUNDTRUTH_FILE, &gt, &gt_count) == 0) {
        has_gt = 1;
        printf("  Loaded ground truth for %zu queries.\n", gt_count);
    } else {
        printf("  Ground truth not found, skipping recall eval.\n");
    }

    /* ---- Step 5: Search rounds ---- */
    int total_rounds = WARMUP_ROUNDS + MEASURE_ROUNDS;
    printf("\n[Step 5] Running %d rounds (%d warmup + %d measured), "
           "%zu queries/round ...\n",
           total_rounds, WARMUP_ROUNDS, MEASURE_ROUNDS, num_queries);

    double round_qps[MEASURE_ROUNDS];
    double round_recall[MEASURE_ROUNDS];
    int measure_idx = 0;

    for (int rnd = 0; rnd < total_rounds; rnd++) {
        int is_warmup = rnd < WARMUP_ROUNDS;
        double search_start = now_seconds();
        double total_recall = 0.0;
        int matched = 0;

        for (size_t q = 0; q < num_queries; q++) {
            zvec_vector_query_t *vq = zvec_vector_query_create();
            zvec_vector_query_set_field_name(vq, "embedding");
            zvec_vector_query_set_query_vector(vq, queries[q].vec,
                                               DIMENSION * sizeof(float));
            zvec_vector_query_set_topk(vq, TOPK);
            zvec_vector_query_set_include_doc_id(vq, true);
            zvec_vector_query_set_include_vector(vq, false);

            zvec_hnsw_query_params_t *hqp =
                zvec_query_params_hnsw_create(EF_SEARCH, 0.0f, false, false);
            zvec_vector_query_set_hnsw_params(vq, hqp);

            zvec_doc_t **results = NULL;
            size_t result_count = 0;
            err = zvec_collection_query(collection,
                                        (const zvec_vector_query_t *)vq,
                                        &results, &result_count);

            if (has_gt && err == ZVEC_OK) {
                const gt_entry_t *g = find_gt(gt, gt_count, queries[q].key);

                /* Debug: print first query info */
                if (q == 0) {
                    printf("  [DEBUG] query_key='%s' result_count=%zu gt_entry=%p gt_id_count=%zu\n",
                           queries[q].key, result_count, (void*)g, g ? g->id_count : 0);
                    for (size_t r = 0; r < (result_count < 5 ? result_count : 5); r++) {
                        const char *did = zvec_doc_get_pk_pointer(results[r]);
                        printf("    result[%zu] pk='%s'\n", r, did ? did : "(null)");
                    }
                    if (g) {
                        printf("    gt first 5:");
                        for (size_t i = 0; i < (g->id_count < 5 ? g->id_count : 5); i++)
                            printf(" %s", g->ids[i]);
                        printf("\n");
                    }
                }

                if (g && g->id_count > 0) {
                    int hits = 0;
                    size_t limit = g->id_count < (size_t)TOPK
                                       ? g->id_count
                                       : (size_t)TOPK;
                    for (size_t r = 0; r < result_count; r++) {
                        const char *doc_id = zvec_doc_get_pk_pointer(results[r]);
                        if (doc_id && gt_contains(g, (size_t)TOPK, doc_id)) {
                            hits++;
                        }
                    }
                    total_recall += (double)hits / (double)limit;
                    matched++;
                }
            }

            if (results) zvec_docs_free(results, result_count);
            zvec_vector_query_destroy(vq);
        }

        double search_elapsed = now_seconds() - search_start;
        double qps = num_queries / search_elapsed;
        double recall = matched > 0 ? (total_recall / matched * 100.0) : 0.0;

        if (is_warmup) {
            printf("  [Round %d/%d] warmup: QPS=%.1f  recall@%d=%.2f%%"
                   "  (discarded)\n",
                   rnd + 1, total_rounds, qps, TOPK, recall);
        } else {
            round_qps[measure_idx] = qps;
            round_recall[measure_idx] = recall;
            printf("  [Round %d/%d] measured-%d: QPS=%.1f  recall@%d=%.2f%%\n",
                   rnd + 1, total_rounds, measure_idx + 1, qps, TOPK, recall);
            measure_idx++;
        }
    }

    /* ---- Step 6: Summary ---- */
    printf("\n[Step 6] Summary\n");
    printf("  Warmup rounds  : %d\n", WARMUP_ROUNDS);
    printf("  Measured rounds: %d\n", MEASURE_ROUNDS);
    printf("  Queries/round  : %zu\n", num_queries);

    if (measure_idx > 0) {
        double sum_qps = 0, sum_recall = 0;
        double min_qps = round_qps[0], max_qps = round_qps[0];
        for (int i = 0; i < measure_idx; i++) {
            sum_qps += round_qps[i];
            sum_recall += round_recall[i];
            if (round_qps[i] < min_qps) min_qps = round_qps[i];
            if (round_qps[i] > max_qps) max_qps = round_qps[i];
        }
        printf("  Avg QPS        : %.1f  (min=%.1f, max=%.1f)\n",
               sum_qps / measure_idx, min_qps, max_qps);
        if (has_gt) {
            printf("  Avg recall@%d  : %.2f%%\n",
                   TOPK, sum_recall / measure_idx);
        } else {
            printf("  Avg recall@%d  : N/A (no ground truth)\n", TOPK);
        }
    }

    printf("\n============================================================\n");

    free(queries);
    if (has_gt) free_groundtruth(gt, gt_count);
    zvec_collection_close(collection);
    zvec_shutdown();
    return 0;
}
