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

// Virtual-table test for the zvec SQLite extension. Builds small collections
// with the C API (one per metric), then checks through SQL that:
//   - MATCH + k returns the exact neighbours (recall vs brute force),
//   - distance is "smaller is better" for every metric,
//   - scalar columns come back, bad input yields SQL errors, not crashes,
//   - several tables/connections share one collection handle, and dropping
//     them releases zvec's file lock.
//
// Usage: vtab_test <path-to-zvec-extension>
// Exit codes: 0 pass, 1 fail, 77 skipped.

#define _XOPEN_SOURCE 700

#include <ftw.h>
#include <math.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "zvec/c_api.h"

#define SKIP_EXIT_CODE 77
#define N_DOCS 2000
#define DIM 32
#define N_QUERIES 50
#define TOPK 10
#define MIN_RECALL 0.95

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, ...)                                   \
  do {                                                     \
    g_checks++;                                            \
    if (!(cond)) {                                         \
      g_failures++;                                        \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

// ---- deterministic data (xorshift64*, seeded per vector index) ------------

static float gen_component(uint64_t *s) {
  *s ^= *s >> 12;
  *s ^= *s << 25;
  *s ^= *s >> 27;
  uint64_t r = *s * 2685821657736338717ULL;
  return (float)((double)(r >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static void gen_vec(uint64_t i, float *out) {
  uint64_t s = 0x9E3779B97F4A7C15ULL ^ (i * 0xD1B54A32D192ED03ULL + 1);
  for (int d = 0; d < DIM; d++) {
    out[d] = gen_component(&s);
  }
}

// "Smaller is better", the same convention as the distance column.
static double ref_dist(int metric, const float *a, const float *b) {
  double dot = 0, na = 0, nb = 0, l2 = 0;
  for (int d = 0; d < DIM; d++) {
    double x = a[d], y = b[d];
    dot += x * y;
    na += x * x;
    nb += y * y;
    l2 += (x - y) * (x - y);
  }
  if (metric == ZVEC_METRIC_TYPE_L2) return l2;
  if (metric == ZVEC_METRIC_TYPE_IP) return -dot;
  return 1.0 - dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

// ---- host SQLite helpers ---------------------------------------------------

static int enable_load_extension(sqlite3 *db) {
#ifdef SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION
  return sqlite3_db_config(db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, NULL);
#else
  return sqlite3_enable_load_extension(db, 1);
#endif
}

static sqlite3 *open_db(const char *ext_path) {
  sqlite3 *db = NULL;
  if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
    fprintf(stderr, "cannot open :memory: database\n");
    exit(1);
  }
  int rc = enable_load_extension(db);
  if (rc != SQLITE_OK) {
    printf("SKIP: this SQLite build cannot load extensions (rc=%d)\n", rc);
    sqlite3_close(db);
    exit(SKIP_EXIT_CODE);
  }
  char *err = NULL;
  rc = sqlite3_load_extension(db, ext_path, "sqlite3_zvec_init", &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "cannot load %s: %s\n", ext_path, err ? err : "?");
    sqlite3_free(err);
    exit(1);
  }
  return db;
}

// Runs one statement and returns the sqlite result code; the message (if any)
// is copied into msg.
static int exec_msg(sqlite3 *db, const char *sql, char *msg, size_t msg_len) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  snprintf(msg, msg_len, "%s", err ? err : "");
  sqlite3_free(err);
  return rc;
}

// Steps a prepared statement to the end; returns the final rc and message.
static int drain(sqlite3_stmt *st, sqlite3 *db, int *rows, char *msg,
                 size_t msg_len) {
  int rc;
  *rows = 0;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    (*rows)++;
  }
  snprintf(msg, msg_len, "%s", rc == SQLITE_DONE ? "" : sqlite3_errmsg(db));
  return rc;
}

// ---- fixture ---------------------------------------------------------------

static int rm_cb(const char *p, const struct stat *sb, int flag,
                 struct FTW *ftw) {
  (void)sb;
  (void)flag;
  (void)ftw;
  return remove(p);
}

static void rm_rf(const char *path) {
  nftw(path, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
}

// Creates <dir>/m<metric> with N_DOCS vectors (pk "d<i>") and an INT32 scalar
// "grp" = i % 7, builds the HNSW index, and closes the collection again.
static int make_collection(const char *path, int metric) {
  zvec_collection_schema_t *schema = zvec_collection_schema_create("t");
  zvec_field_schema_t *grp =
      zvec_field_schema_create("grp", ZVEC_DATA_TYPE_INT32, false, 0);
  zvec_field_schema_t *emb =
      zvec_field_schema_create("emb", ZVEC_DATA_TYPE_VECTOR_FP32, false, DIM);
  zvec_index_params_t *ip = zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  if (!schema || !grp || !emb || !ip) return 1;
  if (zvec_index_params_set_metric_type(ip, (zvec_metric_type_t)metric) !=
          ZVEC_OK ||
      zvec_index_params_set_hnsw_params(ip, 16, 200) != ZVEC_OK ||
      zvec_field_schema_set_index_params(emb, ip) != ZVEC_OK ||
      zvec_collection_schema_add_field(schema, grp) != ZVEC_OK ||
      zvec_collection_schema_add_field(schema, emb) != ZVEC_OK) {
    return 1;
  }
  zvec_collection_t *col = NULL;
  if (zvec_collection_create_and_open(path, schema, NULL, &col) != ZVEC_OK) {
    return 1;
  }

  enum { kBatch = 250 };
  const zvec_doc_t *docs[kBatch];
  float vec[DIM];
  for (int base = 0; base < N_DOCS; base += kBatch) {
    for (int j = 0; j < kBatch; j++) {
      int i = base + j;
      char pk[32];
      snprintf(pk, sizeof pk, "d%d", i);
      int32_t g = i % 7;
      zvec_doc_t *doc = zvec_doc_create();
      zvec_doc_set_pk(doc, pk);
      gen_vec((uint64_t)i, vec);
      if (zvec_doc_add_field_by_value(doc, "grp", ZVEC_DATA_TYPE_INT32, &g,
                                      sizeof g) != ZVEC_OK ||
          zvec_doc_add_field_by_value(doc, "emb", ZVEC_DATA_TYPE_VECTOR_FP32,
                                      vec, sizeof vec) != ZVEC_OK) {
        return 1;
      }
      docs[j] = doc;
    }
    size_t ok = 0, bad = 0;
    zvec_error_code_t rc = zvec_collection_insert(col, docs, kBatch, &ok, &bad);
    for (int j = 0; j < kBatch; j++) zvec_doc_destroy((zvec_doc_t *)docs[j]);
    if (rc != ZVEC_OK || bad != 0 || ok != kBatch) return 1;
  }
  if (zvec_collection_flush(col) != ZVEC_OK ||
      zvec_collection_optimize(col) != ZVEC_OK ||
      zvec_collection_close(col) != ZVEC_OK) {
    return 1;
  }
  return 0;
}

// ---- tests -----------------------------------------------------------------

static const char *metric_name(int m) {
  return m == ZVEC_METRIC_TYPE_L2   ? "L2"
         : m == ZVEC_METRIC_TYPE_IP ? "IP"
                                    : "COSINE";
}

// Recall, distance ordering/accuracy, self-hit and scalar column.
static void test_search(sqlite3 *db, int metric, const float *data) {
  sqlite3_stmt *st = NULL;
  int rc = sqlite3_prepare_v2(
      db, "SELECT pk, grp, distance FROM p WHERE emb MATCH ?1 AND k = ?2", -1,
      &st, NULL);
  CHECK(rc == SQLITE_OK, "%s: prepare: %s", metric_name(metric),
        sqlite3_errmsg(db));
  if (rc != SQLITE_OK) return;

  uint64_t rng = 0x123456789ABCDEFULL;
  double sum = 0, min_recall = 1.0, max_err = 0;
  int ordered = 1, self_miss = 0, grp_bad = 0, short_result = 0;

  for (int qi = 0; qi < N_QUERIES; qi++) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    int src = (int)(rng % N_DOCS);
    float q[DIM];
    memcpy(q, data + (size_t)src * DIM, sizeof q);
    if (qi >= 10) {
      for (int d = 0; d < DIM; d++) q[d] += 0.1f * gen_component(&rng);
    }

    // exact top-k by insertion into a sorted array
    double bd[TOPK];
    int bi[TOPK], bn = 0;
    for (int i = 0; i < N_DOCS; i++) {
      double d = ref_dist(metric, q, data + (size_t)i * DIM);
      if (bn == TOPK && d >= bd[TOPK - 1]) continue;
      int p = bn < TOPK ? bn++ : TOPK - 1;
      while (p > 0 && bd[p - 1] > d) {
        bd[p] = bd[p - 1];
        bi[p] = bi[p - 1];
        p--;
      }
      bd[p] = d;
      bi[p] = i;
    }

    sqlite3_reset(st);
    sqlite3_bind_blob(st, 1, q, (int)sizeof q, SQLITE_STATIC);
    sqlite3_bind_int(st, 2, TOPK);
    int hits = 0, got = 0;
    double prev = -INFINITY;
    while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
      int id = atoi((const char *)sqlite3_column_text(st, 0) + 1);
      int grp = sqlite3_column_int(st, 1);
      double d = sqlite3_column_double(st, 2);
      if (grp != id % 7) grp_bad++;
      if (got == 0 && qi < 10 && metric != ZVEC_METRIC_TYPE_IP && id != src) {
        self_miss++;  // IP: a vector is not necessarily its own best match
      }
      if (d < prev) ordered = 0;
      prev = d;
      for (int j = 0; j < TOPK; j++) {
        if (bi[j] == id) {
          hits++;
          break;
        }
      }
      double e = fabs(d - ref_dist(metric, q, data + (size_t)id * DIM));
      if (e > max_err) max_err = e;
      got++;
    }
    CHECK(rc == SQLITE_DONE, "%s: step: %s", metric_name(metric),
          sqlite3_errmsg(db));
    if (got != TOPK) short_result++;
    double r = (double)hits / TOPK;
    sum += r;
    if (r < min_recall) min_recall = r;
  }
  sqlite3_finalize(st);

  double mean = sum / N_QUERIES;
  printf("%-6s recall@%d mean=%.4f min=%.2f max_dist_err=%.2g\n",
         metric_name(metric), TOPK, mean, min_recall, max_err);
  CHECK(mean >= MIN_RECALL, "%s: mean recall %.4f < %.2f", metric_name(metric),
        mean, MIN_RECALL);
  CHECK(short_result == 0, "%s: %d queries returned fewer than k rows",
        metric_name(metric), short_result);
  CHECK(ordered, "%s: distance not ascending", metric_name(metric));
  CHECK(self_miss == 0, "%s: %d/10 self queries missed", metric_name(metric),
        self_miss);
  CHECK(grp_bad == 0, "%s: %d wrong scalar values", metric_name(metric),
        grp_bad);
  CHECK(max_err < 1e-3, "%s: distance differs from reference by %g",
        metric_name(metric), max_err);
}

// Every malformed use must surface as an SQL error, never a crash.
static void test_errors(sqlite3 *db, const char *missing_path) {
  char msg[512];
  float q[DIM] = {0};
  float short_q[DIM - 1] = {0};
  sqlite3_stmt *st = NULL;
  int rows = 0, rc;

  rc = sqlite3_prepare_v2(db, "SELECT pk FROM p", -1, &st, NULL);
  if (rc == SQLITE_OK) {
    rc = drain(st, db, &rows, msg, sizeof msg);
  } else {
    snprintf(msg, sizeof msg, "%s", sqlite3_errmsg(db));  // rejected at prepare
  }
  sqlite3_finalize(st);
  CHECK(rc != SQLITE_DONE && rc != SQLITE_ROW, "scan without MATCH succeeded");
  CHECK(msg[0] != '\0', "scan without MATCH: empty error message");

  const char *bad_sql = "SELECT pk FROM p WHERE emb MATCH ?1 AND k = ?2";
  struct {
    const char *what;
    int kind;  // 0 blob, 1 short blob, 2 text
    int k;
  } cases[] = {{"k = 0", 0, 0},
               {"k < 0", 0, -3},
               {"short blob", 1, TOPK},
               {"text query", 2, TOPK}};
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    rc = sqlite3_prepare_v2(db, bad_sql, -1, &st, NULL);
    CHECK(rc == SQLITE_OK, "%s: prepare failed", cases[i].what);
    if (rc != SQLITE_OK) continue;
    if (cases[i].kind == 0) {
      sqlite3_bind_blob(st, 1, q, (int)sizeof q, SQLITE_STATIC);
    } else if (cases[i].kind == 1) {
      sqlite3_bind_blob(st, 1, short_q, (int)sizeof short_q, SQLITE_STATIC);
    } else {
      sqlite3_bind_text(st, 1, "hello", -1, SQLITE_STATIC);
    }
    sqlite3_bind_int(st, 2, cases[i].k);
    rc = drain(st, db, &rows, msg, sizeof msg);
    sqlite3_finalize(st);
    CHECK(rc != SQLITE_DONE && rc != SQLITE_ROW, "%s: did not fail",
          cases[i].what);
    CHECK(msg[0] != '\0', "%s: empty error message", cases[i].what);
  }

  char *sql =
      sqlite3_mprintf("CREATE VIRTUAL TABLE x USING zvec('%q')", missing_path);
  rc = exec_msg(db, sql, msg, sizeof msg);
  sqlite3_free(sql);
  CHECK(rc != SQLITE_OK, "CREATE on a missing collection succeeded");
  CHECK(strstr(msg, missing_path) != NULL, "error does not name the path: %s",
        msg);
}

// Two tables and a second connection on one collection share the registry
// handle; once all are dropped zvec's exclusive file lock is free again.
static void test_sharing_and_release(sqlite3 *db, const char *ext_path,
                                     const char *path) {
  char msg[256];
  char *sql = sqlite3_mprintf("CREATE VIRTUAL TABLE p2 USING zvec('%q')", path);
  int rc = exec_msg(db, sql, msg, sizeof msg);
  CHECK(rc == SQLITE_OK, "second table on same collection: %s", msg);

  sqlite3 *db2 = open_db(ext_path);
  sqlite3_free(sql);
  sql = sqlite3_mprintf("CREATE VIRTUAL TABLE p3 USING zvec('%q')", path);
  rc = exec_msg(db2, sql, msg, sizeof msg);
  sqlite3_free(sql);
  CHECK(rc == SQLITE_OK, "second connection on same collection: %s", msg);

  const char *q =
      "SELECT count(*) FROM %s WHERE emb MATCH zeroblob(%d) AND k = 3";
  for (int t = 0; t < 3; t++) {
    sqlite3 *h = t == 2 ? db2 : db;
    const char *name = t == 0 ? "p" : t == 1 ? "p2" : "p3";
    sqlite3_stmt *st = NULL;
    sql = sqlite3_mprintf(q, name, (int)(DIM * sizeof(float)));
    rc = sqlite3_prepare_v2(h, sql, -1, &st, NULL);
    sqlite3_free(sql);
    CHECK(rc == SQLITE_OK, "%s: prepare: %s", name, sqlite3_errmsg(h));
    if (rc == SQLITE_OK) {
      rc = sqlite3_step(st);
      CHECK(rc == SQLITE_ROW && sqlite3_column_int(st, 0) == 3,
            "%s: expected 3 rows", name);
    }
    sqlite3_finalize(st);
  }

  CHECK(exec_msg(db, "DROP TABLE p2", msg, sizeof msg) == SQLITE_OK, "%s", msg);
  CHECK(exec_msg(db, "DROP TABLE p", msg, sizeof msg) == SQLITE_OK, "%s", msg);
  sqlite3_close(db2);  // drops p3 with the connection

  zvec_collection_t *col = NULL;
  zvec_error_code_t zrc = zvec_collection_open(path, NULL, &col);
  CHECK(zrc == ZVEC_OK, "collection still locked after all tables closed");
  if (zrc == ZVEC_OK) zvec_collection_close(col);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <path-to-zvec-extension>\n", argv[0]);
    return 1;
  }
  const char *ext_path = argv[1];

  char tmpl[] = "/tmp/zvec_vtab_test_XXXXXX";
  char *dir = mkdtemp(tmpl);
  if (dir == NULL) {
    fprintf(stderr, "mkdtemp failed\n");
    return 1;
  }

  float *data = malloc(sizeof(float) * N_DOCS * DIM);
  if (data == NULL) return 1;
  for (int i = 0; i < N_DOCS; i++) gen_vec((uint64_t)i, data + (size_t)i * DIM);

  const int metrics[] = {ZVEC_METRIC_TYPE_L2, ZVEC_METRIC_TYPE_IP,
                         ZVEC_METRIC_TYPE_COSINE};
  char l2_path[256] = "";
  for (size_t m = 0; m < sizeof metrics / sizeof metrics[0]; m++) {
    char path[256];
    snprintf(path, sizeof path, "%s/m%d", dir, metrics[m]);
    if (make_collection(path, metrics[m]) != 0) {
      fprintf(stderr, "cannot build %s fixture\n", metric_name(metrics[m]));
      rm_rf(dir);
      return 1;
    }
    sqlite3 *db = open_db(ext_path);
    char *sql =
        sqlite3_mprintf("CREATE VIRTUAL TABLE p USING zvec('%q')", path);
    char msg[256];
    int rc = exec_msg(db, sql, msg, sizeof msg);
    sqlite3_free(sql);
    CHECK(rc == SQLITE_OK, "%s: create: %s", metric_name(metrics[m]), msg);
    if (rc == SQLITE_OK) {
      test_search(db, metrics[m], data);
      if (metrics[m] == ZVEC_METRIC_TYPE_L2) {
        snprintf(l2_path, sizeof l2_path, "%s", path);
        char missing[300];
        snprintf(missing, sizeof missing, "%s/does_not_exist", dir);
        test_errors(db, missing);
        test_sharing_and_release(db, ext_path, l2_path);
      }
    }
    sqlite3_close(db);
  }

  free(data);
  rm_rf(dir);
  printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
