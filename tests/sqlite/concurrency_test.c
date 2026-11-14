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

// Several threads, one SQLite connection each, share one derived index:
// writers insert and sync, readers search (and so sync before searching).
// Meant to run under ThreadSanitizer as well as plainly. At the end one more
// sync must leave the index identical to the table.
//
// Usage: concurrency_test <path-to-zvec-extension>
// Exit codes: 0 pass, 1 fail, 77 skipped.

#define _XOPEN_SOURCE 700

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
int main(void) {
  printf("SKIP: concurrency_test uses POSIX threads\n");
  return 77;
}
#else

#include <sys/stat.h>
#include <ftw.h>
#include <pthread.h>
#include <unistd.h>

#define SKIP_EXIT_CODE 77
#define WRITERS 3
#define READERS 3
#define ROWS_PER_WRITER 400

static const char *g_ext = NULL;
static const char *g_db = NULL;
static int g_writers_done = 0; /* accessed with __atomic builtins */
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_checks = 0, g_failures = 0, g_searches = 0;

#define CHECK(cond, ...)                                   \
  do {                                                     \
    pthread_mutex_lock(&g_mu);                             \
    g_checks++;                                            \
    if (!(cond)) {                                         \
      g_failures++;                                        \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
    pthread_mutex_unlock(&g_mu);                           \
  } while (0)

static sqlite3 *open_db(void) {
  sqlite3 *db = NULL;
  if (sqlite3_open(g_db, &db) != SQLITE_OK) {
    fprintf(stderr, "cannot open %s\n", g_db);
    exit(1);
  }
  sqlite3_busy_timeout(db, 10000);
#ifdef SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION
  int rc =
      sqlite3_db_config(db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, NULL);
#else
  int rc = sqlite3_enable_load_extension(db, 1);
#endif
  if (rc != SQLITE_OK) {
    printf("SKIP: this SQLite build cannot load extensions (rc=%d)\n", rc);
    exit(SKIP_EXIT_CODE);
  }
  char *err = NULL;
  if (sqlite3_load_extension(db, g_ext, "sqlite3_zvec_init", &err) !=
      SQLITE_OK) {
    fprintf(stderr, "cannot load %s: %s\n", g_ext, err ? err : "?");
    exit(1);
  }
  return db;
}

static const char *one(sqlite3 *db, const char *sql, char *buf, size_t len) {
  sqlite3_stmt *st = NULL;
  buf[0] = '\0';
  if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK) {
    snprintf(buf, len, "ERR: %s", sqlite3_errmsg(db));
    return buf;
  }
  int rc = sqlite3_step(st);
  if (rc == SQLITE_ROW) {
    const char *t = (const char *)sqlite3_column_text(st, 0);
    snprintf(buf, len, "%s", t ? t : "");
  } else if (rc != SQLITE_DONE) {
    snprintf(buf, len, "ERR: %s", sqlite3_errmsg(db));
  }
  sqlite3_finalize(st);
  return buf;
}

static void *writer(void *arg) {
  long w = (long)arg;
  sqlite3 *db = open_db();
  sqlite3_stmt *ins = NULL;
  sqlite3_prepare_v2(db, "INSERT INTO docs(id, embedding) VALUES(?1, ?2)", -1,
                     &ins, NULL);
  char buf[512];
  for (int i = 0; i < ROWS_PER_WRITER; i++) {
    float v[4] = {(float)w, (float)i, 1.0f, 0.0f};
    sqlite3_reset(ins);
    sqlite3_bind_int64(ins, 1, w * 100000 + i);
    sqlite3_bind_blob(ins, 2, v, sizeof v, SQLITE_TRANSIENT);
    int rc = sqlite3_step(ins);
    CHECK(rc == SQLITE_DONE, "writer %ld: insert failed: %s", w,
          sqlite3_errmsg(db));
    if (i % 25 == 24) {
      const char *r = one(db, "SELECT zvec_sync('docs_idx')", buf, sizeof buf);
      // Losing the write-lock race to another connection is fine: the queue
      // stays.
      CHECK(strncmp(r, "ERR", 3) != 0 || strstr(r, "locked") != NULL,
            "writer %ld: sync: %s", w, r);
    }
  }
  sqlite3_finalize(ins);
  sqlite3_close(db);
  return NULL;
}

static void *reader(void *arg) {
  long r = (long)arg;
  sqlite3 *db = open_db();
  sqlite3_stmt *st = NULL;
  sqlite3_prepare_v2(db,
                     "SELECT count(*) FROM (SELECT pk FROM docs_idx WHERE "
                     "embedding MATCH ?1 AND k = 5)",
                     -1, &st, NULL);
  unsigned seed = (unsigned)r + 7;
  while (!__atomic_load_n(&g_writers_done, __ATOMIC_ACQUIRE)) {
    float v[4] = {(float)(rand_r(&seed) % WRITERS),
                  (float)(rand_r(&seed) % ROWS_PER_WRITER), 1.0f, 0.0f};
    sqlite3_reset(st);
    sqlite3_bind_blob(st, 1, v, sizeof v, SQLITE_TRANSIENT);
    int rc = sqlite3_step(st);
    CHECK(rc == SQLITE_ROW || rc == SQLITE_DONE,
          "reader %ld: search failed: %s", r, sqlite3_errmsg(db));
    pthread_mutex_lock(&g_mu);
    g_searches++;
    pthread_mutex_unlock(&g_mu);
  }
  sqlite3_finalize(st);
  sqlite3_close(db);
  return NULL;
}

static int rm_cb(const char *p, const struct stat *sb, int flag,
                 struct FTW *ftw) {
  (void)sb;
  (void)flag;
  (void)ftw;
  return remove(p);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <path-to-zvec-extension>\n", argv[0]);
    return 1;
  }
  g_ext = argv[1];
  char tmpl[] = "/tmp/zvec_conc_XXXXXX";
  char *dir = mkdtemp(tmpl);
  if (dir == NULL) return 1;
  char dbfile[512];
  snprintf(dbfile, sizeof dbfile, "%s/c.db", dir);
  g_db = dbfile;

  sqlite3 *db = open_db();
  char *err = NULL;
  int rc =
      sqlite3_exec(db,
                   "CREATE TABLE docs(id INTEGER PRIMARY KEY, embedding BLOB);"
                   "CREATE VIRTUAL TABLE docs_idx USING zvec(base=docs, "
                   "key=id, column=embedding, dim=4, metric=l2)",
                   NULL, NULL, &err);
  CHECK(rc == SQLITE_OK, "setup: %s", err ? err : "?");
  sqlite3_free(err);
  sqlite3_close(db);

  pthread_t tw[WRITERS], tr[READERS];
  for (long i = 0; i < READERS; i++)
    pthread_create(&tr[i], NULL, reader, (void *)i);
  for (long i = 0; i < WRITERS; i++)
    pthread_create(&tw[i], NULL, writer, (void *)i);
  for (int i = 0; i < WRITERS; i++) pthread_join(tw[i], NULL);
  __atomic_store_n(&g_writers_done, 1, __ATOMIC_RELEASE);
  for (int i = 0; i < READERS; i++) pthread_join(tr[i], NULL);

  char buf[512];
  db = open_db();
  one(db, "SELECT zvec_sync('docs_idx')", buf, sizeof buf);
  CHECK(strncmp(buf, "ERR", 3) != 0, "final sync: %s", buf);
  one(db, "SELECT zvec_check('docs_idx')", buf, sizeof buf);
  CHECK(strstr(buf, "\"status\":\"ok\"") != NULL,
        "index differs from the table: %s", buf);
  char want[64];
  snprintf(want, sizeof want, "\"base\":%d,", WRITERS * ROWS_PER_WRITER);
  CHECK(strstr(buf, want) != NULL, "row count: %s", buf);
  sqlite3_close(db);
  nftw(dir, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
  printf("%d writers x %d rows, %d readers, %d searches: %s\n", WRITERS,
         ROWS_PER_WRITER, READERS, g_searches, buf);
  printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
#endif
