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

// Crash safety of the derived index: a child process keeps writing rows and
// syncing; the parent kills it with SIGKILL at a random moment, then reopens
// the database and requires that one more sync leaves the index identical to
// the base table (zvec_check reports "ok") and that searches find the rows.
// Runs once with sync=manual and once with sync=query.
//
// Usage: crash_test <path-to-zvec-extension>
// Environment: ZVEC_CRASH_ITERATIONS (default 12 per mode), ZVEC_CRASH_SEED.
// Exit codes: 0 pass, 1 fail, 77 skipped (no POSIX fork, or no extension loading).

#define _XOPEN_SOURCE 700

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
int main(void) {
  printf("SKIP: crash_test needs fork()/SIGKILL\n");
  return 77;
}
#else

#include <ftw.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define SKIP_EXIT_CODE 77
#define DIM 4

static int g_checks = 0;
static int g_failures = 0;
static const char *g_ext = NULL;

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

static sqlite3 *open_db(const char *file) {
  sqlite3 *db = NULL;
  if (sqlite3_open(file, &db) != SQLITE_OK) {
    fprintf(stderr, "cannot open %s\n", file);
    exit(1);
  }
  sqlite3_busy_timeout(db, 2000);
#ifdef SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION
  int rc = sqlite3_db_config(db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, NULL);
#else
  int rc = sqlite3_enable_load_extension(db, 1);
#endif
  if (rc != SQLITE_OK) {
    printf("SKIP: this SQLite build cannot load extensions (rc=%d)\n", rc);
    exit(SKIP_EXIT_CODE);
  }
  char *err = NULL;
  if (sqlite3_load_extension(db, g_ext, "sqlite3_zvec_init", &err) != SQLITE_OK) {
    fprintf(stderr, "cannot load %s: %s\n", g_ext, err ? err : "?");
    exit(1);
  }
  return db;
}

static const char *one(sqlite3 *db, const char *sql) {
  static char buf[1024];
  sqlite3_stmt *st = NULL;
  buf[0] = '\0';
  if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK) {
    snprintf(buf, sizeof buf, "ERR: %s", sqlite3_errmsg(db));
    return buf;
  }
  int rc = sqlite3_step(st);
  if (rc == SQLITE_ROW) {
    const char *t = (const char *)sqlite3_column_text(st, 0);
    snprintf(buf, sizeof buf, "%s", t ? t : "");
  } else if (rc != SQLITE_DONE) {
    snprintf(buf, sizeof buf, "ERR: %s", sqlite3_errmsg(db));
  }
  sqlite3_finalize(st);
  return buf;
}

static int rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw) {
  (void)sb;
  (void)flag;
  (void)ftw;
  return remove(p);
}

static void vec_for(float *v, long id, long version) {
  v[0] = (float)id;
  v[1] = (float)version;
  v[2] = 1.0f;
  v[3] = 0.0f;
}

// The child: write, update, delete, sync, search, until killed.
static void child_main(const char *dbfile, int query_mode) {
  sqlite3 *db = open_db(dbfile);
  sqlite3_stmt *ins = NULL, *upd = NULL, *del = NULL, *srch = NULL;
  sqlite3_prepare_v2(db, "INSERT INTO docs(id, embedding) VALUES(?1, ?2)", -1, &ins, NULL);
  sqlite3_prepare_v2(db, "UPDATE docs SET embedding = ?2 WHERE id = ?1", -1, &upd, NULL);
  sqlite3_prepare_v2(db, "DELETE FROM docs WHERE id = ?1", -1, &del, NULL);
  sqlite3_prepare_v2(db,
                     "SELECT count(*) FROM (SELECT pk FROM docs_idx WHERE embedding MATCH ?1 AND k = 3)",
                     -1, &srch, NULL);
  long next = atol(one(db, "SELECT coalesce(max(id), 0) + 1 FROM docs"));
  unsigned seed = (unsigned)getpid();
  for (long round = 1;; round++) {
    if (sqlite3_exec(db, "BEGIN IMMEDIATE", NULL, NULL, NULL) != SQLITE_OK) continue;
    float v[DIM];
    for (int j = 0; j < 20; j++, next++) {
      vec_for(v, next, 0);
      sqlite3_reset(ins);
      sqlite3_bind_int64(ins, 1, next);
      sqlite3_bind_blob(ins, 2, v, sizeof v, SQLITE_TRANSIENT);
      sqlite3_step(ins);
    }
    if (round % 3 == 0 && next > 10) {
      long id = 1 + (long)(rand_r(&seed) % (unsigned)(next - 1));
      vec_for(v, id, round);
      sqlite3_reset(upd);
      sqlite3_bind_int64(upd, 1, id);
      sqlite3_bind_blob(upd, 2, v, sizeof v, SQLITE_TRANSIENT);
      sqlite3_step(upd);
      id = 1 + (long)(rand_r(&seed) % (unsigned)(next - 1));
      sqlite3_reset(del);
      sqlite3_bind_int64(del, 1, id);
      sqlite3_step(del);
    }
    sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    if (round % 2 == 0) {
      if (query_mode) {
        vec_for(v, next - 1, 0);
        sqlite3_reset(srch);
        sqlite3_bind_blob(srch, 1, v, sizeof v, SQLITE_TRANSIENT);
        sqlite3_step(srch);
      } else {
        sqlite3_exec(db, "SELECT zvec_sync('docs_idx')", NULL, NULL, NULL);
      }
    }
  }
}

static void verify(const char *dbfile, int iter, int query_mode) {
  sqlite3 *db = open_db(dbfile);
  const char *r = one(db, "SELECT zvec_check('docs_idx')");
  CHECK(strncmp(r, "ERR", 3) != 0, "iter %d: zvec_check failed after the crash: %s", iter, r);
  r = one(db, "SELECT zvec_sync('docs_idx')");
  CHECK(strncmp(r, "ERR", 3) != 0, "iter %d: sync failed after the crash: %s", iter, r);
  r = one(db, "SELECT zvec_check('docs_idx')");
  CHECK(strstr(r, "\"status\":\"ok\"") != NULL, "iter %d (%s): index differs from the table after a sync: %s", iter,
        query_mode ? "sync=query" : "sync=manual", r);

  // Spot-check searches: a stored vector must find its own row first.
  sqlite3_stmt *pick = NULL, *find = NULL;
  sqlite3_prepare_v2(db, "SELECT id, embedding FROM docs ORDER BY random() LIMIT 5", -1, &pick, NULL);
  sqlite3_prepare_v2(db,
                     "SELECT pk FROM docs_idx WHERE embedding MATCH ?1 AND k = 1",
                     -1, &find, NULL);
  while (sqlite3_step(pick) == SQLITE_ROW) {
    sqlite3_int64 id = sqlite3_column_int64(pick, 0);
    sqlite3_reset(find);
    sqlite3_bind_blob(find, 1, sqlite3_column_blob(pick, 1), sqlite3_column_bytes(pick, 1), SQLITE_TRANSIENT);
    int rc = sqlite3_step(find);
    const char *pk = rc == SQLITE_ROW ? (const char *)sqlite3_column_text(find, 0) : "";
    CHECK(rc == SQLITE_ROW && atoll(pk) == id, "iter %d: row %lld not found by its own vector (got %s)", iter,
          (long long)id, pk);
  }
  sqlite3_finalize(pick);
  sqlite3_finalize(find);
  sqlite3_close(db);
}

static void run_mode(const char *dir, int query_mode, int iterations) {
  char dbfile[512], buf[512];
  snprintf(dbfile, sizeof dbfile, "%s/%s.db", dir, query_mode ? "query" : "manual");
  sqlite3 *db = open_db(dbfile);
  snprintf(buf, sizeof buf,
           "CREATE TABLE docs(id INTEGER PRIMARY KEY, embedding BLOB);"
           "CREATE VIRTUAL TABLE docs_idx USING zvec(base=docs, key=id, column=embedding, "
           "dim=%d, metric=l2, sync=%s)",
           DIM, query_mode ? "query" : "manual");
  char *err = NULL;
  int rc = sqlite3_exec(db, buf, NULL, NULL, &err);
  CHECK(rc == SQLITE_OK, "setup: %s", err ? err : "?");
  sqlite3_free(err);
  sqlite3_close(db);

  for (int i = 0; i < iterations; i++) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
      child_main(dbfile, query_mode);
      _exit(0);
    }
    struct timespec ts = {0, (long)(30 + rand() % 220) * 1000000L};
    nanosleep(&ts, NULL);
    kill(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status), "iter %d: the writer exited by itself (status %d)", i, status);
    verify(dbfile, i, query_mode);
  }
  db = open_db(dbfile);
  printf("%s: %d kills survived, %s rows, index consistent\n", query_mode ? "sync=query " : "sync=manual",
         iterations, one(db, "SELECT count(*) FROM docs"));
  sqlite3_close(db);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <path-to-zvec-extension>\n", argv[0]);
    return 1;
  }
  g_ext = argv[1];
  const char *n = getenv("ZVEC_CRASH_ITERATIONS");
  int iterations = n != NULL && atoi(n) > 0 ? atoi(n) : 12;
  const char *s = getenv("ZVEC_CRASH_SEED");
  srand(s != NULL ? (unsigned)atoi(s) : (unsigned)time(NULL));

  char tmpl[] = "/tmp/zvec_crash_XXXXXX";
  char *dir = mkdtemp(tmpl);
  if (dir == NULL) {
    fprintf(stderr, "mkdtemp failed\n");
    return 1;
  }
  run_mode(dir, 0, iterations);
  run_mode(dir, 1, iterations);
  nftw(dir, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
  printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
#endif
