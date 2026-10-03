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

// Smoke test for the zvec SQLite loadable extension: the module loads into a
// real SQLite host, registers its functions, and reaches the zvec core.
//
// Usage: sqlite_ext_test <path-to-zvec-extension>
// Exit codes: 0 pass, 1 fail, 77 skipped (host SQLite cannot load extensions).

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zvec/c_api.h"

#define SKIP_EXIT_CODE 77

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, ...)                                        \
  do {                                                          \
    g_checks++;                                                 \
    if (!(cond)) {                                              \
      g_failures++;                                             \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
      fprintf(stderr, __VA_ARGS__);                             \
      fprintf(stderr, "\n");                                    \
    }                                                           \
  } while (0)

static int enable_load_extension(sqlite3 *db) {
#ifdef SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION
  return sqlite3_db_config(db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, NULL);
#else
  return sqlite3_enable_load_extension(db, 1);
#endif
}

// Opens an in-memory database with extension loading enabled. Loading is a
// per-connection setting, so every connection goes through here.
static sqlite3 *open_db(void) {
  sqlite3 *db = NULL;
  if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
    fprintf(stderr, "cannot open :memory: database: %s\n",
            db ? sqlite3_errmsg(db) : "out of memory");
    exit(1);
  }
  int rc = enable_load_extension(db);
  if (rc != SQLITE_OK) {
    printf("SKIP: this SQLite build cannot load extensions (rc=%d)\n", rc);
    sqlite3_close(db);
    exit(SKIP_EXIT_CODE);
  }
  return db;
}

// Runs a query and copies the first column of the first row as text into buf.
// Returns the sqlite result code of the last step (SQLITE_ROW on success).
static int query_text(sqlite3 *db, const char *sql, char *buf, size_t buf_len,
                      int *is_null) {
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  buf[0] = '\0';
  if (is_null) {
    *is_null = 0;
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    if (sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
      if (is_null) {
        *is_null = 1;
      }
    } else {
      const unsigned char *text = sqlite3_column_text(stmt, 0);
      snprintf(buf, buf_len, "%s", text ? (const char *)text : "");
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static long long query_int(sqlite3 *db, const char *sql, int *rc_out) {
  sqlite3_stmt *stmt = NULL;
  long long value = -1;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      value = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
  }
  if (rc_out) {
    *rc_out = rc;
  }
  return value;
}

static void test_functions_absent_before_load(void) {
  sqlite3 *db = open_db();
  char buf[256];
  int rc = query_text(db, "SELECT zvec_version()", buf, sizeof(buf), NULL);
  CHECK(rc != SQLITE_ROW && rc != SQLITE_OK,
        "zvec_version() must not exist before the extension is loaded");
  sqlite3_close(db);
}

static void test_load_default_entry_point(const char *ext_path) {
  sqlite3 *db = open_db();
  char *err = NULL;
  int rc = sqlite3_load_extension(db, ext_path, NULL, &err);
  CHECK(rc == SQLITE_OK, "load with default entry point failed: %s",
        err ? err : "(no message)");
  sqlite3_free(err);

  // zvec_version() returns the same string as the C API of the loaded core.
  char buf[512];
  rc = query_text(db, "SELECT zvec_version()", buf, sizeof(buf), NULL);
  CHECK(rc == SQLITE_ROW, "SELECT zvec_version() rc=%d (%s)", rc,
        sqlite3_errmsg(db));
  CHECK(buf[0] != '\0', "zvec_version() returned an empty string");
  CHECK(strcmp(buf, zvec_get_version()) == 0,
        "zvec_version() = '%s' but zvec_get_version() = '%s'", buf,
        zvec_get_version());

  // Deterministic scalar usable per row (version bound as a parameter).
  char sql[128];
  long long n = -1;
  {
    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(
        db,
        "SELECT count(*) FROM (SELECT zvec_version() AS v FROM "
        "(SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3)) WHERE v = ?1",
        -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, buf, -1, SQLITE_TRANSIENT);
      rc = sqlite3_step(stmt);
      if (rc == SQLITE_ROW) {
        n = sqlite3_column_int64(stmt, 0);
      }
      sqlite3_finalize(stmt);
    }
  }
  CHECK(rc == SQLITE_ROW && n == 3, "per-row zvec_version(): rc=%d n=%lld", rc,
        n);

  // zvec_check_version agrees with the C API.
  CHECK(query_int(db, "SELECT zvec_check_version(0, 0, 0)", &rc) == 1 &&
            rc == SQLITE_ROW,
        "zvec_check_version(0,0,0) should be 1");
  CHECK(query_int(db, "SELECT zvec_check_version(9999, 0, 0)", &rc) == 0 &&
            rc == SQLITE_ROW,
        "zvec_check_version(9999,0,0) should be 0");
  snprintf(sql, sizeof(sql), "SELECT zvec_check_version(%d, %d, %d)",
           zvec_get_version_major(), zvec_get_version_minor(),
           zvec_get_version_patch());
  CHECK(query_int(db, sql, &rc) == 1 && rc == SQLITE_ROW,
        "current version must satisfy zvec_check_version: %s", sql);

  // NULL propagates.
  int is_null = 0;
  rc = query_text(db, "SELECT zvec_check_version(NULL, 0, 0)", buf,
                  sizeof(buf), &is_null);
  CHECK(rc == SQLITE_ROW && is_null, "NULL input should give NULL output");

  // Bad arguments produce SQL errors, not crashes.
  rc = query_text(db, "SELECT zvec_check_version(1)", buf, sizeof(buf), NULL);
  CHECK(rc != SQLITE_ROW && rc != SQLITE_OK,
        "wrong argument count should be an error");
  rc = query_text(db, "SELECT zvec_check_version('a', 'b', 'c')", buf,
                  sizeof(buf), NULL);
  CHECK(rc != SQLITE_ROW && rc != SQLITE_OK,
        "non-integer arguments should be an error");
  rc = query_text(db, "SELECT zvec_check_version(9223372036854775807, 0, 0)",
                  buf, sizeof(buf), NULL);
  CHECK(rc != SQLITE_ROW && rc != SQLITE_OK,
        "out-of-range integer argument should be an error");

  sqlite3_close(db);
}

static void test_load_explicit_entry_point(const char *ext_path) {
  sqlite3 *db = open_db();
  char *err = NULL;
  int rc = sqlite3_load_extension(db, ext_path, "sqlite3_zvec_init", &err);
  CHECK(rc == SQLITE_OK, "load with explicit entry point failed: %s",
        err ? err : "(no message)");
  sqlite3_free(err);

  char buf[512];
  rc = query_text(db, "SELECT zvec_version()", buf, sizeof(buf), NULL);
  CHECK(rc == SQLITE_ROW && buf[0] != '\0', "zvec_version() rc=%d", rc);
  sqlite3_close(db);
}

static void test_load_wrong_entry_point(const char *ext_path) {
  sqlite3 *db = open_db();
  char *err = NULL;
  int rc = sqlite3_load_extension(db, ext_path, "sqlite3_no_such_init", &err);
  CHECK(rc != SQLITE_OK, "unknown entry point must fail to load");
  CHECK(err != NULL, "failed load should report an error message");
  sqlite3_free(err);
  sqlite3_close(db);
}

static void test_load_twice_and_reopen(const char *ext_path) {
  // Loading into a second connection after the first was closed must work
  // (the module stays resident; no double-registration problems).
  int i;
  for (i = 0; i < 2; i++) {
    sqlite3 *db = open_db();
    char *err = NULL;
    int rc = sqlite3_load_extension(db, ext_path, NULL, &err);
    CHECK(rc == SQLITE_OK, "load #%d failed: %s", i,
          err ? err : "(no message)");
    sqlite3_free(err);
    // Loading the same module again into the same connection is harmless.
    rc = sqlite3_load_extension(db, ext_path, NULL, &err);
    CHECK(rc == SQLITE_OK, "repeat load #%d failed: %s", i,
          err ? err : "(no message)");
    sqlite3_free(err);
    sqlite3_close(db);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <path-to-zvec-extension>\n", argv[0]);
    return 1;
  }
  const char *ext_path = argv[1];

  printf("sqlite %s, zvec %s\n", sqlite3_libversion(), zvec_get_version());

  test_functions_absent_before_load();
  test_load_default_entry_point(ext_path);
  test_load_explicit_entry_point(ext_path);
  test_load_wrong_entry_point(ext_path);
  test_load_twice_and_reopen(ext_path);

  printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
