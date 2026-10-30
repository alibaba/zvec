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

// Derived-index mode of the zvec SQLite extension: a virtual table declared
// with base=/key=/column= keeps a rebuildable zvec index of a table that stays
// the source of truth. Everything here goes through SQL only.
//
// Usage: derived_test <path-to-zvec-extension>
// Exit codes: 0 pass, 1 fail, 77 skipped.

#define _XOPEN_SOURCE 700

#include <sys/stat.h>
#include <ftw.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SKIP_EXIT_CODE 77

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

// float32 vectors of dimension 4
#define V_X "x'0000803f000000000000000000000000'"  // (1,0,0,0)
#define V_Y "x'000000000000803f0000000000000000'"  // (0,1,0,0)

static const char *g_ext = NULL;

static int enable_load_extension(sqlite3 *db) {
#ifdef SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION
  return sqlite3_db_config(db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, NULL);
#else
  return sqlite3_enable_load_extension(db, 1);
#endif
}

static sqlite3 *open_db(const char *file) {
  sqlite3 *db = NULL;
  if (sqlite3_open(file, &db) != SQLITE_OK) {
    fprintf(stderr, "cannot open %s\n", file);
    exit(1);
  }
  int rc = enable_load_extension(db);
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

static sqlite3 *open_db_ro(const char *file) {
  sqlite3 *db = NULL;
  if (sqlite3_open_v2(file, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
    fprintf(stderr, "cannot open %s read-only\n", file);
    exit(1);
  }
  enable_load_extension(db);
  char *err = NULL;
  if (sqlite3_load_extension(db, g_ext, "sqlite3_zvec_init", &err) !=
      SQLITE_OK) {
    fprintf(stderr, "cannot load %s: %s\n", g_ext, err ? err : "?");
    exit(1);
  }
  return db;
}

// Runs one statement; returns the result code and copies the message.
static int run(sqlite3 *db, const char *sql, char *msg, size_t len) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  snprintf(msg, len, "%s", err ? err : "");
  sqlite3_free(err);
  return rc;
}

// First column of the first row as text ("" for no row, "ERR: ..." on error).
// Valid until the next call.
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

#define EXPECT_ONE(db, sql, want)                                             \
  do {                                                                        \
    const char *got_ = one((db), (sql));                                      \
    CHECK(strcmp(got_, (want)) == 0,                                          \
          "%s\n        got:    %s\n        want:   %s", (sql), got_, (want)); \
  } while (0)

#define MUST(db, sql)                               \
  do {                                              \
    char m_[512];                                   \
    int rc_ = run((db), (sql), m_, sizeof m_);      \
    CHECK(rc_ == SQLITE_OK, "%s -> %s", (sql), m_); \
  } while (0)

// The statement must fail and the message must contain `needle`.
#define MUST_FAIL(db, sql, needle)                                         \
  do {                                                                     \
    char m_[512];                                                          \
    int rc_ = run((db), (sql), m_, sizeof m_);                             \
    CHECK(rc_ != SQLITE_OK, "unexpectedly succeeded: %s", (sql));          \
    CHECK(strstr(m_, (needle)) != NULL,                                    \
          "%s\n        message: %s\n        wanted:  ...%s...", (sql), m_, \
          (needle));                                                       \
  } while (0)

static int is_dir(const char *path) {
  struct stat sb;
  return stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
}

static int rm_cb(const char *p, const struct stat *sb, int flag,
                 struct FTW *ftw) {
  (void)sb;
  (void)flag;
  (void)ftw;
  return remove(p);
}

static const char *kOutbox =
    "SELECT coalesce(group_concat(op || ':' || pk, ','), '') FROM "
    "(SELECT op, pk FROM docs_idx_outbox ORDER BY seq)";
static const char *kSearchX =
    "SELECT coalesce(group_concat(pk || '|' || distance, ','), '') FROM "
    "(SELECT pk, distance FROM docs_idx WHERE embedding MATCH " V_X
    " AND k = 5 ORDER BY distance, pk)";
static const char *kKeysX =
    "SELECT coalesce(group_concat(pk, ','), '') FROM "
    "(SELECT pk FROM docs_idx WHERE embedding MATCH " V_X
    " AND k = 5 ORDER BY distance, pk)";

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <path-to-zvec-extension>\n", argv[0]);
    return 1;
  }
  g_ext = argv[1];

  char tmpl[] = "/tmp/zvec_derived_test_XXXXXX";
  char *dir = mkdtemp(tmpl);
  if (dir == NULL) {
    fprintf(stderr, "mkdtemp failed\n");
    return 1;
  }
  char dbfile[256], idxdir[300], buf[1500];
  snprintf(dbfile, sizeof dbfile, "%s/t.db", dir);
  snprintf(idxdir, sizeof idxdir, "%s/t.db.zvec/docs_idx", dir);

  sqlite3 *db = open_db(dbfile);
  MUST(db,
       "CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT, embedding BLOB)");

  // ---- create ----
  MUST(db,
       "CREATE VIRTUAL TABLE docs_idx USING zvec(base=docs, key=id, "
       "column=embedding, dim=4, metric=l2, sync=manual)");
  EXPECT_ONE(
      db,
      "SELECT group_concat(name, ',') FROM (SELECT name FROM sqlite_master "
      "WHERE name LIKE 'docs_idx%' AND type = 'table' ORDER BY name)",
      "docs_idx,docs_idx_config,docs_idx_outbox,docs_idx_state");
  EXPECT_ONE(db, "SELECT count(*) FROM sqlite_master WHERE type = 'trigger'",
             "3");
  EXPECT_ONE(db, "SELECT value FROM docs_idx_config WHERE key = 'dim'", "4");
  EXPECT_ONE(db, "SELECT value FROM docs_idx_config WHERE key = 'metric'",
             "l2");
  snprintf(buf, sizeof buf,
           "SELECT value = '%s' FROM docs_idx_config WHERE key = 'path'",
           idxdir);
  EXPECT_ONE(db, buf, "1");
  CHECK(is_dir(idxdir), "index directory %s was not created", idxdir);

  // ---- triggers queue exactly the changed keys ----
  MUST(db, "INSERT INTO docs VALUES(1, 'a', " V_X ")");
  MUST(db, "INSERT INTO docs VALUES(2, 'b', NULL)");
  MUST(db, "UPDATE docs SET embedding = " V_X " WHERE id = 1");  // unchanged
  MUST(db, "UPDATE docs SET body = 'a2' WHERE id = 1");          // other column
  MUST(db, "UPDATE docs SET id = 7 WHERE id = 1");               // key change
  MUST(db, "DELETE FROM docs WHERE id = 2");
  EXPECT_ONE(db, kOutbox, "1:1,1:2,2:1,1:7,2:2");

  // ---- sync and search ----
  EXPECT_ONE(db, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":1,\"deleted\":2,\"skipped\":0}");
  EXPECT_ONE(db, "SELECT count(*) FROM docs_idx_outbox", "0");
  EXPECT_ONE(db, "SELECT watermark FROM docs_idx_state", "5");
  EXPECT_ONE(db, kSearchX, "7|0.0");
  EXPECT_ONE(db, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":0,\"deleted\":0,\"skipped\":0}");

  MUST(db, "INSERT INTO docs VALUES(8, 'h', " V_Y ")");
  MUST(db, "INSERT INTO docs VALUES(9, 'i', NULL)");
  MUST(db, "INSERT INTO docs VALUES(10, 'j', zeroblob(7))");  // wrong size
  EXPECT_ONE(db, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":1,\"deleted\":1,\"skipped\":1}");
  EXPECT_ONE(db, kSearchX, "7|0.0,8|2.0");

  // ---- join back to the base table; a lagging index never leaks deleted rows
  // ----
  const char *kJoin =
      "SELECT coalesce(group_concat(body, ','), '') FROM (SELECT d.body AS "
      "body "
      "FROM docs_idx i JOIN docs d ON d.id = i.pk WHERE i.embedding MATCH " V_X
      " AND i.k = 5 ORDER BY i.distance)";
  EXPECT_ONE(db, kJoin, "a2,h");
  MUST(db, "DELETE FROM docs WHERE id = 8");  // no sync yet
  EXPECT_ONE(db, kKeysX, "7,8");              // index lags ...
  EXPECT_ONE(db, kJoin, "a2");                // ... the join hides it
  EXPECT_ONE(db, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":0,\"deleted\":1,\"skipped\":0}");
  EXPECT_ONE(db, kKeysX, "7");

  // ---- defensive mode: shadow state is protected, writes and sync still work
  // ----
#ifdef SQLITE_DBCONFIG_DEFENSIVE
  sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, 1, NULL);
  MUST_FAIL(db, "UPDATE docs_idx_state SET watermark = 99",
            "may not be modified");
  MUST_FAIL(db, "UPDATE docs_idx_config SET value = 'x' WHERE key = 'dim'",
            "may not be modified");
  MUST(db, "INSERT INTO docs VALUES(11, 'k', " V_Y ")");
  EXPECT_ONE(db, kOutbox, "1:11");
  EXPECT_ONE(db, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
  sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, 0, NULL);
#endif
  EXPECT_ONE(db, kKeysX, "7,11");

  // ---- direct-only: not usable from views ----
  MUST(db, "CREATE VIEW v_sync AS SELECT zvec_sync('docs_idx') AS s");
  MUST_FAIL(db, "SELECT * FROM v_sync", "unsafe use of");
  MUST(db,
       "CREATE VIEW v_search AS SELECT pk FROM docs_idx WHERE embedding "
       "MATCH " V_X);
  MUST_FAIL(db, "SELECT * FROM v_search", "unsafe use of");
  MUST(db, "DROP VIEW v_sync");
  MUST(db, "DROP VIEW v_search");

  // ---- persistence and a second connection ----
  sqlite3_close(db);
  db = open_db(dbfile);
  EXPECT_ONE(db, kKeysX, "7,11");
  sqlite3 *db2 = open_db(dbfile);
  EXPECT_ONE(db2, kKeysX, "7,11");
  EXPECT_ONE(db2, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":0,\"deleted\":0,\"skipped\":0}");
  sqlite3_close(db2);

  // ---- many rows: batched sync ----
  MUST(db,
       "WITH RECURSIVE c(x) AS (SELECT 100 UNION ALL SELECT x + 1 FROM c WHERE "
       "x < 1299) "
       "INSERT INTO docs SELECT x, 'bulk', zeroblob(16) FROM c");
  EXPECT_ONE(db, "SELECT zvec_sync('docs_idx')",
             "{\"upserted\":1200,\"deleted\":0,\"skipped\":0}");
  EXPECT_ONE(db,
             "SELECT count(*) FROM (SELECT pk FROM docs_idx WHERE embedding "
             "MATCH zeroblob(16) AND k = 10)",
             "10");

  // ---- TEXT primary key in another table ----
  MUST(db, "CREATE TABLE notes(slug TEXT PRIMARY KEY, v BLOB)");
  snprintf(buf, sizeof buf,
           "CREATE VIRTUAL TABLE notes_idx USING zvec(base=notes, key=slug, "
           "column=v, "
           "dim=4, metric=cosine, path='%s/notes_idx')",
           dir);
  MUST(db, buf);
  MUST(db, "INSERT INTO notes VALUES('alpha', " V_X "), ('beta', " V_Y ")");
  EXPECT_ONE(db, "SELECT zvec_sync('notes_idx')",
             "{\"upserted\":2,\"deleted\":0,\"skipped\":0}");
  EXPECT_ONE(db,
             "SELECT n.slug FROM notes_idx i JOIN notes n ON n.slug = i.pk "
             "WHERE i.v MATCH " V_X " AND i.k = 1",
             "alpha");

  // ---- sync on query (the default) and filter pushdown ----
  MUST(db,
       "CREATE TABLE items(id INTEGER PRIMARY KEY, cat TEXT, price REAL, n "
       "INTEGER, v BLOB)");
  MUST(db, "INSERT INTO items VALUES(1,'a',1.5,10," V_X "),(2,'b',7.25,20," V_Y
           "),(3,'a',9.0,30," V_X ")");
  snprintf(buf, sizeof buf,
           "CREATE VIRTUAL TABLE items_idx USING zvec(base=items, key=id, "
           "column=v, dim=4, "
           "metric=l2, filter='cat,price,n', path='%s/items_idx')",
           dir);
  MUST(db, buf);
#define ITEMS(extra)                                                           \
  "SELECT coalesce(group_concat(pk, ','), '') FROM (SELECT pk FROM items_idx " \
  "WHERE v MATCH " V_X " AND k = 5 " extra " ORDER BY pk)"
  EXPECT_ONE(db, ITEMS(""), "1,2,3");
  MUST(db, "INSERT INTO items VALUES(4,'b',2.0,40," V_X ")");
  EXPECT_ONE(db, ITEMS(""), "1,2,3,4");
  EXPECT_ONE(db, ITEMS("AND cat = 'a'"), "1,3");
  EXPECT_ONE(db, ITEMS("AND price < 8"), "1,2,4");
  EXPECT_ONE(db, ITEMS("AND n >= 20 AND cat = 'b'"), "2,4");
  EXPECT_ONE(db, ITEMS("AND cat = 'a' AND price < 5"), "1");
  EXPECT_ONE(db, ITEMS("AND cat = 'it''s'"), "");
  EXPECT_ONE(db, ITEMS("AND n BETWEEN 15 AND 35"), "2,3");
  MUST(db, "UPDATE items SET cat = 'z' WHERE id = 1");
  EXPECT_ONE(db, ITEMS("AND cat = 'z'"), "1");
  MUST(db, "DELETE FROM items WHERE id = 3");
  EXPECT_ONE(db, ITEMS(""), "1,2,4");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e8 USING zvec(base=items, key=id, column=v, "
            "dim=4, filter='v')",
            "key or the vector");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e9 USING zvec(base=items, key=id, column=v, "
            "dim=4, filter='nope')",
            "not found");
#undef ITEMS

  // ---- bad declarations fail cleanly and leave nothing behind ----
  MUST_FAIL(
      db,
      "CREATE VIRTUAL TABLE e1 USING zvec(base=docs, key=id, column=embedding)",
      "required");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e2 USING zvec(base=nope, key=id, "
            "column=embedding, dim=4)",
            "does not exist");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e3 USING zvec(base=docs, key=body, "
            "column=embedding, dim=4)",
            "PRIMARY KEY");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e4 USING zvec(base=docs, key=id, "
            "column=embedding, dim=0)",
            "dim");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e5 USING zvec(base=docs, key=id, "
            "column=embedding, dim=4, bogus=1)",
            "unknown option");
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE e6 USING zvec(base=docs, key=id x, "
            "column=embedding, dim=4)",
            "plain identifiers");
  snprintf(buf, sizeof buf,
           "CREATE VIRTUAL TABLE e7 USING zvec(base=docs, key=id, "
           "column=embedding, dim=4, path='%s/no/such/parent')",
           dir);
  MUST_FAIL(db, buf, "no/such/parent");
  EXPECT_ONE(db, "SELECT count(*) FROM sqlite_master WHERE name GLOB 'e[0-9]*'",
             "0");

  sqlite3 *mem = open_db(":memory:");
  MUST(mem, "CREATE TABLE docs(id INTEGER PRIMARY KEY, embedding BLOB)");
  MUST_FAIL(mem,
            "CREATE VIRTUAL TABLE t USING zvec(base=docs, key=id, "
            "column=embedding, dim=4)",
            "path=");
  sqlite3_close(mem);

  // ---- DROP removes the tables and triggers, never the directory ----
  MUST(db, "DROP TABLE docs_idx");
  EXPECT_ONE(db,
             "SELECT count(*) FROM sqlite_master WHERE name LIKE 'docs_idx%'",
             "0");
  EXPECT_ONE(db,
             "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' AND "
             "name LIKE 'docs_idx%'",
             "0");
  CHECK(is_dir(idxdir), "DROP TABLE removed the index directory");
  MUST(db, "INSERT INTO docs VALUES(5000, 'after drop', " V_X
           ")");  // no dangling trigger
  MUST_FAIL(db,
            "CREATE VIRTUAL TABLE docs_idx USING zvec(base=docs, key=id, "
            "column=embedding, dim=4, metric=l2)",
            "exist");

  // ---- check, rebuild, read-only open, identity, missing directory,
  // drop_index ----
  {
    char f[512], d1[600], d2[600], d3[600];
    snprintf(f, sizeof f, "%s/life.db", dir);
    snprintf(d1, sizeof d1, "%s.zvec/i1", f);
    snprintf(d2, sizeof d2, "%s.zvec/i2", f);
    snprintf(d3, sizeof d3, "%s.zvec/i3", f);
    sqlite3 *a = open_db(f);
    MUST(a, "CREATE TABLE d(id INTEGER PRIMARY KEY, v BLOB)");
    MUST(a, "INSERT INTO d VALUES(1, " V_X "), (2, " V_Y "), (3, NULL)");
    MUST(a,
         "CREATE VIRTUAL TABLE i1 USING zvec(base=d, key=id, column=v, dim=4, "
         "metric=l2)");
    MUST(a,
         "CREATE VIRTUAL TABLE i2 USING zvec(base=d, key=id, column=v, dim=4, "
         "metric=l2)");
    MUST(a,
         "CREATE VIRTUAL TABLE i3 USING zvec(base=d, key=id, column=v, dim=4, "
         "metric=l2)");
    EXPECT_ONE(a, "SELECT zvec_check('i1')",
               "{\"status\":\"pending\",\"base\":2,\"indexed\":0,\"stale\":0,"
               "\"missing\":2,\"pending\":3}");
    EXPECT_ONE(a, "SELECT zvec_sync('i1')",
               "{\"upserted\":2,\"deleted\":1,\"skipped\":0}");
    EXPECT_ONE(a, "SELECT zvec_check('i1')",
               "{\"status\":\"ok\",\"base\":2,\"indexed\":2,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");
    // A change the triggers never saw: only check can tell, and rebuild repairs
    // it.
    MUST(a, "DROP TRIGGER i1_zvec_ad");
    MUST(a, "DELETE FROM d WHERE id = 2");
    EXPECT_ONE(a, "SELECT zvec_check('i1')",
               "{\"status\":\"repair\",\"base\":1,\"indexed\":2,\"stale\":1,"
               "\"missing\":0,\"pending\":0}");
    MUST(a, "BEGIN");
    MUST_FAIL(a, "SELECT zvec_rebuild('i1')", "inside a transaction");
    MUST(a, "ROLLBACK");
    EXPECT_ONE(a, "SELECT zvec_rebuild('i1')",
               "{\"removed\":1,\"queued\":2,\"upserted\":1,\"deleted\":1,"
               "\"skipped\":0}");
    EXPECT_ONE(a, "SELECT zvec_check('i1')",
               "{\"status\":\"ok\",\"base\":1,\"indexed\":1,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");
    MUST_FAIL(a, "SELECT zvec_check('nope')", "nope");
    MUST(a, "CREATE VIEW vchk AS SELECT zvec_check('i1')");
    MUST_FAIL(a, "SELECT * FROM vchk", "unsafe use of");

    // Identity: a settings row cannot point the table at someone else's
    // directory.
    MUST(a, "SELECT zvec_sync('i2')");
    snprintf(buf, sizeof buf,
             "UPDATE i3_config SET value = '%s' WHERE key = 'path'; "
             "UPDATE i3_config SET value = '0' WHERE key = 'path_default'",
             d2);
    MUST(a, buf);
    sqlite3_close(a);

    // Read-only connection: searches work, changes are refused, nothing is
    // written.
    sqlite3 *ro = open_db_ro(f);
    EXPECT_ONE(ro,
               "SELECT group_concat(pk) FROM (SELECT pk FROM i1 WHERE "
               "embedding_unused IS NULL LIMIT 0)",
               "ERR: no such column: embedding_unused");
    EXPECT_ONE(
        ro,
        "SELECT group_concat(pk) FROM (SELECT pk FROM i1 WHERE v MATCH " V_X
        " AND k = 3)",
        "1");
    MUST_FAIL(ro, "SELECT zvec_sync('i1')", "read-only");
    MUST_FAIL(ro, "SELECT zvec_check('i1')", "read-only");
    MUST_FAIL(ro, "SELECT zvec_rebuild('i1')", "read-only");
    MUST_FAIL(ro, "INSERT INTO d VALUES(9, " V_X ")", "readonly");
    sqlite3_close(ro);

    a = open_db(f);
    MUST_FAIL(a, "SELECT pk FROM i3 WHERE v MATCH " V_X,
              "does not belong to this table");
    MUST(a, "DROP TABLE i3");  // a table that cannot open still drops
    EXPECT_ONE(
        a,
        "SELECT group_concat(pk) FROM (SELECT pk FROM i1 WHERE v MATCH " V_X
        " AND k = 3)",
        "1");

    // Only the database file was copied: the directory is gone.
    sqlite3_close(a);
    snprintf(buf, sizeof buf, "rm -rf '%s'", d1);
    CHECK(system(buf) == 0, "rm -rf %s", d1);
    a = open_db(f);
    MUST_FAIL(a, "SELECT pk FROM i1 WHERE v MATCH " V_X, "zvec_rebuild");
    MUST_FAIL(a, "SELECT zvec_check('i1')", "zvec_rebuild");
    EXPECT_ONE(a, "SELECT zvec_rebuild('i1')",
               "{\"removed\":0,\"queued\":2,\"upserted\":1,\"deleted\":1,"
               "\"skipped\":0}");
    EXPECT_ONE(
        a,
        "SELECT group_concat(pk) FROM (SELECT pk FROM i1 WHERE v MATCH " V_X
        " AND k = 3)",
        "1");

    // zvec_drop_index: table and directory together; still in use elsewhere ->
    // kept.
    sqlite3 *b = open_db(f);
    EXPECT_ONE(
        b,
        "SELECT group_concat(pk) FROM (SELECT pk FROM i2 WHERE v MATCH " V_X
        " AND k = 3)",
        "1");
    snprintf(buf, sizeof buf,
             "{\"dropped\":\"i2\",\"directory\":\"kept: still in use by "
             "another connection or process\",\"path\":\"%s\"}",
             d2);
    EXPECT_ONE(a, "SELECT zvec_drop_index('i2')", buf);
    CHECK(is_dir(d2), "an index still in use was deleted");
    sqlite3_close(b);
    snprintf(buf, sizeof buf,
             "{\"dropped\":\"i1\",\"directory\":\"removed\",\"path\":\"%s\"}",
             d1);
    EXPECT_ONE(a, "SELECT zvec_drop_index('i1')", buf);
    CHECK(!is_dir(d1), "the index directory was not removed");
    MUST_FAIL(a, "SELECT zvec_drop_index('nope')", "nope");
    MUST(a, "BEGIN");
    MUST_FAIL(a, "SELECT zvec_drop_index('nope')", "inside a transaction");
    MUST(a, "ROLLBACK");
    sqlite3_close(a);
    (void)d3;
  }


  // ---- UNIQUE conflicts and transactions --------------------------------
  {
    char f[512];
    snprintf(f, sizeof f, "%s/u.db", dir);
    sqlite3 *a = open_db(f);
    MUST(a, "CREATE TABLE p(id INTEGER PRIMARY KEY, slug TEXT UNIQUE, v BLOB)");
    MUST(a, "INSERT INTO p VALUES(1, 'a', " V_X "), (2, 'b', " V_Y ")");
    MUST(a,
         "CREATE VIRTUAL TABLE pi USING zvec(base=p, key=id, column=v, dim=4, "
         "metric=l2)");
#define PKEYS                                                              \
  "SELECT coalesce(group_concat(pk), '') FROM (SELECT pk FROM pi WHERE v " \
  "MATCH " V_X " AND k = 5 ORDER BY distance, pk)"
    EXPECT_ONE(a, "SELECT zvec_sync('pi')",
               "{\"upserted\":2,\"deleted\":0,\"skipped\":0}");

    // INSERT OR REPLACE removes id 1 (same slug) without a DELETE trigger.
    MUST(a, "INSERT OR REPLACE INTO p VALUES(3, 'a', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('pi')",
               "{\"upserted\":1,\"deleted\":1,\"skipped\":0}");
    EXPECT_ONE(a, PKEYS, "3,2");
    EXPECT_ONE(a, "SELECT zvec_check('pi')",
               "{\"status\":\"ok\",\"base\":2,\"indexed\":2,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");

    // UPDATE OR REPLACE removes id 2.
    MUST(a, "UPDATE OR REPLACE p SET slug = 'b' WHERE id = 3");
    EXPECT_ONE(a, "SELECT zvec_sync('pi')",
               "{\"upserted\":0,\"deleted\":1,\"skipped\":0}");
    EXPECT_ONE(a, PKEYS, "3");

    // OR IGNORE keeps the row; the queued probe changes nothing.
    MUST(a, "INSERT OR IGNORE INTO p VALUES(4, 'b', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('pi')",
               "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
    EXPECT_ONE(a, PKEYS, "3");
    EXPECT_ONE(a, "SELECT zvec_check('pi')",
               "{\"status\":\"ok\",\"base\":1,\"indexed\":1,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");

    // Nothing is synced while this connection has uncommitted changes, so a
    // ROLLBACK cannot leave documents behind.
    MUST(a, "BEGIN");
    MUST(a, "INSERT INTO p VALUES(5, 'e', " V_X ")");
    EXPECT_ONE(a, PKEYS, "3");  // not applied yet: the transaction is open
    MUST_FAIL(a, "SELECT zvec_sync('pi')", "uncommitted changes");
    MUST(a, "ROLLBACK");
    EXPECT_ONE(a, PKEYS, "3");
    EXPECT_ONE(a, "SELECT zvec_check('pi')",
               "{\"status\":\"ok\",\"base\":1,\"indexed\":1,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");
    MUST(a, "BEGIN");
    MUST(a, "INSERT INTO p VALUES(6, 'f', " V_X ")");
    MUST(a, "COMMIT");
    EXPECT_ONE(a, PKEYS, "3,6");  // committed: the next search applies it
    MUST(a, "BEGIN");             // a read transaction may sync
    EXPECT_ONE(a, "SELECT count(*) FROM p", "2");
    MUST(a, "INSERT INTO p VALUES(7, 'g', NULL)");
    MUST(a, "ROLLBACK");

    // Partial and expression UNIQUE indexes cannot be probed: zvec_check sees
    // the stale document and zvec_rebuild repairs it.
    MUST(a, "CREATE TABLE q(id INTEGER PRIMARY KEY, s TEXT, v BLOB)");
    MUST(a, "CREATE UNIQUE INDEX q_s ON q(s) WHERE s IS NOT NULL");
    MUST(a, "INSERT INTO q VALUES(1, 'a', " V_X ")");
    MUST(a,
         "CREATE VIRTUAL TABLE qi USING zvec(base=q, key=id, column=v, dim=4, "
         "metric=l2)");
    EXPECT_ONE(a, "SELECT zvec_sync('qi')",
               "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
    MUST(a, "INSERT OR REPLACE INTO q VALUES(2, 'a', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('qi')",
               "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
    EXPECT_ONE(a, "SELECT zvec_check('qi')",
               "{\"status\":\"repair\",\"base\":1,\"indexed\":2,\"stale\":1,"
               "\"missing\":0,\"pending\":0}");
    EXPECT_ONE(a, "SELECT zvec_rebuild('qi')",
               "{\"removed\":1,\"queued\":1,\"upserted\":1,\"deleted\":0,"
               "\"skipped\":0}");
    EXPECT_ONE(a, "SELECT zvec_check('qi')",
               "{\"status\":\"ok\",\"base\":1,\"indexed\":1,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");

    // A text key on a rowid table: an explicit rowid can replace another row.
    MUST(a, "CREATE TABLE r(slug TEXT PRIMARY KEY, v BLOB)");
    MUST(a, "INSERT INTO r(rowid, slug, v) VALUES(1, 'x', " V_X ")");
    MUST(a,
         "CREATE VIRTUAL TABLE ri USING zvec(base=r, key=slug, column=v, "
         "dim=4, metric=l2)");
    EXPECT_ONE(a, "SELECT zvec_sync('ri')",
               "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
    MUST(a, "INSERT OR REPLACE INTO r(rowid, slug, v) VALUES(1, 'y', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('ri')",
               "{\"upserted\":1,\"deleted\":1,\"skipped\":0}");
    EXPECT_ONE(a, "SELECT zvec_check('ri')",
               "{\"status\":\"ok\",\"base\":1,\"indexed\":1,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");
    // WITHOUT ROWID has no rowid to probe, and still creates cleanly.
    MUST(a,
         "CREATE TABLE w(k TEXT PRIMARY KEY, u TEXT UNIQUE, v BLOB) "
         "WITHOUT ROWID");
    MUST(a, "INSERT INTO w VALUES('k1', 'u1', " V_X ")");
    MUST(a,
         "CREATE VIRTUAL TABLE wi USING zvec(base=w, key=k, column=v, dim=4, "
         "metric=l2)");
    EXPECT_ONE(a, "SELECT zvec_sync('wi')",
               "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
    MUST(a, "INSERT OR REPLACE INTO w VALUES('k2', 'u1', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('wi')",
               "{\"upserted\":1,\"deleted\":1,\"skipped\":0}");

    // The probe must use the collation of the UNIQUE index, not BINARY.
    MUST(a,
         "CREATE TABLE n(id INTEGER PRIMARY KEY, name TEXT UNIQUE COLLATE "
         "NOCASE, tag TEXT, v BLOB)");
    MUST(a, "CREATE UNIQUE INDEX n_tag ON n(tag COLLATE NOCASE)");
    MUST(a, "INSERT INTO n VALUES(1, 'Alice', 'T1', " V_X ")");
    MUST(a,
         "CREATE VIRTUAL TABLE ni USING zvec(base=n, key=id, column=v, dim=4, "
         "metric=l2)");
    EXPECT_ONE(a, "SELECT zvec_sync('ni')",
               "{\"upserted\":1,\"deleted\":0,\"skipped\":0}");
    MUST(a, "INSERT OR REPLACE INTO n VALUES(2, 'ALICE', 'x', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('ni')",
               "{\"upserted\":1,\"deleted\":1,\"skipped\":0}");
    MUST(a, "INSERT OR REPLACE INTO n VALUES(3, 'bob', 'X', " V_X ")");
    EXPECT_ONE(a, "SELECT zvec_sync('ni')",
               "{\"upserted\":1,\"deleted\":1,\"skipped\":0}");
    EXPECT_ONE(a, "SELECT zvec_check('ni')",
               "{\"status\":\"ok\",\"base\":1,\"indexed\":1,\"stale\":0,"
               "\"missing\":0,\"pending\":0}");
#undef PKEYS
    sqlite3_close(a);
  }

  sqlite3_close(db);
  nftw(dir, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
  printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
