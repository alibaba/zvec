#include <sqlite3ext.h>
#include <string.h>
#include "registry.h"
#include "zvec/c_api.h"

SQLITE_EXTENSION_INIT3

typedef struct zvec_vtab {
  sqlite3_vtab base;
  zvec_shared *shared;
} zvec_vtab;

typedef struct zvec_cursor {
  sqlite3_vtab_cursor base;
  sqlite3_int64 pos;
} zvec_cursor;

static char *dequote_copy(const char *arg) {
  size_t len = strlen(arg);
  if (len >= 2 && (arg[0] == '\'' || arg[0] == '"') && arg[len - 1] == arg[0]) {
    return sqlite3_mprintf("%.*s", (int)(len - 2), arg + 1);
  }
  return sqlite3_mprintf("%s", arg);
}

static int zvec_connect(sqlite3 *db, void *p_aux, int argc, const char *const *argv, sqlite3_vtab **pp_vtab, char **pz_err) {
  (void)p_aux;
  if (argc < 4) {
    *pz_err = sqlite3_mprintf("zvec: usage: CREATE VIRTUAL TABLE t USING zvec('<collection_path>')");
    return SQLITE_ERROR;
  }

  char *path = dequote_copy(argv[3]);
  if (path == NULL) return SQLITE_NOMEM;

  char err[256] = "";
  zvec_shared *shared = NULL;
  zvec_error_code_t zrc = zvec_shared_open(path, &shared, err, sizeof err);
  if (zrc != ZVEC_OK) {
    *pz_err = sqlite3_mprintf("zvec: cannot open collection '%s': %s", path, err);
    sqlite3_free(path);
    return SQLITE_ERROR;
  } 

  int rc = sqlite3_declare_vtab(
      db,
      "CREATE TABLE x(score REAL, distance REAL, query BLOB HIDDEN, k INTEGER HIDDEN)");
  if (rc != SQLITE_OK) {
    zvec_shared_release(shared);
    return rc;
  }

  zvec_vtab *vtab = sqlite3_malloc(sizeof *vtab);
  if (vtab == NULL) {
    zvec_shared_release(shared);
    return SQLITE_NOMEM;
  } 
  memset(vtab, 0, sizeof *vtab);
  vtab->shared = shared;
  *pp_vtab = &vtab->base;
  return SQLITE_OK;
}

static int zvec_disconnect(sqlite3_vtab *p_vtab) {
  zvec_vtab *vtab = (zvec_vtab *)p_vtab;
  zvec_shared_release(vtab->shared);
  sqlite3_free(vtab);
  return SQLITE_OK;
}

static int zvec_best_index(sqlite3_vtab *p_vtab, sqlite3_index_info *p_info) {
  (void)p_vtab;
  p_info->estimatedCost = 1000000.0;
  return SQLITE_OK;
}

static int zvec_open_cursor(sqlite3_vtab *p_vtab, sqlite3_vtab_cursor **pp_cursor) {
  (void)p_vtab;
  zvec_cursor *cur = sqlite3_malloc(sizeof *cur);
  if (cur == NULL) return SQLITE_NOMEM;
  memset(cur, 0, sizeof *cur);
  *pp_cursor = &cur->base;
  return SQLITE_OK;
}

static int zvec_close_cursor(sqlite3_vtab_cursor *p_cursor) {
  sqlite3_free(p_cursor);
  return SQLITE_OK;
}

static int zvec_filter(sqlite3_vtab_cursor *p_cursor, int idx_num, const char *idx_str, int argc, sqlite3_value **argv) {
  (void)idx_num; (void)idx_str; (void)argc; (void)argv;
  ((zvec_cursor *)p_cursor)->pos = 0;
  return SQLITE_OK;
}

static int zvec_next(sqlite3_vtab_cursor *p_cursor) {
  ((zvec_cursor *)p_cursor)->pos++;
  return SQLITE_OK;
}

static int zvec_eof(sqlite3_vtab_cursor *p_cursor) {
  (void)p_cursor;
  return 1;
}

static int zvec_column(sqlite3_vtab_cursor *p_cursor, sqlite3_context *ctx, int col) {
  (void)p_cursor; (void)col;
  sqlite3_result_null(ctx);
  return SQLITE_OK;
}

static int zvec_rowid(sqlite3_vtab_cursor *p_cursor, sqlite3_int64 *p_rowid) {
  *p_rowid = ((zvec_cursor *)p_cursor)->pos;
  return SQLITE_OK;
}

static const sqlite3_module zvec_module = {
    .iVersion = 0,
    .xCreate = zvec_connect,
    .xConnect = zvec_connect,
    .xBestIndex = zvec_best_index,
    .xDisconnect = zvec_disconnect,
    .xDestroy = zvec_disconnect, /* DROP must never delete the user's data */
    .xOpen = zvec_open_cursor,
    .xClose = zvec_close_cursor,
    .xFilter = zvec_filter,
    .xNext = zvec_next,
    .xEof = zvec_eof,
    .xColumn = zvec_column,
    .xRowid = zvec_rowid,
};

int zvec_vtab_register(sqlite3 *db) {
  return sqlite3_create_module_v2(db, "zvec", &zvec_module, NULL, NULL);
}

