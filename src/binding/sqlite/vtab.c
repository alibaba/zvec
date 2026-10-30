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

#define _XOPEN_SOURCE 700  // lstat, mkdir, readdir

#include <sys/stat.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <sqlite3ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif
#include "zvec/c_api.h"
#include "args.h"
#include "registry.h"

SQLITE_EXTENSION_INIT3

typedef struct zvec_vtab {
  sqlite3_vtab base;
  sqlite3 *db;          // needed by xDestroy to drop the shadow tables
  char *schema;         // "main", "temp", ...
  char *name;           // the virtual table's own name
  int derived;          // 1: created with base=..., owns shadow tables
  int sync_query;       // 1: drain the outbox before every search
  zvec_shared *shared;  // read-write handle shared by this process, or ...
  zvec_collection_t
      *ro_col;    // ... a private read-only one (database opened read-only)
  char *path;     // derived tables: where the index lives; opened on first use
  char uuid[33];  // derived tables: identity the index directory must carry
  int read_only;  // derived tables: the database file is read-only
  int n_scalar;
  char **scalar_names;
  zvec_data_type_t *scalar_types;
  int n_vec;
  char **vec_names;
  uint32_t *vec_dims;
  zvec_metric_type_t *vec_metrics;
  zvec_data_type_t *vec_types;
} zvec_vtab;

typedef struct zvec_cursor {
  sqlite3_vtab_cursor base;  // must be first
  zvec_doc_t **docs;         // query results, owned by the cursor
  size_t n_docs;
  size_t pos;
  int vec;
} zvec_cursor;

static zvec_collection_t *vtab_col(const zvec_vtab *v) {
  return v->ro_col != NULL ? v->ro_col : zvec_shared_collection(v->shared);
}

static void clear_results(zvec_cursor *cur) {
  if (cur->docs != NULL) zvec_docs_free(cur->docs, cur->n_docs);
  cur->docs = NULL;
  cur->n_docs = 0;
  cur->pos = 0;
}

static int fail_msg(sqlite3_vtab *p_vtab, char *msg) {
  sqlite3_free(p_vtab->zErrMsg);
  p_vtab->zErrMsg = msg;
  return SQLITE_ERROR;
}

static int fail_from_zvec(sqlite3_vtab *p_vtab, const char *what) {
  zvec_error_details_t d = {0};
  const char *msg = (zvec_get_last_error_details(&d) == ZVEC_OK && d.message)
                        ? d.message
                        : "unknown error";
  return fail_msg(p_vtab, sqlite3_mprintf("zvec: %s: %s", what, msg));
}

// Bytes per element for the vector types the virtual table can query.
static size_t vec_elem_size(zvec_data_type_t type) {
  switch (type) {
    case ZVEC_DATA_TYPE_VECTOR_FP16:
      return 2;
    case ZVEC_DATA_TYPE_VECTOR_FP32:
      return 4;
    case ZVEC_DATA_TYPE_VECTOR_FP64:
      return 8;
    case ZVEC_DATA_TYPE_VECTOR_INT8:
      return 1;
    default:
      return 0;
  }
}

static const char *vec_type_name(zvec_data_type_t type) {
  switch (type) {
    case ZVEC_DATA_TYPE_VECTOR_FP16:
      return "float16";
    case ZVEC_DATA_TYPE_VECTOR_FP64:
      return "float64";
    case ZVEC_DATA_TYPE_VECTOR_INT8:
      return "int8";
    default:
      return "float32";
  }
}

static int col_scalar(int i) {
  return 1 + i;
}
static int col_vec(const zvec_vtab *v, int j) {
  return 1 + v->n_scalar + j;
}
static int col_score(const zvec_vtab *v) {
  return 1 + v->n_scalar + v->n_vec;
}
static int col_distance(const zvec_vtab *v) {
  return col_score(v) + 1;
}
static int col_k(const zvec_vtab *v) {
  return col_score(v) + 2;
}

static void result_scalar(sqlite3_context *ctx, const zvec_doc_t *doc,
                          const char *name, zvec_data_type_t type) {
  if (!zvec_doc_has_field_value(doc, name) ||
      zvec_doc_is_field_null(doc, name)) {
    sqlite3_result_null(ctx);
    return;
  }
  switch (type) {
    case ZVEC_DATA_TYPE_BOOL: {
      bool v = false;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_int(ctx, v ? 1 : 0);
      return;
    }
    case ZVEC_DATA_TYPE_INT32: {
      int32_t v = 0;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_int(ctx, v);
      return;
    }
    case ZVEC_DATA_TYPE_INT64: {
      int64_t v = 0;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_int64(ctx, v);
      return;
    }
    case ZVEC_DATA_TYPE_UINT32: {
      uint32_t v = 0;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_int64(ctx, (sqlite3_int64)v);
      return;
    }
    case ZVEC_DATA_TYPE_UINT64: {  // values above 2^63-1 wrap negative
      uint64_t v = 0;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_int64(ctx, (sqlite3_int64)v);
      return;
    }
    case ZVEC_DATA_TYPE_FLOAT: {
      float v = 0;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_double(ctx, (double)v);
      return;
    }
    case ZVEC_DATA_TYPE_DOUBLE: {
      double v = 0;
      zvec_doc_get_field_value_basic(doc, name, type, &v, sizeof v);
      sqlite3_result_double(ctx, v);
      return;
    }
    case ZVEC_DATA_TYPE_STRING:
    case ZVEC_DATA_TYPE_BINARY: {
      const void *p = NULL;
      size_t size = 0;
      if (zvec_doc_get_field_value_pointer(doc, name, type, &p, &size) !=
              ZVEC_OK ||
          p == NULL) {
        sqlite3_result_null(ctx);
      } else if (type == ZVEC_DATA_TYPE_STRING) {
        sqlite3_result_text(ctx, (const char *)p, -1,
                            SQLITE_TRANSIENT);  // NUL-terminated
      } else {
        sqlite3_result_blob(ctx, p, (int)size, SQLITE_TRANSIENT);
      }
      return;
    }
    default:
      sqlite3_result_null(ctx);
  }
}

static int zvec_column(sqlite3_vtab_cursor *p_cursor, sqlite3_context *ctx,
                       int col) {
  const zvec_cursor *cur = (const zvec_cursor *)p_cursor;
  const zvec_vtab *vtab = (const zvec_vtab *)p_cursor->pVtab;
  const zvec_doc_t *doc = cur->docs[cur->pos];

  if (col == 0) {
    const char *pk = zvec_doc_get_pk_pointer(doc);
    if (pk != NULL)
      sqlite3_result_text(ctx, pk, -1, SQLITE_TRANSIENT);
    else
      sqlite3_result_null(ctx);
  } else if (col == col_score(vtab)) {
    sqlite3_result_double(ctx, (double)zvec_doc_get_score(doc));
  } else if (col == col_distance(vtab)) {
    double d = (double)zvec_doc_get_score(doc);
    if (vtab->vec_metrics[cur->vec] == ZVEC_METRIC_TYPE_IP)
      d = -d;  // larger IP = closer
    sqlite3_result_double(ctx, d);
  } else {
    int i = col - col_scalar(0);
    if (i >= 0 && i < vtab->n_scalar) {
      result_scalar(ctx, doc, vtab->scalar_names[i], vtab->scalar_types[i]);
    } else {
      sqlite3_result_null(ctx);  // hidden vector / k columns
    }
  }
  return SQLITE_OK;
}

static const char *sql_type_for(zvec_data_type_t type) {
  switch (type) {
    case ZVEC_DATA_TYPE_BOOL:
    case ZVEC_DATA_TYPE_INT32:
    case ZVEC_DATA_TYPE_INT64:
    case ZVEC_DATA_TYPE_UINT32:
    case ZVEC_DATA_TYPE_UINT64:
      return "INTEGER";
    case ZVEC_DATA_TYPE_FLOAT:
    case ZVEC_DATA_TYPE_DOUBLE:
      return "REAL";
    case ZVEC_DATA_TYPE_STRING:
      return "TEXT";
    case ZVEC_DATA_TYPE_BINARY:
      return "BLOB";
    default:
      return NULL;
  }
}

// The CREATE TABLE text declared to SQLite for this table's columns.
static char *build_decl(const zvec_vtab *vtab) {
  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendall(s, "CREATE TABLE x(pk TEXT");
  for (int i = 0; i < vtab->n_scalar; i++) {
    sqlite3_str_appendf(s, ", \"%w\" %s", vtab->scalar_names[i],
                        sql_type_for(vtab->scalar_types[i]));
  }
  for (int j = 0; j < vtab->n_vec; j++) {
    sqlite3_str_appendf(s, ", \"%w\" BLOB HIDDEN", vtab->vec_names[j]);
  }
  sqlite3_str_appendall(s, ", score REAL, distance REAL, k INTEGER HIDDEN)");
  return sqlite3_str_finish(s);
}

static char *introspect_and_declare(zvec_vtab *vtab, char **pz_err) {
  zvec_collection_schema_t *schema = NULL;
  zvec_field_schema_t **fields = NULL;
  size_t count = 0;
  char *decl = NULL;
  const char *problem = NULL;
  zvec_error_code_t rc;

  rc = zvec_collection_get_schema(vtab_col(vtab), &schema);
  if (rc != ZVEC_OK) {
    problem = "cannot read the collection schema";
    goto done;
  }

  rc = zvec_collection_schema_get_forward_fields(schema, &fields, &count);
  if (rc != ZVEC_OK) {
    problem = "cannot list the scalar fields";
    goto done;
  }
  if (count > 0) {
    vtab->scalar_names = sqlite3_malloc64(count * sizeof *vtab->scalar_names);
    vtab->scalar_types = sqlite3_malloc64(count * sizeof *vtab->scalar_types);
    if (vtab->scalar_names == NULL || vtab->scalar_types == NULL) {
      problem = "out of memory";
      goto done;
    }
  }
  for (size_t i = 0; i < count; i++) {
    zvec_data_type_t type = zvec_field_schema_get_data_type(fields[i]);
    if (sql_type_for(type) == NULL) continue;
    char *name = sqlite3_mprintf("%s", zvec_field_schema_get_name(fields[i]));
    if (name == NULL) {
      problem = "out of memory";
      goto done;
    }
    vtab->scalar_names[vtab->n_scalar] = name;
    vtab->scalar_types[vtab->n_scalar] = type;
    vtab->n_scalar++;
  }
  zvec_free(fields);
  fields = NULL;

  rc = zvec_collection_schema_get_vector_fields(schema, &fields, &count);
  if (rc != ZVEC_OK) {
    problem = "cannot list the vector fields";
    goto done;
  }
  if (count > 0) {
    vtab->vec_names = sqlite3_malloc64(count * sizeof *vtab->vec_names);
    vtab->vec_dims = sqlite3_malloc64(count * sizeof *vtab->vec_dims);
    vtab->vec_metrics = sqlite3_malloc64(count * sizeof *vtab->vec_metrics);
    vtab->vec_types = sqlite3_malloc64(count * sizeof *vtab->vec_types);
    if (vtab->vec_names == NULL || vtab->vec_dims == NULL ||
        vtab->vec_metrics == NULL || vtab->vec_types == NULL) {
      problem = "out of memory";
      goto done;
    }
  }
  for (size_t j = 0; j < count; j++) {
    zvec_data_type_t vtype = zvec_field_schema_get_data_type(fields[j]);
    if (vec_elem_size(vtype) == 0) continue;  // sparse, int4, binary, ...
    char *name = sqlite3_mprintf("%s", zvec_field_schema_get_name(fields[j]));
    if (name == NULL) {
      problem = "out of memory";
      goto done;
    }
    vtab->vec_names[vtab->n_vec] = name;
    vtab->vec_dims[vtab->n_vec] = zvec_field_schema_get_dimension(fields[j]);
    vtab->vec_types[vtab->n_vec] = vtype;
    const zvec_index_params_t *ip =
        zvec_field_schema_get_index_params(fields[j]);
    vtab->vec_metrics[vtab->n_vec] = ip != NULL
                                         ? zvec_index_params_get_metric_type(ip)
                                         : ZVEC_METRIC_TYPE_UNDEFINED;
    vtab->n_vec++;
  }
  zvec_free(fields);
  fields = NULL;
  if (vtab->n_vec == 0) {
    problem =
        "the collection has no vector field of a supported type "
        "(float16, float32, float64, int8)";
    goto done;
  }

  decl = build_decl(vtab);
  if (decl == NULL) problem = "out of memory";

done:
  if (fields != NULL) zvec_free(fields);
  if (schema != NULL) zvec_collection_schema_destroy(schema);
  if (decl == NULL && problem != NULL) {
    *pz_err = sqlite3_mprintf("zvec: %s", problem);
  }
  return decl;
}

static void free_vtab(zvec_vtab *vtab) {
  sqlite3_free(vtab->schema);
  sqlite3_free(vtab->name);
  sqlite3_free(vtab->path);
  sqlite3_free(vtab->vec_types);
  for (int i = 0; i < vtab->n_scalar; i++) sqlite3_free(vtab->scalar_names[i]);
  sqlite3_free(vtab->scalar_names);
  sqlite3_free(vtab->scalar_types);
  for (int j = 0; j < vtab->n_vec; j++) sqlite3_free(vtab->vec_names[j]);
  sqlite3_free(vtab->vec_names);
  sqlite3_free(vtab->vec_dims);
  sqlite3_free(vtab->vec_metrics);
  if (vtab->ro_col != NULL) {
    zvec_collection_close(vtab->ro_col);
  } else if (vtab->shared != NULL) {
    zvec_shared_release(vtab->shared);
  }
  sqlite3_free(vtab);
}

// ---- shadow tables ------------------------------------------------------
// A derived index keeps its settings and its change queue in three ordinary
// tables next to the virtual table: <t>_config, <t>_state, <t>_outbox. All
// helpers return SQLITE_OK or an SQLite error code and, on failure, set *err
// to an sqlite3_mprintf() string for the caller (zErrMsg) to own.

#define SHADOW_VERSION "1"

// sqlite3_str_finish() returns NULL for an empty string; this returns "".
static char *str_done(sqlite3_str *s) {
  int failed = sqlite3_str_errcode(s) != 0;
  char *r = sqlite3_str_finish(s);
  if (r == NULL && !failed) r = sqlite3_mprintf("%s", "");
  return r;
}

static int shadow_fail(char **err, int rc, const char *fmt, const char *a,
                       const char *b) {
  if (err != NULL) {
    *err = sqlite3_mprintf(fmt, a != NULL ? a : "", b != NULL ? b : "");
  }
  return rc;
}

static int shadow_exec(sqlite3 *db, char *sql, char **err) {
  if (sql == NULL)
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  char *msg = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &msg);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    rc = shadow_fail(err, rc, "zvec: %s%s", msg, NULL);
  }
  sqlite3_free(msg);
  return rc;
}

// Table names become a directory name: allow only [A-Za-z0-9_].
static int shadow_safe_component(const char *s) {
  if (s == NULL || s[0] == '\0') return 0;
  for (; *s; s++) {
    char c = *s;
    int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_';
    if (!ok) return 0;
  }
  return 1;
}

static int zvec_default_path(sqlite3 *db, const char *schema, const char *table,
                             char **out_path, char **err) {
  const char *file = sqlite3_db_filename(db, schema);
  if (file == NULL || file[0] == '\0') {
    return shadow_fail(err, SQLITE_NOTFOUND,
                       "zvec: this database has no file; give path='...'%s%s",
                       NULL, NULL);
  }
  if (!shadow_safe_component(table)) {
    return shadow_fail(
        err, SQLITE_NOTFOUND,
        "zvec: table name '%s' cannot name a directory; give path='...'%s",
        table, NULL);
  }
  *out_path = sqlite3_mprintf("%s.zvec/%s", file, table);
  if (*out_path == NULL)
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  return SQLITE_OK;
}

// SQLite's affinity rules, reduced to what the index can store.
static zvec_ftype filter_type_from_decl(const char *decl) {
  if (decl == NULL) return ZVEC_FT_UNKNOWN;
  char up[64];
  size_t n = 0;
  for (; decl[n] != '\0' && n + 1 < sizeof up; n++)
    up[n] = (char)toupper((unsigned char)decl[n]);
  up[n] = '\0';
  if (strstr(up, "INT") != NULL) return ZVEC_FT_INT;
  if (strstr(up, "CHAR") != NULL || strstr(up, "CLOB") != NULL ||
      strstr(up, "TEXT") != NULL)
    return ZVEC_FT_TEXT;
  if (strstr(up, "BLOB") != NULL || up[0] == '\0') return ZVEC_FT_UNKNOWN;
  if (strstr(up, "REAL") != NULL || strstr(up, "FLOA") != NULL ||
      strstr(up, "DOUB") != NULL)
    return ZVEC_FT_REAL;
  return ZVEC_FT_UNKNOWN;  // NUMERIC and friends: ask the user
}

static int zvec_shadow_check_base(sqlite3 *db, const char *schema, zvec_args *a,
                                  char **err) {
  sqlite3_stmt *st = NULL;
  int rc;

  // Must be an ordinary table (not a view, not another virtual table).
  char *sql = sqlite3_mprintf(
      "SELECT 1 FROM \"%w\".sqlite_master WHERE type='table' AND name=%Q "
      "AND sql NOT LIKE 'CREATE VIRTUAL%%'",
      schema, a->base);
  if (sql == NULL)
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
  rc = sqlite3_step(st);
  sqlite3_finalize(st);
  if (rc != SQLITE_ROW) {
    return shadow_fail(
        err, SQLITE_ERROR,
        "zvec: base table '%s' does not exist or is not an ordinary table%s",
        a->base, NULL);
  }

  rc = sqlite3_prepare_v2(
      db, "SELECT name, pk, type FROM pragma_table_xinfo(?1, ?2)", -1, &st,
      NULL);
  if (rc != SQLITE_OK)
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
  sqlite3_bind_text(st, 1, a->base, -1, SQLITE_STATIC);
  sqlite3_bind_text(st, 2, schema, -1, SQLITE_STATIC);
  int have_key = 0, key_is_pk = 0, have_col = 0, pk_cols = 0;
  int found[ZVEC_MAX_FILTER] = {0};
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    const char *name = (const char *)sqlite3_column_text(st, 0);
    int pk = sqlite3_column_int(st, 1);
    if (pk > 0) pk_cols++;
    if (name != NULL && sqlite3_stricmp(name, a->key) == 0) {
      have_key = 1;
      key_is_pk = pk > 0;
    }
    if (name != NULL && sqlite3_stricmp(name, a->column) == 0) have_col = 1;
    for (int f = 0; name != NULL && f < a->n_filter; f++) {
      if (sqlite3_stricmp(name, a->filter_name[f]) != 0) continue;
      found[f] = 1;
      if (a->filter_type[f] == ZVEC_FT_UNKNOWN) {
        a->filter_type[f] =
            filter_type_from_decl((const char *)sqlite3_column_text(st, 2));
      }
    }
  }
  sqlite3_finalize(st);
  if (rc != SQLITE_DONE)
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);

  if (!have_key)
    return shadow_fail(err, SQLITE_ERROR,
                       "zvec: key column '%s' not found in '%s'", a->key,
                       a->base);
  if (!have_col)
    return shadow_fail(err, SQLITE_ERROR,
                       "zvec: vector column '%s' not found in '%s'", a->column,
                       a->base);
  for (int f = 0; f < a->n_filter; f++) {
    if (!found[f]) {
      return shadow_fail(err, SQLITE_ERROR,
                         "zvec: filter column '%s' not found in '%s'",
                         a->filter_name[f], a->base);
    }
    if (a->filter_type[f] == ZVEC_FT_UNKNOWN) {
      return shadow_fail(
          err, SQLITE_ERROR,
          "zvec: cannot tell if filter column '%s' is int, real or text; "
          "write it as name:int, name:real or name:text%s",
          a->filter_name[f], NULL);
    }
  }
  if (!key_is_pk || pk_cols != 1) {
    return shadow_fail(
        err, SQLITE_ERROR,
        "zvec: key '%s' must be the single-column PRIMARY KEY of '%s'", a->key,
        a->base);
  }
  if (sqlite3_stricmp(a->key, a->column) == 0) {
    return shadow_fail(err, SQLITE_ERROR,
                       "zvec: key and column must differ%s%s", NULL, NULL);
  }
  return SQLITE_OK;
}

static const char *shadow_metric_name(int m) {
  return m == 2 ? "ip" : m == 3 ? "cosine" : "l2";
}
static const char *shadow_elem_name(zvec_elem e) {
  switch (e) {
    case ZVEC_ELEM_F16:
      return "float16";
    case ZVEC_ELEM_F64:
      return "float64";
    case ZVEC_ELEM_I8:
      return "int8";
    default:
      return "float32";
  }
}

// Triggers on the base table queue the keys that changed, in the same
// transaction as the change itself. The outbox stores no vectors: sync reads
// the base table's current state for each queued key, so replays are harmless.
// Trigger bodies may not qualify table names, hence the bare "<t>_outbox".
// ---- UNIQUE conflicts --------------------------------------------------
// INSERT OR REPLACE / UPDATE OR REPLACE delete the rows that conflict on a
// UNIQUE constraint, and SQLite does not run DELETE triggers for those deletes
// unless recursive_triggers is on. The index would keep a document for a row
// that no longer exists. So, for every plain UNIQUE index of the base table,
// BEFORE triggers queue the key of whatever row currently holds the same
// value. Sync always applies the *current* state of a key, so queueing a row
// that survives (INSERT OR IGNORE, or the row itself) is harmless.
// Partial and expression indexes cannot be probed this way: they are counted
// in *n_unsupported and left to recursive_triggers and zvec_check.

typedef struct unique_index {
  int n;
  char *cols[8];
  char *colls[8];
} unique_index;

static void unique_index_free(unique_index *u, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < u[i].n; j++) {
      sqlite3_free(u[i].cols[j]);
      sqlite3_free(u[i].colls[j]);
    }
  }
  sqlite3_free(u);
}

enum { MAX_UNIQUE = 16 };

static int list_unique_indexes(sqlite3 *db, const char *schema,
                               const zvec_args *a, unique_index **out,
                               int *n_out, int *n_unsupported, char **err) {
  *out = NULL;
  *n_out = 0;
  *n_unsupported = 0;
  char *names[MAX_UNIQUE];
  int partial[MAX_UNIQUE];
  int n = 0, rc;
  sqlite3_stmt *st = NULL;
  rc = sqlite3_prepare_v2(db,
                          "SELECT name, partial FROM pragma_index_list(?1, ?2) "
                          "WHERE \"unique\" = 1 AND origin <> 'pk'",
                          -1, &st, NULL);
  if (rc != SQLITE_OK)
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
  sqlite3_bind_text(st, 1, a->base, -1, SQLITE_STATIC);
  sqlite3_bind_text(st, 2, schema, -1, SQLITE_STATIC);
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    if (n >= MAX_UNIQUE) {
      (*n_unsupported)++;
      continue;
    }
    names[n] = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(st, 0));
    partial[n] = sqlite3_column_int(st, 1);
    if (names[n] == NULL) {
      rc = SQLITE_NOMEM;
      break;
    }
    n++;
  }
  sqlite3_finalize(st);
  unique_index *u = NULL;
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
    u = sqlite3_malloc64((sqlite3_uint64)(n > 0 ? n : 1) * sizeof *u);
    if (u == NULL) rc = SQLITE_NOMEM;
    if (u != NULL) memset(u, 0, (size_t)(n > 0 ? n : 1) * sizeof *u);
  }
  int used = 0;
  for (int i = 0; i < n && rc == SQLITE_OK; i++) {
    sqlite3_stmt *xi = NULL;
    rc = sqlite3_prepare_v2(
        db,
        "SELECT cid, name, coll FROM pragma_index_xinfo(?1, ?2) "
        "WHERE key = 1 ORDER BY seqno",
        -1, &xi, NULL);
    if (rc != SQLITE_OK) break;
    sqlite3_bind_text(xi, 1, names[i], -1, SQLITE_STATIC);
    sqlite3_bind_text(xi, 2, schema, -1, SQLITE_STATIC);
    unique_index cur;
    memset(&cur, 0, sizeof cur);
    int supported = !partial[i];
    while ((rc = sqlite3_step(xi)) == SQLITE_ROW) {
      const char *cn = (const char *)sqlite3_column_text(xi, 1);
      const char *coll = (const char *)sqlite3_column_text(xi, 2);
      if (sqlite3_column_int(xi, 0) < 0 || cn == NULL || cur.n >= 8) {
        supported = 0;  // expression or rowid column, or too wide
        continue;
      }
      cur.cols[cur.n] = sqlite3_mprintf("%s", cn);
      cur.colls[cur.n] = sqlite3_mprintf("%s", coll != NULL ? coll : "BINARY");
      if (cur.cols[cur.n] == NULL || cur.colls[cur.n] == NULL) {
        sqlite3_free(cur.cols[cur.n]);
        sqlite3_free(cur.colls[cur.n]);
        rc = SQLITE_NOMEM;
        break;
      }
      cur.n++;
    }
    sqlite3_finalize(xi);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
    if (rc != SQLITE_OK) {
      for (int j = 0; j < cur.n; j++) {
        sqlite3_free(cur.cols[j]);
        sqlite3_free(cur.colls[j]);
      }
      break;
    }
    if (supported && cur.n > 0) {
      u[used++] = cur;
    } else {
      for (int j = 0; j < cur.n; j++) {
        sqlite3_free(cur.cols[j]);
        sqlite3_free(cur.colls[j]);
      }
      (*n_unsupported)++;
    }
  }
  for (int i = 0; i < n; i++) sqlite3_free(names[i]);
  if (rc != SQLITE_OK) {
    unique_index_free(u, used);
    return shadow_fail(
        err, rc, "zvec: %s%s",
        rc == SQLITE_NOMEM ? "out of memory" : sqlite3_errmsg(db), NULL);
  }
  *out = u;
  *n_out = used;
  return SQLITE_OK;
}

// The statements that queue the rows holding the same value as `new`.
static char *unique_probe_sql(const char *table, const zvec_args *a,
                              const unique_index *u, int n, int with_rowid) {
  sqlite3_str *q = sqlite3_str_new(NULL);
  for (int i = 0; i < n; i++) {
    sqlite3_str_appendf(q,
                        "INSERT INTO \"%w_outbox\"(op, pk) SELECT 2, "
                        "CAST(\"%w\" AS TEXT) FROM \"%w\" WHERE ",
                        table, a->key, a->base);
    for (int j = 0; j < u[i].n; j++) {
      sqlite3_str_appendf(q, "%s\"%w\" = new.\"%w\" COLLATE \"%w\"",
                          j ? " AND " : "", u[i].cols[j], u[i].cols[j],
                          u[i].colls[j]);
    }
    sqlite3_str_appendall(q, "; ");
  }
  if (with_rowid) {
    sqlite3_str_appendf(q,
                        "INSERT INTO \"%w_outbox\"(op, pk) SELECT 2, "
                        "CAST(\"%w\" AS TEXT) FROM \"%w\" WHERE rowid = "
                        "new.rowid; ",
                        table, a->key, a->base);
  }
  return str_done(q);
}

static int is_rowid_table(sqlite3 *db, const char *schema, const char *base) {
  sqlite3_stmt *st = NULL;
  char *sql = sqlite3_mprintf(
      "SELECT sql FROM \"%w\".sqlite_master WHERE type='table' AND name=%Q",
      schema, base);
  if (sql == NULL) return 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) return 0;
  int rowid = 1;
  if (sqlite3_step(st) == SQLITE_ROW) {
    const char *text = (const char *)sqlite3_column_text(st, 0);
    // "WITHOUT ROWID" can only follow the closing parenthesis.
    const char *close = text != NULL ? strrchr(text, ')') : NULL;
    if (close != NULL) {
      for (const char *p = close + 1; *p != '\0'; p++) {
        if (sqlite3_strnicmp(p, "WITHOUT", 7) == 0) {
          rowid = 0;
          break;
        }
      }
    }
  }
  sqlite3_finalize(st);
  return rowid;
}

// True when the key is the table's INTEGER PRIMARY KEY (an alias of rowid), so
// a rowid conflict is a conflict on the key itself.
static int key_is_rowid_alias(sqlite3 *db, const char *schema,
                              const zvec_args *a) {
  sqlite3_stmt *st = NULL;
  int alias = 0;
  if (sqlite3_prepare_v2(db,
                         "SELECT type FROM pragma_table_xinfo(?1, ?2) WHERE "
                         "name = ?3 COLLATE nocase AND pk = 1",
                         -1, &st, NULL) != SQLITE_OK)
    return 0;
  sqlite3_bind_text(st, 1, a->base, -1, SQLITE_STATIC);
  sqlite3_bind_text(st, 2, schema, -1, SQLITE_STATIC);
  sqlite3_bind_text(st, 3, a->key, -1, SQLITE_STATIC);
  if (sqlite3_step(st) == SQLITE_ROW) {
    const char *t = (const char *)sqlite3_column_text(st, 0);
    alias = t != NULL && sqlite3_stricmp(t, "INTEGER") == 0;
  }
  sqlite3_finalize(st);
  return alias;
}

static char *shadow_trigger_sql(sqlite3 *db, const char *schema,
                                const char *table, const zvec_args *a,
                                char **err) {
  unique_index *uniq = NULL;
  int n_uniq = 0, n_unsupported = 0;
  if (list_unique_indexes(db, schema, a, &uniq, &n_uniq, &n_unsupported, err) !=
      SQLITE_OK)
    return NULL;
  int with_rowid =
      is_rowid_table(db, schema, a->base) && !key_is_rowid_alias(db, schema, a);
  char *probes = unique_probe_sql(table, a, uniq, n_uniq, with_rowid);
  if (probes == NULL) {
    unique_index_free(uniq, n_uniq);
    *err = sqlite3_mprintf("zvec: out of memory");
    return NULL;
  }
  sqlite3_str *q = sqlite3_str_new(NULL);
  if (probes[0] != '\0') {
    sqlite3_str_appendf(q,
                        "CREATE TRIGGER \"%w\".\"%w_zvec_bi\" BEFORE INSERT ON "
                        "\"%w\" BEGIN %s END;",
                        schema, table, a->base, probes);
  }
  if (n_uniq > 0) {
    sqlite3_str_appendf(
        q, "CREATE TRIGGER \"%w\".\"%w_zvec_bu\" BEFORE UPDATE OF ", schema,
        table);
    int first = 1;
    for (int i = 0; i < n_uniq; i++) {
      for (int j = 0; j < uniq[i].n; j++) {
        sqlite3_str_appendf(q, "%s\"%w\"", first ? "" : ", ", uniq[i].cols[j]);
        first = 0;
      }
    }
    // The rowid probe is only meaningful for INSERT.
    char *update_probes = unique_probe_sql(table, a, uniq, n_uniq, 0);
    sqlite3_str_appendf(q, " ON \"%w\" BEGIN %s END;", a->base,
                        update_probes != NULL ? update_probes : "");
    sqlite3_free(update_probes);
  }
  sqlite3_free(probes);
  unique_index_free(uniq, n_uniq);

  sqlite3_str_appendf(
      q,
      "CREATE TRIGGER \"%w\".\"%w_zvec_ai\" AFTER INSERT ON \"%w\" BEGIN "
      "INSERT INTO \"%w_outbox\"(op, pk) VALUES(1, CAST(new.\"%w\" AS TEXT)); "
      "END;",
      schema, table, a->base, table, a->key);
  // Fires for the vector, the key and every mirrored filter column.
  sqlite3_str_appendf(
      q, "CREATE TRIGGER \"%w\".\"%w_zvec_au\" AFTER UPDATE OF \"%w\", \"%w\"",
      schema, table, a->column, a->key);
  for (int i = 0; i < a->n_filter; i++)
    sqlite3_str_appendf(q, ", \"%w\"", a->filter_name[i]);
  sqlite3_str_appendf(q,
                      " ON \"%w\" WHEN new.\"%w\" IS NOT old.\"%w\" OR "
                      "new.\"%w\" IS NOT old.\"%w\"",
                      a->base, a->column, a->column, a->key, a->key);
  for (int i = 0; i < a->n_filter; i++) {
    sqlite3_str_appendf(q, " OR new.\"%w\" IS NOT old.\"%w\"",
                        a->filter_name[i], a->filter_name[i]);
  }
  sqlite3_str_appendall(q, " BEGIN ");
  sqlite3_str_appendf(
      q,
      "INSERT INTO \"%w_outbox\"(op, pk) SELECT 2, CAST(old.\"%w\" AS TEXT) "
      "WHERE new.\"%w\" IS NOT old.\"%w\"; ",
      table, a->key, a->key, a->key);
  sqlite3_str_appendf(q,
                      "INSERT INTO \"%w_outbox\"(op, pk) VALUES(1, "
                      "CAST(new.\"%w\" AS TEXT)); END;",
                      table, a->key);
  sqlite3_str_appendf(
      q,
      "CREATE TRIGGER \"%w\".\"%w_zvec_ad\" AFTER DELETE ON \"%w\" BEGIN "
      "INSERT INTO \"%w_outbox\"(op, pk) VALUES(2, CAST(old.\"%w\" AS TEXT)); "
      "END;",
      schema, table, a->base, table, a->key);
  return sqlite3_str_finish(q);
}

static int zvec_shadow_create(sqlite3 *db, const char *schema,
                              const char *table, const zvec_args *a,
                              const char *resolved_path, int path_is_default,
                              char **err) {
  int rc = shadow_exec(
      db,
      sqlite3_mprintf(
          "CREATE TABLE \"%w\".\"%w_config\"(key TEXT PRIMARY KEY, value TEXT "
          "NOT NULL) WITHOUT ROWID;"
          "CREATE TABLE \"%w\".\"%w_state\"(id INTEGER PRIMARY KEY "
          "CHECK(id=1), watermark INTEGER NOT NULL DEFAULT 0);"
          "INSERT INTO \"%w\".\"%w_state\"(id, watermark) VALUES(1, 0);"
          "CREATE TABLE \"%w\".\"%w_outbox\"(seq INTEGER PRIMARY KEY "
          "AUTOINCREMENT, "
          "op INTEGER NOT NULL CHECK(op IN (1,2)), pk TEXT NOT NULL);",
          schema, table, schema, table, schema, table, schema, table),
      err);
  if (rc != SQLITE_OK) return rc;

  char dim[16];
  snprintf(dim, sizeof dim, "%u", (unsigned)a->dim);
  sqlite3_str *fs = sqlite3_str_new(NULL);
  for (int i = 0; i < a->n_filter; i++) {
    sqlite3_str_appendf(fs, "%s%s:%s", i ? "," : "", a->filter_name[i],
                        zvec_ftype_name(a->filter_type[i]));
  }
  char *filter_list = str_done(fs);
  if (filter_list == NULL)
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  const char *rows[][2] = {{"version", SHADOW_VERSION},
                           {"base", a->base},
                           {"key", a->key},
                           {"column", a->column},
                           {"dim", dim},
                           {"metric", shadow_metric_name(a->metric)},
                           {"type", shadow_elem_name(a->elem)},
                           {"path", resolved_path},
                           {"path_default", path_is_default ? "1" : "0"},
                           {"sync", a->sync_query ? "query" : "manual"},
                           {"uuid", a->uuid},
                           {"filter", filter_list}};
  size_t n_rows = a->n_filter > 0 ? 12 : 11;

  char *sql = sqlite3_mprintf(
      "INSERT INTO \"%w\".\"%w_config\"(key, value) VALUES(?1, ?2)", schema,
      table);
  if (sql == NULL) {
    sqlite3_free(filter_list);
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  }
  sqlite3_stmt *st = NULL;
  rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    sqlite3_free(filter_list);
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
  }
  for (size_t i = 0; i < n_rows; i++) {
    sqlite3_reset(st);
    sqlite3_bind_text(st, 1, rows[i][0], -1, SQLITE_STATIC);
    sqlite3_bind_text(st, 2, rows[i][1], -1, SQLITE_STATIC);
    rc = sqlite3_step(st);
    if (rc != SQLITE_DONE) {
      rc = shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
      sqlite3_finalize(st);
      sqlite3_free(filter_list);
      return rc;
    }
  }
  sqlite3_finalize(st);
  sqlite3_free(filter_list);
  char *trigger_sql = shadow_trigger_sql(db, schema, table, a, err);
  if (trigger_sql == NULL) return *err != NULL ? SQLITE_ERROR : SQLITE_NOMEM;
  rc = shadow_exec(db, trigger_sql, err);
  if (rc != SQLITE_OK) return rc;
  // Rows that already exist are queued too, so the first sync builds the index.
  return shadow_exec(
      db,
      sqlite3_mprintf("INSERT INTO \"%w\".\"%w_outbox\"(op, pk) SELECT 1, "
                      "CAST(\"%w\" AS TEXT) FROM \"%w\".\"%w\"",
                      schema, table, a->key, schema, a->base),
      err);
}

// Wraps v in single quotes ('' escapes), the form zvec_args_parse expects.
static char *shadow_quote_value(const char *k, const char *v) {
  return sqlite3_mprintf("%s=%Q", k, v);
}

static int zvec_shadow_load(sqlite3 *db, const char *schema, const char *table,
                            zvec_args *out, int *path_is_default, char **err) {
  char *sql = sqlite3_mprintf("SELECT key, value FROM \"%w\".\"%w_config\"",
                              schema, table);
  if (sql == NULL)
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  sqlite3_stmt *st = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    return shadow_fail(err, rc, "zvec: cannot read settings of '%s': %s", table,
                       sqlite3_errmsg(db));

  enum { kMax = 16 };
  char *args[kMax];
  int n = 0, version_ok = 0, def = 0;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    const char *k = (const char *)sqlite3_column_text(st, 0);
    const char *v = (const char *)sqlite3_column_text(st, 1);
    if (k == NULL || v == NULL) continue;
    if (strcmp(k, "version") == 0) {
      version_ok = strcmp(v, SHADOW_VERSION) == 0;
      continue;
    }
    if (strcmp(k, "path_default") == 0) {
      def = v[0] == '1';
      continue;
    }
    if (n < kMax) args[n++] = shadow_quote_value(k, v);
  }
  sqlite3_finalize(st);

  const char *argv[3 + kMax];
  argv[0] = "zvec";
  argv[1] = schema;
  argv[2] = table;
  int bad = rc != SQLITE_DONE || !version_ok;
  for (int i = 0; i < n; i++) {
    if (args[i] == NULL) bad = 1;
    argv[3 + i] = args[i] != NULL ? args[i] : "";
  }
  char msg[256] = "";
  int prc = bad ? -1 : zvec_args_parse(3 + n, argv, out, msg, sizeof msg);
  if (prc == 0 && out->uuid[0] == '\0') {
    zvec_args_free(out);
    prc = -1;
    bad = 1;
  }
  for (int i = 0; i < n; i++) sqlite3_free(args[i]);
  if (prc != 0) {
    return shadow_fail(
        err, SQLITE_ERROR, "zvec: settings of '%s' are unusable: %s", table,
        bad ? "missing, incomplete or from a newer version" : msg);
  }
  if (path_is_default != NULL) *path_is_default = def;
  return SQLITE_OK;
}

static int zvec_shadow_drop(sqlite3 *db, const char *schema, const char *table,
                            char **err) {
  return shadow_exec(
      db,
      sqlite3_mprintf("DROP TRIGGER IF EXISTS \"%w\".\"%w_zvec_ai\";"
                      "DROP TRIGGER IF EXISTS \"%w\".\"%w_zvec_au\";"
                      "DROP TRIGGER IF EXISTS \"%w\".\"%w_zvec_ad\";"
                      "DROP TRIGGER IF EXISTS \"%w\".\"%w_zvec_bi\";"
                      "DROP TRIGGER IF EXISTS \"%w\".\"%w_zvec_bu\";"
                      "DROP TABLE IF EXISTS \"%w\".\"%w_outbox\";"
                      "DROP TABLE IF EXISTS \"%w\".\"%w_state\";"
                      "DROP TABLE IF EXISTS \"%w\".\"%w_config\";",
                      schema, table, schema, table, schema, table, schema,
                      table, schema, table, schema, table, schema, table,
                      schema, table),
      err);
}

// ---- creating, connecting and destroying -------------------------------

// Deletes a directory tree (symlinks are removed, never followed). Returns 0
// on success. Callers only pass directories this extension created itself.
#ifdef _WIN32
static int remove_tree(const char *path) {
  char *pattern = sqlite3_mprintf("%s\\*", path);
  WIN32_FIND_DATAA fd;
  HANDLE h =
      pattern != NULL ? FindFirstFileA(pattern, &fd) : INVALID_HANDLE_VALUE;
  sqlite3_free(pattern);
  if (h == INVALID_HANDLE_VALUE) return -1;
  int rc = 0;
  do {
    if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0)
      continue;
    char *child = sqlite3_mprintf("%s\\%s", path, fd.cFileName);
    if (child == NULL) {
      rc = -1;
      break;
    }
    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      if (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
        if (!RemoveDirectoryA(child)) rc = -1;
      } else if (remove_tree(child) != 0) {
        rc = -1;
      }
    } else {
      SetFileAttributesA(child, FILE_ATTRIBUTE_NORMAL);  // clears read-only
      if (!DeleteFileA(child)) rc = -1;
    }
    sqlite3_free(child);
  } while (FindNextFileA(h, &fd));
  FindClose(h);
  return (rc == 0 && RemoveDirectoryA(path)) ? 0 : -1;
}
#else
static int remove_tree_at(const char *path, int depth) {
  DIR *d = depth < 32 ? opendir(path) : NULL;
  if (d == NULL) return -1;
  int rc = 0;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
    char *child = sqlite3_mprintf("%s/%s", path, e->d_name);
    struct stat sb;
    if (child == NULL) {
      rc = -1;
      break;
    }
    if (lstat(child, &sb) != 0) {
      rc = -1;
    } else if (S_ISDIR(sb.st_mode)) {
      if (remove_tree_at(child, depth + 1) != 0) rc = -1;
    } else if (unlink(child) != 0) {
      rc = -1;
    }
    sqlite3_free(child);
  }
  closedir(d);
  return (rc == 0 && rmdir(path) == 0) ? 0 : -1;
}

static int remove_tree(const char *path) {
  return remove_tree_at(path, 0);
}
#endif

static int path_exists(const char *path) {
#ifdef _WIN32
  struct _stat sb;
  return _stat(path, &sb) == 0;
#else
  struct stat sb;
  return stat(path, &sb) == 0;
#endif
}

// Best effort: removes the directory above `path` if it is now empty.
static void remove_empty_parent(const char *path) {
  const char *slash = strrchr(path, '/');
#ifdef _WIN32
  const char *back = strrchr(path, '\\');
  if (back != NULL && (slash == NULL || back > slash)) slash = back;
#endif
  if (slash == NULL) return;
  char *dir = sqlite3_mprintf("%.*s", (int)(slash - path), path);
  if (dir == NULL) return;
#ifdef _WIN32
  _rmdir(dir);
#else
  rmdir(dir);
#endif
  sqlite3_free(dir);
}

// Creates the directory above <dbfile>.zvec/<name> (the default location).
static int make_default_parent(const char *path) {
  const char *slash = strrchr(path, '/');
#ifdef _WIN32
  const char *back = strrchr(path, '\\');
  if (back != NULL && (slash == NULL || back > slash)) slash = back;
#endif
  if (slash == NULL) return 0;
  char *dir = sqlite3_mprintf("%.*s", (int)(slash - path), path);
  if (dir == NULL) return -1;
#ifdef _WIN32
  int rc = _mkdir(dir);
#else
  int rc = mkdir(dir, 0700);
#endif
  int exists = rc != 0 && errno == EEXIST;
  sqlite3_free(dir);
  return (rc == 0 || exists) ? 0 : -1;
}

static zvec_data_type_t data_type_for(zvec_elem e) {
  switch (e) {
    case ZVEC_ELEM_F16:
      return ZVEC_DATA_TYPE_VECTOR_FP16;
    case ZVEC_ELEM_F64:
      return ZVEC_DATA_TYPE_VECTOR_FP64;
    case ZVEC_ELEM_I8:
      return ZVEC_DATA_TYPE_VECTOR_INT8;
    default:
      return ZVEC_DATA_TYPE_VECTOR_FP32;
  }
}

// Builds a one-field collection (the vector column, HNSW) and creates it
// through the registry.
static zvec_data_type_t filter_data_type(zvec_ftype t) {
  switch (t) {
    case ZVEC_FT_INT:
      return ZVEC_DATA_TYPE_INT64;
    case ZVEC_FT_REAL:
      return ZVEC_DATA_TYPE_DOUBLE;
    default:
      return ZVEC_DATA_TYPE_STRING;
  }
}

// Nullable scalar field with an inverted index, one per filter column.
static int add_filter_fields(zvec_collection_schema_t *schema,
                             const zvec_args *a) {
  for (int i = 0; i < a->n_filter; i++) {
    zvec_field_schema_t *f = zvec_field_schema_create(
        a->filter_name[i], filter_data_type(a->filter_type[i]), true, 0);
    zvec_index_params_t *ip = zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
    int ok = f != NULL && ip != NULL &&
             zvec_index_params_set_invert_params(ip, true, false) == ZVEC_OK &&
             zvec_field_schema_set_index_params(f, ip) == ZVEC_OK &&
             zvec_collection_schema_add_field(schema, f) == ZVEC_OK;
    if (ip != NULL) zvec_index_params_destroy(ip);
    if (f != NULL) zvec_field_schema_destroy(f);
    if (!ok) return -1;
  }
  return 0;
}

static zvec_error_code_t create_collection(const zvec_args *a, const char *name,
                                           const char *path, zvec_shared **out,
                                           char *err, size_t err_len) {
  zvec_collection_schema_t *schema = zvec_collection_schema_create(name);
  zvec_field_schema_t *field = zvec_field_schema_create(
      a->column, data_type_for(a->elem), false, a->dim);
  zvec_index_params_t *ip = zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  zvec_error_code_t rc = ZVEC_OK;

  if (schema == NULL || field == NULL || ip == NULL) {
    snprintf(err, err_len, "cannot build the collection schema");
    rc = ZVEC_ERROR_INTERNAL_ERROR;
  } else if (zvec_index_params_set_metric_type(
                 ip, (zvec_metric_type_t)a->metric) != ZVEC_OK ||
             zvec_index_params_set_hnsw_params(ip, 16, 200) != ZVEC_OK ||
             zvec_field_schema_set_index_params(field, ip) != ZVEC_OK ||
             zvec_collection_schema_add_field(schema, field) != ZVEC_OK) {
    snprintf(err, err_len, "invalid vector settings (dim=%u)",
             (unsigned)a->dim);
    rc = ZVEC_ERROR_INVALID_ARGUMENT;
  } else if (add_filter_fields(schema, a) != 0) {
    snprintf(err, err_len, "cannot add the filter columns to the schema");
    rc = ZVEC_ERROR_INVALID_ARGUMENT;
  } else {
    rc = zvec_shared_create(path, schema, out, err, err_len);
  }
  if (ip != NULL) zvec_index_params_destroy(ip);
  if (field != NULL) zvec_field_schema_destroy(field);
  if (schema != NULL) zvec_collection_schema_destroy(schema);
  return rc;
}

// USING zvec('<path>'): open a collection that already exists.
static int open_existing(const char *path, zvec_shared **out, char **pz_err) {
  char err[256] = "";
  if (zvec_shared_open(path, out, err, sizeof err) != ZVEC_OK) {
    *pz_err =
        sqlite3_mprintf("zvec: cannot open collection '%s': %s", path, err);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

// Read-only open: a private handle that takes only a shared lock, so several
// processes can search one index and a read-only file system works.
static int open_readonly(const char *path, zvec_collection_t **out,
                         char **pz_err) {
  zvec_collection_options_t *opts = zvec_collection_options_create();
  zvec_error_code_t rc = ZVEC_ERROR_INTERNAL_ERROR;
  if (opts != NULL) {
    zvec_collection_options_set_read_only(opts, true);
    rc = zvec_collection_open(path, opts, out);
    zvec_collection_options_destroy(opts);
  }
  if (rc != ZVEC_OK) {
    zvec_error_details_t d = {0};
    *pz_err = sqlite3_mprintf(
        "zvec: cannot open collection '%s' read-only: %s", path,
        (zvec_get_last_error_details(&d) == ZVEC_OK && d.message)
            ? d.message
            : "unknown error");
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

// The index directory carries the table's random id as its schema name, so a
// settings row (possibly from an untrusted database file) cannot make the
// extension write to some other zvec collection, and a wrong or replaced
// directory is noticed.
static int verify_identity(zvec_collection_t *col, const char *uuid,
                           const char *path, char **pz_err) {
  zvec_collection_schema_t *schema = NULL;
  if (zvec_collection_get_schema(col, &schema) != ZVEC_OK) {
    *pz_err = sqlite3_mprintf("zvec: cannot read the schema of '%s'", path);
    return SQLITE_ERROR;
  }
  const char *name = zvec_collection_schema_get_name(schema);
  char want[48];
  snprintf(want, sizeof want, "zvec_%s", uuid);
  int ok = name != NULL && strcmp(name, want) == 0;
  zvec_collection_schema_destroy(schema);
  if (!ok) {
    *pz_err = sqlite3_mprintf(
        "zvec: the collection at '%s' does not belong to this table "
        "(wrong or replaced directory)",
        path);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

// USING zvec(base=..., ...) at CREATE time. Shadow tables first: if anything
// later fails, SQLite rolls the statement back and nothing is left behind.
// *created_path is set (caller frees) once a new collection directory exists.
static int create_derived(sqlite3 *db, const char *schema, const char *name,
                          zvec_args *a, zvec_shared **out, char **created_path,
                          char **pz_err) {
  char *msg = NULL;
  int rc = zvec_shadow_check_base(db, schema, a, &msg);
  if (rc != SQLITE_OK) {
    *pz_err = msg;
    return SQLITE_ERROR;
  }

  char *path = NULL;
  int is_default = 0;
  if (a->path != NULL) {
    path = sqlite3_mprintf("%s", a->path);
    if (path == NULL) return SQLITE_NOMEM;
  } else {
    rc = zvec_default_path(db, schema, name, &path, &msg);
    if (rc != SQLITE_OK) {
      *pz_err = msg;
      return SQLITE_ERROR;
    }
    is_default = 1;
  }

  if (path_exists(path)) {
    *pz_err = sqlite3_mprintf(
        "zvec: '%s' already exists (an index left over from an earlier "
        "table?); remove it or choose another path=",
        path);
    sqlite3_free(path);
    return SQLITE_ERROR;
  }

  if (a->uuid[0] != '\0') {
    *pz_err = sqlite3_mprintf("zvec: uuid is assigned automatically");
    sqlite3_free(path);
    return SQLITE_ERROR;
  }
  {
    unsigned char rnd[16];
    sqlite3_randomness((int)sizeof rnd, rnd);
    for (int i = 0; i < 16; i++) snprintf(a->uuid + 2 * i, 3, "%02x", rnd[i]);
  }
  rc = zvec_shadow_create(db, schema, name, a, path, is_default, &msg);
  if (rc != SQLITE_OK) {
    *pz_err = msg;
    sqlite3_free(path);
    return SQLITE_ERROR;
  }
  if (is_default && make_default_parent(path) != 0) {
    *pz_err =
        sqlite3_mprintf("zvec: cannot create the directory for '%s'", path);
    sqlite3_free(path);
    return SQLITE_ERROR;
  }

  char err[256] = "";
  char coll_name[48];
  snprintf(coll_name, sizeof coll_name, "zvec_%s", a->uuid);
  if (create_collection(a, coll_name, path, out, err, sizeof err) != ZVEC_OK) {
    *pz_err =
        sqlite3_mprintf("zvec: cannot create the index at '%s': %s", path, err);
    sqlite3_free(path);
    return SQLITE_ERROR;
  }
  *created_path = path;
  return SQLITE_OK;
}

// USING zvec(base=..., ...) at connect time: the stored settings win over the
// arguments, and a default location follows the database file. Nothing is
// opened here: the index is opened by the first search (vtab_ensure_open), so
// connecting, DROP TABLE and a missing or foreign directory never fail.
static int load_derived(sqlite3 *db, const char *schema, const char *name,
                        zvec_args *cfg, char **path, char **pz_err) {
  int is_default = 0;
  char *msg = NULL;
  int rc = zvec_shadow_load(db, schema, name, cfg, &is_default, &msg);
  if (rc != SQLITE_OK) {
    *pz_err = msg;
    return SQLITE_ERROR;
  }
  if (is_default) {
    rc = zvec_default_path(db, schema, name, path, &msg);
    if (rc != SQLITE_OK) *pz_err = msg;
  } else {
    *path = sqlite3_mprintf("%s", cfg->path);
    rc = *path != NULL ? SQLITE_OK : SQLITE_NOMEM;
  }
  if (rc != SQLITE_OK) {
    zvec_args_free(cfg);
    return rc == SQLITE_NOMEM ? rc : SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static char *missing_index_message(const char *path, const char *table) {
  return sqlite3_mprintf(
      "zvec: the index directory '%s' is missing (only the database file was "
      "copied?); "
      "SELECT zvec_rebuild('%s') recreates it from the table",
      path, table);
}

// Opens the index of a derived table when it is first needed.
static int vtab_ensure_open(zvec_vtab *v, char **pz_err) {
  if (v->shared != NULL || v->ro_col != NULL) return SQLITE_OK;
  if (!path_exists(v->path)) {
    *pz_err = missing_index_message(v->path, v->name);
    return SQLITE_ERROR;
  }
  int rc;
  if (v->read_only) {
    rc = open_readonly(v->path, &v->ro_col, pz_err);
    if (rc == SQLITE_OK)
      rc = verify_identity(v->ro_col, v->uuid, v->path, pz_err);
    if (rc != SQLITE_OK && v->ro_col != NULL) {
      zvec_collection_close(v->ro_col);
      v->ro_col = NULL;
    }
  } else {
    rc = open_existing(v->path, &v->shared, pz_err);
    if (rc == SQLITE_OK)
      rc = verify_identity(zvec_shared_collection(v->shared), v->uuid, v->path,
                           pz_err);
    if (rc != SQLITE_OK && v->shared != NULL) {
      zvec_shared_release(v->shared);
      v->shared = NULL;
    }
  }
  return rc;
}

// A derived table's columns follow from its stored settings.
static int fill_from_args(zvec_vtab *v, const zvec_args *a) {
  if (a->n_filter > 0) {
    v->scalar_names =
        sqlite3_malloc64((sqlite3_uint64)a->n_filter * sizeof *v->scalar_names);
    v->scalar_types =
        sqlite3_malloc64((sqlite3_uint64)a->n_filter * sizeof *v->scalar_types);
    if (v->scalar_names == NULL || v->scalar_types == NULL) return SQLITE_NOMEM;
  }
  for (int i = 0; i < a->n_filter; i++) {
    v->scalar_names[i] = sqlite3_mprintf("%s", a->filter_name[i]);
    if (v->scalar_names[i] == NULL) return SQLITE_NOMEM;
    v->scalar_types[i] = filter_data_type(a->filter_type[i]);
    v->n_scalar++;
  }
  v->vec_names = sqlite3_malloc64(sizeof *v->vec_names);
  v->vec_dims = sqlite3_malloc64(sizeof *v->vec_dims);
  v->vec_metrics = sqlite3_malloc64(sizeof *v->vec_metrics);
  v->vec_types = sqlite3_malloc64(sizeof *v->vec_types);
  if (v->vec_names == NULL || v->vec_dims == NULL || v->vec_metrics == NULL ||
      v->vec_types == NULL) {
    return SQLITE_NOMEM;
  }
  v->vec_names[0] = sqlite3_mprintf("%s", a->column);
  if (v->vec_names[0] == NULL) return SQLITE_NOMEM;
  v->vec_dims[0] = a->dim;
  v->vec_metrics[0] = (zvec_metric_type_t)a->metric;
  v->vec_types[0] = data_type_for(a->elem);
  v->n_vec = 1;
  return SQLITE_OK;
}

// Takes ownership of `shared` or `ro_col`: released again if anything below
// fails.
static int make_vtab(sqlite3 *db, const char *schema, const char *name,
                     int derived, zvec_shared *shared,
                     zvec_collection_t *ro_col, const zvec_args *cfg,
                     sqlite3_vtab **pp_vtab, char **pz_err) {
  zvec_vtab *vtab = sqlite3_malloc(sizeof *vtab);
  if (vtab == NULL) {
    if (ro_col != NULL)
      zvec_collection_close(ro_col);
    else
      zvec_shared_release(shared);
    return SQLITE_NOMEM;
  }
  memset(vtab, 0, sizeof *vtab);
  vtab->shared = shared;
  vtab->ro_col = ro_col;
  vtab->db = db;
  vtab->derived = derived;
  vtab->schema = sqlite3_mprintf("%s", schema);
  vtab->name = sqlite3_mprintf("%s", name);
  if (vtab->schema == NULL || vtab->name == NULL) {
    free_vtab(vtab);
    return SQLITE_NOMEM;
  }

  char *decl = NULL;
  if (cfg != NULL) {  // derived: columns come from the settings, no I/O
    int frc = fill_from_args(vtab, cfg);
    if (frc != SQLITE_OK) {
      free_vtab(vtab);
      return frc;
    }
    decl = build_decl(vtab);
    if (decl == NULL) {
      free_vtab(vtab);
      return SQLITE_NOMEM;
    }
  } else {
    decl = introspect_and_declare(vtab, pz_err);
    if (decl == NULL) {
      free_vtab(vtab);
      return SQLITE_ERROR;
    }
  }
  int rc = sqlite3_declare_vtab(db, decl);
  sqlite3_free(decl);
  if (rc != SQLITE_OK) {
    free_vtab(vtab);
    return rc;
  }
#ifdef SQLITE_VTAB_DIRECTONLY
  // Not usable from views, triggers or CHECK constraints: it opens files and
  // (once sync exists) writes them.
  sqlite3_vtab_config(db, SQLITE_VTAB_DIRECTONLY);
#endif
  *pp_vtab = &vtab->base;
  return SQLITE_OK;
}

static int zvec_open_table(sqlite3 *db, int argc, const char *const *argv,
                           sqlite3_vtab **pp_vtab, char **pz_err,
                           int creating) {
  zvec_args a;
  char err[256] = "";
  if (zvec_args_parse(argc, argv, &a, err, sizeof err) != 0) {
    *pz_err = sqlite3_mprintf("%s", err);
    return SQLITE_ERROR;
  }
  int derived = !a.attach;
  zvec_shared *shared = NULL;
  zvec_collection_t *ro_col = NULL;
  char *created_path = NULL, *path = NULL;
  int rc;
  int read_only = !creating && sqlite3_db_readonly(db, argv[1]) == 1;
  zvec_args *cfg =
      NULL;  // settings the columns are built from (derived tables)
  if (!derived) {
    rc = read_only ? open_readonly(a.path, &ro_col, pz_err)
                   : open_existing(a.path, &shared, pz_err);
  } else if (creating) {
    rc = create_derived(db, argv[1], argv[2], &a, &shared, &created_path,
                        pz_err);
    cfg = &a;
    if (rc == SQLITE_OK) path = sqlite3_mprintf("%s", created_path);
  } else {
    zvec_args_free(&a);  // the stored settings win over the arguments
    rc = load_derived(db, argv[1], argv[2], &a, &path, pz_err);
    cfg = &a;
  }
  if (rc != SQLITE_OK) {
    zvec_args_free(&a);
    return rc;
  }

  rc = make_vtab(db, argv[1], argv[2], derived, shared, ro_col, cfg, pp_vtab,
                 pz_err);
  if (rc == SQLITE_OK && derived) {
    zvec_vtab *v = (zvec_vtab *)*pp_vtab;
    v->sync_query = a.sync_query;
    v->read_only = read_only;
    memcpy(v->uuid, a.uuid, sizeof v->uuid);
    v->path = path;  // the vtab owns it now
    path = NULL;
  }
  if (rc != SQLITE_OK && created_path != NULL) {
    (void)remove_tree(created_path);  // make_vtab already released the handle
  }
  sqlite3_free(path);
  sqlite3_free(created_path);
  zvec_args_free(&a);
  return rc;
}

static int zvec_create(sqlite3 *db, void *p_aux, int argc,
                       const char *const *argv, sqlite3_vtab **pp_vtab,
                       char **pz_err) {
  (void)p_aux;
  return zvec_open_table(db, argc, argv, pp_vtab, pz_err, 1);
}

static int zvec_connect(sqlite3 *db, void *p_aux, int argc,
                        const char *const *argv, sqlite3_vtab **pp_vtab,
                        char **pz_err) {
  (void)p_aux;
  return zvec_open_table(db, argc, argv, pp_vtab, pz_err, 0);
}

static int zvec_disconnect(sqlite3_vtab *p_vtab) {
  free_vtab((zvec_vtab *)p_vtab);
  return SQLITE_OK;
}

// DROP TABLE: remove the shadow tables, never the collection directory (a
// rolled-back DROP could not bring deleted files back).
static int zvec_destroy(sqlite3_vtab *p_vtab) {
  zvec_vtab *vtab = (zvec_vtab *)p_vtab;
  if (vtab->derived) {
    char *msg = NULL;
    if (zvec_shadow_drop(vtab->db, vtab->schema, vtab->name, &msg) !=
        SQLITE_OK) {
      return fail_msg(p_vtab, msg);
    }
  }
  return zvec_disconnect(p_vtab);
}

// ---- zvec_sync() ----------------------------------------------------------
// Drains <t>_outbox into the zvec collection. For every queued key the base
// table's *current* row decides what happens, so replays and duplicates are
// harmless:  vector present and well-formed -> upsert; row gone or vector
// NULL -> delete; wrong-sized vector -> delete and count as skipped.
//
// Order matters for crash safety: write zvec, flush it, and only then advance
// the watermark and delete the processed outbox rows (one SAVEPOINT). A crash
// anywhere earlier leaves the rows queued and the next sync redoes them.
// Rows are deleted by seq, not by key, so a key that changes while a batch is
// running keeps its newer outbox row and is picked up by the next batch.

enum { SYNC_BATCH = 500 };

typedef struct sync_stats {
  sqlite3_int64 upserted;
  sqlite3_int64 deleted;
  sqlite3_int64 skipped;
} sync_stats;

static int sync_zvec_fail(char **err, const char *what) {
  zvec_error_details_t d = {0};
  const char *msg = (zvec_get_last_error_details(&d) == ZVEC_OK && d.message)
                        ? d.message
                        : "unknown error";
  *err = sqlite3_mprintf("zvec_sync: %s: %s", what, msg);
  return SQLITE_ERROR;
}

// Returns SQLite's own result code (SQLITE_BUSY and friends stay visible).
static int sync_sql_fail(sqlite3 *db, char **err) {
  int code = sqlite3_errcode(db) & 0xff;
  *err = sqlite3_mprintf("zvec_sync: %s", sqlite3_errmsg(db));
  return code != SQLITE_OK ? code : SQLITE_ERROR;
}

static sqlite3_stmt *sync_prepare(sqlite3 *db, char *sql) {
  sqlite3_stmt *st = NULL;
  if (sql == NULL) return NULL;
  if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK) st = NULL;
  sqlite3_free(sql);
  return st;
}

// Copies the mirrored columns (fetch columns 1..n) into the document. A value
// of the wrong storage class is left out, which reads back as NULL.
static int sync_add_filter_values(zvec_doc_t *doc, const zvec_args *cfg,
                                  sqlite3_stmt *fetch) {
  for (int i = 0; i < cfg->n_filter; i++) {
    int col = 1 + i, t = sqlite3_column_type(fetch, col);
    const char *name = cfg->filter_name[i];
    zvec_error_code_t rc = ZVEC_OK;
    if (cfg->filter_type[i] == ZVEC_FT_INT && t == SQLITE_INTEGER) {
      int64_t v = sqlite3_column_int64(fetch, col);
      rc = zvec_doc_add_field_by_value(doc, name, ZVEC_DATA_TYPE_INT64, &v,
                                       sizeof v);
    } else if (cfg->filter_type[i] == ZVEC_FT_REAL &&
               (t == SQLITE_FLOAT || t == SQLITE_INTEGER)) {
      double v = sqlite3_column_double(fetch, col);
      rc = zvec_doc_add_field_by_value(doc, name, ZVEC_DATA_TYPE_DOUBLE, &v,
                                       sizeof v);
    } else if (cfg->filter_type[i] == ZVEC_FT_TEXT && t == SQLITE_TEXT) {
      rc = zvec_doc_add_field_by_value(
          doc, name, ZVEC_DATA_TYPE_STRING, sqlite3_column_text(fetch, col),
          (size_t)sqlite3_column_bytes(fetch, col));
    }
    if (rc != ZVEC_OK) return -1;
  }
  return 0;
}

// One batch: `n` queued keys (duplicates allowed). Fills the stats.
static int sync_apply_batch(sqlite3 *db, zvec_collection_t *col,
                            const zvec_args *cfg, sqlite3_stmt *fetch,
                            char **pks, int n, sync_stats *stats, char **err) {
  const zvec_doc_t **docs = sqlite3_malloc64((sqlite3_uint64)n * sizeof *docs);
  const char **del = sqlite3_malloc64((sqlite3_uint64)n * sizeof *del);
  int n_docs = 0, n_del = 0, rc = SQLITE_OK;
  sqlite3_int64 skipped = 0;
  size_t want = (size_t)cfg->dim * zvec_elem_size(cfg->elem);
  if (docs == NULL || del == NULL) {
    sqlite3_free(docs);
    sqlite3_free(del);
    *err = sqlite3_mprintf("zvec_sync: out of memory");
    return SQLITE_NOMEM;
  }

  for (int i = 0; i < n && rc == SQLITE_OK; i++) {
    int dup = 0;
    for (int j = 0; j < i; j++) {
      if (strcmp(pks[i], pks[j]) == 0) {
        dup = 1;
        break;
      }
    }
    if (dup) continue;

    sqlite3_reset(fetch);
    sqlite3_bind_text(fetch, 1, pks[i], -1, SQLITE_STATIC);
    int step = sqlite3_step(fetch);
    if (step == SQLITE_ROW && sqlite3_column_type(fetch, 0) != SQLITE_NULL) {
      if (sqlite3_column_type(fetch, 0) == SQLITE_BLOB &&
          (size_t)sqlite3_column_bytes(fetch, 0) == want) {
        zvec_doc_t *doc = zvec_doc_create();
        if (doc == NULL) {
          rc = SQLITE_NOMEM;
          break;
        }
        zvec_doc_set_pk(doc, pks[i]);
        if (zvec_doc_add_field_by_value(
                doc, cfg->column, data_type_for(cfg->elem),
                sqlite3_column_blob(fetch, 0), want) != ZVEC_OK) {
          zvec_doc_destroy(doc);
          rc = sync_zvec_fail(err, "cannot build a document");
          break;
        }
        if (sync_add_filter_values(doc, cfg, fetch) != 0) {
          zvec_doc_destroy(doc);
          rc = sync_zvec_fail(err, "cannot add a filter column to a document");
          break;
        }
        docs[n_docs++] = doc;
      } else {
        skipped++;  // wrong type or length: keep it out of the index
        del[n_del++] = pks[i];
      }
    } else if (step == SQLITE_ROW || step == SQLITE_DONE) {
      del[n_del++] = pks[i];  // row gone, or its vector is NULL
    } else {
      rc = sync_sql_fail(db, err);
    }
  }
  sqlite3_reset(fetch);

  if (rc == SQLITE_NOMEM && *err == NULL)
    *err = sqlite3_mprintf("zvec_sync: out of memory");
  if (rc == SQLITE_OK && n_docs > 0) {
    size_t ok = 0, bad = 0;
    if (zvec_collection_upsert(col, docs, (size_t)n_docs, &ok, &bad) !=
        ZVEC_OK) {
      rc = sync_zvec_fail(err, "upsert failed");
    } else if (bad != 0 || ok != (size_t)n_docs) {
      *err = sqlite3_mprintf("zvec_sync: zvec rejected %lld of %d documents",
                             (sqlite3_int64)bad, n_docs);
      rc = SQLITE_ERROR;
    }
  }
  if (rc == SQLITE_OK && n_del > 0) {
    // A key that was never indexed counts as an error per key in zvec; for an
    // idempotent sync that is fine, only a failed call is.
    size_t ok = 0, bad = 0;
    if (zvec_collection_delete(col, del, (size_t)n_del, &ok, &bad) != ZVEC_OK) {
      rc = sync_zvec_fail(err, "delete failed");
    }
  }
  if (rc == SQLITE_OK && zvec_collection_flush(col) != ZVEC_OK) {
    rc = sync_zvec_fail(err, "flush failed");
  }
  for (int i = 0; i < n_docs; i++) zvec_doc_destroy((zvec_doc_t *)docs[i]);
  sqlite3_free(docs);
  sqlite3_free(del);
  if (rc == SQLITE_OK) {
    stats->upserted += n_docs;
    stats->deleted += n_del - skipped;
    stats->skipped += skipped;
  }
  return rc;
}

// `open` is an already open handle to reuse (sync before a search), or NULL.
static int sync_run(sqlite3 *db, const char *schema, const char *name,
                    zvec_shared *open, sync_stats *stats, char **err) {
  zvec_args cfg;
  int is_default = 0;
  char *msg = NULL;
  if (zvec_shadow_load(db, schema, name, &cfg, &is_default, &msg) !=
      SQLITE_OK) {
    *err = msg;
    return SQLITE_ERROR;
  }
  char *path = NULL;
  int rc = SQLITE_OK;
  zvec_shared *shared = open;
  if (shared == NULL) {
    if (is_default) {
      rc = zvec_default_path(db, schema, name, &path, &msg);
      if (rc != SQLITE_OK) *err = msg;
    } else {
      path = sqlite3_mprintf("%s", cfg.path);
      rc = path != NULL ? SQLITE_OK : SQLITE_NOMEM;
    }
    if (rc == SQLITE_OK) rc = open_existing(path, &shared, err);
    if (rc == SQLITE_OK) {
      rc = verify_identity(zvec_shared_collection(shared), cfg.uuid, path, err);
      if (rc != SQLITE_OK) {
        zvec_shared_release(shared);
        shared = NULL;
      }
    }
    sqlite3_free(path);
  }
  if (rc != SQLITE_OK) {
    zvec_args_free(&cfg);
    return rc == SQLITE_NOMEM ? rc : SQLITE_ERROR;
  }
  zvec_collection_t *col = zvec_shared_collection(shared);

  sqlite3_stmt *next = sync_prepare(
      db, sqlite3_mprintf("SELECT seq, pk FROM \"%w\".\"%w_outbox\" "
                          "WHERE seq > (SELECT watermark FROM "
                          "\"%w\".\"%w_state\" WHERE id = 1) "
                          "ORDER BY seq LIMIT %d",
                          schema, name, schema, name, (int)SYNC_BATCH));
  sqlite3_str *cols = sqlite3_str_new(NULL);
  sqlite3_str_appendf(cols, "\"%w\"", cfg.column);
  for (int i = 0; i < cfg.n_filter; i++)
    sqlite3_str_appendf(cols, ", \"%w\"", cfg.filter_name[i]);
  char *col_list = sqlite3_str_finish(cols);
  sqlite3_stmt *fetch = sync_prepare(
      db,
      col_list == NULL
          ? NULL
          : sqlite3_mprintf("SELECT %s FROM \"%w\".\"%w\" WHERE \"%w\" = ?1",
                            col_list, schema, cfg.base, cfg.key));
  sqlite3_free(col_list);
  sqlite3_stmt *done = sync_prepare(
      db, sqlite3_mprintf(
              "UPDATE \"%w\".\"%w_state\" SET watermark = ?1 WHERE id = 1; ",
              schema, name));
  sqlite3_stmt *purge = sync_prepare(
      db, sqlite3_mprintf("DELETE FROM \"%w\".\"%w_outbox\" WHERE seq <= ?1",
                          schema, name));
  rc = (next && fetch && done && purge) ? SQLITE_OK : sync_sql_fail(db, err);

  while (rc == SQLITE_OK) {
    char *pks[SYNC_BATCH];
    int n = 0;
    sqlite3_int64 max_seq = 0;
    sqlite3_reset(next);
    int step_rc = SQLITE_ROW;
    while (n < SYNC_BATCH && (step_rc = sqlite3_step(next)) == SQLITE_ROW) {
      max_seq = sqlite3_column_int64(next, 0);
      pks[n] =
          sqlite3_mprintf("%s", (const char *)sqlite3_column_text(next, 1));
      if (pks[n] == NULL) {
        rc = SQLITE_NOMEM;
        *err = sqlite3_mprintf("zvec_sync: out of memory");
        break;
      }
      n++;
    }
    if (rc == SQLITE_OK && step_rc != SQLITE_ROW && step_rc != SQLITE_DONE) {
      rc = sync_sql_fail(
          db, err);  // e.g. SQLITE_BUSY: not the same as "queue empty"
    }
    sqlite3_reset(next);
    if (rc == SQLITE_OK && n == 0) break;

    if (rc == SQLITE_OK) {
      rc = sync_apply_batch(db, col, &cfg, fetch, pks, n, stats, err);
    }
    for (int i = 0; i < n; i++) sqlite3_free(pks[i]);
    if (rc != SQLITE_OK) break;

    if (sqlite3_exec(db, "SAVEPOINT zvec_sync_batch", NULL, NULL, NULL) !=
        SQLITE_OK) {
      rc = sync_sql_fail(db, err);
      break;
    }
    sqlite3_bind_int64(done, 1, max_seq);
    sqlite3_bind_int64(purge, 1, max_seq);
    int ok = sqlite3_step(done) == SQLITE_DONE;
    sqlite3_reset(done);
    ok = ok && sqlite3_step(purge) == SQLITE_DONE;
    sqlite3_reset(purge);
    if (ok) {
      ok = sqlite3_exec(db, "RELEASE zvec_sync_batch", NULL, NULL, NULL) ==
           SQLITE_OK;
    }
    if (!ok) {
      rc = sync_sql_fail(db, err);
      sqlite3_exec(db, "ROLLBACK TO zvec_sync_batch; RELEASE zvec_sync_batch",
                   NULL, NULL, NULL);
    }
  }

  sqlite3_finalize(next);
  sqlite3_finalize(fetch);
  sqlite3_finalize(done);
  sqlite3_finalize(purge);
  if (open == NULL) zvec_shared_release(shared);
  zvec_args_free(&cfg);
  return rc;
}

// zvec_sync(name [, schema]) -> '{"upserted":N,"deleted":N,"skipped":N}'
// The index cannot take part in a transaction. Applying changes that this
// connection has not committed yet would leave documents in the index for rows
// that a ROLLBACK then removes, so nothing is synced while the connection has
// uncommitted writes. A plain read transaction is fine.
static int has_uncommitted_changes(sqlite3 *db) {
#ifdef SQLITE_TXN_WRITE
  if (sqlite3_libversion_number() >= 3034000) {
    return sqlite3_txn_state(db, NULL) == SQLITE_TXN_WRITE;
  }
#endif
  return !sqlite3_get_autocommit(db);
}

static int refuse_if_read_only(sqlite3_context *ctx, sqlite3 *db,
                               const char *schema, const char *fn) {
  if (sqlite3_db_readonly(db, schema) != 1) return 0;
  char *m = sqlite3_mprintf(
      "%s: the database is read-only; searches work, changes cannot be applied",
      fn);
  sqlite3_result_error(ctx, m != NULL ? m : "zvec: the database is read-only",
                       -1);
  sqlite3_free(m);
  return 1;
}

static void zvec_sync_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *name = (const char *)sqlite3_value_text(argv[0]);
  const char *schema =
      argc > 1 ? (const char *)sqlite3_value_text(argv[1]) : "main";
  if (name == NULL || schema == NULL) {
    sqlite3_result_error(ctx, "zvec_sync: the table name must be text", -1);
    return;
  }
  if (refuse_if_read_only(ctx, db, schema, "zvec_sync")) return;
  if (has_uncommitted_changes(db)) {
    sqlite3_result_error(
        ctx,
        "zvec_sync: this connection has uncommitted changes; commit first "
        "(the index cannot be rolled back)",
        -1);
    return;
  }
  sync_stats stats = {0, 0, 0};
  char *err = NULL;
  if (sync_run(db, schema, name, NULL, &stats, &err) != SQLITE_OK) {
    sqlite3_result_error(ctx, err != NULL ? err : "zvec_sync: failed", -1);
    sqlite3_free(err);
    return;
  }
  sqlite3_result_text(
      ctx,
      sqlite3_mprintf("{\"upserted\":%lld,\"deleted\":%lld,\"skipped\":%lld}",
                      stats.upserted, stats.deleted, stats.skipped),
      -1, sqlite3_free);
}

// ---- zvec_drop_index() ----------------------------------------------------
// DROP TABLE plus deleting the index directory, as one explicit call. Safety:
// - not inside a transaction (file deletion cannot be rolled back);
// - only the default location, which is recomputed from the database file
// name, is ever deleted, never a path read from the (untrusted) config;
// - not while another connection or process still holds the collection.

static int index_in_use(const char *path) {
  zvec_collection_options_t *opts = zvec_collection_options_create();
  zvec_collection_t *col = NULL;
  zvec_error_code_t rc = ZVEC_ERROR_INTERNAL_ERROR;
  if (opts != NULL) {
    zvec_collection_options_set_read_only(opts, true);  // shared lock only
    rc = zvec_collection_open(path, opts, &col);
    zvec_collection_options_destroy(opts);
  }
  if (rc == ZVEC_OK) {
    zvec_collection_close(col);
    return 0;
  }
  return 1;  // cannot open it: assume somebody holds the lock
}

static void json_string(sqlite3_str *s, const char *v) {
  sqlite3_str_appendchar(s, 1, '"');
  for (; *v; v++) {
    unsigned char c = (unsigned char)*v;
    if (c == '"' || c == '\\') {
      sqlite3_str_appendchar(s, 1, '\\');
      sqlite3_str_appendchar(s, 1, (char)c);
    } else if (c < 0x20) {
      sqlite3_str_appendf(s, "\\u%04x", c);
    } else {
      sqlite3_str_appendchar(s, 1, (char)c);
    }
  }
  sqlite3_str_appendchar(s, 1, '"');
}

// zvec_drop_index(name [, schema]) ->
// '{"dropped":"<name>","directory":"removed|kept: ...","path":"<dir>"}'
static void zvec_drop_index_fn(sqlite3_context *ctx, int argc,
                               sqlite3_value **argv) {
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *name = (const char *)sqlite3_value_text(argv[0]);
  const char *schema =
      argc > 1 ? (const char *)sqlite3_value_text(argv[1]) : "main";
  if (name == NULL || schema == NULL) {
    sqlite3_result_error(ctx, "zvec_drop_index: the table name must be text",
                         -1);
    return;
  }
  if (!sqlite3_get_autocommit(db)) {
    sqlite3_result_error(ctx,
                         "zvec_drop_index: cannot run inside a transaction "
                         "(deleting files cannot be rolled back)",
                         -1);
    return;
  }

  zvec_args cfg;
  int is_default = 0;
  char *msg = NULL;
  if (zvec_shadow_load(db, schema, name, &cfg, &is_default, &msg) !=
      SQLITE_OK) {
    sqlite3_result_error(ctx, msg != NULL ? msg : "zvec_drop_index: failed",
                         -1);
    sqlite3_free(msg);
    return;
  }
  char *path = NULL;
  if (is_default) {
    if (zvec_default_path(db, schema, name, &path, &msg) != SQLITE_OK) {
      zvec_args_free(&cfg);
      sqlite3_result_error(ctx, msg != NULL ? msg : "zvec_drop_index: failed",
                           -1);
      sqlite3_free(msg);
      return;
    }
  } else {
    path = sqlite3_mprintf("%s", cfg.path);
  }
  zvec_args_free(&cfg);
  if (path == NULL) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  char *sql = sqlite3_mprintf("DROP TABLE \"%w\".\"%w\"", schema, name);
  int rc = sql != NULL ? sqlite3_exec(db, sql, NULL, NULL, &msg) : SQLITE_NOMEM;
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    char *e = sqlite3_mprintf("zvec_drop_index: %s",
                              msg != NULL ? msg : "cannot drop the table");
    sqlite3_result_error(ctx, e != NULL ? e : "zvec_drop_index: failed", -1);
    sqlite3_free(e);
    sqlite3_free(msg);
    sqlite3_free(path);
    return;
  }

  const char *outcome;
  if (!is_default) {
    outcome = "kept: explicit path, remove it yourself";
  } else if (!path_exists(path)) {
    outcome = "none: the directory did not exist";
  } else if (index_in_use(path)) {
    outcome = "kept: still in use by another connection or process";
  } else if (remove_tree(path) != 0) {
    outcome = "kept: could not delete it";
  } else {
    remove_empty_parent(path);
    outcome = "removed";
  }

  sqlite3_str *out = sqlite3_str_new(NULL);
  sqlite3_str_appendall(out, "{\"dropped\":");
  json_string(out, name);
  sqlite3_str_appendall(out, ",\"directory\":");
  json_string(out, outcome);
  sqlite3_str_appendall(out, ",\"path\":");
  json_string(out, path);
  sqlite3_str_appendchar(out, 1, '}');
  sqlite3_free(path);
  char *text = sqlite3_str_finish(out);
  if (text == NULL) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  sqlite3_result_text(ctx, text, -1, sqlite3_free);
}

// ---- zvec_check() and zvec_rebuild() ----------------------------------------
// check:   compares the index with the base table and reports; changes nothing.
// rebuild: removes index entries that have no valid base row, then re-sends
// every base row. The repair for anything sync cannot see (REPLACE
// conflicts, rolled-back syncs, an index restored from a backup).

typedef struct scan_result {
  sqlite3_int64 base;     // base rows holding a vector of the right size
  sqlite3_int64 indexed;  // documents in the index
  sqlite3_int64 stale;    // indexed, but no valid base row
  sqlite3_int64 missing;  // valid base rows the index lacks
  sqlite3_int64 pending;  // queued changes not yet applied
  sqlite3_int64 removed;  // stale documents deleted (rebuild only)
} scan_result;

// create_missing: recreate an absent index directory (zvec_rebuild fills it).
static int open_index(sqlite3 *db, const char *schema, const char *name,
                      int create_missing, zvec_args *cfg, zvec_shared **shared,
                      char **err) {
  int is_default = 0;
  char *msg = NULL;
  if (zvec_shadow_load(db, schema, name, cfg, &is_default, &msg) != SQLITE_OK) {
    *err = msg;
    return SQLITE_ERROR;
  }
  char *path = NULL;
  int rc;
  if (is_default) {
    rc = zvec_default_path(db, schema, name, &path, &msg);
    if (rc != SQLITE_OK) *err = msg;
  } else {
    path = sqlite3_mprintf("%s", cfg->path);
    rc = path != NULL ? SQLITE_OK : SQLITE_NOMEM;
  }
  int created = 0;  // a collection created just now needs no identity check
  if (rc == SQLITE_OK && !path_exists(path)) {
    if (create_missing) {
      char cname[48], cerr[256] = "";
      snprintf(cname, sizeof cname, "zvec_%s", cfg->uuid);
      if (is_default && make_default_parent(path) != 0) {
        *err =
            sqlite3_mprintf("zvec: cannot create the directory for '%s'", path);
        rc = SQLITE_ERROR;
      } else if (create_collection(cfg, cname, path, shared, cerr,
                                   sizeof cerr) != ZVEC_OK) {
        *err = sqlite3_mprintf("zvec: cannot create the index at '%s': %s",
                               path, cerr);
        rc = SQLITE_ERROR;
      } else {
        created = 1;
      }
    } else {
      *err = missing_index_message(path, name);
      rc = SQLITE_ERROR;
    }
  }
  if (rc == SQLITE_OK && !created) rc = open_existing(path, shared, err);
  if (rc == SQLITE_OK && !created) {
    rc = verify_identity(zvec_shared_collection(*shared), cfg->uuid, path, err);
    if (rc != SQLITE_OK) {
      zvec_shared_release(*shared);
      *shared = NULL;
    }
  }
  sqlite3_free(path);
  if (rc != SQLITE_OK) {
    zvec_args_free(cfg);
    return rc == SQLITE_NOMEM ? rc : SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int scalar_query(sqlite3 *db, char *sql, sqlite3_int64 *out,
                        char **err) {
  sqlite3_stmt *st = NULL;
  if (sql == NULL) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW) {
    *out = sqlite3_column_int64(st, 0);
  } else {
    rc = SQLITE_ERROR;
    *err = sqlite3_mprintf("zvec: %s", sqlite3_errmsg(db));
  }
  sqlite3_finalize(st);
  return rc;
}

// Walks every document in the index. With do_delete, stale ones are removed.
static int scan_index(sqlite3 *db, const char *schema, const char *name,
                      const zvec_args *cfg, zvec_collection_t *col,
                      int do_delete, scan_result *r, char **err) {
  memset(r, 0, sizeof *r);
  long long want = (long long)cfg->dim * (long long)zvec_elem_size(cfg->elem);
  int rc = scalar_query(
      db,
      sqlite3_mprintf("SELECT count(*) FROM \"%w\".\"%w\" WHERE typeof(\"%w\") "
                      "= 'blob' AND length(\"%w\") = %lld",
                      schema, cfg->base, cfg->column, cfg->column, want),
      &r->base, err);
  if (rc == SQLITE_OK) {
    rc = scalar_query(
        db,
        sqlite3_mprintf(
            "SELECT count(*) FROM \"%w\".\"%w_outbox\" WHERE seq > "
            "(SELECT watermark FROM \"%w\".\"%w_state\" WHERE id = 1)",
            schema, name, schema, name),
        &r->pending, err);
  }
  if (rc != SQLITE_OK) return rc;

  sqlite3_stmt *look = NULL;
  char *sql = sqlite3_mprintf(
      "SELECT typeof(\"%w\") = 'blob' AND length(\"%w\") = %lld FROM "
      "\"%w\".\"%w\" "
      "WHERE \"%w\" = ?1 AND CAST(\"%w\" AS TEXT) = ?1",
      cfg->column, cfg->column, want, schema, cfg->base, cfg->key, cfg->key);
  if (sql == NULL ||
      sqlite3_prepare_v2(db, sql, -1, &look, NULL) != SQLITE_OK) {
    *err = sqlite3_mprintf("zvec: %s", sqlite3_errmsg(db));
    sqlite3_free(sql);
    return SQLITE_ERROR;
  }
  sqlite3_free(sql);

  char **stale = NULL;
  size_t n_stale = 0, cap = 0;
  const char *no_fields[1] = {NULL};
  zvec_iterator_options_t *opts = zvec_iterator_options_create();
  zvec_doc_iterator_t *it = NULL;
  if (opts == NULL ||
      zvec_iterator_options_set_output_fields(opts, no_fields, 0) != ZVEC_OK ||
      zvec_iterator_options_set_include_vector(opts, false) != ZVEC_OK ||
      zvec_collection_create_iterator(col, opts, &it) != ZVEC_OK) {
    rc = sync_zvec_fail(err, "cannot walk the index");
  }
  if (opts != NULL) zvec_iterator_options_destroy(opts);

  while (rc == SQLITE_OK) {
    zvec_doc_t *doc = NULL;
    if (zvec_doc_iterator_next(it, &doc) != ZVEC_OK) {
      rc = sync_zvec_fail(err, "walking the index failed");
      break;
    }
    if (doc == NULL) break;
    const char *pk = zvec_doc_get_pk_pointer(doc);
    r->indexed++;
    int valid = 0;
    if (pk != NULL) {
      sqlite3_reset(look);
      sqlite3_bind_text(look, 1, pk, -1, SQLITE_TRANSIENT);
      valid =
          sqlite3_step(look) == SQLITE_ROW && sqlite3_column_int(look, 0) == 1;
    }
    if (!valid) {
      r->stale++;
      if (do_delete && pk != NULL) {
        if (n_stale == cap) {
          size_t ncap = cap ? cap * 2 : 64;
          char **grown = sqlite3_realloc64(stale, ncap * sizeof *stale);
          if (grown == NULL) {
            rc = SQLITE_NOMEM;
            *err = sqlite3_mprintf("zvec: out of memory");
          } else {
            stale = grown;
            cap = ncap;
          }
        }
        if (rc == SQLITE_OK) {
          stale[n_stale] = sqlite3_mprintf("%s", pk);
          if (stale[n_stale] == NULL) {
            rc = SQLITE_NOMEM;
            *err = sqlite3_mprintf("zvec: out of memory");
          } else {
            n_stale++;
          }
        }
      }
    }
    zvec_doc_destroy(doc);
  }
  zvec_doc_iterator_close(
      it);  // before any delete: an open iterator pins the collection
  sqlite3_finalize(look);

  for (size_t i = 0; rc == SQLITE_OK && i < n_stale; i += SYNC_BATCH) {
    size_t n = n_stale - i < SYNC_BATCH ? n_stale - i : SYNC_BATCH, ok = 0,
           bad = 0;
    if (zvec_collection_delete(col, (const char *const *)(stale + i), n, &ok,
                               &bad) != ZVEC_OK) {
      rc = sync_zvec_fail(err, "delete failed");
    } else {
      r->removed += (sqlite3_int64)ok;
    }
  }
  if (rc == SQLITE_OK && n_stale > 0 && zvec_collection_flush(col) != ZVEC_OK) {
    rc = sync_zvec_fail(err, "flush failed");
  }
  for (size_t i = 0; i < n_stale; i++) sqlite3_free(stale[i]);
  sqlite3_free(stale);

  if (rc == SQLITE_OK) {
    r->missing = r->base - (r->indexed - r->stale);
    if (r->missing < 0) r->missing = 0;
  }
  return rc;
}

static int name_args(sqlite3_context *ctx, int argc, sqlite3_value **argv,
                     const char *fn, const char **name, const char **schema) {
  *name = (const char *)sqlite3_value_text(argv[0]);
  *schema = argc > 1 ? (const char *)sqlite3_value_text(argv[1]) : "main";
  if (*name == NULL || *schema == NULL) {
    char *m = sqlite3_mprintf("%s: the table name must be text", fn);
    sqlite3_result_error(ctx, m != NULL ? m : "zvec: bad arguments", -1);
    sqlite3_free(m);
    return -1;
  }
  return 0;
}

static void result_json(sqlite3_context *ctx, sqlite3_str *s) {
  char *text = sqlite3_str_finish(s);
  if (text == NULL) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  sqlite3_result_text(ctx, text, -1, sqlite3_free);
}

static void result_failure(sqlite3_context *ctx, char *err,
                           const char *fallback) {
  sqlite3_result_error(ctx, err != NULL ? err : fallback, -1);
  sqlite3_free(err);
}

// zvec_check(name [, schema]) ->
// '{"status":"ok|pending|repair","base":N,"indexed":N,"stale":N,"missing":N,"pending":N}'
// "pending": run zvec_sync (or just search). "repair": run zvec_rebuild.
static void zvec_check_fn(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *name, *schema;
  if (name_args(ctx, argc, argv, "zvec_check", &name, &schema) != 0) return;
  if (refuse_if_read_only(ctx, db, schema, "zvec_check")) return;
  zvec_args cfg;
  zvec_shared *shared = NULL;
  char *err = NULL;
  if (open_index(db, schema, name, 0, &cfg, &shared, &err) != SQLITE_OK) {
    result_failure(ctx, err, "zvec_check: failed");
    return;
  }
  scan_result r;
  int rc = scan_index(db, schema, name, &cfg, zvec_shared_collection(shared), 0,
                      &r, &err);
  zvec_shared_release(shared);
  zvec_args_free(&cfg);
  if (rc != SQLITE_OK) {
    result_failure(ctx, err, "zvec_check: failed");
    return;
  }
  const char *status = r.pending > 0                    ? "pending"
                       : (r.stale > 0 || r.missing > 0) ? "repair"
                                                        : "ok";
  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendf(
      s,
      "{\"status\":\"%s\",\"base\":%lld,\"indexed\":%lld,\"stale\":%lld,"
      "\"missing\":%lld,\"pending\":%lld}",
      status, r.base, r.indexed, r.stale, r.missing, r.pending);
  result_json(ctx, s);
}

// zvec_rebuild(name [, schema]) ->
// '{"removed":N,"queued":N,"upserted":N,"deleted":N,"skipped":N}'
static void zvec_rebuild_fn(sqlite3_context *ctx, int argc,
                            sqlite3_value **argv) {
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *name, *schema;
  if (name_args(ctx, argc, argv, "zvec_rebuild", &name, &schema) != 0) return;
  if (refuse_if_read_only(ctx, db, schema, "zvec_rebuild")) return;
  if (!sqlite3_get_autocommit(db)) {
    sqlite3_result_error(ctx,
                         "zvec_rebuild: cannot run inside a transaction "
                         "(changes to the index cannot be rolled back)",
                         -1);
    return;
  }
  zvec_args cfg;
  zvec_shared *shared = NULL;
  char *err = NULL;
  if (open_index(db, schema, name, 1, &cfg, &shared, &err) != SQLITE_OK) {
    result_failure(ctx, err, "zvec_rebuild: failed");
    return;
  }
  scan_result r;
  sync_stats stats = {0, 0, 0};
  sqlite3_int64 queued = 0;
  int rc = scan_index(db, schema, name, &cfg, zvec_shared_collection(shared), 1,
                      &r, &err);
  if (rc == SQLITE_OK) {
    char *sql = sqlite3_mprintf(
        "INSERT INTO \"%w\".\"%w_outbox\"(op, pk) SELECT 1, CAST(\"%w\" AS "
        "TEXT) FROM \"%w\".\"%w\"",
        schema, name, cfg.key, schema, cfg.base);
    char *msg = NULL;
    rc = sql != NULL ? sqlite3_exec(db, sql, NULL, NULL, &msg) : SQLITE_NOMEM;
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
      err = sqlite3_mprintf("zvec_rebuild: %s",
                            msg != NULL ? msg : "cannot queue the rows");
      sqlite3_free(msg);
    } else {
      queued = sqlite3_changes64(db);
      rc = sync_run(db, schema, name, shared, &stats, &err);
    }
  }
  zvec_shared_release(shared);
  zvec_args_free(&cfg);
  if (rc != SQLITE_OK) {
    result_failure(ctx, err, "zvec_rebuild: failed");
    return;
  }
  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendf(
      s,
      "{\"removed\":%lld,\"queued\":%lld,\"upserted\":%lld,\"deleted\":%lld,"
      "\"skipped\":%lld}",
      r.removed, queued, stats.upserted, stats.deleted, stats.skipped);
  result_json(ctx, s);
}

// Which scalar columns the zvec filter language can compare, and with what.
enum { PUSH_NONE = 0, PUSH_INT, PUSH_REAL, PUSH_TEXT };

static int push_class(zvec_data_type_t t) {
  switch (t) {
    case ZVEC_DATA_TYPE_INT32:
    case ZVEC_DATA_TYPE_INT64:
    case ZVEC_DATA_TYPE_UINT32:
    case ZVEC_DATA_TYPE_UINT64:
      return PUSH_INT;
    case ZVEC_DATA_TYPE_FLOAT:
    case ZVEC_DATA_TYPE_DOUBLE:
      return PUSH_REAL;
    case ZVEC_DATA_TYPE_STRING:
      return PUSH_TEXT;
    default:
      return PUSH_NONE;
  }
}

static int plain_name(const char *s) {
  if (s == NULL || !(isalpha((unsigned char)s[0]) || s[0] == '_')) return 0;
  for (; *s; s++) {
    if (!(isalnum((unsigned char)*s) || *s == '_')) return 0;
  }
  return 1;
}

// '=' 'n'(!=) '<' 'l'(<=) '>' 'g'(>=), or 0 when the operator is not pushed.
static char push_op(unsigned char op) {
  switch (op) {
    case SQLITE_INDEX_CONSTRAINT_EQ:
      return '=';
    case SQLITE_INDEX_CONSTRAINT_NE:
      return 'n';
    case SQLITE_INDEX_CONSTRAINT_LT:
      return '<';
    case SQLITE_INDEX_CONSTRAINT_LE:
      return 'l';
    case SQLITE_INDEX_CONSTRAINT_GT:
      return '>';
    case SQLITE_INDEX_CONSTRAINT_GE:
      return 'g';
    default:
      return 0;
  }
}

enum { MAX_PUSH = 16 };

static int zvec_best_index(sqlite3_vtab *p_vtab, sqlite3_index_info *p_info) {
  const zvec_vtab *vtab = (const zvec_vtab *)p_vtab;
  int match_idx = -1;
  int match_vec = -1;
  int k_idx = -1;

  for (int i = 0; i < p_info->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &p_info->aConstraint[i];
    if (!c->usable) continue;
    if (c->op == SQLITE_INDEX_CONSTRAINT_MATCH) {
      for (int j = 0; j < vtab->n_vec; j++) {
        if (c->iColumn == col_vec(vtab, j) && match_idx < 0) {
          match_idx = i;
          match_vec = j;
        }
      }
    } else if (c->op == SQLITE_INDEX_CONSTRAINT_EQ &&
               c->iColumn == col_k(vtab)) {
      k_idx = i;
    }
  }

  if (match_idx < 0) {
    p_info->idxNum = -1;
    p_info->estimatedCost = 1e12;
    return SQLITE_OK;
  }

  int argv_pos = 0;
  p_info->aConstraintUsage[match_idx].argvIndex = ++argv_pos;
  p_info->aConstraintUsage[match_idx].omit = 1;
  if (k_idx >= 0) {
    p_info->aConstraintUsage[k_idx].argvIndex = ++argv_pos;
    p_info->aConstraintUsage[k_idx].omit = 1;
  }

  // Comparisons on scalar columns become a zvec filter. They are NOT marked
  // as handled (omit stays 0): SQLite re-checks every returned row, so a value
  // the filter language cannot express is simply not pushed (see xFilter).
  sqlite3_str *spec = sqlite3_str_new(NULL);
  int n_push = 0;
  for (int i = 0; i < p_info->nConstraint && n_push < MAX_PUSH; i++) {
    const struct sqlite3_index_constraint *c = &p_info->aConstraint[i];
    int si = c->iColumn - col_scalar(0);
    char op = push_op(c->op);
    if (!c->usable || op == 0 || si < 0 || si >= vtab->n_scalar) continue;
    if (push_class(vtab->scalar_types[si]) == PUSH_NONE ||
        !plain_name(vtab->scalar_names[si]))
      continue;
    p_info->aConstraintUsage[i].argvIndex = ++argv_pos;
    p_info->aConstraintUsage[i].omit = 0;
    sqlite3_str_appendf(spec, "%d%c;", si, op);
    n_push++;
  }
  char *spec_str = str_done(spec);
  if (spec_str == NULL) return SQLITE_NOMEM;
  if (n_push > 0) {
    p_info->idxStr = spec_str;
    p_info->needToFreeIdxStr = 1;
  } else {
    sqlite3_free(spec_str);
  }

  p_info->idxNum = (match_vec << 1) | (k_idx >= 0 ? 1 : 0);
  p_info->estimatedCost = n_push > 0 ? 50.0 : 100.0;
  p_info->estimatedRows = 10;
  return SQLITE_OK;
}

// Appends `name <op> literal` for one pushed comparison. Returns 0 when this
// value cannot be written safely in the filter language (it is then left to
// SQLite's own re-check).
static int append_comparison(sqlite3_str *out, const zvec_vtab *vtab, int si,
                             char op, sqlite3_value *val) {
  int cls = push_class(vtab->scalar_types[si]);
  int vt = sqlite3_value_type(val);
  const char *op_text = op == '='   ? "="
                        : op == 'n' ? "!="
                        : op == '<' ? "<"
                        : op == 'l' ? "<="
                        : op == '>' ? ">"
                                    : ">=";
  sqlite3_str *lit = sqlite3_str_new(NULL);

  if (cls == PUSH_INT && vt == SQLITE_INTEGER) {
    sqlite3_str_appendf(lit, "%lld", sqlite3_value_int64(val));
  } else if (cls == PUSH_REAL && (vt == SQLITE_INTEGER || vt == SQLITE_FLOAT)) {
    double d = sqlite3_value_double(val);
    char buf[40];
    if (!isfinite(d)) goto skip;
    snprintf(buf, sizeof buf, "%.17g", d);
    if (strpbrk(buf, "eEnNiI") != NULL)
      goto skip;  // exponent forms: not worth the risk
    sqlite3_str_appendall(lit, buf);
    if (strchr(buf, '.') == NULL) sqlite3_str_appendall(lit, ".0");
  } else if (cls == PUSH_TEXT && vt == SQLITE_TEXT) {
    const unsigned char *t = sqlite3_value_text(val);
    int n = sqlite3_value_bytes(val);
    if (t == NULL || (int)strlen((const char *)t) != n)
      goto skip;  // embedded NUL
    for (int i = 0; i < n; i++) {
      if (t[i] == '\\' || t[i] < 0x20)
        goto skip;  // the filter lexer's escapes are not reliable
    }
    sqlite3_str_appendchar(lit, 1, '\'');
    for (int i = 0; i < n; i++) {
      if (t[i] == '\'') sqlite3_str_appendchar(lit, 1, '\\');
      sqlite3_str_appendchar(lit, 1, (char)t[i]);
    }
    sqlite3_str_appendchar(lit, 1, '\'');
  } else {
    goto skip;
  }
  {
    char *l = sqlite3_str_finish(lit);
    if (l == NULL) return 0;
    if (sqlite3_str_length(out) > 0) sqlite3_str_appendall(out, " AND ");
    sqlite3_str_appendf(out, "%s %s %s", vtab->scalar_names[si], op_text, l);
    sqlite3_free(l);
    return 1;
  }
skip:
  sqlite3_free(sqlite3_str_finish(lit));
  return 0;
}

// True when changes are waiting in the outbox.
static int outbox_pending(sqlite3 *db, const char *schema, const char *name) {
  char *sql = sqlite3_mprintf(
      "SELECT 1 FROM \"%w\".\"%w_outbox\" WHERE seq > "
      "(SELECT watermark FROM \"%w\".\"%w_state\" WHERE id = 1) LIMIT 1",
      schema, name, schema, name);
  sqlite3_stmt *st = NULL;
  int pending = 1;  // if in doubt, sync and let it report the problem
  if (sql != NULL && sqlite3_prepare_v2(db, sql, -1, &st, NULL) == SQLITE_OK) {
    int rc = sqlite3_step(st);
    pending = rc == SQLITE_ROW || (rc != SQLITE_DONE);
  }
  sqlite3_finalize(st);
  sqlite3_free(sql);
  return pending;
}

static int zvec_open_cursor(sqlite3_vtab *p_vtab,
                            sqlite3_vtab_cursor **pp_cursor) {
  (void)p_vtab;
  zvec_cursor *cur = sqlite3_malloc(sizeof *cur);
  if (cur == NULL) return SQLITE_NOMEM;
  memset(cur, 0, sizeof *cur);
  *pp_cursor = &cur->base;
  return SQLITE_OK;
}

static int zvec_close_cursor(sqlite3_vtab_cursor *p_cursor) {
  clear_results((zvec_cursor *)p_cursor);
  sqlite3_free(p_cursor);
  return SQLITE_OK;
}

static int zvec_filter(sqlite3_vtab_cursor *p_cursor, int idx_num,
                       const char *idx_str, int argc, sqlite3_value **argv) {
  zvec_cursor *cur = (zvec_cursor *)p_cursor;
  zvec_vtab *vtab = (zvec_vtab *)p_cursor->pVtab;
  clear_results(cur);

  if (idx_num < 0) {
    return fail_msg(
        p_cursor->pVtab,
        sqlite3_mprintf("zvec: a MATCH on a vector column is "
                        "required (WHERE <vector column> MATCH ?)"));
  }
  int vec = idx_num >> 1;
  cur->vec = vec;
  int has_k = idx_num & 1;
  if (argc < 1 + has_k) {
    return fail_msg(p_cursor->pVtab,
                    sqlite3_mprintf("zvec: internal error: missing arguments"));
  }

  // A NULL query vector matches nothing, as in ordinary SQL.
  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) return SQLITE_OK;

  size_t want =
      (size_t)vtab->vec_dims[vec] * vec_elem_size(vtab->vec_types[vec]);
  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
      (size_t)sqlite3_value_bytes(argv[0]) != want) {
    return fail_msg(
        p_cursor->pVtab,
        sqlite3_mprintf("zvec: query vector for \"%s\" must be a BLOB of "
                        "%u %s values (%lld bytes)",
                        vtab->vec_names[vec], (unsigned)vtab->vec_dims[vec],
                        vec_type_name(vtab->vec_types[vec]),
                        (sqlite3_int64)want));
  }

  int k = 10;  // default when k is not given
  if (has_k) {
    if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER) {
      return fail_msg(p_cursor->pVtab,
                      sqlite3_mprintf("zvec: k must be an integer"));
    }
    sqlite3_int64 kk = sqlite3_value_int64(argv[1]);
    if (kk < 1 || kk > 10000) {
      return fail_msg(p_cursor->pVtab,
                      sqlite3_mprintf("zvec: k must be between 1 and 10000"));
    }
    k = (int)kk;
  }

  char *open_err = NULL;
  if (vtab->derived && vtab_ensure_open(vtab, &open_err) != SQLITE_OK) {
    return fail_msg(p_cursor->pVtab, open_err);
  }

  // sync=query: make the index current first (read-your-writes). A database
  // opened read-only cannot be synced; it is searched as it is.
  if (vtab->derived && vtab->sync_query && vtab->shared != NULL &&
      sqlite3_db_readonly(vtab->db, vtab->schema) != 1 &&
      !has_uncommitted_changes(vtab->db) &&
      outbox_pending(vtab->db, vtab->schema, vtab->name)) {
    sync_stats stats = {0, 0, 0};
    char *err = NULL;
    int src = sync_run(vtab->db, vtab->schema, vtab->name, vtab->shared, &stats,
                       &err);
    // Another connection holds the write lock: search what is indexed now.
    // The queued changes stay queued and the next search or sync applies them.
    if (src == SQLITE_BUSY || src == SQLITE_LOCKED) src = SQLITE_OK;
    char *msg =
        src == SQLITE_OK
            ? NULL
            : sqlite3_mprintf("zvec: syncing before the search failed (%s)",
                              err != NULL ? err : "unknown error");
    sqlite3_free(err);
    if (src != SQLITE_OK) return fail_msg(p_cursor->pVtab, msg);
  }

  // Pushed-down comparisons: idx_str is "<scalar index><op>;..." in argv order.
  sqlite3_str *fstr = sqlite3_str_new(NULL);
  int next_arg = 1 + has_k;
  for (const char *p = idx_str; p != NULL && *p != '\0' && next_arg < argc;) {
    char *end = NULL;
    long si = strtol(p, &end, 10);
    char op = end != NULL ? *end : '\0';
    if (end == p || op == '\0' || end[1] != ';' || si < 0 ||
        si >= vtab->n_scalar)
      break;
    append_comparison(fstr, vtab, (int)si, op, argv[next_arg++]);
    p = end + 2;
  }
  char *filter = str_done(fstr);
  if (filter == NULL) return SQLITE_NOMEM;

  zvec_doc_t **docs = NULL;
  size_t n = 0;
  zvec_error_code_t rc = ZVEC_ERROR_INTERNAL_ERROR;
  for (int attempt = 0; attempt < 2; attempt++) {
    zvec_vector_query_t *q = zvec_vector_query_create();
    if (q == NULL) {
      sqlite3_free(filter);
      return SQLITE_NOMEM;
    }
    zvec_vector_query_set_topk(q, k);
    zvec_vector_query_set_field_name(q, vtab->vec_names[vec]);
    zvec_vector_query_set_include_vector(q, false);
    zvec_vector_query_set_query_vector(q, sqlite3_value_blob(argv[0]), want);
    if (attempt == 0 && filter[0] != '\0')
      zvec_vector_query_set_filter(q, filter);
    rc = zvec_collection_query(vtab_col(vtab), q, &docs, &n);
    zvec_vector_query_destroy(q);
    // If zvec rejects the filter text (an unusual column name, say), search
    // without it: SQLite re-checks every row, so the result stays correct.
    if (rc == ZVEC_OK || filter[0] == '\0') break;
  }
  sqlite3_free(filter);
  if (rc != ZVEC_OK) return fail_from_zvec(p_cursor->pVtab, "query failed");

  cur->docs = docs;
  cur->n_docs = n;
  cur->pos = 0;
  return SQLITE_OK;
}

static int zvec_next(sqlite3_vtab_cursor *p_cursor) {
  ((zvec_cursor *)p_cursor)->pos++;
  return SQLITE_OK;
}

static int zvec_eof(sqlite3_vtab_cursor *p_cursor) {
  const zvec_cursor *cur = (const zvec_cursor *)p_cursor;
  return cur->pos >= cur->n_docs;
}

static int zvec_rowid(sqlite3_vtab_cursor *p_cursor, sqlite3_int64 *p_rowid) {
  *p_rowid = (sqlite3_int64)((zvec_cursor *)p_cursor)->pos;
  return SQLITE_OK;
}

// Only tables that ordinary SQL never has to write. The outbox is written by
// triggers, which SQLite's defensive mode would block.
static int zvec_shadow_name(const char *suffix) {
  return strcmp(suffix, "config") == 0 || strcmp(suffix, "state") == 0;
}

static const sqlite3_module zvec_module = {
    .iVersion = 3,
    .xCreate = zvec_create,
    .xConnect = zvec_connect,
    .xBestIndex = zvec_best_index,
    .xDisconnect = zvec_disconnect,
    .xDestroy = zvec_destroy,
    .xOpen = zvec_open_cursor,
    .xClose = zvec_close_cursor,
    .xFilter = zvec_filter,
    .xNext = zvec_next,
    .xEof = zvec_eof,
    .xColumn = zvec_column,
    .xRowid = zvec_rowid,
    .xShadowName = zvec_shadow_name,
};

int zvec_vtab_register(sqlite3 *db) {
  int rc = sqlite3_create_module_v2(db, "zvec", &zvec_module, NULL, NULL);
  if (rc != SQLITE_OK) return rc;
  int flags = SQLITE_UTF8;
#ifdef SQLITE_DIRECTONLY
  flags |= SQLITE_DIRECTONLY;  // touches files: no views, triggers or CHECKs
#endif
  static const struct {
    const char *name;
    int n_arg;
    void (*fn)(sqlite3_context *, int, sqlite3_value **);
  } funcs[] = {{"zvec_sync", 1, zvec_sync_fn},
               {"zvec_sync", 2, zvec_sync_fn},
               {"zvec_drop_index", 1, zvec_drop_index_fn},
               {"zvec_drop_index", 2, zvec_drop_index_fn},
               {"zvec_check", 1, zvec_check_fn},
               {"zvec_check", 2, zvec_check_fn},
               {"zvec_rebuild", 1, zvec_rebuild_fn},
               {"zvec_rebuild", 2, zvec_rebuild_fn}};
  for (size_t i = 0; i < sizeof funcs / sizeof funcs[0] && rc == SQLITE_OK;
       i++) {
    rc = sqlite3_create_function_v2(db, funcs[i].name, funcs[i].n_arg, flags,
                                    NULL, funcs[i].fn, NULL, NULL, NULL);
  }
  return rc;
}
