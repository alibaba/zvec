#define _XOPEN_SOURCE 700 /* nftw, mkdir */

#include <sys/stat.h>
#include <errno.h>
#include <sqlite3ext.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <ftw.h>
#endif
#include "zvec/c_api.h"
#include "args.h"
#include "registry.h"

SQLITE_EXTENSION_INIT3

typedef struct zvec_vtab {
  sqlite3_vtab base;
  sqlite3 *db;  /* needed by xDestroy to drop the shadow tables */
  char *schema; /* "main", "temp", ... */
  char *name;   /* the virtual table's own name */
  int derived;  /* 1: created with base=..., owns shadow tables */
  zvec_shared *shared;
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
  sqlite3_vtab_cursor base; /* must be first */
  zvec_doc_t **docs;        /* query results, owned by the cursor */
  size_t n_docs;
  size_t pos;
  int vec;
} zvec_cursor;

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

/* Bytes per element for the vector types the virtual table can query. */
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
    case ZVEC_DATA_TYPE_UINT64: { /* values above 2^63-1 wrap negative */
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
                            SQLITE_TRANSIENT); /* NUL-terminated */
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
      d = -d; /* larger IP = closer */
    sqlite3_result_double(ctx, d);
  } else {
    int i = col - col_scalar(0);
    if (i >= 0 && i < vtab->n_scalar) {
      result_scalar(ctx, doc, vtab->scalar_names[i], vtab->scalar_types[i]);
    } else {
      sqlite3_result_null(ctx); /* hidden vector / k columns */
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

static char *introspect_and_declare(zvec_vtab *vtab, char **pz_err) {
  zvec_collection_schema_t *schema = NULL;
  zvec_field_schema_t **fields = NULL;
  size_t count = 0;
  char *decl = NULL;
  const char *problem = NULL;
  zvec_error_code_t rc;

  rc =
      zvec_collection_get_schema(zvec_shared_collection(vtab->shared), &schema);
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
    if (vec_elem_size(vtype) == 0) continue; /* sparse, int4, binary, ... */
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
  decl = sqlite3_str_finish(s);
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
  sqlite3_free(vtab->vec_types);
  for (int i = 0; i < vtab->n_scalar; i++) sqlite3_free(vtab->scalar_names[i]);
  sqlite3_free(vtab->scalar_names);
  sqlite3_free(vtab->scalar_types);
  for (int j = 0; j < vtab->n_vec; j++) sqlite3_free(vtab->vec_names[j]);
  sqlite3_free(vtab->vec_names);
  sqlite3_free(vtab->vec_dims);
  sqlite3_free(vtab->vec_metrics);
  zvec_shared_release(vtab->shared);
  sqlite3_free(vtab);
}

/* ---- shadow tables ------------------------------------------------------
 * A derived index keeps its settings and its change queue in three ordinary
 * tables next to the virtual table: <t>_config, <t>_state, <t>_outbox. All
 * helpers return SQLITE_OK or an SQLite error code and, on failure, set *err
 * to an sqlite3_mprintf() string for the caller (zErrMsg) to own. */

#define SHADOW_VERSION "1"

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

/* Table names become a directory name: allow only [A-Za-z0-9_]. */
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

static int zvec_shadow_check_base(sqlite3 *db, const char *schema,
                                  const zvec_args *a, char **err) {
  sqlite3_stmt *st = NULL;
  int rc;

  /* Must be an ordinary table (not a view, not another virtual table). */
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

  rc = sqlite3_prepare_v2(db, "SELECT name, pk FROM pragma_table_xinfo(?1, ?2)",
                          -1, &st, NULL);
  if (rc != SQLITE_OK)
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
  sqlite3_bind_text(st, 1, a->base, -1, SQLITE_STATIC);
  sqlite3_bind_text(st, 2, schema, -1, SQLITE_STATIC);
  int have_key = 0, key_is_pk = 0, have_col = 0, pk_cols = 0;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    const char *name = (const char *)sqlite3_column_text(st, 0);
    int pk = sqlite3_column_int(st, 1);
    if (pk > 0) pk_cols++;
    if (name != NULL && sqlite3_stricmp(name, a->key) == 0) {
      have_key = 1;
      key_is_pk = pk > 0;
    }
    if (name != NULL && sqlite3_stricmp(name, a->column) == 0) have_col = 1;
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

/* Triggers on the base table queue the keys that changed, in the same
 * transaction as the change itself. The outbox stores no vectors: sync reads
 * the base table's current state for each queued key, so replays are harmless.
 * Trigger bodies may not qualify table names, hence the bare "<t>_outbox". */
static char *shadow_trigger_sql(const char *schema, const char *table,
                                const zvec_args *a) {
  sqlite3_str *q = sqlite3_str_new(NULL);
  sqlite3_str_appendf(
      q,
      "CREATE TRIGGER \"%w\".\"%w_zvec_ai\" AFTER INSERT ON \"%w\" BEGIN "
      "INSERT INTO \"%w_outbox\"(op, pk) VALUES(1, CAST(new.\"%w\" AS TEXT)); "
      "END;",
      schema, table, a->base, table, a->key);
  sqlite3_str_appendf(q,
                      "CREATE TRIGGER \"%w\".\"%w_zvec_au\" AFTER UPDATE OF "
                      "\"%w\", \"%w\" ON \"%w\" "
                      "WHEN new.\"%w\" IS NOT old.\"%w\" OR new.\"%w\" IS NOT "
                      "old.\"%w\" BEGIN ",
                      schema, table, a->column, a->key, a->base, a->column,
                      a->column, a->key, a->key);
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
  const char *rows[][2] = {{"version", SHADOW_VERSION},
                           {"base", a->base},
                           {"key", a->key},
                           {"column", a->column},
                           {"dim", dim},
                           {"metric", shadow_metric_name(a->metric)},
                           {"type", shadow_elem_name(a->elem)},
                           {"path", resolved_path},
                           {"path_default", path_is_default ? "1" : "0"}};

  char *sql = sqlite3_mprintf(
      "INSERT INTO \"%w\".\"%w_config\"(key, value) VALUES(?1, ?2)", schema,
      table);
  if (sql == NULL)
    return shadow_fail(err, SQLITE_NOMEM, "zvec: out of memory%s%s", NULL,
                       NULL);
  sqlite3_stmt *st = NULL;
  rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    return shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
  for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++) {
    sqlite3_reset(st);
    sqlite3_bind_text(st, 1, rows[i][0], -1, SQLITE_STATIC);
    sqlite3_bind_text(st, 2, rows[i][1], -1, SQLITE_STATIC);
    rc = sqlite3_step(st);
    if (rc != SQLITE_DONE) {
      rc = shadow_fail(err, rc, "zvec: %s%s", sqlite3_errmsg(db), NULL);
      sqlite3_finalize(st);
      return rc;
    }
  }
  sqlite3_finalize(st);
  return shadow_exec(db, shadow_trigger_sql(schema, table, a), err);
}

/* Wraps v in single quotes ('' escapes), the form zvec_args_parse expects. */
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
                      "DROP TABLE IF EXISTS \"%w\".\"%w_outbox\";"
                      "DROP TABLE IF EXISTS \"%w\".\"%w_state\";"
                      "DROP TABLE IF EXISTS \"%w\".\"%w_config\";",
                      schema, table, schema, table, schema, table, schema,
                      table, schema, table, schema, table),
      err);
}

/* ---- creating, connecting and destroying ------------------------------- */

#ifndef _WIN32
static int rm_cb(const char *p, const struct stat *sb, int flag,
                 struct FTW *ftw) {
  (void)sb;
  (void)flag;
  (void)ftw;
  return remove(p);
}
#endif

/* Only ever called on a directory this very call just created. */
static void remove_tree(const char *path) {
#ifndef _WIN32
  nftw(path, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
#else
  (void)path; /* TODO(windows): not implemented, the directory is left behind */
#endif
}

/* Creates the directory above <dbfile>.zvec/<name> (the default location). */
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

/* Builds a one-field collection (the vector column, HNSW) and creates it
 * through the registry. */
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
  } else {
    rc = zvec_shared_create(path, schema, out, err, err_len);
  }
  if (ip != NULL) zvec_index_params_destroy(ip);
  if (field != NULL) zvec_field_schema_destroy(field);
  if (schema != NULL) zvec_collection_schema_destroy(schema);
  return rc;
}

/* USING zvec('<path>'): open a collection that already exists. */
static int open_existing(const char *path, zvec_shared **out, char **pz_err) {
  char err[256] = "";
  if (zvec_shared_open(path, out, err, sizeof err) != ZVEC_OK) {
    *pz_err =
        sqlite3_mprintf("zvec: cannot open collection '%s': %s", path, err);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

/* USING zvec(base=..., ...) at CREATE time. Shadow tables first: if anything
 * later fails, SQLite rolls the statement back and nothing is left behind.
 * *created_path is set (caller frees) once a new collection directory exists.
 */
static int create_derived(sqlite3 *db, const char *schema, const char *name,
                          const zvec_args *a, zvec_shared **out,
                          char **created_path, char **pz_err) {
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
  if (create_collection(a, name, path, out, err, sizeof err) != ZVEC_OK) {
    *pz_err =
        sqlite3_mprintf("zvec: cannot create the index at '%s': %s", path, err);
    sqlite3_free(path);
    return SQLITE_ERROR;
  }
  *created_path = path;
  return SQLITE_OK;
}

/* USING zvec(base=..., ...) at connect time: the stored settings win over the
 * arguments, and a default location follows the database file. */
static int open_derived(sqlite3 *db, const char *schema, const char *name,
                        zvec_shared **out, char **pz_err) {
  zvec_args cfg;
  int is_default = 0;
  char *msg = NULL;
  int rc = zvec_shadow_load(db, schema, name, &cfg, &is_default, &msg);
  if (rc != SQLITE_OK) {
    *pz_err = msg;
    return SQLITE_ERROR;
  }
  char *path = NULL;
  if (is_default) {
    rc = zvec_default_path(db, schema, name, &path, &msg);
    if (rc != SQLITE_OK) *pz_err = msg;
  } else {
    path = sqlite3_mprintf("%s", cfg.path);
    rc = path != NULL ? SQLITE_OK : SQLITE_NOMEM;
  }
  zvec_args_free(&cfg);
  if (rc != SQLITE_OK) return rc == SQLITE_NOMEM ? rc : SQLITE_ERROR;
  rc = open_existing(path, out, pz_err);
  sqlite3_free(path);
  return rc;
}

/* Takes ownership of `shared`: released again if anything below fails. */
static int make_vtab(sqlite3 *db, const char *schema, const char *name,
                     int derived, zvec_shared *shared, sqlite3_vtab **pp_vtab,
                     char **pz_err) {
  zvec_vtab *vtab = sqlite3_malloc(sizeof *vtab);
  if (vtab == NULL) {
    zvec_shared_release(shared);
    return SQLITE_NOMEM;
  }
  memset(vtab, 0, sizeof *vtab);
  vtab->shared = shared;
  vtab->db = db;
  vtab->derived = derived;
  vtab->schema = sqlite3_mprintf("%s", schema);
  vtab->name = sqlite3_mprintf("%s", name);
  if (vtab->schema == NULL || vtab->name == NULL) {
    free_vtab(vtab);
    return SQLITE_NOMEM;
  }

  char *decl = introspect_and_declare(vtab, pz_err);
  if (decl == NULL) {
    free_vtab(vtab);
    return SQLITE_ERROR;
  }
  int rc = sqlite3_declare_vtab(db, decl);
  sqlite3_free(decl);
  if (rc != SQLITE_OK) {
    free_vtab(vtab);
    return rc;
  }
#ifdef SQLITE_VTAB_DIRECTONLY
  /* Not usable from views, triggers or CHECK constraints: it opens files and
   * (once sync exists) writes them. */
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
  char *created_path = NULL;
  int rc;
  if (!derived) {
    rc = open_existing(a.path, &shared, pz_err);
  } else if (creating) {
    rc = create_derived(db, argv[1], argv[2], &a, &shared, &created_path,
                        pz_err);
  } else {
    rc = open_derived(db, argv[1], argv[2], &shared, pz_err);
  }
  zvec_args_free(&a);
  if (rc != SQLITE_OK) return rc;

  rc = make_vtab(db, argv[1], argv[2], derived, shared, pp_vtab, pz_err);
  if (rc != SQLITE_OK && created_path != NULL) {
    remove_tree(created_path); /* make_vtab already released the handle */
  }
  sqlite3_free(created_path);
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

/* DROP TABLE: remove the shadow tables, never the collection directory (a
 * rolled-back DROP could not bring deleted files back). */
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

/* ---- zvec_sync() ----------------------------------------------------------
 * Drains <t>_outbox into the zvec collection. For every queued key the base
 * table's *current* row decides what happens, so replays and duplicates are
 * harmless:  vector present and well-formed -> upsert; row gone or vector
 * NULL -> delete; wrong-sized vector -> delete and count as skipped.
 *
 * Order matters for crash safety: write zvec, flush it, and only then advance
 * the watermark and delete the processed outbox rows (one SAVEPOINT). A crash
 * anywhere earlier leaves the rows queued and the next sync redoes them.
 * Rows are deleted by seq, not by key, so a key that changes while a batch is
 * running keeps its newer outbox row and is picked up by the next batch. */

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

static int sync_sql_fail(sqlite3 *db, char **err) {
  *err = sqlite3_mprintf("zvec_sync: %s", sqlite3_errmsg(db));
  return SQLITE_ERROR;
}

static sqlite3_stmt *sync_prepare(sqlite3 *db, char *sql) {
  sqlite3_stmt *st = NULL;
  if (sql == NULL) return NULL;
  if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK) st = NULL;
  sqlite3_free(sql);
  return st;
}

/* One batch: `n` queued keys (duplicates allowed). Fills the stats. */
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
        docs[n_docs++] = doc;
      } else {
        skipped++; /* wrong type or length: keep it out of the index */
        del[n_del++] = pks[i];
      }
    } else if (step == SQLITE_ROW || step == SQLITE_DONE) {
      del[n_del++] = pks[i]; /* row gone, or its vector is NULL */
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
    /* A key that was never indexed counts as an error per key in zvec; for an
     * idempotent sync that is fine, only a failed call is. */
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

static int sync_run(sqlite3 *db, const char *schema, const char *name,
                    sync_stats *stats, char **err) {
  zvec_args cfg;
  int is_default = 0;
  char *msg = NULL;
  if (zvec_shadow_load(db, schema, name, &cfg, &is_default, &msg) !=
      SQLITE_OK) {
    *err = msg;
    return SQLITE_ERROR;
  }
  char *path = NULL;
  int rc;
  if (is_default) {
    rc = zvec_default_path(db, schema, name, &path, &msg);
    if (rc != SQLITE_OK) *err = msg;
  } else {
    path = sqlite3_mprintf("%s", cfg.path);
    rc = path != NULL ? SQLITE_OK : SQLITE_NOMEM;
  }
  zvec_shared *shared = NULL;
  if (rc == SQLITE_OK) rc = open_existing(path, &shared, err);
  sqlite3_free(path);
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
  sqlite3_stmt *fetch = sync_prepare(
      db, sqlite3_mprintf("SELECT \"%w\" FROM \"%w\".\"%w\" WHERE \"%w\" = ?1",
                          cfg.column, schema, cfg.base, cfg.key));
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
    while (n < SYNC_BATCH && sqlite3_step(next) == SQLITE_ROW) {
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
      sync_sql_fail(db, err);
      sqlite3_exec(db, "ROLLBACK TO zvec_sync_batch; RELEASE zvec_sync_batch",
                   NULL, NULL, NULL);
      rc = SQLITE_ERROR;
    }
  }

  sqlite3_finalize(next);
  sqlite3_finalize(fetch);
  sqlite3_finalize(done);
  sqlite3_finalize(purge);
  zvec_shared_release(shared);
  zvec_args_free(&cfg);
  return rc;
}

/* zvec_sync(name [, schema]) -> '{"upserted":N,"deleted":N,"skipped":N}' */
static void zvec_sync_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *name = (const char *)sqlite3_value_text(argv[0]);
  const char *schema =
      argc > 1 ? (const char *)sqlite3_value_text(argv[1]) : "main";
  if (name == NULL || schema == NULL) {
    sqlite3_result_error(ctx, "zvec_sync: the table name must be text", -1);
    return;
  }
  sync_stats stats = {0, 0, 0};
  char *err = NULL;
  if (sync_run(db, schema, name, &stats, &err) != SQLITE_OK) {
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
  p_info->idxNum = (match_vec << 1) | (k_idx >= 0 ? 1 : 0);
  p_info->estimatedCost = 100.0;
  p_info->estimatedRows = 10;
  return SQLITE_OK;
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
  (void)idx_str;
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

  /* A NULL query vector matches nothing, as in ordinary SQL. */
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

  int k = 10; /* default when k is not given */
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

  zvec_vector_query_t *q = zvec_vector_query_create();
  if (q == NULL) return SQLITE_NOMEM;
  zvec_vector_query_set_topk(q, k);
  zvec_vector_query_set_field_name(q, vtab->vec_names[vec]);
  zvec_vector_query_set_include_vector(q, false);
  zvec_vector_query_set_query_vector(q, sqlite3_value_blob(argv[0]), want);

  zvec_doc_t **docs = NULL;
  size_t n = 0;
  zvec_error_code_t rc =
      zvec_collection_query(zvec_shared_collection(vtab->shared), q, &docs, &n);
  zvec_vector_query_destroy(q);
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

/* Only tables that ordinary SQL never has to write. The outbox is written by
 * triggers, which SQLite's defensive mode would block. */
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
  flags |= SQLITE_DIRECTONLY; /* touches files: no views, triggers or CHECKs */
#endif
  rc = sqlite3_create_function_v2(db, "zvec_sync", 1, flags, NULL, zvec_sync_fn,
                                  NULL, NULL, NULL);
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function_v2(db, "zvec_sync", 2, flags, NULL,
                                    zvec_sync_fn, NULL, NULL, NULL);
  }
  return rc;
}
