#include <sqlite3ext.h>
#include <string.h>
#include "registry.h"
#include "zvec/c_api.h"

SQLITE_EXTENSION_INIT3

typedef struct zvec_vtab {
  sqlite3_vtab base;
  zvec_shared *shared;
  int n_scalar;
  char **scalar_names;
  zvec_data_type_t *scalar_types;
  int n_vec;
  char **vec_names;
  uint32_t *vec_dims;
  zvec_metric_type_t *vec_metrics;
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

static int col_scalar(int i) { return 1 + i; }
static int col_vec(const zvec_vtab *v, int j) { return 1 + v->n_scalar + j; }
static int col_score(const zvec_vtab *v) { return 1 + v->n_scalar + v->n_vec; }
static int col_distance(const zvec_vtab *v) { return col_score(v) + 1; }
static int col_k(const zvec_vtab *v) { return col_score(v) + 2; }

static void result_scalar(sqlite3_context *ctx, const zvec_doc_t *doc,
                          const char *name, zvec_data_type_t type) {
  if (!zvec_doc_has_field_value(doc, name) || zvec_doc_is_field_null(doc, name)) {
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
      if (zvec_doc_get_field_value_pointer(doc, name, type, &p, &size) != ZVEC_OK ||
          p == NULL) {
        sqlite3_result_null(ctx);
      } else if (type == ZVEC_DATA_TYPE_STRING) {
        sqlite3_result_text(ctx, (const char *)p, -1, SQLITE_TRANSIENT); /* NUL-terminated */
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
    if (pk != NULL) sqlite3_result_text(ctx, pk, -1, SQLITE_TRANSIENT);
    else sqlite3_result_null(ctx);
  } else if (col == col_score(vtab)) {
    sqlite3_result_double(ctx, (double)zvec_doc_get_score(doc));
  } else if (col == col_distance(vtab)) {
    double d = (double)zvec_doc_get_score(doc);
    if (vtab->vec_metrics[cur->vec] == ZVEC_METRIC_TYPE_IP) d = -d; /* larger IP = closer */
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

  rc = zvec_collection_get_schema(zvec_shared_collection(vtab->shared),
                                  &schema);
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
    vtab->scalar_names =
        sqlite3_malloc64(count * sizeof *vtab->scalar_names);
    vtab->scalar_types =
        sqlite3_malloc64(count * sizeof *vtab->scalar_types);
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
    if (vtab->vec_names == NULL || vtab->vec_dims == NULL || vtab->vec_metrics == NULL) {
      problem = "out of memory";
      goto done;
    }
  }
  for (size_t j = 0; j < count; j++) {
    if (zvec_field_schema_get_data_type(fields[j]) !=
        ZVEC_DATA_TYPE_VECTOR_FP32) {
      continue;
    }
    char *name = sqlite3_mprintf("%s", zvec_field_schema_get_name(fields[j]));
    if (name == NULL) {
      problem = "out of memory";
      goto done;
    }
    vtab->vec_names[vtab->n_vec] = name;
    vtab->vec_dims[vtab->n_vec] = zvec_field_schema_get_dimension(fields[j]);
    const zvec_index_params_t *ip = zvec_field_schema_get_index_params(fields[j]);
    vtab->vec_metrics[vtab->n_vec] =
        ip != NULL ? zvec_index_params_get_metric_type(ip)
                   : ZVEC_METRIC_TYPE_UNDEFINED;
    vtab->n_vec++;
  }
  zvec_free(fields);
  fields = NULL;

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

  zvec_vtab *vtab = sqlite3_malloc(sizeof *vtab);
  if (vtab == NULL) {
    zvec_shared_release(shared);
    return SQLITE_NOMEM;
  }
  memset(vtab, 0, sizeof *vtab);
  vtab->shared = shared;

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

static int zvec_open_cursor(sqlite3_vtab *p_vtab, sqlite3_vtab_cursor **pp_cursor) {
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
    return fail_msg(p_cursor->pVtab,
                    sqlite3_mprintf("zvec: a MATCH on a vector column is "
                                    "required (WHERE <vector column> MATCH ?)"));
  }
  int vec = idx_num >> 1;
  cur->vec = vec;
  int has_k = idx_num & 1;
  if (argc < 1 + has_k) {
    return fail_msg(p_cursor->pVtab, sqlite3_mprintf("zvec: internal error: missing arguments"));
  }

  /* A NULL query vector matches nothing, as in ordinary SQL. */
  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) return SQLITE_OK;

  size_t want = (size_t)vtab->vec_dims[vec] * sizeof(float);
  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
      (size_t)sqlite3_value_bytes(argv[0]) != want) {
    return fail_msg(p_cursor->pVtab,
                    sqlite3_mprintf("zvec: query vector for \"%s\" must be a BLOB of "
                                    "%u float32 values (%lld bytes)",
                                    vtab->vec_names[vec],
                                    (unsigned)vtab->vec_dims[vec], (sqlite3_int64)want));
  }

  int k = 10; /* default when k is not given */
  if (has_k) {
    if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER) {
      return fail_msg(p_cursor->pVtab, sqlite3_mprintf("zvec: k must be an integer"));
    }
    sqlite3_int64 kk = sqlite3_value_int64(argv[1]);
    if (kk < 1 || kk > 10000) {
      return fail_msg(p_cursor->pVtab, sqlite3_mprintf("zvec: k must be between 1 and 10000"));
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
  zvec_error_code_t rc = zvec_collection_query(
      zvec_shared_collection(vtab->shared), q, &docs, &n);
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

static const sqlite3_module zvec_module = {
    .iVersion = 0,
    .xCreate = zvec_connect,
    .xConnect = zvec_connect,
    .xBestIndex = zvec_best_index,
    .xDisconnect = zvec_disconnect,
    .xDestroy = zvec_disconnect, 
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

