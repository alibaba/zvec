#include "args.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zvec/c_api.h"

static void set_err(char *err, size_t len, const char *fmt, const char *a) {
  if (err != NULL && len > 0) {
    snprintf(err, len, fmt, a != NULL ? a : "");
  }
}

static char *dup_n(const char *s, size_t n) {
  char *p = malloc(n + 1);
  if (p != NULL) {
    memcpy(p, s, n);
    p[n] = '\0';
  }
  return p;
}

/* Trims blanks and removes one level of '...' quoting ('' is a literal '). */
static char *clean_value(const char *s, size_t n) {
  while (n > 0 && isspace((unsigned char)*s)) {
    s++;
    n--;
  }
  while (n > 0 && isspace((unsigned char)s[n - 1])) {
    n--;
  }
  if (n >= 2 && s[0] == '\'' && s[n - 1] == '\'') {
    char *out = malloc(n);
    if (out == NULL) return NULL;
    size_t j = 0;
    for (size_t i = 1; i + 1 < n; i++) {
      out[j++] = s[i];
      if (s[i] == '\'' && i + 2 < n && s[i + 1] == '\'') i++;
    }
    out[j] = '\0';
    return out;
  }
  return dup_n(s, n);
}

/* Identifiers end up inside generated trigger SQL, so only plain ones. */
static int is_plain_ident(const char *s) {
  if (s == NULL || !(isalpha((unsigned char)s[0]) || s[0] == '_')) return 0;
  for (; *s; s++) {
    if (!(isalnum((unsigned char)*s) || *s == '_')) return 0;
  }
  return 1;
}

size_t zvec_elem_size(zvec_elem e) {
  switch (e) {
    case ZVEC_ELEM_F16:
      return 2;
    case ZVEC_ELEM_F64:
      return 8;
    case ZVEC_ELEM_I8:
      return 1;
    default:
      return 4;
  }
}

const char *zvec_ftype_name(zvec_ftype t) {
  switch (t) {
    case ZVEC_FT_INT:
      return "int";
    case ZVEC_FT_REAL:
      return "real";
    case ZVEC_FT_TEXT:
      return "text";
    default:
      return "unknown";
  }
}

void zvec_args_free(zvec_args *a) {
  if (a == NULL) return;
  for (int i = 0; i < a->n_filter; i++) free(a->filter_name[i]);
  free(a->path);
  free(a->base);
  free(a->key);
  free(a->column);
  memset(a, 0, sizeof *a);
}

static int set_str(char **slot, char *val, const char *name, char *err,
                   size_t err_len) {
  if (val == NULL) {
    set_err(err, err_len, "zvec: out of memory%s", "");
    return -1;
  }
  if (*slot != NULL) {
    free(val);
    set_err(err, err_len, "zvec: option '%s' given twice", name);
    return -1;
  }
  *slot = val;
  return 0;
}

/* "a, b:real, c:text" -> out->filter_*. Types are optional. */
static int parse_filter(const char *list, zvec_args *out, char *err,
                        size_t err_len) {
  const char *p = list;
  while (1) {
    while (isspace((unsigned char)*p)) p++;
    const char *end = p;
    while (*end != '\0' && *end != ',') end++;
    size_t n = (size_t)(end - p);
    while (n > 0 && isspace((unsigned char)p[n - 1])) n--;
    char *item = dup_n(p, n);
    if (item == NULL) {
      set_err(err, err_len, "zvec: out of memory%s", "");
      return -1;
    }
    zvec_ftype t = ZVEC_FT_UNKNOWN;
    char *colon = strchr(item, ':');
    if (colon != NULL) {
      *colon++ = '\0';
      if (strcmp(colon, "int") == 0)
        t = ZVEC_FT_INT;
      else if (strcmp(colon, "real") == 0)
        t = ZVEC_FT_REAL;
      else if (strcmp(colon, "text") == 0)
        t = ZVEC_FT_TEXT;
      else {
        set_err(err, err_len,
                "zvec: filter column type must be int, real or text (got '%s')",
                colon);
        free(item);
        return -1;
      }
    }
    if (!is_plain_ident(item)) {
      set_err(err, err_len,
              "zvec: filter columns must be plain identifiers (got '%s')",
              item);
      free(item);
      return -1;
    }
    for (int i = 0; i < out->n_filter; i++) {
      if (strcmp(out->filter_name[i], item) == 0) {
        set_err(err, err_len, "zvec: filter column '%s' listed twice", item);
        free(item);
        return -1;
      }
    }
    if (out->n_filter >= ZVEC_MAX_FILTER) {
      set_err(err, err_len, "zvec: at most 8 filter columns%s", "");
      free(item);
      return -1;
    }
    out->filter_name[out->n_filter] = item;
    out->filter_type[out->n_filter++] = t;
    if (*end == '\0') return 0;
    p = end + 1;
  }
}

int zvec_args_parse(int argc, const char *const *argv, zvec_args *out,
                    char *err, size_t err_len) {
  memset(out, 0, sizeof *out);
  out->metric = ZVEC_METRIC_TYPE_L2;
  out->elem = ZVEC_ELEM_F32;
  out->sync_query = 1;
  int sync_seen = 0;
  int dim_seen = 0, metric_seen = 0, type_seen = 0;

  if (argc < 4) {
    set_err(err, err_len, "zvec: missing arguments%s", "");
    return -1;
  }

  /* Legacy form: exactly one argument without '=' is a collection path. */
  if (argc == 4 && strchr(argv[3], '=') == NULL) {
    out->attach = 1;
    out->path = clean_value(argv[3], strlen(argv[3]));
    if (out->path == NULL || out->path[0] == '\0') {
      set_err(err, err_len, "zvec: empty collection path%s", "");
      zvec_args_free(out);
      return -1;
    }
    return 0;
  }

  for (int i = 3; i < argc; i++) {
    const char *eq = strchr(argv[i], '=');
    if (eq == NULL) {
      set_err(err, err_len, "zvec: expected key=value, got '%s'", argv[i]);
      goto fail;
    }
    const char *k = argv[i];
    size_t kn = (size_t)(eq - k);
    while (kn > 0 && isspace((unsigned char)*k)) {
      k++;
      kn--;
    }
    while (kn > 0 && isspace((unsigned char)k[kn - 1])) {
      kn--;
    }
    char *key = dup_n(k, kn);
    char *val = clean_value(eq + 1, strlen(eq + 1));
    if (key == NULL || val == NULL) {
      free(key);
      free(val);
      set_err(err, err_len, "zvec: out of memory%s", "");
      goto fail;
    }
    int rc = 0;
    if (strcmp(key, "base") == 0) {
      rc = set_str(&out->base, val, key, err, err_len);
      val = NULL;
    } else if (strcmp(key, "key") == 0) {
      rc = set_str(&out->key, val, key, err, err_len);
      val = NULL;
    } else if (strcmp(key, "column") == 0) {
      rc = set_str(&out->column, val, key, err, err_len);
      val = NULL;
    } else if (strcmp(key, "path") == 0) {
      rc = set_str(&out->path, val, key, err, err_len);
      val = NULL;
    } else if (strcmp(key, "dim") == 0) {
      char *end = NULL;
      long v = strtol(val, &end, 10);
      if (dim_seen || end == val || *end != '\0' || v < 1 || v > 65536) {
        set_err(err, err_len,
                "zvec: dim must be an integer in 1..65536 (got '%s')", val);
        rc = -1;
      } else {
        out->dim = (uint32_t)v;
        dim_seen = 1;
      }
    } else if (strcmp(key, "metric") == 0) {
      if (metric_seen) {
        set_err(err, err_len, "zvec: option '%s' given twice", key);
        rc = -1;
      } else if (strcmp(val, "l2") == 0)
        out->metric = ZVEC_METRIC_TYPE_L2;
      else if (strcmp(val, "ip") == 0)
        out->metric = ZVEC_METRIC_TYPE_IP;
      else if (strcmp(val, "cosine") == 0)
        out->metric = ZVEC_METRIC_TYPE_COSINE;
      else {
        set_err(err, err_len,
                "zvec: metric must be l2, ip or cosine (got '%s')", val);
        rc = -1;
      }
      metric_seen = 1;
    } else if (strcmp(key, "type") == 0) {
      if (type_seen) {
        set_err(err, err_len, "zvec: option '%s' given twice", key);
        rc = -1;
      } else if (strcmp(val, "float32") == 0)
        out->elem = ZVEC_ELEM_F32;
      else if (strcmp(val, "float16") == 0)
        out->elem = ZVEC_ELEM_F16;
      else if (strcmp(val, "float64") == 0)
        out->elem = ZVEC_ELEM_F64;
      else if (strcmp(val, "int8") == 0)
        out->elem = ZVEC_ELEM_I8;
      else {
        set_err(
            err, err_len,
            "zvec: type must be float32, float16, float64 or int8 (got '%s')",
            val);
        rc = -1;
      }
      type_seen = 1;
    } else if (strcmp(key, "sync") == 0) {
      if (sync_seen) {
        set_err(err, err_len, "zvec: option '%s' given twice", key);
        rc = -1;
      } else if (strcmp(val, "query") == 0)
        out->sync_query = 1;
      else if (strcmp(val, "manual") == 0)
        out->sync_query = 0;
      else {
        set_err(err, err_len, "zvec: sync must be query or manual (got '%s')",
                val);
        rc = -1;
      }
      sync_seen = 1;
    } else if (strcmp(key, "filter") == 0) {
      rc = parse_filter(val, out, err,
                        err_len); /* may repeat; quote lists: filter='a,b' */
    } else {
      set_err(err, err_len, "zvec: unknown option '%s'", key);
      rc = -1;
    }
    free(key);
    free(val);
    if (rc != 0) goto fail;
  }

  if (out->base == NULL || out->key == NULL || out->column == NULL ||
      !dim_seen) {
    set_err(err, err_len, "zvec: base, key, column and dim are required%s", "");
    goto fail;
  }
  if (!is_plain_ident(out->base) || !is_plain_ident(out->key) ||
      !is_plain_ident(out->column)) {
    set_err(err, err_len,
            "zvec: base, key and column must be plain identifiers%s", "");
    goto fail;
  }
  for (int i = 0; i < out->n_filter; i++) {
    if (strcmp(out->filter_name[i], out->key) == 0 ||
        strcmp(out->filter_name[i], out->column) == 0) {
      set_err(err, err_len,
              "zvec: filter column '%s' is the key or the vector column",
              out->filter_name[i]);
      goto fail;
    }
  }
  return 0;

fail:
  zvec_args_free(out);
  return -1;
}
