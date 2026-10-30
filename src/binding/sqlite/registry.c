#define _XOPEN_SOURCE 700
#include "registry.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static SRWLOCK g_lock = SRWLOCK_INIT;
#define LOCK() AcquireSRWLockExclusive(&g_lock);
#define UNLOCK() ReleaseSRWLockExclusive(&g_lock);
#else
#include <pthread.h>
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() pthread_mutex_lock(&g_lock)
#define UNLOCK() pthread_mutex_unlock(&g_lock)
#endif

#ifdef ZVEC_SQLITE_TRACE
#define ZTRACE(...)                                \
  do {                                             \
    fprintf(stderr, "[zvec_sqlite] " __VA_ARGS__); \
    fputc('\n', stderr);                           \
  } while (0)
#else
#define ZTRACE(...) ((void)0)
#endif

struct ZvecShared {
  char *path;             /* normalized key (heap copy) */
  zvec_collection_t *col; /* shared handle */
  int refcount;
  struct ZvecShared *next;
};

static struct ZvecShared *g_head = NULL;

static struct ZvecShared *find_locked(const char *key) {
  for (struct ZvecShared *e = g_head; e != NULL; e = e->next) {
    if (strcmp(e->path, key) == 0) return e;
  }
  return NULL;
}

static void set_err(char *err, size_t len, const char *msg) {
  if (err != NULL && len > 0) {
    snprintf(err, len, "%s", msg ? msg : "unknown error");
  }
}

static void set_err_from_zvec(char *err, size_t len) {
  zvec_error_details_t d = {0};
  if (zvec_get_last_error_details(&d) == ZVEC_OK)
    set_err(err, len, d.message);
  else
    set_err(err, len, "unknown error");
}

static zvec_error_code_t normalize_path(const char *path, char *out,
                                        size_t out_len) {
  if (path == NULL || path[0] == '\0' || out == NULL || out_len == 0)
    return ZVEC_ERROR_INVALID_ARGUMENT;

#ifdef _WIN32
  return _fullpath(out, path, out_len) ? ZVEC_OK : ZVEC_ERROR_NOT_FOUND;
#else
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

  char resolved[PATH_MAX];
  if (realpath(path, resolved) != NULL)
    return snprintf(out, out_len, "%s", resolved) < (int)out_len
               ? ZVEC_OK
               : ZVEC_ERROR_NOT_FOUND;

  char tmp[PATH_MAX];
  size_t len = strlen(path);
  if (len >= sizeof tmp) return ZVEC_ERROR_INVALID_ARGUMENT;
  memcpy(tmp, path, len + 1);
  while (len > 1 && tmp[len - 1] == '/') tmp[--len] = '\0';

  const char *base;
  const char *parent;
  char *slash = strrchr(tmp, '/');
  if (slash == NULL) {
    parent = ".";
    base = tmp;
  } else if (slash == tmp) {
    parent = "/";
    base = tmp + 1;
  } else {
    *slash = '\0';
    parent = tmp;
    base = slash + 1;
  }
  if (base[0] == '\0') return ZVEC_ERROR_NOT_FOUND;

  if (realpath(parent, resolved) == NULL) return ZVEC_ERROR_NOT_FOUND;
  const char *sep = strcmp(resolved, "/") == 0 ? "" : "/";
  int n = snprintf(out, out_len, "%s%s%s", resolved, sep, base);

  return (n < 0 || (size_t)n >= out_len) ? ZVEC_ERROR_NOT_FOUND : ZVEC_OK;
#endif
}

static zvec_error_code_t acquire(const char *path,
                                 const zvec_collection_schema_t *schema,
                                 zvec_shared **out, char *err, size_t err_len) {
  char key[PATH_MAX];
  zvec_collection_t *col = NULL;
  zvec_error_code_t rc;

  if (out == NULL) {
    set_err(err, err_len, "out parameter is NULL");
    return ZVEC_ERROR_NOT_FOUND;
  }
  *out = NULL;

  rc = normalize_path(path, key, sizeof key);
  if (rc != ZVEC_OK) {
    set_err(err, err_len, "error resolving path");
    return rc;
  }

  LOCK();

  zvec_shared *node = find_locked(key);

  if (node != NULL) {
    if (schema != NULL) {
      set_err(err, err_len, "collection is already open in this process");
      rc = ZVEC_ERROR_ALREADY_EXISTS;
    } else {
      node->refcount++;
      *out = node;
      rc = ZVEC_OK;
    }
    goto done;
  }

  rc = (schema != NULL)
           ? zvec_collection_create_and_open(key, schema, NULL, &col)
           : zvec_collection_open(key, NULL, &col);
  if (rc != ZVEC_OK) {
    set_err_from_zvec(err, err_len);
    goto done;
  }

  node = malloc(sizeof *node);
  if (node != NULL) {
    node->path = strdup(key);
  }

  if (node == NULL || node->path == NULL) {
    free(node);
    zvec_collection_close(col);
    set_err(err, err_len, "out of memory");
    rc = ZVEC_ERROR_RESOURCE_EXHAUSTED;
    goto done;
  }
  node->col = col;
  node->refcount = 1;
  node->next = g_head;
  g_head = node;
  *out = node;
  rc = ZVEC_OK;

done:
  UNLOCK();
  return rc;
}

void zvec_shared_release(zvec_shared *s) {
  if (s == NULL) return;

  LOCK();
  if (--s->refcount == 0) {
    zvec_shared **link = &g_head;
    while (*link != NULL && *link != s) {
      link = &(*link)->next;
    }
    if (*link == s) {
      *link = s->next;
      zvec_error_code_t rc = zvec_collection_close(s->col);
      if (rc != ZVEC_OK) {
        ZTRACE("close failed for %s: rc=%d", s->path, (int)rc);
      }
      free(s->path);
      free(s);
    }
  }
  UNLOCK();
}

zvec_error_code_t zvec_shared_open(const char *path, zvec_shared **out,
                                   char *err, size_t err_len) {
  return acquire(path, NULL, out, err, err_len);
}

zvec_error_code_t zvec_shared_create(const char *path,
                                     const zvec_collection_schema_t *schema,
                                     zvec_shared **out, char *err,
                                     size_t err_len) {
  if (schema == NULL) {
    set_err(err, err_len, "schema is required to create a collection");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  return acquire(path, schema, out, err, err_len);
}

zvec_collection_t *zvec_shared_collection(const zvec_shared *s) {
  return s != NULL ? s->col : NULL;
}

#ifdef NORMALIZE_TEST
int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    char out[PATH_MAX];
    int rc = normalize_path(argv[i], out, sizeof out);
    printf("%-24s -> rc=%d  %s\n", argv[i], rc, rc == 0 ? out : "");
  }
  return 0;
}
#endif
