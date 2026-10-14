#ifndef ZVEC_SQLITE_ARGS_H
#define ZVEC_SQLITE_ARGS_H

#include <stddef.h>
#include <stdint.h>

/* Element type of the vector BLOB stored in the base table. */
typedef enum zvec_elem {
  ZVEC_ELEM_F32 = 0,
  ZVEC_ELEM_F16,
  ZVEC_ELEM_F64,
  ZVEC_ELEM_I8
} zvec_elem;

/* Scalar columns mirrored into the index so searches can be filtered. */
#define ZVEC_MAX_FILTER 8
typedef enum zvec_ftype {
  ZVEC_FT_UNKNOWN =
      0, /* not resolved yet: taken from the base column's affinity */
  ZVEC_FT_INT,
  ZVEC_FT_REAL,
  ZVEC_FT_TEXT
} zvec_ftype;

typedef struct zvec_args {
  int attach; /* 1: legacy USING zvec('path') -- open an existing collection */
  char *path; /* attach: required; derived mode: optional */
  char *base; /* derived mode only from here down */
  char *key;
  char *column;
  uint32_t dim;
  int metric; /* ZVEC_METRIC_TYPE_* */
  zvec_elem elem;
  int n_filter; /* filter=a,b:real,... */
  char *filter_name[ZVEC_MAX_FILTER];
  zvec_ftype filter_type[ZVEC_MAX_FILTER];
  int sync_query; /* sync=query (default, 1) | manual (0) */
} zvec_args;

/* Parses the argv that SQLite hands to xCreate/xConnect (argv[0] is the module
 * name, [1] the schema, [2] the table name, [3..] the user arguments).
 * Returns 0 and fills *out, or -1 and writes a message to err. On success the
 * caller owns the strings and must call zvec_args_free(). */
int zvec_args_parse(int argc, const char *const *argv, zvec_args *out,
                    char *err, size_t err_len);
void zvec_args_free(zvec_args *a);

/* Bytes per element, and the BLOB length a vector of `dim` elements must have.
 */
size_t zvec_elem_size(zvec_elem e);

/* "int", "real" or "text". */
const char *zvec_ftype_name(zvec_ftype t);

#endif
