#ifndef ZVEC_SQLITE_REGISTRY_H
#define ZVEC_SQLITE_REGISTRY_H

#include<stddef.h>
#include"zvec/c_api.h"

// One entry per open path
typedef struct zvec_shared zvec_shared;

/* Open an existing collection, or reuse the process' open handle
 * for the same normalized path. Returns ZVEC_OK and bumps the 
 * refcount to success. On failure writes a message into err 
 * (may be NULL) */
zvec_error_code_t zvec_shared_open(const char *path, 
    zvec_shared **out, 
    char *err, 
    size_t err_len);

/* Creates a new collection (path must exist) and registers it. */
zvec_error_code_t zvec_shared_create(const char *path, 
    const zvec_collection_schema_t *schema, 
    zvec_shared **out, 
    char *err, 
    size_t err_len);

/* The shared handle. Valid until the matching release. 
 * Not owned by caller. */
zvec_collection_t *zvec_shared_collection(const zvec_shared *s);

/* Drop one reference. At zero, the collection is 
 * closed and freed. */
void zvec_shared_release(zvec_shared *s);

#endif
