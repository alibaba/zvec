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

// SQLite loadable extension for zvec.
//
// Load with `.load ./zvec` (sqlite3 shell) or sqlite3_load_extension(). The
// default entry point SQLite derives from the file name `zvec.<ext>` is
// `sqlite3_zvec_init`, defined below.
//
// This file only talks to zvec through the public C API (zvec/c_api.h) and to
// SQLite through the extension function table (sqlite3ext.h), so the module
// never links against libsqlite3.

#include <limits.h>
#include <sqlite3ext.h>

#include "zvec/c_api.h"

SQLITE_EXTENSION_INIT1

#if defined(_WIN32) || defined(__CYGWIN__)
#define ZVEC_SQLITE_EXPORT __declspec(dllexport)
#else
#define ZVEC_SQLITE_EXPORT __attribute__((visibility("default")))
#endif

// Function flags. SQLITE_INNOCUOUS only exists in SQLite >= 3.31.
#ifdef SQLITE_INNOCUOUS
#define ZVEC_SQLITE_FN_FLAGS \
  (SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS)
#else
#define ZVEC_SQLITE_FN_FLAGS (SQLITE_UTF8 | SQLITE_DETERMINISTIC)
#endif

// zvec_version() -> TEXT
static void zvec_version_fn(sqlite3_context *ctx, int argc,
                            sqlite3_value **argv) {
  (void)argc;
  (void)argv;
  const char *version = zvec_get_version();
  // The C API documents the string as library-managed; copy it anyway so the
  // result stays valid regardless of how the library manages that storage.
  sqlite3_result_text(ctx, version ? version : "", -1, SQLITE_TRANSIENT);
}

static int int_arg(sqlite3_value *value, int *out) {
  if (sqlite3_value_type(value) != SQLITE_INTEGER) {
    return 0;
  }
  sqlite3_int64 n = sqlite3_value_int64(value);
  if (n < INT_MIN || n > INT_MAX) {
    return 0;
  }
  *out = (int)n;
  return 1;
}

// zvec_check_version(major, minor, patch) -> INTEGER (1 if the loaded zvec
// library is at least that version, else 0). NULL in, NULL out.
static void zvec_check_version_fn(sqlite3_context *ctx, int argc,
                                  sqlite3_value **argv) {
  int v[3] = {0, 0, 0};
  int i;
  if (argc != 3) {
    sqlite3_result_error(ctx, "zvec_check_version() takes 3 arguments", -1);
    return;
  }
  for (i = 0; i < 3; i++) {
    if (sqlite3_value_type(argv[i]) == SQLITE_NULL) {
      sqlite3_result_null(ctx);
      return;
    }
  }
  for (i = 0; i < 3; i++) {
    if (!int_arg(argv[i], &v[i])) {
      sqlite3_result_error(
          ctx, "zvec_check_version() expects three 32-bit integer arguments",
          -1);
      return;
    }
  }
  sqlite3_result_int(ctx, zvec_check_version(v[0], v[1], v[2]) ? 1 : 0);
}

ZVEC_SQLITE_EXPORT int sqlite3_zvec_init(sqlite3 *db, char **pzErrMsg,
                                         const sqlite3_api_routines *pApi) {
  int rc;
  SQLITE_EXTENSION_INIT2(pApi);
  (void)pzErrMsg;

  rc = sqlite3_create_function_v2(db, "zvec_version", 0, ZVEC_SQLITE_FN_FLAGS,
                                  0, zvec_version_fn, 0, 0, 0);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_function_v2(db, "zvec_check_version", 3,
                                  ZVEC_SQLITE_FN_FLAGS, 0,
                                  zvec_check_version_fn, 0, 0, 0);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // The virtual table module (CREATE VIRTUAL TABLE ... USING zvec) will be
  // registered here with sqlite3_create_module_v2().

#ifdef SQLITE_OK_LOAD_PERMANENTLY
  // Keep the library resident after the connection closes. zvec owns threads
  // and process-wide state, so unloading it under a live process is unsafe.
  return SQLITE_OK_LOAD_PERMANENTLY;
#else
  return SQLITE_OK;
#endif
}
