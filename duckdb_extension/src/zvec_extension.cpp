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

#define DUCKDB_EXTENSION_MAIN

#include "zvec_extension.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb.hpp"

namespace duckdb {

// Forward declarations for function registrations
void RegisterZvecCreate(DatabaseInstance &db);
void RegisterZvecInsert(DatabaseInstance &db);
void RegisterZvecSearch(DatabaseInstance &db);
void RegisterZvecFetch(DatabaseInstance &db);

void ZvecExtension::Load(DuckDB &db) {
  RegisterZvecCreate(*db.instance);
  RegisterZvecInsert(*db.instance);
  RegisterZvecSearch(*db.instance);
  RegisterZvecFetch(*db.instance);
}

std::string ZvecExtension::Name() {
  return "zvec";
}

std::string ZvecExtension::Version() const {
#ifdef EXT_VERSION_ZVEC
  return EXT_VERSION_ZVEC;
#else
  return "0.1.0";
#endif
}

}  // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void zvec_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadExtension<duckdb::ZvecExtension>();
}

DUCKDB_EXTENSION_API const char *zvec_version() {
  return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
