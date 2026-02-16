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

#include "duckdb/main/extension_util.hpp"
#include "collection_registry.hpp"
#include "duckdb.hpp"
#include "schema_parser.hpp"
#include "type_conversion.hpp"

namespace duckdb {

static void ZvecCreateFunction(DataChunk &args, ExpressionState &state,
                               Vector &result) {
  auto &path_vec = args.data[0];
  auto &schema_vec = args.data[1];
  auto count = args.size();

  auto paths = FlatVector::GetData<string_t>(path_vec);
  auto schemas = FlatVector::GetData<string_t>(schema_vec);
  auto result_data = FlatVector::GetData<string_t>(result);

  for (idx_t i = 0; i < count; i++) {
    if (FlatVector::IsNull(path_vec, i) || FlatVector::IsNull(schema_vec, i)) {
      FlatVector::SetNull(result, i, true);
      continue;
    }
    auto path = paths[i].GetString();
    auto schema_json = schemas[i].GetString();

    auto schema = ParseSchemaJson(schema_json);
    auto &registry = CollectionRegistry::Instance();
    registry.Create(path, *schema);

    auto msg = "Created collection at " + path;
    result_data[i] = StringVector::AddString(result, msg);
  }
}

void RegisterZvecCreate(DatabaseInstance &db) {
  ScalarFunctionSet func_set("zvec_create");
  func_set.AddFunction(
      ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR},
                     LogicalType::VARCHAR, ZvecCreateFunction));
  ExtensionUtil::RegisterFunction(db, func_set);
}

}  // namespace duckdb
