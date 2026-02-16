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

#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "collection_registry.hpp"
#include "duckdb.hpp"
#include "type_conversion.hpp"

namespace duckdb {

struct ZvecFetchBindData : public TableFunctionData {
  std::string path;
  std::string pk;

  struct FieldInfo {
    std::string name;
    zvec::DataType type;
  };
  std::vector<FieldInfo> output_fields;
};

struct ZvecFetchState : public GlobalTableFunctionState {
  zvec::DocPtrMap results;
  bool done = false;
};

static unique_ptr<FunctionData> ZvecFetchBind(ClientContext &context,
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<string> &names) {
  auto bind_data = make_uniq<ZvecFetchBindData>();
  bind_data->path = input.inputs[0].GetValue<string>();
  bind_data->pk = input.inputs[1].GetValue<string>();

  auto &registry = CollectionRegistry::Instance();
  auto collection = registry.GetOrOpen(bind_data->path);
  auto schema = UnwrapOrThrow(collection->Schema(), "get schema");

  // Output: pk VARCHAR, then all forward fields
  names.push_back("pk");
  return_types.push_back(LogicalType::VARCHAR);

  for (auto &field : schema.forward_fields()) {
    bind_data->output_fields.push_back({field->name(), field->data_type()});
    names.push_back(field->name());
    return_types.push_back(ZvecTypeToDuckDB(field->data_type()));
  }

  return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> ZvecFetchInit(
    ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<ZvecFetchBindData>();
  auto state = make_uniq<ZvecFetchState>();

  auto &registry = CollectionRegistry::Instance();
  auto collection = registry.GetOrOpen(bind_data.path);

  state->results = UnwrapOrThrow(collection->Fetch({bind_data.pk}), "fetch");
  return std::move(state);
}

static void ZvecFetchScan(ClientContext &context, TableFunctionInput &data,
                          DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<ZvecFetchBindData>();
  auto &state = data.global_state->Cast<ZvecFetchState>();

  if (state.done) {
    output.SetCardinality(0);
    return;
  }

  idx_t count = 0;
  for (auto &[pk_key, doc_ptr] : state.results) {
    if (!doc_ptr) {
      continue;
    }

    // pk
    FlatVector::GetData<string_t>(output.data[0])[count] =
        StringVector::AddString(output.data[0], doc_ptr->pk());

    // forward fields
    for (idx_t f = 0; f < bind_data.output_fields.size(); f++) {
      auto &field_info = bind_data.output_fields[f];
      DocFieldToDuckDB(*doc_ptr, field_info.name, field_info.type,
                       output.data[f + 1], count);
    }
    count++;
  }

  state.done = true;
  output.SetCardinality(count);
}

void RegisterZvecFetch(DatabaseInstance &db) {
  TableFunction func("zvec_fetch", {LogicalType::VARCHAR, LogicalType::VARCHAR},
                     ZvecFetchScan, ZvecFetchBind, ZvecFetchInit);
  ExtensionUtil::RegisterFunction(db, func);
}

}  // namespace duckdb
