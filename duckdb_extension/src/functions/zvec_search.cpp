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

struct ZvecSearchBindData : public TableFunctionData {
  std::string path;
  std::string field_name;
  std::vector<float> query_vector;
  int topk;

  // Schema info for output columns
  struct FieldInfo {
    std::string name;
    zvec::DataType type;
  };
  std::vector<FieldInfo> output_fields;
};

struct ZvecSearchState : public GlobalTableFunctionState {
  zvec::DocPtrList results;
  idx_t current_idx = 0;
  bool done = false;
};

static unique_ptr<FunctionData> ZvecSearchBind(
    ClientContext &context, TableFunctionBindInput &input,
    vector<LogicalType> &return_types, vector<string> &names) {
  auto bind_data = make_uniq<ZvecSearchBindData>();
  bind_data->path = input.inputs[0].GetValue<string>();
  bind_data->field_name = input.inputs[1].GetValue<string>();

  // Extract FLOAT[] from the third argument
  auto &list_val = input.inputs[2];
  auto list_children = ListValue::GetChildren(list_val);
  for (auto &child : list_children) {
    bind_data->query_vector.push_back(child.GetValue<float>());
  }

  bind_data->topk = input.inputs[3].GetValue<int32_t>();

  // Open collection and read schema for output column definitions
  auto &registry = CollectionRegistry::Instance();
  auto collection = registry.GetOrOpen(bind_data->path);
  auto schema = UnwrapOrThrow(collection->Schema(), "get schema");

  // Output: pk VARCHAR, score FLOAT, then all forward fields
  names.push_back("pk");
  return_types.push_back(LogicalType::VARCHAR);

  names.push_back("score");
  return_types.push_back(LogicalType::FLOAT);

  for (auto &field : schema.forward_fields()) {
    bind_data->output_fields.push_back({field->name(), field->data_type()});
    names.push_back(field->name());
    return_types.push_back(ZvecTypeToDuckDB(field->data_type()));
  }

  return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> ZvecSearchInit(
    ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<ZvecSearchBindData>();
  auto state = make_uniq<ZvecSearchState>();

  auto &registry = CollectionRegistry::Instance();
  auto collection = registry.GetOrOpen(bind_data.path);

  zvec::VectorQuery query;
  query.topk_ = bind_data.topk;
  query.field_name_ = bind_data.field_name;
  query.query_vector_ = FloatArrayToQueryVector(bind_data.query_vector);

  state->results = UnwrapOrThrow(collection->Query(query), "vector search");
  return std::move(state);
}

static void ZvecSearchScan(ClientContext &context, TableFunctionInput &data,
                           DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<ZvecSearchBindData>();
  auto &state = data.global_state->Cast<ZvecSearchState>();

  if (state.done) {
    output.SetCardinality(0);
    return;
  }

  idx_t count = 0;
  idx_t max_count = STANDARD_VECTOR_SIZE;

  while (state.current_idx < state.results.size() && count < max_count) {
    auto &doc = state.results[state.current_idx];

    // pk
    FlatVector::GetData<string_t>(output.data[0])[count] =
        StringVector::AddString(output.data[0], doc->pk());

    // score
    FlatVector::GetData<float>(output.data[1])[count] = doc->score();

    // forward fields
    for (idx_t f = 0; f < bind_data.output_fields.size(); f++) {
      auto &field_info = bind_data.output_fields[f];
      DocFieldToDuckDB(*doc, field_info.name, field_info.type,
                       output.data[f + 2], count);
    }

    count++;
    state.current_idx++;
  }

  if (state.current_idx >= state.results.size()) {
    state.done = true;
  }
  output.SetCardinality(count);
}

void RegisterZvecSearch(DatabaseInstance &db) {
  TableFunction func(
      "zvec_search",
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
      ZvecSearchScan, ZvecSearchBind, ZvecSearchInit);
  ExtensionUtil::RegisterFunction(db, func);
}

}  // namespace duckdb
