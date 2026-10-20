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

#pragma once

#include <string>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>

namespace zvec::checkpoint_test {

constexpr int kInitialDocs = 64;
constexpr int kSegmentDocs = 1000;

inline std::string PrimaryKey(int id) {
  return "pk_" + std::to_string(id);
}

inline CollectionSchema MakeSchema(bool fts) {
  CollectionSchema schema("checkpoint_recovery");
  schema.set_max_doc_count_per_segment(kSegmentDocs);
  schema.add_field(
      std::make_shared<FieldSchema>("generation", DataType::INT32, false,
                                    std::make_shared<InvertIndexParams>()));
  for (const auto *name : {"text", "title"}) {
    schema.add_field(std::make_shared<FieldSchema>(
        name, DataType::STRING, false,
        fts ? std::make_shared<FtsIndexParams>("whitespace") : nullptr));
  }
  schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, 4, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  return schema;
}

inline Doc MakeDoc(int id, int generation) {
  Doc doc;
  doc.set_pk(PrimaryKey(id));
  doc.set<int32_t>("generation", generation);
  for (const auto *field : {"text", "title"}) {
    std::string value =
        std::string(field) + "version" + std::to_string(generation);
    const int frequency =
        std::string(field) == "text" ? id % 5 + 1 : 5 - id % 5;
    for (int i = 0; i < frequency; ++i)
      value += " " + std::string(field) + "rank";
    for (int i = 0; i < id % 3; ++i) value += " padding";
    doc.set<std::string>(field, value);
  }
  doc.set<std::vector<float>>("vec",
                              {float(id), float(generation), 1.0f, 0.0f});
  return doc;
}

}  // namespace zvec::checkpoint_test
