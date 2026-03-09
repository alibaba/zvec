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


#include <zvec/db/collection.h>
#include <zvec/db/doc.h>


namespace zvec {

/**
 * @brief Create a test schema with deterministic field definitions.
 *
 * @param name The collection name (default: "crash_recovery_test")
 * @return CollectionSchema::Ptr The test schema
 */
inline CollectionSchema::Ptr CreateTestSchema(
    const std::string &name = "crash_recovery_test") {
  auto schema = std::make_shared<CollectionSchema>(name);
  schema->set_max_doc_count_per_segment(2000);

  schema->add_field(
      std::make_shared<FieldSchema>("int32_field", DataType::INT32, false));
  schema->add_field(
      std::make_shared<FieldSchema>("int64_field", DataType::INT64, true));
  schema->add_field(
      std::make_shared<FieldSchema>("float_field", DataType::FLOAT, true));
  schema->add_field(
      std::make_shared<FieldSchema>("string_field", DataType::STRING, false));
  schema->add_field(
      std::make_shared<FieldSchema>("bool_field", DataType::BOOL, false));
  schema->add_field(std::make_shared<FieldSchema>("array_int32_field",
                                                  DataType::ARRAY_INT32, true));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_string_field", DataType::ARRAY_STRING, false));
  schema->add_field(std::make_shared<FieldSchema>(
      "dense_fp32_field", DataType::VECTOR_FP32, 128, false,
      std::make_shared<HnswIndexParams>(MetricType::COSINE)));
  schema->add_field(std::make_shared<FieldSchema>(
      "sparse_fp32_field", DataType::SPARSE_VECTOR_FP32, 0, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));

  return schema;
}


/**
 * @brief Create a test document with deterministic values based on doc_id.
 *
 * Document pattern:
 * - pk: "pk_{doc_id}"
 * - int32_field: doc_id (cast to int32)
 * - int64_field: doc_id, null if doc_id % 60 == 0
 * - float_field: doc_id / 1000.0, null if doc_id % 70 == 0
 * - string_field: "value_{doc_id}" or "updated_value_{doc_id}" if updated
 * - bool_field: doc_id % 2 == 0 or flipped if is_update
 * - array_int32_field: [doc_id, doc_id+1, doc_id+2], null if doc_id % 100 == 0
 * - array_string_field: ["str_0", ...] or ["updated_str_0", ...] if updated
 * - dense_fp32_field: vector where dense[i] = (doc_id + i) / 1000.0f
 * - sparse_fp32_field: sparse vector with indices [0, 10, ...]
 *
 * @param doc_id The document ID (determines all field values)
 * @param is_update If true, create updated version of the document
 * @return Doc The created document
 */
inline Doc CreateTestDoc(uint64_t doc_id, bool is_update = false) {
  Doc doc;

  // Set primary key
  std::string pk = "pk_" + std::to_string(doc_id);
  doc.set_pk(pk);

  // Set scalar fields
  doc.set<int32_t>("int32_field", static_cast<int32_t>(doc_id));

  // int64_field: nullable, null if doc_id % 60 == 0
  if (doc_id % 60 != 0) {
    doc.set<int64_t>("int64_field", static_cast<int64_t>(doc_id));
  }

  // float_field: nullable, null if doc_id % 70 == 0
  if (doc_id % 70 != 0) {
    doc.set<float>("float_field", static_cast<float>(doc_id) / 1000.0f);
  }

  // string_field: "value_{id}" or "updated_value_{id}"
  std::string string_value = is_update
                                 ? "updated_value_" + std::to_string(doc_id)
                                 : "value_" + std::to_string(doc_id);
  doc.set<std::string>("string_field", string_value);

  // bool_field: alternating based on doc_id, flipped if updated
  bool bool_value = (doc_id % 2 == 0);
  if (is_update) {
    bool_value = !bool_value;
  }
  doc.set<bool>("bool_field", bool_value);

  // array_int32_field: nullable, null if doc_id % 100 == 0
  if (doc_id % 100 != 0) {
    std::vector<int32_t> array_int32;
    for (int i = 0; i < 3; i++) {
      array_int32.push_back(static_cast<int32_t>(doc_id + i));
    }
    doc.set<std::vector<int32_t>>("array_int32_field", array_int32);
  }

  // array_string_field: ["str_0", "str_1", ...] or ["updated_str_0", ...]
  std::vector<std::string> array_string;
  size_t array_size = doc_id % 5 + 1;  // 1 to 5 elements
  for (size_t i = 0; i < array_size; i++) {
    if (is_update) {
      array_string.push_back("updated_str_" + std::to_string(i));
    } else {
      array_string.push_back("str_" + std::to_string(i));
    }
  }
  doc.set<std::vector<std::string>>("array_string_field", array_string);

  // dense_fp32_field: deterministic pattern
  std::vector<float> dense(128);
  for (int i = 0; i < 128; i++) {
    dense[i] = static_cast<float>(doc_id + i) / 1000.0f;
  }
  doc.set<std::vector<float>>("dense_fp32_field", dense);

  // sparse_fp32_field: sparse vector with indices [0, 10, 20, ..., 100]
  // Values based on doc_id: value = (doc_id + index) / 1000.0
  std::vector<uint32_t> sparse_indices;
  std::vector<float> sparse_values;
  for (uint32_t idx = 0; idx <= 100; idx += 10) {
    sparse_indices.push_back(idx);
    sparse_values.push_back(static_cast<float>(doc_id + idx) / 1000.0f);
  }
  doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
      "sparse_fp32_field", std::make_pair(sparse_indices, sparse_values));

  return doc;
}


}  // namespace zvec
