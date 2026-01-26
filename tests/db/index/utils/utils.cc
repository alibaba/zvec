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

#include "utils.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "ailego/logger/logger.h"
#include "zvec/db/collection.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"
#include "zvec/db/schema.h"
#include "zvec/db/status.h"
#include "zvec/db/type.h"

using namespace zvec;
using namespace zvec::test;

CollectionSchema::Ptr TestHelper::CreateTempSchema() {
  auto schema = std::make_shared<CollectionSchema>("demo");
  schema->set_max_doc_count_per_segment(1000);

  schema->add_field(std::make_shared<FieldSchema>(
      "id", DataType::INT64, false, std::make_shared<InvertIndexParams>(true)));
  schema->add_field(std::make_shared<FieldSchema>(
      "name", DataType::STRING, false,
      std::make_shared<InvertIndexParams>(false)));
  schema->add_field(
      std::make_shared<FieldSchema>("weight", DataType::FLOAT, true));

  schema->add_field(std::make_shared<FieldSchema>(
      "dense", DataType::VECTOR_FP32, 128, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));
  schema->add_field(std::make_shared<FieldSchema>(
      "sparse", DataType::SPARSE_VECTOR_FP32, 0, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));
  return schema;
}

CollectionSchema::Ptr TestHelper::CreateScalarSchema() {
  auto schema = std::make_shared<CollectionSchema>("demo");

  // scalar
  schema->add_field(std::make_shared<FieldSchema>("int32", DataType::INT32));
  schema->add_field(std::make_shared<FieldSchema>("string", DataType::STRING));

  return schema;
}

// Helper function
CollectionSchema::Ptr TestHelper::CreateNormalSchema(
    bool nullable, std::string name, IndexParams::Ptr scalar_index_params,
    IndexParams::Ptr vector_index_params, uint64_t max_doc_count) {
  auto schema = std::make_shared<CollectionSchema>(name);
  schema->set_max_doc_count_per_segment(max_doc_count);

  // scalar
  schema->add_field(std::make_shared<FieldSchema>(
      "int32", DataType::INT32, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "string", DataType::STRING, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "uint32", DataType::UINT32, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "bool", DataType::BOOL, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "float", DataType::FLOAT, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "double", DataType::DOUBLE, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "int64", DataType::INT64, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "uint64", DataType::UINT64, nullable, scalar_index_params));

  // array
  schema->add_field(std::make_shared<FieldSchema>(
      "array_int32", DataType::ARRAY_INT32, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_string", DataType::ARRAY_STRING, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_uint32", DataType::ARRAY_UINT32, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_bool", DataType::ARRAY_BOOL, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_float", DataType::ARRAY_FLOAT, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_double", DataType::ARRAY_DOUBLE, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_int64", DataType::ARRAY_INT64, nullable, scalar_index_params));
  schema->add_field(std::make_shared<FieldSchema>(
      "array_uint64", DataType::ARRAY_UINT64, nullable, scalar_index_params));

  schema->add_field(std::make_shared<FieldSchema>(
      "dense_fp32", DataType::VECTOR_FP32, 128, false,
      vector_index_params ? vector_index_params
                          : std::make_shared<FlatIndexParams>(MetricType::IP)));
  schema->add_field(std::make_shared<FieldSchema>(
      "dense_fp16", DataType::VECTOR_FP16, 128, false,
      std::make_shared<FlatIndexParams>(MetricType::IP)));
  schema->add_field(std::make_shared<FieldSchema>(
      "dense_int8", DataType::VECTOR_INT8, 128, false,
      std::make_shared<FlatIndexParams>(MetricType::IP)));

  IndexParams::Ptr sparse_index_params;
  if (vector_index_params) {
    sparse_index_params = vector_index_params->clone();
    auto v = std::dynamic_pointer_cast<VectorIndexParams>(sparse_index_params);
    // sparse always use IP
    v->set_metric_type(MetricType::IP);
  }
  schema->add_field(std::make_shared<FieldSchema>(
      "sparse_fp32", DataType::SPARSE_VECTOR_FP32, 128, false,
      sparse_index_params ? sparse_index_params
                          : std::make_shared<FlatIndexParams>(MetricType::IP)));
  schema->add_field(std::make_shared<FieldSchema>(
      "sparse_fp16", DataType::SPARSE_VECTOR_FP16, 128, false,
      std::make_shared<FlatIndexParams>(MetricType::IP)));

  return schema;
}

CollectionSchema::Ptr TestHelper::CreateSchemaWithScalarIndex(
    bool nullable, bool enable_optimize, std::string name) {
  return CreateNormalSchema(
      nullable, name, std::make_shared<InvertIndexParams>(enable_optimize));
}

CollectionSchema::Ptr TestHelper::CreateSchemaWithVectorIndex(
    bool nullable, std::string name, IndexParams::Ptr vector_index_params) {
  return CreateNormalSchema(
      nullable, name, nullptr,
      vector_index_params ? vector_index_params
                          : std::make_shared<HnswIndexParams>(MetricType::IP));
}

CollectionSchema::Ptr TestHelper::CreateSchemaWithMaxDocCount(
    uint64_t doc_count) {
  return CreateNormalSchema(false, "demo", nullptr, nullptr, doc_count);
}

std::string TestHelper::MakePK(const uint64_t doc_id) {
  return "pk_" + std::to_string(doc_id);
}

uint64_t TestHelper::ExtractDocId(const std::string &pk) {
  return std::stoull(pk.substr(3));
}

Doc TestHelper::CreateDoc(const uint64_t doc_id, const CollectionSchema &schema,
                          std::string pk) {
  Doc new_doc;
  if (pk.empty()) {
    pk = MakePK(doc_id);
  }
  new_doc.set_pk(pk);

  for (auto &field : schema.fields()) {
    switch (field->data_type()) {
      case DataType::BINARY: {
        std::string binary_str("binary_" + std::to_string(doc_id));
        new_doc.set<std::string>(field->name(), binary_str);
        break;
      }
      case DataType::BOOL:
        new_doc.set<bool>(field->name(), doc_id % 10 == 0);
        break;
      case DataType::INT32:
        new_doc.set<int32_t>(field->name(), (int32_t)doc_id);
        break;
      case DataType::INT64:
        new_doc.set<int64_t>(field->name(), (int64_t)doc_id);
        break;
      case DataType::UINT32:
        new_doc.set<uint32_t>(field->name(), (uint32_t)doc_id);
        break;
      case DataType::UINT64:
        new_doc.set<uint64_t>(field->name(), (uint64_t)doc_id);
        break;
      case DataType::FLOAT:
        new_doc.set<float>(field->name(), (float)doc_id);
        break;
      case DataType::DOUBLE:
        new_doc.set<double>(field->name(), (double)doc_id);
        break;
      case DataType::STRING:
        new_doc.set<std::string>(field->name(),
                                 "value_" + std::to_string(doc_id));
        break;
      case DataType::ARRAY_BINARY: {
        std::vector<std::string> bin_vec;
        for (size_t i = 0; i < (doc_id % 10); i++) {
          bin_vec.push_back("bin_" + std::to_string(i));
        }
        new_doc.set<std::vector<std::string>>(field->name(), bin_vec);
        break;
      }
      case DataType::ARRAY_BOOL:
        new_doc.set<std::vector<bool>>(field->name(),
                                       std::vector<bool>(10, doc_id % 10 == 0));
        break;
      case DataType::ARRAY_INT32:
        new_doc.set<std::vector<int32_t>>(
            field->name(), std::vector<int32_t>(10, (int32_t)doc_id));
        break;
      case DataType::ARRAY_INT64:
        new_doc.set<std::vector<int64_t>>(
            field->name(), std::vector<int64_t>(10, (int64_t)doc_id));
        break;
      case DataType::ARRAY_UINT32:
        new_doc.set<std::vector<uint32_t>>(
            field->name(), std::vector<uint32_t>(10, (uint32_t)doc_id));
        break;
      case DataType::ARRAY_UINT64:
        new_doc.set<std::vector<uint64_t>>(
            field->name(), std::vector<uint64_t>(10, (uint64_t)doc_id));
        break;
      case DataType::ARRAY_FLOAT:
        new_doc.set<std::vector<float>>(field->name(),
                                        std::vector<float>(10, (float)doc_id));
        break;
      case DataType::ARRAY_DOUBLE:
        new_doc.set<std::vector<double>>(
            field->name(), std::vector<double>(10, (double)doc_id));
        break;
      case DataType::ARRAY_STRING:
        new_doc.set<std::vector<std::string>>(
            field->name(),
            std::vector<std::string>(10, "value_" + std::to_string(doc_id)));
        break;
      case DataType::VECTOR_BINARY32:
        new_doc.set<std::vector<uint32_t>>(
            field->name(),
            std::vector<uint32_t>(field->dimension(), uint32_t(doc_id + 0.1)));
        break;
      case DataType::VECTOR_BINARY64:
        new_doc.set<std::vector<uint64_t>>(
            field->name(),
            std::vector<uint64_t>(field->dimension(), uint64_t(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP32:
        new_doc.set<std::vector<float>>(
            field->name(),
            std::vector<float>(field->dimension(), float(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP64:
        new_doc.set<std::vector<double>>(
            field->name(),
            std::vector<double>(field->dimension(), double(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP16:
        new_doc.set<std::vector<float16_t>>(
            field->name(), std::vector<float16_t>(
                               field->dimension(),
                               static_cast<float16_t>(float(doc_id + 0.1))));
        break;
      case DataType::VECTOR_INT8:
        new_doc.set<std::vector<int8_t>>(
            field->name(),
            std::vector<int8_t>(field->dimension(), (int8_t)doc_id));
        break;
      case DataType::VECTOR_INT16:
        new_doc.set<std::vector<int16_t>>(
            field->name(),
            std::vector<int16_t>(field->dimension(), (int16_t)doc_id));
        break;
      case DataType::SPARSE_VECTOR_FP16: {
        std::vector<uint32_t> indices;
        std::vector<float16_t> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float16_t(float(doc_id + 0.1)));
        }
        std::pair<std::vector<uint32_t>, std::vector<float16_t>>
            sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
            field->name(), sparse_float_vec);
        break;
      }
      case DataType::SPARSE_VECTOR_FP32: {
        std::vector<uint32_t> indices;
        std::vector<float> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float(doc_id + 0.1));
        }
        std::pair<std::vector<uint32_t>, std::vector<float>> sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            field->name(), sparse_float_vec);
        break;
      }
      default:
        std::cout << "Unsupported data type: " << field->name() << std::endl;
        throw std::runtime_error("Unsupported vector data type");
    }
  }

  return new_doc;
}

Doc TestHelper::CreateDocNull(const uint64_t doc_id,
                              const CollectionSchema &schema, std::string pk) {
  Doc new_doc;
  if (pk.empty()) {
    pk = "pk_" + std::to_string(doc_id);
  }
  new_doc.set_pk(pk);

  for (auto &field : schema.fields()) {
    switch (field->data_type()) {
      case DataType::BINARY:
      case DataType::BOOL:
      case DataType::INT32:
      case DataType::INT64:
      case DataType::UINT32:
      case DataType::UINT64:
      case DataType::FLOAT:
      case DataType::DOUBLE:
      case DataType::STRING:
      case DataType::ARRAY_BINARY:
      case DataType::ARRAY_BOOL:
      case DataType::ARRAY_INT32:
      case DataType::ARRAY_INT64:
      case DataType::ARRAY_UINT32:
      case DataType::ARRAY_UINT64:
      case DataType::ARRAY_FLOAT:
      case DataType::ARRAY_DOUBLE:
      case DataType::ARRAY_STRING:
        break;
      case DataType::VECTOR_FP32:
        new_doc.set<std::vector<float>>(
            field->name(),
            std::vector<float>(field->dimension(), float(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP64:
        new_doc.set<std::vector<double>>(
            field->name(),
            std::vector<double>(field->dimension(), double(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP16:
        new_doc.set<std::vector<float16_t>>(
            field->name(), std::vector<float16_t>(
                               field->dimension(),
                               static_cast<float16_t>(float(doc_id + 0.1))));
        break;
      case DataType::VECTOR_INT8:
        new_doc.set<std::vector<int8_t>>(
            field->name(),
            std::vector<int8_t>(field->dimension(), (int8_t)doc_id));
        break;
      case DataType::VECTOR_INT16:
        new_doc.set<std::vector<int16_t>>(
            field->name(),
            std::vector<int16_t>(field->dimension(), (int16_t)doc_id));
        break;
      case DataType::SPARSE_VECTOR_FP16: {
        std::vector<uint32_t> indices;
        std::vector<float16_t> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float16_t(float(doc_id + 0.1)));
        }
        std::pair<std::vector<uint32_t>, std::vector<float16_t>>
            sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
            field->name(), sparse_float_vec);
        break;
      }
      case DataType::SPARSE_VECTOR_FP32: {
        std::vector<uint32_t> indices;
        std::vector<float> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float(doc_id + 0.1));
        }
        std::pair<std::vector<uint32_t>, std::vector<float>> sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            field->name(), sparse_float_vec);
        break;
      }
      default:
        throw std::runtime_error("Unsupported vector data type");
    }
  }

  return new_doc;
}

Status TestHelper::SegmentInsertDoc(const Segment::Ptr &segment,
                                    const CollectionSchema &schema,
                                    const uint64_t start_doc_id,
                                    const uint64_t end_doc_id, bool nullable,
                                    bool upsert, bool batch) {
  for (auto doc_id = start_doc_id; doc_id < end_doc_id; doc_id++) {
    if (segment) {
      Doc new_doc;
      if (nullable) {
        new_doc = CreateDocNull(doc_id, schema);
      } else {
        new_doc = CreateDoc(doc_id, schema);
      }

      Status s;
      if (upsert) {
        s = segment->Upsert(new_doc);
        CHECK_RETURN_STATUS(s);
      } else {
        s = segment->Insert(new_doc);
        CHECK_RETURN_STATUS(s);
      }
    }
  }
  return Status::OK();
}

Status TestHelper::CollectionInsertDoc(const Collection::Ptr &collection,
                                       const uint64_t start_doc_id,
                                       const uint64_t end_doc_id, bool nullable,
                                       bool upsert, bool batch) {
  if (!collection) {
    return Status::InvalidArgument("collection is nullptr");
  }
  auto schema = collection->Schema().value();
  auto make_doc = [&](uint64_t doc_id) -> Doc {
    return nullable ? CreateDocNull(doc_id, schema) : CreateDoc(doc_id, schema);
  };
  auto exec_write = [&](std::vector<Doc> &docs) -> Status {
    Result<WriteResults> result =
        upsert ? collection->Upsert(docs) : collection->Insert(docs);

    if (!result.has_value()) {
      LOG_ERROR("Failed to %s docs (count=%zu), error: %s.",
                upsert ? "upsert" : "insert", docs.size(),
                result.error().message().c_str());
      return result.error();
    }

    const auto &write_results = result.value();
    if (write_results.empty()) {
      return Status::InternalError("WriteResults is unexpectedly empty");
    }

    for (const auto &wr : write_results) {
      if (!wr.ok()) {
        return wr;
      }
    }
    return Status::OK();
  };

  if (batch) {
    std::vector<Doc> docs;
    docs.reserve(end_doc_id - start_doc_id);
    for (uint64_t doc_id = start_doc_id; doc_id < end_doc_id; ++doc_id) {
      docs.emplace_back(make_doc(doc_id));
    }
    return exec_write(docs);
  } else {
    std::vector<Doc> single_doc;
    single_doc.reserve(1);  // 可选优化

    for (uint64_t doc_id = start_doc_id; doc_id < end_doc_id; ++doc_id) {
      single_doc.clear();
      single_doc.push_back(make_doc(doc_id));
      Status s = exec_write(single_doc);
      if (!s.ok()) {
        LOG_ERROR("Failed at doc_id=%" PRIu64 ", doc: %s", doc_id,
                  single_doc[0].to_detail_string().c_str());
        return s;
      }
    }
  }
  return Status::OK();
}

Status TestHelper::CollectionUpsertDoc(const Collection::Ptr &collection,
                                       const uint64_t start_doc_id,
                                       const uint64_t end_doc_id, bool nullable,
                                       bool batch) {
  return CollectionInsertDoc(collection, start_doc_id, end_doc_id, nullable,
                             true, batch);
}

Segment::Ptr TestHelper::CreateSegmentWithDoc(
    const std::string &col_path, const CollectionSchema &schema,
    SegmentID segment_id, uint64_t min_doc_id, const IDMap::Ptr &id_map,
    const DeleteStore::Ptr &delete_store,
    const VersionManager::Ptr &version_manager, const SegmentOptions &options,
    uint64_t start_doc_id, uint32_t doc_count, bool nullable, bool upsert) {
  auto result =
      Segment::CreateAndOpen(col_path, schema, segment_id, min_doc_id, id_map,
                             delete_store, version_manager, options);

  if (!result.has_value()) {
    return nullptr;
  }

  auto segment = std::move(result).value();

  auto s = SegmentInsertDoc(segment, schema, start_doc_id,
                            start_doc_id + doc_count, nullable, upsert);
  if (!s.ok()) {
    LOG_ERROR("Failed to insert doc, err: %s", s.message().c_str());
    return nullptr;
  }

  return segment;
}

Collection::Ptr TestHelper::CreateCollectionWithDoc(
    const std::string &path, const CollectionSchema &schema,
    const CollectionOptions &options, uint64_t start_doc_id, uint32_t doc_count,
    bool nullable, bool upsert) {
  auto result = Collection::CreateAndOpen(path, schema, options);

  if (!result.has_value()) {
    LOG_ERROR("Failed to create collection, err: %s",
              result.error().message().c_str());
    return nullptr;
  }

  auto collection = std::move(result).value();

  auto s = CollectionInsertDoc(collection, start_doc_id,
                               start_doc_id + doc_count, nullable, upsert);
  if (!s.ok()) {
    LOG_ERROR("Failed to insert doc, err: %s", s.message().c_str());
    return nullptr;
  }

  return collection;
}

arrow::Status TestHelper::WriteTestFile(const std::string &filepath,
                                        FileFormat format,
                                        uint32_t start_doc_id,
                                        uint32_t end_doc_id,
                                        uint32_t batch_size) {
  // Define schema with additional list types
  auto schema = arrow::schema(
      {arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
       arrow::field(USER_ID, arrow::utf8()), arrow::field("id", arrow::int32()),
       arrow::field("name", arrow::utf8()),
       arrow::field("score", arrow::float64()),
       arrow::field("list_binary", arrow::list(arrow::binary())),
       arrow::field("list_utf8", arrow::list(arrow::utf8())),
       arrow::field("list_boolean", arrow::list(arrow::boolean())),
       arrow::field("list_int32", arrow::list(arrow::int32())),
       arrow::field("list_int64", arrow::list(arrow::int64())),
       arrow::field("list_uint32", arrow::list(arrow::uint32())),
       arrow::field("list_uint64", arrow::list(arrow::uint64())),
       arrow::field("list_float32", arrow::list(arrow::float32())),
       arrow::field("list_float64", arrow::list(arrow::float64()))});

  // Create builders
  auto g_doc_id_builder = std::make_shared<arrow::UInt64Builder>();
  auto uid_builder = std::make_shared<arrow::StringBuilder>();
  auto id_builder = std::make_shared<arrow::Int32Builder>();
  auto name_builder = std::make_shared<arrow::StringBuilder>();
  auto score_builder = std::make_shared<arrow::DoubleBuilder>();

  // Array field builders
  auto list_binary_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::BinaryBuilder>());
  auto list_utf8_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::StringBuilder>());
  auto list_boolean_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::BooleanBuilder>());
  auto list_int32_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::Int32Builder>());
  auto list_int64_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>());
  auto list_uint32_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::UInt32Builder>());
  auto list_uint64_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::UInt64Builder>());
  auto list_float32_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto list_float64_builder = std::make_shared<arrow::ListBuilder>(
      arrow::default_memory_pool(), std::make_shared<arrow::DoubleBuilder>());

  // Cast child builders for easier access
  auto binary_builder =
      static_cast<arrow::BinaryBuilder *>(list_binary_builder->value_builder());
  auto utf8_child_builder =
      static_cast<arrow::StringBuilder *>(list_utf8_builder->value_builder());
  auto boolean_child_builder = static_cast<arrow::BooleanBuilder *>(
      list_boolean_builder->value_builder());
  auto int32_child_builder =
      static_cast<arrow::Int32Builder *>(list_int32_builder->value_builder());
  auto int64_child_builder =
      static_cast<arrow::Int64Builder *>(list_int64_builder->value_builder());
  auto uint32_child_builder =
      static_cast<arrow::UInt32Builder *>(list_uint32_builder->value_builder());
  auto uint64_child_builder =
      static_cast<arrow::UInt64Builder *>(list_uint64_builder->value_builder());
  auto float32_child_builder =
      static_cast<arrow::FloatBuilder *>(list_float32_builder->value_builder());
  auto float64_child_builder = static_cast<arrow::DoubleBuilder *>(
      list_float64_builder->value_builder());

  // Fill data
  for (uint32_t i = start_doc_id; i < end_doc_id; ++i) {
    ARROW_RETURN_NOT_OK(g_doc_id_builder->Append(i + 1));
    ARROW_RETURN_NOT_OK(uid_builder->Append("user_" + std::to_string(i + 1)));
    ARROW_RETURN_NOT_OK(id_builder->Append(i + 1));
    ARROW_RETURN_NOT_OK(name_builder->Append("Name" + std::to_string(i)));
    ARROW_RETURN_NOT_OK(score_builder->Append(80.0 + i));

    const int dim = 128;
    // Append list_binary data
    ARROW_RETURN_NOT_OK(list_binary_builder->Append());
    for (int j = 0; j < dim; ++j) {
      std::string binary_data =
          "binary_" + std::to_string(i) + "_" + std::to_string(j);
      ARROW_RETURN_NOT_OK(binary_builder->Append(binary_data));
    }

    // Append list_utf8 data
    ARROW_RETURN_NOT_OK(list_utf8_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(utf8_child_builder->Append(
          "string_" + std::to_string(i) + "_" + std::to_string(j)));
    }

    // Append list_boolean data
    ARROW_RETURN_NOT_OK(list_boolean_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(boolean_child_builder->Append((i + j) % 2 == 0));
    }

    // Append list_int32 data
    ARROW_RETURN_NOT_OK(list_int32_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(int32_child_builder->Append(i * 10 + j));
    }

    // Append list_int64 data
    ARROW_RETURN_NOT_OK(list_int64_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(
          int64_child_builder->Append(static_cast<int64_t>(i) * 100 + j));
    }

    // Append list_uint32 data
    ARROW_RETURN_NOT_OK(list_uint32_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(
          uint32_child_builder->Append(static_cast<uint32_t>(i) * 10 + j));
    }

    // Append list_uint64 data
    ARROW_RETURN_NOT_OK(list_uint64_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(
          uint64_child_builder->Append(static_cast<uint64_t>(i) * 100 + j));
    }

    // Append list_float32 data
    ARROW_RETURN_NOT_OK(list_float32_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(
          float32_child_builder->Append(static_cast<float>(i) + j * 0.1f));
    }

    // Append list_float64 data
    ARROW_RETURN_NOT_OK(list_float64_builder->Append());
    for (int j = 0; j < dim; ++j) {
      ARROW_RETURN_NOT_OK(
          float64_child_builder->Append(static_cast<double>(i) + j * 0.01));
    }
  }

  // Construct arrays
  std::shared_ptr<arrow::Array> g_doc_id_array, uid_array, id_array, name_array,
      score_array, list_binary_array, list_utf8_array, list_boolean_array,
      list_int32_array, list_int64_array, list_uint32_array, list_uint64_array,
      list_float32_array, list_float64_array;

  ARROW_RETURN_NOT_OK(g_doc_id_builder->Finish(&g_doc_id_array));
  ARROW_RETURN_NOT_OK(uid_builder->Finish(&uid_array));
  ARROW_RETURN_NOT_OK(id_builder->Finish(&id_array));
  ARROW_RETURN_NOT_OK(name_builder->Finish(&name_array));
  ARROW_RETURN_NOT_OK(score_builder->Finish(&score_array));
  ARROW_RETURN_NOT_OK(list_binary_builder->Finish(&list_binary_array));
  ARROW_RETURN_NOT_OK(list_utf8_builder->Finish(&list_utf8_array));
  ARROW_RETURN_NOT_OK(list_boolean_builder->Finish(&list_boolean_array));
  ARROW_RETURN_NOT_OK(list_int32_builder->Finish(&list_int32_array));
  ARROW_RETURN_NOT_OK(list_int64_builder->Finish(&list_int64_array));
  ARROW_RETURN_NOT_OK(list_uint32_builder->Finish(&list_uint32_array));
  ARROW_RETURN_NOT_OK(list_uint64_builder->Finish(&list_uint64_array));
  ARROW_RETURN_NOT_OK(list_float32_builder->Finish(&list_float32_array));
  ARROW_RETURN_NOT_OK(list_float64_builder->Finish(&list_float64_array));

  // Set rows per batch
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;

  // Split data into multiple batches
  auto doc_count = (int)(end_doc_id - start_doc_id);
  for (int start = 0; start < doc_count; start += batch_size) {
    int current_batch_size = std::min((int)batch_size, doc_count - start);

    auto g_doc_id_slice = g_doc_id_array->Slice(start, current_batch_size);
    auto uid_slice = uid_array->Slice(start, current_batch_size);
    auto id_slice = id_array->Slice(start, current_batch_size);
    auto name_slice = name_array->Slice(start, current_batch_size);
    auto score_slice = score_array->Slice(start, current_batch_size);
    auto list_binary_slice =
        list_binary_array->Slice(start, current_batch_size);
    auto list_utf8_slice = list_utf8_array->Slice(start, current_batch_size);
    auto list_boolean_slice =
        list_boolean_array->Slice(start, current_batch_size);
    auto list_int32_slice = list_int32_array->Slice(start, current_batch_size);
    auto list_int64_slice = list_int64_array->Slice(start, current_batch_size);
    auto list_uint32_slice =
        list_uint32_array->Slice(start, current_batch_size);
    auto list_uint64_slice =
        list_uint64_array->Slice(start, current_batch_size);
    auto list_float32_slice =
        list_float32_array->Slice(start, current_batch_size);
    auto list_float64_slice =
        list_float64_array->Slice(start, current_batch_size);

    auto batch = arrow::RecordBatch::Make(
        schema, current_batch_size,
        {g_doc_id_slice, uid_slice, id_slice, name_slice, score_slice,
         list_binary_slice, list_utf8_slice, list_boolean_slice,
         list_int32_slice, list_int64_slice, list_uint32_slice,
         list_uint64_slice, list_float32_slice, list_float64_slice});
    batches.push_back(batch);
  }

  // Open output stream
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(filepath));

  if (format == FileFormat::PARQUET) {
    // Parquet write logic - create table with multiple record batches
    auto table = arrow::Table::Make(
        schema, {g_doc_id_array, uid_array, id_array, name_array, score_array,
                 list_binary_array, list_utf8_array, list_boolean_array,
                 list_int32_array, list_int64_array, list_uint32_array,
                 list_uint64_array, list_float32_array, list_float64_array});

    parquet::WriterProperties::Builder builder;
    builder.data_pagesize(1024);
    // 3 rows per row group
    builder.max_row_group_length(batch_size);
    auto props = builder.build();

    auto status = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out, batch_size, props);
    if (!status.ok()) {
      std::cerr << "Write failed: " << status.ToString() << std::endl;
      return status;
    }

    std::cout << "Wrote test Parquet file with multiple row groups: "
              << filepath << std::endl;
  } else if (format == FileFormat::IPC) {
    // IPC write logic - write multiple record batches
    auto writer_result = arrow::ipc::MakeFileWriter(out, schema);
    ARROW_RETURN_NOT_OK(writer_result.status());
    auto writer = std::move(writer_result).ValueOrDie();

    // Write multiple batches
    for (const auto &batch : batches) {
      ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    }

    ARROW_RETURN_NOT_OK(writer->Close());

    std::cout << "Wrote test IPC file with " << batches.size()
              << " batches: " << filepath << std::endl;
  }

  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}