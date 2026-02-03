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
// limitations under the License

#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "zvec/db/index_params.h"
#include "zvec/db/schema.h"
#include "zvec/db/type.h"

namespace zvec {

inline CollectionSchema::Ptr GetCollectionSchema() {
  auto invert_params = std::make_shared<InvertIndexParams>(true);
  auto collection_schema = std::make_shared<CollectionSchema>(
      "test_collection",
      std::vector<FieldSchema::Ptr>{
          std::make_shared<FieldSchema>("id", DataType::UINT64, false, nullptr),
          std::make_shared<FieldSchema>("invert_id", DataType::UINT64, false,
                                        invert_params),

          std::make_shared<FieldSchema>("bool", DataType::BOOL, false, nullptr),
          std::make_shared<FieldSchema>("invert_bool", DataType::BOOL, false,
                                        invert_params),

          std::make_shared<FieldSchema>("bool_array", DataType::ARRAY_BOOL,
                                        false, nullptr),
          std::make_shared<FieldSchema>(
              "invert_bool_array", DataType::ARRAY_BOOL, false, invert_params),

          std::make_shared<FieldSchema>("name", DataType::STRING, false,
                                        nullptr),
          std::make_shared<FieldSchema>("invert_name", DataType::STRING, false,
                                        invert_params),

          std::make_shared<FieldSchema>("age", DataType::INT32, false, nullptr),
          std::make_shared<FieldSchema>(
              "invert_age", DataType::INT32, false,
              std::make_shared<InvertIndexParams>(true)),

          std::make_shared<FieldSchema>("score", DataType::DOUBLE, false,
                                        nullptr),

          std::make_shared<FieldSchema>("optional_age", DataType::UINT32, true,
                                        nullptr),
          std::make_shared<FieldSchema>("invert_optional_age", DataType::UINT32,
                                        true, invert_params),

          std::make_shared<FieldSchema>("category_set", DataType::ARRAY_INT32,
                                        true, nullptr),
          std::make_shared<FieldSchema>("invert_category_set",
                                        DataType::ARRAY_INT32, true,
                                        invert_params),

          // add vector field
          std::make_shared<FieldSchema>(
              "dense", DataType::VECTOR_FP32, 4, false,
              std::make_shared<FlatIndexParams>(MetricType::L2)),

          // add sparse vector
          std::make_shared<FieldSchema>(
              "sparse", DataType::SPARSE_VECTOR_FP32, 0, false,
              std::make_shared<FlatIndexParams>(MetricType::IP)),
      });

  return collection_schema;
}

inline Doc CreateDoc(const uint64_t doc_id) {
  Doc new_doc;
  new_doc.set_pk("pk_" + std::to_string(doc_id));
  new_doc.set_doc_id(doc_id);

  new_doc.set<uint64_t>("id", doc_id);
  new_doc.set<uint64_t>("invert_id", doc_id);
  new_doc.set<bool>("bool", doc_id % 100 == 0);
  new_doc.set<bool>("invert_bool", doc_id % 100 == 0);
  new_doc.set<int32_t>("age", doc_id % 100);
  new_doc.set<int32_t>("invert_age", doc_id % 100);
  if (uint32_t v = doc_id % 100; v) {
    new_doc.set("optional_age", v);
    new_doc.set("invert_optional_age", v);
  }
  auto name = "user_" + std::to_string(doc_id % 100);
  new_doc.set<std::string>("name", name);
  new_doc.set<std::string>("invert_name", name);
  new_doc.set<double>("score", static_cast<double>(rand() % 1000) / 10.0);

  // vector
  std::vector<float> vv;
  for (uint32_t i = 0; i < 4; i++) {
    vv.push_back(static_cast<float>(doc_id));
  }
  new_doc.set<std::vector<float>>("dense", vv);

  // sparse vector
  {
    std::vector<uint32_t> indices;
    std::vector<float> values;
    for (uint32_t i = 0; i < doc_id % 100; i++) {
      indices.push_back(i);
      values.push_back(static_cast<float>(doc_id));
    }
    new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
        "sparse", std::make_pair(indices, values));
  }

  auto category_size = doc_id % 100;
  if (category_size > 0) {
    std::vector<int32_t> category;
    for (uint32_t i = 1; i <= category_size; i++) {
      category.push_back(i);
    }
    new_doc.set("category_set", category);
    new_doc.set("invert_category_set", category);
  }

  if (doc_id % 3 == 0) {
    new_doc.set<std::vector<bool>>("bool_array", {true, false, true});
    new_doc.set<std::vector<bool>>("invert_bool_array", {true, false, true});
  } else if (doc_id % 3 == 1) {
    new_doc.set<std::vector<bool>>("bool_array", {true, true, true});
    new_doc.set<std::vector<bool>>("invert_bool_array", {true, true, true});
  } else {
    new_doc.set<std::vector<bool>>("bool_array", {false, false, false});
    new_doc.set<std::vector<bool>>("invert_bool_array", {false, false, false});
  }

  return new_doc;
}

inline Status InsertDoc(const Segment::Ptr &segment,
                        const uint64_t start_doc_id,
                        const uint64_t end_doc_id) {
  srand(time(NULL));
  long long create_total = 0;
  long long insert_total = 0;
  for (auto doc_id = start_doc_id; doc_id < end_doc_id; doc_id++) {
    if (segment) {
      auto start = std::chrono::system_clock::now();
      Doc new_doc = CreateDoc(doc_id);
      auto end = std::chrono::system_clock::now();
      auto create_cost =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      create_total += create_cost;

      start = std::chrono::system_clock::now();
      auto status = segment->Insert(new_doc);
      if (!status.ok()) {
        return status;
      }
      end = std::chrono::system_clock::now();
      auto insert_cost =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      insert_total += insert_cost;
    }
  }
  std::cout << "pure create cost " << create_total << "us" << std::endl;
  std::cout << "pure insert cost " << insert_total << "us" << std::endl;
  return Status::OK();
}

class RecallTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);

    collection_schema_ = GetCollectionSchema();
    auto segment = create_segment();
    if (segment == nullptr) {
      LOG_ERROR("create segment failed");
      EXPECT_TRUE(segment != nullptr);
      std::exit(EXIT_FAILURE);
    }
    auto status = InsertDoc(segment, 0, 10000);
    if (!status.ok()) {
      LOG_ERROR("insert doc failed: %s", status.c_str());
      EXPECT_TRUE(status.ok());
      std::exit(EXIT_FAILURE);
    }
    segments_.push_back(segment);
  }

  static void TearDownTestSuite() {
    segments_.clear();
    FileHelper::RemoveDirectory(seg_path_);
  }

 public:
  static std::string GetPath() {
    return seg_path_;
  }

  static Segment::Ptr create_segment();

 protected:
  static inline std::string seg_path_ = "./test_collection";
  static inline CollectionSchema::Ptr collection_schema_;
  static inline std::vector<Segment::Ptr> segments_;
};

inline Segment::Ptr RecallTest::create_segment() {
  auto seg_path = GetPath();
  auto segment_meta = std::make_shared<SegmentMeta>();
  segment_meta->set_id(0);

  auto id_map = IDMap::CreateAndOpen("test_collection", GetPath() + "/id_map",
                                     true, false);
  auto delete_store = std::make_shared<DeleteStore>("test_collection");

  Version v1;
  v1.set_schema(*collection_schema_);
  std::string v_path = GetPath() + "/test_manifest";
  FileHelper::CreateDirectory(v_path);
  auto vm = VersionManager::Create(v_path, v1);
  if (!vm.has_value()) {
    LOG_ERROR("create version manager failed: %s", vm.error().c_str());
    return nullptr;
  }

  BlockMeta mem_block;
  mem_block.id_ = 0;
  mem_block.type_ = BlockType::SCALAR;
  mem_block.min_doc_id_ = 0;
  mem_block.max_doc_id_ = 0;
  mem_block.doc_count_ = 0;
  segment_meta->set_writing_forward_block(mem_block);

  SegmentOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result =
      Segment::CreateAndOpen(GetPath(), *collection_schema_, 0, 0, id_map,
                             delete_store, vm.value(), options);

  if (!result) {
    LOG_ERROR("create segment failed: %s", result.error().c_str());
    return nullptr;
  }
  auto segment = result.value();
  return segment;
}

}  // namespace zvec
