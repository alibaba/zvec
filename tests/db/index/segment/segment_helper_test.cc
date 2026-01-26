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

#include "db/index/segment/segment_helper.h"
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <variant>
#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/meta.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "utils/utils.h"
#include "zvec/db/options.h"
#include "zvec/db/schema.h"

using namespace zvec;

class SegmentHelperTest : public testing::Test {
 protected:
  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);

    FileHelper::RemoveDirectory(col_path);
    FileHelper::CreateDirectory(col_path);

    std::string idmap_path =
        FileHelper::MakeFilePath(col_path, FileID::ID_FILE, 0);
    id_map = IDMap::CreateAndOpen(col_name, idmap_path, true, false);
    if (id_map == nullptr) {
      throw std::runtime_error("Failed to create id map");
    }

    std::string delete_store_path =
        FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
    delete_store = std::make_shared<DeleteStore>(col_name);
  }

  void TearDown() override {
    id_map.reset();
    delete_store.reset();

    // FileHelper::RemoveDirectory(col_path);
  }

 public:
  std::string GetColPath() {
    return col_path;
  }

 protected:
  std::string col_name = "test_segment_helper";
  std::string col_path = "./test_collection";
  IDMap::Ptr id_map;
  DeleteStore::Ptr delete_store;
};

TEST_F(SegmentHelperTest, CompactTask_General) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  bool forward_use_parquet = false;
  auto seg_options =
      SegmentOptions{false, !forward_use_parquet, DEFAULT_MAX_BUFFER_SIZE};

  // Create segments
  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      seg_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 1, 1000, id_map, delete_store, version_manager,
      seg_options, 1000, 1000);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg2->flush().ok());
  std::cout << "seg2: " << seg2->meta()->to_string_formatted() << std::endl;

  // Prepare segments for compaction
  std::vector<Segment::Ptr> segments = {seg1, seg2};

  // Create compact task
  SegmentID output_segment_id = 2;
  CompactTask task(GetColPath(), schema, segments,
                   output_segment_id,    // output_segment_id
                   nullptr,              // filter
                   forward_use_parquet,  // forward_use_parquet
                   1                     // concurrency
  );

  // Create segment task
  auto segment_task = SegmentTask::CreateComapctTask(task);

  // Verify task creation
  ASSERT_TRUE(segment_task != nullptr);

  // Execute the task
  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  ASSERT_TRUE(status.ok());

  auto segment_compact_task = std::get<CompactTask>(segment_task->task_info());
  // Verify output segment
  auto output_segment_meta = segment_compact_task.output_segment_meta_;
  ASSERT_EQ(output_segment_meta->id(), output_segment_id);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());

  // Move segment directory
  auto tmp_segment_path =
      FileHelper::MakeTempSegmentPath(GetColPath(), output_segment_id);
  auto new_segment_path =
      FileHelper::MakeSegmentPath(GetColPath(), output_segment_id);
  FileHelper::MoveDirectory(tmp_segment_path, new_segment_path);

  seg_options.read_only_ = true;
  version_manager->set_enable_mmap(!forward_use_parquet);
  auto seg3_ret = Segment::Open(
      GetColPath(), *schema, *segment_compact_task.output_segment_meta_, id_map,
      delete_store, version_manager, seg_options);
  if (!seg3_ret.has_value()) {
    std::cout << seg3_ret.error().message() << std::endl;
    ASSERT_TRUE(false);
  }

  auto seg3 = std::move(seg3_ret.value());
  ASSERT_EQ(seg3->id(), output_segment_id);

  std::cout << seg3->meta()->to_string_formatted() << std::endl;
  ASSERT_EQ(seg3->doc_count(), seg1->doc_count() + seg2->doc_count());

  for (uint64_t i = 0; i < seg3->doc_count(); i++) {
    auto doc = seg3->Fetch(i);
    ASSERT_NE(doc, nullptr);
    auto expect_doc = test::TestHelper::CreateDoc(i, *schema);
    ASSERT_EQ(*doc, expect_doc);
  }

  ASSERT_TRUE(seg1->destroy().ok());
  ASSERT_TRUE(seg2->destroy().ok());
}

TEST_F(SegmentHelperTest, CompactTask_ScalarIndex) {
  auto schema = test::TestHelper::CreateSchemaWithScalarIndex(false);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  bool forward_use_parquet = false;
  auto seg_options =
      SegmentOptions{false, !forward_use_parquet, DEFAULT_MAX_BUFFER_SIZE};

  // Create segments
  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      seg_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 1, 1000, id_map, delete_store, version_manager,
      seg_options, 1000, 1000);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg2->flush().ok());
  std::cout << "seg2: " << seg2->meta()->to_string_formatted() << std::endl;

  // Prepare segments for compaction
  std::vector<Segment::Ptr> segments = {seg1, seg2};

  // Create compact task
  SegmentID output_segment_id = 2;
  CompactTask task(GetColPath(), schema, segments,
                   output_segment_id,    // output_segment_id
                   nullptr,              // filter
                   forward_use_parquet,  // forward_use_parquet
                   1                     // concurrency
  );

  // Create segment task
  auto segment_task = SegmentTask::CreateComapctTask(task);

  // Verify task creation
  ASSERT_TRUE(segment_task != nullptr);

  // Execute the task
  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  ASSERT_TRUE(status.ok());

  auto segment_compact_task = std::get<CompactTask>(segment_task->task_info());
  // Verify output segment
  auto output_segment_meta = segment_compact_task.output_segment_meta_;
  ASSERT_EQ(output_segment_meta->id(), output_segment_id);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());

  // Move segment directory
  auto tmp_segment_path =
      FileHelper::MakeTempSegmentPath(GetColPath(), output_segment_id);
  auto new_segment_path =
      FileHelper::MakeSegmentPath(GetColPath(), output_segment_id);
  FileHelper::MoveDirectory(tmp_segment_path, new_segment_path);

  seg_options.read_only_ = true;
  version_manager->set_enable_mmap(!forward_use_parquet);
  auto seg3_ret = Segment::Open(
      GetColPath(), *schema, *segment_compact_task.output_segment_meta_, id_map,
      delete_store, version_manager, seg_options);
  if (!seg3_ret.has_value()) {
    std::cout << seg3_ret.error().message() << std::endl;
    ASSERT_TRUE(false);
  }

  auto seg3 = std::move(seg3_ret.value());
  ASSERT_EQ(seg3->id(), output_segment_id);

  std::cout << seg3->meta()->to_string_formatted() << std::endl;
  ASSERT_EQ(seg3->doc_count(), seg1->doc_count() + seg2->doc_count());

  for (uint64_t i = 0; i < seg3->doc_count(); i++) {
    auto doc = seg3->Fetch(i);
    ASSERT_NE(doc, nullptr);
    auto expect_doc = test::TestHelper::CreateDoc(i, *schema);
    ASSERT_EQ(*doc, expect_doc);
  }

  ASSERT_TRUE(seg1->destroy().ok());
  ASSERT_TRUE(seg2->destroy().ok());
}

TEST_F(SegmentHelperTest, CompactTask_VectorIndex) {
  auto schema = test::TestHelper::CreateSchemaWithVectorIndex();

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  bool forward_use_parquet = false;
  auto seg_options =
      SegmentOptions{false, !forward_use_parquet, DEFAULT_MAX_BUFFER_SIZE};

  // Create segments
  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      seg_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 1, 1000, id_map, delete_store, version_manager,
      seg_options, 1000, 1000);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg2->flush().ok());
  std::cout << "seg2: " << seg2->meta()->to_string_formatted() << std::endl;

  // Prepare segments for compaction
  std::vector<Segment::Ptr> segments = {seg1, seg2};

  // Create compact task
  SegmentID output_segment_id = 2;
  CompactTask task(GetColPath(), schema, segments,
                   output_segment_id,    // output_segment_id
                   nullptr,              // filter
                   forward_use_parquet,  // forward_use_parquet
                   1                     // concurrency
  );

  // Create segment task
  auto segment_task = SegmentTask::CreateComapctTask(task);

  // Verify task creation
  ASSERT_TRUE(segment_task != nullptr);

  // Execute the task
  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  ASSERT_TRUE(status.ok());

  auto segment_compact_task = std::get<CompactTask>(segment_task->task_info());
  // Verify output segment
  auto output_segment_meta = segment_compact_task.output_segment_meta_;
  ASSERT_EQ(output_segment_meta->id(), output_segment_id);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());

  // Move segment directory
  auto tmp_segment_path =
      FileHelper::MakeTempSegmentPath(GetColPath(), output_segment_id);
  auto new_segment_path =
      FileHelper::MakeSegmentPath(GetColPath(), output_segment_id);
  FileHelper::MoveDirectory(tmp_segment_path, new_segment_path);

  seg_options.read_only_ = true;
  version_manager->set_enable_mmap(!forward_use_parquet);
  auto seg3_ret = Segment::Open(
      GetColPath(), *schema, *segment_compact_task.output_segment_meta_, id_map,
      delete_store, version_manager, seg_options);
  if (!seg3_ret.has_value()) {
    std::cout << seg3_ret.error().message() << std::endl;
    ASSERT_TRUE(false);
  }

  auto seg3 = std::move(seg3_ret.value());
  ASSERT_EQ(seg3->id(), output_segment_id);

  std::cout << seg3->meta()->to_string_formatted() << std::endl;
  ASSERT_EQ(seg3->doc_count(), seg1->doc_count() + seg2->doc_count());

  for (uint64_t i = 0; i < seg3->doc_count(); i++) {
    auto doc = seg3->Fetch(i);
    ASSERT_NE(doc, nullptr);
    auto expect_doc = test::TestHelper::CreateDoc(i, *schema);
    ASSERT_EQ(*doc, expect_doc);
  }

  ASSERT_TRUE(seg1->destroy().ok());
  ASSERT_TRUE(seg2->destroy().ok());
}

TEST_F(SegmentHelperTest, CompactTask_MultipleSegments) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  bool forward_use_parquet = false;
  auto seg_options =
      SegmentOptions{false, !forward_use_parquet, DEFAULT_MAX_BUFFER_SIZE};

  std::vector<Segment::Ptr> input_segs;
  int seg_count = 10;
  int doc_count_per_seg = 100;
  for (int i = 0; i < seg_count; i++) {
    auto seg = test::TestHelper::CreateSegmentWithDoc(
        GetColPath(), *schema, i, i * doc_count_per_seg, id_map, delete_store,
        version_manager, seg_options, i * doc_count_per_seg, doc_count_per_seg);
    ASSERT_TRUE(seg != nullptr);
    ASSERT_TRUE(seg->flush().ok());
    input_segs.push_back(seg);
  }

  // Create compact task
  SegmentID output_segment_id = seg_count;
  CompactTask task(GetColPath(), schema, input_segs,
                   output_segment_id,    // output_segment_id
                   nullptr,              // filter
                   forward_use_parquet,  // forward_use_parquet
                   1                     // concurrency
  );

  // Create segment task
  auto segment_task = SegmentTask::CreateComapctTask(task);

  // Verify task creation
  ASSERT_TRUE(segment_task != nullptr);

  // Execute the task
  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  ASSERT_TRUE(status.ok());

  auto segment_compact_task = std::get<CompactTask>(segment_task->task_info());
  // Verify output segment
  auto output_segment_meta = segment_compact_task.output_segment_meta_;
  ASSERT_EQ(output_segment_meta->id(), output_segment_id);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());

  // Move segment directory
  auto tmp_segment_path =
      FileHelper::MakeTempSegmentPath(GetColPath(), output_segment_id);
  auto new_segment_path =
      FileHelper::MakeSegmentPath(GetColPath(), output_segment_id);
  FileHelper::MoveDirectory(tmp_segment_path, new_segment_path);

  seg_options.read_only_ = true;
  version_manager->set_enable_mmap(!forward_use_parquet);
  auto seg3_ret = Segment::Open(
      GetColPath(), *schema, *segment_compact_task.output_segment_meta_, id_map,
      delete_store, version_manager, seg_options);
  if (!seg3_ret.has_value()) {
    std::cout << seg3_ret.error().message() << std::endl;
    ASSERT_TRUE(false);
  }

  auto seg3 = std::move(seg3_ret.value());
  ASSERT_EQ(seg3->id(), output_segment_id);

  std::cout << seg3->meta()->to_string_formatted() << std::endl;
  ASSERT_EQ(seg3->doc_count(), seg_count * doc_count_per_seg);

  for (uint64_t i = 0; i < seg3->doc_count(); i++) {
    auto doc = seg3->Fetch(i);
    if (doc == nullptr) {
      std::cout << "doc is null: " << i << std::endl;
    }
    ASSERT_NE(doc, nullptr);
    auto expect_doc = test::TestHelper::CreateDoc(i, *schema);
    ASSERT_EQ(*doc, expect_doc);
  }
}

TEST_F(SegmentHelperTest, CompactTask_Filter) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  bool forward_use_parquet = false;
  auto seg_options =
      SegmentOptions{false, !forward_use_parquet, DEFAULT_MAX_BUFFER_SIZE};

  // Create segments
  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      seg_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  // Create a simple filter
  auto filter = std::make_shared<EasyIndexFilter>(
      [&](uint64_t id) -> bool { return id < 10; });
  // Note: Actual filter configuration would depend on the IndexFilter
  // implementation

  // Create compact task with filter
  SegmentID output_segment_id = 1;
  CompactTask task(GetColPath(), schema, {seg1},  // Single segment with filter
                   output_segment_id,             // output_segment_id
                   filter,
                   forward_use_parquet,  // forward_use_parquet
                   1                     // concurrency
  );

  // Create and execute task
  auto segment_task = SegmentTask::CreateComapctTask(task);
  ASSERT_TRUE(segment_task != nullptr);

  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  ASSERT_TRUE(status.ok());

  auto segment_compact_task = std::get<CompactTask>(segment_task->task_info());
  // Verify output segment
  auto output_segment_meta = segment_compact_task.output_segment_meta_;
  std::cout << output_segment_meta->to_string_formatted() << std::endl;
  ASSERT_EQ(output_segment_meta->id(), output_segment_id);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());

  // Move segment directory
  auto tmp_segment_path =
      FileHelper::MakeTempSegmentPath(GetColPath(), output_segment_id);
  auto new_segment_path =
      FileHelper::MakeSegmentPath(GetColPath(), output_segment_id);
  FileHelper::MoveDirectory(tmp_segment_path, new_segment_path);

  seg_options.read_only_ = true;
  version_manager->set_enable_mmap(!forward_use_parquet);
  auto seg2_ret = Segment::Open(
      GetColPath(), *schema, *segment_compact_task.output_segment_meta_, id_map,
      delete_store, version_manager, seg_options);
  if (!seg2_ret.has_value()) {
    std::cout << seg2_ret.error().message() << std::endl;
    ASSERT_TRUE(false);
  }

  auto seg2 = std::move(seg2_ret.value());
  ASSERT_EQ(seg2->id(), output_segment_id);

  std::cout << seg2->meta()->to_string_formatted() << std::endl;
  ASSERT_EQ(seg2->doc_count(), seg1->doc_count() - 10);

  ASSERT_TRUE(seg1->destroy().ok());
}

TEST_F(SegmentHelperTest, CompactTask_FilterAll) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  bool forward_use_parquet = false;
  auto seg_options =
      SegmentOptions{false, !forward_use_parquet, DEFAULT_MAX_BUFFER_SIZE};

  // Create segments
  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      seg_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  // Create a simple filter
  auto filter = std::make_shared<EasyIndexFilter>(
      [&](uint64_t id) -> bool { return true; });
  // Note: Actual filter configuration would depend on the IndexFilter
  // implementation

  // Create compact task with filter
  SegmentID output_segment_id = 1;
  CompactTask task(GetColPath(), schema, {seg1},  // Single segment with filter
                   output_segment_id,             // output_segment_id
                   filter,
                   forward_use_parquet,  // forward_use_parquet
                   1                     // concurrency
  );

  // Create and execute task
  auto segment_task = SegmentTask::CreateComapctTask(task);
  ASSERT_TRUE(segment_task != nullptr);

  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  ASSERT_TRUE(status.ok());

  auto segment_compact_task = std::get<CompactTask>(segment_task->task_info());
  // Verify output segment
  auto output_segment_meta = segment_compact_task.output_segment_meta_;
  ASSERT_EQ(output_segment_meta, nullptr);

  auto tmp_segment_path =
      FileHelper::MakeTempSegmentPath(GetColPath(), output_segment_id);
  ASSERT_FALSE(FileHelper::DirectoryExists(tmp_segment_path));
}

TEST_F(SegmentHelperTest, CreateVectorIndexTask_AllFields) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  // Create a segment
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      SegmentOptions{false, true, DEFAULT_MAX_BUFFER_SIZE}, 0, 1000);
  ASSERT_TRUE(segment != nullptr);
  ASSERT_TRUE(segment->dump().ok());

  // Create index params
  auto index_params =
      std::make_shared<HnswIndexParams>(MetricType::L2,  // metric_type
                                        16,              // m
                                        100              // ef_construction
      );

  // Create create index task
  CreateVectorIndexTask task(
      segment,
      "",  // column_to_build_vector_index (empty means all vector columns)
      index_params,
      1  // concurrency
  );

  // Create segment task
  auto segment_task = SegmentTask::CreateCreateVectorIndexTask(task);

  // Verify task creation
  ASSERT_TRUE(segment_task != nullptr);

  // Execute the task
  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  EXPECT_TRUE(status.ok());

  // Verify output segment meta
  auto index_task = std::get<CreateVectorIndexTask>(segment_task->task_info());
  auto output_segment_meta = index_task.output_segment_meta_;
  std::cout << "output_segment_meta: "
            << output_segment_meta->to_string_formatted() << std::endl;
  ASSERT_EQ(output_segment_meta->id(), 0);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());

  auto segment_meta = std::make_shared<SegmentMeta>(*segment->meta());
  segment_meta->remove_writing_forward_block();
  // create all vector index will not change segment meta
  ASSERT_EQ(*output_segment_meta, *segment_meta);
}

TEST_F(SegmentHelperTest, CreateVectorIndexTask_SingleField) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);

  Version version;
  version.set_schema(*schema);
  auto version_manager_tmp = VersionManager::Create(col_path, version);
  if (!version_manager_tmp.has_value()) {
    throw std::runtime_error("Failed to create version manager");
  }

  auto version_manager = version_manager_tmp.value();

  // Create a segment
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      GetColPath(), *schema, 0, 0, id_map, delete_store, version_manager,
      SegmentOptions{false, true, DEFAULT_MAX_BUFFER_SIZE}, 0, 1000);
  ASSERT_TRUE(segment != nullptr);
  ASSERT_TRUE(segment->dump().ok());

  // Create index params
  auto index_params =
      std::make_shared<HnswIndexParams>(MetricType::IP,  // metric_type
                                        16,              // m
                                        100              // ef_construction
      );

  // Create create index task
  CreateVectorIndexTask task(segment,
                             "dense_fp32",  // column_to_build_vector_index
                                            // (empty means all vector columns)
                             index_params,
                             1  // concurrency
  );

  // Create segment task
  auto segment_task = SegmentTask::CreateCreateVectorIndexTask(task);

  // Verify task creation
  ASSERT_TRUE(segment_task != nullptr);

  // Execute the task
  Status status = SegmentHelper::Execute(segment_task);
  std::cout << "status: " << status.message() << std::endl;
  EXPECT_TRUE(status.ok());

  // Verify output segment meta
  auto index_task = std::get<CreateVectorIndexTask>(segment_task->task_info());
  auto output_segment_meta = index_task.output_segment_meta_;
  std::cout << "output_segment_meta: "
            << output_segment_meta->to_string_formatted() << std::endl;
  ASSERT_EQ(output_segment_meta->id(), 0);
  ASSERT_FALSE(output_segment_meta->writing_forward_block().has_value());
}