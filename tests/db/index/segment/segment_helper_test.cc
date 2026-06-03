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
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
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
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_params.h"
#include "db/index/column/vector_column/vector_index_results.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/meta.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "utils/utils.h"
#include "zvec/db/options.h"
#include "zvec/db/query_params.h"
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
  VersionManager::Ptr CreateVersionManager(const CollectionSchema &schema) {
    Version version;
    version.set_schema(schema);
    auto vm = VersionManager::Create(col_path, version);
    if (!vm.has_value()) {
      throw std::runtime_error("Failed to create version manager");
    }
    return vm.value();
  }

  SegmentOptions WriteOptions() const {
    return SegmentOptions{false, true, DEFAULT_MAX_BUFFER_SIZE};
  }

  struct CompactResult {
    CompactTask compact_task;
    Segment::Ptr output_segment;  // null when filter dropped every doc
  };

  // Execute a CompactTask end-to-end: build it, run it, move the tmp segment
  // dir into place, and reopen the output segment in read-only mode.
  CompactResult RunCompactAndOpen(CollectionSchema::Ptr schema,
                                  std::vector<Segment::Ptr> segments,
                                  SegmentID output_segment_id,
                                  IndexFilter::Ptr filter,
                                  const VersionManager::Ptr &version_manager,
                                  int concurrency = 1) {
    const bool forward_use_parquet = false;
    CompactTask task(col_path, schema, std::move(segments), output_segment_id,
                     std::move(filter), forward_use_parquet, concurrency);
    auto segment_task = SegmentTask::CreateCompactTask(task);
    EXPECT_NE(segment_task, nullptr);
    if (segment_task == nullptr) return {task, nullptr};

    auto status = SegmentHelper::Execute(segment_task);
    EXPECT_TRUE(status.ok()) << status.message();

    auto executed = std::get<CompactTask>(segment_task->task_info());
    if (executed.output_segment_meta_ == nullptr) {
      return {executed, nullptr};
    }

    auto tmp_path =
        FileHelper::MakeTempSegmentPath(col_path, output_segment_id);
    auto dst_path = FileHelper::MakeSegmentPath(col_path, output_segment_id);
    EXPECT_TRUE(FileHelper::MoveDirectory(tmp_path, dst_path));

    SegmentOptions read_options{true, !forward_use_parquet,
                                DEFAULT_MAX_BUFFER_SIZE};
    version_manager->set_enable_mmap(!forward_use_parquet);
    auto seg_ret =
        Segment::Open(col_path, *schema, *executed.output_segment_meta_, id_map,
                      delete_store, version_manager, read_options);
    EXPECT_TRUE(seg_ret.has_value());
    if (!seg_ret.has_value()) return {executed, nullptr};
    return {executed, std::move(seg_ret.value())};
  }

  std::string col_name = "test_segment_helper";
  std::string col_path = "./test_collection";
  IDMap::Ptr id_map;
  DeleteStore::Ptr delete_store;
};

TEST_F(SegmentHelperTest, CompactTask_General) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());
  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 1, 1000, id_map, delete_store, version_manager,
      write_options, 1000, 1000);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg2->flush().ok());

  SegmentID output_segment_id = 2;
  auto [compact_task, seg3] = RunCompactAndOpen(
      schema, {seg1, seg2}, output_segment_id, nullptr, version_manager);

  ASSERT_NE(seg3, nullptr);
  ASSERT_EQ(compact_task.output_segment_meta_->id(), output_segment_id);
  ASSERT_FALSE(
      compact_task.output_segment_meta_->writing_forward_block().has_value());
  ASSERT_EQ(seg3->id(), output_segment_id);
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
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());
  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 1, 1000, id_map, delete_store, version_manager,
      write_options, 1000, 1000);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg2->flush().ok());

  SegmentID output_segment_id = 2;
  auto [compact_task, seg3] = RunCompactAndOpen(
      schema, {seg1, seg2}, output_segment_id, nullptr, version_manager);

  ASSERT_NE(seg3, nullptr);
  ASSERT_EQ(compact_task.output_segment_meta_->id(), output_segment_id);
  ASSERT_FALSE(
      compact_task.output_segment_meta_->writing_forward_block().has_value());
  ASSERT_EQ(seg3->id(), output_segment_id);
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
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());
  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 1, 1000, id_map, delete_store, version_manager,
      write_options, 1000, 1000);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg2->flush().ok());

  SegmentID output_segment_id = 2;
  auto [compact_task, seg3] = RunCompactAndOpen(
      schema, {seg1, seg2}, output_segment_id, nullptr, version_manager);

  ASSERT_NE(seg3, nullptr);
  ASSERT_EQ(compact_task.output_segment_meta_->id(), output_segment_id);
  ASSERT_FALSE(
      compact_task.output_segment_meta_->writing_forward_block().has_value());
  ASSERT_EQ(seg3->id(), output_segment_id);
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
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  std::vector<Segment::Ptr> input_segs;
  const int seg_count = 10;
  const int doc_count_per_seg = 100;
  for (int i = 0; i < seg_count; i++) {
    auto seg = test::TestHelper::CreateSegmentWithDoc(
        col_path, *schema, i, i * doc_count_per_seg, id_map, delete_store,
        version_manager, write_options, i * doc_count_per_seg,
        doc_count_per_seg);
    ASSERT_TRUE(seg != nullptr);
    ASSERT_TRUE(seg->flush().ok());
    input_segs.push_back(seg);
  }

  SegmentID output_segment_id = seg_count;
  auto [compact_task, seg3] = RunCompactAndOpen(
      schema, input_segs, output_segment_id, nullptr, version_manager);

  ASSERT_NE(seg3, nullptr);
  ASSERT_EQ(compact_task.output_segment_meta_->id(), output_segment_id);
  ASSERT_FALSE(
      compact_task.output_segment_meta_->writing_forward_block().has_value());
  ASSERT_EQ(seg3->id(), output_segment_id);
  ASSERT_EQ(seg3->doc_count(), seg_count * doc_count_per_seg);

  for (uint64_t i = 0; i < seg3->doc_count(); i++) {
    auto doc = seg3->Fetch(i);
    ASSERT_NE(doc, nullptr);
    auto expect_doc = test::TestHelper::CreateDoc(i, *schema);
    ASSERT_EQ(*doc, expect_doc);
  }
}

TEST_F(SegmentHelperTest, CompactTask_Filter) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  auto filter = std::make_shared<EasyIndexFilter>(
      [](uint64_t id) -> bool { return id < 10; });

  SegmentID output_segment_id = 1;
  auto [compact_task, seg2] = RunCompactAndOpen(
      schema, {seg1}, output_segment_id, filter, version_manager);

  ASSERT_NE(seg2, nullptr);
  ASSERT_EQ(compact_task.output_segment_meta_->id(), output_segment_id);
  ASSERT_FALSE(
      compact_task.output_segment_meta_->writing_forward_block().has_value());
  ASSERT_EQ(seg2->id(), output_segment_id);
  ASSERT_EQ(seg2->doc_count(), seg1->doc_count() - 10);

  ASSERT_TRUE(seg1->destroy().ok());
}

TEST_F(SegmentHelperTest, CompactTask_FilterAll) {
  auto schema = test::TestHelper::CreateNormalSchema(false, col_name);
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 1000);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());

  auto filter = std::make_shared<EasyIndexFilter>(
      [](uint64_t /*id*/) -> bool { return true; });

  SegmentID output_segment_id = 1;
  auto [compact_task, output_segment] = RunCompactAndOpen(
      schema, {seg1}, output_segment_id, filter, version_manager);

  ASSERT_EQ(compact_task.output_segment_meta_, nullptr);
  ASSERT_EQ(output_segment, nullptr);
  ASSERT_FALSE(FileHelper::DirectoryExists(
      FileHelper::MakeTempSegmentPath(col_path, output_segment_id)));
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

TEST_F(SegmentHelperTest, CompactTask_VectorIndexThreeSegmentsRegression) {
  auto schema = test::TestHelper::CreateSchemaWithVectorIndex();
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 300);
  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 1, 300, id_map, delete_store, version_manager,
      write_options, 300, 300);
  auto seg3 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 2, 600, id_map, delete_store, version_manager,
      write_options, 600, 300);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg3 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());
  ASSERT_TRUE(seg2->flush().ok());
  ASSERT_TRUE(seg3->flush().ok());

  auto [compact_task, output_segment] = RunCompactAndOpen(
      schema, {seg1, seg2, seg3}, 3, nullptr, version_manager);

  ASSERT_NE(output_segment, nullptr);
  ASSERT_EQ(output_segment->doc_count(), 900);
  ASSERT_NE(output_segment->Fetch(0), nullptr);
  ASSERT_NE(output_segment->Fetch(899), nullptr);
}

TEST_F(SegmentHelperTest,
       CompactTask_QuantizedVectorIndexThreeSegmentsRegression) {
  auto schema = test::TestHelper::CreateSchemaWithVectorIndex(
      false, col_name,
      std::make_shared<HnswIndexParams>(MetricType::IP, 16, 20,
                                        QuantizeType::FP16));
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 300);
  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 1, 300, id_map, delete_store, version_manager,
      write_options, 300, 300);
  auto seg3 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 2, 600, id_map, delete_store, version_manager,
      write_options, 600, 300);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg3 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());
  ASSERT_TRUE(seg2->flush().ok());
  ASSERT_TRUE(seg3->flush().ok());
  ASSERT_GT(seg1->get_quant_vector_indexer("dense_fp32").size(), 0u);
  ASSERT_GT(seg2->get_quant_vector_indexer("dense_fp32").size(), 0u);
  ASSERT_GT(seg3->get_quant_vector_indexer("dense_fp32").size(), 0u);

  auto [compact_task, output_segment] = RunCompactAndOpen(
      schema, {seg1, seg2, seg3}, 3, nullptr, version_manager);

  ASSERT_NE(output_segment, nullptr);
  ASSERT_EQ(output_segment->doc_count(), 900);
  ASSERT_NE(output_segment->Fetch(0), nullptr);
  ASSERT_NE(output_segment->Fetch(899), nullptr);
  ASSERT_GT(output_segment->get_vector_indexer("dense_fp32").size(), 0u);
  ASSERT_GT(output_segment->get_quant_vector_indexer("dense_fp32").size(), 0u);
}

struct SegmentCompactReuseParam {
  IndexParams::Ptr vector_index_params;
  IndexType expected_output_type;
};

class SegmentCompactReuseTest
    : public SegmentHelperTest,
      public testing::WithParamInterface<SegmentCompactReuseParam> {
 protected:
  // Returns the indexer's underlying VectorIndexParams::type(), defaulting to
  // FLAT if the params can't be downcast (matches the freshly-inserted state).
  static IndexType IndexerType(const VectorColumnIndexer::Ptr &indexer) {
    auto params = std::dynamic_pointer_cast<VectorIndexParams>(
        indexer->field_schema().index_params());
    return params ? params->type() : IndexType::FLAT;
  }

  static QuantizeType QuantizeTypeOf(const IndexParams::Ptr &params) {
    auto vp = std::dynamic_pointer_cast<VectorIndexParams>(params);
    return vp ? vp->quantize_type() : QuantizeType::UNDEFINED;
  }

  // Run CreateVectorIndexTask on `segment` for `column` with `index_params`,
  // then reload the segment so its in-memory indexer reflects the new index
  // (matching collection.cc's post-optimize reload path).
  void OptimizeSegmentToVectorIndex(const Segment::Ptr &segment,
                                    const CollectionSchema &schema,
                                    const std::string &column,
                                    const IndexParams::Ptr &index_params) {
    CreateVectorIndexTask task(segment, column, index_params, 1);
    auto segment_task = SegmentTask::CreateCreateVectorIndexTask(task);
    ASSERT_NE(segment_task, nullptr);
    ASSERT_TRUE(SegmentHelper::Execute(segment_task).ok());
    auto executed = std::get<CreateVectorIndexTask>(segment_task->task_info());
    ASSERT_NE(executed.output_segment_meta_, nullptr);
    ASSERT_TRUE(
        segment
            ->reload_vector_index(schema, executed.output_segment_meta_,
                                  executed.output_vector_indexers_,
                                  executed.output_quant_vector_indexers_)
            .ok());
  }

  struct ScoredDoc {
    uint64_t doc_id;
    float score;
  };

  // CreateDoc seeds VECTOR_FP32 with a constant vector of value (doc_id+0.1f).
  static std::vector<float> MakeFp32QueryVector(uint64_t doc_id_value,
                                                uint32_t dim) {
    return std::vector<float>(dim, static_cast<float>(doc_id_value) + 0.1f);
  }

  static std::vector<ScoredDoc> RunSearch(
      const VectorColumnIndexer::Ptr &indexer, const std::vector<float> &qvec,
      uint32_t topk, const zvec::QueryParams::Ptr &query_params) {
    vector_column_params::QueryParams qp;
    qp.topk = topk;
    qp.filter = nullptr;
    qp.fetch_vector = false;
    qp.query_params = query_params;
    vector_column_params::VectorData data{
        vector_column_params::DenseVector{qvec.data()}};
    auto results = indexer->Search(data, qp);
    EXPECT_TRUE(results.has_value());
    if (!results.has_value()) return {};
    auto vec_res = dynamic_cast<VectorIndexResults *>(results.value().get());
    EXPECT_NE(vec_res, nullptr);
    if (vec_res == nullptr) return {};
    std::vector<ScoredDoc> out;
    for (auto it = vec_res->create_iterator(); it->valid(); it->next()) {
      out.push_back({it->doc_id(), it->score()});
    }
    return out;
  }

  // All test instantiations use IP — higher score is better.
  static std::set<uint64_t> MergeTopKIds(
      std::vector<std::vector<ScoredDoc>> per_seg, uint32_t topk) {
    std::vector<ScoredDoc> all;
    for (auto &v : per_seg)
      for (auto &d : v) all.push_back(d);
    std::sort(all.begin(), all.end(),
              [](const ScoredDoc &a, const ScoredDoc &b) {
                return a.score > b.score;
              });
    std::set<uint64_t> ids;
    for (size_t i = 0; i < all.size() && ids.size() < topk; ++i) {
      ids.insert(all[i].doc_id);
    }
    return ids;
  }

  static zvec::QueryParams::Ptr MakeIsLinearQueryParam(IndexType type) {
    switch (type) {
      case IndexType::HNSW: {
        auto p = std::make_shared<zvec::HnswQueryParams>();
        p->set_is_linear(true);
        return p;
      }
      case IndexType::IVF: {
        auto p = std::make_shared<zvec::IVFQueryParams>();
        p->set_is_linear(true);
        return p;
      }
      case IndexType::HNSW_RABITQ: {
        auto p = std::make_shared<zvec::HnswRabitqQueryParams>();
        p->set_is_linear(true);
        return p;
      }
      case IndexType::FLAT:
      default:
        return std::make_shared<zvec::FlatQueryParams>();
    }
  }
};

// Mimic the normal insertion lifecycle: small segments accumulate vectors
// in flat storage (no vector index built yet), then compaction merges them
// into a single segment whose vector column is built per schema.
TEST_P(SegmentCompactReuseTest, OptimizedSegmentsReuseFirstIndexer) {
  const auto &param = GetParam();
  auto schema = test::TestHelper::CreateSchemaWithVectorIndex(
      false, col_name, param.vector_index_params);
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  constexpr int kSegCount = 3;
  constexpr int kDocsPerSeg = 300;
  constexpr uint32_t kTopK = 10;
  constexpr uint32_t kDim = 128;
  std::vector<Segment::Ptr> segs;
  for (int i = 0; i < kSegCount; i++) {
    auto seg = test::TestHelper::CreateSegmentWithDoc(
        col_path, *schema, i, i * kDocsPerSeg, id_map, delete_store,
        version_manager, write_options, i * kDocsPerSeg, kDocsPerSeg);
    ASSERT_NE(seg, nullptr);
    ASSERT_TRUE(seg->flush().ok());
    segs.push_back(seg);
  }

  // Capture groundtruth via FlatQuery on each source segment while every
  // segment is still backed by a flat indexer (before seg[0] is optimized).
  const std::vector<uint64_t> query_doc_values{0, kDocsPerSeg,
                                               kSegCount * kDocsPerSeg - 1};
  std::vector<std::set<uint64_t>> groundtruth;
  groundtruth.reserve(query_doc_values.size());
  auto flat_qp = std::make_shared<zvec::FlatQueryParams>();
  for (uint64_t qv : query_doc_values) {
    auto qvec = MakeFp32QueryVector(qv, kDim);
    std::vector<std::vector<ScoredDoc>> per_seg;
    per_seg.reserve(segs.size());
    // Per-segment indexers use block-local doc ids (0..kDocsPerSeg-1).
    // The compacted output indexer reindexes them sequentially across
    // segments, so add segment offset to align id spaces before merging.
    for (size_t s = 0; s < segs.size(); ++s) {
      auto in_indexers = segs[s]->get_vector_indexer("dense_fp32");
      ASSERT_FALSE(in_indexers.empty());
      ASSERT_EQ(IndexerType(in_indexers.front()), IndexType::FLAT);
      auto local = RunSearch(in_indexers.front(), qvec, kTopK, flat_qp);
      const uint64_t offset = static_cast<uint64_t>(s) * kDocsPerSeg;
      for (auto &d : local) d.doc_id += offset;
      per_seg.push_back(std::move(local));
    }
    auto gt = MergeTopKIds(std::move(per_seg), kTopK);
    ASSERT_EQ(gt.size(), kTopK);
    groundtruth.push_back(std::move(gt));
  }

  // Optimize seg[0]'s vector fields to the parametric index type, mimicking
  // the lifecycle the compact path exercises.
  for (const auto &vf : schema->vector_fields()) {
    OptimizeSegmentToVectorIndex(segs[0], *schema, vf->name(),
                                 vf->index_params());
  }

  // For quantized index types (e.g. HNSW_RABITQ) the built index lives in
  // get_quant_vector_indexer; get_vector_indexer keeps the raw FLAT
  // indexer. See CompactTask_QuantizedVectorIndexThreeSegmentsRegression.
  const bool quantized =
      QuantizeTypeOf(param.vector_index_params) != QuantizeType::UNDEFINED;
  for (int i = 0; i < kSegCount; i++) {
    auto in_indexers = quantized
                           ? segs[i]->get_quant_vector_indexer("dense_fp32")
                           : segs[i]->get_vector_indexer("dense_fp32");
    ASSERT_FALSE(in_indexers.empty());
    ASSERT_EQ(IndexerType(in_indexers.front()),
              i == 0 ? param.expected_output_type : IndexType::FLAT);
  }

  auto [compact_task, output_segment] =
      RunCompactAndOpen(schema, segs, kSegCount, nullptr, version_manager);

  ASSERT_NE(output_segment, nullptr);
  ASSERT_EQ(output_segment->doc_count(), kSegCount * kDocsPerSeg);
  ASSERT_NE(output_segment->Fetch(0), nullptr);
  ASSERT_NE(output_segment->Fetch(kSegCount * kDocsPerSeg - 1), nullptr);

  auto out_indexers =
      quantized ? output_segment->get_quant_vector_indexer("dense_fp32")
                : output_segment->get_vector_indexer("dense_fp32");
  ASSERT_FALSE(out_indexers.empty());
  EXPECT_EQ(IndexerType(out_indexers.front()), param.expected_output_type);

  // is_linear queries on the merged indexer must reproduce the pre-compact
  // groundtruth. Quantized indexers are allowed a small recall hit.
  auto linear_qp = MakeIsLinearQueryParam(param.expected_output_type);
  const double kMinRecall = quantized ? 0.8 : 1.0;
  for (size_t qi = 0; qi < query_doc_values.size(); ++qi) {
    auto qvec = MakeFp32QueryVector(query_doc_values[qi], kDim);
    auto hits = RunSearch(out_indexers.front(), qvec, kTopK, linear_qp);
    ASSERT_EQ(hits.size(), kTopK);
    size_t intersect = 0;
    for (const auto &h : hits) {
      if (groundtruth[qi].count(h.doc_id)) intersect++;
    }
    double recall = static_cast<double>(intersect) / kTopK;
    EXPECT_GE(recall, kMinRecall)
        << "query[" << qi << "] (value=" << query_doc_values[qi]
        << ") recall=" << recall;
  }
}

INSTANTIATE_TEST_SUITE_P(Hnsw, SegmentCompactReuseTest,
                         testing::Values(SegmentCompactReuseParam{
                             std::make_shared<HnswIndexParams>(MetricType::IP,
                                                               16, 200),
                             IndexType::HNSW}));

// CreateNormalSchema() only puts the test's vector_index_params on dense_fp32.
// The other 4 vector fields are hardcoded — dense_fp16/dense_int8/sparse_fp16
// are always FlatIndexParams, and sparse_fp32 gets the
//   cloned params only if supports_sparse is true (utils.cc:117-124), which
//   excludes IVF and HNSW_RABITQ — so for IVF it also falls back to FLAT.

INSTANTIATE_TEST_SUITE_P(
    Ivf, SegmentCompactReuseTest,
    testing::Values(SegmentCompactReuseParam{
        std::make_shared<IVFIndexParams>(MetricType::IP, 10, 4, false,
                                         QuantizeType::UNDEFINED),
        IndexType::IVF}));

#if RABITQ_SUPPORTED
INSTANTIATE_TEST_SUITE_P(HnswRabitq, SegmentCompactReuseTest,
                         testing::Values(SegmentCompactReuseParam{
                             std::make_shared<HnswRabitqIndexParams>(
                                 MetricType::IP, 7, 256, 16, 200, 0),
                             IndexType::HNSW_RABITQ}));
#endif

TEST_F(SegmentHelperTest, CompactTask_FilterMultiSegmentsRegression) {
  auto schema = test::TestHelper::CreateSchemaWithVectorIndex();
  auto version_manager = CreateVersionManager(*schema);
  auto write_options = WriteOptions();

  auto seg1 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager,
      write_options, 0, 400);
  auto seg2 = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 1, 400, id_map, delete_store, version_manager,
      write_options, 400, 400);
  ASSERT_TRUE(seg1 != nullptr);
  ASSERT_TRUE(seg2 != nullptr);
  ASSERT_TRUE(seg1->flush().ok());
  ASSERT_TRUE(seg2->flush().ok());

  auto filter = std::make_shared<EasyIndexFilter>(
      [](uint64_t id) -> bool { return id < 100 || (id >= 400 && id < 450); });

  auto [compact_task, output_segment] =
      RunCompactAndOpen(schema, {seg1, seg2}, 2, filter, version_manager);

  ASSERT_NE(output_segment, nullptr);
  ASSERT_EQ(output_segment->doc_count(), 650);
}
