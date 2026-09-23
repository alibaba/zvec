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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#if DISKANN_SUPPORTED
#include "algorithm/diskann/diskann_builder.h"
#include "algorithm/diskann/diskann_params.h"

namespace zvec::core_interface {
namespace {

class InspectableBufferedDiskAnn : public DiskAnnIndex {
 public:
  int initialize(const BaseIndexParam &param) {
    proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_THREAD_COUNT, 2U);
    return init(param);
  }

  bool buffered_build_requested() const {
    bool value = false;
    return proxima_index_params_.get(core::PARAM_DISKANN_BUILDER_BUFFERED_BUILD,
                                     &value) &&
           value;
  }

  std::shared_ptr<core::DiskAnnBuilder> builder() const {
    return std::dynamic_pointer_cast<core::DiskAnnBuilder>(builder_);
  }
};

BaseIndexParam::Pointer BuildParam(bool fp16, uint32_t dimension) {
  return DiskAnnIndexParamBuilder()
      .with_dimension(dimension)
      .with_metric_type(MetricType::kL2sq)
      .with_data_type(DataType::DT_FP32)
      .with_quantizer_param(
          QuantizerParam(fp16 ? QuantizerType::kFP16 : QuantizerType::kNone))
      .with_max_degree(16)
      .with_list_size(32)
      .with_pq_chunk_num(4)
      .build();
}

std::vector<float> BuildValues(uint32_t count, uint32_t dimension) {
  std::vector<float> values(static_cast<size_t>(count) * dimension);
  for (uint32_t id = 0; id < count; ++id) {
    for (uint32_t d = 0; d < dimension; ++d) {
      values[static_cast<size_t>(id) * dimension + d] =
          std::sin(static_cast<float>(id * 23 + d * 17 + 1)) +
          0.01f * static_cast<float>(id % 11);
    }
  }
  return values;
}

std::string ReadBytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

class DiskAnnBufferedBuildTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kDocCount = 256;
  const std::filesystem::path directory_{"diskann_buffered_build_test_data"};
  size_t previous_capacity_{0};
  size_t writable_metadata_{0};
  size_t pool_budget_{0};

  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_capacity_ = pool.capacity();
    // A writable store owns 128 writeback pages in addition to its page
    // table and mutexes. In particular, 16 KiB host pages require 2 MiB of
    // staging per store, so a fixed 8 MiB total is not portable.
    writable_metadata_ =
        ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
    pool_budget_ = std::max<size_t>(
        8UL * 1024UL * 1024UL, 3 * writable_metadata_ + 2UL * 1024UL * 1024UL);
    ASSERT_EQ(pool.used(), 0u);
    ASSERT_EQ(pool.init(pool_budget_), 0);
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    ASSERT_FALSE(error);
    ASSERT_TRUE(std::filesystem::create_directory(directory_, error));
    ASSERT_FALSE(error);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous_capacity_), 0);
  }

  std::vector<std::filesystem::path> scratch_directories() const {
    std::vector<std::filesystem::path> result;
    for (const auto &entry : std::filesystem::directory_iterator(directory_)) {
      if (entry.path().filename().string().find(".diskann-build-") !=
          std::string::npos) {
        result.push_back(entry.path());
      }
    }
    return result;
  }

  void add_all(const Index::Pointer &index, const std::vector<float> &values,
               uint32_t dimension) {
    for (uint32_t id = 0; id < kDocCount; ++id) {
      ASSERT_EQ(
          index->add(VectorData{DenseVector{
                         values.data() + static_cast<size_t>(id) * dimension}},
                     id),
          0);
    }
  }

  void fetch_all(const Index::Pointer &index, const std::vector<float> &values,
                 uint32_t dimension, bool fp16) {
    ASSERT_EQ(index->get_doc_count(), kDocCount);
    for (uint32_t id = 0; id < kDocCount; ++id) {
      SCOPED_TRACE(id);
      VectorDataBuffer result;
      // These core-interface document IDs are dense ordinals. In particular,
      // Flat merge preserves the source ordinal range, including any holes.
      ASSERT_EQ(index->fetch(id, &result), 0);
      const auto &bytes =
          std::get<DenseVectorBuffer>(result.vector_buffer).data;
      ASSERT_EQ(bytes.size(), static_cast<size_t>(dimension) * sizeof(float));
      for (uint32_t d = 0; d < dimension; ++d) {
        float actual = 0;
        std::memcpy(&actual, bytes.data() + d * sizeof(float), sizeof(actual));
        ASSERT_NEAR(actual, values[static_cast<size_t>(id) * dimension + d],
                    fp16 ? 0.002f : 0.00001f);
      }
    }
    VectorDataBuffer missing;
    EXPECT_NE(index->fetch(kDocCount, &missing), 0);
  }

  using QueryResult = std::vector<std::pair<uint64_t, float>>;
  void query(const Index::Pointer &index, const std::vector<float> &values,
             uint32_t dimension, uint32_t id, QueryResult *output) {
    auto params = std::make_shared<DiskAnnQueryParam>();
    params->topk = 5;
    params->list_size = 64;
    SearchResult result;
    ASSERT_EQ(
        index->search(VectorData{DenseVector{
                          values.data() + static_cast<size_t>(id) * dimension}},
                      params, &result),
        0);
    ASSERT_EQ(result.doc_list_.size(), 5u);
    EXPECT_EQ(result.doc_list_[0].key(), id);
    output->clear();
    for (const auto &doc : result.doc_list_) {
      output->emplace_back(doc.key(), doc.score());
    }
  }

  void run_lifecycle(bool fp16, bool merge, bool pressure) {
    const uint32_t dimension = pressure ? 512 : 32;
    const auto values = BuildValues(kDocCount, dimension);
    const auto path = (directory_ / "target.index").string();
    const auto source_path = (directory_ / "source.index").string();
    auto param = BuildParam(fp16, dimension);
    auto inspected = std::make_shared<InspectableBufferedDiskAnn>();
    ASSERT_EQ(inspected->initialize(*param), 0);
    Index::Pointer target = inspected;
    ASSERT_EQ(
        target->open(path, {StorageOptions::StorageType::kBufferPool, true}),
        0);
    ASSERT_TRUE(inspected->buffered_build_requested());

    auto &pool = ailego::MemoryLimitPool::get_instance();
    size_t reservation = 0;
    auto release = ailego::ScopeGuard::Make([&]() {
      if (reservation != 0) pool.release_external(reservation);
    });
    if (pressure) {
      // Keep room for all three writable page tables (Flat source + vectors +
      // graph), but not the source plus build vector working set.
      const size_t available = 3 * writable_metadata_ + 256UL * 1024UL;
      const size_t resident_vector_bytes =
          static_cast<size_t>(kDocCount) * dimension *
          (fp16 ? sizeof(uint16_t) : sizeof(float));
      ASSERT_GT(values.size() * sizeof(float) + resident_vector_bytes,
                256UL * 1024UL);
      ASSERT_LT(available, pool_budget_);
      const size_t requested = pool_budget_ - available;
      ASSERT_TRUE(pool.try_charge_external(requested));
      reservation = requested;
    }

    Index::Pointer source;
    if (merge) {
      auto flat_param = FlatIndexParamBuilder()
                            .with_dimension(dimension)
                            .with_metric_type(MetricType::kL2sq)
                            .with_data_type(DataType::DT_FP32)
                            .build();
      source = IndexFactory::CreateAndInitIndex(*flat_param);
      ASSERT_NE(source, nullptr);
      ASSERT_EQ(source->open(source_path,
                             {StorageOptions::StorageType::kBufferPool, true}),
                0);
    }
    ASSERT_NO_FATAL_FAILURE(
        add_all(merge ? source : target, values, dimension));
    ASSERT_EQ(
        merge ? target->merge({source}, {}, {2, nullptr}) : target->train(), 0);
    if (pressure) {
      // Timing determines whether foreground admission or background
      // writeback reclaims a page. Assert the fixed reservation and budget,
      // then validate every result, rather than requiring one reclaim path.
      EXPECT_EQ(pool.external_used(), reservation);
      EXPECT_LE(pool.used(), pool_budget_);
    }
    EXPECT_TRUE(target->is_trained());
    EXPECT_TRUE(scratch_directories().empty());
    ASSERT_NO_FATAL_FAILURE(fetch_all(target, values, dimension, fp16));
    QueryResult built_result;
    ASSERT_NO_FATAL_FAILURE(
        query(target, values, dimension, 73, &built_result));
    if (source) {
      ASSERT_EQ(source->close(), 0);
      source.reset();
    }
    ASSERT_EQ(target->close(), 0);
    target.reset();
    inspected.reset();
    EXPECT_TRUE(scratch_directories().empty());

    // Reopen exactly the same bytes through both readers: no rebuild may
    // obscure a storage-dependent result difference.
    for (auto mode : {StorageOptions::StorageType::kBufferPool,
                      StorageOptions::StorageType::kMMAP}) {
      auto reopened = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(reopened, nullptr);
      ASSERT_EQ(reopened->open(path, {mode, false, true}), 0);
      ASSERT_NO_FATAL_FAILURE(fetch_all(reopened, values, dimension, fp16));
      QueryResult reopened_result;
      ASSERT_NO_FATAL_FAILURE(
          query(reopened, values, dimension, 73, &reopened_result));
      EXPECT_EQ(reopened_result, built_result);
      ASSERT_EQ(reopened->close(), 0);
    }
    EXPECT_TRUE(scratch_directories().empty());
    ASSERT_TRUE(std::filesystem::remove(path));
    if (merge) {
      ASSERT_TRUE(std::filesystem::remove(source_path));
    }
  }
};

TEST_F(DiskAnnBufferedBuildTest, DirectAndMergedFp16AndFp32RoundTrip) {
  for (bool fp16 : {false, true}) {
    for (bool merge : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "fp16=" << fp16 << " merge=" << merge);
      ASSERT_NO_FATAL_FAILURE(run_lifecycle(fp16, merge, false));
    }
  }
}

TEST_F(DiskAnnBufferedBuildTest, MergedBuildSurvivesPagePressure) {
  for (bool fp16 : {false, true}) {
    SCOPED_TRACE(fp16);
    ASSERT_NO_FATAL_FAILURE(run_lifecycle(fp16, true, true));
  }
}

TEST_F(DiskAnnBufferedBuildTest, FailedDumpRetainsScratchAndRetryCleansIt) {
  for (bool fp16 : {false, true}) {
    SCOPED_TRACE(fp16);
    constexpr uint32_t kDimension = 32;
    auto param = BuildParam(fp16, kDimension);
    const auto values = BuildValues(kDocCount, kDimension);
    const auto output = directory_ / "blocked.index";
    ASSERT_TRUE(std::filesystem::create_directory(output));
    const auto marker = output / "do-not-remove";
    std::ofstream(marker) << "unrelated output-directory content";
    auto inspected = std::make_shared<InspectableBufferedDiskAnn>();
    ASSERT_EQ(inspected->initialize(*param), 0);
    Index::Pointer target = inspected;
    ASSERT_EQ(target->open(output.string(),
                           {StorageOptions::StorageType::kBufferPool, true}),
              0);
    ASSERT_NO_FATAL_FAILURE(add_all(target, values, kDimension));
    ASSERT_NE(target->train(), 0);
    EXPECT_FALSE(target->is_trained());
    auto builder = inspected->builder();
    ASSERT_NE(builder, nullptr);
    ASSERT_TRUE(builder->buffered_build());
    EXPECT_EQ(builder->stats().built_count(), kDocCount);
    const auto scratch = scratch_directories();
    ASSERT_EQ(scratch.size(), 1u);
    EXPECT_TRUE(std::filesystem::exists(scratch[0] / "graph"));
    EXPECT_FALSE(std::filesystem::exists(scratch[0] / "vectors"));
    const auto codes = ReadBytes(scratch[0] / "codes");
    ASSERT_FALSE(codes.empty());
    EXPECT_EQ(ReadBytes(marker), "unrelated output-directory content");

    EXPECT_NE(target->train(), 0);
    EXPECT_EQ(inspected->builder(), builder);
    EXPECT_EQ(ReadBytes(scratch[0] / "codes"), codes);
    EXPECT_TRUE(std::filesystem::exists(marker));
    ASSERT_TRUE(std::filesystem::remove(marker));
    ASSERT_TRUE(std::filesystem::remove(output));
    std::weak_ptr<core::DiskAnnBuilder> previous_builder = builder;
    builder.reset();
    ASSERT_EQ(target->train(), 0);
    EXPECT_TRUE(previous_builder.expired());
    EXPECT_FALSE(std::filesystem::exists(scratch[0]));
    ASSERT_NO_FATAL_FAILURE(fetch_all(target, values, kDimension, fp16));
    ASSERT_EQ(target->close(), 0);
    target.reset();
    inspected.reset();
    EXPECT_TRUE(scratch_directories().empty());
    ASSERT_TRUE(std::filesystem::remove(output));
  }
}

TEST_F(DiskAnnBufferedBuildTest, AbandonedFailedDumpCleansOwnedScratchOnly) {
  constexpr uint32_t kDimension = 32;
  const auto output = directory_ / "abandoned.index";
  ASSERT_TRUE(std::filesystem::create_directory(output));
  const auto marker = output / "keep";
  std::ofstream(marker) << "keep";
  {
    auto inspected = std::make_shared<InspectableBufferedDiskAnn>();
    ASSERT_EQ(inspected->initialize(*BuildParam(false, kDimension)), 0);
    Index::Pointer target = inspected;
    ASSERT_EQ(target->open(output.string(),
                           {StorageOptions::StorageType::kBufferPool, true}),
              0);
    const auto values = BuildValues(kDocCount, kDimension);
    ASSERT_NO_FATAL_FAILURE(add_all(target, values, kDimension));
    ASSERT_NE(target->train(), 0);
    ASSERT_TRUE(inspected->builder()->buffered_build());
    ASSERT_FALSE(scratch_directories().empty());
  }
  EXPECT_TRUE(scratch_directories().empty());
  EXPECT_EQ(ReadBytes(marker), "keep");
}

}  // namespace
}  // namespace zvec::core_interface
#endif  // DISKANN_SUPPORTED
