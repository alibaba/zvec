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

#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/ivf/ivf_params.h"
#include "utility/temporary_buffer_storage.h"

namespace zvec::core_interface {
namespace {

class InspectableBufferedIVF : public IVFIndex {
 public:
  int initialize(const BaseIndexParam &param) {
    proxima_index_params_.set(core::PARAM_IVF_BUILDER_THREAD_COUNT, 2U);
    return init(param);
  }
  bool buffered_build_requested() const {
    return !proxima_index_params_
                .get_as_string(core::PARAM_IVF_BUILDER_BUILD_STORAGE_PATH)
                .empty();
  }
};

class IVFBufferedBuildTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kCount = 256;
  static constexpr uint32_t kDimension = 512;
  const std::filesystem::path directory_{"ivf_buffered_build_data"};
  size_t capacity_{0};
  size_t writable_metadata_{0};
  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    writable_metadata_ =
        ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
    // Two scratch stores may coexist with a writable Flat source. The data
    // headroom is smaller than either FP32 matrix, without starving metadata.
    ASSERT_EQ(pool.init(3 * writable_metadata_ + 128 * 1024), 0);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }
  void TearDown() override {
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    EXPECT_TRUE(std::filesystem::remove(directory_));
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(capacity_), 0);
  }
  std::vector<float> values() {
    std::vector<float> result(kCount * kDimension);
    for (uint32_t id = 0; id < kCount; ++id) {
      for (uint32_t d = 0; d < kDimension; ++d) {
        result[id * kDimension + d] =
            std::sin(static_cast<float>(id * 11 + d * 3)) * 0.1f;
      }
    }
    return result;
  }
  void verify(const Index::Pointer &index, const std::vector<float> &data,
              bool fp16) {
    ASSERT_EQ(index->get_doc_count(), kCount);
    for (uint32_t id = 0; id < kCount; ++id) {
      VectorDataBuffer output;
      ASSERT_EQ(index->fetch(id, &output), 0);
      const auto &bytes =
          std::get<DenseVectorBuffer>(output.vector_buffer).data;
      ASSERT_EQ(bytes.size(), kDimension * sizeof(float));
      for (uint32_t d = 0; d < kDimension; ++d) {
        float actual;
        std::memcpy(&actual, bytes.data() + d * sizeof(float), sizeof(float));
        ASSERT_NEAR(actual, data[id * kDimension + d], fp16 ? 0.0001f : 0.0f);
      }
    }
    auto query = std::make_shared<IVFQueryParam>();
    query->topk = 5;
    query->nprobe = 8;
    SearchResult result;
    ASSERT_EQ(
        index->search(VectorData{DenseVector{data.data()}}, query, &result), 0);
    ASSERT_EQ(result.doc_list_.size(), 5u);
    EXPECT_EQ(result.doc_list_[0].key(), 0u);
  }
  void run(bool fp16, bool merge, bool pressure = false) {
    auto param = IVFIndexParamBuilder()
                     .with_dimension(kDimension)
                     .with_metric_type(MetricType::kL2sq)
                     .with_data_type(DataType::DT_FP32)
                     .with_quantizer_param(QuantizerParam(
                         fp16 ? QuantizerType::kFP16 : QuantizerType::kNone))
                     .with_n_list(8)
                     .with_n_iters(2)
                     .build();
    auto target_impl = std::make_shared<InspectableBufferedIVF>();
    ASSERT_EQ(target_impl->initialize(*param), 0);
    Index::Pointer target = target_impl;
    const auto path = (directory_ / "target").string();
    ASSERT_EQ(
        target->open(path, {StorageOptions::StorageType::kBufferPool, true}),
        0);
    EXPECT_TRUE(target_impl->buffered_build_requested());
    auto &pool = ailego::MemoryLimitPool::get_instance();
    size_t reservation = 0;
    auto release = ailego::ScopeGuard::Make([&]() {
      if (reservation != 0) pool.release_external(reservation);
    });
    size_t training_metadata = 0;
    if (pressure) {
      // Measure the actual metadata charged by an identically sized scratch
      // file, including BufferStorage's headers and file-growth rounding.
      const size_t matrix_bytes = static_cast<size_t>(kCount) * kDimension *
                                  (fp16 ? sizeof(uint16_t) : sizeof(float));
      ASSERT_GT(matrix_bytes, 128u * 1024u);
      const size_t before = pool.metadata_used();
      core::TemporaryBufferStorage::Pointer probe;
      ASSERT_EQ(
          core::TemporaryBufferStorage::Create(
              (directory_ / "sizing-probe").string(), matrix_bytes, &probe),
          0);
      training_metadata = pool.metadata_used() - before;
      ASSERT_GT(training_metadata, 0u);
      probe.reset();
      ASSERT_EQ(pool.metadata_used(), before);
    }
    Index::Pointer source;
    const auto source_path = (directory_ / "source").string();
    if (merge) {
      auto flat = FlatIndexParamBuilder()
                      .with_dimension(kDimension)
                      .with_metric_type(MetricType::kL2sq)
                      .with_data_type(DataType::DT_FP32)
                      .build();
      source = IndexFactory::CreateAndInitIndex(*flat);
      ASSERT_NE(source, nullptr);
      ASSERT_EQ(source->open(source_path,
                             {StorageOptions::StorageType::kBufferPool, true}),
                0);
    }
    const auto data = values();
    for (uint32_t id = 0; id < kCount; ++id) {
      ASSERT_EQ(
          (merge ? source : target)
              ->add(VectorData{DenseVector{data.data() + id * kDimension}}, id),
          0);
    }
    if (pressure) {
      // Use clean source pages so the setup reservation need not race dirty
      // writeback; training itself still competes with this source for pages.
      if (source) ASSERT_EQ(source->flush(), 0);
      const size_t retained_metadata = pool.metadata_used();
      ASSERT_GT(pool.capacity(),
                retained_metadata + training_metadata + 128 * 1024);
      const size_t amount =
          pool.capacity() - retained_metadata - training_metadata - 128 * 1024;
      ASSERT_TRUE(pool.try_charge_external(amount));
      reservation = amount;
      ASSERT_EQ(pool.capacity() - pool.external_used() - pool.metadata_used() -
                    training_metadata,
                128u * 1024u);
    }
    ASSERT_EQ(
        merge ? target->merge({source}, {}, {2, nullptr}) : target->train(), 0);
    ASSERT_NO_FATAL_FAILURE(verify(target, data, fp16));
    EXPECT_LE(ailego::MemoryLimitPool::get_instance().used(),
              ailego::MemoryLimitPool::get_instance().capacity());
    if (source) {
      ASSERT_EQ(source->close(), 0);
      source.reset();
      ASSERT_TRUE(std::filesystem::remove(source_path));
    }
    ASSERT_EQ(target->close(), 0);
    target.reset();
    target_impl.reset();
    for (auto mode : {StorageOptions::StorageType::kMMAP,
                      StorageOptions::StorageType::kBufferPool}) {
      auto reopened = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_EQ(reopened->open(path, {mode, false, true}), 0);
      ASSERT_NO_FATAL_FAILURE(verify(reopened, data, fp16));
      ASSERT_EQ(reopened->close(), 0);
    }
    ASSERT_TRUE(std::filesystem::remove(path));
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  }
};

TEST_F(IVFBufferedBuildTest, DirectAndMergedFp16AndFp32RoundTrip) {
  for (bool fp16 : {false, true}) {
    for (bool merge : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "fp16=" << fp16 << " merge=" << merge);
      ASSERT_NO_FATAL_FAILURE(run(fp16, merge));
    }
  }
}

TEST_F(IVFBufferedBuildTest, DirectAndMergedBuildsSurvivePagePressure) {
  for (bool fp16 : {false, true}) {
    for (bool merge : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "fp16=" << fp16 << " merge=" << merge);
      ASSERT_NO_FATAL_FAILURE(run(fp16, merge, true));
    }
  }
}

TEST_F(IVFBufferedBuildTest, MmapLeavesBufferedTrainingDisabled) {
  auto param = IVFIndexParamBuilder()
                   .with_dimension(8)
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .build();
  auto target = std::make_shared<InspectableBufferedIVF>();
  ASSERT_EQ(target->initialize(*param), 0);
  Index::Pointer index = target;
  ASSERT_EQ(index->open((directory_ / "mmap").string(),
                        {StorageOptions::StorageType::kMMAP, true}),
            0);
  EXPECT_FALSE(target->buffered_build_requested());
  ASSERT_EQ(index->close(), 0);
}

TEST_F(IVFBufferedBuildTest, FailedDumpPreservesInputForFetchAndRetry) {
  auto param = IVFIndexParamBuilder()
                   .with_dimension(kDimension)
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
                   .with_n_list(8)
                   .with_n_iters(2)
                   .build();
  auto inspected = std::make_shared<InspectableBufferedIVF>();
  ASSERT_EQ(inspected->initialize(*param), 0);
  Index::Pointer target = inspected;
  const auto path = directory_ / "blocked-output";
  ASSERT_TRUE(std::filesystem::create_directory(path));
  ASSERT_EQ(target->open(path.string(),
                         {StorageOptions::StorageType::kBufferPool, true}),
            0);
  const auto data = values();
  for (uint32_t id = 0; id < kCount; ++id) {
    ASSERT_EQ(
        target->add(VectorData{DenseVector{data.data() + id * kDimension}}, id),
        0);
  }
  ASSERT_NE(target->train(), 0);
  EXPECT_FALSE(target->is_trained());
  VectorDataBuffer output;
  ASSERT_EQ(target->fetch(7, &output), 0);
  const auto &bytes = std::get<DenseVectorBuffer>(output.vector_buffer).data;
  ASSERT_EQ(bytes.size(), kDimension * sizeof(float));
  EXPECT_EQ(
      std::memcmp(bytes.data(), data.data() + 7 * kDimension, bytes.size()), 0);
  EXPECT_NE(target->add(VectorData{DenseVector{data.data()}}, kCount), 0);
  ASSERT_TRUE(std::filesystem::remove(path));
  ASSERT_EQ(target->train(), 0);
  ASSERT_NO_FATAL_FAILURE(verify(target, data, true));
  ASSERT_EQ(target->close(), 0);
  target.reset();
  inspected.reset();
  ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_F(IVFBufferedBuildTest, NestedUtf8OutputCreatesParentsAndCleansScratch) {
  auto param = IVFIndexParamBuilder()
                   .with_dimension(kDimension)
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_n_list(8)
                   .with_n_iters(2)
                   .build();
  auto inspected = std::make_shared<InspectableBufferedIVF>();
  ASSERT_EQ(inspected->initialize(*param), 0);
  Index::Pointer target = inspected;
  const auto parent = directory_ / std::filesystem::u8path(u8"嵌套");
  const auto nested = parent / std::filesystem::u8path(u8"构建");
  const auto path = nested / "index";
  ASSERT_FALSE(std::filesystem::exists(parent));
  ASSERT_EQ(target->open(path.u8string(),
                         {StorageOptions::StorageType::kBufferPool, true}),
            0);
  // Opening remains lazy: it must not allocate any scratch or create dirs.
  EXPECT_FALSE(std::filesystem::exists(parent));
  const auto data = values();
  for (uint32_t id = 0; id < kCount; ++id) {
    ASSERT_EQ(
        target->add(VectorData{DenseVector{data.data() + id * kDimension}}, id),
        0);
  }
  ASSERT_EQ(target->train(), 0);
  ASSERT_NO_FATAL_FAILURE(verify(target, data, false));
  ASSERT_EQ(target->close(), 0);
  target.reset();
  inspected.reset();
  ASSERT_TRUE(std::filesystem::remove(path));
  EXPECT_TRUE(std::filesystem::is_empty(nested));
  ASSERT_TRUE(std::filesystem::remove(nested));
  EXPECT_TRUE(std::filesystem::is_empty(parent));
  ASSERT_TRUE(std::filesystem::remove(parent));
}

}  // namespace
}  // namespace zvec::core_interface
