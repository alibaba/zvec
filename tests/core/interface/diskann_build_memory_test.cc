// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/diskann/diskann_params.h"
#include "mixed_reducer/merged_provider_index_holder.h"
#include "tests/test_util.h"
#include "utility/ordinal_access_holder.h"

#if DISKANN_SUPPORTED
namespace zvec::core_interface {
namespace {
class InspectableDiskAnn : public DiskAnnIndex {
 public:
  int initialize(const BaseIndexParam &param) {
    proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_THREAD_COUNT, 2U);
    return init(param);
  }
  std::weak_ptr<core::IndexBuilder> builder() const {
    return builder_;
  }
  std::weak_ptr<core::IndexConverter> converter() const {
    return converter_;
  }
  core::IndexReformer::Pointer replace_reformer(
      core::IndexReformer::Pointer value) {
    return std::exchange(reformer_, std::move(value));
  }
  core::IndexHolder::Pointer converted() const {
    return converter_ ? converter_->result() : nullptr;
  }
};
auto parameter(bool fp16, MetricType metric = MetricType::kL2sq) {
  return DiskAnnIndexParamBuilder()
      .with_dimension(16)
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_quantizer_param(
          QuantizerParam(fp16 ? QuantizerType::kFP16 : QuantizerType::kNone))
      .with_max_degree(16)
      .with_list_size(20)
      .with_pq_chunk_num(4)
      .build();
}
std::vector<float> values(uint32_t id, size_t dim = 16) {
  std::vector<float> data(dim);
  for (size_t d = 0; d < dim; ++d) data[d] = std::sin(float(id * 13 + d + 1));
  return data;
}
TEST(DiskAnnBuildMemory, ReleaseStateAndReopenDirectAndMerged) {
  for (bool fp16 : {false, true}) {
    for (bool merge : {false, true}) {
      for (auto metric : {MetricType::kL2sq, MetricType::kCosine}) {
        SCOPED_TRACE(::testing::Message()
                     << fp16 << " " << merge << " " << int(metric));
        const std::string path = "diskann_memory_target",
                          source_path = "diskann_memory_source";
        test_util::RemoveTestFiles(path);
        test_util::RemoveTestFiles(source_path);
        auto param = parameter(fp16, metric);
        auto inspected = std::make_shared<InspectableDiskAnn>();
        ASSERT_EQ(0, inspected->initialize(*param));
        Index::Pointer target = inspected;
        auto converter = inspected->converter();
        ASSERT_EQ(
            0, target->open(path, {StorageOptions::StorageType::kMMAP, true}));
        auto flat = FlatIndexParamBuilder()
                        .with_dimension(16)
                        .with_metric_type(metric)
                        .with_data_type(DataType::DT_FP32)
                        .build();
        auto source = IndexFactory::CreateAndInitIndex(*flat);
        ASSERT_EQ(0, source->open(source_path,
                                  {StorageOptions::StorageType::kMMAP, true}));
        for (uint32_t id = 0; id < 64; ++id) {
          auto data = values(id);
          ASSERT_EQ(0, (merge ? source : target)
                           ->add(VectorData{DenseVector{data.data()}}, id));
        }
        ASSERT_EQ(0, merge ? target->merge({source}, {}) : target->train());
        EXPECT_TRUE(converter.expired());
        EXPECT_EQ(nullptr, inspected->converted());
        EXPECT_EQ(0U, inspected->builder().lock()->stats().built_count());
        EXPECT_EQ(64U, target->get_doc_count());
        std::weak_ptr<core::IndexStreamer> source_state =
            source->index_searcher();
        ASSERT_EQ(0, source->close());
        source.reset();
        EXPECT_TRUE(source_state.expired());
        EXPECT_EQ(0, target->train());
        EXPECT_EQ(0, target->merge({}, {}));
        VectorDataBuffer before;
        ASSERT_EQ(0, target->fetch(7, &before));
        const auto data = values(7);
        const auto &bytes =
            std::get<DenseVectorBuffer>(before.vector_buffer).data;
        ASSERT_EQ(data.size() * sizeof(float), bytes.size());
        for (size_t d = 0; d < data.size(); ++d) {
          EXPECT_NEAR(data[d], reinterpret_cast<const float *>(bytes.data())[d],
                      fp16 ? 0.002 : 0.00001);
        }
        ASSERT_EQ(0, target->close());
        target = IndexFactory::CreateAndInitIndex(*param);
        ASSERT_EQ(0, target->open(path, {StorageOptions::StorageType::kMMAP,
                                         false, true}));
        VectorDataBuffer after;
        ASSERT_EQ(0, target->fetch(7, &after));
        EXPECT_EQ(bytes, std::get<DenseVectorBuffer>(after.vector_buffer).data);
        auto query = std::make_shared<DiskAnnQueryParam>();
        query->topk = 5;
        SearchResult result;
        ASSERT_EQ(0, target->search(VectorData{DenseVector{data.data()}}, query,
                                    &result));
        ASSERT_FALSE(result.doc_list_.empty());
        EXPECT_EQ(7U, result.doc_list_[0].key());
        ASSERT_EQ(0, target->close());
        test_util::RemoveTestFiles(path);
        test_util::RemoveTestFiles(source_path);
      }
    }
  }
}
TEST(DiskAnnBuildMemory, FailedDumpRetainsInputAndRetries) {
  for (bool fp16 : {false, true}) {
    const std::string parent = "diskann_memory_blocker";
    test_util::RemoveTestFiles(parent);
    std::ofstream(parent).close();
    auto inspected = std::make_shared<InspectableDiskAnn>();
    ASSERT_EQ(0, inspected->initialize(*parameter(fp16)));
    Index::Pointer target = inspected;
    ASSERT_EQ(0, target->open(parent + "/index",
                              {StorageOptions::StorageType::kMMAP, true}));
    std::string expected;
    for (uint32_t id = 0; id < 32; ++id) {
      auto data = values(id);
      if (id == 7)
        expected.assign(reinterpret_cast<const char *>(data.data()),
                        16 * sizeof(float));
      ASSERT_EQ(0, target->add(VectorData{DenseVector{data.data()}}, id * 2));
    }
    EXPECT_NE(0, target->train());
    auto builder = inspected->builder();
    auto converted = inspected->converted();
    EXPECT_EQ(32U, builder.lock()->stats().built_count());
    EXPECT_FALSE(target->is_trained());
    VectorDataBuffer fetched;
    ASSERT_EQ(0, target->fetch(14, &fetched));
    auto data = values(7);
    EXPECT_EQ(expected,
              std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
    EXPECT_NE(0, target->fetch(15, &fetched));
    EXPECT_NE(0, target->add(VectorData{DenseVector{data.data()}}, 65));
    EXPECT_NE(0, target->train());
    EXPECT_FALSE(builder.expired());
    EXPECT_EQ(converted, inspected->converted());
    converted.reset();
    test_util::RemoveTestFiles(parent);
    ASSERT_EQ(0, target->train());
    EXPECT_TRUE(builder.expired());
    ASSERT_EQ(0, target->fetch(14, &fetched));
    ASSERT_EQ(0, target->close());
    test_util::RemoveTestFiles(parent);
  }
}

TEST(DiskAnnBuildMemory, FailedOpenRetriesWithoutRebuildingOrRedumping) {
  class FailingLoad : public core::IndexReformer {
   public:
    int init(const ailego::Params &) override {
      return 0;
    }
    int cleanup() override {
      return 0;
    }
    int unload() override {
      return 0;
    }
    int load(core::IndexStorage::Pointer) override {
      return core::IndexError_ReadData;
    }
  };
  const std::string path = "diskann_memory_open_retry";
  test_util::RemoveTestFiles(path);
  auto inspected = std::make_shared<InspectableDiskAnn>();
  ASSERT_EQ(0, inspected->initialize(*parameter(true)));
  auto reformer = inspected->replace_reformer(std::make_shared<FailingLoad>());
  auto conversion = inspected->converter();
  Index::Pointer target = inspected;
  ASSERT_EQ(0, target->open(path, {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t id = 0; id < 32; ++id) {
    auto data = values(id);
    ASSERT_EQ(0, target->add(VectorData{DenseVector{data.data()}}, id));
  }
  EXPECT_EQ(core::IndexError_ReadData, target->train());
  EXPECT_FALSE(target->is_trained());
  EXPECT_FALSE(conversion.expired());
  auto replacement = inspected->builder();
  EXPECT_EQ(0U, replacement.lock()->stats().built_count());
  std::ifstream file_before(path, std::ios::binary);
  const std::string before((std::istreambuf_iterator<char>(file_before)), {});
  ASSERT_FALSE(before.empty());
  file_before.close();
  inspected->replace_reformer(std::move(reformer));
  ASSERT_EQ(0, target->train());
  EXPECT_TRUE(target->is_trained());
  EXPECT_FALSE(replacement.expired());
  EXPECT_TRUE(conversion.expired());
  std::ifstream file_after(path, std::ios::binary);
  const std::string after((std::istreambuf_iterator<char>(file_after)), {});
  EXPECT_EQ(before, after);
  file_after.close();
  VectorDataBuffer fetched;
  ASSERT_EQ(0, target->fetch(7, &fetched));
  ASSERT_EQ(0, target->close());
  test_util::RemoveTestFiles(path);
}

class FailingOrdinalHolder : public core::IndexHolder,
                             public core::OrdinalAccessHolder {
 public:
  explicit FailingOrdinalHolder(core::RandomAccessIndexHolder::Pointer source)
      : source_(std::move(source)), fail_(std::make_shared<bool>(false)) {}
  size_t count() const override {
    return source_->count();
  }
  size_t dimension() const override {
    return source_->dimension();
  }
  core::IndexMeta::DataType data_type() const override {
    return source_->data_type();
  }
  size_t element_size() const override {
    return source_->element_size();
  }
  bool multipass() const override {
    return true;
  }
  core::IndexHolder::Iterator::Pointer create_iterator() override {
    return source_->create_iterator();
  }
  void fail(bool value) {
    *fail_ = value;
  }
  class Reader : public core::OrdinalAccessHolder::Reader {
   public:
    Reader(core::RandomAccessIndexHolder::Pointer source,
           std::shared_ptr<bool> fail)
        : source_(std::move(source)), fail_(std::move(fail)) {}
    int read(size_t ordinal, uint64_t *key, const void **data) override {
      if (!key || !data) return core::IndexError_InvalidArgument;
      *data = nullptr;
      if (ordinal >= source_->count()) return core::IndexError_OutOfRange;
      if (*fail_ && ordinal == 1) return core::IndexError_ReadData;
      *key = source_->key(ordinal);
      *data = source_->element(ordinal);
      return 0;
    }
    void reset() override {}

   private:
    core::RandomAccessIndexHolder::Pointer source_;
    std::shared_ptr<bool> fail_;
  };
  int create_ordinal_reader(
      core::OrdinalAccessHolder::Reader::Pointer *reader) override {
    if (!reader) return core::IndexError_InvalidArgument;
    *reader = std::make_unique<Reader>(source_, fail_);
    return 0;
  }

 private:
  core::RandomAccessIndexHolder::Pointer source_;
  std::shared_ptr<bool> fail_;
};
TEST(DiskAnnBuildMemory, DeferredSourceReadFailureAndRepeatedDump) {
  // Exercise both single-sector and multi-sector node layouts.
  for (size_t dim : {16U, 1536U}) {
    core::IndexMeta meta(core::IndexMeta::DataType::DT_FP32, dim);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto source = std::make_shared<core::RandomAccessIndexHolder>(meta);
    for (uint64_t id = 0; id < 16; ++id) {
      auto data = values(id, dim);
      source->emplace(id * 3, data.data());
    }
    auto holder = std::make_shared<FailingOrdinalHolder>(source);
    auto builder = core::IndexFactory::CreateBuilder("DiskAnnBuilder");
    ailego::Params params;
    params.set(core::PARAM_DISKANN_BUILDER_THREAD_COUNT, 2U);
    params.set(core::PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 4U);
    ASSERT_EQ(0, builder->init(meta, params));
    ASSERT_EQ(0, builder->train(holder));
    ASSERT_EQ(0, builder->build(holder));
    const std::string path = "diskann_memory_deferred";
    for (bool fail : {true, false, false}) {
      holder->fail(fail);
      auto dumper = core::IndexFactory::CreateDumper("FileDumper");
      ASSERT_EQ(0, dumper->create(path));
      if (fail)
        EXPECT_EQ(core::IndexError_ReadData, builder->dump(dumper));
      else
        EXPECT_EQ(0, builder->dump(dumper));
      int close_result = dumper->close();
      if (!fail) ASSERT_EQ(0, close_result);
    }
    ASSERT_EQ(0, builder->cleanup());
    test_util::RemoveTestFiles(path);
  }
}

class DeferredReadFailureReformer : public core::IndexReformer {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int load(core::IndexStorage::Pointer) override {
    return 0;
  }
  int unload() override {
    return 0;
  }
  int revert(const void *data, const core::IndexQueryMeta &meta,
             std::string *output) const override {
    if (fail && *static_cast<const float *>(data) == failed_value) {
      return core::IndexError_ReadData;
    }
    output->assign(static_cast<const char *>(data), meta.element_size());
    return 0;
  }
  bool fail{false};
  float failed_value{0.0F};
};

TEST(DiskAnnBuildMemory, DeferredDecodedSourceReadFailureDuringDump) {
  constexpr uint32_t kDocCount = 16;
  // Cover both node layouts, including failure on the final vector where no
  // subsequent iteration can notice the invalidated source iterator.
  for (size_t dim : {16U, 1536U}) {
    for (uint32_t failed_id : {0U, kDocCount - 1}) {
      SCOPED_TRACE(::testing::Message() << dim << " " << failed_id);
      const std::string source_path = "diskann_memory_decoded_source";
      const std::string path = "diskann_memory_decoded_target";
      test_util::RemoveTestFiles(source_path);
      test_util::RemoveTestFiles(path);
      auto source_param = FlatIndexParamBuilder()
                              .with_dimension(dim)
                              .with_metric_type(MetricType::kL2sq)
                              .with_data_type(DataType::DT_FP32)
                              .build();
      auto source = IndexFactory::CreateAndInitIndex(*source_param);
      ASSERT_NE(nullptr, source);
      ASSERT_EQ(0, source->open(source_path,
                                {StorageOptions::StorageType::kMMAP, true}));
      for (uint32_t id = 0; id < kDocCount; ++id) {
        auto data = values(id, dim);
        data[0] = static_cast<float>(id + 1);
        ASSERT_EQ(0, source->add(VectorData{DenseVector{data.data()}}, id));
      }

      auto reformer = std::make_shared<DeferredReadFailureReformer>();
      core::IndexQueryMeta query_meta(core::IndexMeta::DT_FP32, dim);
      core::MergedProviderIndexHolder::Source merged_source;
      merged_source.owner = source->index_searcher();
      merged_source.reformer = reformer;
      merged_source.provider_meta = query_meta;
      merged_source.need_revert = true;
      auto holder = std::make_shared<core::MergedProviderIndexHolder>(
          query_meta,
          std::vector<core::MergedProviderIndexHolder::Source>{merged_source});
      ASSERT_EQ(0, holder->init({}));
      core::OrdinalAccessHolder::Reader::Pointer reader;
      ASSERT_EQ(core::IndexError_NotImplemented,
                holder->create_ordinal_reader(&reader));

      core::IndexMeta meta(core::IndexMeta::DT_FP32, dim);
      meta.set_metric("SquaredEuclidean", 0, ailego::Params());
      ailego::Params params;
      params.set(core::PARAM_DISKANN_BUILDER_THREAD_COUNT, 2U);
      params.set(core::PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 4U);
      auto builder = core::IndexFactory::CreateBuilder("DiskAnnBuilder");
      ASSERT_NE(nullptr, builder);
      ASSERT_EQ(0, builder->init(meta, params));
      ASSERT_EQ(0, builder->train(holder));
      ASSERT_EQ(0, builder->build(holder));

      // First verify a successful dump through the iterator fallback. Then
      // fail decoding: MergedProviderIndexHolder returns a non-null zero
      // placeholder, so checking only data() for nullptr is insufficient.
      for (bool fail : {false, true}) {
        reformer->failed_value = static_cast<float>(failed_id + 1);
        reformer->fail = fail;
        auto dumper = core::IndexFactory::CreateDumper("FileDumper");
        ASSERT_NE(nullptr, dumper);
        ASSERT_EQ(0, dumper->create(path));
        EXPECT_EQ(fail ? core::IndexError_ReadData : 0, builder->dump(dumper));
        EXPECT_EQ(fail ? core::IndexError_ReadData : 0, holder->status());
        const int close_result = dumper->close();
        if (!fail) EXPECT_EQ(0, close_result);
      }
      ASSERT_EQ(0, builder->cleanup());
      holder.reset();
      ASSERT_EQ(0, source->close());
      test_util::RemoveTestFiles(source_path);
      test_util::RemoveTestFiles(path);
    }
  }
}
}  // namespace
}  // namespace zvec::core_interface
#endif
