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

#include "ivf_rabitq_streamer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <rabitqlib/quantization/data_layout.hpp>
#include "algorithm/hnsw_rabitq/rabitq_utils.h"
#include "zvec/ailego/container/params.h"
#include "zvec/ailego/logger/logger.h"
#include "zvec/ailego/utility/file_helper.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_holder.h"
#include "zvec/core/framework/index_memory.h"
#include "zvec/core/framework/index_streamer.h"
#include "ivf_rabitq_builder.h"
#include "ivf_rabitq_context.h"
#include "ivf_rabitq_entity.h"
#include "ivf_rabitq_params.h"
#include "ivf_rabitq_reformer.h"
#include "ivf_rabitq_util.h"
#include "rabitq_converter.h"
#include "rabitq_params.h"

using namespace std;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static kDim = 128;

class IvfRabitqStreamerTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string IvfRabitqStreamerTest::dir_("ivfRabitqStreamerTest");
shared_ptr<IndexMeta> IvfRabitqStreamerTest::index_meta_ptr_;

namespace {

void FillTestVector(size_t seed, NumericalVector<float> *vec) {
  ASSERT_NE(nullptr, vec);
  for (size_t j = 0; j < vec->dimension(); ++j) {
    size_t value = (seed + 1) * 1103515245ULL + (j + 1) * 12345ULL +
                   (seed + 17) * (j + 31);
    value ^= (value >> 11);
    (*vec)[j] = static_cast<float>(value % 1000003ULL) / 1000003.0f;
  }
}

IndexHolder::Pointer BuildHolder(size_t dim, size_t doc_cnt) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    FillTestVector(i, &vec);
    EXPECT_TRUE(holder->emplace(i, vec));
  }
  return holder;
}

IndexHolder::Pointer BuildDirectionalHolder(size_t dim, size_t doc_cnt) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  for (size_t i = 0; i < doc_cnt; ++i) {
    NumericalVector<float> vec(dim);
    float norm_sq = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = std::sin(static_cast<float>((i + 1) * (j + 3)) * 0.017f) +
               std::cos(static_cast<float>((i + 5) * (j + 1)) * 0.011f);
      norm_sq += vec[j] * vec[j];
    }
    const float inverse_norm = 1.0f / std::sqrt(norm_sq);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] *= inverse_norm;
    }
    EXPECT_TRUE(holder->emplace(i, vec));
  }
  return holder;
}

class TrackingIndexThreads : public IndexThreads {
 public:
  TrackingIndexThreads() : threads_(2, false) {}

  size_t count(void) const override {
    return threads_.count();
  }

  void stop(void) override {
    threads_.stop();
  }

  void submit(ailego::ClosureHandler &&task) override {
    threads_.submit(std::move(task));
  }

  TaskGroup::Pointer make_group(void) override {
    make_group_count_.fetch_add(1, std::memory_order_relaxed);
    return threads_.make_group();
  }

  int indexof_this(void) const override {
    return threads_.indexof_this();
  }

  size_t make_group_count(void) const {
    return make_group_count_.load(std::memory_order_relaxed);
  }

 private:
  SingleQueueIndexThreads threads_;
  std::atomic<size_t> make_group_count_{0};
};

void BuildIndexFile(const IndexMeta &meta, const IndexHolder::Pointer &holder,
                    const ailego::Params &params, const std::string &path) {
  auto builder = std::make_shared<IvfRabitqBuilder>();
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(nullptr, holder));
  ASSERT_EQ(0, builder->build(nullptr, holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
}

void BuildAndOpenStreamer(const IndexMeta &meta,
                          const IndexHolder::Pointer &holder,
                          const ailego::Params &params, const std::string &path,
                          IndexStreamer::Pointer *streamer) {
  ASSERT_NE(nullptr, streamer);
  BuildIndexFile(meta, holder, params, path);

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(path, false));

  *streamer = std::make_shared<IvfRabitqStreamer>();
  ASSERT_EQ(0, (*streamer)->init(meta, params));
  ASSERT_EQ(0, (*streamer)->open(storage));
}

void RunBuildSearchAndReopen(const IndexMeta &meta,
                             const IndexHolder::Pointer &holder,
                             const void *query,
                             const IndexQueryMeta &query_meta,
                             const ailego::Params &params,
                             const std::string &path) {
  BuildIndexFile(meta, holder, params, path);

  {
    IndexStreamer::Pointer streamer = std::make_shared<IvfRabitqStreamer>();
    ASSERT_EQ(0, streamer->init(meta, params));
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_EQ(0, streamer->open(storage));

    auto context = streamer->create_context();
    context->set_topk(10);
    ailego::Params search_params;
    search_params.set(PARAM_IVF_RABITQ_NPROBE, 16U);
    ASSERT_EQ(0, context->update(search_params));
    ASSERT_EQ(0, streamer->search_impl(query, query_meta, 1, context));
    EXPECT_GT(context->result(0).size(), 0UL);
    const auto first_result = context->result(0);
    ASSERT_EQ(0, streamer->search_impl(query, query_meta, 1, context));
    const auto &repeated_result = context->result(0);
    ASSERT_EQ(first_result.size(), repeated_result.size());
    for (size_t i = 0; i < first_result.size(); ++i) {
      EXPECT_EQ(first_result[i].key(), repeated_result[i].key());
      EXPECT_FLOAT_EQ(first_result[i].score(), repeated_result[i].score());
    }

    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());
  }

  {
    IndexStreamer::Pointer streamer = std::make_shared<IvfRabitqStreamer>();
    ASSERT_EQ(0, streamer->init(meta, params));
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_EQ(0, streamer->open(storage));

    auto context = streamer->create_context();
    context->set_topk(10);
    ailego::Params search_params;
    search_params.set(PARAM_IVF_RABITQ_NPROBE, 16U);
    ASSERT_EQ(0, context->update(search_params));
    ASSERT_EQ(0, streamer->search_impl(query, query_meta, 1, context));
    EXPECT_GT(context->result(0).size(), 0UL);
  }
}

void BuildLoadedReformer(const IndexMeta &rabitq_meta,
                         const std::string &metric_name,
                         const IndexHolder::Pointer &holder,
                         const std::string &file_id,
                         std::shared_ptr<IvfRabitqReformer> *out_reformer,
                         std::vector<float> *rotated_centroids = nullptr,
                         uint32_t num_clusters = 16U) {
  ASSERT_NE(nullptr, out_reformer);

  RabitqConverter converter;
  ailego::Params params;
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  params.set(PARAM_RABITQ_NUM_CLUSTERS, num_clusters);
  ASSERT_EQ(0, converter.init(rabitq_meta, params));
  ASSERT_EQ(0, converter.train(holder));

  auto memory_dumper = IndexFactory::CreateDumper("MemoryDumper");
  ASSERT_NE(nullptr, memory_dumper);
  ASSERT_EQ(0, memory_dumper->init(ailego::Params()));
  ASSERT_EQ(0, memory_dumper->create(file_id));
  ASSERT_EQ(0, converter.dump(memory_dumper));
  ASSERT_EQ(0, memory_dumper->close());

  auto reformer = std::make_shared<IvfRabitqReformer>();
  ASSERT_EQ(0, reformer->init(metric_name));
  auto memory_storage = IndexFactory::CreateStorage("MemoryReadStorage");
  ASSERT_NE(nullptr, memory_storage);
  ASSERT_EQ(0, memory_storage->open(file_id, false));
  ASSERT_EQ(0, reformer->load(memory_storage));
  if (rotated_centroids) {
    auto segment = memory_storage->get(RABITQ_CONVERTER_SEG_ID);
    ASSERT_NE(nullptr, segment);
    const void *header_data = nullptr;
    ASSERT_EQ(sizeof(RabitqConverterHeader),
              segment->read(0, &header_data, sizeof(RabitqConverterHeader)));
    const auto *header =
        static_cast<const RabitqConverterHeader *>(header_data);
    const size_t centroid_count =
        static_cast<size_t>(header->num_clusters) * header->padded_dim;
    const void *centroid_data = nullptr;
    ASSERT_EQ(centroid_count * sizeof(float),
              segment->read(sizeof(RabitqConverterHeader), &centroid_data,
                            centroid_count * sizeof(float)));
    const auto *centroids = static_cast<const float *>(centroid_data);
    rotated_centroids->assign(centroids, centroids + centroid_count);
  }
  IndexMemory::Instance()->remove(file_id);

  *out_reformer = reformer;
}

}  // namespace

void IvfRabitqStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, kDim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void IvfRabitqStreamerTest::TearDown(void) {
  ailego::FileHelper::RemovePath(dir_.c_str());
}

TEST_F(IvfRabitqStreamerTest, TestParamValidation) {
  auto init_builder = [&](const ailego::Params &params) {
    auto builder = std::make_shared<IvfRabitqBuilder>();
    return builder->init(*index_meta_ptr_, params);
  };

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 0);
  EXPECT_EQ(IndexError_InvalidArgument, init_builder(params));

  params.set(PARAM_IVF_RABITQ_NLIST, 1025);
  EXPECT_EQ(0, init_builder(params));

  params.set(PARAM_IVF_RABITQ_NLIST, 32);
  params.set(PARAM_RABITQ_SAMPLE_COUNT, -1);
  EXPECT_EQ(IndexError_InvalidArgument, init_builder(params));

  IvfRabitqContext context;
  ailego::Params search_params;
  search_params.set(PARAM_IVF_RABITQ_NPROBE, 1025);
  EXPECT_EQ(0, context.update(search_params));

  auto init_streamer = [&](const ailego::Params &streamer_params) {
    auto streamer = std::make_shared<IvfRabitqStreamer>();
    return streamer->init(*index_meta_ptr_, streamer_params);
  };
  EXPECT_EQ(0, init_streamer(search_params));

  search_params.set(PARAM_IVF_RABITQ_NPROBE, -1);
  EXPECT_EQ(IndexError_InvalidArgument, context.update(search_params));
  EXPECT_EQ(IndexError_InvalidArgument, init_streamer(search_params));

  search_params.set(PARAM_IVF_RABITQ_NPROBE, 0);
  EXPECT_EQ(0, init_streamer(search_params));
}

TEST_F(IvfRabitqStreamerTest, TestBuilderUsesProvidedThreads) {
  auto holder = BuildHolder(kDim, 256);
  auto threads = std::make_shared<TrackingIndexThreads>();
  auto builder = std::make_shared<IvfRabitqBuilder>();

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 8U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 1U);
  ASSERT_EQ(0, builder->init(*index_meta_ptr_, params));

  ASSERT_EQ(0, builder->train(threads, holder));
  size_t train_group_count = threads->make_group_count();
  EXPECT_GT(train_group_count, 0U);

  ASSERT_EQ(0, builder->build(threads, holder));
  EXPECT_GT(threads->make_group_count(), train_group_count);
}

TEST_F(IvfRabitqStreamerTest, TestBuilderBuildsWithConfiguredInternalThreads) {
  auto holder = BuildHolder(kDim, 256);
  auto builder = std::make_shared<IvfRabitqBuilder>();

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 8U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 1U);
  params.set(PARAM_IVF_RABITQ_BUILDER_THREAD_COUNT, 2U);
  ASSERT_EQ(0, builder->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, builder->train(nullptr, holder));
  ASSERT_EQ(0, builder->build(nullptr, holder));
}

TEST_F(IvfRabitqStreamerTest, TestRejectInvalidFormatMagicAndVersion) {
  auto holder = BuildHolder(kDim, 64);
  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 16U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 4U);

  auto builder = std::make_shared<IvfRabitqBuilder>();
  ASSERT_EQ(0, builder->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, builder->train(nullptr, holder));
  ASSERT_EQ(0, builder->build(nullptr, holder));

  const std::string file_id{"ivf_rabitq_invalid_format"};
  IndexMemory::Instance()->remove(file_id);
  auto dumper = IndexFactory::CreateDumper("MemoryDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->init(ailego::Params()));
  ASSERT_EQ(0, dumper->create(file_id));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(file_id, false));

  auto header_segment = storage->get(IVF_RABITQ_HEADER_SEG_ID);
  ASSERT_NE(nullptr, header_segment);
  const void *header_data = nullptr;
  ASSERT_EQ(sizeof(IvfRabitqHeader),
            header_segment->read(0, &header_data, sizeof(IvfRabitqHeader)));
  auto *header = const_cast<IvfRabitqHeader *>(
      static_cast<const IvfRabitqHeader *>(header_data));
  ASSERT_EQ(kIvfRabitqIndexMagic, header->magic);
  ASSERT_EQ(kIvfRabitqIndexVersion, header->version);

  header->magic = 0;
  IvfRabitqEntity invalid_magic_entity;
  EXPECT_EQ(IndexError_InvalidFormat, invalid_magic_entity.load(storage));

  header->magic = kIvfRabitqIndexMagic;
  header->version = kIvfRabitqIndexVersion + 1;
  IvfRabitqEntity invalid_version_entity;
  EXPECT_EQ(IndexError_InvalidFormat, invalid_version_entity.load(storage));

  ASSERT_EQ(0, storage->close());
  IndexMemory::Instance()->remove(file_id);
}

TEST_F(IvfRabitqStreamerTest, TestBuildAndSearch) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i * kDim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set("proxima.ivf_rabitq.nlist", 32U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestBuildAndSearch", &streamer);
  ASSERT_NE(nullptr, streamer);

  // Search
  NumericalVector<float> query_vec(kDim);
  for (size_t j = 0; j < kDim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  auto context = streamer->create_context();
  context->set_topk(10);
  ailego::Params search_params;
  search_params.set("proxima.ivf_rabitq.nprobe", 8U);
  context->update(search_params);

  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &result = context->result(0);
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);

  // The nearest vector should be key=0 (same direction as query)
  LOG_INFO("IVF RabitQ search returned %zu results, top key=%zu, score=%f",
           result.size(), (size_t)result[0].key(), result[0].score());
}

TEST_F(IvfRabitqStreamerTest, TestProviderVectorAccessUnsupported) {
  auto holder = BuildHolder(kDim, 300UL);

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 16U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 4U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestCreateProvider", &streamer);
  ASSERT_NE(nullptr, streamer);

  auto provider = streamer->create_provider();
  ASSERT_NE(nullptr, provider);
  EXPECT_EQ(holder->count(), provider->count());
  EXPECT_EQ(kDim, provider->dimension());
  EXPECT_EQ(IndexMeta::DataType::DT_UNDEFINED, provider->data_type());
  EXPECT_EQ(0UL, provider->element_size());

  size_t seen = 0;
  auto iter = provider->create_iterator();
  ASSERT_NE(nullptr, iter);
  while (iter->is_valid()) {
    EXPECT_EQ(nullptr, iter->data());
    ++seen;
    iter->next();
  }
  EXPECT_EQ(provider->count(), seen);

  EXPECT_EQ(nullptr, provider->get_vector(17));
  EXPECT_EQ(nullptr, streamer->get_vector_by_id(17));
  IndexStorage::MemoryBlock block;
  EXPECT_EQ(IndexError_Unsupported, streamer->get_vector_by_id(17, block));
  EXPECT_EQ(IndexError_Unsupported, provider->get_vector(17, block));
}

TEST_F(IvfRabitqStreamerTest, TestTotalBitsOneBuildSearchAndReopen) {
  auto holder = BuildHolder(kDim, 1000UL);
  NumericalVector<float> query_vec(kDim);
  FillTestVector(17, &query_vec);

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 32U);
  params.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, 0U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 1U);

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  RunBuildSearchAndReopen(*index_meta_ptr_, holder, query_vec.data(),
                          query_meta, params,
                          dir_ + "/TestTotalBitsOneBuildSearchAndReopen");
}

TEST_F(IvfRabitqStreamerTest, TestInnerProductBuildSearchAndReopen) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric("InnerProduct", 0, ailego::Params());

  auto holder = BuildHolder(kDim, 1000UL);
  NumericalVector<float> query_vec(kDim);
  FillTestVector(31, &query_vec);

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 32U);
  params.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, 0U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  RunBuildSearchAndReopen(meta, holder, query_vec.data(), query_meta, params,
                          dir_ + "/TestInnerProductBuildSearchAndReopen");
}

TEST_F(IvfRabitqStreamerTest, TestCosineBuildSearchAndReopen) {
  IndexMeta raw_meta(IndexMeta::DataType::DT_FP32, kDim);
  raw_meta.set_metric("Cosine", 0, ailego::Params());

  auto raw_holder = BuildHolder(kDim, 1000UL);
  auto converter = IndexFactory::CreateConverter("CosineNormalizeConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(raw_meta, ailego::Params()));
  IndexMeta transformed_meta = converter->meta();
  ASSERT_EQ(kDim + 1, transformed_meta.dimension());
  ASSERT_EQ(0, converter->transform(raw_holder));
  auto holder = converter->result();
  ASSERT_NE(nullptr, holder);

  NumericalVector<float> raw_query_vec(kDim);
  FillTestVector(43, &raw_query_vec);
  auto reformer =
      IndexFactory::CreateReformer(transformed_meta.reformer_name());
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(transformed_meta.reformer_params()));

  IndexQueryMeta raw_query_meta(IndexMeta::DataType::DT_FP32, kDim);
  IndexQueryMeta transformed_query_meta;
  std::string transformed_query;
  ASSERT_EQ(0,
            reformer->transform(raw_query_vec.data(), raw_query_meta,
                                &transformed_query, &transformed_query_meta));
  ASSERT_EQ(kDim + 1, transformed_query_meta.dimension());

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 32U);
  params.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, 0U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  params.set(PARAM_RABITQ_GENERAL_DIMENSION, static_cast<uint32_t>(kDim));

  RunBuildSearchAndReopen(transformed_meta, holder, transformed_query.data(),
                          transformed_query_meta, params,
                          dir_ + "/TestCosineBuildSearchAndReopen");
}

TEST_F(IvfRabitqStreamerTest, TestContextResetClearsSearchState) {
  IvfRabitqContext context;
  context.set_topk(3);
  context.set_fetch_vector(true);
  context.set_filter([](uint64_t key) { return key == 7; });
  context.set_threshold(0.5f);
  context.set_group_by([](uint64_t key) { return std::to_string(key % 3U); });
  context.set_group_params(/*group_num=*/3, /*group_topk=*/2);
  context.query_state.rotated_query.push_back(1.0f);
  context.query_state.query_norm_sq = 2.0f;
  context.query_state.batch_query = std::make_shared<int>(1);
  context.probe_centroids.push_back({3U, 2.0f, 1.0f});

  ASSERT_TRUE(context.fetch_vector());
  ASSERT_TRUE(context.filter().is_valid());
  ASSERT_TRUE(context.group_by_search());
  ASSERT_NE(std::numeric_limits<float>::max(), context.threshold());

  context.reset();

  EXPECT_FALSE(context.fetch_vector());
  EXPECT_FALSE(context.filter().is_valid());
  EXPECT_FALSE(context.group_by_search());
  EXPECT_FALSE(context.group_by().is_valid());
  EXPECT_EQ(std::numeric_limits<float>::max(), context.threshold());
  EXPECT_TRUE(context.query_state.rotated_query.empty());
  EXPECT_FLOAT_EQ(0.0f, context.query_state.query_norm_sq);
  EXPECT_EQ(nullptr, context.query_state.batch_query);
  EXPECT_TRUE(context.probe_centroids.empty());
}

TEST_F(IvfRabitqStreamerTest, TestProbeCentroidsUseAssignmentMetric) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric("InnerProduct", 0, ailego::Params());
  auto holder = BuildDirectionalHolder(kDim, 1000UL);

  std::shared_ptr<IvfRabitqReformer> reformer;
  BuildLoadedReformer(meta, "InnerProduct", holder,
                      dir_ + "/TestProbeCentroidsUseAssignmentMetric",
                      &reformer);
  ASSERT_NE(nullptr, reformer);

  NumericalVector<float> query_vec(kDim);
  FillTestVector(71, &query_vec);

  uint32_t assigned = reformer->find_nearest_centroid(query_vec.data());
  std::vector<uint32_t> probe_centroids;
  ASSERT_EQ(0, reformer->select_probe_centroids(query_vec.data(), 1,
                                                &probe_centroids));
  ASSERT_EQ(1UL, probe_centroids.size());
  EXPECT_EQ(assigned, probe_centroids[0]);
}

TEST_F(IvfRabitqStreamerTest, TestCosineProbeCentroidsUseInnerProductMetric) {
  IndexMeta raw_meta(IndexMeta::DataType::DT_FP32, kDim);
  raw_meta.set_metric("Cosine", 0, ailego::Params());
  auto raw_holder = BuildHolder(kDim, 1000UL);

  auto converter = IndexFactory::CreateConverter("CosineNormalizeConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(raw_meta, ailego::Params()));
  IndexMeta transformed_meta = converter->meta();
  ASSERT_EQ(0, converter->transform(raw_holder));
  auto holder = converter->result();
  ASSERT_NE(nullptr, holder);

  ailego::Params params;
  params.set(PARAM_RABITQ_GENERAL_DIMENSION, static_cast<uint32_t>(kDim));
  IndexMeta rabitq_meta;
  std::string metric_name;
  ASSERT_EQ(0, PrepareAndCheckIvfRabitqInternalMeta(
                   transformed_meta, params, &rabitq_meta, &metric_name));
  ASSERT_EQ("InnerProduct", metric_name);
  ASSERT_EQ(kDim, rabitq_meta.dimension());

  std::shared_ptr<IvfRabitqReformer> reformer;
  BuildLoadedReformer(rabitq_meta, metric_name, holder,
                      dir_ + "/TestCosineProbeCentroidsUseInnerProductMetric",
                      &reformer);
  ASSERT_NE(nullptr, reformer);

  NumericalVector<float> raw_query_vec(kDim);
  FillTestVector(79, &raw_query_vec);
  auto query_reformer =
      IndexFactory::CreateReformer(transformed_meta.reformer_name());
  ASSERT_NE(nullptr, query_reformer);
  ASSERT_EQ(0, query_reformer->init(transformed_meta.reformer_params()));

  IndexQueryMeta raw_query_meta(IndexMeta::DataType::DT_FP32, kDim);
  IndexQueryMeta transformed_query_meta;
  std::string transformed_query;
  ASSERT_EQ(0, query_reformer->transform(raw_query_vec.data(), raw_query_meta,
                                         &transformed_query,
                                         &transformed_query_meta));

  const float *query =
      reinterpret_cast<const float *>(transformed_query.data());
  uint32_t assigned = reformer->find_nearest_centroid(query);
  std::vector<uint32_t> probe_centroids;
  ASSERT_EQ(0, reformer->select_probe_centroids(query, 1, &probe_centroids));
  ASSERT_EQ(1UL, probe_centroids.size());
  EXPECT_EQ(assigned, probe_centroids[0]);
}

TEST_F(IvfRabitqStreamerTest,
       TestInnerProductProbeReusesCoarseScoreForCentroidPreparation) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric("InnerProduct", 0, ailego::Params());
  auto holder = BuildDirectionalHolder(kDim, 1000UL);

  std::shared_ptr<IvfRabitqReformer> reformer;
  std::vector<float> rotated_centroids;
  BuildLoadedReformer(
      meta, "InnerProduct", holder,
      dir_ + "/TestInnerProductProbeReusesCoarseScoreForCentroidPreparation",
      &reformer, &rotated_centroids, 4U);
  ASSERT_NE(nullptr, reformer);

  NumericalVector<float> query(kDim);
  FillTestVector(97, &query);
  for (size_t i = 0; i < query.dimension(); ++i) {
    query[i] *= 3.0f;
  }
  IvfRabitqQueryState state;
  ASSERT_EQ(0, reformer->create_query_state(query.data(), &state));

  std::vector<IvfRabitqProbeCentroid> probes;
  ASSERT_EQ(0,
            reformer->select_probe_centroids(query.data(), 3, &state, &probes));
  ASSERT_EQ(3UL, probes.size());

  std::vector<std::pair<float, uint32_t>> expected;
  const size_t centroid_count =
      rotated_centroids.size() / reformer->padded_dim();
  expected.reserve(centroid_count);
  for (size_t id = 0; id < centroid_count; ++id) {
    const float *centroid =
        rotated_centroids.data() + id * reformer->padded_dim();
    float inner_product = 0.0f;
    for (size_t i = 0; i < reformer->padded_dim(); ++i) {
      inner_product += state.rotated_query[i] * centroid[i];
    }
    expected.emplace_back(-inner_product, static_cast<uint32_t>(id));
  }
  std::sort(expected.begin(), expected.end());

  for (const auto &probe : probes) {
    const float *centroid =
        rotated_centroids.data() + probe.id * reformer->padded_dim();
    float direct_dist_sq = 0.0f;
    float direct_ip = 0.0f;
    for (size_t i = 0; i < reformer->padded_dim(); ++i) {
      const float diff = state.rotated_query[i] - centroid[i];
      direct_dist_sq += diff * diff;
      direct_ip += state.rotated_query[i] * centroid[i];
    }
    EXPECT_NEAR(std::sqrt(direct_dist_sq), probe.residual_norm, 1e-4f);
    EXPECT_NEAR(direct_ip, probe.inner_product, 1e-4f);
    EXPECT_EQ(0, reformer->prepare_for_cluster(probe, &state));
  }
  for (size_t i = 0; i < probes.size(); ++i) {
    EXPECT_EQ(expected[i].second, probes[i].id);
  }

  std::vector<uint32_t> legacy_ids;
  ASSERT_EQ(0, reformer->select_probe_centroids(query.data(), 3, &legacy_ids));
  ASSERT_EQ(probes.size(), legacy_ids.size());
  for (size_t i = 0; i < probes.size(); ++i) {
    EXPECT_EQ(probes[i].id, legacy_ids[i]);
  }

  std::vector<IvfRabitqProbeCentroid> single_probe;
  ASSERT_EQ(0, reformer->select_probe_centroids(query.data(), 1, &state,
                                                &single_probe));
  ASSERT_EQ(1UL, single_probe.size());
  EXPECT_EQ(expected[0].second, single_probe[0].id);
  const float *single_centroid =
      rotated_centroids.data() + single_probe[0].id * reformer->padded_dim();
  float single_dist_sq = 0.0f;
  float single_ip = 0.0f;
  for (size_t i = 0; i < reformer->padded_dim(); ++i) {
    const float diff = state.rotated_query[i] - single_centroid[i];
    single_dist_sq += diff * diff;
    single_ip += state.rotated_query[i] * single_centroid[i];
  }
  EXPECT_NEAR(std::sqrt(single_dist_sq), single_probe[0].residual_norm, 1e-4f);
  EXPECT_NEAR(single_ip, single_probe[0].inner_product, 1e-4f);

  NumericalVector<float> zero_query(kDim);
  std::fill(zero_query.data(), zero_query.data() + zero_query.dimension(),
            0.0f);
  IvfRabitqQueryState zero_state;
  ASSERT_EQ(0, reformer->create_query_state(zero_query.data(), &zero_state));
  ASSERT_EQ(0, reformer->select_probe_centroids(zero_query.data(), 3,
                                                &zero_state, &probes));
  ASSERT_EQ(3UL, probes.size());
  for (const auto &probe : probes) {
    EXPECT_TRUE(std::isfinite(probe.residual_norm));
    EXPECT_FLOAT_EQ(0.0f, probe.inner_product);
    EXPECT_EQ(0, reformer->prepare_for_cluster(probe, &zero_state));
  }
}

TEST_F(IvfRabitqStreamerTest, TestL2ProbeCarriesCoarseResidualNorm) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto holder = BuildDirectionalHolder(kDim, 1000UL);

  std::shared_ptr<IvfRabitqReformer> reformer;
  std::vector<float> rotated_centroids;
  BuildLoadedReformer(meta, "SquaredEuclidean", holder,
                      dir_ + "/TestL2ProbeCarriesCoarseResidualNorm", &reformer,
                      &rotated_centroids);
  ASSERT_NE(nullptr, reformer);

  NumericalVector<float> query(kDim);
  FillTestVector(101, &query);
  IvfRabitqQueryState state;
  ASSERT_EQ(0, reformer->create_query_state(query.data(), &state));

  std::vector<IvfRabitqProbeCentroid> probes;
  ASSERT_EQ(0,
            reformer->select_probe_centroids(query.data(), 4, &state, &probes));
  ASSERT_EQ(4UL, probes.size());
  std::vector<std::pair<float, uint32_t>> expected;
  const size_t centroid_count =
      rotated_centroids.size() / reformer->padded_dim();
  expected.reserve(centroid_count);
  for (size_t id = 0; id < centroid_count; ++id) {
    const float *centroid =
        rotated_centroids.data() + id * reformer->padded_dim();
    float direct_dist_sq = 0.0f;
    for (size_t i = 0; i < reformer->padded_dim(); ++i) {
      const float diff = state.rotated_query[i] - centroid[i];
      direct_dist_sq += diff * diff;
    }
    expected.emplace_back(direct_dist_sq, static_cast<uint32_t>(id));
  }
  std::sort(expected.begin(), expected.end());
  for (const auto &probe : probes) {
    const float *centroid =
        rotated_centroids.data() + probe.id * reformer->padded_dim();
    float direct_dist_sq = 0.0f;
    for (size_t i = 0; i < reformer->padded_dim(); ++i) {
      const float diff = state.rotated_query[i] - centroid[i];
      direct_dist_sq += diff * diff;
    }
    EXPECT_NEAR(std::sqrt(direct_dist_sq), probe.residual_norm, 1e-4f);
    EXPECT_FLOAT_EQ(0.0f, probe.inner_product);
    EXPECT_EQ(0, reformer->prepare_for_cluster(probe, &state));
  }
  for (size_t i = 0; i < probes.size(); ++i) {
    EXPECT_EQ(expected[i].second, probes[i].id);
  }
}

TEST_F(IvfRabitqStreamerTest, TestIncrementalAddUnsupported) {
  NumericalVector<float> query_vec(kDim);
  FillTestVector(83, &query_vec);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);

  {
    IndexStreamer::Pointer streamer = std::make_shared<IvfRabitqStreamer>();
    ailego::Params params;
    params.set(PARAM_IVF_RABITQ_NLIST, 32U);
    params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
    ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));

    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0,
              storage->open(dir_ + "/TestUnsupportedFlushBeforeBuild", true));
    ASSERT_NE(0, streamer->open(storage));

    auto context = streamer->create_context();
    EXPECT_EQ(IndexError_NotImplemented,
              streamer->add_impl(1, query_vec.data(), query_meta, context));
    EXPECT_EQ(0, streamer->flush(0UL));
  }

  {
    auto holder = BuildHolder(kDim, 1000UL);
    ailego::Params params;
    params.set(PARAM_IVF_RABITQ_NLIST, 32U);
    params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
    IndexStreamer::Pointer streamer;
    BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                         dir_ + "/TestUnsupportedAddAfterBuild", &streamer);
    ASSERT_NE(nullptr, streamer);

    auto context = streamer->create_context();
    EXPECT_EQ(IndexError_NotImplemented,
              streamer->add_impl(1001, query_vec.data(), query_meta, context));
    EXPECT_EQ(0, streamer->flush(0UL));
  }
}

TEST_F(IvfRabitqStreamerTest, TestSearchScanLimitParams) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>((i + 1) * (j + 3)) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 32U);
  params.set(PARAM_IVF_RABITQ_NPROBE, 0U);
  params.set(PARAM_IVF_RABITQ_SCAN_RATIO, 0.05f);
  params.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, 80U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestSearchScanLimitParams", &streamer);
  ASSERT_NE(nullptr, streamer);

  NumericalVector<float> query_vec(kDim);
  for (size_t j = 0; j < kDim; ++j) {
    query_vec[j] = static_cast<float>(j + 3) / 1000.0f;
  }
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);

  auto base_context = streamer->create_context();
  auto *context = dynamic_cast<IvfRabitqContext *>(base_context.get());
  ASSERT_NE(nullptr, context);
  context->set_topk(10);

  ASSERT_EQ(
      0, streamer->search_impl(query_vec.data(), query_meta, 1, base_context));
  EXPECT_EQ(80U, context->max_scan_count());
  EXPECT_GT(context->result(0).size(), 0UL);

  ailego::Params nprobe_params;
  nprobe_params.set(PARAM_IVF_RABITQ_NPROBE, 8U);
  nprobe_params.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, 80U);
  ASSERT_EQ(0, context->update(nprobe_params));
  ASSERT_EQ(
      0, streamer->search_impl(query_vec.data(), query_meta, 1, base_context));
  EXPECT_EQ(static_cast<uint32_t>(doc_cnt), context->max_scan_count());

  ailego::Params bf_params;
  bf_params.set(PARAM_IVF_RABITQ_NPROBE, 1U);
  bf_params.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD,
                static_cast<uint32_t>(doc_cnt));
  ASSERT_EQ(0, context->update(bf_params));
  ASSERT_EQ(
      0, streamer->search_impl(query_vec.data(), query_meta, 1, base_context));
  EXPECT_EQ(static_cast<uint32_t>(doc_cnt), context->max_scan_count());
}

TEST_F(IvfRabitqStreamerTest, TestSearchWithFilter) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i * kDim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set("proxima.ivf_rabitq.nlist", 32U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestSearchWithFilter", &streamer);
  ASSERT_NE(nullptr, streamer);

  NumericalVector<float> query_vec(kDim);
  for (size_t j = 0; j < kDim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  auto context = streamer->create_context();
  context->set_topk(10);
  ailego::Params search_params;
  search_params.set("proxima.ivf_rabitq.nprobe", 16U);
  context->update(search_params);

  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  const auto &unfiltered_result = context->result(0);
  ASSERT_GT(unfiltered_result.size(), 0UL);
  uint64_t filtered_key = unfiltered_result[0].key();

  context->set_filter(
      [filtered_key](uint64_t key) { return key == filtered_key; });
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  const auto &filtered_result = context->result(0);
  auto iter = std::find_if(filtered_result.begin(), filtered_result.end(),
                           [filtered_key](const IndexDocument &doc) {
                             return doc.key() == filtered_key;
                           });
  EXPECT_EQ(filtered_result.end(), iter);

  context->reset();
  context->set_topk(10);
  ASSERT_EQ(0, context->update(search_params));
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  const auto &reset_result = context->result(0);
  auto reset_iter = std::find_if(reset_result.begin(), reset_result.end(),
                                 [filtered_key](const IndexDocument &doc) {
                                   return doc.key() == filtered_key;
                                 });
  EXPECT_NE(reset_result.end(), reset_iter);
}

TEST_F(IvfRabitqStreamerTest, TestSearchByPrimaryKeys) {
  constexpr size_t kDocCount = 256;
  auto holder = BuildHolder(kDim, kDocCount);

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 16U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 1U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestSearchByPrimaryKeys", &streamer);
  ASSERT_NE(nullptr, streamer);

  NumericalVector<float> query_vec0(kDim);
  NumericalVector<float> query_vec1(kDim);
  FillTestVector(37, &query_vec0);
  FillTestVector(42, &query_vec1);
  std::vector<float> query_vectors(kDim * 2);
  memcpy(query_vectors.data(), query_vec0.data(), kDim * sizeof(float));
  memcpy(query_vectors.data() + kDim, query_vec1.data(), kDim * sizeof(float));
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  auto context = streamer->create_context();
  context->set_topk(3);

  const std::vector<std::vector<uint64_t>> keys{{5, 37, 91, 9999},
                                                {7, 42, 100, 9999}};
  ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(query_vectors.data(), keys,
                                                  query_meta, 2, context));
  ASSERT_EQ(3UL, context->result(0).size());
  EXPECT_EQ(37UL, context->result(0)[0].key());
  for (const auto &doc : context->result(0)) {
    EXPECT_NE(keys[0].end(),
              std::find(keys[0].begin(), keys[0].end(), doc.key()));
  }
  ASSERT_EQ(3UL, context->result(1).size());
  EXPECT_EQ(42UL, context->result(1)[0].key());
  for (const auto &doc : context->result(1)) {
    EXPECT_NE(keys[1].end(),
              std::find(keys[1].begin(), keys[1].end(), doc.key()));
  }

  context->set_filter([](uint64_t key) { return key == 37 || key == 42; });
  ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(query_vectors.data(), keys,
                                                  query_meta, 2, context));
  ASSERT_EQ(2UL, context->result(0).size());
  ASSERT_EQ(2UL, context->result(1).size());
  for (uint32_t q = 0; q < 2; ++q) {
    for (const auto &doc : context->result(q)) {
      EXPECT_NE(37UL, doc.key());
      EXPECT_NE(42UL, doc.key());
    }
  }
}

TEST_F(IvfRabitqStreamerTest, TestGroupBySearchByPrimaryKeys) {
  constexpr size_t kDocCount = 256;
  auto holder = BuildHolder(kDim, kDocCount);

  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 16U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 1U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestGroupBySearchByPrimaryKeys", &streamer);
  ASSERT_NE(nullptr, streamer);

  NumericalVector<float> query_vec(kDim);
  FillTestVector(37, &query_vec);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  auto context = streamer->create_context();
  context->set_group_by([](uint64_t key) { return std::to_string(key % 3U); });
  context->set_topk(10);

  auto expect_top2_matches = [](const IndexGroupDocumentList &expected,
                                const IndexGroupDocumentList &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(expected[i].group_id(), actual[i].group_id());
      ASSERT_EQ(2UL, actual[i].docs().size());
      ASSERT_GE(expected[i].docs().size(), actual[i].docs().size());
      for (size_t j = 0; j < actual[i].docs().size(); ++j) {
        EXPECT_EQ(expected[i].docs()[j].key(), actual[i].docs()[j].key());
        EXPECT_FLOAT_EQ(expected[i].docs()[j].score(),
                        actual[i].docs()[j].score());
      }
    }
  };

  context->set_group_params(/*group_num=*/3, /*group_topk=*/kDocCount);
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  const auto baseline_groups = context->group_result(0);

  context->set_group_params(/*group_num=*/3, /*group_topk=*/2);
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  expect_top2_matches(baseline_groups, context->group_result(0));

  const uint64_t filtered_key = baseline_groups[0].docs()[0].key();
  context->set_filter(
      [filtered_key](uint64_t key) { return key == filtered_key; });
  context->set_group_params(/*group_num=*/3, /*group_topk=*/kDocCount);
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  const auto filtered_baseline_groups = context->group_result(0);

  context->set_group_params(/*group_num=*/3, /*group_topk=*/2);
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  expect_top2_matches(filtered_baseline_groups, context->group_result(0));
  context->reset_filter();

  const std::vector<std::vector<uint64_t>> keys{
      {3, 4, 5, 36, 37, 38, 90, 91, 92}};
  ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(query_vec.data(), keys,
                                                  query_meta, 1, context));

  const auto &groups = context->group_result(0);
  ASSERT_EQ(3UL, groups.size());
  for (const auto &group : groups) {
    ASSERT_EQ(2UL, group.docs().size());
    const uint64_t expected_group = std::stoull(group.group_id());
    for (const auto &doc : group.docs()) {
      EXPECT_EQ(expected_group, doc.key() % 3U);
    }
  }
}

TEST_F(IvfRabitqStreamerTest, TestRecallQuality) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 2000UL;
  srand(42);
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set("proxima.ivf_rabitq.nlist", 64U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params, dir_ + "/TestRecall",
                       &streamer);
  ASSERT_NE(nullptr, streamer);

  // Use high nprobe (all clusters) as ground truth, low nprobe as test
  size_t topk = 10;
  int total_hits = 0;
  int total_cnts = 0;
  size_t query_cnt = 100;
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);

  auto gt_ctx = streamer->create_context();
  auto test_ctx = streamer->create_context();
  gt_ctx->set_topk(topk);
  test_ctx->set_topk(topk);

  // Ground truth: search all clusters
  ailego::Params gt_params;
  gt_params.set("proxima.ivf_rabitq.nprobe", 64U);
  gt_ctx->update(gt_params);

  // Test: search with limited nprobe
  ailego::Params test_search_params;
  test_search_params.set("proxima.ivf_rabitq.nprobe", 16U);
  test_ctx->update(test_search_params);

  srand(123);
  for (size_t q = 0; q < query_cnt; q++) {
    NumericalVector<float> query_vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      query_vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    ASSERT_EQ(0,
              streamer->search_impl(query_vec.data(), query_meta, 1, gt_ctx));
    ASSERT_EQ(0,
              streamer->search_impl(query_vec.data(), query_meta, 1, test_ctx));

    auto &gt_result = gt_ctx->result(0);
    auto &test_result = test_ctx->result(0);

    ASSERT_GT(gt_result.size(), 0UL);
    ASSERT_GT(test_result.size(), 0UL);

    for (size_t k = 0; k < test_result.size(); ++k) {
      total_cnts++;
      for (size_t j = 0; j < gt_result.size(); ++j) {
        if (gt_result[j].key() == test_result[k].key()) {
          total_hits++;
          break;
        }
      }
    }
  }

  float recall = total_hits * 1.0f / total_cnts;
  LOG_INFO("IVF RabitQ recall@%zu (nprobe=16 vs nprobe=64): %.2f%% (%d/%d)",
           topk, recall * 100.0f, total_hits, total_cnts);
  EXPECT_GT(recall, 0.50f) << "Recall too low";
}

TEST_F(IvfRabitqStreamerTest, TestCloseAndReopen) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set("proxima.ivf_rabitq.nlist", 16U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 4U);

  std::string path = dir_ + "/TestCloseReopen";
  BuildIndexFile(*index_meta_ptr_, holder, params, path);

  // Open and close the built index once before reopening it.
  {
    IndexStreamer::Pointer streamer = std::make_shared<IvfRabitqStreamer>();
    ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_EQ(0, streamer->open(storage));
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());
  }

  // Reopen and search
  {
    IndexStreamer::Pointer streamer2 = std::make_shared<IvfRabitqStreamer>();
    ASSERT_EQ(0, streamer2->init(*index_meta_ptr_, params));
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_EQ(0, streamer2->open(storage));

    NumericalVector<float> query_vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      query_vec[j] = 10.0f;
    }

    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
    auto context = streamer2->create_context();
    context->set_topk(5);
    ailego::Params search_params;
    search_params.set("proxima.ivf_rabitq.nprobe", 8U);
    context->update(search_params);

    ASSERT_EQ(0,
              streamer2->search_impl(query_vec.data(), query_meta, 1, context));
    const auto &result = context->result(0);
    ASSERT_GT(result.size(), 0UL);
    // Key 10 should be the nearest
    ASSERT_EQ(10UL, result[0].key());
  }
}

TEST_F(IvfRabitqStreamerTest, TestCosineCloseAndReopen) {
  IndexMeta raw_meta(IndexMeta::DataType::DT_FP32, kDim);
  raw_meta.set_metric("Cosine", 0, ailego::Params());

  auto raw_holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 1000UL;
  std::vector<NumericalVector<float>> raw_vectors;
  raw_vectors.reserve(doc_cnt);
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(((i + 3) * (j + 5)) % 97 + 1) / 97.0f;
    }
    raw_vectors.push_back(vec);
    ASSERT_TRUE(raw_holder->emplace(i, raw_vectors.back()));
  }

  auto converter = IndexFactory::CreateConverter("CosineNormalizeConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(raw_meta, ailego::Params()));
  IndexMeta transformed_meta = converter->meta();
  ASSERT_EQ(kDim + 1, transformed_meta.dimension());
  ASSERT_EQ(0, converter->transform(raw_holder));
  auto holder = converter->result();
  ASSERT_NE(nullptr, holder);

  ailego::Params params;
  params.set("proxima.ivf_rabitq.nlist", 32U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 7U);
  params.set(PARAM_RABITQ_GENERAL_DIMENSION, static_cast<uint32_t>(kDim));

  std::string path = dir_ + "/TestCosineCloseAndReopen";
  BuildIndexFile(transformed_meta, holder, params, path);

  {
    IndexStreamer::Pointer streamer = std::make_shared<IvfRabitqStreamer>();
    ASSERT_EQ(0, streamer->init(transformed_meta, params));
    ASSERT_EQ(kDim + 1, streamer->meta().dimension());
    ASSERT_EQ("Cosine", streamer->meta().metric_name());
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_EQ(0, streamer->open(storage));
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());
  }

  IndexStreamer::Pointer streamer2 = std::make_shared<IvfRabitqStreamer>();
  ASSERT_EQ(0, streamer2->init(transformed_meta, params));
  ASSERT_EQ(kDim + 1, streamer2->meta().dimension());
  ASSERT_EQ("Cosine", streamer2->meta().metric_name());
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, streamer2->open(storage));

  auto reformer =
      IndexFactory::CreateReformer(transformed_meta.reformer_name());
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(transformed_meta.reformer_params()));

  IndexQueryMeta raw_query_meta(IndexMeta::DataType::DT_FP32, kDim);
  IndexQueryMeta transformed_query_meta;
  std::string transformed_query0;
  std::string transformed_query1;
  ASSERT_EQ(0,
            reformer->transform(raw_vectors[17].data(), raw_query_meta,
                                &transformed_query0, &transformed_query_meta));
  IndexQueryMeta transformed_query_meta1;
  ASSERT_EQ(0,
            reformer->transform(raw_vectors[101].data(), raw_query_meta,
                                &transformed_query1, &transformed_query_meta1));
  ASSERT_EQ(transformed_query_meta.dimension(),
            transformed_query_meta1.dimension());
  ASSERT_EQ(kDim + 1, transformed_query_meta.dimension());

  std::vector<float> query_buffer(2 * transformed_query_meta.dimension());
  std::memcpy(query_buffer.data(), transformed_query0.data(),
              transformed_query0.size());
  std::memcpy(query_buffer.data() + transformed_query_meta.dimension(),
              transformed_query1.data(), transformed_query1.size());

  auto context = streamer2->create_context();
  context->set_topk(10);
  ailego::Params search_params;
  search_params.set("proxima.ivf_rabitq.nprobe", 16U);
  context->update(search_params);

  ASSERT_EQ(0, streamer2->search_impl(query_buffer.data(),
                                      transformed_query_meta, 2, context));
  EXPECT_GT(context->result(0).size(), 0UL);
  EXPECT_GT(context->result(1).size(), 0UL);
}

TEST_F(IvfRabitqStreamerTest, TestSmallDataset) {
  // Test with a small but valid dataset. nlist must be significantly smaller
  // than doc_count because rabitqlib's f_rescale = 1/||residual|| becomes NaN
  // when a cluster has only 1 vector (residual ≈ 0). Use 200 vectors / 32
  // clusters ≈ 6 vectors per cluster to stay well within the valid range.
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(kDim);
  size_t doc_cnt = 200UL;
  srand(7);
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set("proxima.ivf_rabitq.nlist", 32U);
  params.set(PARAM_RABITQ_TOTAL_BITS, 3U);
  IndexStreamer::Pointer streamer;
  BuildAndOpenStreamer(*index_meta_ptr_, holder, params,
                       dir_ + "/TestSmallDataset", &streamer);
  ASSERT_NE(nullptr, streamer);

  NumericalVector<float> query_vec(kDim);
  srand(99);
  for (size_t j = 0; j < kDim; ++j) {
    query_vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  auto context = streamer->create_context();
  context->set_topk(5);
  ailego::Params search_params;
  search_params.set("proxima.ivf_rabitq.nprobe", 32U);
  context->update(search_params);

  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));
  const auto &result = context->result(0);
  ASSERT_GT(result.size(), 0UL);
  LOG_INFO("TestSmallDataset: %zu results, top key=%zu, score=%f",
           result.size(), (size_t)result[0].key(), result[0].score());
}

}  // namespace core
}  // namespace zvec
