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
#include "ivf_searcher.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "zvec/core/framework/index_framework.h"
#include "ivf_builder.h"
#include "ivf_dumper.h"
#include "ivf_streamer.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

class FixedCentroidTrainer : public IndexTrainer {
 public:
  FixedCentroidTrainer(const IndexMeta &meta, IndexBundle::Pointer bundle)
      : meta_(meta), bundle_(std::move(bundle)) {}

  int init(const IndexMeta &, const Params &) override {
    return 0;
  }

  int cleanup() override {
    return 0;
  }

  int train(IndexThreads::Pointer, IndexHolder::Pointer) override {
    return 0;
  }

  int load(IndexStorage::Pointer) override {
    return 0;
  }

  int dump(const IndexDumper::Pointer &) override {
    return 0;
  }

  const IndexMeta &meta() const override {
    return meta_;
  }

  const IndexTrainer::Stats &stats() const override {
    return stats_;
  }

  IndexBundle::Pointer indexes() const override {
    return bundle_;
  }

 private:
  IndexMeta meta_{};
  IndexTrainer::Stats stats_{};
  IndexBundle::Pointer bundle_{};
};

static IndexTrainer::Pointer CreateFixedCentroidTrainer(
    const IndexMeta &meta, const std::vector<float> &centroid_values) {
  IndexCluster::CentroidList centroids;
  centroids.reserve(centroid_values.size());
  for (float value : centroid_values) {
    NumericalVector<float> centroid(meta.dimension());
    for (size_t i = 0; i < meta.dimension(); ++i) {
      centroid[i] = value;
    }
    IndexCluster::Centroid item;
    item.set_feature(centroid);
    centroids.emplace_back(std::move(item));
  }

  IndexBundle::Pointer bundle;
  int ret = IndexCluster::Serialize(meta, centroids, &bundle);
  EXPECT_EQ(0, ret);
  if (ret != 0) {
    return IndexTrainer::Pointer();
  }

  return std::make_shared<FixedCentroidTrainer>(meta, std::move(bundle));
}

class IVFSearcherTest : public testing::Test {
 public:
 protected:
  void SetUp() override;
  void TearDown() override;
  void prepare_index_holder(uint32_t base_key, uint32_t num);

  void prepare_rand_index_holder(uint32_t base_key, uint32_t num);

  void prepare_fp16_index_holder(uint32_t base_key, uint32_t num);

  void prepare_fp32_index_holder(uint32_t base_key, uint32_t num);

  void prepare_binary_index_holder(uint32_t base_key, uint32_t num);

  void prepare_int8_index_holder(uint32_t base_key, uint32_t num);

  void prepare_same_index_holder(uint32_t base_key, uint32_t num);

  IndexMeta index_meta_;
  Params params_;
  uint32_t dimension_;
  IndexHolder::Pointer holder_;
  std::string index_path_;
  IndexThreads::Pointer threads_{};
};

// Compare candidate-key BF against full BF with the equivalent predicate.
// Empty candidate batches, missing keys and a second predicate must not cause
// scanning unrelated vectors or retaining results from an earlier query.
template <typename Searcher>
static void CheckKeyedBruteForce(
    Searcher *searcher, const void *queries, const IndexQueryMeta &qmeta,
    const std::vector<std::vector<uint64_t>> &keys) {
  auto context = searcher->create_context();
  auto baseline = searcher->create_context();
  ASSERT_NE(nullptr, context);
  ASSERT_NE(nullptr, baseline);
  context->set_topk(16);
  baseline->set_topk(16);
  const auto count = static_cast<uint32_t>(keys.size());
  for (bool filtered : {false, true}) {
    for (float threshold : {std::numeric_limits<float>::max(), 0.25f}) {
      context->set_threshold(threshold);
      baseline->set_threshold(threshold);
      context->set_filter(
          [filtered](uint64_t key) { return filtered && key % 2 == 0; });
      ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(queries, keys, qmeta,
                                                      count, context));
      for (uint32_t q = 0; q < count; ++q) {
        SCOPED_TRACE(q);
        baseline->set_filter([&keys, q, filtered](uint64_t key) {
          return (filtered && key % 2 == 0) ||
                 std::find(keys[q].begin(), keys[q].end(), key) ==
                     keys[q].end();
        });
        const void *query = static_cast<const char *>(queries) +
                            q * static_cast<size_t>(qmeta.element_size());
        ASSERT_EQ(0, searcher->search_bf_impl(query, qmeta, baseline));
        const auto &expected = baseline->result();
        const auto &actual = context->result(q);
        ASSERT_EQ(expected.size(), actual.size());
        for (const auto &doc : actual) {
          const auto found = std::find_if(
              expected.begin(), expected.end(),
              [&doc](const auto &item) { return item.key() == doc.key(); });
          ASSERT_NE(expected.end(), found);
          EXPECT_NEAR(found->score(), doc.score(),
                      1e-4f * std::max(1.0f, std::abs(found->score())));
        }
        // Unlike full BF, work must be bounded by the requested candidates.
        const auto *ivf_context =
            dynamic_cast<const IVFSearcherContext *>(context.get());
        ASSERT_NE(nullptr, ivf_context);
        EXPECT_LE(ivf_context->stats(q).dist_calced_count(), keys[q].size());
      }
    }
  }
  const std::vector<std::vector<uint64_t>> empty_keys(1);
  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(queries, empty_keys, qmeta,
                                                  context));
  EXPECT_TRUE(context->result().empty());
  EXPECT_EQ(IndexError_InvalidArgument, searcher->search_bf_by_p_keys_impl(
                                            queries, keys, qmeta, 0, context));
  EXPECT_EQ(IndexError_InvalidArgument,
            searcher->search_bf_by_p_keys_impl(queries, keys, qmeta, count + 1,
                                               context));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      searcher->search_bf_by_p_keys_impl(nullptr, keys, qmeta, count, context));
  IndexRunner::Context::Pointer missing_context;
  EXPECT_EQ(IndexError_InvalidArgument,
            searcher->search_bf_by_p_keys_impl(queries, keys, qmeta, count,
                                               missing_context));
  IndexQueryMeta wrong_meta(qmeta.data_type(), qmeta.dimension() + 1);
  EXPECT_EQ(IndexError_InvalidArgument,
            searcher->search_bf_by_p_keys_impl(queries, keys, wrong_meta, count,
                                               context));
}

void IVFSearcherTest::SetUp() {
  dimension_ = 8U;

  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());

  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  index_path_ = "./ivf_searcher.index";
  std::mt19937 gen((std::random_device())());
  bool v = std::uniform_int_distribution<size_t>(0, 1)(gen);
  if (v) {
    threads_ = std::make_shared<SingleQueueIndexThreads>();
  }
}

void IVFSearcherTest::TearDown() {
  File::RemovePath(index_path_);
}

void IVFSearcherTest::prepare_index_holder(uint32_t base_key, uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 1.0f * i;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_rand_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = std::rand() % 1000 * 1.0;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_fp32_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 0.01f * i;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_fp16_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 0.01f * i;
    }
    holder->emplace(key + i, vec);
  }

  IndexConverter::Pointer conveter =
      IndexFactory::CreateConverter("HalfFloatConverter");
  conveter->init(index_meta_, Params());
  IndexHolder::Pointer new_holder(holder);
  conveter->transform(new_holder);
  holder_ = conveter->result();
}

void IVFSearcherTest::prepare_int8_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_INT8> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_INT8>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<int8_t> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = (int8_t)(i % 128);
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_binary_index_holder(uint32_t base_key,
                                                  uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_BINARY32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_BINARY32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    BinaryVector<uint32_t> vec(dimension_);
    for (size_t j = 0; j < dimension_ && j < i; ++j) {
      vec.set(j);
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_same_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 8;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

TEST_F(IVFSearcherTest, TestInit) {
  IVFSearcher searcher;
  int ret = searcher.init(params_);
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, LegacyQuantizerDescriptorDoesNotSelectTurboFormat) {
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    SCOPED_TRACE(order);
    index_meta_.set_major_order(order);
    index_meta_.set_quantizer("Fp32Quantizer", 0, Params());
    params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
    params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
    prepare_index_holder(0, 33);
    IVFBuilder builder;
    // The legacy overload preserves quantizer metadata but does not encode
    // postings with Turbo or write a Turbo quantizer segment.
    ASSERT_EQ(0, builder.init(index_meta_, params_));
    auto centroid_meta = index_meta_;
    centroid_meta.set_quantizer("", 0, Params());
    ASSERT_EQ(
        0, builder.train(CreateFixedCentroidTrainer(centroid_meta, {16.0f})));
    ASSERT_EQ(0, builder.build(threads_, holder_));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(index_path_));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());

    for (const char *storage_name :
         {"FileReadStorage", "MMapFileReadStorage", "BufferReadStorage"}) {
      SCOPED_TRACE(storage_name);
      auto storage = IndexFactory::CreateStorage(storage_name);
      ASSERT_NE(nullptr, storage);
      ASSERT_EQ(0, storage->init(Params()));
      ASSERT_EQ(0, storage->open(index_path_, false));
      EXPECT_FALSE(storage->has(IVF_TURBO_QUANTIZER_SEG_ID));
      IVFEntity entity;
      ASSERT_EQ(0, entity.load(storage));
      EXPECT_EQ("Fp32Quantizer", entity.meta().quantizer_name());
      EXPECT_FALSE(
          entity.meta().attributes().has(IVF_TURBO_FORMAT_VERSION_ATTRIBUTE));
      EXPECT_EQ(nullptr, entity.quantizer());
      Params search_params;
      search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
      search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
      const std::vector<float> query(dimension_, 32.0f);
      const IndexQueryMeta qmeta(IndexMeta::DT_FP32, dimension_);
      auto check_search = [&](auto *searcher) {
        EXPECT_EQ(nullptr, searcher->quantizer());
        auto context = searcher->create_context();
        ASSERT_NE(nullptr, context);
        context->set_topk(3);
        for (bool brute_force : {false, true}) {
          ASSERT_EQ(0,
                    brute_force
                        ? searcher->search_bf_impl(query.data(), qmeta, context)
                        : searcher->search_impl(query.data(), qmeta, context));
          const auto &result = context->result(0);
          ASSERT_EQ(3U, result.size());
          for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(32U - i, result[i].key());
            EXPECT_FLOAT_EQ(static_cast<float>(i * i * dimension_),
                            result[i].score());
          }
        }
        CheckKeyedBruteForce(searcher, query.data(), qmeta,
                             {{30, 2, 31, 9999, kInvalidKey}});
      };
      IVFSearcher searcher;
      ASSERT_EQ(0, searcher.init(search_params));
      ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
      check_search(&searcher);
      ASSERT_EQ(0, searcher.unload());
      IVFStreamer streamer;
      ASSERT_EQ(0, streamer.init(index_meta_, search_params));
      ASSERT_EQ(0, streamer.open(storage));
      check_search(&streamer);
      ASSERT_EQ(0, streamer.close());
    }
  }
}

TEST_F(IVFSearcherTest,
       MultipleDistanceBatchesAndDeletedSlotsMatchExactSearch) {
  constexpr uint32_t dim = 256;
  constexpr uint32_t count = 2 * 512 + 7;
  const std::vector<uint32_t> deleted{0,   31,   32,   511,
                                      512, 1023, 1024, count - 1};
  std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
  std::mt19937 random(7193);
  std::uniform_real_distribution<float> value(-2.0f, 3.0f);
  for (auto &vector : vectors) {
    for (auto &component : vector) {
      component = value(random);
    }
  }
  const auto key = [&deleted](uint32_t i) {
    return std::find(deleted.begin(), deleted.end(), i) != deleted.end()
               ? kInvalidKey
               : uint64_t{10000} + 13 * i;
  };

  auto &memory_pool = MemoryLimitPool::get_instance();
  const size_t previous_capacity = memory_pool.capacity();
  ASSERT_EQ(0u, memory_pool.used());
  ASSERT_EQ(0, memory_pool.init(8 * 1024 * 1024));
  auto cleanup = ScopeGuard::Make([&]() {
    EXPECT_EQ(0u, memory_pool.used());
    EXPECT_EQ(0, memory_pool.init(previous_capacity));
  });

  // Cover multiple reads and a short final batch in both legacy layouts and
  // Turbo's pointer batch path. Four Turbo cosine rows leave padding in each
  // block.
  for (auto format : {std::make_tuple(false, IndexMeta::MO_ROW, 32U),
                      std::make_tuple(false, IndexMeta::MO_COLUMN, 32U),
                      std::make_tuple(true, IndexMeta::MO_ROW, 32U),
                      std::make_tuple(true, IndexMeta::MO_ROW, 4U)}) {
    const bool turbo = std::get<0>(format);
    const auto order = std::get<1>(format);
    const uint32_t block_vectors = std::get<2>(format);
    SCOPED_TRACE(turbo);
    SCOPED_TRACE(order);
    SCOPED_TRACE(block_vectors);
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, dim);
    meta.set_metric(turbo ? "Cosine" : "SquaredEuclidean", 0, Params());
    meta.set_major_order(order);
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
    for (uint32_t i = 0; i < count; ++i) {
      NumericalVector<float> vector(dim);
      std::copy(vectors[i].begin(), vectors[i].end(), vector.data());
      // Invalid keys represent deleted posting slots; their vectors remain
      // present so skipping them cannot change the layout of later blocks.
      holder->emplace(key(i), vector);
    }

    Params build_params;
    build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
    build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
    build_params.set(PARAM_IVF_BUILDER_THREAD_COUNT, 1U);
    build_params.set(PARAM_IVF_BUILDER_BLOCK_VECTOR_COUNT, block_vectors);
    IVFBuilder builder;
    if (turbo) {
      meta.set_quantizer("Fp32Quantizer", 0, Params());
      auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
      ASSERT_NE(nullptr, quantizer);
      ASSERT_EQ(0, quantizer->init(meta, Params()));
      ASSERT_EQ(0, builder.init(meta, build_params, quantizer));
    } else {
      ASSERT_EQ(0, builder.init(meta, build_params));
    }
    ASSERT_EQ(0, builder.train(IndexThreads::Pointer(), holder));
    ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(index_path_));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    ASSERT_EQ(0, builder.cleanup());

    for (const char *storage_name :
         {"MMapFileReadStorage", "BufferReadStorage"}) {
      SCOPED_TRACE(storage_name);
      auto storage = IndexFactory::CreateStorage(storage_name);
      ASSERT_NE(nullptr, storage);
      Params storage_params;
      // utility_params.h conflicts with quantizer_params.h on legacy MIPS
      // constants; keep this storage-only setting local to the test.
      storage_params.set("proxima.buffer.read_storage.warmup_mode",
                         "sequential");
      ASSERT_EQ(0, storage->init(storage_params));
      ASSERT_EQ(0, storage->open(index_path_, false));
      {
        IVFEntity entity;
        ASSERT_EQ(0, entity.load(storage));
        ASSERT_EQ(order, entity.meta().major_order());
        ASSERT_EQ(turbo, entity.quantizer() != nullptr);
        ASSERT_EQ(count, entity.vector_count());
        ASSERT_EQ(1u, entity.inverted_list_count());
        ASSERT_EQ(block_vectors, entity.block_vector_count());
        const size_t block_vector_bytes =
            block_vectors * entity.meta().element_size();
        if (block_vectors == 4) {
          ASSERT_GT(entity.inverted_block_size(), block_vector_bytes);
        } else {
          ASSERT_EQ(entity.inverted_block_size(), block_vector_bytes);
        }
        IndexStorage::MemoryBlock list_block;
        const auto *list = entity.inverted_list_meta(0, list_block);
        ASSERT_NE(nullptr, list);
        ASSERT_EQ((count + block_vectors - 1) / block_vectors,
                  list->block_count);
        ASSERT_EQ(count, list->vector_count);
        auto body = storage->get(IVF_INVERTED_BODY_SEG_ID);
        ASSERT_NE(nullptr, body);
        const size_t bytes =
            (512 / block_vectors) * entity.inverted_block_size();
        IndexStorage::Segment::ScatterBlock spans;
        ASSERT_EQ(bytes, body->read_scatter(list->offset, spans, bytes));
        if (std::string(storage_name) == "BufferReadStorage") {
          // Resident pages exercise the same query across noncontiguous
          // spans, including vectors split by a page boundary.
          ASSERT_GT(spans.spans.size(), 1u);
        } else {
          ASSERT_EQ(1u, spans.spans.size());
        }
      }

      Params search_params;
      search_params.set(PARAM_IVF_SEARCHER_NPROBE, 1U);
      search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1U);
      IVFSearcher searcher;
      ASSERT_EQ(0, searcher.init(search_params));
      ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
      auto context = searcher.create_context();
      ASSERT_NE(nullptr, context);
      const auto *ivf_context =
          dynamic_cast<const IVFSearcherContext *>(context.get());
      ASSERT_NE(nullptr, ivf_context);
      const IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
      for (uint32_t query_id : {0U, 512U, count - 2}) {
        SCOPED_TRACE(query_id);
        const auto &query = vectors[query_id];
        std::vector<std::pair<double, uint64_t>> expected;
        for (uint32_t i = 0; i < count; ++i) {
          if (key(i) == kInvalidKey) {
            continue;
          }
          double dot = 0.0;
          double lhs_norm = 0.0;
          double rhs_norm = 0.0;
          double squared_distance = 0.0;
          for (uint32_t d = 0; d < dim; ++d) {
            const double lhs = vectors[i][d];
            const double rhs = query[d];
            dot += lhs * rhs;
            lhs_norm += lhs * lhs;
            rhs_norm += rhs * rhs;
            squared_distance += (lhs - rhs) * (lhs - rhs);
          }
          const double distance =
              turbo ? 1.0 - dot / std::sqrt(lhs_norm * rhs_norm)
                    : squared_distance;
          expected.emplace_back(distance, key(i));
        }
        std::sort(expected.begin(), expected.end());
        std::unordered_map<uint64_t, double> reference;
        for (const auto &item : expected) {
          reference.emplace(item.second, item.first);
        }
        const auto tolerance = [](double distance) {
          return 2e-6 * std::max(1.0, std::abs(distance));
        };
        for (size_t topk : {size_t{17}, expected.size()}) {
          SCOPED_TRACE(topk);
          context->set_topk(topk);
          for (bool brute_force : {false, true}) {
            SCOPED_TRACE(brute_force);
            ASSERT_EQ(
                0, brute_force
                       ? searcher.search_bf_impl(query.data(), qmeta, context)
                       : searcher.search_impl(query.data(), qmeta, context));
            const auto &result = context->result(0);
            ASSERT_EQ(topk, result.size());
            const double cutoff = expected[topk - 1].first;
            std::unordered_set<uint64_t> returned_keys;
            double greatest_lower_bound = 0.0;
            for (size_t i = 0; i < result.size(); ++i) {
              const auto found = reference.find(result[i].key());
              ASSERT_NE(reference.end(), found);
              EXPECT_TRUE(returned_keys.insert(result[i].key()).second);
              const double exact = found->second;
              const double error = tolerance(exact);
              EXPECT_NEAR(exact, result[i].score(), error);
              EXPECT_LE(exact - error, cutoff + tolerance(cutoff));
              if (i != 0) {
                EXPECT_LE(result[i - 1].score(), result[i].score());
                // Different FP32 reductions may exchange nearly tied keys.
                // Distinguishable reference distances must stay ordered,
                // including nonadjacent pairs in a chain of close scores.
                EXPECT_LE(greatest_lower_bound, exact + error);
                greatest_lower_bound =
                    std::max(greatest_lower_bound, exact - error);
              } else {
                greatest_lower_bound = exact - error;
              }
            }
            for (const auto &item : expected) {
              // Full results must contain every live key. Top-k results must
              // contain every key clearly nearer than the boundary; only
              // keys within the score-error interval may cross that boundary.
              if (topk == expected.size() ||
                  item.first + tolerance(item.first) <
                      cutoff - tolerance(cutoff)) {
                EXPECT_EQ(1u, returned_keys.count(item.second));
              }
            }
            EXPECT_EQ(count, ivf_context->stats(0).dist_calced_count());
          }
        }
      }
      context.reset();
      ASSERT_EQ(0, searcher.unload());
      ASSERT_EQ(0, storage->close());
    }
  }
}

// Keep key-read accounting separate from vector storage so the contiguous
// scan still uses the production MMap base_data() fast path.
class IVFCountingKeysSegment : public IndexSegmentStorage::Segment {
 public:
  explicit IVFCountingKeysSegment(const IndexStorage::Segment::Pointer &keys)
      : Segment(keys, 0, keys->data_size(), keys->padding_size(),
                keys->data_crc()) {}

  size_t read(size_t offset, IndexStorage::MemoryBlock &data,
              size_t len) override {
    ++reads;
    return Segment::read(offset, data, len);
  }

  size_t reads{0};
};

namespace zvec {
namespace core {

class IVFEntityTestPeer {
 public:
  static bool all_keys_valid(const IVFEntity &entity) {
    return entity.all_keys_valid_;
  }

  static std::shared_ptr<IVFCountingKeysSegment> count_key_reads(
      IVFEntity &entity) {
    auto counted = std::make_shared<IVFCountingKeysSegment>(entity.keys_);
    entity.keys_ = counted;
    return counted;
  }
};

}  // namespace core
}  // namespace zvec

TEST_F(IVFSearcherTest, AllValidKeysSkipReadsAndSurviveCloneReload) {
  constexpr uint32_t first_list_count = 17;
  constexpr uint32_t second_list_count = 2 * 512 + 7;
  constexpr uint32_t count = first_list_count + second_list_count;
  constexpr uint32_t invalid_id = first_list_count + 512;
  const auto key = [=](uint32_t id) {
    // The largest key is not the last stored key, and keys are not local IDs.
    return uint64_t{10000} + 13 * ((id * 67) % count);
  };
  index_meta_.set_major_order(IndexMeta::MO_ROW);
  const std::vector<std::string> paths{
      index_path_ + ".valid", index_path_ + ".invalid", index_path_ + ".empty"};
  auto cleanup = ScopeGuard::Make([&]() {
    for (const auto &path : paths) {
      File::RemovePath(path);
    }
  });
  IndexCluster::CentroidList centroids;
  for (float value : {0.0f, static_cast<float>(count - 1)}) {
    NumericalVector<float> vector(dimension_);
    std::fill(vector.data(), vector.data() + dimension_, 0.0f);
    vector[0] = value;
    IndexCluster::Centroid centroid;
    centroid.set_feature(vector);
    centroids.emplace_back(std::move(centroid));
  }
  IVFCentroidIndex centroid_index;
  ASSERT_EQ(0, centroid_index.init(index_meta_, Params()));
  ASSERT_EQ(0, centroid_index.build(centroids));

  std::vector<IndexStorage::Pointer> storages;
  for (size_t state = 0; state < paths.size(); ++state) {
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(paths[state]));
    IVFDumper postings(index_meta_, dumper, 2, 32);
    if (state != 2) {
      for (uint32_t id = 0; id < count; ++id) {
        NumericalVector<float> vector(dimension_);
        std::fill(vector.data(), vector.data() + dimension_, 0.0f);
        vector[0] = static_cast<float>(id);
        const uint64_t stored_key =
            state == 1 && id == invalid_id ? kInvalidKey : key(id);
        ASSERT_EQ(0,
                  postings.dump_inverted_vector(id < first_list_count ? 0 : 1,
                                                stored_key, vector.data()));
      }
    }
    ASSERT_EQ(0, postings.dump_inverted_vector_finished());
    ASSERT_EQ(0, postings.dump_centroid_index(centroid_index.data(),
                                              centroid_index.size()));
    ASSERT_EQ(0, IndexHelper::SerializeToDumper(index_meta_, dumper.get()));
    ASSERT_EQ(0, dumper->close());
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(Params()));
    ASSERT_EQ(0, storage->open(paths[state], false));
    auto body = storage->get(IVF_INVERTED_BODY_SEG_ID);
    ASSERT_NE(nullptr, body);
    if (state != 2) {
      ASSERT_NE(nullptr, body->base_data());
    }
    storages.emplace_back(std::move(storage));
  }

  IVFEntity entity;
  EXPECT_FALSE(IVFEntityTestPeer::all_keys_valid(entity));
  const IndexQueryMeta qmeta(IndexMeta::DT_FP32, dimension_);
  std::vector<float> query(dimension_, 0.0f);
  query[0] = static_cast<float>(count - 1);
  // Reusing the same entity must recompute the state in both directions.
  for (size_t state : {0U, 1U, 0U, 2U, 1U}) {
    SCOPED_TRACE(state);
    ASSERT_EQ(0, entity.load(storages[state]));
    EXPECT_EQ(state != 1, IVFEntityTestPeer::all_keys_valid(entity));
    auto clone = std::make_shared<IVFEntity>();
    ASSERT_EQ(clone, entity.clone(clone));
    EXPECT_EQ(IVFEntityTestPeer::all_keys_valid(entity),
              IVFEntityTestPeer::all_keys_valid(*clone));
    if (state == 2) {
      EXPECT_EQ(0U, clone->vector_count());
      continue;
    }
    ASSERT_EQ(32 * clone->meta().element_size(), clone->inverted_block_size());
    ASSERT_EQ(0, clone->bind_query(query.data(), qmeta));
    auto counted_keys = IVFEntityTestPeer::count_key_reads(*clone);
    for (size_t topk : {size_t{17}, size_t{second_list_count}}) {
      SCOPED_TRACE(topk);
      IndexDocumentHeap heap(topk);
      IndexContext::Stats stats;
      uint32_t scan_count = 0;
      counted_keys->reads = 0;
      ASSERT_EQ(0, clone->search(1, query.data(), &scan_count, &heap, &stats));
      EXPECT_EQ(second_list_count, scan_count);
      EXPECT_EQ(second_list_count, stats.dist_calced_count());
      if (state == 0) {
        EXPECT_EQ(0U, counted_keys->reads);
      } else {
        EXPECT_GT(counted_keys->reads, 0U);
      }
      heap.sort();
      const size_t reads_before_mapping = counted_keys->reads;
      // Entity callers resolve IDs only for the candidates retained by top-k.
      ASSERT_EQ(0, clone->retrieve_keys(&heap));
      EXPECT_EQ(heap.size(), counted_keys->reads - reads_before_mapping);
      const size_t valid_count = second_list_count - (state == 1 ? 1 : 0);
      ASSERT_EQ(std::min(topk, valid_count), heap.size());
      uint32_t expected_id = count - 1;
      for (const auto &doc : heap.container()) {
        if (state == 1 && expected_id == invalid_id) {
          --expected_id;
        }
        EXPECT_EQ(expected_id, doc.index());
        EXPECT_EQ(key(expected_id), doc.key());
        const float delta = static_cast<float>(count - 1 - expected_id);
        EXPECT_FLOAT_EQ(delta * delta, doc.score());
        --expected_id;
      }
    }
  }

  // Query across the list boundary so neither list alone contains top-k.
  // Quarter-integer coordinates give distinct, exactly representable scores.
  std::vector<float> boundary_query(dimension_, 0.0f);
  boundary_query[0] = static_cast<float>(first_list_count) - 0.25f;
  constexpr size_t boundary_topk = 7;
  const auto expected_topk = [&](float threshold, uint32_t begin,
                                 uint32_t end) {
    std::vector<std::pair<float, uint32_t>> expected;
    for (uint32_t id = begin; id < end; ++id) {
      const float delta = boundary_query[0] - static_cast<float>(id);
      const float score = delta * delta;
      if (score <= threshold) {
        expected.emplace_back(score, id);
      }
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(std::min(expected.size(), boundary_topk));
    return expected;
  };
  ASSERT_EQ(0, entity.load(storages[0]));
  ASSERT_TRUE(IVFEntityTestPeer::all_keys_valid(entity));
  ASSERT_EQ(0, entity.bind_query(boundary_query.data(), qmeta));
  auto counted_keys = IVFEntityTestPeer::count_key_reads(entity);
  for (bool reverse_lists : {false, true}) {
    SCOPED_TRACE(reverse_lists);
    for (float threshold :
         {std::numeric_limits<float>::max(), 7.5625f, 0.0625f, -1.0f}) {
      SCOPED_TRACE(threshold);
      IndexDocumentHeap heap(boundary_topk, threshold);
      IndexContext::Stats stats;
      counted_keys->reads = 0;
      for (uint32_t pass = 0; pass < 2; ++pass) {
        const uint32_t list = reverse_lists ? 1 - pass : pass;
        SCOPED_TRACE(list);
        uint32_t scan_count = 0;
        ASSERT_EQ(0, entity.search(list, boundary_query.data(), &scan_count,
                                   &heap, &stats));
        EXPECT_EQ(list == 0 ? first_list_count : second_list_count, scan_count);
        EXPECT_EQ(0U, counted_keys->reads);
        const uint32_t begin = pass == 1 || list == 0 ? 0 : first_list_count;
        const uint32_t end = pass == 1 || list == 1 ? count : first_list_count;
        const auto expected = expected_topk(threshold, begin, end);
        std::vector<std::pair<float, uint32_t>> actual;
        for (const auto &doc : heap.container()) {
          actual.emplace_back(doc.score(), doc.index());
        }
        std::sort(actual.begin(), actual.end());
        // On pass two the prefilled heap contains both nearer and farther
        // candidates than this list. Preserve the former and replace the
        // latter, regardless of list order or a partially filled radius heap.
        EXPECT_EQ(expected, actual);
        if (threshold == std::numeric_limits<float>::max()) {
          EXPECT_TRUE(heap.full());
        }
      }
      EXPECT_EQ(count, stats.dist_calced_count());
      heap.sort();
      ASSERT_EQ(0, entity.retrieve_keys(&heap));
      EXPECT_EQ(heap.size(), counted_keys->reads);
      const auto expected = expected_topk(threshold, 0, count);
      ASSERT_EQ(expected.size(), heap.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(key(expected[i].second), heap.container()[i].key());
        EXPECT_FLOAT_EQ(expected[i].first, heap.container()[i].score());
      }
    }
  }

  Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_NPROBE, 2U);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1U);
  // A deletion/filter installed after loading must still inspect real keys;
  // all stored keys being valid says nothing about this dynamic predicate.
  auto check_search = [&](auto *searcher) {
    auto context = searcher->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(17);
    for (bool filtered : {false, true}) {
      if (filtered) {
        context->set_filter(
            [&](uint64_t candidate) { return candidate == key(count - 1); });
      }
      for (bool brute_force : {false, true}) {
        SCOPED_TRACE(filtered);
        SCOPED_TRACE(brute_force);
        ASSERT_EQ(0,
                  brute_force
                      ? searcher->search_bf_impl(query.data(), qmeta, context)
                      : searcher->search_impl(query.data(), qmeta, context));
        const auto &result = context->result(0);
        ASSERT_EQ(17U, result.size());
        for (size_t i = 0; i < result.size(); ++i) {
          const uint32_t id = count - 1 - i - (filtered ? 1 : 0);
          EXPECT_EQ(key(id), result[i].key());
          const float delta = static_cast<float>(count - 1 - id);
          EXPECT_FLOAT_EQ(delta * delta, result[i].score());
        }
      }
    }
    context->reset_filter();
    context->set_topk(boundary_topk);
    for (float threshold :
         {std::numeric_limits<float>::max(), 7.5625f, 0.0625f, -1.0f}) {
      SCOPED_TRACE(threshold);
      context->set_threshold(threshold);
      const auto expected = expected_topk(threshold, 0, count);
      for (bool brute_force : {false, true}) {
        SCOPED_TRACE(brute_force);
        ASSERT_EQ(0, brute_force ? searcher->search_bf_impl(
                                       boundary_query.data(), qmeta, context)
                                 : searcher->search_impl(boundary_query.data(),
                                                         qmeta, context));
        const auto &result = context->result(0);
        ASSERT_EQ(expected.size(), result.size());
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(key(expected[i].second), result[i].key());
          EXPECT_FLOAT_EQ(expected[i].first, result[i].score());
        }
        auto *ivf_context = dynamic_cast<IVFSearcherContext *>(context.get());
        ASSERT_NE(nullptr, ivf_context);
        EXPECT_EQ(count, ivf_context->stats(0).dist_calced_count());
      }
    }
  };
  IVFSearcher searcher;
  ASSERT_EQ(0, searcher.init(search_params));
  ASSERT_EQ(0, searcher.load(storages[0], IndexMetric::Pointer()));
  check_search(&searcher);
  ASSERT_EQ(0, searcher.unload());
  IVFStreamer streamer;
  ASSERT_EQ(0, streamer.init(index_meta_, search_params));
  ASSERT_EQ(0, streamer.open(storages[0]));
  check_search(&streamer);
  ASSERT_EQ(0, streamer.close());
}

TEST_F(IVFSearcherTest, TestSimple) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 33);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)33, builder.stats().built_count());
  EXPECT_EQ((size_t)33, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(32.0f);
  }

  size_t qnum = 33;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf search
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      ASSERT_EQ((uint64_t)32 - i, result[i].key());
      ASSERT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf search
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)32 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn search
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSimpleCosine) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineNormalizeConverter");
  ASSERT_TRUE(converter != nullptr);
  auto original_index_meta = index_meta_;
  original_index_meta.set_metric("Cosine", 0, Params());
  EXPECT_EQ(0, converter->init(original_index_meta, converter_params));
  IndexMeta index_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);
  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  int ret = builder.init(index_meta, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 33);
  converter->transform(holder_);
  auto holder = converter->result();

  EXPECT_EQ(0, builder.train(threads_, holder));
  EXPECT_EQ(0, builder.build(threads_, holder));
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  EXPECT_EQ(0, dumper->create(index_path_));

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)33, builder.stats().built_count());
  EXPECT_EQ((size_t)33, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(32.0f + i);
  }

  size_t qnum = 33;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }
  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf search
  {
    size_t topk = 33;
    context->set_topk(topk);

    std::string new_vec;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->convert(query.data(), qmeta, &new_vec, &new_meta));

    ret = searcher.search_bf_impl(new_vec.data(), new_meta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < 1; ++i) {
      // ASSERT_EQ(29, result[i].key());
      EXPECT_NEAR(0, result[i].score(), 1e-2);
    }
  }
  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithBuildMemory) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  //    params_.set("proxima.hc.builder.thread_count", 1);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf search
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      ASSERT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf search
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithFilter) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  context->set_filter([](uint64_t key) {
    if (key > 0) {
      return true;
    }
    return false;
  });
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_FLOAT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_FLOAT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_FLOAT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_FLOAT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

///////////////////////////  row major ////////////////////////////////
TEST_F(IVFSearcherTest, TestRowMajorFloatWithBuildMemory) {
  index_meta_.set_major_order(IndexMeta::MO_ROW);
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorFloatWithFilter) {
  index_meta_.set_major_order(IndexMeta::MO_ROW);
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  context->set_filter([](uint64_t key) {
    if (key > 0) {
      return true;
    }
    return false;
  });
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_FLOAT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_FLOAT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_FLOAT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_FLOAT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorFloatWith1LevelAndBuildMemory) {
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "10");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 3;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWith1LevelAndBuildMemory) {
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "10");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 3;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorInt8WithBuildMemory) {
  IVFBuilder builder;
  dimension_ = 12;
  index_meta_.set_meta(IndexMeta::DataType::DT_INT8, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  size_t fnum = 128;
  prepare_int8_index_holder(0, fnum);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)fnum, builder.stats().built_count());
  EXPECT_EQ((size_t)fnum, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<int8_t> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(127);
  }

  size_t qnum = 63;
  std::vector<int8_t> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8, dimension_);

  // single bf serch
  {
    size_t topk = 128;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorInt8WithBuildMemory) {
  IVFBuilder builder;
  dimension_ = 12;
  index_meta_.set_meta(IndexMeta::DataType::DT_INT8, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  size_t fnum = 128;
  prepare_int8_index_holder(0, fnum);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)fnum, builder.stats().built_count());
  EXPECT_EQ((size_t)fnum, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<int8_t> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(127);
  }

  size_t qnum = 63;
  std::vector<int8_t> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8, dimension_);

  // single bf serch
  {
    size_t topk = 128;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSearchWithEmptyCentroid) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "3*3");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);
  size_t doc_cnt = 10;

  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  for (size_t i = 0; i < doc_cnt; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = i % 5;
    }
    holder->emplace(i, vec);
  }
  holder_.reset(holder);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");

  std::string path = "searcher_empty_centroid.index";
  ret = dumper->create(path);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, ret);
  EXPECT_EQ((size_t)10, builder.stats().built_count());
  EXPECT_EQ((size_t)10, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  dumper->close();

  IVFSearcher searcher;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(path, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    auto key1 = result[0].key();
    EXPECT_TRUE(key1 == 4ul || key1 == 9ul);
  }

  // single knn search
  {
    size_t topk = 3;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    auto key1 = result[0].key();
    auto key2 = result[1].key();
    auto key3 = result[2].key();
    EXPECT_TRUE(key1 == 4ul || key1 == 9ul);
    EXPECT_TRUE(key2 == 4ul || key2 == 9ul);
    EXPECT_TRUE(key3 == 3ul || key3 == 8ul);
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFp16WithBuildMemory) {
  const float epsilon = 1e-2;
  dimension_ = 8;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  prepare_fp16_index_holder(0, 1000);
  IVFBuilder builder;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP16, dimension_);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta1(IndexMeta::DataType::DT_FP32, dimension_);

  std::string query_buf;
  query_buf.resize(dimension_ * sizeof(uint16_t));
  std::string query1_buf;
  query1_buf.resize(dimension_ * sizeof(uint16_t) * qnum);

  IndexReformer::Pointer reformer =
      IndexFactory::CreateReformer("HalfFloatReformer");
  IndexQueryMeta qmeta;
  reformer->transform(query.data(), qmeta1, &query_buf, &qmeta);
  reformer->transform(query1.data(), qmeta1, qnum, &query1_buf, &qmeta);
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorFp16WithBuildMemory) {
  const float epsilon = 1e-2;
  dimension_ = 8;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  prepare_fp16_index_holder(0, 1000);
  IVFBuilder builder;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP16, dimension_);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta1(IndexMeta::DataType::DT_FP32, dimension_);

  std::string query_buf;
  query_buf.resize(dimension_ * sizeof(uint16_t));
  std::string query1_buf;
  query1_buf.resize(dimension_ * sizeof(uint16_t) * qnum);

  IndexReformer::Pointer reformer =
      IndexFactory::CreateReformer("HalfFloatReformer");
  IndexQueryMeta qmeta;
  reformer->transform(query.data(), qmeta1, &query_buf, &qmeta);
  reformer->transform(query1.data(), qmeta1, qnum, &query1_buf, &qmeta);
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithHnswGraphType) {
  IVFBuilder builder;
  params_.set("proxima.ivf.builder.graph_type", "hnsw");
  params_.set("proxima.ivf.builder.graph_ef", 200);
  params_.set("proxima.ivf.builder.graph_scan_ratio", 1.0);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithSsgGraphType) {
  IVFBuilder builder;
  params_.set("proxima.ivf.builder.graph_type", "ssg");
  params_.set("proxima.ivf.builder.graph_ef", 200);
  params_.set("proxima.ivf.builder.graph_scan_ratio", 1.0);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithInt8Converter) {
  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "Int8QuantizerConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithFloat16Quantizer) {
  const float epsilon = 1e-2;

  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "HalfFloatConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_fp32_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    context->set_filter([](uint64_t) { return false; });
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithConverterAndQuantizer) {
  const float epsilon = 1e-2;
  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "Int8QuantizerConverter");
  build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "HalfFloatConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_fp32_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    context->set_filter([](uint64_t) { return false; });
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      ASSERT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, KeyedBruteForceWithEmptyQuantizedCentroids) {
  const std::vector<std::vector<float>> centroid_layouts{
      {1.0f, -1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f}};
  for (const char *quantizer :
       {"Int8QuantizerConverter", "Int4QuantizerConverter"}) {
    SCOPED_TRACE(quantizer);
    for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
      SCOPED_TRACE(order);
      for (const auto &centroids : centroid_layouts) {
        SCOPED_TRACE(testing::PrintToString(centroids));
        auto meta = index_meta_;
        meta.set_metric("InnerProduct", 0, Params());
        meta.set_major_order(order);
        IndexCluster::CentroidList centroid_list;
        std::vector<IndexConverter::Pointer> quantizers;
        for (float value : centroids) {
          NumericalVector<float> centroid(dimension_);
          for (size_t d = 0; d < dimension_; ++d) {
            centroid[d] = value;
          }
          IndexCluster::Centroid item;
          item.set_feature(centroid);
          centroid_list.emplace_back(std::move(item));

          auto holder =
              std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(
                  dimension_);
          for (uint64_t i = 0; i < 10; ++i) {
            NumericalVector<float> vector(dimension_);
            for (size_t d = 0; d < dimension_; ++d) {
              vector[d] = value < 0 ? -2.0f * float(i + 1) : float(i + 1);
            }
            holder->emplace((value < 0 ? 110 : 100) + i, vector);
          }
          auto converter = IndexFactory::CreateConverter(quantizer);
          ASSERT_NE(nullptr, converter);
          ASSERT_EQ(0, converter->init(meta, Params()));
          ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, holder));
          quantizers.emplace_back(std::move(converter));
        }
        IVFCentroidIndex centroid_index;
        ASSERT_EQ(0, centroid_index.init(meta, Params()));
        ASSERT_EQ(0, centroid_index.build(centroid_list));
        auto dumper = IndexFactory::CreateDumper("FileDumper");
        ASSERT_NE(nullptr, dumper);
        ASSERT_EQ(0, dumper->create(index_path_));
        auto posting_meta = quantizers.front()->meta();
        posting_meta.set_major_order(order);
        IVFDumper postings(posting_meta, dumper, centroids.size());
        for (size_t i = 0; i < centroids.size(); ++i) {
          // Write only two lists, leaving leading, interior and trailing gaps.
          if (centroids[i] == 0.0f) {
            continue;
          }
          auto iter = quantizers[i]->result()->create_iterator();
          for (; iter->is_valid(); iter->next()) {
            ASSERT_EQ(
                0, postings.dump_inverted_vector(i, iter->key(), iter->data()));
          }
        }
        ASSERT_EQ(0, postings.dump_inverted_vector_finished());
        ASSERT_EQ(0, postings.dump_quantizer_params(quantizers));
        ASSERT_EQ(0, postings.dump_centroid_index(centroid_index.data(),
                                                  centroid_index.size()));
        ASSERT_EQ(0, IndexHelper::SerializeToDumper(meta, dumper.get()));
        ASSERT_EQ(0, dumper->close());

        for (const char *storage_name :
             {"FileReadStorage", "MMapFileReadStorage", "BufferReadStorage"}) {
          SCOPED_TRACE(storage_name);
          auto storage = IndexFactory::CreateStorage(storage_name);
          ASSERT_NE(nullptr, storage);
          ASSERT_EQ(0, storage->init(Params()));
          ASSERT_EQ(0, storage->open(index_path_, false));
          ASSERT_TRUE(storage->has(IVF_INT8_QUANTIZED_PARAMS_SEG_ID) ||
                      storage->has(IVF_INT4_QUANTIZED_PARAMS_SEG_ID));
          IVFEntity entity;
          ASSERT_EQ(0, entity.load(storage));
          ASSERT_EQ(nullptr, entity.quantizer());
          ASSERT_EQ(centroids.size(), entity.inverted_list_count());
          for (size_t i = 0; i < centroids.size(); ++i) {
            IndexStorage::MemoryBlock meta_block;
            const auto *list = entity.inverted_list_meta(i, meta_block);
            ASSERT_NE(nullptr, list);
            EXPECT_EQ(centroids[i] == 0.0f ? 0u : 10u, list->vector_count);
          }
          // The real Dumper leaves the trailing empty list's offset at zero.
          IndexStorage::MemoryBlock meta_block;
          const auto *last_list =
              entity.inverted_list_meta(centroids.size() - 1, meta_block);
          ASSERT_NE(nullptr, last_list);
          EXPECT_EQ(0u, last_list->id_offset);

          Params search_params;
          search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
          const std::vector<float> queries(dimension_ * 2, 5.0f);
          const IndexQueryMeta qmeta(IndexMeta::DT_FP32, dimension_);
          const std::vector<std::vector<uint64_t>> keys{
              {100, 109, 110, 119, 9999, kInvalidKey}, {101, 111}};
          IVFSearcher searcher;
          ASSERT_EQ(0, searcher.init(search_params));
          ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
          CheckKeyedBruteForce(&searcher, queries.data(), qmeta, keys);
          ASSERT_EQ(0, searcher.unload());
          IVFStreamer streamer;
          ASSERT_EQ(0, streamer.init(meta, search_params));
          ASSERT_EQ(0, streamer.open(storage));
          CheckKeyedBruteForce(&streamer, queries.data(), qmeta, keys);
          ASSERT_EQ(0, streamer.close());
        }
      }
    }
  }
}

TEST_F(IVFSearcherTest, TestQuantizedPerCentroid) {
  IVFBuilder builder;
  auto build_params = params_;
  auto meta = index_meta_;
  meta.set_metric("InnerProduct", 0, Params());
  build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");
  build_params.set(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, true);
  int ret = builder.init(meta, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(500.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 1; i <= dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  CheckKeyedBruteForce(&searcher, query.data(), qmeta,
                       {{11, 251, 502, 991, 10000}});

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    context->set_filter([](uint64_t) { return false; });
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    uint64_t max_rank_error = 0;
    uint64_t total_rank_error = 0;
    for (size_t i = 0; i < topk; ++i) {
      const uint64_t expected_key = (uint64_t)(total - 1) - i;
      const uint64_t actual_key = result[i].key();
      const uint64_t rank_error = expected_key > actual_key
                                      ? expected_key - actual_key
                                      : actual_key - expected_key;
      max_rank_error = std::max(max_rank_error, rank_error);
      total_rank_error += rank_error;
      float expect = (float)result[i].key() * 500.0f * dimension_;
      ASSERT_NEAR(expect, std::abs(result[i].score()), expect * 0.2 + 500000);
    }
    // K-means initialization and int8 score ties can vary by platform. Check
    // the overall approximate ranking quality without rejecting one outlier.
    EXPECT_LE(max_rank_error, 200U);
    EXPECT_LE(total_rank_error, topk * 30U);
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      ASSERT_NEAR((uint64_t)(total - 1) - q, result[0].key(), 100);
      // EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 10;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_NEAR((uint64_t)total - i - 1, result[i].key(), 100);
      // EXPECT_FLOAT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      ASSERT_NEAR((uint64_t)(total - 1) - q, result[0].key(), 100);
      // EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSharedContext) {
  size_t dim = dimension_;
  auto gen_holder = [&](int start, size_t doc_cnt) {
    auto holder =
        make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
    uint64_t key = start;
    for (size_t i = 0; i < doc_cnt; i++) {
      NumericalVector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = i;
      }
      key += 3;
      holder->emplace(key, vec);
    }
    return holder;
  };
  auto gen_index = [&](int start, size_t docs, std::string path) {
    auto holder = gen_holder(start, docs);
    IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
    Params params;
    params.set("proxima.ivf.builder.centroid_count", "16");
    builder->init(index_meta_, params);
    builder->train(holder);
    builder->build(holder);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    dumper->create(path);
    builder->dump(dumper);
    dumper->close();

    IndexSearcher::Pointer searcher =
        IndexFactory::CreateSearcher("IVFSearcher");
    auto name = rand() % 2 ? "FileReadStorage" : "MMapFileReadStorage";
    auto container = IndexFactory::CreateStorage(name);
    bool alone_file_handle = std::rand() % 2;
    bool lock_hot = std::rand() % 2;
    params.set("proxima.file.read_storage.alone_file_handle",
               alone_file_handle);
    params.set("proxima.file.read_storage.lock_hot_in_memory", lock_hot);
    container->init(params);
    container->open(path, false);
    searcher->init(Params());
    searcher->load(container, IndexMetric::Pointer());
    return searcher;
  };

  srand(Realtime::MilliSeconds());
  size_t docs1 = rand() % 500 + 100;
  size_t docs2 = rand() % 5000 + 100;
  size_t docs3 = rand() % 50000 + 100;
  auto path1 = "unittest-index/TestSharedContext.index1";
  auto path2 = "unittest-index/TestSharedContext.index2";
  auto path3 = "unittest-index/TestSharedContext.index3";
  auto searcher1 = gen_index(0, docs1, path1);
  auto searcher2 = gen_index(1, docs2, path2);
  auto searcher3 = gen_index(2, docs3, path3);

  srand(Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto do_test = [&]() {
    IndexSearcher::Context::Pointer ctx;
    switch (rand() % 3) {
      case 0:
        ctx = searcher1->create_context();
        if (rand() % 2 == 0) {
          ctx->set_filter([](uint64_t) { return false; });
        }
        break;
      case 1:
        ctx = searcher2->create_context();
        if (rand() % 2 == 0) {
          ctx->set_filter([](uint64_t) { return false; });
        }
        break;
      case 2:
        ctx = searcher3->create_context();
        if (rand() % 2 == 0) {
          ctx->set_filter([](uint64_t) { return false; });
        }
        break;
    }
    ctx->set_topk(10);

    int ret = 0;
    for (int i = 0; i < 100; ++i) {
      NumericalVector<float> query(dim);
      for (size_t j = 0; j < dim; ++j) {
        query[j] = i + 0.1f;
      }

      auto code = rand() % 6;
      switch (code) {
        case 0:
          ret = searcher1->search_impl(query.data(), qmeta, ctx);
          break;
        case 1:
          ret = searcher2->search_impl(query.data(), qmeta, ctx);
          break;
        case 2:
          ret = searcher3->search_impl(query.data(), qmeta, ctx);
          break;
        case 3:
          ret = searcher1->search_bf_impl(query.data(), qmeta, ctx);
          break;
        case 4:
          ret = searcher2->search_bf_impl(query.data(), qmeta, ctx);
          break;
        case 5:
          ret = searcher3->search_bf_impl(query.data(), qmeta, ctx);
          break;
      }

      ASSERT_EQ(0, ret);
      auto &results = ctx->result();
      EXPECT_EQ(10, results.size());
      for (int k = 0; k < 10; ++k) {
        EXPECT_EQ(code % 3, results[k].key() % 3);
      }
    }
  };
  auto t1 = std::async(std::launch::async, do_test);
  auto t2 = std::async(std::launch::async, do_test);
  t1.wait();
  t2.wait();
}

TEST_F(IVFSearcherTest, TestRnnSearch) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  size_t dim = 16;
  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = "IVFSearcherTest.TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(0, builder->cleanup());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(Params()));

  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  float radius = 1000.0f;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  EXPECT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }
  File::RemovePath(path);
}

TEST_F(IVFSearcherTest, TestProvider) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      dimension_);
  size_t doc_cnt = 5000UL;
  std::vector<uint64_t> keys(doc_cnt);
  srand(Realtime::MilliSeconds());
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < doc_cnt; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](uint64_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = keys[i];
    }
    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  ASSERT_EQ(0, builder->init(index_meta_, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = index_path_ + "/TestProvider";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcher_params;
  ASSERT_EQ(0, searcher->init(searcher_params));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
  for (size_t i = 0; i < keys.size(); ++i) {
    const float *d1 =
        reinterpret_cast<const float *>(provider->get_vector(keys[i]));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d1[j], keys[i]);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dimension_, provider->dimension());
  ASSERT_EQ(index_meta_.element_size(), provider->element_size());
  ASSERT_EQ(index_meta_.data_type(), provider->data_type());
}

TEST_F(IVFSearcherTest, ProviderReleasesStorageBeforeThreadExit) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  const size_t previous_capacity = memory_pool.capacity();
  ASSERT_EQ(0u, memory_pool.used());
  ASSERT_EQ(0, memory_pool.init(8 * 1024 * 1024));
  auto cleanup = ScopeGuard::Make([&]() {
    EXPECT_EQ(0u, memory_pool.used());
    EXPECT_EQ(0, memory_pool.init(previous_capacity));
  });

  prepare_index_holder(0, 32);
  for (bool quantized : {false, true}) {
    SCOPED_TRACE(quantized);
    Params build_params;
    build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
    if (quantized) {
      build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "HalfFloatConverter");
    }
    IVFBuilder builder;
    ASSERT_EQ(0, builder.init(index_meta_, build_params));
    ASSERT_EQ(0, builder.train(threads_, holder_));
    ASSERT_EQ(0, builder.build(threads_, holder_));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(index_path_));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    ASSERT_EQ(0, builder.cleanup());

    for (bool iterate : {true, false}) {
      SCOPED_TRACE(iterate);
      auto storage = IndexFactory::CreateStorage("BufferReadStorage");
      ASSERT_NE(nullptr, storage);
      ASSERT_EQ(0, storage->init(Params()));
      ASSERT_EQ(0, storage->open(index_path_, false));
      IVFSearcher searcher;
      ASSERT_EQ(0, searcher.init(Params()));
      ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
      auto provider = searcher.create_provider();
      ASSERT_NE(nullptr, provider);
      {
        auto iterator = provider->create_iterator();
        ASSERT_NE(nullptr, iterator);
        ASSERT_TRUE(iterator->is_valid());
        const uint64_t key = iterator->key();
        const void *data =
            iterate ? iterator->data() : provider->get_vector(key);
        ASSERT_NE(nullptr, data);
        EXPECT_FLOAT_EQ(static_cast<float>(key),
                        static_cast<const float *>(data)[0]);
      }
      ASSERT_GT(memory_pool.used(), 0u);
      provider.reset();
      ASSERT_EQ(0, searcher.unload());
      ASSERT_EQ(0, storage->close());
      storage.reset();
      // The calling thread remains alive. A TLS read block must not retain
      // the pool or its file handle after every index owner has been released.
      EXPECT_EQ(0u, memory_pool.used());
      EXPECT_EQ(0u, memory_pool.metadata_used());
    }
    // Windows refuses this removal if a provider left a file handle open.
    EXPECT_TRUE(File::RemovePath(index_path_));
  }
}

TEST_F(IVFSearcherTest, TestProviderInt8) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      dimension_);
  size_t doc_cnt = 5000UL;
  std::vector<uint64_t> keys(doc_cnt);
  srand(Realtime::MilliSeconds());
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < doc_cnt; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](uint64_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = keys[i];
    }
    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  params.set("proxima.ivf.builder.retain_original_features", false);
  auto meta = index_meta_;
  meta.set_metric("InnerProduct", 0, Params());
  params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");
  params.set(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, true);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = index_path_ + "/TestProvider";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcher_params;
  ASSERT_EQ(0, searcher->init(searcher_params));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_TRUE(!!provider);
  ASSERT_EQ(IndexMeta::DataType::DT_INT8, provider->data_type());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto d1 = reinterpret_cast<const int8_t *>(provider->get_vector(keys[i]));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_LT(d1[j], 255);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    const int8_t *d = reinterpret_cast<const int8_t *>(iter->data());
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_LT(d[j], 255);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dimension_, provider->dimension());
  ASSERT_EQ(index_meta_.element_size(), provider->element_size() * 4);
}

TEST_F(IVFSearcherTest, TestSearcherReuse) {
  auto build_index = [](IndexMeta &meta, size_t base, size_t doc_cnt,
                        std::string &path) {
    IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
    ASSERT_NE(builder, nullptr);
    IndexHolder::Pointer holder;
    if (meta.data_type() == IndexMeta::DataType::DT_INT8) {
      auto h = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_INT8>>(
          meta.dimension());
      for (size_t i = base; i < doc_cnt; i++) {
        NumericalVector<int8_t> vec(meta.dimension());
        for (size_t j = 0; j < meta.dimension(); ++j) {
          vec[j] = i;
        }
        ASSERT_TRUE(h->emplace(i, vec));
      }
      holder = h;
    } else if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
      auto h = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          meta.dimension());
      for (size_t i = base; i < doc_cnt; i++) {
        NumericalVector<float> vec(meta.dimension());
        for (size_t j = 0; j < meta.dimension(); ++j) {
          vec[j] = i;
        }
        ASSERT_TRUE(h->emplace(i, vec));
      }
      holder = h;
    }
    Params params;
    LOG_DEBUG("Build index %s count=%zu", path.c_str(), holder->count());
    params.set("proxima.ivf.builder.centroid_count", "10");
    ASSERT_EQ(0, builder->init(meta, params));
    ASSERT_EQ(0, builder->train(holder));
    ASSERT_EQ(0, builder->build(holder));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, builder->dump(dumper));
    ASSERT_EQ(0, dumper->close());
    ASSERT_EQ(0, builder->cleanup());
  };

  auto path1 = index_path_ + "/index1";
  auto path2 = index_path_ + "/index2";
  IndexMeta meta1(IndexMeta::DataType::DT_INT8, 16);
  IndexMeta meta2(IndexMeta::DataType::DT_FP32, 31);
  build_index(meta1, 10, 200, path1);
  build_index(meta2, 2000, 3000, path2);

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcher_params;
  ASSERT_EQ(0, searcher->init(searcher_params));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path1, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_INT8, searcher->meta().data_type());
  ASSERT_EQ(190UL, searcher->stats().loaded_count());
  ASSERT_EQ(190UL, provider->count());
  ASSERT_EQ("IVFSearcher", provider->owner_class());
  for (size_t i = 10; i < 200ul; ++i) {
    const int8_t *d1 =
        reinterpret_cast<const int8_t *>(provider->get_vector(i));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < meta1.dimension(); ++j) {
      ASSERT_EQ(d1[j], (int8_t)i);
    }
  }
  ASSERT_EQ(meta1.dimension(), provider->dimension());
  ASSERT_EQ(meta1.element_size(), provider->element_size());
  ASSERT_EQ(meta1.data_type(), provider->data_type());
  ASSERT_EQ(0, searcher->unload());
  ASSERT_EQ(0, searcher->cleanup());

  auto container2 = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container2->open(path2, false));
  ASSERT_EQ(0, searcher->init(searcher_params));
  ASSERT_EQ(0, searcher->load(container2, IndexMetric::Pointer()));

  auto provider2 = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, searcher->meta().data_type());
  for (size_t i = 2000; i < 3000ul; ++i) {
    const float *d1 = reinterpret_cast<const float *>(provider2->get_vector(i));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < meta2.dimension(); ++j) {
      ASSERT_FLOAT_EQ(d1[j], i);
    }
  }
  ASSERT_EQ(meta2.dimension(), provider2->dimension());
  ASSERT_EQ(meta2.element_size(), provider2->element_size());
  ASSERT_EQ(meta2.data_type(), provider2->data_type());
  ASSERT_EQ(1000UL, provider2->count());
  ASSERT_EQ(1000UL, searcher->stats().loaded_count());
}

TEST_F(IVFSearcherTest, TestInt8QuantizerWithL2) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      dimension_);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  params.set("proxima.ivf.builder.store_original_features", true);
  auto meta = index_meta_;
  params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = index_path_ + "/TestQuantizer";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcher_params;
  ASSERT_EQ(0, searcher->init(searcher_params));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
  for (size_t i = 0; i < doc_cnt; ++i) {
    const float *d1 = reinterpret_cast<const float *>(provider->get_vector(i));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d1[j], i);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dimension_, provider->dimension());
  ASSERT_EQ(index_meta_.element_size(), provider->element_size());
  ASSERT_EQ(index_meta_.data_type(), provider->data_type());

  auto context = searcher->create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  const std::vector<float> candidate_query(dimension_, 321.0f);
  CheckKeyedBruteForce(searcher.get(), candidate_query.data(), qmeta,
                       {{7, 301, 998, 4321, 10000}});
  size_t topk = 1;
  context->set_topk(topk);
  context->set_filter([](uint64_t) { return false; });
  for (size_t i = 0; i < doc_cnt; i += 20) {
    NumericalVector<float> query(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      query[j] = i;
    }
    int ret = searcher->search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);
    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    ASSERT_NEAR(i, result[0].key(), 100);
  }
}

TEST_F(IVFSearcherTest, TestMipsEuclideanMetric) {
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("MipsSquaredEuclidean", 0, Params());
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  const size_t COUNT = 10000UL;
  for (size_t i = 0; i < COUNT; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 100.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  Params builder_params;
  builder_params.set("proxima.ivf.builder.centroid_count", 1024);
  ASSERT_EQ(0, builder->init(meta, builder_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  std::string path = "IVFTestMipsEuclideanMetric";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params params;
  params.set("proxima.ivf.searcher.scan_ratio", 0.1f);
  ASSERT_EQ(0, searcher->init(params));

  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 10;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  EXPECT_EQ(results.size(), topk);
  EXPECT_NEAR((uint64_t)(COUNT - 1), results[0].key(), 10);
  File::RemovePath(path);
}

TEST_F(IVFSearcherTest, TestSameValue) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params_.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_same_index_holder(0, 33);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)33, builder.stats().built_count());
  EXPECT_EQ((size_t)33, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(32.0f);
  }

  size_t qnum = 33;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      // std::cout << "i: " << i << ", key: " << result[i].key() << ", score: "
      // << result[i].score() << std::endl;
      ASSERT_EQ(0, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_FLOAT_EQ((float)0, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_FLOAT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

// Test: Builder with CONVERTER_CLASS produces index that auto-converts FP32
// queries
TEST_F(IVFSearcherTest, TestConverterClassEndToEnd) {
  const float epsilon = 1e-2;

  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  build_params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "HalfFloatConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);

  int total = 1000;
  prepare_fp32_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ(0, dumper->close());

  // Load and search with FP32 query - reformer should auto-convert
  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // The searcher should automatically apply reformer to convert FP32 query
  // to FP16 before searching
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ(topk, result.size());
    // First result should be key 0 (closest to query -0.1)
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ(topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

// Test: nprobe parameter overrides scan_ratio for centroid selection
TEST_F(IVFSearcherTest, TestNprobeParameter) {
  // Build index with 16 centroids, 1000 vectors
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  dimension_ = 32;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);

  // Prepare random data so centroids get varying amount of vectors
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  std::srand(42);
  for (size_t i = 0; i < 1000; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = std::rand() % 1000 * 1.0f;
    }
    holder->emplace(i, vec);
  }
  holder_.reset(holder);

  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);
  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());

  // Load searcher
  IVFSearcher searcher;
  Params search_params;
  // scan_ratio=0.1 → selects ~2 centroids out of 16
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = searcher.init(search_params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);
  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);
  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query(dimension_, 500.0f);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  auto context = searcher.create_context();
  size_t topk = 10;
  context->set_topk(topk);

  // Search with scan_ratio only (few centroids)
  ret = searcher.search_impl(query.data(), qmeta, context);
  EXPECT_EQ(0, ret);
  const IndexDocumentList &result_low = context->result(0);
  size_t found_low = result_low.size();
  EXPECT_GT(found_low, 0u);

  // Now update context with nprobe=16 (all centroids)
  Params nprobe_params;
  nprobe_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  nprobe_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)16);
  nprobe_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(nprobe_params);
  EXPECT_EQ(0, ret);

  context->set_topk(topk);
  ret = searcher.search_impl(query.data(), qmeta, context);
  EXPECT_EQ(0, ret);
  const IndexDocumentList &result_high = context->result(0);
  size_t found_high = result_high.size();
  EXPECT_EQ(found_high, topk);

  // With more centroids searched, the best score should be <= the score
  // from fewer centroids (at least as good)
  EXPECT_LE(result_high[0].score(), result_low[0].score());

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

// Test: nprobe=1 should only search 1 centroid
TEST_F(IVFSearcherTest, TestNprobeOne) {
  // Build index with many centroids
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "8");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);
  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = searcher.init(search_params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);
  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);
  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query(dimension_, 999.0f);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  auto context = searcher.create_context();

  // Update with nprobe=1, so only 1 centroid is selected
  Params nprobe_params;
  nprobe_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  nprobe_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)1);
  nprobe_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(nprobe_params);
  EXPECT_EQ(0, ret);

  size_t topk = 1000;
  context->set_topk(topk);
  ret = searcher.search_impl(query.data(), qmeta, context);
  EXPECT_EQ(0, ret);

  const IndexDocumentList &result = context->result(0);
  // With nprobe=1, we should get fewer results than total since only
  // 1 out of 8 centroids is searched
  EXPECT_GT(result.size(), 0u);
  EXPECT_LT(result.size(), 1000u);

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

// Test: explicit nprobe scans every selected inverted list.
TEST_F(IVFSearcherTest, TestNprobeScansAllSelectedLists) {
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);

  auto trainer =
      CreateFixedCentroidTrainer(index_meta_, {0.0f, 100.0f, 200.0f, 300.0f});
  ASSERT_TRUE(!!trainer);
  ret = builder.train(trainer);
  EXPECT_EQ(0, ret);

  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  auto append_vectors = [&](uint32_t base_key, size_t count, float value) {
    for (size_t i = 0; i < count; ++i) {
      NumericalVector<float> vec(dimension_);
      for (size_t j = 0; j < dimension_; ++j) {
        vec[j] = value;
      }
      holder->emplace(base_key + i, vec);
    }
  };
  append_vectors(0, 80, 0.0f);
  append_vectors(80, 10, 100.0f);
  append_vectors(90, 5, 200.0f);
  append_vectors(95, 5, 300.0f);
  holder_.reset(holder);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);
  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = searcher.init(search_params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);
  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);
  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query(dimension_, 0.0f);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  auto context = searcher.create_context();
  auto *ivf_ctx = dynamic_cast<IVFSearcherContext *>(context.get());
  ASSERT_NE(ivf_ctx, nullptr);

  Params nprobe_params;
  nprobe_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  nprobe_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)2);
  nprobe_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(nprobe_params);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(ivf_ctx->max_scan_count(), 100u);

  size_t topk = 100;
  context->set_topk(topk);
  ret = searcher.search_impl(query.data(), qmeta, context);
  EXPECT_EQ(0, ret);
  const IndexDocumentList &result = context->result(0);
  EXPECT_EQ(result.size(), 90u);

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

// Test: verify max_scan_count value directly via IVFSearcherContext
TEST_F(IVFSearcherTest, TestNprobeMaxScanCountValue) {
  // Build a small index: 4 centroids, 400 vectors
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 400);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);
  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = searcher.init(search_params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);
  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);
  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  auto context = searcher.create_context();
  auto *ivf_ctx = dynamic_cast<IVFSearcherContext *>(context.get());
  ASSERT_NE(ivf_ctx, nullptr);

  // Default: scan_ratio=0.1, 400 vectors -> max_scan_count = ceil(400*0.1) = 40
  EXPECT_EQ(ivf_ctx->max_scan_count(), 40u);

  // Explicit nprobe uses vector_count as the scan cap.
  Params nprobe2_params;
  nprobe2_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  nprobe2_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)2);
  nprobe2_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(nprobe2_params);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(ivf_ctx->max_scan_count(), 400u);

  // Set nprobe=4 (all clusters).
  Params nprobe4_params;
  nprobe4_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  nprobe4_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)4);
  nprobe4_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(nprobe4_params);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(ivf_ctx->max_scan_count(), 400u);

  // Set nprobe=1.
  Params nprobe1_params;
  nprobe1_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  nprobe1_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)1);
  nprobe1_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(nprobe1_params);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(ivf_ctx->max_scan_count(), 400u);

  // Without nprobe (nprobe=0), falls back to scan_ratio
  Params no_nprobe_params;
  no_nprobe_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.5);
  no_nprobe_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(no_nprobe_params);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(ivf_ctx->max_scan_count(), 200u);  // ceil(400*0.5) = 200

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

// Test: nprobe exceeding inverted_list_count is clamped
TEST_F(IVFSearcherTest, TestNprobeClampToListCount) {
  // Build index with 4 centroids and 400 vectors
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 400);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);
  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = searcher.init(search_params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);
  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);
  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  auto context = searcher.create_context();
  auto *ivf_ctx = dynamic_cast<IVFSearcherContext *>(context.get());
  ASSERT_NE(ivf_ctx, nullptr);

  // Set nprobe=100, far exceeding 4 centroids.
  // It should be clamped to 4 selected centroids and scan those lists fully.
  Params over_params;
  over_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 0.1);
  over_params.set(PARAM_IVF_SEARCHER_NPROBE, (uint32_t)100);
  over_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
  ret = context->update(over_params);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(ivf_ctx->max_scan_count(), 400u);

  // Verify search still works correctly with clamped nprobe
  std::vector<float> query(dimension_, 1.0f);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  context->set_topk(400);
  ret = searcher.search_impl(query.data(), qmeta, context);
  EXPECT_EQ(0, ret);
  const IndexDocumentList &result = context->result(0);
  EXPECT_EQ(result.size(), 400u);

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

namespace zvec {
namespace core {
namespace {

// The dimension produces INT4 records with a byte count not divisible by four;
// queries and encoded postings have different strides and block padding.
constexpr uint32_t kDimension = 18;
constexpr size_t kVectorCount = 67;
constexpr uint32_t kTopK = 11;
constexpr uint32_t kQueryCount = 3;

uint64_t Key(size_t i) {
  return 100 + 13 * i;
}

using TurboCase = std::tuple<const char *, const char *>;

// Preserve the file's structure while withholding the serialized codebook.
class MissingQuantizerStateDumper : public IndexDumper {
 public:
  MissingQuantizerStateDumper()
      : dumper_(IndexFactory::CreateDumper("FileDumper")) {}

  int init(const ailego::Params &params) override {
    return dumper_->init(params);
  }
  int cleanup() override {
    return dumper_->cleanup();
  }
  int create(const std::string &path) override {
    return dumper_->create(path);
  }
  int close() override {
    return dumper_->close();
  }
  int append(const std::string &id, size_t size, size_t padding,
             uint32_t crc) override {
    return dumper_->append(
        id == IVF_TURBO_QUANTIZER_SEG_ID ? "missing_state" : id, size, padding,
        crc);
  }
  size_t write(const void *data, size_t size) override {
    return dumper_->write(data, size);
  }
  uint32_t magic() const override {
    return dumper_->magic();
  }

 private:
  IndexDumper::Pointer dumper_;
};

class IVFTurboTest : public testing::TestWithParam<TurboCase> {
 protected:
  void SetUp() override {
    const auto *info = testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::string("ivf_turbo_") + info->name() + ".index";
    std::replace(path_.begin(), path_.end(), '/', '_');
    ailego::File::RemovePath(path_);

    meta_.set_meta(IndexMeta::DataType::DT_FP32, kDimension);
    meta_.set_metric(std::get<1>(GetParam()), 0, ailego::Params());
    ailego::Params quantizer_params;
    if (std::string(std::get<0>(GetParam())) == "PqInt4Quantizer") {
      quantizer_params.set("num_chunk", 3U);
      quantizer_params.set("use_zero_mean", true);
      quantizer_params.set("thread_count", 1U);
    }
    meta_.set_quantizer(std::get<0>(GetParam()), 0, quantizer_params);
    quantizer_ = IndexFactory::CreateQuantizer(std::get<0>(GetParam()));
    ASSERT_NE(nullptr, quantizer_);
    ASSERT_EQ(0, quantizer_->init(meta_, quantizer_params));

    std::mt19937 random(20260910);
    std::uniform_real_distribution<float> value(-2.0f, 3.0f);
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
            kDimension);
    for (size_t i = 0; i < kVectorCount; ++i) {
      ailego::NumericalVector<float> row(kDimension);
      for (uint32_t j = 0; j < kDimension; ++j) {
        // Vary the norms so cosine exercises normalization and its extra
        // metadata instead of accidentally receiving unit vectors.
        row[j] = value(random) * (0.4f + 0.07f * (i % 9));
      }
      vectors_.emplace_back(row.data(), row.data() + kDimension);
      holder->emplace(Key(i), row);
    }
    for (size_t i : {7U, 31U, 52U}) {
      queries_.insert(queries_.end(), vectors_[i].begin(), vectors_[i].end());
    }
    queries_[kDimension + 2] += 0.17f;

    ailego::Params params;
    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
    params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
    params.set(PARAM_IVF_BUILDER_THREAD_COUNT, 1U);
    params.set(PARAM_IVF_BUILDER_BLOCK_VECTOR_COUNT, 8U);
    IVFBuilder builder;
    ASSERT_EQ(0, builder.init(meta_, params, quantizer_));
    ASSERT_EQ(0, builder.train(IndexThreads::Pointer(), holder));
    ASSERT_NE(nullptr, builder.centroid_index());
    EXPECT_EQ(4U, builder.centroid_index()->centroids_count());
    EXPECT_EQ(IndexMeta::DataType::DT_FP32,
              builder.centroid_index()->meta().data_type());
    ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    EXPECT_EQ(kVectorCount, builder.stats().dumped_count());
    auto missing_state = std::make_shared<MissingQuantizerStateDumper>();
    ASSERT_EQ(0, missing_state->create(path_ + ".missing"));
    ASSERT_EQ(0, builder.dump(missing_state));
    ASSERT_EQ(0, missing_state->close());

    for (const auto &row : vectors_) {
      codes_.emplace_back(quantizer_->quantized_datapoint_vector_length(),
                          '\0');
      quantizer_->quantize_data(row.data(), codes_.back().data());
    }
  }

  void TearDown() override {
    ailego::File::RemovePath(path_);
    ailego::File::RemovePath(path_ + ".missing");
    ailego::File::RemovePath(path_ + ".format");
  }

  ailego::Params search_params() const {
    ailego::Params params;
    params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
    params.set(PARAM_IVF_SEARCHER_NPROBE, 4U);
    // Force the ANN entry point to select and scan the centroid lists.
    params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1U);
    return params;
  }

  IndexStorage::Pointer open_storage(const char *name) {
    auto storage = IndexFactory::CreateStorage(name);
    EXPECT_NE(nullptr, storage);
    if (!storage) {
      return nullptr;
    }
    EXPECT_EQ(0, storage->init(ailego::Params()));
    EXPECT_EQ(0, storage->open(path_, false));
    return storage;
  }

  void check_result(const IndexDocumentList &result, const float *query,
                    bool filtered) const {
    std::vector<std::pair<uint64_t, float>> expected;
    for (size_t i = 0; i < codes_.size(); ++i) {
      if (filtered && Key(i) % 2 == 0) {
        continue;
      }
      expected.emplace_back(Key(i),
                            quantizer_->calc_distance_dp_query_unquantized(
                                codes_[i].data(), query));
    }
    std::sort(expected.begin(), expected.end(),
              [](const auto &left, const auto &right) {
                return left.second < right.second;
              });
    ASSERT_EQ(kTopK, result.size());
    std::vector<uint64_t> seen;
    for (size_t rank = 0; rank < result.size(); ++rank) {
      SCOPED_TRACE(testing::Message() << "rank=" << rank);
      const auto found = std::find_if(
          expected.begin(), expected.end(),
          [&](const auto &row) { return row.first == result[rank].key(); });
      ASSERT_NE(expected.end(), found);
      EXPECT_EQ(seen.end(),
                std::find(seen.begin(), seen.end(), result[rank].key()));
      seen.push_back(result[rank].key());
      const float tolerance = 1e-4f * std::max(1.0f, std::abs(found->second));
      EXPECT_NEAR(found->second, result[rank].score(), tolerance);
      // Compare ranks by score so legitimate quantization ties need not use
      // the same key ordering as std::sort.
      EXPECT_NEAR(expected[rank].second, result[rank].score(), tolerance);
    }
  }

  template <class Searcher>
  void check_searches(Searcher *searcher) const {
    const auto original_queries = queries_;
    const IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDimension);
    auto context = searcher->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(kTopK);
    for (bool filtered : {false, true}) {
      context->set_filter(
          [filtered](uint64_t key) { return filtered && key % 2 == 0; });
      ASSERT_EQ(0, searcher->search_bf_impl(queries_.data(), qmeta, context));
      check_result(context->result(0), queries_.data(), filtered);
      ASSERT_EQ(0, searcher->search_impl(queries_.data(), qmeta, context));
      check_result(context->result(0), queries_.data(), filtered);
      ASSERT_EQ(0, searcher->search_bf_impl(queries_.data(), qmeta, kQueryCount,
                                            context));
      for (uint32_t q = 0; q < kQueryCount; ++q) {
        check_result(context->result(q), queries_.data() + q * kDimension,
                     filtered);
      }
      ASSERT_EQ(0, searcher->search_impl(queries_.data(), qmeta, kQueryCount,
                                         context));
      for (uint32_t q = 0; q < kQueryCount; ++q) {
        check_result(context->result(q), queries_.data() + q * kDimension,
                     filtered);
      }
    }
    CheckKeyedBruteForce(
        searcher, queries_.data(), qmeta,
        {{Key(52), Key(7), Key(31), Key(kVectorCount + 1), kInvalidKey},
         {},
         {Key(19), Key(44)}});
    EXPECT_EQ(original_queries, queries_);
  }

  void check_decoded_provider(IndexProvider::Pointer provider) const {
    ASSERT_NE(nullptr, provider);
    EXPECT_EQ(kVectorCount, provider->count());
    EXPECT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
    ASSERT_EQ(kDimension * sizeof(float), provider->element_size());
    IndexQueryMeta encoded_meta;
    std::string encoded;
    ASSERT_EQ(0, quantizer_->quantize(
                     vectors_[0].data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                     &encoded, &encoded_meta));
    auto iterator = provider->create_iterator();
    ASSERT_NE(nullptr, iterator);
    for (size_t i = 0; i < kVectorCount; ++i) {
      ASSERT_TRUE(iterator->is_valid());
      EXPECT_EQ(Key(i), iterator->key());
      std::string decoded;
      ASSERT_EQ(
          0, quantizer_->dequantize(codes_[i].data(), encoded_meta, &decoded));
      ASSERT_EQ(provider->element_size(), decoded.size());
      EXPECT_EQ(0,
                std::memcmp(decoded.data(), iterator->data(), decoded.size()));
      iterator->next();
    }
    EXPECT_FALSE(iterator->is_valid());
  }

  IndexMeta meta_;
  turbo::Quantizer::Pointer quantizer_;
  std::vector<std::vector<float>> vectors_;
  std::vector<float> queries_;
  std::vector<std::string> codes_;
  std::string path_;
};

TEST_P(IVFTurboTest, PostingFormatDetection) {
  for (bool future_version : {false, true}) {
    SCOPED_TRACE(future_version);
    auto posting_meta = quantizer_->meta();
    posting_meta.set_quantizer(meta_.quantizer_name(),
                               meta_.quantizer_revision(),
                               meta_.quantizer_params());
    posting_meta.set_major_order(IndexMeta::MO_ROW);
    posting_meta.mutable_attributes()->erase(
        IVF_TURBO_FORMAT_VERSION_ATTRIBUTE);
    if (future_version) {
      posting_meta.mutable_attributes()->set(IVF_TURBO_FORMAT_VERSION_ATTRIBUTE,
                                             kIVFTurboFormatVersion + 1);
    }
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_ + ".format"));
    // Reproduce the earlier Turbo format without the new metadata attribute.
    IVFDumper postings(posting_meta, dumper, 1, 8);
    for (size_t i = 0; i < codes_.size(); ++i) {
      ASSERT_EQ(0, postings.dump_inverted_vector(0, Key(i), codes_[i].data()));
    }
    ASSERT_EQ(0, postings.dump_inverted_vector_finished());
    ASSERT_EQ(0, postings.dump_turbo_quantizer(quantizer_));
    ASSERT_EQ(0, IndexHelper::SerializeToDumper(meta_, dumper.get()));
    ASSERT_EQ(0, dumper->close());
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path_ + ".format", false));
    IVFEntity entity;
    if (future_version) {
      EXPECT_EQ(IndexError_InvalidFormat, entity.load(storage));
    } else {
      ASSERT_EQ(0, entity.load(storage));
      ASSERT_NE(nullptr, entity.quantizer());
      for (size_t i = 0; i < codes_.size(); ++i) {
        IndexStorage::MemoryBlock code;
        ASSERT_EQ(0, entity.get_vector_by_key(Key(i), code));
        ASSERT_NE(nullptr, code.data());
        EXPECT_EQ(0,
                  std::memcmp(codes_[i].data(), code.data(), codes_[i].size()));
      }
    }
  }
}

TEST_P(IVFTurboTest, ProviderConcurrentReadsKeepReturnedBytesStable) {
  for (const char *storage_name :
       {"FileReadStorage", "MMapFileReadStorage", "BufferReadStorage"}) {
    SCOPED_TRACE(storage_name);
    auto storage = open_storage(storage_name);
    ASSERT_NE(nullptr, storage);
    IVFSearcher searcher;
    ASSERT_EQ(0, searcher.init(search_params()));
    ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
    auto provider = searcher.create_provider();
    ASSERT_NE(nullptr, provider);
    auto iterator = provider->create_iterator();
    ASSERT_NE(nullptr, iterator);

    const auto *first = static_cast<const char *>(provider->get_vector(Key(0)));
    ASSERT_NE(nullptr, first);
    const std::string expected(first, provider->element_size());
    IndexStorage::MemoryBlock owned;
    ASSERT_EQ(0, provider->get_vector(Key(0), owned));
    // Start without priming iterator data or its lazy mapping cache.
    std::vector<std::future<void>> readers;
    for (size_t worker = 1; worker <= 4; ++worker) {
      readers.push_back(std::async(std::launch::async, [&, worker]() {
        auto other = provider->create_iterator();
        ASSERT_NE(nullptr, other);
        for (size_t i = 0; i < worker; ++i) {
          other->next();
        }
        for (size_t repeat = 0; repeat < 32; ++repeat) {
          ASSERT_TRUE(iterator->is_valid());
          EXPECT_EQ(Key(0), iterator->key());
          const void *data = iterator->data();
          ASSERT_NE(nullptr, data);
          ASSERT_NE(nullptr, other->data());
          ASSERT_NE(nullptr, provider->get_vector(Key(worker)));
          EXPECT_EQ(0, std::memcmp(expected.data(), data, expected.size()));
          EXPECT_EQ(0, iterator->status());
        }
      }));
    }
    for (auto &reader : readers) {
      reader.get();
    }
    EXPECT_EQ(0, std::memcmp(expected.data(), first, expected.size()));
    ASSERT_NE(nullptr, provider->get_vector(Key(5)));
    EXPECT_EQ(0, std::memcmp(expected.data(), owned.data(), expected.size()));
    EXPECT_EQ(0,
              std::memcmp(expected.data(), iterator->data(), expected.size()));
    iterator->next();
    EXPECT_EQ(Key(1), iterator->key());
    const void *second = provider->get_vector(Key(1));
    ASSERT_NE(nullptr, second);
    ASSERT_NE(nullptr, iterator->data());
    EXPECT_EQ(0, std::memcmp(second, iterator->data(), expected.size()));
  }
}

TEST_P(IVFTurboTest, RawQueriesAndPostingCodesSurviveReopen) {
  // No quantizer is injected: all storage implementations must reconstruct
  // it from the dedicated persisted metadata and serialized state.
  for (const char *storage_name :
       {"FileReadStorage", "MMapFileReadStorage", "BufferReadStorage"}) {
    SCOPED_TRACE(storage_name);
    auto storage = open_storage(storage_name);
    ASSERT_NE(nullptr, storage);
    IVFSearcher searcher;
    ASSERT_EQ(0, searcher.init(search_params()));
    ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
    ASSERT_NE(nullptr, searcher.quantizer());
    EXPECT_EQ(kDimension, searcher.meta().dimension());
    EXPECT_EQ(IndexMeta::DataType::DT_FP32, searcher.meta().data_type());
    check_decoded_provider(searcher.create_provider());
    check_searches(&searcher);
    auto retained_context = searcher.create_context();
    ASSERT_NE(nullptr, retained_context);
    retained_context->set_topk(kTopK);
    const IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDimension);
    ASSERT_EQ(0,
              searcher.search_impl(queries_.data(), qmeta, retained_context));
    ASSERT_EQ(0, searcher.unload());
    ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
    ASSERT_EQ(0,
              searcher.search_impl(queries_.data(), qmeta, retained_context));
    check_result(retained_context->result(0), queries_.data(), false);
    ASSERT_EQ(0, searcher.unload());
  }

  auto storage = open_storage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  IVFStreamer streamer;
  ASSERT_EQ(0, streamer.init(meta_, search_params()));
  ASSERT_EQ(0, streamer.open(storage));
  ASSERT_NE(nullptr, streamer.quantizer());
  check_decoded_provider(streamer.create_provider());
  for (size_t i = 0; i < codes_.size(); ++i) {
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0,
              streamer.get_vector_by_id(static_cast<uint32_t>(Key(i)), block));
    ASSERT_NE(nullptr, block.data());
    EXPECT_EQ(0, std::memcmp(codes_[i].data(), block.data(), codes_[i].size()));
  }
  check_searches(&streamer);
  ASSERT_EQ(0, streamer.close());

  {
    // Even stateless Turbo quantizers require the state segment when marked.
    auto missing_state = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, missing_state);
    ASSERT_EQ(0, missing_state->init(ailego::Params()));
    ASSERT_EQ(0, missing_state->open(path_ + ".missing", false));
    IVFSearcher searcher;
    ASSERT_EQ(0, searcher.init(search_params()));
    EXPECT_EQ(IndexError_InvalidFormat,
              searcher.load(missing_state, IndexMetric::Pointer()));
    IVFStreamer missing_streamer;
    ASSERT_EQ(0, missing_streamer.init(meta_, search_params()));
    EXPECT_EQ(IndexError_InvalidFormat, missing_streamer.open(missing_state));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ScalarFormatsAndMetrics, IVFTurboTest,
    testing::Combine(testing::Values("Fp32Quantizer", "Fp16Quantizer",
                                     "Int8Quantizer", "Int4Quantizer"),
                     testing::Values("SquaredEuclidean", "InnerProduct",
                                     "Cosine")),
    [](const testing::TestParamInfo<TurboCase> &info) {
      return std::string(std::get<0>(info.param)) + "_" +
             std::get<1>(info.param);
    });

INSTANTIATE_TEST_SUITE_P(TrainedCodebook, IVFTurboTest,
                         testing::Values(TurboCase{"PqInt4Quantizer",
                                                   "SquaredEuclidean"}));

TEST(IVFTurboConfiguration, NullOptionalQuantizerUsesLegacyInitialization) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  IVFSearcher searcher;
  EXPECT_EQ(0, searcher.init(ailego::Params(), nullptr));
  IVFStreamer streamer;
  EXPECT_EQ(0, streamer.init(meta, ailego::Params(), nullptr));
  IVFBuilder builder;
  ailego::Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  EXPECT_EQ(0, builder.init(meta, params, nullptr));
}

class FixedAngularTrainer : public IndexTrainer {
 public:
  FixedAngularTrainer(IndexMeta meta, IndexBundle::Pointer bundle)
      : meta_(std::move(meta)), bundle_(std::move(bundle)) {}
  int init(const IndexMeta &, const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int train(IndexThreads::Pointer, IndexHolder::Pointer) override {
    return 0;
  }
  int load(IndexStorage::Pointer) override {
    return 0;
  }
  int dump(const IndexDumper::Pointer &) override {
    return 0;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  IndexBundle::Pointer indexes() const override {
    return bundle_;
  }

 private:
  IndexMeta meta_;
  IndexBundle::Pointer bundle_;
  Stats stats_;
};

TEST(IVFTurboCentroids, CosineNprobeUsesEveryCoordinateAndIgnoresVectorNorm) {
  constexpr uint32_t dim = 8;
  constexpr uint32_t group_size = 6;
  const std::string path = "ivf_turbo_cosine_nprobe.index";
  struct Cleanup {
    const std::string &path;
    ~Cleanup() {
      ailego::File::RemovePath(path);
    }
  } cleanup{path};
  ailego::File::RemovePath(path);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, dim);
  meta.set_metric("Cosine", 0, ailego::Params());
  meta.set_quantizer("Int8Quantizer", 0, ailego::Params());
  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  for (uint32_t group = 0; group < 2; ++group) {
    for (uint32_t i = 0; i < group_size; ++i) {
      ailego::NumericalVector<float> vector(dim);
      std::fill(vector.data(), vector.data() + dim, 0.0f);
      const float scale = static_cast<float>(1U << i) + group * 0.25f;
      vector[0] = scale;
      // The final logical coordinate distinguishes the two angular groups.
      // Dropping it as an implicit norm tail makes their directions identical.
      vector[dim - 1] = (group == 0 ? 3.0f : -3.0f) * scale;
      holder->emplace(group * group_size + i, vector);
    }
  }
  ailego::Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  params.set(PARAM_IVF_BUILDER_THREAD_COUNT, 1U);
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params, quantizer));
  // Known centers isolate centroid assignment and query transformation from
  // K-means random initialization on a corpus with only two distinct angles.
  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  auto trainer_meta = converter->meta();
  trainer_meta.set_quantizer("", 0, ailego::Params());
  IndexCluster::CentroidList centroids;
  const float norm = std::sqrt(10.0f);
  for (uint32_t group = 0; group < 2; ++group) {
    ailego::NumericalVector<float> center(dim + 1);
    std::fill(center.data(), center.data() + dim + 1, 0.0f);
    center[0] = 1.0f / norm;
    center[dim - 1] = (group == 0 ? 3.0f : -3.0f) / norm;
    center[dim] = norm;
    IndexCluster::Centroid centroid;
    centroid.set_feature(center);
    centroids.emplace_back(std::move(centroid));
  }
  IndexBundle::Pointer bundle;
  ASSERT_EQ(0, IndexCluster::Serialize(trainer_meta, centroids, &bundle));
  ASSERT_EQ(0, builder.train(std::make_shared<FixedAngularTrainer>(
                   trainer_meta, std::move(bundle))));
  ASSERT_NE(nullptr, builder.centroid_index());
  ASSERT_EQ(2U, builder.centroid_index()->centroids_count());
  EXPECT_EQ("CosineFp32Converter",
            builder.centroid_index()->meta().converter_name());
  EXPECT_EQ(dim + 1, builder.centroid_index()->meta().dimension());
  ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder.dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, false));
  auto segment = storage->get(IVF_CENTROID_SEG_ID);
  ASSERT_NE(nullptr, segment);
  IndexSegmentStorage centroid_storage(segment);
  ASSERT_EQ(0, centroid_storage.open("", false));
  IndexMeta centroid_meta;
  ASSERT_EQ(0, IndexHelper::DeserializeFromStorage(&centroid_storage,
                                                   &centroid_meta));
  EXPECT_EQ("CosineFp32Converter", centroid_meta.converter_name());
  EXPECT_EQ(dim + 1, centroid_meta.dimension());

  ailego::Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_NPROBE, 1U);
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0U);
  IVFSearcher searcher;
  ASSERT_EQ(0, searcher.init(search_params));
  ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
  auto context = searcher.create_context();
  ASSERT_NE(nullptr, context);
  auto *ivf_context = dynamic_cast<IVFSearcherContext *>(context.get());
  ASSERT_NE(nullptr, ivf_context);
  EXPECT_EQ(0U, ivf_context->bruteforce_threshold());
  // BF would return all twelve records; nprobe=1 must return only six.
  context->set_topk(2 * group_size);
  std::vector<float> queries(2 * dim, 0.0f);
  queries[0] = 0.125f;
  queries[dim - 1] = 0.375f;
  queries[dim] = 40.0f;
  queries[2 * dim - 1] = -120.0f;
  ASSERT_EQ(0, searcher.search_impl(queries.data(),
                                    IndexQueryMeta(IndexMeta::DT_FP32, dim), 2,
                                    context));
  for (uint32_t group = 0; group < 2; ++group) {
    EXPECT_EQ(1U, ivf_context->centroid_searcher_ctx()->result(group).size());
    const auto &result = context->result(group);
    ASSERT_EQ(group_size, result.size());
    for (const auto &doc : result) {
      EXPECT_EQ(group, doc.key() / group_size);
      EXPECT_NEAR(0.0f, doc.score(), 0.001f);
    }
  }
  ASSERT_EQ(0, searcher.unload());
}

}  // namespace
}  // namespace core
}  // namespace zvec
