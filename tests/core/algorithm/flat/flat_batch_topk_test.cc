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
#include <cstring>
#include <future>
#include <set>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "flat/flat_streamer.h"
#include "flat/flat_streamer_context.h"
#include "flat/flat_utility.h"
#include "quantizer/distance_quantizer.h"
#include "tests/test_util.h"

using namespace zvec;
using namespace zvec::core;

namespace {
constexpr size_t kDimension = 19;
constexpr size_t kCount = 99;
constexpr uint64_t kKeyBase = uint64_t{1} << 48;
uint64_t Key(size_t i) {
  return kKeyBase + i * 37 + 11;
}
size_t Id(uint64_t key) {
  return (key - kKeyBase - 11) / 37;
}

struct Cleanup {
  std::string path;
  ~Cleanup() {
    test_util::RemoveTestPath(path);
  }
};

template <typename T>
std::vector<T> Data() {
  std::vector<T> data(kCount * kDimension);
  for (size_t i = 0; i < kCount; ++i) {
    for (size_t d = 0; d < kDimension; ++d) {
      data[i * kDimension + d] = static_cast<T>(
          (static_cast<int>((i * 31 + d * 7 + i * d) % 127) - 63) * 0.03125f);
    }
  }
  return data;
}

template <typename T>
void Check(const IndexDocumentList &actual, const std::vector<T> &data,
           const T *query, size_t topk, float threshold,
           const std::set<size_t> &excluded, bool fetch_vector = false,
           bool static_index = false) {
  std::vector<float> all(kCount);
  std::vector<float> expected;
  for (size_t i = 0; i < kCount; ++i) {
    float score = 0;
    for (size_t d = 0; d < kDimension; ++d) {
      const float delta = static_cast<float>(data[i * kDimension + d]) -
                          static_cast<float>(query[d]);
      score += delta * delta;
    }
    all[i] = score;
    if (!excluded.count(i) && score <= threshold) expected.push_back(score);
  }
  std::sort(expected.begin(), expected.end());
  expected.resize(std::min(topk, expected.size()));
  ASSERT_EQ(expected.size(), actual.size());
  std::set<uint64_t> seen;
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_GE(actual[i].key(), kKeyBase);
    const size_t id = Id(actual[i].key());
    ASSERT_LT(id, kCount);
    EXPECT_EQ(Key(id), actual[i].key());
    EXPECT_FALSE(excluded.count(id));
    EXPECT_TRUE(seen.insert(actual[i].key()).second);
    EXPECT_NEAR(expected[i], actual[i].score(), 0.03f);
    EXPECT_NEAR(all[id], actual[i].score(), 0.03f);
    if (static_index) EXPECT_EQ(id, actual[i].index());
    if (fetch_vector) {
      ASSERT_NE(nullptr, actual[i].vector());
      const auto *values = static_cast<const T *>(actual[i].vector());
      for (size_t d = 0; d < kDimension; ++d) {
        EXPECT_FLOAT_EQ(static_cast<float>(data[id * kDimension + d]),
                        static_cast<float>(values[d]));
      }
    }
  }
}

template <IndexMeta::DataType Type, typename T>
void CheckStatic() {
  Cleanup cleanup{"flat_batch_topk_static"};
  test_util::RemoveTestPath(cleanup.path);
  ailego::MemoryLimitPool::get_instance().init(64UL * 1024 * 1024);
  auto data = Data<T>();
  IndexMeta meta(Type, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  meta.set_major_order(IndexMeta::MO_ROW);
  auto holder = std::make_shared<MultiPassIndexHolder<Type>>(kDimension);
  for (size_t i = 0; i < kCount; ++i) {
    ailego::NumericalVector<T> row(kDimension);
    std::copy_n(data.data() + i * kDimension, kDimension, row.data());
    ASSERT_TRUE(holder->emplace(Key(i), row));
  }
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, dumper->create(cleanup.path + "/index"));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
  for (const char *backend :
       {"MMapFileReadStorage", "FileReadStorage", "BufferReadStorage"}) {
    for (bool quantized : {false, true}) {
      SCOPED_TRACE(backend);
      SCOPED_TRACE(quantized);
      auto storage = IndexFactory::CreateStorage(backend);
      ASSERT_EQ(0, storage->init(ailego::Params()));
      ASSERT_EQ(0, storage->open(cleanup.path + "/index", false));
      auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
      ailego::Params params;
      // Several short-lived storage reads, with a final three-row tail.
      params.set(PARAM_FLAT_READ_BLOCK_SIZE,
                 uint32_t{32 * sizeof(T) * kDimension});
      ASSERT_EQ(0, quantized
                       ? searcher->init(params, CreateDistanceQuantizer(meta))
                       : searcher->init(params));
      ASSERT_EQ(0, searcher->load(storage, nullptr));
      auto context = searcher->create_context();
      IndexQueryMeta qmeta(Type, kDimension);
      for (uint32_t topk : {1U, 10U, 65U, 120U, 10U}) {
        for (int mode : {0, 1, 2}) {
          SCOPED_TRACE(topk);
          SCOPED_TRACE(mode);
          std::set<size_t> excluded;
          for (size_t i = 0; i < kCount; ++i) {
            if (mode == 2 || (mode == 1 && i % 3 == 0)) excluded.insert(i);
          }
          context->set_topk(topk);
          context->set_threshold(40.0f);
          size_t filtered_calls = 0;
          if (mode == 0)
            context->reset_filter();
          else
            context->set_filter([excluded, &filtered_calls](uint64_t key) {
              ++filtered_calls;
              return excluded.count(Id(key)) != 0;
            });
          // Grow, shrink and reuse the per-query selectors, including both
          // the single-query specialization and a partial final data batch.
          for (uint32_t query_count : {3U, 7U, 2U, 1U, 7U}) {
            SCOPED_TRACE(query_count);
            filtered_calls = 0;
            ASSERT_EQ(0, searcher->search_impl(data.data(), qmeta, query_count,
                                               context));
            EXPECT_EQ(mode == 0 ? 0U : kCount, filtered_calls);
            for (size_t q = 0; q < query_count; ++q) {
              Check(context->result(q), data, data.data() + q * kDimension,
                    topk, 40.0f, excluded, false, true);
            }
          }
          context->reset_filter();
        }
      }
      context->reset_filter();
      context->reset_threshold();
      context->set_topk(7);
      ASSERT_EQ(0, searcher->search_impl(data.data() + 70 * kDimension, qmeta,
                                         context));
      Check(context->result(), data, data.data() + 70 * kDimension, 7,
            std::numeric_limits<float>::max(), {}, false, true);
      // Different query contexts may search the same immutable index
      // concurrently.
      auto run = [&](size_t id) {
        auto independent = searcher->create_context();
        independent->set_topk(10);
        EXPECT_EQ(0, searcher->search_impl(data.data() + id * kDimension, qmeta,
                                           independent));
        Check(independent->result(), data, data.data() + id * kDimension, 10,
              std::numeric_limits<float>::max(), {}, false, true);
      };
      auto first = std::async(std::launch::async, run, 17);
      auto second = std::async(std::launch::async, run, 83);
      first.get();
      second.get();
      context.reset();
      ASSERT_EQ(0, searcher->unload());
      ASSERT_EQ(0, storage->close());
    }
  }
}

template <IndexMeta::DataType Type, typename T>
void CheckStreamer() {
  Cleanup cleanup{"flat_batch_topk_streamer"};
  test_util::RemoveTestPath(cleanup.path);
  ailego::MemoryLimitPool::get_instance().init(64UL * 1024 * 1024);
  const auto data = Data<T>();
  IndexMeta meta(Type, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  meta.set_major_order(IndexMeta::MO_ROW);
  IndexQueryMeta qmeta(Type, kDimension);
  for (bool contiguous : {false, true}) {
    for (bool quantized : {false, true}) {
      SCOPED_TRACE(contiguous);
      SCOPED_TRACE(quantized);
      auto storage = IndexFactory::CreateStorage("MMapFileStorage");
      ASSERT_EQ(0, storage->init(ailego::Params()));
      const std::string path = cleanup.path + "/" + std::to_string(contiguous) +
                               std::to_string(quantized);
      ASSERT_EQ(0, storage->open(path, true));
      auto streamer = IndexFactory::CreateStreamer("FlatStreamer");
      ailego::Params params;
      params.set(PARAM_FLAT_USE_ID_MAP, true);
      params.set(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, contiguous);
      ASSERT_EQ(0, quantized ? streamer->init(meta, params,
                                              CreateDistanceQuantizer(meta))
                             : streamer->init(meta, params));
      ASSERT_EQ(0, streamer->open(storage));
      auto context = streamer->create_context();
      for (size_t i = 0; i < kCount; ++i) {
        ASSERT_EQ(0, streamer->add_impl(Key(i), data.data() + i * kDimension,
                                        qmeta, context));
      }
      // Seed on-disk deletion bits and an invalid slot through the storage
      // interface: Flat does not expose a public remove operation. Reopening
      // exercises both normal block scans and contiguous-cache reconstruction.
      auto *flat = dynamic_cast<FlatStreamer<32> *>(streamer.get());
      ASSERT_NE(nullptr, flat);
      const size_t block_size = flat->entity().linear_block_size();
      auto segment = storage->get(FLAT_SEGMENT_FEATURES_SEG_ID + "1");
      ASSERT_NE(nullptr, segment);
      for (size_t id : {7U, 98U}) {
        DeletionMap deleted;
        deleted.set(id % 32);
        const size_t offset = (id / 32 + 1) * block_size - sizeof(BlockHeader) -
                              sizeof(DeletionMap);
        ASSERT_EQ(sizeof(deleted),
                  segment->write(offset, &deleted, sizeof(deleted)));
      }
      const uint64_t invalid = kInvalidKey;
      const size_t key_offset =
          block_size + 32 * meta.element_size() + 2 * sizeof(uint64_t);
      ASSERT_EQ(sizeof(invalid),
                segment->write(key_offset, &invalid, sizeof(invalid)));
      segment.reset();
      ASSERT_EQ(0, streamer->flush(0));
      ASSERT_EQ(0, streamer->close());
      ASSERT_EQ(0, streamer->open(storage));
      context = streamer->create_context();
      auto *flat_context =
          dynamic_cast<FlatStreamerContext<32> *>(context.get());
      ASSERT_NE(nullptr, flat_context);
      for (uint32_t topk : {1U, 10U, 65U, 120U, 10U}) {
        for (int mode : {0, 1, 2}) {
          std::set<size_t> excluded{7, 34, 98};
          for (size_t i = 0; i < kCount; ++i) {
            if (mode == 2 || (mode == 1 && i % 3 == 0)) excluded.insert(i);
          }
          context->set_topk(topk);
          context->set_fetch_vector(true);
          context->set_threshold(40.0f);
          if (mode == 0)
            context->reset_filter();
          else
            context->set_filter([excluded](uint64_t key) {
              return excluded.count(Id(key)) != 0;
            });
          ASSERT_EQ(0,
                    streamer->search_bf_impl(data.data(), qmeta, 3, context));
          for (size_t q = 0; q < 3; ++q) {
            Check(context->result(q), data, data.data() + q * kDimension, topk,
                  40.0f, excluded, true);
            // Contiguous reconstruction omits the two deleted rows entirely.
            // The retained legacy K>=N unfiltered path computes clean blocks
            // before discarding invalid keys; preserve that existing counter.
            const bool legacy_invalid =
                !contiguous && topk >= kCount && mode == 0;
            EXPECT_EQ(excluded.size() - (contiguous ? 2 : 0) - legacy_invalid,
                      flat_context->mutable_stats(q)->filtered_count());
            EXPECT_EQ(kCount - excluded.size() + legacy_invalid,
                      flat_context->mutable_stats(q)->dist_calced_count());
          }
        }
      }
      context->reset_filter();
      context->reset_threshold();
      context->set_topk(7);
      ASSERT_EQ(0, streamer->search_bf_impl(data.data() + 70 * kDimension,
                                            qmeta, context));
      Check(context->result(), data, data.data() + 70 * kDimension, 7,
            std::numeric_limits<float>::max(), {7, 34, 98}, true);
      context->set_threshold(0.0f);
      ASSERT_EQ(0, streamer->search_bf_impl(data.data() + 70 * kDimension,
                                            qmeta, context));
      Check(context->result(), data, data.data() + 70 * kDimension, 7, 0.0f,
            {7, 34, 98}, true);
      context.reset();
      ASSERT_EQ(0, streamer->close());
      ASSERT_EQ(0, storage->close());
    }
  }
}
}  // namespace

TEST(FlatBatchTopK, Fp32StaticStorageFilteringAndReuse) {
  CheckStatic<IndexMeta::DT_FP32, float>();
}
TEST(FlatBatchTopK, Fp16StaticStorageFilteringAndReuse) {
  CheckStatic<IndexMeta::DT_FP16, ailego::Float16>();
}
TEST(FlatBatchTopK, StaticMultiQueryPreprocessingAndExtraValues) {
  constexpr size_t kEncodedDimension = kDimension + sizeof(uint32_t);
  Cleanup cleanup{"flat_batch_topk_static_preprocess"};
  test_util::RemoveTestPath(cleanup.path);
  ailego::MemoryLimitPool::get_instance().init(64UL * 1024 * 1024);
  ailego::Params metric_params;
  metric_params.set("proxima.uniform_uint8.metric.origin_metric_name",
                    std::string("SquaredEuclidean"));
  IndexMeta meta(IndexMeta::DT_INT8, kEncodedDimension);
  meta.set_metric("UniformUint8", 0, metric_params);
  meta.set_major_order(IndexMeta::MO_ROW);
  auto metric = IndexFactory::CreateMetric("UniformUint8");
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(meta, metric_params));
  ASSERT_TRUE(metric->batch_distance());
  ASSERT_TRUE(metric->get_query_preprocess_func());
  ASSERT_EQ(sizeof(uint32_t), metric->extra_values_size_per_vector());

  std::vector<int8_t> data(kCount * kDimension);
  std::string queries;
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_INT8>>(
      kEncodedDimension);
  for (size_t i = 0; i < kCount; ++i) {
    ailego::NumericalVector<int8_t> row(kEncodedDimension);
    uint32_t sum_squared = 0;
    for (size_t d = 0; d < kDimension; ++d) {
      const uint8_t code = static_cast<uint8_t>(i * 73 + d * 29 + d * i * 3);
      row[d] = static_cast<int8_t>(static_cast<int>(code) - 128);
      data[i * kDimension + d] = row[d];
      sum_squared += static_cast<uint32_t>(code) * code;
    }
    std::memcpy(row.data() + kDimension, &sum_squared, sizeof(sum_squared));
    ASSERT_TRUE(holder->emplace(Key(i), row));
    if (i < 7) {
      queries.append(reinterpret_cast<const char *>(row.data()),
                     kEncodedDimension);
    }
  }
  const auto original_queries = queries;
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, dumper->create(cleanup.path + "/index"));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(cleanup.path + "/index", false));
  auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
  ailego::Params params;
  params.set(PARAM_FLAT_READ_BLOCK_SIZE, uint32_t{32 * kEncodedDimension});
  ASSERT_EQ(0, searcher->init(params));
  ASSERT_EQ(0, searcher->load(storage, nullptr));
  auto context = searcher->create_context();
  context->set_topk(10);
  IndexQueryMeta qmeta(IndexMeta::DT_INT8, kEncodedDimension);
  for (bool filtered : {false, true}) {
    SCOPED_TRACE(filtered);
    std::set<size_t> excluded;
    if (filtered) {
      for (size_t i = 0; i < kCount; i += 3) excluded.insert(i);
      context->set_filter([](uint64_t key) { return Id(key) % 3 == 0; });
    } else {
      context->reset_filter();
    }
    // Unaligned norm tails and distinct queries exercise both byte strides.
    for (uint32_t count : {3U, 7U, 2U, 1U, 7U}) {
      SCOPED_TRACE(count);
      ASSERT_EQ(0,
                searcher->search_impl(queries.data(), qmeta, count, context));
      EXPECT_EQ(original_queries, queries);
      for (size_t q = 0; q < count; ++q) {
        Check(context->result(q), data,
              reinterpret_cast<const int8_t *>(queries.data()) +
                  q * kEncodedDimension,
              10, std::numeric_limits<float>::max(), excluded, false, true);
      }
    }
  }
  context.reset();
  ASSERT_EQ(0, searcher->unload());
  ASSERT_EQ(0, storage->close());
}
TEST(FlatBatchTopK, Fp32StreamerFilteringStatsAndVectors) {
  CheckStreamer<IndexMeta::DT_FP32, float>();
}
TEST(FlatBatchTopK, Fp16StreamerFilteringStatsAndVectors) {
  CheckStreamer<IndexMeta::DT_FP16, ailego::Float16>();
}
