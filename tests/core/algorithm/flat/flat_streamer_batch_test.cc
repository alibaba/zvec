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
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "flat/flat_index_format.h"
#include "flat/flat_streamer.h"
#include "flat/flat_utility.h"
#include "quantizer/fp32_quantizer/fp32_quantizer.h"
#include "tests/test_util.h"

using namespace zvec;
using namespace zvec::core;

namespace {

bool ReadSegment(const IndexStorage::Segment::Pointer &segment, size_t offset,
                 void *output, size_t size) {
  IndexStorage::MemoryBlock block;
  if (segment->read(offset, block, size) != size || !block.data()) {
    return false;
  }
  std::memcpy(output, block.data(), size);
  return true;
}

// Keep the actual FP32 dispatch and encoding, while observing the interface
// chosen by the streamer. Scalar result checks below are independent of it.
class RecordingFp32Quantizer : public turbo::Fp32Quantizer {
 public:
  struct Call {
    size_t stride;
    int count;
    bool contiguous;
  };

  void calc_distance_dp_query_batch(const void *const *candidates, int count,
                                    const void *query,
                                    float *distances) const override {
    const size_t stride = quantized_datapoint_vector_length();
    bool contiguous = true;
    for (int i = 1; i < count; ++i) {
      contiguous &= reinterpret_cast<uintptr_t>(candidates[0]) + i * stride ==
                    reinterpret_cast<uintptr_t>(candidates[i]);
    }
    calls.push_back({stride, count, contiguous});
    if (before_first_batch) {
      auto callback = std::move(before_first_batch);
      before_first_batch = nullptr;
      callback();
    }
    Fp32Quantizer::calc_distance_dp_query_batch(candidates, count, query,
                                                distances);
  }

  mutable std::vector<Call> calls;
  mutable std::function<void()> before_first_batch;
};

class FlatStreamerBatchTest
    : public testing::TestWithParam<std::tuple<const char *, bool>> {
 protected:
  static constexpr size_t kDimension = 19;
  static constexpr size_t kCount = 1073;
  static constexpr size_t kTopk = 23;
  const std::string path_ = "flat_streamer_batch_test/index";
  IndexStreamer::Pointer streamer_;
  IndexStorage::Pointer storage_;

  static uint64_t key(size_t position) {
    return (uint64_t{1} << 44) + 1000003 + position * 17;
  }

  void SetUp() override {
    test_util::RemoveTestPath("flat_streamer_batch_test");
    ailego::MemoryLimitPool::get_instance().init(256UL * 1024 * 1024);
  }

  void TearDown() override {
    if (streamer_) streamer_->close();
    streamer_.reset();
    if (storage_) storage_->close();
    storage_.reset();
    test_util::RemoveTestPath("flat_streamer_batch_test");
  }

  // A deletion map can be loaded from an existing Flat index even though the
  // streamer exposes no public erase method. Seed that real persisted state,
  // including an invalid key, rather than mocking the search implementation.
  void mark_deleted_and_invalid(size_t element_size) {
    auto metadata = storage_->get(FLAT_LINEAR_META_SEG_ID);
    auto head = storage_->get(FLAT_LINEAR_LIST_HEAD_SEG_ID);
    ASSERT_NE(nullptr, metadata);
    ASSERT_NE(nullptr, head);
    StreamerLinearMeta meta;
    BlockLocation location;
    ASSERT_TRUE(ReadSegment(metadata, 0, &meta, sizeof(meta)));
    ASSERT_TRUE(ReadSegment(head, 0, &location, sizeof(location)));
    size_t changed = 0;
    while (location.segment_id != 0) {
      auto segment = storage_->get(FLAT_SEGMENT_FEATURES_SEG_ID +
                                   std::to_string(location.segment_id));
      ASSERT_NE(nullptr, segment);
      const size_t begin = location.block_index * meta.header.block_size;
      const size_t header_offset =
          begin + meta.header.block_size - sizeof(BlockHeader);
      const size_t deletion_offset = header_offset - sizeof(DeletionMap);
      const size_t keys_offset =
          begin + meta.header.block_vector_count * element_size;
      BlockHeader header;
      DeletionMap deletion;
      ASSERT_TRUE(ReadSegment(segment, header_offset, &header, sizeof(header)));
      ASSERT_TRUE(
          ReadSegment(segment, deletion_offset, &deletion, sizeof(deletion)));
      std::vector<uint64_t> keys(header.vector_count);
      const size_t key_bytes = keys.size() * sizeof(uint64_t);
      ASSERT_TRUE(ReadSegment(segment, keys_offset, keys.data(), key_bytes));
      for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == key(2) || keys[i] == key(1024)) {
          deletion.set(i);
          ++changed;
        } else if (keys[i] == key(256)) {
          keys[i] = kInvalidKey;
          ++changed;
        }
      }
      ASSERT_EQ(sizeof(deletion),
                segment->write(deletion_offset, &deletion, sizeof(deletion)));
      ASSERT_EQ(key_bytes, segment->write(keys_offset, keys.data(), key_bytes));
      location = header.next;
    }
    ASSERT_EQ(3U, changed);
    ASSERT_EQ(0, storage_->flush());
  }

  void check_result(const IndexDocumentList &actual,
                    const std::vector<std::string> &codes, const void *query,
                    const char *metric, int filter_mode, bool deleted) {
    const auto *q = static_cast<const float *>(query);
    std::vector<std::pair<float, uint64_t>> expected;
    for (size_t i = 0; i < codes.size(); ++i) {
      if ((deleted && (i == 2 || i == 256 || i == 1024)) ||
          (filter_mode == 1 && i % 3 == 0) || (filter_mode == 2 && i == 12)) {
        continue;
      }
      const auto *row = reinterpret_cast<const float *>(codes[i].data());
      double score = std::string(metric) == "Cosine" ? 1.0 : 0.0;
      for (size_t d = 0; d < kDimension; ++d) {
        if (std::string(metric) == "SquaredEuclidean") {
          const double delta = row[d] - q[d];
          score += delta * delta;
        } else {
          score -= double(row[d]) * q[d];
        }
      }
      expected.emplace_back(static_cast<float>(score), key(i));
    }
    std::partial_sort(expected.begin(), expected.begin() + kTopk,
                      expected.end());
    ASSERT_EQ(kTopk, actual.size());
    for (size_t i = 0; i < kTopk; ++i) {
      EXPECT_EQ(expected[i].second, actual[i].key());
      EXPECT_NEAR(expected[i].first, actual[i].score(),
                  1e-4f * std::max(1.0f, std::abs(expected[i].first)));
    }
  }

  void check_calls(const RecordingFp32Quantizer &quantizer, size_t query_count,
                   bool filtered, bool deleted, size_t topk = kTopk) {
    const auto *flat = dynamic_cast<const FlatStreamer<32> *>(streamer_.get());
    ASSERT_NE(nullptr, flat);
    const auto *contiguous =
        dynamic_cast<const FlatContiguousStreamerEntity *>(&flat->entity());
    const bool snapshot = contiguous && contiguous->is_contiguous();
    if (snapshot || filtered || deleted || topk == 0 ||
        topk == std::numeric_limits<size_t>::max()) {
      // These paths also use pointer batches, with their own grouping rules.
      // Result and statistics checks below cover their scan semantics.
      return;
    }
    ASSERT_FALSE(quantizer.calls.empty());
    size_t count = 0;
    size_t full_blocks = 0;
    bool has_tail = false;
    for (const auto &call : quantizer.calls) {
      const size_t encoded_size = quantizer.quantized_datapoint_vector_length();
      EXPECT_EQ(encoded_size, call.stride);
      EXPECT_TRUE(call.contiguous);
      EXPECT_GT(call.count, 0);
      EXPECT_LE(call.count, 32);
      count += call.count;
      full_blocks += call.count == 32;
      has_tail |= call.count % 32 != 0;
    }
    // The 17-row head must not split later full physical blocks.
    EXPECT_EQ(kCount * query_count, count);
    EXPECT_TRUE(has_tail);
    EXPECT_EQ(((kCount + 31) / 32) * query_count, quantizer.calls.size());
    EXPECT_EQ((kCount / 32) * query_count, full_blocks);
  }

  void open_generic_entity(RecordingFp32Quantizer *quantizer,
                           const std::shared_ptr<RecordingFp32Quantizer> &owner,
                           bool use_key_map, IndexStreamer::Stats *stats,
                           FlatStreamerEntity::Pointer *entity) {
    if (!storage_) {
      storage_ = IndexFactory::CreateStorage(std::get<0>(GetParam()));
      ASSERT_NE(nullptr, storage_);
      ASSERT_EQ(0, storage_->init(ailego::Params()));
      ASSERT_EQ(0, storage_->open(path_, true));
    }
    *entity = std::make_shared<FlatStreamerEntity>(*stats);
    (*entity)->set_block_vector_count(32);
    // One 19D FP32 physical block fits in this segment. Small segments make
    // clone/cache extension and physical candidate IDs observable cheaply.
    (*entity)->set_segment_size(4096);
    (*entity)->set_linear_list_count(1);
    (*entity)->set_use_key_info_map(use_key_map);
    (*entity)->set_quantizer(owner);
    *(*entity)->mutable_meta() = quantizer->meta();
    ASSERT_EQ(0, (*entity)->open(storage_, quantizer->meta()));
  }

  static std::vector<float> ip_row(size_t position) {
    std::vector<float> row(kDimension, 0.0f);
    row[0] = static_cast<float>(position + 1);
    return row;
  }

  void init_ip_quantizer(std::shared_ptr<RecordingFp32Quantizer> *quantizer) {
    IndexMeta meta(IndexMeta::DT_FP32, kDimension);
    meta.set_metric("InnerProduct", 0, ailego::Params());
    meta.set_major_order(IndexMeta::MO_ROW);
    *quantizer = std::make_shared<RecordingFp32Quantizer>();
    ASSERT_EQ(0, (*quantizer)->init(meta, ailego::Params()));
  }

  void check_ip_entity(const FlatStreamerEntity &entity,
                       RecordingFp32Quantizer *quantizer,
                       std::vector<std::pair<float, uint64_t>> expected,
                       size_t physical_count, bool eligible,
                       size_t deleted_rows = 0) {
    const std::vector<float> query = ip_row(0);
    IndexDocumentHeap heap(kTopk);
    IndexContext::Stats stats;
    FlatSearchScratch scratch;
    uint32_t scanned = 0;
    quantizer->calls.clear();
    ASSERT_EQ(0, entity.search(query.data(), IndexFilter(), &scanned, &heap,
                               &stats, &scratch));
    EXPECT_EQ(physical_count, scanned);
    EXPECT_EQ(physical_count - deleted_rows, stats.dist_calced_count());
    EXPECT_EQ(deleted_rows, stats.filtered_count());
    if (eligible) {
      EXPECT_FALSE(quantizer->calls.empty());
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(std::min(kTopk, expected.size()));
    heap.sort();
    ASSERT_EQ(expected.size(), heap.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i].second, heap[i].key());
      EXPECT_FLOAT_EQ(expected[i].first, heap[i].score());
    }
  }

  void check_prefilled_heap(const std::vector<std::string> &codes,
                            const void *query, const char *metric,
                            RecordingFp32Quantizer *quantizer, bool deleted) {
    auto *flat = dynamic_cast<FlatStreamer<32> *>(streamer_.get());
    ASSERT_NE(nullptr, flat);
    const auto *contiguous =
        dynamic_cast<const FlatContiguousStreamerEntity *>(&flat->entity());
    const bool snapshot = contiguous && contiguous->is_contiguous();
    const auto *q = static_cast<const float *>(query);
    constexpr uint64_t other_key = (uint64_t{1} << 61) + 123;
    constexpr size_t unlimited = std::numeric_limits<size_t>::max();
    for (size_t topk : {size_t{1}, size_t{511}, size_t{512}, size_t{513},
                        size_t{1078}, unlimited}) {
      for (float threshold : {11.0f, std::numeric_limits<float>::max()}) {
        SCOPED_TRACE(topk);
        SCOPED_TRACE(threshold);
        for (bool filtered : {false, true}) {
          SCOPED_TRACE(filtered);
          // The default constructor supplies an unlimited heap without
          // attempting to reserve SIZE_MAX entries.
          IndexDocumentHeap heap;
          if (topk != unlimited) heap.limit(topk);
          heap.set_threshold(threshold);
          // A candidate from an earlier entity must remain part of the final
          // top-k when the fixed-size local selection buffer is reset.
          heap.emplace(other_key, -1.0f);
          std::vector<std::pair<float, uint64_t>> expected{{-1.0f, other_key}};
          std::unordered_map<uint64_t, float> expected_scores{
              {other_key, -1.0f}};
          size_t valid_count = 0;
          for (size_t i = 0; i < codes.size(); ++i) {
            if ((deleted && (i == 2 || i == 256 || i == 1024)) ||
                (filtered && i % 3 == 0)) {
              continue;
            }
            ++valid_count;
            const auto *row = reinterpret_cast<const float *>(codes[i].data());
            double score = std::string(metric) == "Cosine" ? 1.0 : 0.0;
            for (size_t d = 0; d < kDimension; ++d) {
              if (std::string(metric) == "SquaredEuclidean") {
                const double delta = row[d] - q[d];
                score += delta * delta;
              } else {
                score -= double(row[d]) * q[d];
              }
            }
            if (score <= threshold) {
              expected.emplace_back(static_cast<float>(score), key(i));
              expected_scores.emplace(key(i), static_cast<float>(score));
            }
          }
          std::sort(expected.begin(), expected.end());
          expected.resize(std::min(topk, expected.size()));
          FlatSearchScratch scratch;
          IndexContext::Stats stats;
          uint32_t scan_count = 0;
          IndexFilter filter;
          if (filtered) {
            filter.set([](uint64_t value) {
              return ((value - key(0)) / 17) % 3 == 0;
            });
          }
          quantizer->calls.clear();
          ASSERT_EQ(0, flat->entity().search(query, filter, &scan_count, &heap,
                                             &stats, &scratch));
          check_calls(*quantizer, 1, filtered, deleted, topk);
          const size_t expected_scan_count =
              kCount - (snapshot && deleted ? 2 : 0);
          const size_t expected_calced_count =
              snapshot || filtered ? valid_count : kCount - (deleted ? 2 : 0);
          const size_t expected_filtered_count =
              snapshot || filtered ? expected_scan_count - valid_count
                                   : (deleted ? 2 : 0);
          EXPECT_EQ(expected_scan_count, scan_count);
          EXPECT_EQ(expected_calced_count, stats.dist_calced_count());
          EXPECT_EQ(expected_filtered_count, stats.filtered_count());
          heap.sort();
          ASSERT_EQ(expected.size(), heap.size());
          std::vector<uint64_t> actual_keys, expected_keys;
          for (size_t i = 0; i < heap.size(); ++i) {
            actual_keys.push_back(heap[i].key());
            expected_keys.push_back(expected[i].second);
            const auto expected_score = expected_scores.find(heap[i].key());
            ASSERT_NE(expected_scores.end(), expected_score);
            EXPECT_NEAR(
                expected_score->second, heap[i].score(),
                1e-4f * std::max(1.0f, std::abs(expected_score->second)));
            if (i != 0) EXPECT_LE(heap[i - 1].score(), heap[i].score());
          }
          std::sort(actual_keys.begin(), actual_keys.end());
          std::sort(expected_keys.begin(), expected_keys.end());
          EXPECT_EQ(expected_keys, actual_keys);
        }
      }
    }
  }
};

TEST_P(FlatStreamerBatchTest, FullScanUsesBoundedBatchDistances) {
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    SCOPED_TRACE(metric);
    IndexMeta raw_meta(IndexMeta::DT_FP32, kDimension);
    raw_meta.set_metric(metric, 0, ailego::Params());
    auto quantizer = std::make_shared<RecordingFp32Quantizer>();
    ASSERT_EQ(0, quantizer->init(raw_meta, ailego::Params()));
    IndexMeta meta = quantizer->meta();
    meta.set_quantizer("Fp32Quantizer", 0, ailego::Params());
    ailego::Params params;
    params.set(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, std::get<1>(GetParam()));
    storage_ = IndexFactory::CreateStorage(std::get<0>(GetParam()));
    ASSERT_NE(nullptr, storage_);
    ASSERT_EQ(0, storage_->init(ailego::Params()));
    ASSERT_EQ(0, storage_->open(path_, true));
    streamer_ = IndexFactory::CreateStreamer("FlatStreamer");
    ASSERT_NE(nullptr, streamer_);
    ASSERT_EQ(0, streamer_->init(meta, params, quantizer));
    ASSERT_EQ(0, streamer_->open(storage_));
    auto context = streamer_->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(kTopk);
    std::mt19937 random(20261022);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> raw(kDimension);
    std::vector<std::string> codes(kCount,
                                   std::string(meta.element_size(), '\0'));
    IndexQueryMeta qmeta;
    std::vector<std::string> queries(3);
    for (auto &query : queries) {
      for (float &value : raw) value = distribution(random);
      ASSERT_EQ(
          0, quantizer->quantize(raw.data(),
                                 IndexQueryMeta(IndexMeta::DT_FP32, kDimension),
                                 &query, &qmeta));
    }
    for (size_t i = 0; i < kCount; ++i) {
      for (float &value : raw) value = distribution(random);
      quantizer->quantize_data(raw.data(), codes[i].data());
      ASSERT_EQ(0,
                streamer_->add_impl(key(i), codes[i].data(), qmeta, context));
    }
    // Cover the writer before flush, and the loaded representation after
    // deletion and invalid-key markers have been persisted.
    for (bool deleted : {false, true}) {
      SCOPED_TRACE(deleted);
      if (deleted) {
        ASSERT_EQ(0, streamer_->flush(0));
        ASSERT_EQ(0, streamer_->close());
        mark_deleted_and_invalid(meta.element_size());
        ASSERT_EQ(0, streamer_->open(storage_));
        context = streamer_->create_context();
        ASSERT_NE(nullptr, context);
        context->set_topk(kTopk);
      }
      for (int filter_mode : {0, 1, 2, 0}) {
        SCOPED_TRACE(filter_mode);
        const bool filtered = filter_mode != 0;
        if (filter_mode == 1) {
          context->set_filter(
              [](uint64_t value) { return ((value - key(0)) / 17) % 3 == 0; });
        } else if (filter_mode == 2) {
          context->set_filter([](uint64_t value) { return value == key(12); });
        } else {
          context->reset_filter();
        }
        std::string batch = queries[0] + queries[1] + queries[2];
        quantizer->calls.clear();
        ASSERT_EQ(0, streamer_->search_impl(batch.data(), qmeta, 3, context));
        check_calls(*quantizer, 3, filtered, deleted);
        for (size_t q = 0; q < queries.size(); ++q) {
          check_result(context->result(q), codes, queries[q].data(), metric,
                       filter_mode, deleted);
        }
        quantizer->calls.clear();
        ASSERT_EQ(0,
                  streamer_->search_bf_impl(queries[1].data(), qmeta, context));
        check_calls(*quantizer, 1, filtered, deleted);
        check_result(context->result(), codes, queries[1].data(), metric,
                     filter_mode, deleted);
      }
      check_prefilled_heap(codes, queries[0].data(), metric, quantizer.get(),
                           deleted);
    }
    ASSERT_EQ(0, streamer_->close());
    streamer_.reset();
    ASSERT_EQ(0, storage_->close());
    storage_.reset();
    test_util::RemoveTestPath("flat_streamer_batch_test");
  }
}

TEST_P(FlatStreamerBatchTest, ReloadChecksValidityWithEitherKeyMapMode) {
  for (bool use_key_map : {false, true}) {
    SCOPED_TRACE(use_key_map);
    std::shared_ptr<RecordingFp32Quantizer> quantizer;
    init_ip_quantizer(&quantizer);
    IndexStreamer::Stats stats;
    FlatStreamerEntity::Pointer entity;
    open_generic_entity(quantizer.get(), quantizer, use_key_map, &stats,
                        &entity);
    std::vector<std::pair<float, uint64_t>> expected;
    for (size_t i = 0; i < kCount; ++i) {
      const auto row = ip_row(i);
      ASSERT_EQ(0, entity->add(key(i), row.data(), row.size() * sizeof(float)));
      expected.emplace_back(-row[0], key(i));
    }
    ASSERT_EQ(0, entity->flush(0));
    ASSERT_EQ(0, entity->close());
    open_generic_entity(quantizer.get(), quantizer, use_key_map, &stats,
                        &entity);
    check_ip_entity(*entity, quantizer.get(), expected, kCount, true);
    ASSERT_EQ(0, entity->close());
    mark_deleted_and_invalid(quantizer->meta().element_size());
    open_generic_entity(quantizer.get(), quantizer, use_key_map, &stats,
                        &entity);
    expected.clear();
    for (size_t i = 0; i < kCount; ++i) {
      if (i != 2 && i != 256 && i != 1024) {
        expected.emplace_back(-ip_row(i)[0], key(i));
      }
    }
    check_ip_entity(*entity, quantizer.get(), expected, kCount, false, 2);
    ASSERT_EQ(0, entity->close());
    entity.reset();
    ASSERT_EQ(0, storage_->close());
    storage_.reset();
    test_util::RemoveTestPath("flat_streamer_batch_test");
  }
}

TEST_P(FlatStreamerBatchTest, FillingSparseIdsKeepsConservativeFallback) {
  for (bool use_key_map : {false, true}) {
    SCOPED_TRACE(use_key_map);
    std::shared_ptr<RecordingFp32Quantizer> quantizer;
    init_ip_quantizer(&quantizer);
    IndexStreamer::Stats stats;
    FlatStreamerEntity::Pointer entity;
    open_generic_entity(quantizer.get(), quantizer, use_key_map, &stats,
                        &entity);
    std::vector<std::pair<float, uint64_t>> expected;
    for (uint32_t i = 0; i < 32; ++i) {
      const auto row = ip_row(i);
      ASSERT_EQ(0, entity->add_vector_with_id(i, row.data(),
                                              row.size() * sizeof(float)));
      expected.emplace_back(-row[0], i);
    }
    check_ip_entity(*entity, quantizer.get(), expected, 32, true);
    auto row = ip_row(65);
    ASSERT_EQ(0, entity->add_vector_with_id(65, row.data(),
                                            row.size() * sizeof(float)));
    expected.emplace_back(-row[0], 65);
    check_ip_entity(*entity, quantizer.get(), expected, 66, false);
    for (uint32_t i = 32; i < 65; ++i) {
      row = ip_row(i);
      ASSERT_EQ(0, entity->add_vector_with_id(i, row.data(),
                                              row.size() * sizeof(float)));
      expected.emplace_back(-row[0], i);
    }
    // Filling all holes does not restore the live generation's flag.
    check_ip_entity(*entity, quantizer.get(), expected, 66, false);
    ASSERT_EQ(0, entity->flush(0));
    ASSERT_EQ(0, entity->close());
    open_generic_entity(quantizer.get(), quantizer, use_key_map, &stats,
                        &entity);
    check_ip_entity(*entity, quantizer.get(), expected, 66, true);
    ASSERT_EQ(0, entity->close());
    entity.reset();
    ASSERT_EQ(0, storage_->close());
    storage_.reset();
    test_util::RemoveTestPath("flat_streamer_batch_test");
  }
}

TEST_P(FlatStreamerBatchTest, CloneSharesInvalidationAndExtendsSegmentCache) {
  std::shared_ptr<RecordingFp32Quantizer> quantizer;
  init_ip_quantizer(&quantizer);
  IndexStreamer::Stats stats;
  FlatStreamerEntity::Pointer entity;
  open_generic_entity(quantizer.get(), quantizer, true, &stats, &entity);
  std::vector<std::pair<float, uint64_t>> expected;
  for (size_t i = 0; i < 32; ++i) {
    const auto row = ip_row(i);
    ASSERT_EQ(0, entity->add(key(i), row.data(), row.size() * sizeof(float)));
    expected.emplace_back(-row[0], key(i));
  }
  auto clone = entity->clone();
  ASSERT_NE(nullptr, clone);
  check_ip_entity(*clone, quantizer.get(), expected, 32, true);
  // Each extra 32-row block requires another 4 KiB segment. The clone was
  // created before those segments existed and must grow its own lazy cache.
  for (size_t i = 32; i < 96; ++i) {
    const auto row = ip_row(i);
    ASSERT_EQ(0, entity->add(key(i), row.data(), row.size() * sizeof(float)));
    expected.emplace_back(-row[0], key(i));
  }
  check_ip_entity(*clone, quantizer.get(), expected, 96, true);
  const auto invalid = ip_row(1000);
  ASSERT_EQ(0, entity->add(kInvalidKey, invalid.data(),
                           invalid.size() * sizeof(float)));
  // The invalid key was published by the original after clone creation.
  check_ip_entity(*clone, quantizer.get(), expected, 97, false);
  check_ip_entity(*entity, quantizer.get(), expected, 97, false);
  ASSERT_EQ(0, clone->close());
  ASSERT_EQ(0, entity->close());
}

TEST_P(FlatStreamerBatchTest, InvalidAppendDuringDistanceRestartsOnce) {
  if (std::get<0>(GetParam()) == std::string("BufferStorage")) {
    GTEST_SKIP() << "A buffer-pool pin can block same-thread append";
  }
  std::shared_ptr<RecordingFp32Quantizer> quantizer;
  init_ip_quantizer(&quantizer);
  IndexStreamer::Stats stats;
  FlatStreamerEntity::Pointer entity;
  open_generic_entity(quantizer.get(), quantizer, true, &stats, &entity);
  std::vector<std::pair<float, uint64_t>> expected;
  for (size_t i = 0; i < 128; ++i) {
    const auto row = ip_row(i);
    ASSERT_EQ(0, entity->add(key(i), row.data(), row.size() * sizeof(float)));
    expected.emplace_back(-row[0], key(i));
  }
  bool appended = false;
  quantizer->before_first_batch = [&]() {
    const auto invalid = ip_row(1000);
    ASSERT_EQ(0, entity->add(kInvalidKey, invalid.data(),
                             invalid.size() * sizeof(float)));
    const auto best = ip_row(5000);
    ASSERT_EQ(0,
              entity->add(key(5000), best.data(), best.size() * sizeof(float)));
    appended = true;
  };
  constexpr uint64_t previous_key = (uint64_t{1} << 61) + 17;
  IndexDocumentHeap heap(kTopk);
  heap.emplace(previous_key, -10000.0f);
  expected.emplace_back(-10000.0f, previous_key);
  expected.emplace_back(-5001.0f, key(5000));
  const auto query = ip_row(0);
  IndexContext::Stats query_stats;
  query_stats.set_dist_calced_count(7);
  query_stats.set_filtered_count(9);
  uint32_t scanned = 5;
  quantizer->calls.clear();
  ASSERT_EQ(0, entity->search(query.data(), IndexFilter(), &scanned, &heap,
                              &query_stats));
  ASSERT_TRUE(appended);
  ASSERT_FALSE(quantizer->calls.empty());
  EXPECT_EQ(135U, scanned);
  EXPECT_EQ(137U, query_stats.dist_calced_count());
  EXPECT_EQ(9U, query_stats.filtered_count());
  std::sort(expected.begin(), expected.end());
  expected.resize(kTopk);
  heap.sort();
  ASSERT_EQ(kTopk, heap.size());
  for (size_t i = 0; i < heap.size(); ++i) {
    EXPECT_EQ(expected[i].second, heap[i].key());
    EXPECT_FLOAT_EQ(expected[i].first, heap[i].score());
  }
  ASSERT_EQ(0, entity->close());
}

TEST_P(FlatStreamerBatchTest, LegacyMetricAndColumnConfigurationRemainUsable) {
  for (bool column : {false, true}) {
    for (const char *metric : {"SquaredEuclidean", "Euclidean"}) {
      SCOPED_TRACE(column);
      SCOPED_TRACE(metric);
      IndexMeta meta(IndexMeta::DT_FP32, kDimension);
      meta.set_metric(metric, 0, ailego::Params());
      ailego::Params params;
      params.set(PARAM_FLAT_COLUMN_MAJOR_ORDER, column);
      params.set(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, std::get<1>(GetParam()));
      storage_ = IndexFactory::CreateStorage(std::get<0>(GetParam()));
      ASSERT_NE(nullptr, storage_);
      ASSERT_EQ(0, storage_->init(ailego::Params()));
      ASSERT_EQ(0, storage_->open(path_, true));
      streamer_ = IndexFactory::CreateStreamer("FlatStreamer");
      ASSERT_NE(nullptr, streamer_);
      ASSERT_EQ(0, streamer_->init(meta, params));
      ASSERT_EQ(0, streamer_->open(storage_));
      auto context = streamer_->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(7);
      IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDimension);
      std::vector<float> row(kDimension);
      for (size_t i = 0; i < 67; ++i) {
        std::fill(row.begin(), row.end(), float(i));
        ASSERT_EQ(0, streamer_->add_impl(key(i), row.data(), qmeta, context));
      }
      ASSERT_EQ(0, streamer_->flush(0));
      ASSERT_EQ(0, streamer_->close());
      ASSERT_EQ(0, streamer_->open(storage_));
      context = streamer_->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(7);
      std::fill(row.begin(), row.end(), 12.125f);
      for (bool filtered : {false, true}) {
        if (filtered) {
          context->set_filter([](uint64_t value) { return value == key(12); });
        } else {
          context->reset_filter();
        }
        ASSERT_EQ(0, streamer_->search_impl(row.data(), qmeta, context));
        std::vector<std::pair<float, uint64_t>> expected;
        for (size_t i = 0; i < 67; ++i) {
          if (filtered && i == 12) continue;
          float score =
              kDimension * (float(i) - 12.125f) * (float(i) - 12.125f);
          if (std::string(metric) == "Euclidean") score = std::sqrt(score);
          expected.emplace_back(score, key(i));
        }
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(7U, context->result().size());
        for (size_t i = 0; i < 7; ++i) {
          EXPECT_EQ(expected[i].second, context->result()[i].key());
          EXPECT_NEAR(expected[i].first, context->result()[i].score(), 1e-4f);
        }
      }
      ASSERT_EQ(0, streamer_->close());
      streamer_.reset();
      ASSERT_EQ(0, storage_->close());
      storage_.reset();
      test_util::RemoveTestPath("flat_streamer_batch_test");
    }
  }
}

INSTANTIATE_TEST_SUITE_P(StorageLayouts, FlatStreamerBatchTest,
                         testing::Combine(testing::Values("MMapFileStorage",
                                                          "BufferStorage"),
                                          testing::Bool()));

}  // namespace
