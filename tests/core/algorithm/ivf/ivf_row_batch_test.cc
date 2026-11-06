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
#include <cstring>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include "algorithm/ivf/ivf_entity.h"
#include "quantizer/distance_quantizer.h"

namespace zvec::core {
namespace {

// Separate page allocations expose accidental reads through span boundaries.
class RowBatchSegment : public IndexStorage::Segment {
 public:
  explicit RowBatchSegment(std::vector<uint8_t> bytes, size_t page_size = 0)
      : bytes_(std::move(bytes)), page_size_(page_size) {
    if (page_size_) {
      for (size_t off = 0; off < bytes_.size(); off += page_size_) {
        const size_t size = std::min(page_size_, bytes_.size() - off);
        pages_.emplace_back(bytes_.begin() + off, bytes_.begin() + off + size);
      }
    }
  }
  size_t data_size() const override {
    return bytes_.size();
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return bytes_.size();
  }
  size_t fetch(size_t off, void *out, size_t len) const override {
    if (off > bytes_.size()) return 0;
    len = std::min(len, bytes_.size() - off);
    std::memcpy(out, bytes_.data() + off, len);
    return len;
  }
  size_t read(size_t off, IndexStorage::MemoryBlock &out, size_t len) override {
    if (off > bytes_.size()) return 0;
    len = std::min(len, bytes_.size() - off);
    out.reset(bytes_.data() + off);
    return len;
  }
  size_t read_scatter(size_t off, ScatterBlock &out, size_t len) override {
    if (!page_size_) return Segment::read_scatter(off, out, len);
    if (off > bytes_.size()) return 0;
    len = std::min(len, bytes_.size() - off);
    std::vector<ReadSpan> spans;
    for (size_t processed = 0; processed < len;) {
      const size_t pos = off + processed;
      const size_t page = pos / page_size_;
      const size_t in_page = pos % page_size_;
      const size_t size =
          std::min(len - processed, pages_[page].size() - in_page);
      spans.push_back({pages_[page].data() + in_page, size});
      processed += size;
    }
    out.reset_scattered(std::move(spans), {}, len);
    return len;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<RowBatchSegment>(*this);
  }

  bool contains(const void *data, size_t size) const {
    const auto addr = reinterpret_cast<uintptr_t>(data);
    const auto within = [&](const std::vector<uint8_t> &bytes) {
      const auto base = reinterpret_cast<uintptr_t>(bytes.data());
      return addr >= base && addr - base <= bytes.size() &&
             size <= bytes.size() - (addr - base);
    };
    if (!page_size_) return within(bytes_);
    return std::any_of(pages_.begin(), pages_.end(), within);
  }

 private:
  std::vector<uint8_t> bytes_;
  size_t page_size_;
  std::vector<std::vector<uint8_t>> pages_;
};

template <typename T>
std::vector<uint8_t> Bytes(const std::vector<T> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

class RowBatchEntity : public IVFEntity {
 public:
  static constexpr uint32_t kCount = 392;
  static constexpr uint32_t kIdOffset = 9;
  static uint64_t Key(uint32_t row) {
    return 1000 + 13 * row;
  }
  static bool Invalid(uint32_t row) {
    return row == 17 || row == 366;
  }

  void prepare(IndexMeta::DataType type, bool paged = false,
               size_t block_padding = 0, bool pad_tail = false,
               size_t dimension = 8) {
    const size_t unit = type == IndexMeta::DT_FP16 ? 2 : 4;
    // Legacy cosine describes its float norm tail as additional dimensions.
    meta_.set_meta(type, dimension + sizeof(float) / unit);
    meta_.set_metric("Cosine", 0, ailego::Params());
    meta_.set_major_order(IndexMeta::MO_ROW);
    distance_quantizer_ = CreateDistanceQuantizer(meta_);
    ASSERT_TRUE(distance_quantizer_);
    ASSERT_EQ(distance_quantizer_->dim(), dimension);
    const size_t stride = meta_.element_size();
    ASSERT_EQ(stride, dimension * unit + sizeof(float));
    header_.block_vector_count = 32;
    header_.block_size = 32 * stride + block_padding;
    header_.total_vector_count = kIdOffset + kCount;
    header_.inverted_list_count = 2;
    norm_value_ = 1.f;

    std::vector<InvertedListMeta> lists(2);
    std::vector<uint8_t> body;
    uint32_t ordinal = 0;
    const uint32_t counts[] = {355, 37};
    for (size_t list = 0; list < lists.size(); ++list) {
      auto &meta = lists[list];
      meta.offset = body.size();
      meta.vector_count = counts[list];
      meta.block_count = (counts[list] + 31) / 32;
      meta.id_offset = kIdOffset + ordinal;
      header_.block_count += meta.block_count;
      for (uint32_t block = 0; block < meta.block_count; ++block) {
        const size_t rows = std::min(uint32_t(32), counts[list] - block * 32);
        const size_t size =
            (pad_tail || block_padding) ? header_.block_size : rows * stride;
        const size_t start = body.size();
        body.resize(start + size, 0xee);
        for (size_t row = 0; row < rows; ++row, ++ordinal)
          std::memcpy(body.data() + start + row * stride, &ordinal,
                      sizeof(ordinal));
      }
    }
    header_.inverted_body_size = body.size();
    postings_ = std::make_shared<RowBatchSegment>(std::move(body),
                                                  paged ? 5 * stride + 4 : 0);
    inverted_ = postings_;
    inverted_meta_ = std::make_shared<RowBatchSegment>(Bytes(lists));
    std::vector<uint64_t> keys(kIdOffset + kCount, kInvalidKey);
    for (uint32_t row = 0; row < kCount; ++row)
      keys[kIdOffset + row] = Invalid(row) ? kInvalidKey : Key(row);
    keys_ = std::make_shared<RowBatchSegment>(Bytes(keys));
    calculator_ = std::make_shared<IVFDistanceCalculator>(meta_, nullptr, 32);
    const auto scalar = [](const void *, const void *, size_t, float *) {};
    const auto batch = [this, stride](const void **rows, const void *, size_t n,
                                      size_t, float *out, const void **) {
      batches.push_back(n);
      for (size_t i = 0; i < n; ++i) {
        uint32_t row;
        std::memcpy(&row, rows[i], sizeof(row));
        EXPECT_LT(row, kCount);  // Padding/next-cluster records must not leak.
        seen.push_back(row);
        out[i] = static_cast<float>(row);
        if (postings_->contains(rows[i], stride)) {
          if (i)
            EXPECT_EQ(static_cast<const uint8_t *>(rows[i]),
                      static_cast<const uint8_t *>(rows[i - 1]) + stride);
        } else {
          EXPECT_EQ(n, 1u);  // Only one split row may use existing scratch.
          ++split_rows;
        }
      }
      if (n > 1) EXPECT_TRUE(postings_->contains(rows[0], n * stride));
    };
    query_distance_ = turbo::DistanceImpl(scalar, batch, {}, dimension);
  }

  std::vector<size_t> batches;
  std::vector<uint32_t> seen;
  size_t split_rows = 0;

 private:
  std::shared_ptr<RowBatchSegment> postings_;
};

void CheckSearch(RowBatchEntity &entity, bool filtered = false,
                 bool reject_all = false) {
  IndexDocumentHeap heap(RowBatchEntity::kCount);
  IndexContext::Stats stats;
  IndexFilter filter;
  filter.set([reject_all](uint64_t key) {
    return reject_all || (key != kInvalidKey && (key - 1000) / 13 % 5 == 0);
  });
  for (size_t list = 0; list < 2; ++list) {
    uint32_t scanned = 0;
    const int result =
        filtered ? entity.search(list, nullptr, filter, &scanned, &heap, &stats)
                 : entity.search(list, nullptr, &scanned, &heap, &stats);
    ASSERT_EQ(result, 0);
    EXPECT_EQ(scanned, list == 0 ? 355u : 37u);
  }
  size_t valid = 0, removed = 0;
  for (uint32_t row = 0; row < RowBatchEntity::kCount; ++row) {
    const uint64_t key =
        RowBatchEntity::Invalid(row) ? kInvalidKey : RowBatchEntity::Key(row);
    if (filtered && filter(key))
      ++removed;
    else if (key != kInvalidKey)
      ++valid;
  }
  EXPECT_EQ(heap.size(), valid);
  EXPECT_EQ(stats.filtered_count(), removed);
  EXPECT_EQ(stats.dist_calced_count(),
            reject_all ? 0u : RowBatchEntity::kCount);
  for (const auto &doc : heap) {
    const uint32_t row = doc.index() - RowBatchEntity::kIdOffset;
    EXPECT_EQ(doc.key(), RowBatchEntity::Key(row));
    EXPECT_FALSE(RowBatchEntity::Invalid(row));
    EXPECT_FLOAT_EQ(doc.score(), static_cast<float>(row));
    if (filtered) EXPECT_FALSE(filter(doc.key()));
  }
  if (!reject_all) {
    ASSERT_EQ(entity.seen.size(), RowBatchEntity::kCount);
    for (uint32_t row = 0; row < RowBatchEntity::kCount; ++row)
      EXPECT_EQ(entity.seen[row], row);
  } else {
    EXPECT_TRUE(entity.seen.empty());
  }
}

TEST(IVFRowBatch, ContiguousNormTailsAndClusterTails) {
  for (auto type : {IndexMeta::DT_FP32, IndexMeta::DT_FP16})
    for (bool pad_tail : {false, true}) {
      RowBatchEntity entity;
      ASSERT_NO_FATAL_FAILURE(entity.prepare(type, false, 0, pad_tail));
      CheckSearch(entity);
      ASSERT_FALSE(entity.batches.empty());
      const size_t largest =
          *std::max_element(entity.batches.begin(), entity.batches.end());
      if (type == IndexMeta::DT_FP32) {
        EXPECT_GT(largest, 32u);
      } else {
        EXPECT_LE(largest, 32u);
      }
      EXPECT_EQ(entity.split_rows, 0u);
    }
}

TEST(IVFRowBatch, ScatterSpansOnlyCopySplitRows) {
  for (auto type : {IndexMeta::DT_FP32, IndexMeta::DT_FP16}) {
    RowBatchEntity entity;
    ASSERT_NO_FATAL_FAILURE(entity.prepare(type, true));
    CheckSearch(entity);
    ASSERT_FALSE(entity.batches.empty());
    EXPECT_LE(*std::max_element(entity.batches.begin(), entity.batches.end()),
              5u);
    EXPECT_GT(entity.split_rows, 0u);
  }
}

TEST(IVFRowBatch, PayloadBudgetAdaptsToDimension) {
  // A batch never combines clusters, includes padding, or splits a storage
  // block just to meet the target. Odd dimensions keep their norm-tail stride.
  const std::pair<size_t, size_t> cases[] = {{256, 355},  {1024, 256},
                                             {1537, 160}, {4096, 64},
                                             {8192, 32},  {16384, 32}};
  for (const auto &test : cases) {
    RowBatchEntity entity;
    ASSERT_NO_FATAL_FAILURE(
        entity.prepare(IndexMeta::DT_FP32, false, 0, false, test.first));
    CheckSearch(entity);
    ASSERT_FALSE(entity.batches.empty());
    EXPECT_EQ(*std::max_element(entity.batches.begin(), entity.batches.end()),
              test.second);
    EXPECT_EQ(entity.split_rows, 0u);
  }
  RowBatchEntity half;
  ASSERT_NO_FATAL_FAILURE(
      half.prepare(IndexMeta::DT_FP16, false, 0, false, 1024));
  CheckSearch(half);
  EXPECT_LE(*std::max_element(half.batches.begin(), half.batches.end()), 32u);
}

TEST(IVFRowBatch, PhysicalPaddingRetainsOriginalBlocks) {
  RowBatchEntity entity;
  ASSERT_NO_FATAL_FAILURE(entity.prepare(IndexMeta::DT_FP32, false, 16));
  CheckSearch(entity);
  ASSERT_FALSE(entity.batches.empty());
  EXPECT_LE(*std::max_element(entity.batches.begin(), entity.batches.end()),
            32u);
}

TEST(IVFRowBatch, FilterRetainsOriginalBlocksAndStats) {
  for (bool reject_all : {false, true}) {
    RowBatchEntity entity;
    ASSERT_NO_FATAL_FAILURE(entity.prepare(IndexMeta::DT_FP32, true));
    CheckSearch(entity, true, reject_all);
    if (!reject_all) {
      ASSERT_FALSE(entity.batches.empty());
      EXPECT_LE(*std::max_element(entity.batches.begin(), entity.batches.end()),
                32u);
    }
  }
}

TEST(IVFRowBatch, CalculatorRetainsDefault32AndAcceptsCallerCapacity) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, 9);
  IVFDistanceCalculator calculator(meta, nullptr, 32);
  std::vector<float> rows(97 * 9), output(97);
  std::vector<size_t> calls;
  const auto batch = [&](const void **pointers, const void *, size_t n, size_t,
                         float *out, const void **) {
    calls.push_back(n);
    for (size_t i = 0; i < n; ++i) {
      const size_t row =
          (static_cast<const float *>(pointers[i]) - rows.data()) / 9;
      out[i] = static_cast<float>(row);
    }
  };
  turbo::DistanceImpl distance({}, batch, {}, 8);
  calculator.query_features_distance(distance, rows.data(), 97, output.data());
  EXPECT_EQ(calls, (std::vector<size_t>{32, 32, 32, 1}));
  calls.clear();
  const void *pointers[64];
  calculator.query_features_distance(distance, rows.data(), 97, output.data(),
                                     pointers, 64);
  EXPECT_EQ(calls, (std::vector<size_t>{64, 33}));
  for (size_t i = 0; i < output.size(); ++i)
    EXPECT_FLOAT_EQ(output[i], float(i));
}

}  // namespace
}  // namespace zvec::core
