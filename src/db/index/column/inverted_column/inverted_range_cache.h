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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <roaring/roaring.h>
#include "db/common/rocksdb_context.h"

namespace zvec {

enum class RangeQueryPath {
  ScanLeft,
  ScanRight,
  PrefixAdd,
  PrefixSubtract,
  PrefixSubtractAddBoundary,
};

struct RangeQueryPlan {
  RangeQueryPath path{RangeQueryPath::ScanLeft};
  size_t prefix_index{SIZE_MAX};
  size_t range_begin{0};
  size_t range_end{0};
  size_t boundary_bucket{SIZE_MAX};
  bool left_inclusive{false};
  bool flip{false};
  uint64_t estimated_work{0};
  // Optional prefix ending at the boundary bucket. The fields above describe
  // the original plan if the bounded correction cannot complete.
  size_t boundary_subtract_prefix{SIZE_MAX};
  uint64_t boundary_subtract_budget{0};
};

// Immutable cumulative bitmaps built from sealed RANGES. The fixed sixteenth
// grid has at most 15 entries; it has no cache budget or eviction policy.
class InvertedRangeCache {
 public:
  // Byte-work proxy for setting up an owned container and its payload.
  static constexpr uint64_t kContainerSetupWork = 128;
  static constexpr size_t kMaxBoundaryProbeTerms = 2048;
  static constexpr unsigned kQuantiles = 16;
  static constexpr const char *kNoCacheSealValue = "sealed-no-range-prefix-v1";
  static uint64_t short_term_work(const rocksdb::Slice &value);
  struct BitmapDeleter {
    void operator()(roaring_bitmap_t *bitmap) const {
      roaring_bitmap_free(bitmap);
    }
  };
  using BitmapPtr = std::unique_ptr<roaring_bitmap_t, BitmapDeleter>;
  struct Bucket {
    std::string key;
    std::string begin;
    std::string end;
    uint64_t cumulative_cardinality{0};
    uint64_t cumulative_decode_work{0};
  };
  struct SuffixCheckpoint {
    std::string first_term;
    uint64_t work;
  };
  struct Prefix {
    size_t end_bucket{0};
    BitmapPtr bitmap;
    uint64_t copy_bytes{0};
    std::vector<SuffixCheckpoint> suffix_checkpoints;
  };

  // Returns no cache when source buckets cannot safely support prefix plans.
  static std::unique_ptr<InvertedRangeCache> Build(
      RocksdbContext &context, rocksdb::ColumnFamilyHandle *ranges,
      rocksdb::ColumnFamilyHandle *terms, uint32_t max_id,
      const roaring_bitmap_t *null_bitmap);

  // Store the bitmaps separately to avoid assembling a single large value.
  // The caller publishes the resulting metadata as the column's seal value.
  rocksdb::Status store(RocksdbContext &context, const std::string &field,
                        std::string &metadata) const;
  static std::unique_ptr<InvertedRangeCache> Load(
      RocksdbContext &context, const std::string &field,
      const std::string &metadata, uint32_t max_id,
      const roaring_bitmap_t *null_bitmap);
  static rocksdb::Status Delete(rocksdb::WriteBatch &batch,
                                const std::string &field, size_t keep = 0);

  std::vector<Bucket> buckets;
  std::vector<Prefix> prefixes;
  uint64_t total_cardinality{0};
};

}  // namespace zvec
