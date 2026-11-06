// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <roaring/roaring.h>
#include "db/common/rocksdb_context.h"

namespace zvec {

// Runtime policy only: neither this policy nor the derived cache is persisted.
struct RangePrefixCacheOptions {
  size_t max_bytes{16ULL * 1024 * 1024};
  bool allow_subtraction{true};
};

enum class RangeQueryPath {
  Legacy,
  ScanLeft,
  ScanRight,
  PrefixAdd,
  PrefixSubtract
};

struct RangeQueryPlan {
  RangeQueryPath path{RangeQueryPath::Legacy};
  size_t prefix_index{SIZE_MAX};
  size_t range_begin{0};
  size_t range_end{0};
  size_t boundary_bucket{SIZE_MAX};
  bool left_inclusive{false};
  bool flip{false};
  uint64_t residual_bytes{0};
  uint64_t residual_cardinality{0};
  uint64_t estimated_work{0};
};

// Immutable, optional acceleration over the existing sealed RANGES CF.
// A failed admission/build leaves the original search path available.
class InvertedRangeCache {
 public:
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
    uint64_t cardinality{0};
    uint64_t bytes{0};
    uint64_t cumulative_cardinality{0};
    uint64_t cumulative_bytes{0};
  };

  struct Prefix {
    // Exclusive end in buckets; the exact value boundary is buckets[end-1].end.
    size_t end_bucket{0};
    BitmapPtr bitmap;
    uint64_t cardinality{0};
    uint64_t copy_bytes{0};
  };

  struct Stats {
    double build_ms{0};
    size_t accounted_bytes{0};
    size_t build_reservation_bytes{0};
    uint64_t source_bytes{0};
  };

  static std::unique_ptr<InvertedRangeCache> Build(
      RocksdbContext &context, rocksdb::ColumnFamilyHandle *ranges,
      uint32_t max_id, const roaring_bitmap_t *null_bitmap, size_t max_bytes);

  ~InvertedRangeCache();
  InvertedRangeCache(const InvertedRangeCache &) = delete;
  InvertedRangeCache &operator=(const InvertedRangeCache &) = delete;

  std::vector<Bucket> buckets;
  std::vector<Prefix> prefixes;
  uint64_t total_cardinality{0};
  Stats stats;

 private:
  InvertedRangeCache() = default;
  size_t charged_bytes_{0};
};

}  // namespace zvec
