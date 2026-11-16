// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include "inverted_range_cache.h"
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "inverted_codec.h"

namespace zvec {
namespace {

size_t metadata_bytes(const InvertedRangeCache &cache) {
  size_t bytes = sizeof(cache) +
                 cache.buckets.capacity() * sizeof(InvertedRangeCache::Bucket) +
                 cache.prefixes.capacity() * sizeof(InvertedRangeCache::Prefix);
  for (const auto &bucket : cache.buckets) {
    // Counting SSO storage twice is intentional: admission is conservative.
    bytes += bucket.key.capacity() + bucket.begin.capacity() +
             bucket.end.capacity() + 3;
  }
  return bytes;
}

size_t bitmap_bytes(const roaring_bitmap_t *bitmap) {
  // Arrays and runs are shrunk before publication. Portable bytes account for
  // their payload; the per-container allowance covers heads, pointer/key/type
  // arrays and allocator rounding without depending on a platform allocator.
  return roaring_bitmap_portable_size_in_bytes(bitmap) + sizeof(*bitmap) +
         256ULL * bitmap->high_low_container.size;
}

}  // namespace

InvertedRangeCache::~InvertedRangeCache() {
  std::vector<Prefix>().swap(prefixes);
  std::vector<Bucket>().swap(buckets);
  if (charged_bytes_) {
    ailego::MemoryLimitPool::get_instance().release_external(charged_bytes_);
  }
}

std::unique_ptr<InvertedRangeCache> InvertedRangeCache::Build(
    RocksdbContext &context, rocksdb::ColumnFamilyHandle *ranges,
    uint32_t max_id, const roaring_bitmap_t *null_bitmap, size_t max_bytes) {
  if (!ranges || !max_bytes) {
    return nullptr;
  }
  const uint64_t document_domain = static_cast<uint64_t>(max_id) + 1;
  const uint64_t null_count =
      null_bitmap ? roaring_bitmap_get_cardinality(null_bitmap) : 0;
  if (null_count > document_domain ||
      (null_count && roaring_bitmap_maximum(null_bitmap) > max_id)) {
    return nullptr;
  }
  const auto start = std::chrono::steady_clock::now();
  auto cache = std::unique_ptr<InvertedRangeCache>(new InvertedRangeCache());
  const uint64_t blocks = (static_cast<uint64_t>(max_id) >> 16) + 1;
  // Bound the rolling bitmap, one decoded input and one snapshot/conversion.
  // Even an unoptimized array/run container cannot exceed 65536 uint16 slots
  // (or 32768 start/length pairs). This is logical admission, not allocation.
  const uint64_t scratch = 3 * (blocks * (131072 + 256) + 256);
  if (scratch > std::numeric_limits<size_t>::max() - max_bytes) {
    return nullptr;
  }
  const size_t reservation = max_bytes + static_cast<size_t>(scratch);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  if (!pool.try_charge_external(reservation)) {
    return nullptr;
  }
  cache->charged_bytes_ = reservation;
  cache->stats.build_reservation_bytes = reservation;

  rocksdb::ReadOptions read_options = context.read_opts_;
  read_options.fill_cache = false;
  std::unique_ptr<rocksdb::Iterator> iter(
      context.db_->NewIterator(read_options, ranges));
  uint64_t total_bytes = 0;
  size_t text_bytes = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    const size_t metadata = sizeof(*cache) + text_bytes +
                            cache->buckets.capacity() * sizeof(Bucket);
    if (iter->key().size() > max_bytes / 4 ||
        metadata + 4 * iter->key().size() + 2 * sizeof(Bucket) > max_bytes) {
      return nullptr;
    }
    const auto key = iter->key();
    if (key.size() < 9) {
      return nullptr;
    }
    uint64_t encoded_begin_size = 0;
    std::memcpy(&encoded_begin_size, key.data() + key.size() - 8, 8);
    if (encoded_begin_size > key.size() - 9 ||
        key[encoded_begin_size] != '\0') {
      return nullptr;
    }
    char *begin = nullptr, *end = nullptr;
    size_t begin_size = 0, end_size = 0;
    InvertedIndexCodec::decode_range_key(iter->key().data(), iter->key().size(),
                                         &begin, &begin_size, &end, &end_size);
    if (InvertedIndexCodec::CMP(begin, begin_size, end, end_size) > 0 ||
        (!cache->buckets.empty() &&
         InvertedIndexCodec::CMP(cache->buckets.back().end.data(),
                                 cache->buckets.back().end.size(), begin,
                                 begin_size) >= 0)) {
      // Binary search requires ordered disjoint value intervals. Reject the
      // optional cache when old/corrupt metadata does not satisfy that
      // contract.
      return nullptr;
    }
    roaring_bitmap_t *raw = nullptr;
    auto status = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                  iter->value().size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) {
      LOG_WARN("Skipping range prefix cache: failed to decode range");
      return nullptr;
    }
    const auto cardinality = roaring_bitmap_get_cardinality(bitmap.get());
    cache->total_cardinality += cardinality;
    total_bytes += iter->value().size();
    cache->buckets.push_back(
        {iter->key().ToString(), std::string(begin, begin_size),
         std::string(end, end_size), cardinality, iter->value().size(),
         cache->total_cardinality, total_bytes});
    const auto &added = cache->buckets.back();
    text_bytes += added.key.capacity() + added.begin.capacity() +
                  added.end.capacity() + 3;
    if (cache->total_cardinality > document_domain - null_count ||
        sizeof(*cache) + text_bytes +
                cache->buckets.capacity() * sizeof(Bucket) >
            max_bytes) {
      return nullptr;
    }
  }
  if (!iter->status().ok() || cache->buckets.size() < 2 ||
      cache->total_cardinality == 0 ||
      cache->total_cardinality + null_count != document_domain) {
    // The planner may introduce a complement that the original query did not
    // use. Its [0,max_id] domain must contain only indexed or NULL documents;
    // otherwise flipping a bitmap would manufacture matches for missing IDs.
    return nullptr;
  }

  cache->prefixes.reserve(9);
  BitmapPtr cumulative(roaring_bitmap_create());
  if (!cumulative) {
    return nullptr;
  }
  size_t bitmap_charge = 0;
  size_t bucket_index = 0;
  unsigned decile = 1;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next(), ++bucket_index) {
    if (bucket_index >= cache->buckets.size() ||
        iter->key() != cache->buckets[bucket_index].key) {
      return nullptr;
    }
    auto status = InvertedIndexCodec::merge_or(
        iter->value().data(), iter->value().size(), false, cumulative.get());
    if (!status.ok()) {
      return nullptr;
    }
    const auto cardinality =
        cache->buckets[bucket_index].cumulative_cardinality;
    if (decile <= 9 &&
        cardinality >= (cache->total_cardinality * decile + 9) / 10) {
      // A large term may cross several deciles. Keep its boundary once;
      // never split a term or create duplicate checkpoints.
      while (decile <= 9 &&
             cardinality >= (cache->total_cardinality * decile + 9) / 10) {
        ++decile;
      }
      if (bucket_index + 1 == cache->buckets.size()) {
        continue;
      }
      BitmapPtr prefix(roaring_bitmap_copy(cumulative.get()));
      if (!prefix) {
        return nullptr;
      }
      roaring_bitmap_run_optimize(prefix.get());
      roaring_bitmap_shrink_to_fit(prefix.get());
      bitmap_charge += bitmap_bytes(prefix.get());
      if (bitmap_charge + metadata_bytes(*cache) > max_bytes) {
        return nullptr;
      }
      const auto copy_bytes =
          roaring_bitmap_portable_size_in_bytes(prefix.get());
      cache->prefixes.push_back(
          {bucket_index + 1, std::move(prefix), cardinality, copy_bytes});
    }
  }
  if (!iter->status().ok() || bucket_index != cache->buckets.size() ||
      cache->prefixes.empty()) {
    return nullptr;
  }
  if (roaring_bitmap_get_cardinality(cumulative.get()) !=
          cache->total_cardinality ||
      roaring_bitmap_maximum(cumulative.get()) > max_id ||
      (null_count && roaring_bitmap_intersect(cumulative.get(), null_bitmap))) {
    // Equal posting and union cardinalities establish disjoint range sets.
    // Together with the NULL checks and domain size above, they also establish
    // full domain coverage without allocating another persistent bitmap.
    return nullptr;
  }
  cumulative.reset();
  iter.reset();
  cache->stats.source_bytes = 2 * total_bytes;
  cache->stats.accounted_bytes = bitmap_charge + metadata_bytes(*cache);
  pool.release_external(reservation - cache->stats.accounted_bytes);
  cache->charged_bytes_ = cache->stats.accounted_bytes;
  cache->stats.build_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  return cache;
}

}  // namespace zvec
