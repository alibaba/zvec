// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include "inverted_range_cache.h"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <limits>
#include <optional>
#include "inverted_codec.h"

namespace zvec {
namespace {

constexpr size_t kRoaringBlockCount = 1U << 16;
constexpr size_t kContainerOverhead = 256;

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
         kContainerOverhead * bitmap->high_low_container.size;
}

}  // namespace

InvertedRangeCache::~InvertedRangeCache() {
  std::vector<Prefix>().swap(prefixes);
  std::vector<Bucket>().swap(buckets);
  if (stats.accounted_bytes) {
    ailego::MemoryLimitPool::get_instance().release_heap_cache(
        stats.accounted_bytes, charge_stamp_);
  }
}

InvertedRangeCacheSlot::~InvertedRangeCacheSlot() {
  if (generation_) {
    ailego::BlockEvictionQueue::get_instance().set_invalid(this);
  }
}

void InvertedRangeCacheSlot::publish(
    std::unique_ptr<InvertedRangeCache> cache) {
  if (!cache) return;

  std::shared_ptr<const InvertedRangeCache> published(std::move(cache));
  auto &queue = ailego::BlockEvictionQueue::get_instance();
  queue.set_valid(this);
  std::lock_guard<std::mutex> lock(mutex_);
  generation_ = queue.next_version();
  cache_ = std::move(published);
  if (!queue.add_single_block({0, generation_, this},
                              ailego::BlockEvictionQueue::kProbationPriority)) {
    cache_.reset();
  }
}

std::shared_ptr<const InvertedRangeCache> InvertedRangeCacheSlot::pin() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_;
}

bool InvertedRangeCacheSlot::is_dead_block(ailego::eviction_key_t,
                                           ailego::version_t version) {
  std::lock_guard<std::mutex> lock(mutex_);
  return !cache_ || generation_ != version;
}

size_t InvertedRangeCacheSlot::eviction_bytes(ailego::eviction_key_t) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_ ? cache_->stats.accounted_bytes : 0;
}

bool InvertedRangeCacheSlot::evict_block(ailego::eviction_key_t) {
  std::shared_ptr<const InvertedRangeCache> retired;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Existing queries own the snapshot. Removing the slot's reference
    // stops new pins and lets the last reader return the charge directly.
    retired = std::move(cache_);
  }
  return retired.use_count() == 1;
}

void InvertedRangeCacheSlot::eviction_requeue_failed(
    ailego::eviction_key_t, ailego::version_t version) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (generation_ == version) {
    cache_.reset();
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
  auto &pool = ailego::MemoryLimitPool::get_instance();
  using PendingAllocation = ailego::MemoryLimitPool::PendingAllocation;
  constexpr auto kind = PendingAllocation::Kind::kEvictableCache;
  PendingAllocation budget(pool, kind);
  if (!pool.can_reuse_cache_backing() || !budget.reserve(max_bytes)) {
    return nullptr;
  }
  std::optional<PendingAllocation> scratch_budget;
  size_t peak_reservation = max_bytes;
  const auto reserve_scratch = [&](uint64_t scratch) {
    if (scratch > std::numeric_limits<size_t>::max() - max_bytes ||
        !pool.can_reuse_cache_backing()) {
      return false;
    }
    // Resize only between postings or build phases, with no live scratch.
    scratch_budget.emplace(pool, kind);
    if (!scratch_budget->reserve(static_cast<size_t>(scratch))) {
      return false;
    }
    peak_reservation =
        std::max(peak_reservation, max_bytes + static_cast<size_t>(scratch));
    return true;
  };
  std::bitset<kRoaringBlockCount> occupied_blocks;
  if (!reserve_scratch(sizeof(occupied_blocks))) {
    return nullptr;
  }
  const auto start = std::chrono::steady_clock::now();
  auto cache = std::unique_ptr<InvertedRangeCache>(new InvertedRangeCache());

  rocksdb::ReadOptions read_options = context.read_opts_;
  read_options.fill_cache = false;
  std::unique_ptr<rocksdb::Iterator> iter(
      context.db_->NewIterator(read_options, ranges));
  uint64_t total_bytes = 0;
  size_t text_bytes = 0;
  size_t max_posting_blocks = 0;
  size_t max_decode_scratch = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    if (!pool.can_reuse_cache_backing()) return nullptr;
    const size_t metadata = sizeof(*cache) + text_bytes +
                            cache->buckets.capacity() * sizeof(Bucket);
    if (iter->key().size() > max_bytes / 4 ||
        metadata + 4 * iter->key().size() + 2 * sizeof(Bucket) > max_bytes) {
      return nullptr;
    }
    const auto key = iter->key();
    const char *begin = nullptr, *end = nullptr;
    size_t begin_size = 0, end_size = 0;
    if (!InvertedIndexCodec::decode_range_key(key.data(), key.size(), &begin,
                                              &begin_size, &end, &end_size)) {
      return nullptr;
    }
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
    const auto value = iter->value();
    // Safe decoding copies the payload. A portable bitmap has at least four
    // header bytes per container; an encoded ID list has four bytes per ID.
    const size_t decode_overhead =
        kContainerOverhead *
            (1 + std::min(kRoaringBlockCount, value.size() / 4)) +
        sizeof(occupied_blocks);
    if (value.size() > std::numeric_limits<size_t>::max() - decode_overhead) {
      return nullptr;
    }
    const size_t decode_scratch = value.size() + decode_overhead;
    if (decode_scratch > max_decode_scratch) {
      if (!reserve_scratch(decode_scratch)) return nullptr;
      max_decode_scratch = decode_scratch;
    }
    roaring_bitmap_t *raw = nullptr;
    auto status =
        InvertedIndexCodec::Deserialize(value.data(), value.size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) {
      LOG_WARN("Skipping range prefix cache: failed to decode range");
      return nullptr;
    }
    const auto cardinality = roaring_bitmap_get_cardinality(bitmap.get());
    const auto &containers = bitmap->high_low_container;
    for (int32_t i = 0; i < containers.size; ++i) {
      occupied_blocks.set(containers.keys[i]);
    }
    max_posting_blocks =
        std::max(max_posting_blocks, static_cast<size_t>(containers.size));
    cache->total_cardinality += cardinality;
    total_bytes += value.size();
    cache->buckets.push_back(
        {iter->key().ToString(), std::string(begin, begin_size),
         std::string(end, end_size), cardinality, value.size(),
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

  // Reserve for the rolling bitmap, a checkpoint copy and a container
  // conversion. Only platforms supporting frozen postings avoid a decode
  // buffer for the source bitmap.
  const uint64_t block_bytes = 2ULL * kRoaringBlockCount + kContainerOverhead;
  const uint64_t source_scratch =
      InvertedIndexCodec::kFrozenPostings
          ? kContainerOverhead * (max_posting_blocks + 3) +
                sizeof(occupied_blocks)
          : max_decode_scratch;
  const uint64_t merge_scratch =
      (2 * occupied_blocks.count() + 1) * block_bytes + source_scratch;
  if (!reserve_scratch(merge_scratch)) {
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
    if (!pool.can_reuse_cache_backing()) return nullptr;
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

  // Preserve all original decile checkpoints. Spend only remaining budget on
  // a bounded, query-independent refinement: half, quarters, eighths, etc. of
  // serialized posting work. Exact checkpoints and original plans stay usable;
  // a new threshold can merge fewer residual postings without a result cache.
  constexpr size_t max_prefixes = 32;
  uint64_t extra_source_bytes = 0;
  bool full = false;
  for (uint64_t denominator = 2;
       denominator <= 64 && cache->prefixes.size() < max_prefixes && !full;
       denominator *= 2) {
    for (uint64_t numerator = 1;
         numerator < denominator && cache->prefixes.size() < max_prefixes;
         numerator += 2) {
      // This form avoids overflowing total_bytes * numerator.
      const uint64_t target =
          total_bytes / denominator * numerator +
          ((total_bytes % denominator) * numerator + denominator - 1) /
              denominator;
      auto bucket =
          std::lower_bound(cache->buckets.begin(), cache->buckets.end(), target,
                           [](const Bucket &entry, uint64_t bytes) {
                             return entry.cumulative_bytes < bytes;
                           });
      const size_t end = bucket - cache->buckets.begin() + 1;
      if (end >= cache->buckets.size()) continue;
      auto position =
          std::lower_bound(cache->prefixes.begin(), cache->prefixes.end(), end,
                           [](const Prefix &prefix, size_t boundary) {
                             return prefix.end_bucket < boundary;
                           });
      if (position != cache->prefixes.end() && position->end_bucket == end)
        continue;
      const size_t begin =
          position == cache->prefixes.begin() ? 0 : (position - 1)->end_bucket;
      BitmapPtr prefix(begin ? roaring_bitmap_copy((position - 1)->bitmap.get())
                             : roaring_bitmap_create());
      if (!prefix) {
        full = true;
        break;
      }
      iter->Seek(cache->buckets[begin].key);
      for (size_t i = begin; i < end; ++i, iter->Next()) {
        if (!pool.can_reuse_cache_backing()) return nullptr;
        if (!iter->Valid() || iter->key() != cache->buckets[i].key)
          return nullptr;
        const auto value = iter->value();
        auto status = InvertedIndexCodec::merge_or(value.data(), value.size(),
                                                   false, prefix.get());
        if (!status.ok()) return nullptr;
        extra_source_bytes += value.size();
      }
      if (!iter->status().ok() ||
          roaring_bitmap_get_cardinality(prefix.get()) !=
              cache->buckets[end - 1].cumulative_cardinality)
        return nullptr;
      roaring_bitmap_run_optimize(prefix.get());
      roaring_bitmap_shrink_to_fit(prefix.get());
      const size_t charge = bitmap_bytes(prefix.get());
      const size_t growth =
          cache->prefixes.capacity() < max_prefixes
              ? (max_prefixes - cache->prefixes.capacity()) * sizeof(Prefix)
              : 0;
      const size_t accounted = bitmap_charge + metadata_bytes(*cache);
      if (accounted > max_bytes) return nullptr;
      const size_t remaining = max_bytes - accounted;
      if (charge > remaining || growth > remaining - charge) {
        full = true;
        break;
      }
      if (growth) cache->prefixes.reserve(max_prefixes);
      position =
          std::lower_bound(cache->prefixes.begin(), cache->prefixes.end(), end,
                           [](const Prefix &entry, size_t boundary) {
                             return entry.end_bucket < boundary;
                           });
      const auto copy_bytes =
          roaring_bitmap_portable_size_in_bytes(prefix.get());
      cache->prefixes.insert(
          position,
          {end, std::move(prefix),
           cache->buckets[end - 1].cumulative_cardinality, copy_bytes});
      bitmap_charge += charge;
    }
  }
  iter.reset();
  if (!pool.can_reuse_cache_backing()) return nullptr;
  cache->stats.build_reservation_bytes = peak_reservation;
  cache->stats.source_bytes = 2 * total_bytes + extra_source_bytes;
  cache->stats.accounted_bytes = bitmap_charge + metadata_bytes(*cache);
  cache->charge_stamp_ = budget.commit(cache->stats.accounted_bytes);
  cache->stats.build_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  return cache;
}

}  // namespace zvec
