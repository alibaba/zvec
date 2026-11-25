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

#include "inverted_range_cache.h"
#include <cstring>
#include "inverted_codec.h"

namespace zvec {
namespace {

bool decode_bucket(const std::string &key, InvertedRangeCache::Bucket &bucket) {
  if (key.size() < 9) return false;
  uint64_t size = 0;
  std::memcpy(&size, key.data() + key.size() - 8, 8);
  if (size > key.size() - 9 || key[size] != '\0') return false;
  bucket.key = key;
  bucket.begin.assign(key.data(), size);
  bucket.end.assign(key.data() + size + 1, key.size() - size - 9);
  return InvertedIndexCodec::CMP(bucket.begin.data(), bucket.begin.size(),
                                 bucket.end.data(), bucket.end.size()) <= 0;
}

// Record upper bounds on work for suffixes of this bucket. Keep only powers
// of two and the final eligible suffix to bound memory per prefix.
std::vector<InvertedRangeCache::SuffixCheckpoint> build_suffix_checkpoints(
    rocksdb::Iterator &terms, const InvertedRangeCache::Bucket &bucket) {
  std::vector<InvertedRangeCache::SuffixCheckpoint> checkpoints;
  std::string first_term;
  uint64_t work = 0;
  terms.SeekForPrev(bucket.end);
  for (size_t n = 1;
       n <= InvertedRangeCache::kMaxBoundaryProbeTerms && terms.Valid();
       ++n, terms.Prev()) {
    const auto key = terms.key();
    // The optional metadata should stay small even for long string keys.
    if (key.size() > 128 ||
        InvertedIndexCodec::CMP(key.data(), key.size(), bucket.begin.data(),
                                bucket.begin.size()) < 0) {
      break;
    }
    const uint64_t term_work =
        InvertedRangeCache::short_term_work(terms.value());
    if (term_work == 0) break;
    work += term_work;
    first_term.assign(key.data(), key.size());
    if ((n & (n - 1)) == 0) checkpoints.push_back({first_term, work});
  }
  if (!terms.status().ok()) return {};
  if (work && checkpoints.back().work != work) {
    checkpoints.push_back({std::move(first_term), work});
  }
  return checkpoints;
}

}  // namespace

uint64_t InvertedRangeCache::short_term_work(const rocksdb::Slice &value) {
  if (value.empty()) return 0;
  const auto count = static_cast<unsigned char>(value[0]);
  if (count == 0 || count > INVERT_ID_LIST_SIZE_THRESHOLD ||
      value.size() != 1 + count * sizeof(uint32_t)) {
    return 0;
  }
  return kContainerSetupWork + 2 * value.size();
}

std::unique_ptr<InvertedRangeCache> InvertedRangeCache::Build(
    RocksdbContext &context, rocksdb::ColumnFamilyHandle *ranges,
    rocksdb::ColumnFamilyHandle *terms, uint32_t max_id,
    const roaring_bitmap_t *null_bitmap) {
  if (!ranges) return nullptr;
  const uint64_t domain = uint64_t(max_id) + 1;
  const uint64_t null_count =
      null_bitmap ? roaring_bitmap_get_cardinality(null_bitmap) : 0;
  if (null_count > domain ||
      (null_count && roaring_bitmap_maximum(null_bitmap) > max_id))
    return nullptr;
  auto cache = std::make_unique<InvertedRangeCache>();
  rocksdb::ReadOptions options = context.read_opts_;
  options.fill_cache = false;
  std::unique_ptr<rocksdb::Iterator> iter(
      context.db_->NewIterator(options, ranges));
  uint64_t source_bytes = 0;
  uint64_t decode_work = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    Bucket bucket;
    if (!decode_bucket(iter->key().ToString(), bucket) ||
        (!cache->buckets.empty() &&
         InvertedIndexCodec::CMP(
             cache->buckets.back().end.data(), cache->buckets.back().end.size(),
             bucket.begin.data(), bucket.begin.size()) >= 0))
      return nullptr;
    roaring_bitmap_t *raw = nullptr;
    auto status = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                  iter->value().size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) return nullptr;
    bucket.cardinality = roaring_bitmap_get_cardinality(bitmap.get());
    bucket.bytes = iter->value().size();
    if (bucket.cardinality > domain - null_count - cache->total_cardinality ||
        bucket.bytes > UINT64_MAX - source_bytes)
      return nullptr;
    cache->total_cardinality += bucket.cardinality;
    source_bytes += bucket.bytes;
    bucket.cumulative_cardinality = cache->total_cardinality;
    bucket.cumulative_bytes = source_bytes;
    // Conservative byte-work bound: an unaligned view copies its payload
    // before merging. Keep descriptor setup separate from those data passes.
    decode_work +=
        bucket.bytes * 2 +
        uint64_t(bitmap->high_low_container.size) * kContainerSetupWork;
    bucket.cumulative_decode_work = decode_work;
    cache->buckets.push_back(std::move(bucket));
  }
  // Complement plans require a complete DocID domain with NULLs excluded.
  if (!iter->status().ok() || cache->buckets.size() < 2 ||
      !cache->total_cardinality ||
      cache->total_cardinality + null_count != domain)
    return nullptr;

  cache->prefixes.reserve(15);
  BitmapPtr cumulative(roaring_bitmap_create());
  if (!cumulative) return nullptr;
  size_t bucket_index = 0;
  unsigned quantile = 1;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next(), ++bucket_index) {
    if (bucket_index >= cache->buckets.size() ||
        iter->key() != cache->buckets[bucket_index].key)
      return nullptr;
    roaring_bitmap_t *raw = nullptr;
    auto status = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                  iter->value().size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) return nullptr;
    roaring_bitmap_or_inplace(cumulative.get(), bitmap.get());
    const uint64_t cardinality =
        cache->buckets[bucket_index].cumulative_cardinality;
    if (quantile > 15 ||
        cardinality < (cache->total_cardinality * quantile + 15) / 16)
      continue;
    do {
      ++quantile;
    } while (quantile <= 15 &&
             cardinality >= (cache->total_cardinality * quantile + 15) / 16);
    // A repeated value may cross multiple quantiles. Store that boundary once,
    // and omit the final full-domain prefix.
    if (bucket_index + 1 == cache->buckets.size()) continue;
    BitmapPtr prefix(roaring_bitmap_copy(cumulative.get()));
    if (!prefix) return nullptr;
    roaring_bitmap_run_optimize(prefix.get());
    roaring_bitmap_shrink_to_fit(prefix.get());
    const auto copy_bytes = roaring_bitmap_portable_size_in_bytes(prefix.get());
    cache->prefixes.push_back(
        {bucket_index + 1, std::move(prefix), copy_bytes, {}});
  }
  if (!iter->status().ok() || bucket_index != cache->buckets.size() ||
      cache->prefixes.empty() ||
      roaring_bitmap_get_cardinality(cumulative.get()) !=
          cache->total_cardinality ||
      roaring_bitmap_maximum(cumulative.get()) > max_id ||
      (null_count && roaring_bitmap_intersect(cumulative.get(), null_bitmap)))
    return nullptr;
  if (terms) {
    try {
      std::unique_ptr<rocksdb::Iterator> term_iter(
          context.db_->NewIterator(options, terms));
      for (auto &prefix : cache->prefixes) {
        prefix.suffix_checkpoints = build_suffix_checkpoints(
            *term_iter, cache->buckets[prefix.end_bucket - 1]);
      }
    } catch (const std::bad_alloc &) {
      // The prefix cache is usable without the optional suffix checkpoints.
    }
  }
  return cache;
}

}  // namespace zvec
