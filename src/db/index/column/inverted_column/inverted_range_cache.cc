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
#include <string_view>
#include "inverted_codec.h"

namespace zvec {
namespace {

constexpr std::string_view kMetadataVersion = "zvec-range-prefix-v1";
constexpr size_t kMaxBoundaryTermBytes = 128;
constexpr size_t kMaxSuffixCheckpoints = 13;

std::string prefix_key(const std::string &field, size_t index) {
  return field + "$ZVEC$RANGE_PREFIX$" + static_cast<char>('a' + index);
}

template <class T>
void append_number(std::string &out, T number) {
  for (size_t i = 0; i < sizeof(T); ++i) {
    out.push_back(static_cast<char>(number & 0xff));
    number >>= 8;
  }
}

void append_key(std::string &out, const std::string &key) {
  append_number(out, uint64_t(key.size()));
  out.append(key);
}

template <class T>
bool take_number(std::string_view &input, T &number) {
  if (input.size() < sizeof(T)) return false;
  number = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    number |= T(static_cast<unsigned char>(input[i])) << (8 * i);
  }
  input.remove_prefix(sizeof(T));
  return true;
}

bool take_key(std::string_view &input, std::string_view &key) {
  uint64_t size;
  if (!take_number(input, size) || size > input.size()) return false;
  key = input.substr(0, size);
  input.remove_prefix(size);
  return true;
}

bool before(std::string_view a, std::string_view b) {
  return InvertedIndexCodec::CMP(a.data(), a.size(), b.data(), b.size()) < 0;
}

bool decode_bucket(std::string_view key, InvertedRangeCache::Bucket &bucket) {
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
    if (key.size() > kMaxBoundaryTermBytes ||
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
  uint64_t decode_work = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    Bucket bucket;
    if (!decode_bucket(iter->key().ToString(), bucket) ||
        (!cache->buckets.empty() &&
         !before(cache->buckets.back().end, bucket.begin)))
      return nullptr;
    roaring_bitmap_t *raw = nullptr;
    auto status = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                  iter->value().size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) return nullptr;
    const uint64_t cardinality = roaring_bitmap_get_cardinality(bitmap.get());
    if (cardinality > domain - null_count - cache->total_cardinality)
      return nullptr;
    cache->total_cardinality += cardinality;
    bucket.cumulative_cardinality = cache->total_cardinality;
    // Conservative byte-work bound: an unaligned view copies its payload
    // before merging. Keep descriptor setup separate from those data passes.
    decode_work +=
        uint64_t(iter->value().size()) * 2 +
        uint64_t(bitmap->high_low_container.size) * kContainerSetupWork;
    bucket.cumulative_decode_work = decode_work;
    cache->buckets.push_back(std::move(bucket));
  }
  // Complement plans require a complete DocID domain with NULLs excluded.
  if (!iter->status().ok() || cache->buckets.size() < 2 ||
      !cache->total_cardinality ||
      cache->total_cardinality + null_count != domain)
    return nullptr;

  cache->prefixes.reserve(kQuantiles - 1);
  BitmapPtr cumulative(roaring_bitmap_create());
  if (!cumulative) return nullptr;
  size_t bucket_index = 0;
  unsigned quantile = 1;
  const auto cutoff = [&](unsigned q) {
    return (cache->total_cardinality * q + kQuantiles - 1) / kQuantiles;
  };
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
    if (quantile >= kQuantiles || cardinality < cutoff(quantile)) continue;
    do {
      ++quantile;
    } while (quantile < kQuantiles && cardinality >= cutoff(quantile));
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

rocksdb::Status InvertedRangeCache::store(RocksdbContext &context,
                                          const std::string &field,
                                          std::string &metadata) const {
  std::string encoded(kMetadataVersion);
  append_number(encoded, uint64_t(buckets.size()));
  for (const auto &bucket : buckets) {
    append_key(encoded, bucket.key);
    append_number(encoded, bucket.cumulative_cardinality);
    append_number(encoded, bucket.cumulative_decode_work);
  }

  append_number(encoded, uint32_t(prefixes.size()));
  std::string value;
  for (unsigned i = 0; i < prefixes.size(); ++i) {
    const auto &prefix = prefixes[i];
    append_number(encoded, uint64_t(prefix.end_bucket));
    append_number(encoded, uint32_t(prefix.suffix_checkpoints.size()));
    for (const auto &checkpoint : prefix.suffix_checkpoints) {
      append_key(encoded, checkpoint.first_term);
      append_number(encoded, checkpoint.work);
    }
    value.resize(prefix.copy_bytes);
    roaring_bitmap_portable_serialize(prefix.bitmap.get(), value.data());
    auto status =
        context.db_->Put(context.write_opts_, prefix_key(field, i), value);
    if (!status.ok()) return status;
  }
  metadata = std::move(encoded);
  return rocksdb::Status::OK();
}

std::unique_ptr<InvertedRangeCache> InvertedRangeCache::Load(
    RocksdbContext &context, const std::string &field,
    const std::string &metadata, uint32_t max_id,
    const roaring_bitmap_t *null_bitmap) {
  std::string_view input(metadata);
  if (input.substr(0, kMetadataVersion.size()) != kMetadataVersion)
    return nullptr;
  input.remove_prefix(kMetadataVersion.size());
  uint64_t bucket_count;
  if (!take_number(input, bucket_count) || bucket_count < 2 ||
      bucket_count > input.size() / (3 * sizeof(uint64_t)))
    return nullptr;

  const uint64_t domain = uint64_t(max_id) + 1;
  const uint64_t null_count = roaring_bitmap_get_cardinality(null_bitmap);
  if (null_count >= domain ||
      (null_count && roaring_bitmap_maximum(null_bitmap) > max_id))
    return nullptr;
  auto cache = std::make_unique<InvertedRangeCache>();
  cache->buckets.reserve(bucket_count);
  uint64_t previous_cardinality = 0;
  uint64_t previous_work = 0;
  for (uint64_t i = 0; i < bucket_count; ++i) {
    Bucket bucket;
    std::string_view key;
    if (!take_key(input, key) || !decode_bucket(key, bucket) ||
        (!cache->buckets.empty() &&
         !before(cache->buckets.back().end, bucket.begin)) ||
        !take_number(input, bucket.cumulative_cardinality) ||
        !take_number(input, bucket.cumulative_decode_work) ||
        bucket.cumulative_cardinality <= previous_cardinality ||
        bucket.cumulative_cardinality > domain - null_count ||
        bucket.cumulative_decode_work <= previous_work)
      return nullptr;
    previous_cardinality = bucket.cumulative_cardinality;
    previous_work = bucket.cumulative_decode_work;
    cache->buckets.push_back(std::move(bucket));
  }
  if (previous_cardinality != domain - null_count) return nullptr;
  cache->total_cardinality = previous_cardinality;

  uint32_t prefix_count;
  if (!take_number(input, prefix_count) || !prefix_count ||
      prefix_count >= kQuantiles)
    return nullptr;
  cache->prefixes.reserve(prefix_count);
  rocksdb::ReadOptions options = context.read_opts_;
  options.fill_cache = false;
  uint64_t previous_end = 0;
  for (unsigned i = 0; i < prefix_count; ++i) {
    uint64_t end;
    uint32_t checkpoint_count;
    if (!take_number(input, end) || end <= previous_end ||
        end >= bucket_count || !take_number(input, checkpoint_count) ||
        checkpoint_count > kMaxSuffixCheckpoints)
      return nullptr;
    previous_end = end;
    const auto &bucket = cache->buckets[end - 1];
    Prefix prefix;
    prefix.end_bucket = end;
    prefix.suffix_checkpoints.reserve(checkpoint_count);
    for (unsigned j = 0; j < checkpoint_count; ++j) {
      std::string_view term;
      uint64_t work;
      if (!take_key(input, term) || term.size() > kMaxBoundaryTermBytes ||
          !take_number(input, work) || !work || before(term, bucket.begin) ||
          before(bucket.end, term) ||
          (j && (work <= prefix.suffix_checkpoints.back().work ||
                 !before(term, prefix.suffix_checkpoints.back().first_term))))
        return nullptr;
      prefix.suffix_checkpoints.push_back({std::string(term), work});
    }
    rocksdb::PinnableSlice value;
    const auto status =
        context.db_->Get(options, context.db_->DefaultColumnFamily(),
                         prefix_key(field, i), &value);
    if (!status.ok() || !value.size() ||
        roaring_bitmap_portable_deserialize_size(value.data(), value.size()) !=
            value.size())
      return nullptr;
    prefix.bitmap.reset(
        roaring_bitmap_portable_deserialize_safe(value.data(), value.size()));
    const char *validation_error = nullptr;
    if (!prefix.bitmap ||
        !roaring_bitmap_internal_validate(prefix.bitmap.get(),
                                          &validation_error) ||
        roaring_bitmap_get_cardinality(prefix.bitmap.get()) !=
            bucket.cumulative_cardinality ||
        roaring_bitmap_maximum(prefix.bitmap.get()) > max_id)
      return nullptr;
    prefix.copy_bytes = value.size();
    cache->prefixes.push_back(std::move(prefix));
  }
  if (!input.empty()) return nullptr;
  return cache;
}

rocksdb::Status InvertedRangeCache::Delete(rocksdb::WriteBatch &batch,
                                           const std::string &field,
                                           size_t keep) {
  for (size_t i = keep; i < kQuantiles - 1; ++i) {
    auto status = batch.Delete(prefix_key(field, i));
    if (!status.ok()) return status;
  }
  return rocksdb::Status::OK();
}

}  // namespace zvec
