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
#include <memory>
#include <ailego/pattern/defer.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/write_batch.h>
#include "inverted_column_indexer.h"

namespace zvec {

namespace {

// Include a conservative allowance for container headers and pointer arrays.
// Frozen payloads belong to the backing string and must not be counted twice.
size_t BitmapMemoryBytes(const roaring_bitmap_t *bitmap) {
  if (!bitmap) return 0;
  size_t bytes =
      sizeof(*bitmap) + size_t(bitmap->high_low_container.allocation_size) * 64;
  if (!(bitmap->high_low_container.flags & ROARING_FLAG_FROZEN)) {
    roaring_statistics_t stats;
    roaring_bitmap_statistics(bitmap, &stats);
    bytes += uint64_t(stats.n_bytes_array_containers) +
             stats.n_bytes_run_containers + stats.n_bytes_bitset_containers;
  }
  return bytes;
}

struct CachedBsiSlice {
  std::string buffer;
  roaring_bitmap_t *bitmap{nullptr};
  ~CachedBsiSlice() {
    if (bitmap) roaring_bitmap_free(bitmap);
  }
  size_t charge() const {
    return sizeof(*this) + buffer.capacity() + BitmapMemoryBytes(bitmap);
  }
};

const rocksdb::Cache::CacheItemHelper kBsiCacheHelper{
    rocksdb::CacheEntryRole::kMisc,
    [](void *value, rocksdb::MemoryAllocator *) {
      delete static_cast<CachedBsiSlice *>(value);
    }};

}  // namespace

Status InvertedColumnIndexer::restore_bsi_state() {
  auto state = std::make_unique<BsiState>();
  // Only sealed indexes can read immutable slices without a snapshot.
  const auto params =
      std::static_pointer_cast<InvertIndexParams>(field_.index_params());
  const auto cache_mode =
      sealed_ ? params->bsi_cache_mode() : BsiCacheMode::RESIDENT;
  auto status = load_bsi_state(state.get(), read_only_, cache_mode);
  if (!status.ok()) return status;
  bsi_ = std::move(state);
  return Status::OK();
}

Status InvertedColumnIndexer::load_bsi_state(BsiState *state, bool frozen,
                                             BsiCacheMode cache_mode) {
  const size_t width = field_.element_data_size() * 8;
  if (cache_mode != BsiCacheMode::RESIDENT) {
    if (cache_mode == BsiCacheMode::LRU) {
      const auto params =
          std::static_pointer_cast<InvertIndexParams>(field_.index_params());
      // A single shard avoids partitioning the budget across just 32/64 keys.
      state->cache = rocksdb::NewLRUCache(params->bsi_cache_bytes(), 0, true);
    }
    return Status::OK();
  }
  state->buffers.resize(width);
  state->slices.resize(width);
  for (size_t bit = 0; bit < width; ++bit) {
    auto &buffer = state->buffers[bit];
    auto status =
        ctx_.db_->Get(ctx_.read_opts_, cf_bsi_,
                      std::string(1, static_cast<char>(bit)), &buffer);
    roaring_bitmap_t *bitmap = nullptr;
    if (status.ok()) {
      auto decoded = frozen ? InvertedIndexCodec::DeserializeFrozen(
                                  buffer.data(), buffer.size(), &bitmap)
                            : InvertedIndexCodec::Deserialize(
                                  buffer.data(), buffer.size(), &bitmap);
      if (!decoded.ok()) return decoded;
    } else if (status.IsNotFound()) {
      bitmap = roaring_bitmap_create();
    } else {
      return Status::InternalError(status.ToString());
    }
    if (!bitmap) return Status::InternalError("Cannot allocate BSI slice");
    state->slices[bit].reset(bitmap);
    if (!frozen) std::string{}.swap(buffer);
  }
  return Status::OK();
}

void InvertedColumnIndexer::update_bsi(uint32_t id, const std::string &term) {
  // Lock only the mutable slices; the caller publishes the column bound after
  // this returns. The owner serializes inserts against flush, seal and drop.
  std::unique_lock lock(bsi_->mutex);
  // Like term postings, replay assumes the same value for the same docID.
  // Always update slices even if a prior term write survived a partial flush.
  // Bit zero is the most significant bit of the order-preserving encoding.
  for (size_t bit = 0; bit < term.size() * 8; ++bit) {
    const auto byte = static_cast<unsigned char>(term[bit / 8]);
    if ((byte & (0x80u >> (bit % 8))) != 0) {
      if (roaring_bitmap_add_checked(bsi_->slices[bit].get(), id)) {
        bsi_->dirty_slices |= uint64_t{1} << bit;
      }
    }
  }
}

Status InvertedColumnIndexer::append_bsi_snapshot(rocksdb::WriteBatch *batch) {
  for (size_t bit = 0; bit < bsi_->slices.size(); ++bit) {
    if (!(bsi_->dirty_slices & (uint64_t{1} << bit))) continue;
    const std::string key(1, static_cast<char>(bit));
    std::string value;
    auto encoded =
        InvertedIndexCodec::Serialize(bsi_->slices[bit].get(), &value);
    if (!encoded.ok()) return encoded;
    auto status = batch->Put(cf_bsi_, key, value);
    if (!status.ok()) return Status::InternalError(status.ToString());
  }
  return Status::OK();
}

Status InvertedColumnIndexer::freeze_bsi_state() {
  const auto params =
      std::static_pointer_cast<InvertIndexParams>(field_.index_params());
  // Prepare everything before publishing sealed state. A failed read or
  // allocation leaves the existing mutable state available for retry.
  BsiState state;
  auto status = load_bsi_state(&state, true, params->bsi_cache_mode());
  if (!status.ok()) return status;
  auto rs = ctx_.db_->Put(ctx_.write_opts_, key_sealed(), "sealed");
  if (!rs.ok()) return Status::InternalError(rs.ToString());
  bsi_->slices.swap(state.slices);
  bsi_->buffers.swap(state.buffers);
  bsi_->cache = std::move(state.cache);
  read_only_ = true;
  sealed_ = true;
  return Status::OK();
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_bsi(
    const std::string &term, CompareOp op) const {
  // Keep the state object stable across seal, including readers that entered
  // while the column was still writable. Query results own their payloads.
  std::shared_lock lock(bsi_->mutex);
  // Read the bound before NULL state. NULL inserts publish their bitmap first
  // and do not take the BSI lock. Use this same bound for universe and flip.
  const uint64_t doc_id_limit = doc_id_limit_.load();

  // Compute GT or GE from the least significant bit. A null accumulator
  // represents the initial empty (GT) or universal (GE) set without allocating
  // it. Leading identity operations need neither a slice read nor a bitmap op.
  const bool inclusive = op == CompareOp::GE || op == CompareOp::LT;
  const bool complement = op == CompareOp::LT || op == CompareOp::LE;

  // A nonresident query pins only the slices reached by the same recurrence.
  // All-zero GE and all-one GT bounds need no slice reads at all.
  // Declare views after values so frozen headers die before their backing data.
  std::vector<PinnableSlice> values;
  std::vector<BitmapPtr> views;
  std::vector<std::unique_ptr<CachedBsiSlice>> uncached;
  std::vector<rocksdb::Cache::Handle *> handles;
  AILEGO_DEFER([&]() {
    for (auto *handle : handles) {
      if (handle) bsi_->cache->Release(handle);
    }
  });
  std::vector<const roaring_bitmap_t *> selected;
  const bool nonresident = bsi_->slices.empty();
  if (nonresident) {
    size_t needed = term.size() * 8;
    while (needed != 0) {
      const size_t bit = needed - 1;
      const auto byte = static_cast<unsigned char>(term[bit / 8]);
      const bool one = (byte & (0x80u >> (bit % 8))) != 0;
      if (one == inclusive) break;
      --needed;
    }
    if (needed != 0) {
      selected.resize(needed);
      handles.resize(needed);
      std::vector<std::string> keys;
      std::vector<size_t> missing;
      keys.reserve(needed);
      for (size_t bit = 0; bit < needed; ++bit) {
        const std::string key(1, static_cast<char>(bit));
        if (bsi_->cache) handles[bit] = bsi_->cache->Lookup(key);
        if (handles[bit]) {
          selected[bit] =
              static_cast<CachedBsiSlice *>(bsi_->cache->Value(handles[bit]))
                  ->bitmap;
        } else {
          keys.push_back(key);
          missing.push_back(bit);
        }
      }
      std::vector<Slice> slices(keys.begin(), keys.end());
      std::vector<rocksdb::Status> statuses(keys.size());
      values.resize(keys.size());
      views.resize(keys.size());
      if (!keys.empty()) {
        ctx_.db_->MultiGet(ctx_.read_opts_, cf_bsi_, keys.size(), slices.data(),
                           values.data(), statuses.data(), true);
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        const auto bit = missing[i];
        if (!statuses[i].ok() && !statuses[i].IsNotFound()) {
          return tl::make_unexpected(
              Status::InternalError(statuses[i].ToString()));
        }
        if (bsi_->cache) {
          auto entry = std::make_unique<CachedBsiSlice>();
          if (statuses[i].ok()) {
            entry->buffer.assign(values[i].data(), values[i].size());
            auto status = InvertedIndexCodec::DeserializeFrozen(
                entry->buffer.data(), entry->buffer.size(), &entry->bitmap);
            if (!status.ok()) return tl::make_unexpected(status);
          }
          selected[bit] = entry->bitmap;
          auto status =
              bsi_->cache->Insert(keys[i], entry.get(), &kBsiCacheHelper,
                                  entry->charge(), &handles[bit]);
          if (status.ok()) {
            entry.release();
          } else {
            // Oversized slices or a fully pinned cache still work without
            // exceeding its capacity. The query owns these temporary entries.
            uncached.push_back(std::move(entry));
          }
          values[i].Reset();
        } else if (statuses[i].ok()) {
          roaring_bitmap_t *bitmap = nullptr;
          auto status = InvertedIndexCodec::DeserializeFrozen(
              values[i].data(), values[i].size(), &bitmap);
          if (!status.ok()) return tl::make_unexpected(status);
          views[i].reset(bitmap);
          selected[bit] = bitmap;
        }
      }
    }
  }
  roaring_bitmap_t empty;
  roaring_bitmap_init_cleared(&empty);
  BitmapPtr result;
  for (size_t bit = term.size() * 8; bit-- > 0;) {
    const auto byte = static_cast<unsigned char>(term[bit / 8]);
    const bool one = (byte & (0x80u >> (bit % 8))) != 0;
    if (!result && one != inclusive) continue;
    const auto *slice = nonresident ? selected[bit] : bsi_->slices[bit].get();
    if (!slice) slice = &empty;
    if (!result) {
      result.reset(roaring_bitmap_copy(slice));
      if (!result)
        return tl::make_unexpected(
            Status::InternalError("Cannot allocate BSI result"));
    } else if (one) {
      roaring_bitmap_and_inplace(result.get(), slice);
    } else {
      roaring_bitmap_or_inplace(result.get(), slice);
    }
  }
  if (!result) {
    result.reset(roaring_bitmap_create());
    if (!result)
      return tl::make_unexpected(
          Status::InternalError("Cannot allocate BSI result"));
    if (inclusive) roaring_bitmap_add_range(result.get(), 0, doc_id_limit);
  }
  if (complement) {
    roaring_bitmap_flip_inplace(result.get(), 0, doc_id_limit);
  }
  // NULL has no value slices, but complements and the unsigned minimum bound
  // can include it. Keep NULL exclusion independent of hole tolerance.
  if (null_bitmap_.cardinality() != 0) {
    BitmapPtr nulls(null_bitmap_.copy());
    if (!nulls)
      return tl::make_unexpected(
          Status::InternalError("Cannot copy NULL bitmap"));
    roaring_bitmap_andnot_inplace(result.get(), nulls.get());
  }
  return result.release();
}

InvertedSearchResult::Ptr InvertedColumnIndexer::search_filter(
    const std::string &value, CompareOp op) const {
  if (!supports_bsi(op)) {
    return search(value, op);
  }
  if (value.size() != field_.element_data_size()) return nullptr;
  try {
    auto bitmap = get_bitmap_bsi(encode(value), op);
    if (!bitmap) {
      LOG_ERROR("BSI filter failed for %s: %s", id().c_str(),
                bitmap.error().message().c_str());
      return nullptr;
    }
    BitmapPtr owned(bitmap.value());
    auto result = std::make_shared<InvertedSearchResult>(owned.get());
    owned.release();
    return result;
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Cannot allocate BSI query state for %s", id().c_str());
    return nullptr;
  }
}

}  // namespace zvec
