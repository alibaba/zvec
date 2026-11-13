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
#include <cstdlib>
#include <memory>
#include <rocksdb/write_batch.h>
#include "inverted_column_indexer.h"

namespace zvec {

Status InvertedColumnIndexer::restore_bsi_state() {
  auto state = std::make_unique<BsiState>();
  const size_t width = field_.element_data_size() * 8;
  std::string value;
  auto status = ctx_.db_->Get(ctx_.read_opts_, key_bsi_doc_count(), &value);
  if (status.ok()) {
    try {
      size_t parsed = 0;
      state->doc_id_limit = std::stoull(value, &parsed);
      if (parsed != value.size() || state->doc_id_limit > (uint64_t{1} << 32)) {
        return Status::InternalError("Invalid BSI document bound");
      }
    } catch (const std::exception &) {
      return Status::InternalError("Invalid BSI document bound");
    }
  } else if (!status.IsNotFound()) {
    return Status::InternalError(status.ToString());
  }
  // Benchmark-only switch, fixed when opening the index. Writing indexes
  // always keep mutable slices; an empty slice vector denotes on-demand reads.
  const auto *nonresident = std::getenv("ZVEC_BSI_BENCH_NONRESIDENT");
  if (sealed_ && read_only_ && nonresident && std::string(nonresident) == "1") {
    bsi_ = std::move(state);
    return Status::OK();
  }
  state->buffers.resize(width);
  state->slices.resize(width);
  for (size_t bit = 0; bit < width; ++bit) {
    auto &buffer = state->buffers[bit];
    status = ctx_.db_->Get(ctx_.read_opts_, cf_bsi_,
                           std::string(1, static_cast<char>(bit)), &buffer);
    roaring_bitmap_t *bitmap = nullptr;
    if (status.ok()) {
      auto decoded = read_only_ ? InvertedIndexCodec::DeserializeFrozen(
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
    if (!read_only_) std::string{}.swap(buffer);
  }
  bsi_ = std::move(state);
  return Status::OK();
}

void InvertedColumnIndexer::update_bsi(uint32_t id, const std::string &term) {
  std::unique_lock lock(bsi_->mutex);
  // Like term postings, replay assumes the same value for the same docID.
  // Always update slices even if a prior term write survived a partial flush.
  // Bit zero is the most significant bit of the order-preserving encoding.
  for (size_t bit = 0; bit < term.size() * 8; ++bit) {
    const auto byte = static_cast<unsigned char>(term[bit / 8]);
    if ((byte & (0x80u >> (bit % 8))) != 0) {
      roaring_bitmap_add(bsi_->slices[bit].get(), id);
    }
  }
  bsi_->doc_id_limit = std::max(bsi_->doc_id_limit, uint64_t{id} + 1);
  bsi_->dirty = true;
}

Status InvertedColumnIndexer::flush_bsi() {
  std::unique_lock lock(bsi_->mutex);
  if (!bsi_->dirty) return Status::OK();
  rocksdb::WriteBatch batch;
  for (size_t bit = 0; bit < bsi_->slices.size(); ++bit) {
    const std::string key(1, static_cast<char>(bit));
    if (roaring_bitmap_is_empty(bsi_->slices[bit].get())) {
      auto status = batch.Delete(cf_bsi_, key);
      if (!status.ok()) return Status::InternalError(status.ToString());
      continue;
    }
    std::string value;
    auto encoded =
        InvertedIndexCodec::Serialize(bsi_->slices[bit].get(), &value);
    if (!encoded.ok()) return encoded;
    auto status = batch.Put(cf_bsi_, key, value);
    if (!status.ok()) return Status::InternalError(status.ToString());
  }
  auto status =
      batch.Put(key_bsi_doc_count(), std::to_string(bsi_->doc_id_limit));
  if (!status.ok()) return Status::InternalError(status.ToString());
  status = ctx_.db_->Write(ctx_.write_opts_, &batch);
  if (!status.ok()) return Status::InternalError(status.ToString());
  // The owner still flushes all CFs before the segment commits and clears WAL.
  bsi_->dirty = false;
  return Status::OK();
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_bsi(
    const std::string &term, CompareOp op) const {
  std::shared_lock lock(bsi_->mutex, std::defer_lock);
  if (!read_only_) lock.lock();

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
      std::vector<std::string> keys;
      keys.reserve(needed);
      for (size_t bit = 0; bit < needed; ++bit) {
        keys.emplace_back(1, static_cast<char>(bit));
      }
      std::vector<Slice> slices(keys.begin(), keys.end());
      std::vector<rocksdb::Status> statuses(needed);
      values.resize(needed);
      views.resize(needed);
      ctx_.db_->MultiGet(ctx_.read_opts_, cf_bsi_, needed, slices.data(),
                         values.data(), statuses.data(), true);
      for (size_t bit = 0; bit < needed; ++bit) {
        if (statuses[bit].ok()) {
          roaring_bitmap_t *bitmap = nullptr;
          auto status = InvertedIndexCodec::DeserializeFrozen(
              values[bit].data(), values[bit].size(), &bitmap);
          if (!status.ok()) return tl::make_unexpected(status);
          views[bit].reset(bitmap);
        } else if (!statuses[bit].IsNotFound()) {
          return tl::make_unexpected(
              Status::InternalError(statuses[bit].ToString()));
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
    const auto *slice =
        nonresident ? views[bit].get() : bsi_->slices[bit].get();
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
    if (inclusive)
      roaring_bitmap_add_range(result.get(), 0, bsi_->doc_id_limit);
  }
  if (complement) {
    roaring_bitmap_flip_inplace(result.get(), 0, bsi_->doc_id_limit);
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
  if (!bsi_ || (op != CompareOp::LT && op != CompareOp::LE &&
                op != CompareOp::GT && op != CompareOp::GE)) {
    return search(value, op);
  }
  if (value.size() != field_.element_data_size()) return nullptr;
  auto bitmap = get_bitmap_bsi(encode(value), op);
  if (!bitmap) return nullptr;
  return std::make_shared<InvertedSearchResult>(bitmap.value());
}

}  // namespace zvec
