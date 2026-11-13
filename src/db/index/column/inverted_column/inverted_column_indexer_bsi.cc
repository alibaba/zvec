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


#include <memory>
#include <ailego/pattern/defer.h>
#include <rocksdb/write_batch.h>
#include "inverted_column_indexer.h"

namespace zvec {

void InvertedColumnIndexer::record_bsi_id(uint32_t id) {
  update_max_id(id);
  if (id == bsi_doc_count_) {
    ++bsi_doc_count_;
    while (!bsi_pending_ids_.empty() &&
           *bsi_pending_ids_.begin() == bsi_doc_count_) {
      bsi_pending_ids_.erase(bsi_pending_ids_.begin());
      ++bsi_doc_count_;
    }
  } else if (id > bsi_doc_count_) {
    bsi_pending_ids_.insert(id);
  }
}

Status InvertedColumnIndexer::restore_bsi_state() {
  std::string value;
  auto status = ctx_.db_->Get(ctx_.read_opts_, key_bsi_doc_count(), &value);
  if (status.ok()) {
    try {
      size_t parsed = 0;
      bsi_doc_count_ = std::stoull(value, &parsed);
      if (parsed != value.size() || bsi_doc_count_ > (uint64_t{1} << 32)) {
        return Status::InternalError("Invalid BSI document count");
      }
    } catch (const std::exception &) {
      return Status::InternalError("Invalid BSI document count");
    }
  } else if (!status.IsNotFound()) {
    return Status::InternalError(status.ToString());
  }
  if (bsi_doc_count_ > max_id_) {
    return Status::OK();
  }

  // Only incomplete prefixes need reconstruction. This also handles an empty
  // index, whose legacy max_id is zero, and unflushed metadata after replay.
  std::unique_ptr<rocksdb::Iterator> iter(
      ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_));
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    roaring_bitmap_t *raw = nullptr;
    auto decoded = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                   iter->value().size(), &raw);
    if (!decoded.ok()) {
      return decoded;
    }
    roaring::Roaring ids(raw);
    ids.removeRange(0, bsi_doc_count_);
    for (uint32_t id : ids) {
      record_bsi_id(id);
    }
  }
  if (!iter->status().ok()) {
    return Status::InternalError(iter->status().ToString());
  }
  roaring::Roaring nulls(null_bitmap_.copy());
  nulls.removeRange(0, bsi_doc_count_);
  for (uint32_t id : nulls) {
    record_bsi_id(id);
  }
  return Status::OK();
}

Status InvertedColumnIndexer::insert_bsi(uint32_t id,
                                         const std::string *value) {
  std::unique_lock lock(bsi_mutex_);
  if (read_only_) {
    return Status::PermissionDenied();
  }
  if (value && value->size() != field_.element_data_size()) {
    return Status::InvalidArgument("Invalid BSI value size");
  }
  std::string term = value ? encode(*value) : std::string{};
  // Like term postings, replay assumes the same value for the same docID.
  // Always merge both terms and slices: either CF can survive a partial flush
  // independently of the persisted document count.
  if (!value) {
    null_bitmap_.add(id);
    record_bsi_id(id);
    return Status::OK();
  }

  std::string operand(1, char{1});
  operand.append(reinterpret_cast<const char *>(&id), sizeof(id));
  rocksdb::WriteBatch batch;
  auto status = batch.Merge(cf_terms_, term, operand);
  if (!status.ok()) {
    return Status::InternalError(status.ToString());
  }
  // Bit zero is the most significant bit of the order-preserving encoding.
  for (size_t bit = 0; bit < term.size() * 8; ++bit) {
    auto byte = static_cast<unsigned char>(term[bit / 8]);
    if ((byte & (0x80u >> (bit % 8))) != 0) {
      status =
          batch.Merge(cf_bsi_, std::string(1, static_cast<char>(bit)), operand);
      if (!status.ok()) {
        return Status::InternalError(status.ToString());
      }
    }
  }
  status = ctx_.db_->Write(ctx_.write_opts_, &batch);
  if (!status.ok()) {
    return Status::InternalError(status.ToString());
  }
  record_bsi_id(id);
  return Status::OK();
}

roaring_bitmap_t *InvertedColumnIndexer::get_bsi_non_null() const {
  auto *bitmap = null_bitmap_.copy();
  roaring_bitmap_remove_range(bitmap, bsi_doc_count_, uint64_t{1} << 32);
  roaring_bitmap_flip_inplace(bitmap, 0, bsi_doc_count_);
  return bitmap;
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_bsi(
    const std::string &term, CompareOp op) const {
  roaring::Roaring valid;
  rocksdb::ReadOptions options = ctx_.read_opts_;
  {
    std::shared_lock lock(bsi_mutex_);
    valid = roaring::Roaring(get_bsi_non_null());
    options.snapshot = ctx_.db_->GetSnapshot();
  }
  if (!options.snapshot) {
    return tl::make_unexpected(Status::InternalError("Cannot snapshot BSI"));
  }
  AILEGO_DEFER([&]() { ctx_.db_->ReleaseSnapshot(options.snapshot); });

  const size_t width = term.size() * 8;
  std::vector<std::string> keys;
  keys.reserve(width);
  for (size_t bit = 0; bit < width; ++bit) {
    keys.emplace_back(1, static_cast<char>(bit));
  }
  std::vector<Slice> slices(keys.begin(), keys.end());
  std::vector<PinnableSlice> values(width);
  std::vector<rocksdb::Status> statuses(width);
  ctx_.db_->MultiGet(options, cf_bsi_, width, slices.data(), values.data(),
                     statuses.data(), true);

  roaring::Roaring equal = valid;
  roaring::Roaring less;
  for (size_t bit = 0; bit < width; ++bit) {
    roaring::Roaring ones;
    if (statuses[bit].ok()) {
      roaring_bitmap_t *raw = nullptr;
      auto status = InvertedIndexCodec::Deserialize(values[bit].data(),
                                                    values[bit].size(), &raw);
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
      ones = roaring::Roaring(raw);
    } else if (!statuses[bit].IsNotFound()) {
      return tl::make_unexpected(
          Status::InternalError(statuses[bit].ToString()));
    }
    auto byte = static_cast<unsigned char>(term[bit / 8]);
    if ((byte & (0x80u >> (bit % 8))) != 0) {
      less |= equal - ones;
      equal &= ones;
    } else {
      equal -= ones;
    }
  }
  switch (op) {
    case CompareOp::LT:
      break;
    case CompareOp::LE:
      less |= equal;
      break;
    case CompareOp::GT:
      less |= equal;
      less = valid - less;
      break;
    case CompareOp::GE:
      less = valid - less;
      break;
    default:
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid BSI operator"));
  }
  return roaring_bitmap_copy(&less.roaring);
}

}  // namespace zvec
