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

#include "fts_rocksdb_merge.h"
#include <cstring>
#include <roaring/roaring.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/column/fts_column/posting/bitpacked_posting_list.h"

namespace zvec::fts {

// ============================================================
// Helper: deserialize a posting value (Roaring Bitmap or BitPacked) into a
// Roaring Bitmap. Caller owns the returned bitmap and must free it.
// Returns nullptr on failure.
// ============================================================

static roaring_bitmap_t *deserialize_posting_to_roaring(const char *data,
                                                        size_t size) {
  if (BitPackedPostingList::is_bitpacked_format(data, size)) {
    // Decode BitPacked format into a new Roaring Bitmap
    BitPackedPostingIterator bp_iter;
    if (bp_iter.open(data, size) != 0) {
      LOG_ERROR(
          "FtsPostingsMerge: failed to open bitpacked posting during merge, "
          "size[%zu]",
          size);
      return nullptr;
    }
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    uint32_t doc_id = bp_iter.next_doc();
    while (doc_id != BitPackedPostingIterator::NO_MORE_DOCS) {
      roaring_bitmap_add(bitmap, doc_id);
      doc_id = bp_iter.next_doc();
    }
    return bitmap;
  }

  // Roaring Bitmap format
  return roaring_bitmap_portable_deserialize_safe(data, size);
}

// ============================================================
// FtsPostingsMerge: Roaring Bitmap OR merge (supports BitPacked input)
// ============================================================

bool FtsPostingsMerge::FullMergeV2(const MergeOperationInput &merge_in,
                                   MergeOperationOutput *merge_out) const {
  // If there is only one operand and no existing_value, return directly
  if (merge_in.existing_value == nullptr && merge_in.operand_list.size() == 1) {
    merge_out->new_value = std::string(merge_in.operand_list[0].data(),
                                       merge_in.operand_list[0].size());
    return true;
  }

  // Deserialize bitmap from existing_value
  roaring_bitmap_t *result_bitmap = roaring_bitmap_create();

  if (merge_in.existing_value != nullptr) {
    roaring_bitmap_t *existing_bitmap = deserialize_posting_to_roaring(
        merge_in.existing_value->data(), merge_in.existing_value->size());
    if (existing_bitmap != nullptr) {
      roaring_bitmap_or_inplace(result_bitmap, existing_bitmap);
      roaring_bitmap_free(existing_bitmap);
    }
  }

  // Merge all operands
  for (const auto &operand : merge_in.operand_list) {
    roaring_bitmap_t *operand_bitmap =
        deserialize_posting_to_roaring(operand.data(), operand.size());
    if (operand_bitmap != nullptr) {
      roaring_bitmap_or_inplace(result_bitmap, operand_bitmap);
      roaring_bitmap_free(operand_bitmap);
    }
  }

  // Serialize result as Roaring Bitmap
  roaring_bitmap_run_optimize(result_bitmap);
  size_t serialized_size = roaring_bitmap_portable_size_in_bytes(result_bitmap);
  merge_out->new_value.resize(serialized_size);
  roaring_bitmap_portable_serialize(result_bitmap, merge_out->new_value.data());
  roaring_bitmap_free(result_bitmap);
  return true;
}

bool FtsPostingsMerge::PartialMerge(const rocksdb::Slice & /*key*/,
                                    const rocksdb::Slice &left_operand,
                                    const rocksdb::Slice &right_operand,
                                    std::string *new_value,
                                    rocksdb::Logger * /*logger*/) const {
  roaring_bitmap_t *left_bitmap =
      deserialize_posting_to_roaring(left_operand.data(), left_operand.size());
  roaring_bitmap_t *right_bitmap = deserialize_posting_to_roaring(
      right_operand.data(), right_operand.size());

  if (left_bitmap == nullptr || right_bitmap == nullptr) {
    LOG_ERROR(
        "FtsPostingsMerge::PartialMerge: failed to deserialize operand. "
        "left_size[%zu] right_size[%zu]",
        left_operand.size(), right_operand.size());
    if (left_bitmap != nullptr) roaring_bitmap_free(left_bitmap);
    if (right_bitmap != nullptr) roaring_bitmap_free(right_bitmap);
    return false;
  }

  roaring_bitmap_or_inplace(left_bitmap, right_bitmap);
  roaring_bitmap_free(right_bitmap);

  size_t serialized_size = roaring_bitmap_portable_size_in_bytes(left_bitmap);
  new_value->resize(serialized_size);
  roaring_bitmap_portable_serialize(left_bitmap, new_value->data());
  roaring_bitmap_free(left_bitmap);
  return true;
}

// ============================================================
// FtsMaxTfMerge: uint32_t max merge
// ============================================================

bool FtsMaxTfMerge::FullMergeV2(const MergeOperationInput &merge_in,
                                MergeOperationOutput *merge_out) const {
  uint32_t max_tf = 0;

  if (merge_in.existing_value != nullptr &&
      merge_in.existing_value->size() >= sizeof(uint32_t)) {
    std::memcpy(&max_tf, merge_in.existing_value->data(), sizeof(uint32_t));
  }

  for (const auto &operand : merge_in.operand_list) {
    if (operand.size() >= sizeof(uint32_t)) {
      uint32_t operand_tf = 0;
      std::memcpy(&operand_tf, operand.data(), sizeof(uint32_t));
      if (operand_tf > max_tf) {
        max_tf = operand_tf;
      }
    }
  }

  merge_out->new_value.resize(sizeof(uint32_t));
  std::memcpy(merge_out->new_value.data(), &max_tf, sizeof(uint32_t));
  return true;
}

bool FtsMaxTfMerge::PartialMerge(const rocksdb::Slice & /*key*/,
                                 const rocksdb::Slice &left_operand,
                                 const rocksdb::Slice &right_operand,
                                 std::string *new_value,
                                 rocksdb::Logger * /*logger*/) const {
  if (left_operand.size() < sizeof(uint32_t) ||
      right_operand.size() < sizeof(uint32_t)) {
    LOG_ERROR(
        "FtsMaxTfMerge::PartialMerge: operand too small. "
        "left_size[%zu] right_size[%zu] expected[%zu]",
        left_operand.size(), right_operand.size(), sizeof(uint32_t));
    return false;
  }

  uint32_t left_tf = 0;
  uint32_t right_tf = 0;
  std::memcpy(&left_tf, left_operand.data(), sizeof(uint32_t));
  std::memcpy(&right_tf, right_operand.data(), sizeof(uint32_t));

  uint32_t max_tf = (left_tf > right_tf) ? left_tf : right_tf;
  new_value->resize(sizeof(uint32_t));
  std::memcpy(new_value->data(), &max_tf, sizeof(uint32_t));
  return true;
}

}  // namespace zvec::fts
