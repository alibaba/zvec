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


#include <ailego/pattern/defer.h>
#include <rocksdb/merge_operator.h>
#include "inverted_codec.h"


namespace zvec {


class InvertedRocksdbValueMerger : public rocksdb::MergeOperator {
 public:
  bool FullMergeV2(const MergeOperationInput &merge_in,
                   MergeOperationOutput *merge_out) const override {
    if (merge_in.existing_value == nullptr &&
        merge_in.operand_list.size() == 1) {
      merge_out->new_value = std::string(merge_in.operand_list[0].data(),
                                         merge_in.operand_list[0].size());
      return true;
    }

    merge_out->new_value.clear();

    Status s;
    roaring_bitmap_t *bitmap{nullptr};
    if (merge_in.existing_value != nullptr) {
      s = InvertedIndexCodec::Deserialize(merge_in.existing_value->data(),
                                          merge_in.existing_value->size(),
                                          &bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to deserialize existing value");
        return false;
      }
    } else {
      bitmap = roaring_bitmap_create();
      if (!bitmap) {
        LOG_ERROR("Failed to create bitmap");
        return false;
      }
    }
    AILEGO_DEFER([&]() { roaring_bitmap_free(bitmap); });

    for (const rocksdb::Slice &m : merge_in.operand_list) {
      s = InvertedIndexCodec::Merge_OR(m.data(), m.size(), true, bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to merge bitmap");
        return false;
      }
    }
    roaring_bitmap_repair_after_lazy(bitmap);

    s = InvertedIndexCodec::Serialize(bitmap, &(merge_out->new_value));
    if (s.ok()) {
      return true;
    } else {
      LOG_ERROR("Failed to serialize bitmap");
      return false;
    }
  }


  bool PartialMerge(const rocksdb::Slice & /*key*/,
                    const rocksdb::Slice &left_operand,
                    const rocksdb::Slice &right_operand, std::string *new_value,
                    rocksdb::Logger * /*logger*/) const override {
    roaring_bitmap_t *bitmap{nullptr};
    auto s = InvertedIndexCodec::Deserialize(left_operand.data(),
                                             left_operand.size(), &bitmap);
    if (!s.ok()) {
      LOG_ERROR("Failed to deserialize existing value");
      return false;
    }
    AILEGO_DEFER([&]() { roaring_bitmap_free(bitmap); });

    s = InvertedIndexCodec::Merge_OR(right_operand.data(), right_operand.size(),
                                     false, bitmap);
    if (!s.ok()) {
      LOG_ERROR("Failed to merge bitmap");
      return false;
    }

    s = InvertedIndexCodec::Serialize(bitmap, new_value);
    if (s.ok()) {
      return true;
    } else {
      LOG_ERROR("Failed to serialize bitmap");
      return false;
    }
  }


  const char *Name() const override {
    return "InvertedRocksdbValueMerger";
  }
};


}  // namespace zvec