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

#include <rocksdb/merge_operator.h>

namespace zvec::fts {

/*! FTS postings CF-specific Merge Operator
 *  Performs OR merge on Roaring Bitmap serialized values, used for
 * incrementally updating term document lists
 */
class FtsPostingsMerge : public ROCKSDB_NAMESPACE::MergeOperator {
 public:
  bool FullMergeV2(const MergeOperationInput &merge_in,
                   MergeOperationOutput *merge_out) const override;

  bool PartialMerge(const rocksdb::Slice &key,
                    const rocksdb::Slice &left_operand,
                    const rocksdb::Slice &right_operand, std::string *new_value,
                    rocksdb::Logger *logger) const override;

  const char *Name() const override {
    return "FtsPostingsMerge";
  }
};

/*! FTS $MAX_TF CF-specific Merge Operator
 *  Performs max merge on uint32_t values, used for maintaining the maximum term
 * frequency for each term (WAND upper bound)
 */
class FtsMaxTfMerge : public ROCKSDB_NAMESPACE::MergeOperator {
 public:
  bool FullMergeV2(const MergeOperationInput &merge_in,
                   MergeOperationOutput *merge_out) const override;

  bool PartialMerge(const rocksdb::Slice &key,
                    const rocksdb::Slice &left_operand,
                    const rocksdb::Slice &right_operand, std::string *new_value,
                    rocksdb::Logger *logger) const override;

  const char *Name() const override {
    return "FtsMaxTfMerge";
  }
};

}  // namespace zvec::fts
