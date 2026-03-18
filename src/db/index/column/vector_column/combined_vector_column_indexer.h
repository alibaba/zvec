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

#include <memory>
#include <vector>
#include "db/index/common/index_filter.h"
#include "vector_column_indexer.h"
#include "vector_column_params.h"

namespace zvec {

class CombinedVectorColumnIndexer {
 public:
  using Ptr = std::shared_ptr<CombinedVectorColumnIndexer>;

  explicit CombinedVectorColumnIndexer(
      const std::vector<VectorColumnIndexer::Ptr> &indexers,
      const std::vector<VectorColumnIndexer::Ptr> &normal_indexers,
      const FieldSchema &field_schema, const SegmentMeta &segment_meta,
      std::vector<BlockMeta> blocks, MetricType metric_type,
      bool is_quantized = false);

  virtual ~CombinedVectorColumnIndexer() = default;

  virtual Result<IndexResults::Ptr> Search(
      const vector_column_params::VectorData &vector_data,
      const vector_column_params::QueryParams &query_params);

  // doc_id is segment local id
  virtual Result<vector_column_params::VectorDataBuffer> Fetch(
      uint32_t segment_doc_id) const;


 protected:
  /**
   * A filter wrapper that applies an offset to document IDs before
   * delegating to an inner filter.
   *
   * This is used when multiple blocks with different ID offsets are stored.
   * Each block has its own local ID space, and this filter translates
   * block-level IDs to segment-level IDs before checking the inner filter.
   */
  class BlockOffsetFilter : public IndexFilter {
   public:
    BlockOffsetFilter(const IndexFilter *inner_filter, uint64_t offset)
        : inner_filter_(inner_filter), offset_(offset) {}

    bool is_filtered(uint64_t id) const override {
      return inner_filter_->is_filtered(id + offset_);
    }

   private:
    const IndexFilter *inner_filter_;
    uint64_t offset_;
  };

  // for ut
  CombinedVectorColumnIndexer() = default;


 private:
  FieldSchema field_schema_;
  std::vector<VectorColumnIndexer::Ptr> indexers_;
  std::vector<VectorColumnIndexer::Ptr> normal_indexers_;
  std::vector<BlockMeta> blocks_;
  std::vector<uint32_t> block_offsets_;
  MetricType metric_type_{MetricType::UNDEFINED};
  bool is_quantized_{false};
  uint64_t min_doc_id_{0};
};

}  // namespace zvec