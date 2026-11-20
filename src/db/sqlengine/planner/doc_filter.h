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
#include <arrow/chunked_array.h>
#include <arrow/compute/expression.h>
#include <zvec/db/status.h>
#include "db/index/column/inverted_column/inverted_search_result.h"
#include "db/index/common/index_filter.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/analyzer/query_info.h"
#include "db/sqlengine/analyzer/query_node.h"

namespace zvec::sqlengine {

class DocFilter : public IndexFilter {
 public:
  using Ptr = std::shared_ptr<DocFilter>;

  DocFilter(Segment::Ptr segment, QueryInfo::Ptr query_info,
            std::unique_ptr<arrow::compute::Expression> bitmap_filter,
            std::unique_ptr<arrow::compute::Expression> forward_filter)
      : segment_(std::move(segment)),
        query_info_(std::move(query_info)),
        delete_filter_(segment_->get_filter()),
        invert_cond_(query_info_->invert_cond()),
        bitmap_filter_(std::move(bitmap_filter)),
        forward_filter_expr_(std::move(forward_filter)) {}

  Status compute_filter();

  bool is_filtered(uint64_t id) const override;

  //! When invert cardinality <= \p ratio * doc_count, extract the ids and
  //! clear invert_filter_ so the caller drives evaluation by ids instead of
  //! bitmap-checking. Ratio is per-caller (vector vs FTS use different
  //! GlobalConfig knobs) because per-candidate cost differs.
  std::optional<std::vector<uint64_t>> get_bf_by_keys_and_update(float ratio);

  bool empty() const;

 private:
  friend class zvec::test::ForwardFilterReadTest;

  arrow::Result<std::shared_ptr<arrow::ChunkedArray>> build_forward_bitmap() const;
  std::optional<bool> get_forward_bit(uint64_t id) const;
  std::optional<bool> is_matched_by_forward_filter(uint64_t id) const;

 private:
  Segment::Ptr segment_;
  QueryInfo::Ptr query_info_;
  IndexFilter::Ptr delete_filter_;
  QueryNode::Ptr invert_cond_;
  // The planner selects either a full bitmap or per-candidate evaluation.
  std::unique_ptr<arrow::compute::Expression> bitmap_filter_;
  std::unique_ptr<arrow::compute::Expression> forward_filter_expr_;

  InvertedSearchResult::Ptr invert_result_;
  IndexFilter::Ptr invert_filter_{nullptr};

  std::shared_ptr<arrow::ChunkedArray> forward_bitmap_;
  bool forward_prepared_{false};
  // Rows captured at preparation. The access decision stays fixed even if the
  // segment is sealed later; an owning view rejects candidates beyond it.
  uint64_t forward_rows_{0};
  ForwardReadView::Ptr forward_view_;
  // Sealed IPC replacement for delete_filter_: same results without a
  // per-candidate segment lock. Null keeps delete_filter_.
  IndexFilter::Ptr sealed_deletion_;
};

}  // namespace zvec::sqlengine
