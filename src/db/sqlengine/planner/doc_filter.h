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
#include <arrow/acero/api.h>
#include <arrow/chunked_array.h>
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
            std::unique_ptr<arrow::acero::Declaration> forward_plan,
            std::unique_ptr<arrow::compute::Expression> forward_filter)
      : segment_(std::move(segment)),
        query_info_(std::move(query_info)),
        delete_filter_(segment_->get_filter()),
        invert_cond_(query_info_->invert_cond()),
        forward_plan_(std::move(forward_plan)),
        forward_filter_expr_(std::move(forward_filter)) {}

  Status compute_filter();

  bool is_filtered(uint64_t id) const override;

  //! get brute force by keys and clear `invert_filter_` if suitable
  std::optional<std::vector<uint64_t>> get_bf_by_keys_and_update();

  bool empty() const;

 private:
  std::optional<bool> get_forward_bit(uint64_t id) const;
  std::optional<bool> is_matched_by_forward_filter(uint64_t id) const;

 private:
  Segment::Ptr segment_;
  QueryInfo::Ptr query_info_;
  IndexFilter::Ptr delete_filter_;
  QueryNode::Ptr invert_cond_;
  // either forward_plan_ or forward_expr_ is set
  std::unique_ptr<arrow::acero::Declaration> forward_plan_;
  std::unique_ptr<arrow::compute::Expression> forward_filter_expr_;

  InvertedSearchResult::Ptr invert_result_;
  IndexFilter::Ptr invert_filter_{nullptr};

  std::shared_ptr<arrow::ChunkedArray> forward_bitmap_;
};

}  // namespace zvec::sqlengine