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
#include "db/common/constants.h"
#include "db/index/column/common/index_results.h"
#include "db/index/column/fts_column/fts_column_indexer.h"

namespace zvec {

// IndexResults adapter for FTS search results (doc_id + BM25 score pairs).
// Results are ordered by descending score from FtsColumnIndexer::search().
class FtsIndexResults : public IndexResults,
                        public std::enable_shared_from_this<FtsIndexResults> {
 public:
  using Ptr = std::shared_ptr<FtsIndexResults>;

  explicit FtsIndexResults(std::vector<fts::FtsResult> results)
      : results_(std::move(results)) {}

  size_t count() const override {
    return results_.size();
  }

  const std::vector<fts::FtsResult> &results() const {
    return results_;
  }

  class FtsIterator : public Iterator {
   public:
    explicit FtsIterator(std::shared_ptr<const FtsIndexResults> owner)
        : owner_(std::move(owner)), pos_(0) {}

    idx_t doc_id() const override {
      if (pos_ < owner_->results_.size()) {
        return static_cast<idx_t>(owner_->results_[pos_].doc_id);
      }
      return INVALID_DOC_ID;
    }

    float score() const override {
      if (pos_ < owner_->results_.size()) {
        return owner_->results_[pos_].score;
      }
      return 0.0f;
    }

    void next() override {
      if (pos_ < owner_->results_.size()) {
        ++pos_;
      }
    }

    bool valid() const override {
      return pos_ < owner_->results_.size();
    }

   private:
    std::shared_ptr<const FtsIndexResults> owner_;
    size_t pos_;
  };

  IteratorUPtr create_iterator() override {
    return std::make_unique<FtsIterator>(shared_from_this());
  }

 private:
  std::vector<fts::FtsResult> results_;
};

}  // namespace zvec
