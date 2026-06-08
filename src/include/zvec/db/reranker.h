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

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/type.h>
#include "zvec/db/status.h"

namespace zvec {

//! Reranker abstract base class for re-ranking search results
class Reranker {
 public:
  using Ptr = std::shared_ptr<Reranker>;

  Reranker() = default;
  virtual ~Reranker() = default;

  virtual void bind_schema(CollectionSchema::Ptr /*schema*/,
                           const std::vector<std::string> & /*field_names*/) {}

  //! Re-rank documents from one or more vector queries.
  //! \param query_results Per-query lists of retrieved documents (sorted by
  //!   relevance), in the same order as the sub-queries supplied by the caller.
  //! \param topn Maximum number of documents to return.
  //! \return Re-ranked list of documents (length <= topn), with updated scores.
  virtual Result<DocPtrList> rerank(
      const std::vector<DocPtrList> &query_results, int topn = 10) const = 0;
};

//! Intermediate base for rerankers that compute per-document scores.
//!
//! Implements the common rerank() logic: iterate docs, call rescore() for each,
//! accumulate scores by doc_id, and return topn results in descending order.
//! Subclasses only need to implement rescore().
class ScoreBasedReranker : public Reranker {
 public:
  Result<DocPtrList> rerank(const std::vector<DocPtrList> &query_results,
                            int topn = 10) const override;

 private:
  //! Compute the contribution score for a single document.
  //! \param score The document's raw relevance score from the vector query.
  //! \param rank The document's position (0-based) in the per-query result
  //! list. \param query_index The index (0-based) of the sub-query this result
  //! came from. \return The score contribution to be accumulated for this
  //! document.
  virtual Result<double> rescore(double score, int rank,
                                 int query_index) const = 0;
};

//! Re-ranker using Reciprocal Rank Fusion (RRF) for multi-vector search.
//!
//! RRF combines results from multiple vector queries without requiring
//! relevance scores. The RRF score for a document at rank r is:
//!   score = 1 / (k + r + 1)
//! where k is the rank constant.
class RrfReranker : public ScoreBasedReranker {
 public:
  explicit RrfReranker(int rank_constant = 60)
      : rank_constant_(rank_constant) {}

  int rank_constant() const {
    return rank_constant_;
  }

 private:
  Result<double> rescore(double score, int rank,
                         int query_index) const override;

  int rank_constant_;
};

//! Re-ranker that combines scores from multiple vector fields using weights.
//!
//! Each vector field's relevance score is normalized based on its own metric
//! type, then scaled by a user-provided weight. Final scores are summed across
//! fields. Supported metrics: L2, IP, COSINE.
//!
//! @note NOT thread-safe. The bind_schema() and rerank() calls share mutable
//! state. Each concurrent query must use its own WeightedReranker instance or
//! serialize access externally.
class WeightedReranker : public ScoreBasedReranker {
 public:
  explicit WeightedReranker(const std::vector<double> &weights = {});

  void bind_schema(CollectionSchema::Ptr schema,
                   const std::vector<std::string> &field_names) override;

  const std::vector<double> &weights() const {
    return weights_;
  }

 private:
  Result<double> rescore(double score, int rank,
                         int query_index) const override;

  static Result<double> normalize_score(double score, const FieldSchema &field);

  CollectionSchema::Ptr schema_;
  std::vector<std::string> field_names_;
  std::vector<double> weights_;
};

//! Callback-based re-ranker for cross-language bridging.
//!
//! Wraps a user-provided callback (e.g., a Python callable) as a Reranker.
//! When the callback is a Python function, GIL must be managed by the caller.
class CallbackReranker : public Reranker {
 public:
  using Callback =
      std::function<DocPtrList(const std::vector<DocPtrList> &, int)>;

  explicit CallbackReranker(Callback fn) : callback_(std::move(fn)) {}

  Result<DocPtrList> rerank(const std::vector<DocPtrList> &query_results,
                            int topn = 10) const override {
    if (!callback_) {
      return tl::make_unexpected(
          Status::InvalidArgument("CallbackReranker: callback is empty"));
    }
    return callback_(query_results, topn);
  }

 private:
  Callback callback_;
};

}  // namespace zvec
