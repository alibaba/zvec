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
#include <map>
#include <memory>
#include <string>
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

  virtual void bind_schema(CollectionSchema::Ptr) {}

  //! Re-rank documents from one or more vector queries.
  //! \param query_results Mapping from vector field name to list of retrieved
  //!   documents (sorted by relevance).
  //! \param topn Maximum number of documents to return.
  //! \return Re-ranked list of documents (length <= topn), with updated scores.
  virtual Result<DocPtrList> rerank(
      const std::map<std::string, DocPtrList> &query_results,
      int topn = 10) const = 0;
};

//! Intermediate base for rerankers that compute per-document scores.
//!
//! Implements the common rerank() logic: iterate docs, call rescore() for each,
//! accumulate scores by doc_id, and return topn results in descending order.
//! Subclasses only need to implement rescore().
class ScoreBasedReranker : public Reranker {
 public:
  //! Compute the contribution score for a single document.
  //! \param score The document's raw relevance score from the vector field.
  //! \param rank The document's position (0-based) in the per-field result
  //! list. \param field_name The name of the vector field this result came
  //! from. \return The score contribution to be accumulated for this document.
  virtual Result<double> rescore(double score, int rank,
                                 const std::string &field_name) const = 0;

  Result<DocPtrList> rerank(
      const std::map<std::string, DocPtrList> &query_results,
      int topn = 10) const override;
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

  Result<double> rescore(double score, int rank,
                         const std::string &field_name) const override;

 private:
  int rank_constant_;
};

//! Re-ranker that combines scores from multiple vector fields using weights.
//!
//! Each vector field's relevance score is normalized based on its own metric
//! type, then scaled by a user-provided weight. Final scores are summed across
//! fields. Supported metrics: L2, IP, COSINE.
class WeightedReranker : public ScoreBasedReranker {
 public:
  explicit WeightedReranker(const std::map<std::string, double> &weights = {});

  void bind_schema(CollectionSchema::Ptr schema) override;

  const std::map<std::string, double> &weights() const {
    return weights_;
  }

  Result<double> rescore(double score, int rank,
                         const std::string &field_name) const override;

 private:
  static Result<double> normalize_score(double score, const FieldSchema &field);

  CollectionSchema::Ptr schema_;
  std::map<std::string, double> weights_;
};

//! Callback-based re-ranker for cross-language bridging.
//!
//! Wraps a user-provided callback (e.g., a Python callable) as a Reranker.
//! When the callback is a Python function, GIL must be managed by the caller.
class CallbackReranker : public Reranker {
 public:
  using Callback =
      std::function<DocPtrList(const std::map<std::string, DocPtrList> &, int)>;

  explicit CallbackReranker(Callback fn) : callback_(std::move(fn)) {}

  Result<DocPtrList> rerank(
      const std::map<std::string, DocPtrList> &query_results,
      int topn = 10) const override {
    return callback_(query_results, topn);
  }

 private:
  Callback callback_;
};

}  // namespace zvec
