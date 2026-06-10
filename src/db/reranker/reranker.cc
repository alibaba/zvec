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

#include <algorithm>
#include "zvec/db/status.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <queue>
#include <unordered_map>
#include <utility>
#include <variant>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/index_params.h>
#include <zvec/db/reranker.h>

namespace zvec {
namespace {

// Shared score-based rerank logic used by RRF and Weighted.
// score_fn(doc_score, rank, field_index) -> contribution score
using ScoreFn = std::function<Result<double>(double, int, size_t)>;

Result<DocPtrList> score_based_rerank(const ScoreFn &score_fn,
                                      const std::vector<DocPtrList> &results,
                                      int topn) {
  if (topn <= 0) {
    return DocPtrList();
  }

  std::unordered_map<std::string, double> scores;
  std::unordered_map<std::string, Doc::Ptr> id_to_doc;

  for (size_t field_idx = 0; field_idx < results.size(); ++field_idx) {
    const auto &docs = results[field_idx];
    for (size_t rank = 0; rank < docs.size(); ++rank) {
      const auto &doc = docs[rank];
      const std::string &doc_id = doc->pk();
      auto rs = score_fn(static_cast<double>(doc->score()),
                         static_cast<int>(rank), field_idx);
      if (!rs.has_value()) {
        return tl::make_unexpected(rs.error());
      }
      scores[doc_id] += rs.value();
      if (id_to_doc.find(doc_id) == id_to_doc.end()) {
        id_to_doc[doc_id] = doc;
      }
    }
  }

  using ScorePair = std::pair<std::string, double>;
  auto cmp = [](const ScorePair &a, const ScorePair &b) {
    return a.second > b.second;
  };
  std::priority_queue<ScorePair, std::vector<ScorePair>, decltype(cmp)> pq(cmp);

  for (const auto &[doc_id, score] : scores) {
    if (static_cast<int>(pq.size()) < topn) {
      pq.emplace(doc_id, score);
    } else if (score > pq.top().second) {
      pq.pop();
      pq.emplace(doc_id, score);
    }
  }

  DocPtrList result;
  result.reserve(pq.size());
  while (!pq.empty()) {
    const auto &[doc_id, score] = pq.top();
    auto doc = std::move(id_to_doc[doc_id]);
    doc->set_score(static_cast<float>(score));
    result.push_back(std::move(doc));
    pq.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

Result<double> normalize_score(double score, const FieldSchema &field) {
  if (field.index_type() == IndexType::FTS) {
    // Non-vector FTS/BM25 fields: map positive scores to [0, 1).
    return 2.0 * std::atan(score) / M_PI;
  }
  auto *vip =
      dynamic_cast<const VectorIndexParams *>(field.index_params().get());
  switch (vip->metric_type()) {
    case MetricType::L2:
      return 1.0 - 2.0 * std::atan(score) / M_PI;
    case MetricType::IP:
      return 0.5 + std::atan(score) / M_PI;
    case MetricType::COSINE:
      return 1.0 - score / 2.0;
    default:
      return tl::make_unexpected(Status::InvalidArgument(
          "Unsupported metric type for normalization: ",
          std::to_string(static_cast<int>(vip->metric_type()))));
  }
}

}  // anonymous namespace

namespace reranker {

Result<DocPtrList> rerank(const RerankParams &params,
                          const std::vector<DocPtrList> &results,
                          const std::vector<FieldSchema::Ptr> &fields,
                          int topn) {
  return std::visit(
      [&](const auto &p) -> Result<DocPtrList> {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, RrfParams>) {
          auto score_fn = [&p](double /*score*/, int rank,
                               size_t /*field_idx*/) -> Result<double> {
            return 1.0 / (static_cast<double>(p.rank_constant) +
                          static_cast<double>(rank) + 1.0);
          };
          return score_based_rerank(score_fn, results, topn);

        } else if constexpr (std::is_same_v<T, WeightedParams>) {
          if (p.weights.size() != results.size()) {
            return tl::make_unexpected(Status::InvalidArgument(
                "WeightedParams: weights count (", p.weights.size(),
                ") != results count (", results.size(), ")"));
          }
          if (fields.size() != results.size()) {
            return tl::make_unexpected(Status::InvalidArgument(
                "WeightedParams: fields count (", fields.size(),
                ") != results count (", results.size(), ")"));
          }
          auto score_fn = [&p, &fields](double score, int /*rank*/,
                                        size_t field_idx) -> Result<double> {
            if (!fields[field_idx]) {
              return tl::make_unexpected(Status::InvalidArgument(
                  "WeightedParams: null field schema at index ", field_idx));
            }
            auto normalized = normalize_score(score, *fields[field_idx]);
            if (!normalized.has_value()) {
              return tl::make_unexpected(normalized.error());
            }
            return normalized.value() * p.weights[field_idx];
          };
          return score_based_rerank(score_fn, results, topn);

        } else if constexpr (std::is_same_v<T, CallbackParams>) {
          if (!p.callback) {
            return tl::make_unexpected(
                Status::InvalidArgument("CallbackParams: callback is empty"));
          }
          return p.callback(results, fields, topn);
        }
      },
      params);
}

}  // namespace reranker
}  // namespace zvec
