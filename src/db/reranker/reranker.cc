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
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/index_params.h>
#include <zvec/db/reranker.h>

namespace zvec {

// ==================== ScoreBasedReranker ====================

Result<DocPtrList> ScoreBasedReranker::rerank(
    const std::map<std::string, DocPtrList> &query_results, int topn) const {
  std::unordered_map<std::string, double> scores;
  std::unordered_map<std::string, Doc::Ptr> id_to_doc;

  for (const auto &[field_name, docs] : query_results) {
    for (size_t rank = 0; rank < docs.size(); ++rank) {
      const auto &doc = docs[rank];
      const std::string &doc_id = doc->pk();
      auto rs = rescore(static_cast<double>(doc->score()),
                        static_cast<int>(rank), field_name);
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

  DocPtrList results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    const auto &[doc_id, score] = pq.top();
    auto doc = std::move(id_to_doc[doc_id]);
    doc->set_score(static_cast<float>(score));
    results.push_back(std::move(doc));
    pq.pop();
  }
  std::reverse(results.begin(), results.end());
  return results;
}

// ==================== RrfReranker ====================

Result<double> RrfReranker::rescore(double /*score*/, int rank,
                                    const std::string & /*field_name*/) const {
  return 1.0 / (static_cast<double>(rank_constant_) +
                static_cast<double>(rank) + 1.0);
}

// ==================== WeightedReranker ====================

WeightedReranker::WeightedReranker(const std::map<std::string, double> &weights)
    : weights_(weights) {}

void WeightedReranker::bind_schema(CollectionSchema::Ptr schema) {
  schema_ = std::move(schema);
}

Result<double> WeightedReranker::normalize_score(double score,
                                                 const FieldSchema &field) {
  auto *vip =
      dynamic_cast<const VectorIndexParams *>(field.index_params().get());
  if (!vip) {
    return tl::make_unexpected(
        Status::InvalidArgument("WeightedReranker: field '", field.name(),
                                "' has no vector index params"));
  }
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

Result<double> WeightedReranker::rescore(double score, int /*rank*/,
                                         const std::string &field_name) const {
  const auto *field = schema_->get_vector_field(field_name);
  if (!field) {
    return tl::make_unexpected(Status::InvalidArgument(
        "WeightedReranker: vector field not found: '", field_name + "'"));
  }
  auto normalized = normalize_score(score, *field);
  if (!normalized.has_value()) {
    return tl::make_unexpected(normalized.error());
  }
  double weight = 1.0;
  auto weight_it = weights_.find(field_name);
  if (weight_it != weights_.end()) {
    weight = weight_it->second;
  }
  return normalized.value() * weight;
}

}  // namespace zvec
