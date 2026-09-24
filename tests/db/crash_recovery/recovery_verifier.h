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

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include "checkpoint_recovery.h"

namespace zvec::checkpoint_test {

using ExpectedDocs = std::map<int, int>;

inline void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

inline void Check(const Status &status) {
  Require(status.ok(), status.message());
}

inline void CheckWrite(const Result<WriteResults> &result, size_t count) {
  if (!result) throw std::runtime_error(result.error().message());
  Require(result->size() == count, "Incorrect write result count");
  for (const auto &status : result.value()) Check(status);
}

inline ExpectedDocs InitialDocs(int count = kInitialDocs) {
  ExpectedDocs docs;
  for (int id = 0; id < count; ++id) docs.emplace(id, 0);
  return docs;
}

inline std::map<std::string, float> CheckQuery(
    Collection &collection, const SearchQuery &query,
    const std::set<std::string> &expected) {
  auto result = collection.query(query);
  if (!result) throw std::runtime_error(result.error().message());
  std::set<std::string> actual;
  std::map<std::string, float> scores;
  for (const auto &doc : result.value()) {
    Require(doc != nullptr, "Null query result");
    Require(actual.insert(doc->pk()).second, "Duplicate query result");
    Require(std::isfinite(doc->score()), "Non-finite query score");
    scores[doc->pk()] = doc->score();
  }
  Require(actual == expected,
          "Query key mismatch: " + query.target_.field_name_);
  return scores;
}

inline void CheckTopK(Collection &collection, SearchQuery query,
                      const std::map<std::string, float> &scores,
                      bool descending) {
  query.topk_ = 3;
  std::vector<float> wanted;
  for (const auto &[key, score] : scores) wanted.push_back(score);
  std::sort(wanted.begin(), wanted.end());
  if (descending) std::reverse(wanted.begin(), wanted.end());
  wanted.resize(std::min<size_t>(3, wanted.size()));
  auto result = collection.query(query);
  if (!result) throw std::runtime_error(result.error().message());
  Require(result->size() == wanted.size(), "Incorrect small top-k size");
  std::set<std::string> seen;
  for (size_t i = 0; i < wanted.size(); ++i) {
    const auto &doc = (*result)[i];
    Require(doc && scores.count(doc->pk()) && seen.insert(doc->pk()).second,
            "Invalid small top-k document");
    const float tolerance = 1e-5f * std::max(1.0f, std::abs(wanted[i]));
    Require(std::abs(doc->score() - wanted[i]) <= tolerance &&
                std::abs(scores.at(doc->pk()) - wanted[i]) <= tolerance,
            "Incorrect top-k score or ordering: " + query.target_.field_name_);
  }
}

inline void VerifyState(Collection &collection, bool fts,
                        const ExpectedDocs &expected, int max_id,
                        bool pristine = false) {
  auto stats = collection.stats();
  if (!stats) throw std::runtime_error(stats.error().message());
  Require(stats->doc_count == expected.size(), "Incorrect document count");
  std::vector<std::string> keys;
  for (int id = 0; id <= max_id; ++id) keys.push_back(PrimaryKey(id));
  auto fetched = collection.fetch(keys);
  if (!fetched) throw std::runtime_error(fetched.error().message());
  for (int id = 0; id <= max_id; ++id) {
    const auto actual = fetched->find(PrimaryKey(id));
    Require(actual != fetched->end(), "Missing fetch result");
    auto wanted = expected.find(id);
    if (wanted == expected.end()) {
      Require(!actual->second, "Deleted document survived: " + PrimaryKey(id));
    } else {
      Require(actual->second && *actual->second == MakeDoc(id, wanted->second),
              "Incorrect document: " + PrimaryKey(id));
    }
  }
  SearchQuery query;
  query.topk_ = max_id + 2;
  query.target_.field_name_ = "vec";
  const std::vector<float> vector{0, 0, 1, 0};
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float)));
  std::set<std::string> all_keys;
  std::map<std::string, float> distances;
  for (const auto &[id, generation] : expected) {
    all_keys.insert(PrimaryKey(id));
    distances[PrimaryKey(id)] = float(id * id + generation * generation);
  }
  // Exact distances and small top-k catch errors hidden by full key-set checks.
  const auto actual = CheckQuery(collection, query, all_keys);
  for (const auto &[key, distance] : distances) {
    Require(
        std::abs(actual.at(key) - distance) <= 1e-4f * std::max(1.0f, distance),
        "Incorrect L2 distance: " + key);
  }
  CheckTopK(collection, query, distances, false);
  // Query every generation, including generations whose rows were deleted.
  for (int generation = 0; generation <= 3; ++generation) {
    std::set<std::string> generation_keys;
    for (const auto &[id, value] : expected) {
      if (value == generation) generation_keys.insert(PrimaryKey(id));
    }
    query.filter_ = "generation = " + std::to_string(generation);
    CheckQuery(collection, query, generation_keys);
    if (!fts) continue;
    for (const auto *field : {"text", "title"}) {
      SearchQuery text;
      text.topk_ = max_id + 2;
      text.target_.field_name_ = field;
      FtsClause clause;
      clause.query_string_ =
          std::string(field) + "version" + std::to_string(generation);
      text.target_.clause_ = clause;
      CheckQuery(collection, text, generation_keys);
      clause.query_string_ =
          std::string(field) == "text" ? "titlerank" : "textrank";
      text.target_.clause_ = clause;
      CheckQuery(collection, text, {});
    }
  }
  if (!fts) return;
  for (const auto *field : {"text", "title"}) {
    SearchQuery text;
    text.topk_ = max_id + 2;
    text.target_.field_name_ = field;
    FtsClause clause;
    clause.query_string_ = std::string(field) + "rank";
    text.target_.clause_ = clause;
    auto scores = CheckQuery(collection, text, all_keys);
    // This pristine corpus occupies one segment with known BM25 statistics.
    // Mutated corpora may retain deleted physical rows until compaction.
    if (pristine && expected == InitialDocs()) {
      float total_length = 0;
      for (const auto &[id, generation] : expected) {
        int tf = std::string(field) == "text" ? id % 5 + 1 : 5 - id % 5;
        total_length += tf + 1 + id % 3;
      }
      const float avg = total_length / expected.size();
      const float idf = std::log(1.0f + 0.5f / (expected.size() + 0.5f));
      for (const auto &[id, generation] : expected) {
        float tf = std::string(field) == "text" ? id % 5 + 1 : 5 - id % 5;
        float score = idf * tf * 2.2f /
                      (tf + 1.2f * (0.25f + 0.75f * (tf + 1 + id % 3) / avg));
        Require(std::abs(scores.at(PrimaryKey(id)) - score) < 1e-5f,
                "Incorrect BM25 score: " + std::string(field) + " " +
                    PrimaryKey(id));
      }
    }
    CheckTopK(collection, text, scores, true);
  }
}

}  // namespace zvec::checkpoint_test
