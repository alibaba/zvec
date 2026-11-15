// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <limits>
#include "data_generator.h"

namespace zvec::benchmarks::inverted {
namespace {

int64_t Value(uint32_t term, uint32_t cardinality) {
  return static_cast<int64_t>(term) - cardinality / 2;
}

std::vector<std::string> CaseNames(const DatasetSpec &spec) {
  std::vector<std::string> names;
  if (spec.distribution == "uniform") {
    names = {"eq_hit"};
  } else {
    names = {"eq_hot", "eq_cold"};
  }
  for (const auto *name : {"eq_miss", "ne_hit", "lt_q01", "lt_q50", "lt_q90",
                           "ge_q50", "is_null", "is_not_null"}) {
    names.emplace_back(name);
  }
  return names;
}

int64_t RangeBoundary(const PreparedDataset &dataset, uint32_t percentile) {
  const auto &frequencies = dataset.stats.term_doc_frequencies;
  const uint64_t target =
      (dataset.num_docs - dataset.stats.null_count) * percentile;
  uint64_t cumulative = 0;
  uint64_t best_distance = std::numeric_limits<uint64_t>::max();
  int64_t boundary = Value(0, dataset.spec.cardinality);
  // Candidate i means strictly less than value(i); the final boundary is one
  // beyond the domain. Equal-distance ties select the lower numeric boundary.
  for (size_t i = 0; i <= frequencies.size(); ++i) {
    const uint64_t scaled = cumulative * 100;
    const uint64_t distance =
        scaled > target ? scaled - target : target - scaled;
    if (distance < best_distance) {
      best_distance = distance;
      boundary = Value(static_cast<uint32_t>(i), dataset.spec.cardinality);
    }
    if (i < frequencies.size()) cumulative += frequencies[i];
  }
  return boundary;
}

void MakeCases(PreparedDataset &dataset) {
  std::vector<uint32_t> observed;
  for (uint32_t term = 0; term < dataset.spec.cardinality; ++term) {
    if (dataset.stats.term_doc_frequencies[term] != 0) observed.push_back(term);
  }
  auto add_points = [&](const std::string &name, QueryOp op,
                        const std::vector<uint32_t> &terms) {
    QueryCase query_case{name, {}, false};
    for (uint32_t term : data::Pool(terms)) {
      query_case.inputs.push_back(
          data::ScalarQuery(op, Value(term, dataset.spec.cardinality)));
    }
    dataset.cases.push_back(std::move(query_case));
  };

  std::vector<uint32_t> ne_terms;
  if (dataset.spec.distribution == "uniform") {
    add_points("eq_hit", QueryOp::kEq, observed);
    ne_terms = observed;
  } else {
    std::sort(observed.begin(), observed.end(),
              [&](uint32_t left, uint32_t right) {
                const auto left_df = dataset.stats.term_doc_frequencies[left];
                const auto right_df = dataset.stats.term_doc_frequencies[right];
                return left_df != right_df ? left_df > right_df : left < right;
              });
    const size_t tail_size = std::max<size_t>(1, observed.size() / 10);
    std::vector<uint32_t> hot(observed.begin(), observed.begin() + tail_size);
    std::vector<uint32_t> cold(observed.end() - tail_size, observed.end());
    add_points("eq_hot", QueryOp::kEq, hot);
    add_points("eq_cold", QueryOp::kEq, cold);
    ne_terms = std::move(hot);
  }
  std::vector<uint32_t> misses(data::kQueryPoolSize);
  std::iota(misses.begin(), misses.end(), dataset.spec.cardinality + 17);
  add_points("eq_miss", QueryOp::kEq, misses);
  add_points("ne_hit", QueryOp::kNe, ne_terms);
  for (const auto &[name, percentile] :
       std::vector<std::pair<std::string, uint32_t>>{
           {"lt_q01", 1}, {"lt_q50", 50}, {"lt_q90", 90}}) {
    dataset.cases.push_back(
        {name,
         {data::ScalarQuery(QueryOp::kLt, RangeBoundary(dataset, percentile))},
         false});
  }
  dataset.cases.push_back(
      {"ge_q50",
       {data::ScalarQuery(QueryOp::kGe, RangeBoundary(dataset, 50))},
       false});
  dataset.cases.push_back({"is_null", {{QueryOp::kIsNull, {}, 0}}, false});
  dataset.cases.push_back(
      {"is_not_null", {{QueryOp::kIsNotNull, {}, 0}}, false});
}

PreparedDataset Prepare(const DatasetSpec &spec, uint32_t num_docs,
                        uint64_t seed) {
  if (spec.cardinality == 0 || spec.type != "int64") {
    throw std::invalid_argument("invalid INT64 dataset specification");
  }
  PreparedDataset dataset;
  dataset.spec = spec;
  dataset.num_docs = num_docs;
  dataset.seed = seed;
  const auto kinds = data::RowKinds(num_docs, false, false, seed);
  dataset.is_null.resize(num_docs, 0);
  dataset.values.resize(num_docs);
  dataset.stats.null_count = num_docs / 10;
  dataset.stats.term_doc_frequencies.resize(spec.cardinality, 0);
  const uint32_t nonnull = num_docs - dataset.stats.null_count;
  std::vector<uint32_t> terms(nonnull);

  if (spec.distribution == "uniform") {
    for (uint32_t i = 0; i < nonnull; ++i) terms[i] = i % spec.cardinality;
  } else {
    std::vector<uint32_t> rank_to_term(spec.cardinality);
    std::iota(rank_to_term.begin(), rank_to_term.end(), 0);
    data::Random(seed ^ 0xe7037ed1a0b428dbULL).shuffle(rank_to_term);
    if (spec.distribution == "hotspot") {
      const uint32_t hot_keys = std::max<uint32_t>(1, spec.cardinality / 10);
      if (hot_keys == spec.cardinality) {
        throw std::invalid_argument("hotspot requires at least two values");
      }
      const uint32_t hot_rows = static_cast<uint64_t>(nonnull) * 4 / 5;
      for (uint32_t i = 0; i < nonnull; ++i) {
        const uint32_t rank =
            i < hot_rows
                ? i % hot_keys
                : hot_keys + (i - hot_rows) % (spec.cardinality - hot_keys);
        terms[i] = rank_to_term[rank];
      }
    } else if (spec.distribution == "zipf") {
      std::vector<double> cumulative(spec.cardinality);
      double total = 0;
      for (uint32_t rank = 0; rank < spec.cardinality; ++rank) {
        total += 1.0 / (rank + 1);
        cumulative[rank] = total;
      }
      data::Random values(seed ^ 0x8ebc6af09c88c6e3ULL);
      for (auto &term : terms) {
        auto it = std::upper_bound(cumulative.begin(), cumulative.end(),
                                   values.unit() * total);
        const size_t rank =
            std::min<size_t>(it - cumulative.begin(), spec.cardinality - 1);
        term = rank_to_term[rank];
      }
    } else {
      throw std::invalid_argument("unknown INT64 distribution");
    }
  }
  data::Random(seed ^ 0x589965cc75374cc3ULL).shuffle(terms);
  size_t position = 0;
  for (uint32_t id = 0; id < num_docs; ++id) {
    if (kinds[id] == data::kNull) {
      dataset.is_null[id] = 1;
      continue;
    }
    const uint32_t term = terms[position++];
    dataset.values[id] = data::NativeBytes(Value(term, spec.cardinality));
    ++dataset.stats.term_doc_frequencies[term];
  }
  dataset.stats.raw_elements = nonnull;
  dataset.stats.distinct_elements = nonnull;
  data::FinishStats(dataset);
  MakeCases(dataset);
  data::DeduplicateQueryPools(dataset);
  dataset.fingerprint = data::Fingerprint(dataset);
  return dataset;
}

bool Matches(const PreparedDataset &dataset, uint32_t id,
             const QueryInput &query) {
  const bool is_null = dataset.is_null.at(id) != 0;
  if (query.op == QueryOp::kIsNull) return is_null;
  if (query.op == QueryOp::kIsNotNull) return !is_null;
  if (is_null) return false;
  const int64_t value = data::ReadNative<int64_t>(dataset.values.at(id));
  if (query.values.size() != 1) {
    throw std::invalid_argument("INT64 predicate requires exactly one value");
  }
  const int64_t operand = data::ReadNative<int64_t>(query.values.front());
  switch (query.op) {
    case QueryOp::kEq:
      return value == operand;
    case QueryOp::kNe:
      return value != operand;
    case QueryOp::kLt:
      return value < operand;
    case QueryOp::kGe:
      return value >= operand;
    default:
      throw std::invalid_argument("unsupported INT64 predicate");
  }
}

}  // namespace

TypeSuite Int64Suite() {
  TypeSuite suite;
  suite.name = "int64";
  suite.data_type = DataType::INT64;
  suite.profiles = {{"uniform_k100", "int64", "uniform", 100, 0, false},
                    {"uniform_k10000", "int64", "uniform", 10000, 0, false},
                    {"hotspot_k10000", "int64", "hotspot", 10000, 0, false},
                    {"zipf_k10000", "int64", "zipf", 10000, 0, false}};
  suite.case_names = CaseNames;
  suite.prepare = Prepare;
  suite.matches = Matches;
  return suite;
}

}  // namespace zvec::benchmarks::inverted
