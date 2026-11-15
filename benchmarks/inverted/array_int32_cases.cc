// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "data_generator.h"

namespace zvec::benchmarks::inverted {
namespace {

std::vector<std::string> CaseNames(const DatasetSpec &spec) {
  std::vector<std::string> names{"any_1_hit", "any_8_global", "all_2_adjacent"};
  if (spec.array_length > 1) names.emplace_back("all_m_witness");
  for (const auto *name :
       {"all_8_global", "all_hit_miss", "not_any_2", "not_all_2", "is_null",
        "is_not_null", "len_eq_zero", "len_eq_raw"}) {
    names.emplace_back(name);
  }
  return names;
}

void MakeCases(PreparedDataset &dataset, const std::vector<uint8_t> &kinds) {
  std::vector<int32_t> observed;
  for (uint32_t term = 0; term < dataset.spec.cardinality; ++term) {
    if (dataset.stats.term_doc_frequencies[term] != 0) {
      observed.push_back(static_cast<int32_t>(term));
    }
  }
  QueryCase any_one{"any_1_hit", {}, false};
  QueryCase missing{"all_hit_miss", {}, false};
  for (int32_t term : data::Pool(observed)) {
    any_one.inputs.push_back(
        data::ListQuery(QueryOp::kAny, std::vector<int32_t>{term}));
    missing.inputs.push_back(data::ListQuery(
        QueryOp::kAll,
        std::vector<int32_t>{
            term, static_cast<int32_t>(dataset.spec.cardinality + 17)}));
  }
  QueryCase any_global{"any_8_global", {}, false};
  QueryCase all_global{"all_8_global", {}, false};
  data::Random query_random(dataset.seed ^ 0x1d8e4e27c47d124fULL);
  for (size_t i = 0; i < data::kQueryPoolSize; ++i) {
    std::vector<int32_t> terms;
    while (terms.size() < std::min<uint32_t>(8, dataset.spec.cardinality)) {
      const auto term =
          static_cast<int32_t>(query_random.bounded(dataset.spec.cardinality));
      if (std::find(terms.begin(), terms.end(), term) == terms.end())
        terms.push_back(term);
    }
    std::sort(terms.begin(), terms.end());
    auto query = data::ListQuery(QueryOp::kAny, terms);
    any_global.inputs.push_back(query);
    query.op = QueryOp::kAll;
    all_global.inputs.push_back(std::move(query));
  }
  QueryCase adjacent{"all_2_adjacent", {}, false};
  QueryCase not_any{"not_any_2", {}, false};
  QueryCase not_all{"not_all_2", {}, false};
  for (size_t i = 0; i < data::kQueryPoolSize; ++i) {
    const auto anchor = static_cast<int32_t>(i * dataset.spec.cardinality /
                                             data::kQueryPoolSize);
    const auto next =
        static_cast<int32_t>((anchor + 1) % dataset.spec.cardinality);
    auto query =
        data::ListQuery(QueryOp::kAll, std::vector<int32_t>{anchor, next});
    adjacent.inputs.push_back(query);
    query.op = QueryOp::kNotAny;
    not_any.inputs.push_back(query);
    query.op = QueryOp::kNotAll;
    not_all.inputs.push_back(std::move(query));
  }
  dataset.cases.push_back(std::move(any_one));
  dataset.cases.push_back(std::move(any_global));
  dataset.cases.push_back(std::move(adjacent));
  if (dataset.spec.array_length > 1) {
    QueryCase witnesses{"all_m_witness", {}, true};
    const size_t plain_count =
        std::count(kinds.begin(), kinds.end(), data::kPlain);
    const size_t count = std::min(data::kQueryPoolSize, plain_count);
    size_t ordinal = 0;
    size_t selected = 0;
    for (uint32_t id = 0; id < dataset.num_docs && selected < count; ++id) {
      if (kinds[id] != data::kPlain) continue;
      if (ordinal == selected * plain_count / count) {
        std::vector<int32_t> terms;
        const auto &bytes = dataset.values[id];
        for (uint32_t i = 0;
             i < std::min<uint32_t>(8, dataset.spec.array_length); ++i) {
          terms.push_back(data::ArrayElement(bytes, i));
        }
        witnesses.inputs.push_back(data::ListQuery(QueryOp::kAll, terms));
        ++selected;
      }
      ++ordinal;
    }
    dataset.cases.push_back(std::move(witnesses));
  }
  dataset.cases.push_back(std::move(all_global));
  dataset.cases.push_back(std::move(missing));
  dataset.cases.push_back(std::move(not_any));
  dataset.cases.push_back(std::move(not_all));
  dataset.cases.push_back({"is_null", {{QueryOp::kIsNull, {}, 0}}, false});
  dataset.cases.push_back(
      {"is_not_null", {{QueryOp::kIsNotNull, {}, 0}}, false});
  dataset.cases.push_back(
      {"len_eq_zero", {{QueryOp::kLengthEq, {}, 0}}, false});
  dataset.cases.push_back(
      {"len_eq_raw",
       {{QueryOp::kLengthEq, {}, dataset.spec.array_length}},
       false});
}

PreparedDataset Prepare(const DatasetSpec &spec, uint32_t num_docs,
                        uint64_t seed) {
  if (spec.type != "array_int32" || spec.cardinality < 8 ||
      spec.array_length == 0 || spec.array_length > spec.cardinality ||
      (spec.array_length != 1 && spec.array_length % 2 != 0) ||
      (spec.distribution != "random" && spec.distribution != "window")) {
    throw std::invalid_argument("invalid ARRAY_INT32 dataset specification");
  }
  PreparedDataset dataset;
  dataset.spec = spec;
  dataset.num_docs = num_docs;
  dataset.seed = seed;
  const auto kinds =
      data::RowKinds(num_docs, true, spec.array_length > 1, seed);
  dataset.is_null.resize(num_docs, 0);
  dataset.values.resize(num_docs);
  dataset.stats.term_doc_frequencies.resize(spec.cardinality, 0);
  std::vector<uint32_t> rows;
  rows.reserve(num_docs - 2 * (num_docs / 10));
  for (uint32_t id = 0; id < num_docs; ++id) {
    if (kinds[id] == data::kNull) {
      dataset.is_null[id] = 1;
      ++dataset.stats.null_count;
    } else if (kinds[id] == data::kEmpty) {
      ++dataset.stats.empty_count;
      ++dataset.stats.raw_length_histogram[0];
      ++dataset.stats.unique_length_histogram[0];
    } else {
      rows.push_back(id);
      if (kinds[id] == data::kDuplicate) ++dataset.stats.duplicate_count;
    }
  }
  data::Random(seed ^ 0x589965cc75374cc3ULL).shuffle(rows);
  data::Random values(seed ^ 0x8ebc6af09c88c6e3ULL);
  std::vector<uint32_t> seen(spec.cardinality, 0);
  std::vector<int32_t> members;
  members.reserve(spec.array_length);
  for (size_t position = 0; position < rows.size(); ++position) {
    const uint32_t id = rows[position];
    const bool repeated = kinds[id] == data::kDuplicate;
    const uint32_t unique_length =
        repeated ? spec.array_length / 2 : spec.array_length;
    members.clear();
    if (spec.distribution == "window") {
      const uint32_t start = position % spec.cardinality;
      for (uint32_t i = 0; i < unique_length; ++i) {
        members.push_back(static_cast<int32_t>((start + i) % spec.cardinality));
      }
    } else {
      while (members.size() < unique_length) {
        const uint32_t term = values.bounded(spec.cardinality);
        if (seen[term] == id + 1) continue;
        seen[term] = id + 1;
        members.push_back(static_cast<int32_t>(term));
      }
    }
    auto &bytes = dataset.values[id];
    bytes.resize(spec.array_length * sizeof(int32_t));
    size_t output = 0;
    for (int32_t member : members) {
      ++dataset.stats.term_doc_frequencies[member];
      for (uint32_t copy = 0; copy < (repeated ? 2U : 1U); ++copy) {
        std::memcpy(bytes.data() + output * sizeof(int32_t), &member,
                    sizeof(member));
        ++output;
      }
    }
    dataset.stats.raw_elements += spec.array_length;
    dataset.stats.distinct_elements += unique_length;
    ++dataset.stats.raw_length_histogram[spec.array_length];
    ++dataset.stats.unique_length_histogram[unique_length];
  }
  data::FinishStats(dataset);
  MakeCases(dataset, kinds);
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
  const auto &bytes = dataset.values.at(id);
  if (bytes.size() % sizeof(int32_t) != 0) {
    throw std::invalid_argument("invalid ARRAY_INT32 native payload size");
  }
  const size_t length = bytes.size() / sizeof(int32_t);
  if (query.op == QueryOp::kLengthEq) return length == query.length;
  if (query.values.empty())
    throw std::invalid_argument("empty membership query");
  bool any = false;
  bool all = true;
  for (const auto &operand : query.values) {
    const int32_t term = data::ReadNative<int32_t>(operand);
    bool present = false;
    for (size_t i = 0; i < length; ++i) {
      if (data::ArrayElement(bytes, i) == term) {
        present = true;
        break;
      }
    }
    any = any || present;
    all = all && present;
  }
  switch (query.op) {
    case QueryOp::kAny:
      return any;
    case QueryOp::kAll:
      return all;
    case QueryOp::kNotAny:
      return !any;
    case QueryOp::kNotAll:
      return !all;
    default:
      throw std::invalid_argument("unsupported ARRAY_INT32 predicate");
  }
}

}  // namespace

TypeSuite ArrayInt32Suite() {
  TypeSuite suite;
  suite.name = "array_int32";
  suite.data_type = DataType::ARRAY_INT32;
  suite.profiles = {{"random_l8", "array_int32", "random", 1000, 8, false},
                    {"window_l8", "array_int32", "window", 1000, 8, false},
                    {"random_l1", "array_int32", "random", 1000, 1, true},
                    {"random_l32", "array_int32", "random", 1000, 32, true}};
  suite.case_names = CaseNames;
  suite.prepare = Prepare;
  suite.matches = Matches;
  return suite;
}

}  // namespace zvec::benchmarks::inverted
