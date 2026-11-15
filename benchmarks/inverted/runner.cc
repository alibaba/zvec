// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "runner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include "index_fixture.h"

namespace zvec::benchmarks::inverted {
namespace {
using Clock = std::chrono::steady_clock;
using ailego::JsonValue;

double Seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

std::string Text(const JsonObject &obj, const char *key) {
  if (!obj[key].is_string()) {
    throw std::runtime_error(std::string("Missing string: ") + key);
  }
  return JsonStringText(obj[key]);
}

uint64_t Number(const JsonObject &obj, const char *key) {
  if (!obj[key].is_integer() || obj[key].as_integer() < 0) {
    throw std::runtime_error(std::string("Missing nonnegative integer: ") +
                             key);
  }
  return obj[key].as_integer();
}

JsonObject Report(const char *phase) {
  JsonObject result;
  result.set("schema_version", JsonValue(1));
  result.set("phase", JsonStringValue(phase));
  return result;
}

std::vector<QueryCase> SelectCases(const std::vector<QueryCase> &cases,
                                   const std::string &filter) {
  if (filter.empty()) return cases;
  std::set<std::string> names;
  std::istringstream stream(filter);
  std::string name;
  while (std::getline(stream, name, ',')) {
    if (name.empty() || !names.insert(name).second) {
      throw std::invalid_argument("Empty or duplicate case filter");
    }
  }
  if (names.empty() || filter.back() == ',') {
    throw std::invalid_argument("Empty case filter");
  }
  std::vector<QueryCase> selected;
  for (const auto &query : cases) {
    if (names.erase(query.name)) selected.push_back(query);
  }
  if (!names.empty())
    throw std::invalid_argument("Unknown case: " + *names.begin());
  return selected;
}

JsonObject Manifest(const RunOptions &options) {
  auto manifest = ReadJson(options.manifest);
  if (Number(manifest, "schema_version") != 1 ||
      Text(manifest, "phase") != "build" || !manifest["dataset"].is_object() ||
      !manifest["cases"].is_array()) {
    throw std::runtime_error("Invalid build manifest");
  }
  const auto expected_path = std::filesystem::weakly_canonical(options.index);
  if (std::filesystem::weakly_canonical(Text(manifest, "index_path")) !=
      expected_path) {
    throw std::runtime_error("Index path does not match the build manifest");
  }
  auto spec = ParseSpec(manifest["dataset"].as_object());
  const auto &registered = FindProfile(FindSuite(spec.type), spec.name);
  if (spec.distribution != registered.distribution ||
      spec.cardinality != registered.cardinality ||
      spec.array_length != registered.array_length ||
      spec.supplemental != registered.supplemental) {
    throw std::runtime_error("Dataset definition changed since build");
  }
  return manifest;
}

std::string Hex(uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}

void HashId(uint64_t *hash, uint32_t id) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    *hash = (*hash ^ ((id >> shift) & 0xffU)) * 1099511628211ULL;
  }
}
}  // namespace

const std::vector<TypeSuite> &Suites() {
  static const std::vector<TypeSuite> suites{Int64Suite(), ArrayInt32Suite()};
  return suites;
}

const TypeSuite &FindSuite(const std::string &name) {
  for (const auto &suite : Suites()) {
    if (suite.name == name) return suite;
  }
  throw std::invalid_argument("Unknown type: " + name);
}

const DatasetSpec &FindProfile(const TypeSuite &suite,
                               const std::string &name) {
  for (const auto &spec : suite.profiles) {
    if (spec.name == name) return spec;
  }
  throw std::invalid_argument("Unknown profile: " + name);
}

JsonObject Catalog() {
  JsonObject catalog;
  catalog.set("schema_version", JsonValue(1));
  JsonArray suites;
  for (const auto &suite : Suites()) {
    JsonObject entry;
    entry.set("name", JsonStringValue(suite.name));
    JsonArray profiles;
    for (const auto &profile : suite.profiles) {
      auto spec = SpecJson(profile);
      JsonArray cases;
      for (const auto &name : suite.case_names(profile)) {
        cases.push(JsonStringValue(name));
      }
      spec.set("cases", cases);
      profiles.push(spec);
    }
    entry.set("profiles", profiles);
    suites.push(entry);
  }
  catalog.set("suites", suites);
  return catalog;
}

JsonObject RunBuild(const RunOptions &options) {
  if (options.num_docs == 0)
    throw std::invalid_argument("num_docs must be positive");
  const auto &suite = FindSuite(options.type);
  const auto &spec = FindProfile(suite, options.profile);
  if (std::filesystem::exists(options.index)) {
    throw std::invalid_argument("Build index directory already exists");
  }
  auto data = suite.prepare(spec, options.num_docs, options.seed);
  if (data.values.size() != options.num_docs ||
      data.is_null.size() != options.num_docs) {
    throw std::runtime_error("Generator returned an incomplete dataset");
  }
  auto before = MemorySnapshot();
  IndexFixture index(suite, options.index, false);
  auto &column = index.column();
  const auto begin = Clock::now();
  for (uint32_t id = 0; id < data.num_docs; ++id) {
    auto status = data.is_null[id] ? column.insert_null(id)
                                   : column.insert(id, data.values[id]);
    if (!status.ok())
      throw std::runtime_error("Insert failed at doc " + std::to_string(id));
  }
  const auto inserted = Clock::now();
  index.Seal();
  const auto sealed = Clock::now();
  auto after = MemorySnapshot();
  index.Close();

  auto result = Report("build");
  result.set("dataset", DatasetJson(data));
  result.set("cases", CasesJson(data.cases));
  result.set("index_path",
             JsonStringValue(
                 std::filesystem::weakly_canonical(options.index).string()));
  result.set("insert_seconds", JsonValue(Seconds(begin, inserted)));
  result.set("seal_seconds", JsonValue(Seconds(inserted, sealed)));
  result.set("build_seconds", JsonValue(Seconds(begin, sealed)));
  result.set("docs_per_second",
             JsonValue(data.num_docs / Seconds(begin, sealed)));
  auto disk = DiskUsage(options.index);
  result.set("index_file_bytes", disk["index_file_bytes"]);
  result.set("index_allocated_bytes", disk["index_allocated_bytes"]);
  result.set("bytes_per_doc",
             JsonValue(static_cast<double>(Number(disk, "index_file_bytes")) /
                       data.num_docs));
  result.set("memory_before", before);
  result.set("memory_after", after);
  result.set("environment", EnvironmentJson());
  return result;
}

JsonObject RunVerify(const RunOptions &options) {
  auto manifest = Manifest(options);
  auto description = manifest["dataset"].as_object();
  auto spec = ParseSpec(description);
  const auto &suite = FindSuite(spec.type);
  const auto count = Number(description, "num_docs");
  if (!count || count > UINT32_MAX)
    throw std::runtime_error("Invalid manifest num_docs");
  auto data = suite.prepare(spec, static_cast<uint32_t>(count),
                            Number(description, "seed"));
  if (data.fingerprint != Text(description, "data_fingerprint")) {
    throw std::runtime_error("Regenerated data differs from build input");
  }
  auto cases = SelectCases(ParseCases(manifest["cases"].as_array()),
                           options.case_filter);
  IndexFixture index(suite, options.index, true);
  JsonArray checked;
  for (const auto &query : cases) {
    auto regenerated = SelectCases(data.cases, query.name);
    if (QueryFingerprint(regenerated.front()) != QueryFingerprint(query)) {
      throw std::runtime_error("Regenerated query differs: " + query.name);
    }
    JsonArray counts, fingerprints, ratios, nonnull_ratios;
    for (const auto &input : query.inputs) {
      auto actual = index.Search(input);
      if (!actual) throw std::runtime_error("Query failed: " + query.name);
      auto it = actual->create_iterator();
      uint64_t matches = 0, hash = 14695981039346656037ULL;
      for (uint32_t id = 0; id < data.num_docs; ++id) {
        if (!suite.matches(data, id, input)) continue;
        if (!it->valid() || it->doc_id() != id) {
          throw std::runtime_error("ID mismatch in " + query.name + " at doc " +
                                   std::to_string(id));
        }
        ++matches;
        HashId(&hash, id);
        it->next();
      }
      if (it->valid() || actual->count() != matches ||
          (query.from_documents && matches == 0) ||
          (query.name == "all_hit_miss" && matches != 0)) {
        throw std::runtime_error("Incorrect result in " + query.name);
      }
      counts.push(JsonValue(matches));
      fingerprints.push(JsonStringValue(Hex(hash)));
      ratios.push(JsonValue(static_cast<double>(matches) / data.num_docs));
      if (input.op == QueryOp::kLt || input.op == QueryOp::kGe) {
        nonnull_ratios.push(JsonValue(static_cast<double>(matches) /
                                      (data.num_docs - data.stats.null_count)));
      }
    }
    JsonObject entry;
    entry.set("name", JsonStringValue(query.name));
    entry.set("query_fingerprint", JsonStringValue(QueryFingerprint(query)));
    entry.set("counts", counts);
    entry.set("id_fingerprints", fingerprints);
    entry.set("match_ratios", ratios);
    if (!nonnull_ratios.empty())
      entry.set("nonnull_match_ratios", nonnull_ratios);
    checked.push(entry);
  }
  auto result = Report("verify");
  result.set("passed", JsonValue(true));
  result.set("data_fingerprint", JsonStringValue(data.fingerprint));
  result.set("cases", checked);
  return result;
}

JsonObject RunQuery(const RunOptions &options) {
  if (options.case_filter.empty())
    throw std::invalid_argument("query requires --case");
  if (!std::isfinite(options.warmup_seconds) || options.warmup_seconds < 0 ||
      !std::isfinite(options.measure_seconds) || options.measure_seconds <= 0) {
    throw std::invalid_argument("Invalid timing duration");
  }
  auto manifest = Manifest(options);
  auto description = manifest["dataset"].as_object();
  const auto &suite = FindSuite(Text(description, "type"));
  auto cases = SelectCases(ParseCases(manifest["cases"].as_array()),
                           options.case_filter);
  if (cases.size() != 1 || cases.front().inputs.empty()) {
    throw std::invalid_argument("query requires exactly one nonempty case");
  }
  const auto &query = cases.front();
  auto verified = ReadJson(options.verification);
  if (Text(verified, "phase") != "verify" || !verified["passed"].is_boolean() ||
      !verified["passed"].as_bool() ||
      Text(verified, "data_fingerprint") !=
          Text(description, "data_fingerprint") ||
      !verified["cases"].is_array()) {
    throw std::runtime_error("Valid verification report is required");
  }
  JsonArray expected;
  for (const auto &entry : verified["cases"].as_array()) {
    auto item = entry.as_object();
    if (Text(item, "name") == query.name) {
      if (Text(item, "query_fingerprint") != QueryFingerprint(query) ||
          !item["counts"].is_array()) {
        throw std::runtime_error("Verification query mismatch");
      }
      expected = item["counts"].as_array();
      break;
    }
  }
  if (expected.size() != query.inputs.size())
    throw std::runtime_error("Case was not fully verified");
  std::vector<uint64_t> expected_counts;
  for (const auto &value : expected) {
    if (!value.is_integer() || value.as_integer() < 0 ||
        static_cast<uint64_t>(value.as_integer()) >
            Number(description, "num_docs")) {
      throw std::runtime_error("Invalid verified result count");
    }
    expected_counts.push_back(value.as_integer());
  }

  auto before = MemorySnapshot();
  IndexFixture index(suite, options.index, true);
  uint64_t checksum = 0;
  auto run_pool = [&]() {
    for (size_t i = 0; i < query.inputs.size(); ++i) {
      auto result = index.Search(query.inputs[i]);
      if (!result) {
        throw std::runtime_error("Query failed during timing");
      }
      const auto count = result->count();
      if (count != expected_counts[i]) {
        throw std::runtime_error("Query failed or count changed during timing");
      }
      checksum += count;
    }
  };
  const auto warmup = Clock::now();
  do {
    run_pool();
  } while (Seconds(warmup, Clock::now()) < options.warmup_seconds);

  uint64_t completed = 0;
  checksum = 0;
  const auto begin = Clock::now();
  Clock::time_point end;
  do {
    run_pool();
    completed += query.inputs.size();
    end = Clock::now();
  } while (Seconds(begin, end) < options.measure_seconds);
  const double elapsed = Seconds(begin, end);
  auto after = MemorySnapshot();
  index.Close();
  auto result = Report("query");
  result.set("case", JsonStringValue(query.name));
  result.set("data_fingerprint", description["data_fingerprint"]);
  result.set("query_fingerprint", JsonStringValue(QueryFingerprint(query)));
  result.set("completed_queries", JsonValue(completed));
  result.set("elapsed_seconds", JsonValue(elapsed));
  result.set("mean_ns_per_query", JsonValue(elapsed * 1e9 / completed));
  result.set("qps", JsonValue(completed / elapsed));
  result.set("match_counts", expected);
  result.set("result_checksum", JsonStringValue(Hex(checksum)));
  result.set("warmup_seconds", JsonValue(options.warmup_seconds));
  result.set("measure_seconds", JsonValue(options.measure_seconds));
  result.set("memory_before", before);
  result.set("memory_after", after);
  result.set("environment", EnvironmentJson());
  return result;
}
}  // namespace zvec::benchmarks::inverted
