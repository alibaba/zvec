// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "report_utils.h"
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>

namespace zvec::benchmarks::inverted {

JsonObject InputRecord(const std::string &name, const JsonArray &inputs) {
  if (name.empty() || inputs.empty())
    throw std::invalid_argument(
        "Query name and parameter pool must be nonempty");
  JsonObject result;
  result.set("name", JsonStringValue(name));
  result.set("inputs", inputs);
  result.set("query_fingerprint",
             JsonStringValue(InputFingerprint(name, inputs)));
  return result;
}

std::vector<JsonObject> SelectInputs(const JsonArray &cases,
                                     const std::string &filter) {
  std::set<std::string> requested, seen;
  if (!filter.empty()) {
    std::istringstream stream(filter);
    std::string name;
    while (std::getline(stream, name, ',')) {
      if (name.empty() || !requested.insert(name).second)
        throw std::invalid_argument("Empty or duplicate query filter");
    }
    if (filter.back() == ',') throw std::invalid_argument("Empty query filter");
  }
  std::vector<JsonObject> selected;
  for (const auto &value : cases) {
    if (!value.is_object()) throw std::invalid_argument("Invalid query record");
    const auto entry = value.as_object();
    const auto name = Text(entry, "name");
    if (name.empty() || !seen.insert(name).second ||
        !entry["inputs"].is_array())
      throw std::invalid_argument("Invalid or duplicate query parameters");
    const auto inputs = entry["inputs"].as_array();
    if (inputs.empty() ||
        Text(entry, "query_fingerprint") != InputFingerprint(name, inputs))
      throw std::invalid_argument("Query parameter fingerprint mismatch: " +
                                  name);
    if (filter.empty() || requested.erase(name)) selected.push_back(entry);
  }
  if (!requested.empty())
    throw std::invalid_argument("Unknown query: " + *requested.begin());
  return selected;
}

JsonObject LoadBuild(const std::string &build_report_path,
                     const std::string &index_path) {
  const auto build = ReadJson(build_report_path);
  if (Number(build, "schema_version") != 2 || Text(build, "phase") != "build" ||
      !build["dataset"].is_object() || !build["cases"].is_array())
    throw std::invalid_argument("Invalid build report");
  if (std::filesystem::weakly_canonical(Text(build, "index_path")) !=
      std::filesystem::weakly_canonical(index_path))
    throw std::invalid_argument("Index path does not match build report");
  const auto dataset = build["dataset"].as_object();
  const auto count = Number(dataset, "num_docs");
  if (!count || count > UINT32_MAX || Number(dataset, "seed") > INT64_MAX)
    throw std::invalid_argument("Invalid saved dataset size or seed");
  SelectInputs(build["cases"].as_array(), "");
  return build;
}

JsonObject LoadVerification(const std::string &verification_path,
                            const JsonObject &build, const JsonObject &inputs) {
  const auto verified = ReadJson(verification_path);
  if (Number(verified, "schema_version") != 2 ||
      Text(verified, "phase") != "verify" || !verified["passed"].is_boolean() ||
      !verified["passed"].as_bool() || !verified["cases"].is_array() ||
      Text(verified, "data_fingerprint") !=
          Text(build["dataset"].as_object(), "data_fingerprint"))
    throw std::invalid_argument(
        "Successful verification of this dataset is required");
  const auto name = Text(inputs, "name");
  const auto cases = verified["cases"].as_array();
  std::set<std::string> seen;
  JsonObject result;
  bool found = false;
  for (const auto &value : cases) {
    if (!value.is_object())
      throw std::invalid_argument("Invalid verification record");
    const auto entry = value.as_object();
    const auto entry_name = Text(entry, "name");
    if (!seen.insert(entry_name).second)
      throw std::invalid_argument("Duplicate verification record");
    if (entry_name == name) {
      if (Text(entry, "query_fingerprint") != Text(inputs, "query_fingerprint"))
        throw std::invalid_argument("Verification query fingerprint mismatch");
      result = entry;
      found = true;
    }
  }
  if (!found) throw std::invalid_argument("Query was not verified: " + name);
  return result;
}

void CheckRegenerated(const JsonObject &build, const std::string &fingerprint,
                      const JsonArray &inputs) {
  if (Text(build["dataset"].as_object(), "data_fingerprint") != fingerprint)
    throw std::runtime_error("Regenerated data differs from build input");
  const auto saved = SelectInputs(build["cases"].as_array(), "");
  const auto regenerated = SelectInputs(inputs, "");
  if (saved.size() != regenerated.size())
    throw std::runtime_error("Regenerated query set differs");
  for (size_t i = 0; i < saved.size(); ++i) {
    if (Text(saved[i], "name") != Text(regenerated[i], "name") ||
        Text(saved[i], "query_fingerprint") !=
            Text(regenerated[i], "query_fingerprint"))
      throw std::runtime_error("Regenerated query parameters differ");
  }
}
}  // namespace zvec::benchmarks::inverted
