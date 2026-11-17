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
#include <stdexcept>

namespace zvec::benchmarks::inverted {

JsonObject InputRecord(const std::string &name, const JsonArray &inputs) {
  if (name.empty() || inputs.empty())
    throw std::invalid_argument(
        "Query name and parameter pool must be nonempty");
  JsonObject result;
  result.set("name", JsonStringValue(name));
  result.set("inputs", inputs);
  return result;
}

std::vector<JsonObject> ReadQueryInputs(const JsonArray &cases) {
  if (cases.empty())
    throw std::invalid_argument("Query case list must not be empty");
  std::set<std::string> seen;
  std::vector<JsonObject> records;
  for (const auto &value : cases) {
    if (!value.is_object()) throw std::invalid_argument("Invalid query record");
    const auto entry = value.as_object();
    const auto name = Text(entry, "name");
    if (name.empty() || !seen.insert(name).second ||
        !entry["inputs"].is_array())
      throw std::invalid_argument("Invalid or duplicate query parameters");
    const auto inputs = entry["inputs"].as_array();
    if (inputs.empty())
      throw std::invalid_argument("Query parameter pool must be nonempty: " +
                                  name);
    records.push_back(entry);
  }
  return records;
}

void ValidateBuildReport(const std::string &build_report_path,
                         const std::string &index_path) {
  const auto build = ReadJson(build_report_path);
  if (std::filesystem::weakly_canonical(Text(build, "index_path")) !=
      std::filesystem::weakly_canonical(index_path))
    throw std::invalid_argument("Index path does not match build report");
}

JsonObject LoadVerification(const std::string &verification_path,
                            const JsonObject &inputs) {
  const auto verified = ReadJson(verification_path);
  if (!verified["passed"].is_boolean() || !verified["passed"].as_bool() ||
      !verified["cases"].is_array())
    throw std::invalid_argument(
        "query requires successful verification"
        "\n  run --mode=verify with the same --run_dir first");
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
      result = entry;
      found = true;
    }
  }
  if (!found) throw std::invalid_argument("Query was not verified: " + name);
  return result;
}

}  // namespace zvec::benchmarks::inverted
