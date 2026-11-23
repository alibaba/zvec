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

#include "query_helpers.h"
#include <set>
#include <stdexcept>

namespace zvec::benchmarks::inverted {


JsonObject MakeQueryCase(const std::string &name, const JsonArray &inputs) {
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


}  // namespace zvec::benchmarks::inverted
