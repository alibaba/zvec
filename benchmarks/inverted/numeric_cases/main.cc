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

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <gflags/gflags.h>
#include "numeric_cases.h"

DEFINE_string(mode, "list", "list, build, verify, or query");
DEFINE_string(data_type, "",
              "Numeric data type: int32 or int64 (required to build)");
DEFINE_string(profile, "", "Numeric dataset preset (see list)");
DEFINE_string(config, "{}", "JSON fields overriding the numeric profile");
DEFINE_uint32(num_docs, 100000,
              "Numeric document count: 100000 or 10000000, including NULL");
DEFINE_string(run_dir, "", "Directory containing index and phase reports");
DEFINE_string(case, "",
              "Query name; verify also accepts comma-separated names");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("Inverted-index benchmarks for numeric data.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  namespace bench = zvec::benchmarks::inverted;
  namespace fs = std::filesystem;
  try {
    if (argc != 1)
      throw std::invalid_argument("Unexpected positional argument");
    if (FLAGS_mode != "list" && FLAGS_mode != "build" &&
        FLAGS_mode != "verify" && FLAGS_mode != "query")
      throw std::invalid_argument("Unknown mode: " + FLAGS_mode);
    if (!FLAGS_data_type.empty() && FLAGS_data_type != "int32" &&
        FLAGS_data_type != "int64")
      throw std::invalid_argument("--data_type must be int32 or int64");
    bench::JsonValue config;
    if (!config.parse(FLAGS_config) || !config.is_object())
      throw std::invalid_argument("--config must be a JSON object");
    auto overrides = config.as_object();
    if ((FLAGS_mode == "list" || FLAGS_mode == "build") &&
        !gflags::GetCommandLineFlagInfoOrDie("num_docs").is_default) {
      if (!overrides["num_docs"].is_null())
        throw std::invalid_argument(
            "Specify num_docs once, in --num_docs or --config");
      overrides["num_docs"] = bench::JsonValue(FLAGS_num_docs);
    }

    if (FLAGS_mode == "list") {
      bench::JsonArray profiles;
      for (const auto type : {zvec::DataType::INT32, zvec::DataType::INT64}) {
        const auto name = type == zvec::DataType::INT32 ? "int32" : "int64";
        if (!FLAGS_data_type.empty() && FLAGS_data_type != name) continue;
        const auto catalog =
            bench::NumericCatalog(type, FLAGS_profile, overrides);
        const auto entries = catalog["profiles"].as_array();
        for (const auto &profile : entries) profiles.push(profile);
      }
      bench::JsonObject suite, result;
      suite.set("name", bench::JsonStringValue("numeric"));
      suite.set("profiles", profiles);
      bench::JsonArray suites;
      suites.push(suite);
      result.set("schema_version", bench::JsonValue(2));
      result.set("suites", suites);
      std::cout << bench::JsonValue(result).as_json_string().as_stl_string()
                << '\n';
      return 0;
    }
    if (FLAGS_run_dir.empty())
      throw std::invalid_argument("--run_dir is required");
    const fs::path directory = fs::absolute(FLAGS_run_dir);
    if (FLAGS_mode == "query" &&
        (FLAGS_case.empty() ||
         FLAGS_case.find_first_not_of(
             "abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos))
      throw std::invalid_argument("--case must be a single query name");
    const auto output =
        directory / (FLAGS_mode == "query" ? "query-" + FLAGS_case + ".json"
                                           : FLAGS_mode + ".json");
    if (fs::exists(output))
      throw std::invalid_argument("Output file already exists");

    bench::JsonObject result;
    if (FLAGS_mode == "build") {
      if (FLAGS_data_type.empty())
        throw std::invalid_argument("--data_type is required to build");
      const auto type = FLAGS_data_type == "int32" ? zvec::DataType::INT32
                                                   : zvec::DataType::INT64;
      const auto profile =
          bench::ResolveNumericProfile(type, FLAGS_profile, overrides);
      result = bench::BuildNumeric(profile, directory.string());
      // The preset label is metadata; the typed profile contains only workload
      // fields.
      auto dataset = result["dataset"].as_object();
      dataset["name"] = bench::JsonStringValue(FLAGS_profile);
      result["dataset"] = dataset;
    } else {
      if (!FLAGS_data_type.empty() || !FLAGS_profile.empty() ||
          !config.as_object().empty() ||
          !gflags::GetCommandLineFlagInfoOrDie("num_docs").is_default)
        throw std::invalid_argument(
            "verify/query restore the workload from build.json");
      result = FLAGS_mode == "verify"
                   ? bench::VerifyNumeric(directory.string(), FLAGS_case)
                   : bench::QueryNumeric(directory.string(), FLAGS_case);
    }
    bench::WriteJson(output.string(), result);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "inverted_numeric_bench: " << error.what() << '\n';
    return 1;
  }
}
