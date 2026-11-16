// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <gflags/gflags.h>
#include "array_numeric_cases.h"

DEFINE_string(mode, "list", "list, build, verify, or query");
DEFINE_string(profile, "", "Array dataset preset (see list)");
DEFINE_string(config, "{}",
              "JSON object overriding the selected array dataset preset");
DEFINE_uint32(num_docs, 100000,
              "Number of documents, including NULL and empty arrays");
DEFINE_uint64(seed, 20261115, "Deterministic data seed");
DEFINE_string(index, "",
              "Dedicated index directory; build requires a new path");
DEFINE_string(manifest, "", "Build JSON for verify/query");
DEFINE_string(verification, "", "Successful verification JSON for query");
DEFINE_string(case, "",
              "Query name; verify also accepts comma-separated names");
DEFINE_double(warmup_seconds, 0.2, "Minimum warmup duration per query");
DEFINE_double(measure_seconds, 1.0, "Minimum measured duration per query");
DEFINE_string(output, "", "New JSON output path; list defaults to stdout");

int main(int argc, char **argv) {
  gflags::SetUsageMessage(
      "Inverted ARRAY_INT32 benchmarks. Prefer run_benchmarks.py for full "
      "runs.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  namespace bench = zvec::benchmarks::inverted;
  namespace fs = std::filesystem;
  try {
    if (argc != 1)
      throw std::invalid_argument("Unexpected positional argument");
    if (!FLAGS_num_docs || FLAGS_seed > static_cast<uint64_t>(INT64_MAX))
      throw std::invalid_argument("Invalid data size or seed");
    if (!std::isfinite(FLAGS_warmup_seconds) || FLAGS_warmup_seconds < 0 ||
        !std::isfinite(FLAGS_measure_seconds) || FLAGS_measure_seconds <= 0)
      throw std::invalid_argument("Invalid timing duration");
    if (!FLAGS_output.empty() && fs::exists(FLAGS_output))
      throw std::invalid_argument("Output file already exists");

    bench::ArrayOptions options;
    options.profile = FLAGS_profile;
    options.num_docs = FLAGS_num_docs;
    options.seed = FLAGS_seed;
    options.index_path = FLAGS_index;
    options.build_report_path = FLAGS_manifest;
    options.verification_path = FLAGS_verification;
    options.case_filter = FLAGS_case;
    options.warmup_seconds = FLAGS_warmup_seconds;
    options.measure_seconds = FLAGS_measure_seconds;
    bench::JsonValue config;
    if (!config.parse(FLAGS_config) || !config.is_object())
      throw std::invalid_argument("--config must be a JSON object");
    options.dataset_overrides = config.as_object();

    bench::JsonObject result;
    if (FLAGS_mode == "list") {
      const auto suite = bench::ArrayCatalog(options);
      if (suite["profiles"].as_array().empty())
        throw std::invalid_argument("No matching array profile");
      bench::JsonArray suites;
      suites.push(suite);
      result.set("schema_version", bench::JsonValue(2));
      result.set("suites", suites);
    } else {
      if (options.index_path.empty() || FLAGS_output.empty())
        throw std::invalid_argument("--index and --output are required");
      const auto relative =
          fs::weakly_canonical(FLAGS_output)
              .lexically_relative(fs::weakly_canonical(options.index_path));
      if (!relative.empty() && *relative.begin() != "..")
        throw std::invalid_argument(
            "Reports must be outside the index directory");
      if (FLAGS_mode != "build" && FLAGS_mode != "verify" &&
          FLAGS_mode != "query")
        throw std::invalid_argument("Unknown mode: " + FLAGS_mode);
      if (FLAGS_mode != "build" && !options.dataset_overrides.empty())
        throw std::invalid_argument(
            "verify/query use the saved dataset configuration");
      result = bench::RunArray(FLAGS_mode, options);
      result.set("schema_version", bench::JsonValue(2));
      result.set("phase", bench::JsonStringValue(FLAGS_mode));
    }
    if (FLAGS_output.empty())
      std::cout << bench::JsonValue(result).as_json_string().as_stl_string()
                << '\n';
    else
      bench::WriteJson(FLAGS_output, result);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "inverted_array_bench: " << error.what() << '\n';
    return 1;
  }
}
