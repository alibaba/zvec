// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <gflags/gflags.h>
#include "runner.h"

DEFINE_string(mode, "list", "list, build, verify, or query");
DEFINE_string(type, "int64", "Type suite (see list)");
DEFINE_string(profile, "", "Dataset profile (see list)");
DEFINE_uint32(num_docs, 100000,
              "Number of documents, including NULL and empty");
DEFINE_uint64(seed, 20261115, "Deterministic data seed");
DEFINE_string(index, "",
              "Dedicated index directory; build requires a new path");
DEFINE_string(manifest, "", "Build JSON for verify/query");
DEFINE_string(verification, "", "Successful verification JSON for query");
DEFINE_string(case, "",
              "Exact query case; verify also accepts comma-separated cases");
DEFINE_double(warmup_seconds, 0.2, "Minimum warmup duration per case");
DEFINE_double(measure_seconds, 1.0, "Minimum measured duration per case");
DEFINE_string(output, "", "New JSON output path; list defaults to stdout");

int main(int argc, char **argv) {
  gflags::SetUsageMessage(
      "Inverted subsystem benchmarks. Prefer run_benchmarks.py for full runs.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  namespace bench = zvec::benchmarks::inverted;
  namespace fs = std::filesystem;
  try {
    if (argc != 1)
      throw std::invalid_argument("Unexpected positional argument");
    if (FLAGS_seed >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument("seed must fit a nonnegative int64");
    }
    if (!FLAGS_output.empty() && fs::exists(FLAGS_output)) {
      throw std::invalid_argument("Output file already exists");
    }
    bench::JsonObject result;
    if (FLAGS_mode == "list") {
      result = bench::Catalog();
    } else {
      if (FLAGS_index.empty() || FLAGS_output.empty()) {
        throw std::invalid_argument("--index and --output are required");
      }
      auto index_path = fs::weakly_canonical(FLAGS_index);
      auto output_path = fs::weakly_canonical(FLAGS_output);
      const auto relative = output_path.lexically_relative(index_path);
      if (!relative.empty() && *relative.begin() != "..") {
        throw std::invalid_argument(
            "Reports must be outside the index directory");
      }
      bench::RunOptions options;
      options.type = FLAGS_type;
      options.profile = FLAGS_profile;
      options.num_docs = FLAGS_num_docs;
      options.seed = FLAGS_seed;
      options.index = FLAGS_index;
      options.manifest = FLAGS_manifest;
      options.verification = FLAGS_verification;
      options.case_filter = FLAGS_case;
      options.warmup_seconds = FLAGS_warmup_seconds;
      options.measure_seconds = FLAGS_measure_seconds;
      if (FLAGS_mode == "build")
        result = bench::RunBuild(options);
      else if (FLAGS_mode == "verify")
        result = bench::RunVerify(options);
      else if (FLAGS_mode == "query")
        result = bench::RunQuery(options);
      else
        throw std::invalid_argument("Unknown mode: " + FLAGS_mode);
    }
    if (FLAGS_output.empty()) {
      std::cout
          << zvec::ailego::JsonValue(result).as_json_string().as_stl_string()
          << '\n';
    } else {
      bench::WriteJson(FLAGS_output, result);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "inverted_bench: " << error.what() << '\n';
    return 1;
  }
}
