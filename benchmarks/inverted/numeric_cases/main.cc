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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <gflags/gflags.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include "shared/json.h"
#include "dataset.h"
#include "numeric_cases.h"


DEFINE_string(mode, "", "prepare, build, verify, or query");
DEFINE_string(data_type, "", "int32 or int64");
DEFINE_uint32(num_docs, 100000, "100000 or 10000000");
DEFINE_string(distribution, "uniform",
              "uniform, hotspot_moderate, or hotspot_extreme");
DEFINE_string(value_range, "0_to_30000", "0_to_30000 or 0_to_type_max");
DEFINE_string(value_order, "ascending", "ascending or shuffled");
DEFINE_double(null_ratio, 0, "0, 0.1, or 0.5");
DEFINE_string(run_dir, "", "Directory for this benchmark run");

DECLARE_bool(help);
DECLARE_bool(helpshort);


namespace {


namespace bench = zvec::benchmarks::inverted;
using bench::NumericDistribution;
using bench::NumericDocCount;
using bench::NumericNullRatio;
using bench::NumericProfile;
using bench::NumericValueOrder;
using bench::NumericValueRange;
using zvec::DataType;
using zvec::ailego::FileHelper;

bool IsSpecified(const char *name) {
  return !gflags::GetCommandLineFlagInfoOrDie(name).is_default;
}

void PrintHelp() {
  std::cout << "Usage: " << gflags::ProgramInvocationShortName()
            << " [options]\n\n"
            << gflags::ProgramUsage() << "\n\n";
  for (const auto *name :
       {"mode", "data_type", "num_docs", "distribution", "value_range",
        "value_order", "null_ratio", "run_dir"}) {
    const auto flag = gflags::GetCommandLineFlagInfoOrDie(name);
    std::cout << "  " << std::left << std::setw(16) << ("--" + flag.name)
              << flag.description;
    if (!flag.default_value.empty())
      std::cout << " (default: " << flag.default_value << ')';
    std::cout << '\n';
  }
  std::cout << "\nprepare requires --data_type and --run_dir.\n"
               "build/verify/query require the same --run_dir used for "
               "prepare.\n";
}

[[noreturn]] void InvalidValue(const char *name, const std::string &value) {
  throw std::invalid_argument(
      "invalid value '" + value + "' for --" + name +
      "\n  expected: " + gflags::GetCommandLineFlagInfoOrDie(name).description);
}

std::string DisplayNumber(double value) {
  // Keep decimals readable without rounding an invalid value into a valid one.
  std::string text;
  for (int precision = 1;
       precision <= std::numeric_limits<double>::max_digits10; ++precision) {
    std::ostringstream stream;
    stream << std::setprecision(precision) << value;
    text = stream.str();
    if (!std::isfinite(value) || std::strtod(text.c_str(), nullptr) == value)
      break;
  }
  return text;
}

NumericProfile ProfileFromFlags(DataType type) {
  // Only explicitly supplied flags override NumericProfile's defaults.
  NumericProfile profile(type);
  if (IsSpecified("num_docs")) {
    if (FLAGS_num_docs != 100000 && FLAGS_num_docs != 10000000)
      InvalidValue("num_docs", std::to_string(FLAGS_num_docs));
    profile.num_docs = static_cast<NumericDocCount>(FLAGS_num_docs);
  }
  if (IsSpecified("distribution")) {
    if (FLAGS_distribution == "uniform")
      profile.distribution = NumericDistribution::UNIFORM;
    else if (FLAGS_distribution == "hotspot_moderate")
      profile.distribution = NumericDistribution::HOTSPOT_MODERATE;
    else if (FLAGS_distribution == "hotspot_extreme")
      profile.distribution = NumericDistribution::HOTSPOT_EXTREME;
    else
      InvalidValue("distribution", FLAGS_distribution);
  }
  if (IsSpecified("value_range")) {
    if (FLAGS_value_range == "0_to_30000")
      profile.value_range = NumericValueRange::ZERO_TO_30000;
    else if (FLAGS_value_range == "0_to_type_max")
      profile.value_range = NumericValueRange::ZERO_TO_TYPE_MAX;
    else
      InvalidValue("value_range", FLAGS_value_range);
  }
  if (IsSpecified("value_order")) {
    if (FLAGS_value_order == "ascending")
      profile.value_order = NumericValueOrder::ASCENDING;
    else if (FLAGS_value_order == "shuffled")
      profile.value_order = NumericValueOrder::SHUFFLED;
    else
      InvalidValue("value_order", FLAGS_value_order);
  }
  if (IsSpecified("null_ratio")) {
    if (FLAGS_null_ratio == 0)
      profile.null_ratio = NumericNullRatio::NONE;
    else if (FLAGS_null_ratio == 0.1)
      profile.null_ratio = NumericNullRatio::TEN_PERCENT;
    else if (FLAGS_null_ratio == 0.5)
      profile.null_ratio = NumericNullRatio::FIFTY_PERCENT;
    else
      InvalidValue("null_ratio", DisplayNumber(FLAGS_null_ratio));
  }
  return profile;
}


}  // namespace


#if defined(_WIN32)
int wmain(int argc, wchar_t **wide_argv) {
  // Windows supplies UTF-16 arguments; gflags and ailego use UTF-8.
  std::vector<std::string> arguments;
  arguments.reserve(argc);
  for (int i = 0; i < argc; ++i)
    arguments.push_back(FileHelper::WideToUtf8(wide_argv[i]));
  std::vector<char *> pointers;
  for (auto &argument : arguments) pointers.push_back(argument.data());
  pointers.push_back(nullptr);
  auto argv = pointers.data();
#else
int main(int argc, char **argv) {
#endif
  const bool no_arguments = argc == 1;
  gflags::SetUsageMessage("Inverted-index benchmarks for numeric data.");
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (no_arguments || FLAGS_help || FLAGS_helpshort) {
    PrintHelp();
    return 0;
  }
  gflags::HandleCommandLineHelpFlags();
  namespace fs = std::filesystem;
  try {
    if (argc != 1)
      throw std::invalid_argument("unexpected positional argument '" +
                                  std::string(argv[1]) +
                                  "'\n  expected: named arguments such as "
                                  "--mode=build");
    if (!IsSpecified("mode"))
      throw std::invalid_argument(
          "missing required argument --mode\n  expected: " +
          gflags::GetCommandLineFlagInfoOrDie("mode").description);
    if (FLAGS_mode != "prepare" && FLAGS_mode != "build" &&
        FLAGS_mode != "verify" && FLAGS_mode != "query")
      InvalidValue("mode", FLAGS_mode);

    if (FLAGS_mode == "prepare") {
      if (IsSpecified("data_type") && FLAGS_data_type != "int32" &&
          FLAGS_data_type != "int64")
        InvalidValue("data_type", FLAGS_data_type);
      if (!IsSpecified("data_type"))
        throw std::invalid_argument(
            "missing required argument --data_type "
            "for --mode=prepare\n  expected: " +
            gflags::GetCommandLineFlagInfoOrDie("data_type").description);
    } else {
      for (const auto *name : {"data_type", "num_docs", "distribution",
                               "value_range", "value_order", "null_ratio"}) {
        if (IsSpecified(name))
          throw std::invalid_argument(
              "unexpected argument --" + std::string(name) +
              " for --mode=" + FLAGS_mode +
              "\n  this argument is only supported with --mode=prepare");
      }
    }

    if (FLAGS_run_dir.empty())
      throw std::invalid_argument(
          "missing required argument --run_dir for "
          "--mode=" +
          FLAGS_mode + "\n  expected: a benchmark run directory");
    const auto directory =
        fs::absolute(FileHelper::PathFromUtf8(FLAGS_run_dir));
    const auto run_dir = FileHelper::PathToUtf8(directory);
    if (FLAGS_mode == "prepare") {
      const auto type =
          FLAGS_data_type == "int32" ? DataType::INT32 : DataType::INT64;
      const auto profile = ProfileFromFlags(type);
      if (!zvec::ailego::File::MakePath(run_dir))
        throw std::runtime_error("cannot create run directory: '" + run_dir +
                                 "'");
      bench::NumericDataset::Create(profile, run_dir);
      return 0;
    }

    if (FLAGS_mode == "verify") {
      bench::VerifyNumeric(run_dir);
      std::cout << "Verification passed\n";
      return 0;
    }

    const auto output = FileHelper::PathJoin(run_dir, FLAGS_mode + ".json");
    if (zvec::ailego::File::IsExist(output))
      throw std::invalid_argument(
          "output file already exists: '" + output +
          "'\n  use a new --run_dir to start another run");

    bench::JsonObject result;
    if (FLAGS_mode == "build") {
      result = bench::BuildNumeric(run_dir);
    } else {
      result = bench::QueryNumeric(run_dir);
    }
    bench::WriteJson(output, result);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "inverted_numeric_bench: " << error.what() << '\n';
    return 1;
  }
}
