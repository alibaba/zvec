// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "metrics.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "ailego/utility/memory_helper.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <sys/utsname.h>
#endif
#if __has_include(<zvec/version.h>)
#include <zvec/version.h>
#endif

namespace zvec::benchmarks::inverted {

std::string Text(const JsonObject &object, const char *key) {
  return JsonStringText(object[key]);
}

uint64_t Number(const JsonObject &object, const char *key) {
  if (!object[key].is_integer() || object[key].as_integer() < 0)
    throw std::invalid_argument(std::string("Expected nonnegative integer: ") +
                                key);
  return object[key].as_integer();
}

double Seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

std::string Hex(uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}

void HashId(uint64_t *hash, uint32_t id) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    *hash = (*hash ^ ((id >> shift) & 0xffU)) * 1099511628211ULL;
}

std::string InputFingerprint(const std::string &name, const JsonArray &inputs) {
  uint64_t hash = 14695981039346656037ULL;
  const std::string encoded =
      name + ":" + JsonValue(inputs).as_json_string().as_stl_string();
  for (unsigned char c : encoded) hash = (hash ^ c) * 1099511628211ULL;
  return Hex(hash);
}

JsonValue JsonStringValue(const std::string &value) {
  return JsonValue(ailego::JsonString(value).encode());
}

std::string JsonStringText(const JsonValue &value) {
  if (!value.is_string()) {
    throw std::invalid_argument("Expected JSON string");
  }
  return value.as_string().decode().as_stl_string();
}

void WriteJson(const std::string &path, const JsonObject &object) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.open(path, std::ios::binary | std::ios::trunc);
  output << JsonValue(object).as_json_string().as_stl_string() << '\n';
  output.close();
}

JsonObject ReadJson(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot read JSON file: " + path);
  std::ostringstream content;
  content << input.rdbuf();
  if (input.bad())
    throw std::runtime_error("Failed reading JSON file: " + path);
  JsonValue value;
  if (!value.parse(content.str()) || !value.is_object()) {
    throw std::runtime_error("Invalid JSON object: " + path);
  }
  return value.as_object();
}

JsonObject MemorySnapshot() {
  JsonObject result;
  size_t vsz = 0;
  size_t rss = 0;
  const bool available = ailego::MemoryHelper::SelfUsage(&vsz, &rss) && rss > 0;
  const auto peak = available ? ailego::MemoryHelper::SelfPeakRSS() : 0;
  result.set("rss_bytes", available ? JsonValue(rss) : JsonValue());
  result.set("peak_rss_bytes", peak ? JsonValue(peak) : JsonValue());
  return result;
}

JsonObject DiskUsage(const std::string &path) {
  namespace fs = std::filesystem;
  if (!fs::is_directory(path)) {
    throw std::runtime_error("Cannot measure missing index directory: " + path);
  }
  uint64_t logical_bytes = 0;
  uint64_t allocated_bytes = 0;
  uint64_t file_count = 0;
  for (const auto &entry : fs::recursive_directory_iterator(path)) {
    if (!fs::is_regular_file(entry.symlink_status())) continue;
    logical_bytes += entry.file_size();
    ++file_count;
#if defined(__unix__) || defined(__APPLE__)
    struct stat info {};
    if (::stat(entry.path().c_str(), &info) != 0 || info.st_blocks < 0) {
      throw std::runtime_error("Cannot stat index file: " +
                               entry.path().string());
    }
    allocated_bytes += static_cast<uint64_t>(info.st_blocks) * 512;
#endif
  }
  JsonObject result;
  result.set("index_file_bytes", JsonValue(logical_bytes));
  result.set("index_file_count", JsonValue(file_count));
#if defined(__unix__) || defined(__APPLE__)
  result.set("index_allocated_bytes", JsonValue(allocated_bytes));
#else
  result.set("index_allocated_bytes", JsonValue());
#endif
  return result;
}

JsonObject EnvironmentJson() {
  JsonObject result;
#if defined(__unix__) || defined(__APPLE__)
  struct utsname info {};
  if (::uname(&info) == 0) {
    result.set("platform", JsonStringValue(info.sysname));
    result.set("platform_release", JsonStringValue(info.release));
    result.set("architecture", JsonStringValue(info.machine));
  }
#elif defined(_WIN32)
  result.set("platform", JsonStringValue("Windows"));
#else
  result.set("platform", JsonStringValue("unknown"));
#endif
#if defined(__clang__)
  result.set("compiler", JsonStringValue("Clang " __clang_version__));
#elif defined(__GNUC__)
  result.set("compiler", JsonStringValue("GCC " __VERSION__));
#elif defined(_MSC_VER)
  result.set("compiler", JsonStringValue("MSVC " + std::to_string(_MSC_VER)));
#endif
#ifdef ZVEC_VERSION_STRING
  result.set("zvec_version", JsonStringValue(ZVEC_VERSION_STRING));
#endif
#ifdef NDEBUG
  result.set("ndebug", JsonValue(true));
#else
  result.set("ndebug", JsonValue(false));
#endif
  result.set("cplusplus", JsonValue(static_cast<uint64_t>(__cplusplus)));
  result.set("hardware_threads",
             JsonValue(std::thread::hardware_concurrency()));
  result.set("pointer_bits",
             JsonValue(static_cast<unsigned>(sizeof(void *) * 8)));
  const uint16_t endian_probe = 1;
  result.set("byte_order",
             JsonStringValue(*reinterpret_cast<const uint8_t *>(&endian_probe)
                                 ? "little"
                                 : "big"));
  return result;
}


JsonObject Report(const char *phase) {
  JsonObject result;
  result.set("schema_version", JsonValue(2));
  result.set("phase", JsonStringValue(phase));
  return result;
}

JsonArray BuildMetricNames() {
  JsonArray names;
  for (const char *name :
       {"insert_seconds", "seal_seconds", "build_seconds", "docs_per_second",
        "index_file_bytes", "index_allocated_bytes", "bytes_per_doc",
        "memory_before.rss_bytes", "memory_after.rss_bytes",
        "memory_after.peak_rss_bytes"})
    names.push(JsonStringValue(name));
  return names;
}

JsonObject FrequencyStats(const std::vector<uint64_t> &frequencies) {
  JsonObject result, percentiles;
  JsonArray counts;
  std::vector<uint64_t> observed;
  for (auto count : frequencies) {
    counts.push(JsonValue(count));
    if (count) observed.push_back(count);
  }
  result.set("term_doc_frequencies", counts);
  result.set("observed_cardinality",
             JsonValue(static_cast<uint64_t>(observed.size())));
  if (!observed.empty()) {
    std::sort(observed.begin(), observed.end());
    percentiles.set("min", JsonValue(observed.front()));
    percentiles.set("p50", JsonValue(observed[(observed.size() - 1) / 2]));
    percentiles.set("p95",
                    JsonValue(observed[(observed.size() - 1) * 95 / 100]));
    percentiles.set("max", JsonValue(observed.back()));
  }
  result.set("observed_term_df", percentiles);
  return result;
}

JsonObject QueryReport(const std::string &name, const JsonArray &inputs,
                       const Measurement &measurement, double warmup_seconds,
                       double measure_seconds, const JsonObject &before,
                       const JsonObject &after) {
  auto result = Report("query");
  result.set("case", JsonStringValue(name));
  result.set("query_fingerprint",
             JsonStringValue(InputFingerprint(name, inputs)));
  result.set("completed_queries", JsonValue(measurement.completed));
  result.set("elapsed_seconds", JsonValue(measurement.elapsed_seconds));
  result.set("mean_ns_per_query", JsonValue(measurement.elapsed_seconds * 1e9 /
                                            measurement.completed));
  result.set("qps",
             JsonValue(measurement.completed / measurement.elapsed_seconds));
  result.set("result_checksum", JsonStringValue(Hex(measurement.checksum)));
  result.set("warmup_seconds", JsonValue(warmup_seconds));
  result.set("measure_seconds", JsonValue(measure_seconds));
  result.set("memory_before", before);
  result.set("memory_after", after);
  result.set("environment", EnvironmentJson());
  JsonArray names;
  for (const char *key :
       {"mean_ns_per_query", "qps", "memory_before.rss_bytes",
        "memory_after.rss_bytes", "memory_after.peak_rss_bytes"})
    names.push(JsonStringValue(key));
  result.set("metric_names", names);
  return result;
}

}  // namespace zvec::benchmarks::inverted
