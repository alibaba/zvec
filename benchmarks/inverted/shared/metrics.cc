// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "metrics.h"
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include "ailego/utility/memory_helper.h"
#include "json.h"

namespace zvec::benchmarks::inverted {

namespace {

constexpr double kBytesPerMebibyte = 1024.0 * 1024.0;

}  // namespace

double Seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

JsonObject MemorySnapshot() {
  JsonObject result;
  size_t vsz = 0;
  size_t rss = 0;
  const bool available = ailego::MemoryHelper::SelfUsage(&vsz, &rss) && rss > 0;
  const auto peak = available ? ailego::MemoryHelper::SelfPeakRSS() : 0;
  result.set("rss_mib",
             available ? JsonValue(rss / kBytesPerMebibyte) : JsonValue());
  result.set("peak_rss_mib",
             peak ? JsonValue(peak / kBytesPerMebibyte) : JsonValue());
  return result;
}

JsonObject DiskUsage(const std::string &path) {
  namespace fs = std::filesystem;
  if (!ailego::File::IsDirectory(path)) {
    throw std::runtime_error("Cannot measure missing index directory: " + path);
  }
  uint64_t logical_bytes = 0;
  uint64_t file_count = 0;
  const auto directory = ailego::FileHelper::PathFromUtf8(path);
  for (const auto &entry : fs::recursive_directory_iterator(directory)) {
    if (!fs::is_regular_file(entry.symlink_status())) continue;
    logical_bytes += entry.file_size();
    ++file_count;
  }
  JsonObject result;
  result.set("index_file_mib", JsonValue(logical_bytes / kBytesPerMebibyte));
  result.set("index_file_count", JsonValue(file_count));
  return result;
}

JsonObject EnvironmentJson() {
  JsonObject result;

#if defined(__APPLE__)
  result.set("platform", JsonStringValue("macOS"));
#elif defined(__linux__)
  result.set("platform", JsonStringValue("Linux"));
#elif defined(_WIN32)
  result.set("platform", JsonStringValue("Windows"));
#else
  result.set("platform", JsonStringValue("unknown"));
#endif

  // Report the benchmark binary's target architecture, even under emulation.
#if defined(__aarch64__) || defined(_M_ARM64)
  result.set("architecture", JsonStringValue("arm64"));
#elif defined(__x86_64__) || defined(_M_X64)
  result.set("architecture", JsonStringValue("x86_64"));
#elif defined(__i386__) || defined(_M_IX86)
  result.set("architecture", JsonStringValue("x86"));
#elif defined(__arm__) || defined(_M_ARM)
  result.set("architecture", JsonStringValue("arm"));
#else
  result.set("architecture", JsonStringValue("unknown"));
#endif

#if defined(__clang__)
  result.set("compiler", JsonStringValue("Clang " __clang_version__));
#elif defined(__GNUC__)
  result.set("compiler", JsonStringValue("GCC " __VERSION__));
#elif defined(_MSC_VER)
  result.set("compiler", JsonStringValue("MSVC " + std::to_string(_MSC_VER)));
#endif

  result.set("cplusplus", JsonValue(static_cast<uint64_t>(__cplusplus)));
  result.set("hardware_threads",
             JsonValue(std::thread::hardware_concurrency()));
  return result;
}


JsonObject QueryReport(const std::string &name, double query_time_us) {
  JsonObject result;
  result.set("name", JsonStringValue(name));
  result.set("query_time_us", JsonValue(query_time_us));
  result.set("environment", EnvironmentJson());
  return result;
}

}  // namespace zvec::benchmarks::inverted
