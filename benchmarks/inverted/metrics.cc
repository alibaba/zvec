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
namespace {

const char *OpName(QueryOp op) {
  switch (op) {
    case QueryOp::kEq:
      return "eq";
    case QueryOp::kNe:
      return "ne";
    case QueryOp::kLt:
      return "lt";
    case QueryOp::kGe:
      return "ge";
    case QueryOp::kAny:
      return "any";
    case QueryOp::kAll:
      return "all";
    case QueryOp::kNotAny:
      return "not_any";
    case QueryOp::kNotAll:
      return "not_all";
    case QueryOp::kIsNull:
      return "is_null";
    case QueryOp::kIsNotNull:
      return "is_not_null";
    case QueryOp::kLengthEq:
      return "length_eq";
  }
  throw std::invalid_argument("Unknown query operation");
}

QueryOp ParseOp(const std::string &name) {
  for (auto op :
       {QueryOp::kEq, QueryOp::kNe, QueryOp::kLt, QueryOp::kGe, QueryOp::kAny,
        QueryOp::kAll, QueryOp::kNotAny, QueryOp::kNotAll, QueryOp::kIsNull,
        QueryOp::kIsNotNull, QueryOp::kLengthEq}) {
    if (name == OpName(op)) return op;
  }
  throw std::invalid_argument("Unknown query operation: " + name);
}

uint32_t UInt32(const JsonObject &object, const char *key) {
  const auto value = object[key];
  if (!value.is_integer() || value.as_integer() < 0 ||
      static_cast<uint64_t>(value.as_integer()) >
          std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string("Expected uint32 field: ") + key);
  }
  return static_cast<uint32_t>(value.as_integer());
}

bool Boolean(const JsonObject &object, const char *key) {
  const auto value = object[key];
  if (!value.is_boolean()) {
    throw std::invalid_argument(std::string("Expected boolean field: ") + key);
  }
  return value.as_bool();
}

std::string Hex(const std::string &bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

std::string Unhex(const std::string &text) {
  if (text.size() % 2 != 0) {
    throw std::invalid_argument("Odd-length query hex value");
  }
  const auto nibble = [](char c) -> uint8_t {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::invalid_argument("Invalid query hex value");
  };
  std::string bytes;
  bytes.reserve(text.size() / 2);
  for (size_t i = 0; i < text.size(); i += 2) {
    bytes.push_back(
        static_cast<char>((nibble(text[i]) << 4) | nibble(text[i + 1])));
  }
  return bytes;
}

JsonObject Histogram(const std::map<uint32_t, uint64_t> &histogram) {
  JsonObject result;
  for (const auto &[length, count] : histogram) {
    result.set(std::to_string(length).c_str(), JsonValue(count));
  }
  return result;
}

}  // namespace

JsonValue JsonStringValue(const std::string &value) {
  return JsonValue(ailego::JsonString(value).encode());
}

std::string JsonStringText(const JsonValue &value) {
  if (!value.is_string()) {
    throw std::invalid_argument("Expected JSON string");
  }
  return value.as_string().decode().as_stl_string();
}

JsonObject SpecJson(const DatasetSpec &spec) {
  JsonObject object;
  object.set("name", JsonStringValue(spec.name));
  object.set("type", JsonStringValue(spec.type));
  object.set("distribution", JsonStringValue(spec.distribution));
  object.set("cardinality", JsonValue(spec.cardinality));
  object.set("array_length", JsonValue(spec.array_length));
  object.set("supplemental", JsonValue(spec.supplemental));
  return object;
}

DatasetSpec ParseSpec(const JsonObject &object) {
  DatasetSpec spec;
  spec.name = JsonStringText(object["name"]);
  spec.type = JsonStringText(object["type"]);
  spec.distribution = JsonStringText(object["distribution"]);
  spec.cardinality = UInt32(object, "cardinality");
  spec.array_length = UInt32(object, "array_length");
  spec.supplemental = Boolean(object, "supplemental");
  if (spec.name.empty() || spec.type.empty() || spec.distribution.empty() ||
      spec.cardinality == 0) {
    throw std::invalid_argument("Incomplete dataset specification");
  }
  return spec;
}

JsonObject DatasetJson(const PreparedDataset &dataset) {
  auto object = SpecJson(dataset.spec);
  object.set("num_docs", JsonValue(dataset.num_docs));
  object.set("seed", JsonValue(dataset.seed));
  object.set("data_fingerprint", JsonStringValue(dataset.fingerprint));
  const auto &source = dataset.stats;
  JsonObject stats;
  stats.set("null_count", JsonValue(source.null_count));
  stats.set("empty_count", JsonValue(source.empty_count));
  stats.set("duplicate_count", JsonValue(source.duplicate_count));
  stats.set("raw_elements", JsonValue(source.raw_elements));
  stats.set("distinct_elements", JsonValue(source.distinct_elements));
  stats.set("observed_cardinality", JsonValue(source.observed_cardinality));
  stats.set("raw_length_histogram", Histogram(source.raw_length_histogram));
  stats.set("unique_length_histogram",
            Histogram(source.unique_length_histogram));
  JsonArray frequencies;
  std::vector<uint64_t> observed;
  for (auto count : source.term_doc_frequencies) {
    frequencies.push(JsonValue(count));
    if (count) observed.push_back(count);
  }
  stats.set("term_doc_frequencies", frequencies);
  JsonObject percentiles;
  if (!observed.empty()) {
    std::sort(observed.begin(), observed.end());
    percentiles.set("min", JsonValue(observed.front()));
    percentiles.set("p50", JsonValue(observed[(observed.size() - 1) / 2]));
    percentiles.set("p95",
                    JsonValue(observed[(observed.size() - 1) * 95 / 100]));
    percentiles.set("max", JsonValue(observed.back()));
  }
  stats.set("observed_term_df", percentiles);
  object.set("stats", stats);
  return object;
}

std::string QueryFingerprint(const QueryCase &query_case) {
  uint64_t hash = UINT64_C(14695981039346656037);
  const auto byte = [&hash](uint8_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
  };
  const auto number = [&byte](uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
      byte(static_cast<uint8_t>(value & 255));
      value >>= 8;
    }
  };
  const auto text = [&number, &byte](const std::string &value) {
    number(value.size());
    for (unsigned char c : value) byte(c);
  };
  text(query_case.name);
  number(query_case.from_documents);
  number(query_case.inputs.size());
  for (const auto &input : query_case.inputs) {
    text(OpName(input.op));
    number(input.length);
    number(input.values.size());
    for (const auto &value : input.values) text(value);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

JsonArray CasesJson(const std::vector<QueryCase> &cases) {
  JsonArray result;
  for (const auto &query_case : cases) {
    JsonObject object;
    object.set("name", JsonStringValue(query_case.name));
    object.set("from_documents", JsonValue(query_case.from_documents));
    object.set("query_fingerprint",
               JsonStringValue(QueryFingerprint(query_case)));
    JsonArray inputs;
    for (const auto &input : query_case.inputs) {
      JsonObject item;
      item.set("op", JsonStringValue(OpName(input.op)));
      item.set("length", JsonValue(input.length));
      JsonArray values;
      for (const auto &value : input.values) {
        values.push(JsonStringValue(Hex(value)));
      }
      item.set("values", values);
      inputs.push(item);
    }
    object.set("inputs", inputs);
    result.push(object);
  }
  return result;
}

std::vector<QueryCase> ParseCases(const JsonArray &array) {
  std::vector<QueryCase> result;
  for (size_t i = 0; i < array.size(); ++i) {
    if (!array[i].is_object()) {
      throw std::invalid_argument("Expected query case object");
    }
    const auto object = array[i].as_object();
    QueryCase query_case;
    query_case.name = JsonStringText(object["name"]);
    query_case.from_documents = Boolean(object, "from_documents");
    if (query_case.name.empty() || !object["inputs"].is_array()) {
      throw std::invalid_argument("Incomplete query case");
    }
    const auto inputs = object["inputs"].as_array();
    for (size_t j = 0; j < inputs.size(); ++j) {
      if (!inputs[j].is_object()) {
        throw std::invalid_argument("Expected query input object");
      }
      const auto item = inputs[j].as_object();
      QueryInput input{
          ParseOp(JsonStringText(item["op"])), {}, UInt32(item, "length")};
      if (!item["values"].is_array()) {
        throw std::invalid_argument("Expected query value array");
      }
      const auto values = item["values"].as_array();
      for (size_t k = 0; k < values.size(); ++k) {
        input.values.push_back(Unhex(JsonStringText(values[k])));
      }
      query_case.inputs.push_back(std::move(input));
    }
    if (query_case.inputs.empty()) {
      throw std::invalid_argument("Query case must not have an empty pool");
    }
    if (QueryFingerprint(query_case) !=
        JsonStringText(object["query_fingerprint"])) {
      throw std::invalid_argument("Query fingerprint mismatch: " +
                                  query_case.name);
    }
    result.push_back(std::move(query_case));
  }
  return result;
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

}  // namespace zvec::benchmarks::inverted
