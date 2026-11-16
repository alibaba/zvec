// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#if !defined(_WIN32)
#include <sys/resource.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <libproc.h>
#endif
#endif
#include <arrow/io/interfaces.h>
#include <arrow/util/thread_pool.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query.h>
#include <zvec/db/schema.h>
#include "ailego/utility/memory_helper.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// Keep Windows' DELETE macro out of the SDK's Operator enum. Windows
// declarations must also precede the process-status API header.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#endif

using namespace zvec;
using Clock = std::chrono::steady_clock;
std::vector<std::string> args;
std::string arg(const std::string &key, const std::string &fallback = "") {
  for (size_t i = 0; i + 1 < args.size(); ++i)
    if (args[i] == "--" + key) return args[i + 1];
  return fallback;
}
size_t num(const std::string &key, size_t fallback) {
  return std::stoull(arg(key, std::to_string(fallback)));
}
std::vector<std::string> split(std::string text) {
  std::vector<std::string> out;
  for (size_t at = 0;;) {
    auto end = text.find(',', at);
    out.push_back(text.substr(at, end - at));
    if (end == std::string::npos) break;
    at = end + 1;
  }
  return out;
}
void check(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error(what);
}
void status(const Status &s, const std::string &what) {
  check(s.ok(), what + ": " + s.message());
}
std::string pk(size_t n) {
  return "doc" + std::to_string(n);
}
std::string label(size_t n) {
  return (num("utf8", 0) ? "标签_" : "label_") + std::to_string(n);
}
std::string row_marker(size_t n) {
  return (num("utf8", 0) ? "行" : "row") + std::to_string(n) + ":";
}
bool wide() {
  return arg("profile") == "wide";
}
bool rich() {
  return arg("profile") == "rich" || wide();
}
bool null_field(size_t n, const std::string &name) {
  const auto setting = arg("null-percent");
  if (!setting.empty()) {
    size_t seed = name == "payload" ? 3
                  : name == "opt"   ? 7
                  : name == "tags"  ? 11
                                    : 0;
    if (name != "payload" && name != "opt" && name != "tags")
      for (unsigned char c : name) seed = (seed * 31 + c) % 100;
    return (n * 37 + seed) % 100 < std::stoull(setting);
  }
  return n % (name == "payload" ? 7 : name == "opt" ? 4 : 5) == 0;
}
bool empty_payload(size_t n) {
  return n % 100 < num("empty-string-percent", 0);
}
bool removed(size_t n) {
  return num("mutated", 0) && n % 17 == 0;
}
std::string payload(size_t n, bool apply_mutation = true) {
  if (empty_payload(n))
    return apply_mutation && num("mutated", 0) && n % 19 == 0 ? ":updated" : "";
  std::string s = row_marker(n);
  uint64_t x = (n + 1) * 0x9e3779b97f4a7c15ULL;
  size_t bytes = num("payload-bytes", 1024);
  s.reserve(s.size() + bytes + 8);
  for (size_t i = 0; i < bytes; ++i) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    s.push_back('a' + (x * 2685821657736338717ULL >> 32) % 26);
  }
  if (apply_mutation && num("mutated", 0) && n % 19 == 0) s += ":updated";
  return s;
}
std::string content(size_t n) {
  return "common term" + std::to_string(n % 256) + " group" +
         std::to_string(n % 16);
}
std::vector<int32_t> tags(size_t n) {
  if (n % 100 < num("empty-array-percent", 0)) return {};
  std::vector<int32_t> out;
  out.reserve(num("array-elements", 2));
  for (size_t i = 0; i < num("array-elements", 2); ++i)
    out.push_back(i == 0 ? n % 32 : i == 1 ? n / 32 % 32 : (n + i * 7) % 32);
  return out;
}
bool contains_tag(size_t n, size_t t) {
  if (n % 100 < num("empty-array-percent", 0)) return false;
  const auto len = num("array-elements", 2);
  if (len && n % 32 == t) return true;
  if (len > 1 && n / 32 % 32 == t) return true;
  if (len >= 34) return true;
  for (size_t i = 2; i < len; ++i)
    if ((n + i * 7) % 32 == t) return true;
  return false;
}
std::string binary_value(size_t n) {
  return std::string("bin\0", 4) + std::to_string(n);
}
template <typename T>
std::vector<T> numeric_array(size_t n) {
  std::vector<T> out;
  if (n % 100 < num("empty-array-percent", 0)) return out;
  out.reserve(num("array-elements", 2));
  for (size_t i = 0; i < num("array-elements", 2); ++i)
    out.push_back(static_cast<T>(std::is_same_v<T, bool> ? (n + i) % 2
                                                         : (n % 1024) * 2 + i));
  return out;
}
std::vector<std::string> string_array(size_t n, bool binary = false) {
  std::vector<std::string> out;
  if (n % 100 < num("empty-array-percent", 0)) return out;
  for (size_t i = 0; i < num("array-elements", 2); ++i)
    out.push_back((binary ? binary_value(n) : "标签_" + std::to_string(n)) +
                  ":" + std::to_string(i));
  return out;
}
size_t id(const Doc::Ptr &doc) {
  check(bool(doc), "nonnull returned document");
  auto &s = doc->pk_ref();
  size_t n = 0;
  check(s.size() > 3 && s.compare(0, 3, "doc") == 0, "document key prefix");
  auto r = std::from_chars(s.data() + 3, s.data() + s.size(), n);
  check(r.ec == std::errc{} && r.ptr == s.data() + s.size(),
        "numeric document key");
  return n;
}
struct ProcessUsage {
  double cpu_s{0};
  uint64_t minor_faults{0}, major_faults{0};
};
ProcessUsage process_usage() {
#if defined(_WIN32)
  FILETIME created{}, exited{}, kernel{}, user{};
  check(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user),
        "process CPU times");
  ULARGE_INTEGER k{}, u{};
  k.LowPart = kernel.dwLowDateTime;
  k.HighPart = kernel.dwHighDateTime;
  u.LowPart = user.dwLowDateTime;
  u.HighPart = user.dwHighDateTime;
  return {(k.QuadPart + u.QuadPart) * 1e-7, 0, 0};
#else
  rusage u{};
  check(getrusage(RUSAGE_SELF, &u) == 0, "process usage");
  return {u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 + u.ru_stime.tv_sec +
              u.ru_stime.tv_usec * 1e-6,
          uint64_t(u.ru_minflt), uint64_t(u.ru_majflt)};
#endif
}
uint64_t disk_read() {
#if defined(__APPLE__)
  rusage_info_v4 u{};
  check(proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                        reinterpret_cast<rusage_info_t *>(&u)) == 0,
        "process disk bytes");
  return u.ri_diskio_bytesread;
#elif defined(_WIN32)
  IO_COUNTERS u{};
  check(GetProcessIoCounters(GetCurrentProcess(), &u), "process read bytes");
  return u.ReadTransferCount;
#else
  std::ifstream in("/proc/self/io");
  std::string key;
  uint64_t value;
  while (in >> key >> value)
    if (key == "read_bytes:") return value;
  return 0;
#endif
}
const char *io_metric() {
#if defined(_WIN32)
  return "windows_process_read_transfer_count";
#elif defined(__APPLE__)
  return "darwin_process_diskio_bytesread";
#else
  return "linux_proc_self_io_read_bytes";
#endif
}
std::vector<std::string> fields() {
  std::vector<std::string> out{"key", "label", "bucket", "payload"};
  if (rich())
    for (auto f : {"opt", "tags", "weight", "flag", "content"})
      out.emplace_back(f);
  if (wide())
    for (auto f : {"u32",  "u64",  "f32",  "bin",   "ai64",  "au32", "au64",
                   "af32", "af64", "astr", "abool", "abin",  "ni32", "ni64",
                   "nu32", "nu64", "nf32", "nf64",  "nbool", "nstr", "nbin"})
      out.emplace_back(f);
  return out;
}
template <typename T>
void field(const Doc::Ptr &doc, const std::string &name, T wanted) {
  auto got = doc->get<T>(name);
  check(got.has_value() && got.value() == wanted,
        "field mismatch: " + name + " " + doc->pk_ref());
}
void validate(const Doc::Ptr &doc, size_t n,
              const std::optional<std::vector<std::string>> &projection) {
  const auto output = projection ? *projection : fields();
  size_t expected_fields = 0;
  for (const auto &f : output) {
    if ((f == "payload" && rich() && null_field(n, "payload")) ||
        (f == "opt" && null_field(n, "opt")) ||
        (f == "tags" && null_field(n, "tags")) ||
        (wide() && f != "key" && f != "label" && f != "bucket" &&
         f != "payload" && f != "opt" && f != "tags" && f != "weight" &&
         f != "flag" && f != "content" && null_field(n, f))) {
      // Both original Segment scalar conversion and the optimized direct
      // materializer omit NULL fields from Doc rather than inserting a value.
      check(!doc->has(f), "NULL field must be omitted: " + f);
      continue;
    }
    ++expected_fields;
    check(doc->has(f), "missing projected field: " + f);
    if (f == "key")
      field<int64_t>(doc, f, n);
    else if (f == "label")
      field<std::string>(doc, f, label(n));
    else if (f == "bucket")
      field<int32_t>(doc, f, n % 16);
    else if (f == "payload")
      field<std::string>(doc, f, payload(n));
    else if (f == "opt")
      field<int64_t>(doc, f, n % 128);
    else if (f == "tags")
      field<std::vector<int32_t>>(doc, f, tags(n));
    else if (f == "weight")
      field<double>(doc, f, n * 0.5);
    else if (f == "flag")
      field<bool>(doc, f, n % 2);
    else if (f == "content")
      field<std::string>(doc, f, content(n));
    else if (f == "u32")
      field<uint32_t>(doc, f, uint32_t(0x80000000u + n));
    else if (f == "u64")
      field<uint64_t>(doc, f, (uint64_t{1} << 48) + n);
    else if (f == "f32")
      field<float>(doc, f, float(n * .25));
    else if (f == "bin")
      field<std::string>(doc, f, binary_value(n));
    else if (f == "ai64")
      field<std::vector<int64_t>>(doc, f, numeric_array<int64_t>(n));
    else if (f == "au32")
      field<std::vector<uint32_t>>(doc, f, numeric_array<uint32_t>(n));
    else if (f == "au64")
      field<std::vector<uint64_t>>(doc, f, numeric_array<uint64_t>(n));
    else if (f == "af32")
      field<std::vector<float>>(doc, f, numeric_array<float>(n));
    else if (f == "af64")
      field<std::vector<double>>(doc, f, numeric_array<double>(n));
    else if (f == "astr")
      field<std::vector<std::string>>(doc, f, string_array(n));
    else if (f == "abool")
      field<std::vector<bool>>(doc, f, numeric_array<bool>(n));
    else if (f == "abin")
      field<std::vector<std::string>>(doc, f, string_array(n, true));
    else if (f == "ni32")
      field<int32_t>(doc, f, -int32_t(n % 1024));
    else if (f == "ni64")
      field<int64_t>(doc, f, -int64_t(n) * 3);
    else if (f == "nu32")
      field<uint32_t>(doc, f, uint32_t(0x80000000u + n));
    else if (f == "nu64")
      field<uint64_t>(doc, f, (uint64_t{1} << 48) + n);
    else if (f == "nf32")
      field<float>(doc, f, -float(n * .5));
    else if (f == "nf64")
      field<double>(doc, f, -double(n * .25));
    else if (f == "nbool")
      field<bool>(doc, f, bool(n % 2));
    else if (f == "nstr")
      field<std::string>(doc, f, empty_payload(n) ? "" : label(n));
    else if (f == "nbin")
      field<std::string>(doc, f, binary_value(n));
    else
      check(false, "unknown output field");
  }
  if (projection)
    check(doc->field_names().size() == expected_fields,
          "exact projection field count");
}
struct Request {
  SearchQuery query;
  bool fetch{false};
  std::vector<std::string> keys;
  std::vector<std::string> sorted_keys;
  std::vector<size_t> expected;
  size_t map_size{0};
  size_t wanted_count{0};
  std::optional<std::vector<std::string>> projection;
};
std::vector<Request> make_requests(const std::string &kind, size_t rows) {
  std::mt19937 rng(20261112);
  std::vector<Request> out;
  for (size_t q = 0; q < 64; ++q) {
    Request r;
    r.query.include_vector_ = false;
    r.projection = std::vector<std::string>{};
    size_t k = rng() % (rows - 2048);
    std::function<bool(size_t)> match;
    auto range = [&](size_t begin, size_t length) {
      return "key >= " + std::to_string(begin) + " AND key < " +
             std::to_string(begin + length);
    };
    if (kind == "eq_ids" || kind == "eq_key" || kind == "eq_payload") {
      r.query.filter_ = "key = " + std::to_string(k);
      match = [k](size_t n) { return n == k; };
      if (kind == "eq_key") r.projection = std::vector<std::string>{"key"};
      if (kind == "eq_payload")
        r.projection = std::vector<std::string>{"payload"};
    } else if (kind == "bucket_ids" || kind == "bucket_all") {
      size_t b = q % 16;
      r.query.filter_ = "bucket = " + std::to_string(b);
      match = [b](size_t n) { return n % 16 == b; };
      if (kind == "bucket_all") r.projection = std::nullopt;
    } else if (kind == "range128_all") {
      r.query.filter_ = range(k, 128);
      match = [k](size_t n) { return n >= k && n < k + 128; };
      r.projection = std::nullopt;
    } else if (kind == "in32_ids") {
      std::set<size_t> selected;
      std::string expression = "key IN (";
      for (size_t i = 0; i < 32; ++i) {
        size_t n = (k + i * 1543) % rows;
        selected.insert(n);
        expression += (i ? "," : "") + std::to_string(n);
      }
      r.query.filter_ = expression + ")";
      match = [selected](size_t n) { return selected.count(n); };
    } else if (kind == "or_ids") {
      r.query.filter_ =
          "(" + range(k, 16) + ") OR (" + range(k + 128, 16) + ")";
      match = [k](size_t n) {
        return (n >= k && n < k + 16) || (n >= k + 128 && n < k + 144);
      };
    } else if (kind == "not_in_ids") {
      r.query.filter_ = "(" + range(k, 64) + ") AND key NOT IN (" +
                        std::to_string(k) + "," + std::to_string(k + 1) + ")";
      match = [k](size_t n) { return n >= k + 2 && n < k + 64; };
    } else if (kind == "scan_payload_like") {
      r.query.filter_ = "payload LIKE '" + row_marker(k) + "%'";
      match = [k](size_t n) {
        return n == k && (!rich() || !null_field(n, "payload")) &&
               !empty_payload(n);
      };
    } else if (kind == "like_suffix") {
      size_t suffix = 10 + q % 90;
      r.query.filter_ = "(" + range(k, 1024) + ") AND label LIKE '%" +
                        std::to_string(suffix) + "'";
      match = [k, suffix](size_t n) {
        return n >= k && n < k + 1024 && n % 100 == suffix;
      };
    } else if (kind == "nullable_filter") {
      r.query.filter_ = "(" + range(k, 256) + ") AND opt IS NULL";
      match = [k](size_t n) {
        return n >= k && n < k + 256 && null_field(n, "opt");
      };
      r.projection = std::vector<std::string>{"opt", "payload"};
    } else if (kind == "array_filter") {
      size_t t = q % 32;
      r.query.filter_ = "(" + range(k, 256) + ") AND tags contain_any (" +
                        std::to_string(t) + ")";
      match = [k, t](size_t n) {
        return n >= k && n < k + 256 && !null_field(n, "tags") &&
               (contains_tag(n, t));
      };
      r.projection = std::vector<std::string>{"tags", "payload"};
    } else if (kind == "array_all_filter") {
      const size_t t = q % 32;
      r.query.filter_ = "tags contain_all (" + std::to_string(t) + "," +
                        std::to_string((t + 1) % 32) + ")";
      match = [t](size_t n) {
        return !null_field(n, "tags") && contains_tag(n, t) &&
               contains_tag(n, (t + 1) % 32);
      };
    } else if (kind == "array_i64_filter" || kind == "array_u32_filter" ||
               kind == "array_u64_filter" || kind == "array_float_filter" ||
               kind == "array_double_filter") {
      const std::string column = kind == "array_i64_filter"     ? "ai64"
                                 : kind == "array_u32_filter"   ? "au32"
                                 : kind == "array_u64_filter"   ? "au64"
                                 : kind == "array_float_filter" ? "af32"
                                                                : "af64";
      const size_t target = q * 17 % 2048;
      r.query.filter_ =
          column + " contain_any (" + std::to_string(target) + ")";
      match = [column, target](size_t n) {
        const auto base = (n % 1024) * 2;
        return !null_field(n, column) &&
               n % 100 >= num("empty-array-percent", 0) && target >= base &&
               target < base + num("array-elements", 2);
      };
    } else if (kind == "array_bool_filter") {
      r.query.filter_ = "abool contain_any (true)";
      match = [](size_t n) {
        return !null_field(n, "abool") &&
               n % 100 >= num("empty-array-percent", 0) &&
               (num("array-elements", 2) > 1 ||
                (num("array-elements", 2) == 1 && n % 2));
      };
    } else if (kind == "array_string_filter") {
      r.query.filter_ = "astr contain_any ('标签_" + std::to_string(k) + ":0')";
      match = [k](size_t n) {
        return n == k && !null_field(n, "astr") &&
               n % 100 >= num("empty-array-percent", 0) &&
               num("array-elements", 2) > 0;
      };
    } else if (kind == "scan_weight") {
      r.query.filter_ = "weight >= " + std::to_string(k * 0.5) +
                        " AND weight < " + std::to_string((k + 16) * 0.5);
      match = [k](size_t n) { return n >= k && n < k + 16; };
      r.projection = std::vector<std::string>{"weight"};
    } else if (kind == "i32_filter" || kind == "i64_filter" ||
               kind == "u32_filter" || kind == "u64_filter" ||
               kind == "float_filter" || kind == "double_filter") {
      if (kind == "i32_filter") {
        r.query.filter_ = "ni32 = " + std::to_string(-int32_t(k % 1024));
        match = [k](size_t n) {
          return n % 1024 == k % 1024 && !null_field(n, "ni32");
        };
      } else {
        const std::string column = kind == "i64_filter"     ? "ni64"
                                   : kind == "u32_filter"   ? "nu32"
                                   : kind == "u64_filter"   ? "nu64"
                                   : kind == "float_filter" ? "nf32"
                                                            : "nf64";
        if (kind == "u32_filter" || kind == "u64_filter") {
          const uint64_t base =
              kind == "u32_filter" ? uint64_t{0x80000000u} : uint64_t{1} << 48;
          r.query.filter_ = column + " >= " + std::to_string(base + k) +
                            " AND " + column + " < " +
                            std::to_string(base + k + 16);
        } else if (kind == "i64_filter") {
          r.query.filter_ = column + " <= " + std::to_string(-int64_t(k) * 3) +
                            " AND " + column + " > " +
                            std::to_string(-int64_t(k + 16) * 3);
        } else {
          const double factor = kind == "i64_filter"     ? -3.
                                : kind == "float_filter" ? -.5
                                                         : -.25;
          r.query.filter_ = column + " <= " + std::to_string(k * factor) +
                            " AND " + column + " > " +
                            std::to_string((k + 16) * factor);
        }
        match = [k, column](size_t n) {
          return n >= k && n < k + 16 && !null_field(n, column);
        };
      }
    } else if (kind == "bool_filter") {
      r.query.filter_ = "nbool = true AND key < " + std::to_string(rows / 2);
      match = [rows](size_t n) {
        return n < rows / 2 && n % 2 && !null_field(n, "nbool");
      };
    } else if (kind == "string_filter") {
      r.query.filter_ = "nstr = '" + label(k) + "'";
      match = [k](size_t n) {
        return n == k && !empty_payload(n) && !null_field(n, "nstr");
      };
    } else if (kind == "like_contains") {
      const auto text = std::to_string(q % 100);
      r.query.filter_ = "label LIKE '%" + text + "%'";
      match = [text](size_t n) {
        return std::to_string(n).find(text) != std::string::npos;
      };
    } else if (kind == "range_select_ids" || kind == "range_select_all" ||
               kind == "scan_select") {
      size_t limit = rows * num("selectivity-percent", 10) / 100;
      if (kind == "scan_select")
        r.query.filter_ = "weight < " + std::to_string(limit * .5);
      else
        r.query.filter_ = "key < " + std::to_string(limit);
      match = [limit](size_t n) { return n < limit; };
      if (kind == "range_select_all") r.projection = std::nullopt;
    } else if (kind == "not_null") {
      r.query.filter_ = "opt IS NOT NULL";
      match = [](size_t n) { return !null_field(n, "opt"); };
    } else if (kind == "payload_eq_empty") {
      r.query.filter_ = "payload = ''";
      match = [](size_t n) {
        return (!rich() || !null_field(n, "payload")) && empty_payload(n) &&
               !(num("mutated", 0) && n % 19 == 0);
      };
    } else if (kind == "fts_and" || kind == "fts_or" || kind == "fts_phrase" ||
               kind == "fts_or_filter") {
      size_t term = q % 256;
      r.query.target_.field_name_ = "content";
      FtsClause fts;
      const bool either = kind == "fts_or" || kind == "fts_or_filter";
      if (kind == "fts_and")
        fts.query_string_ = "common AND term" + std::to_string(term);
      else if (kind == "fts_phrase")
        fts.query_string_ = "\"common term" + std::to_string(term) + "\"";
      else
        fts.query_string_ = "term" + std::to_string(term) + " OR term" +
                            std::to_string((term + 1) % 256);
      r.query.target_.clause_ = fts;
      const bool filtered = kind == "fts_or_filter";
      if (filtered) r.query.filter_ = "key < " + std::to_string(rows / 2);
      match = [term, either, filtered, rows](size_t n) {
        return (n % 256 == term || (either && n % 256 == (term + 1) % 256)) &&
               (!filtered || n < rows / 2);
      };
      r.projection = std::vector<std::string>{"content", "payload"};
    } else if (kind == "fts_term" || kind == "fts_filter" ||
               kind == "fts_topk" || kind == "fts_topk_filter") {
      size_t term = q % 256;
      r.query.target_.field_name_ = "content";
      FtsClause fts;
      fts.query_string_ = "term" + std::to_string(term);
      r.query.target_.clause_ = fts;
      const bool filtered = kind == "fts_filter" || kind == "fts_topk_filter";
      if (filtered) r.query.filter_ = "key < " + std::to_string(rows / 2);
      match = [term, filtered, rows](size_t n) {
        return n % 256 == term && (!filtered || n < rows / 2);
      };
      r.projection = std::vector<std::string>{"content", "payload"};
    } else if (kind.rfind("fetch", 0) == 0) {
      r.fetch = true;
      size_t batch = num("fetch-size", 32);
      if (kind == "fetch1_all" || kind == "fetch1_payload" ||
          kind == "fetch1_nullable")
        batch = 1;
      if (kind == "fetch8_key") batch = 8;
      if (kind == "fetch128_payload") batch = 128;
      r.projection = std::vector<std::string>{"payload"};
      if (kind == "fetch8_key" || kind == "fetch_mixed")
        r.projection = std::vector<std::string>{"key"};
      if (kind == "fetch1_all" || kind == "fetch_all")
        r.projection = std::nullopt;
      if (kind == "fetch_projected") {
        const auto selected = arg("projection", "key,payload");
        if (selected == "none")
          r.projection = std::vector<std::string>{};
        else if (selected == "all")
          r.projection = std::nullopt;
        else
          r.projection = split(selected);
      }
      if (kind == "fetch32_null_array")
        r.projection = std::vector<std::string>{"opt", "tags", "payload"};
      if (kind == "fetch1_nullable")
        r.projection = std::vector<std::string>{"opt", "payload"};
      if (kind == "fetch32_hot" && q % 5 != 0) k %= rows / 32;
      std::set<size_t> selected;
      std::set<std::string> unique;
      for (size_t i = 0; i < batch; ++i) {
        size_t n = (k + i * (kind == "fetch32_contiguous" ? 1 : 1543)) % rows;
        if (kind == "fetch_edges") {
          const size_t edges[] = {0,           1,           4095 % rows,
                                  4096 % rows, 4097 % rows, rows / 2,
                                  rows - 2,    rows - 1};
          n = edges[(i + q) % 8];
        }
        if (kind == "fetch32_hot" && q % 5 != 0) n %= rows / 32;
        if (kind == "fetch_mixed" && i % 4 == 0)
          n = std::numeric_limits<size_t>::max() - i;
        if (kind == "fetch_mixed" && i % 4 == 1) n = k;
        unique.insert(pk(n));
        r.keys.push_back(pk(n));
        if (n < rows && !removed(n)) selected.insert(n);
      }
      r.map_size = unique.size();
      r.sorted_keys.assign(unique.begin(), unique.end());
      r.expected.assign(selected.begin(), selected.end());
    } else
      check(false, "unknown scenario " + kind);
    if (!r.fetch) {
      check(bool(match), "query predicate");
      for (size_t n = 0; n < rows; ++n)
        if (!removed(n) && match(n)) r.expected.push_back(n);
      r.query.topk_ = std::max<size_t>(1, r.expected.size());
      r.query.output_fields_ = r.projection;
    }
    r.wanted_count = r.expected.size();
    if (kind == "fts_topk" || kind == "fts_topk_filter") {
      r.query.topk_ = std::max<size_t>(1, num("topk", 10));
      r.wanted_count = std::min(r.wanted_count, size_t(r.query.topk_));
    }
    out.push_back(std::move(r));
  }
  return out;
}
struct Scratch {
  std::vector<uint32_t> seen;
  uint32_t generation{0};
  explicit Scratch(size_t rows) : seen(rows, 0) {}
};
size_t execute(Collection &col, const Request &r, bool full, Scratch &scratch,
               std::vector<Doc::Ptr> *snapshot = nullptr) {
  ++scratch.generation;
  size_t count = 0;
  auto observe = [&](const Doc::Ptr &doc) {
    size_t n = id(doc);
    check(n < scratch.seen.size() &&
              std::binary_search(r.expected.begin(), r.expected.end(), n),
          "unexpected result " + doc->pk_ref());
    check(scratch.seen[n] != scratch.generation, "duplicate result");
    scratch.seen[n] = scratch.generation;
    ++count;
    if (full) validate(doc, n, r.projection);
    if (snapshot) snapshot->push_back(doc);
  };
  if (r.fetch) {
    auto got = col.fetch(r.keys, r.projection, false);
    check(got.has_value(), got.has_value() ? "" : got.error().message());
    check(got->size() == r.map_size, "fetch map size");
    for (auto &item : *got) {
      check(std::binary_search(r.sorted_keys.begin(), r.sorted_keys.end(),
                               item.first),
            "unexpected map key");
      if (item.second) {
        check(item.first == item.second->pk_ref(), "fetch key/value identity");
        observe(item.second);
      } else {
        auto n = std::stoull(item.first.substr(3));
        check(n >= scratch.seen.size() || removed(n), "unexpected null result");
      }
    }
  } else {
    auto got = col.query(r.query);
    check(got.has_value(), got.has_value() ? "" : got.error().message());
    for (auto &doc : *got) observe(doc);
  }
  check(count == r.wanted_count, "exact result count for " + r.query.filter_);
  return count;
}
CollectionSchema schema() {
  CollectionSchema s("scalar_extended");
  auto add = [&](const std::string &n, DataType t, bool nullable,
                 IndexParams::Ptr ix = nullptr) {
    status(s.add_field(std::make_shared<FieldSchema>(n, t, nullable, ix)),
           "schema " + n);
  };
  add("key", DataType::INT64, false, std::make_shared<InvertIndexParams>(true));
  add("label", DataType::STRING, false,
      std::make_shared<InvertIndexParams>(false, true));
  add("bucket", DataType::INT32, false,
      std::make_shared<InvertIndexParams>(true));
  add("payload", DataType::STRING, rich());
  if (rich()) {
    add("opt", DataType::INT64, true,
        std::make_shared<InvertIndexParams>(true));
    add("tags", DataType::ARRAY_INT32, true,
        std::make_shared<InvertIndexParams>());
    add("weight", DataType::DOUBLE, false);
    add("flag", DataType::BOOL, false, std::make_shared<InvertIndexParams>());
    add("content", DataType::STRING, false, std::make_shared<FtsIndexParams>());
  }
  if (wide()) {
    add("u32", DataType::UINT32, true);
    add("u64", DataType::UINT64, true);
    add("f32", DataType::FLOAT, true);
    add("bin", DataType::BINARY, true);
    add("ai64", DataType::ARRAY_INT64, true);
    add("au32", DataType::ARRAY_UINT32, true);
    add("au64", DataType::ARRAY_UINT64, true);
    add("af32", DataType::ARRAY_FLOAT, true);
    add("af64", DataType::ARRAY_DOUBLE, true);
    add("astr", DataType::ARRAY_STRING, true);
    add("abool", DataType::ARRAY_BOOL, true);
    add("abin", DataType::ARRAY_BINARY, true);
    add("ni32", DataType::INT32, true);
    add("ni64", DataType::INT64, true);
    add("nu32", DataType::UINT32, true);
    add("nu64", DataType::UINT64, true);
    add("nf32", DataType::FLOAT, true);
    add("nf64", DataType::DOUBLE, true);
    add("nbool", DataType::BOOL, true);
    add("nstr", DataType::STRING, true);
    add("nbin", DataType::BINARY, true);
  }
  return s;
}
Doc make_doc(size_t n) {
  Doc d;
  d.set_pk(pk(n));
  d.set<int64_t>("key", n);
  d.set<std::string>("label", label(n));
  d.set<int32_t>("bucket", n % 16);
  if (!rich() || !null_field(n, "payload"))
    d.set<std::string>("payload", payload(n, false));
  if (rich()) {
    if (!null_field(n, "opt")) d.set<int64_t>("opt", n % 128);
    if (!null_field(n, "tags")) d.set<std::vector<int32_t>>("tags", tags(n));
    d.set<double>("weight", n * .5);
    d.set<bool>("flag", n % 2);
    d.set<std::string>("content", content(n));
  }
  if (wide()) {
    if (!null_field(n, "u32"))
      d.set<uint32_t>("u32", uint32_t(0x80000000u + n));
    if (!null_field(n, "u64")) d.set<uint64_t>("u64", (uint64_t{1} << 48) + n);
    if (!null_field(n, "f32")) d.set<float>("f32", float(n * .25));
    if (!null_field(n, "bin")) d.set<std::string>("bin", binary_value(n));
    if (!null_field(n, "ai64"))
      d.set<std::vector<int64_t>>("ai64", numeric_array<int64_t>(n));
    if (!null_field(n, "au32"))
      d.set<std::vector<uint32_t>>("au32", numeric_array<uint32_t>(n));
    if (!null_field(n, "au64"))
      d.set<std::vector<uint64_t>>("au64", numeric_array<uint64_t>(n));
    if (!null_field(n, "af32"))
      d.set<std::vector<float>>("af32", numeric_array<float>(n));
    if (!null_field(n, "af64"))
      d.set<std::vector<double>>("af64", numeric_array<double>(n));
    if (!null_field(n, "astr"))
      d.set<std::vector<std::string>>("astr", string_array(n));
    if (!null_field(n, "abool"))
      d.set<std::vector<bool>>("abool", numeric_array<bool>(n));
    if (!null_field(n, "abin"))
      d.set<std::vector<std::string>>("abin", string_array(n, true));
    if (!null_field(n, "ni32")) d.set<int32_t>("ni32", -int32_t(n % 1024));
    if (!null_field(n, "ni64")) d.set<int64_t>("ni64", -int64_t(n) * 3);
    if (!null_field(n, "nu32"))
      d.set<uint32_t>("nu32", uint32_t(0x80000000u + n));
    if (!null_field(n, "nu64"))
      d.set<uint64_t>("nu64", (uint64_t{1} << 48) + n);
    if (!null_field(n, "nf32")) d.set<float>("nf32", -float(n * .5));
    if (!null_field(n, "nf64")) d.set<double>("nf64", -double(n * .25));
    if (!null_field(n, "nbool")) d.set<bool>("nbool", bool(n % 2));
    if (!null_field(n, "nstr"))
      d.set<std::string>("nstr", empty_payload(n) ? "" : label(n));
    if (!null_field(n, "nbin")) d.set<std::string>("nbin", binary_value(n));
  }
  return d;
}
void build(const std::string &path, bool mmap, size_t rows) {
  check(!std::filesystem::exists(path), "fresh data path required");
  auto got = Collection::CreateAndOpen(
      path, schema(),
      CollectionOptions{false, mmap, uint32_t(num("segment-mib", 128) << 20)});
  check(got.has_value(), got.has_value() ? "" : got.error().message());
  auto col = std::move(got).value();
  for (size_t begin = 0; begin < rows; begin += 256) {
    std::vector<Doc> docs;
    for (size_t n = begin; n < std::min(rows, begin + 256); ++n) {
      auto d = make_doc(n);
      docs.push_back(std::move(d));
    }
    auto inserted = col->insert(docs);
    check(inserted.has_value(),
          inserted.has_value() ? "" : inserted.error().message());
    for (auto &s : *inserted) status(s, "insert row");
    if (num("seal-rows", 0) &&
        std::min(rows, begin + 256) % num("seal-rows", 0) == 0) {
      IteratorOptions options;
      options.output_fields_ = std::vector<std::string>{};
      options.include_vector_ = false;
      auto snapshot = col->create_iterator(options);
      check(snapshot.has_value(),
            snapshot.has_value() ? "" : snapshot.error().message());
      snapshot.value()->close();
      snapshot.value().reset();
      std::cout << "SEALED " << std::min(rows, begin + 256) << std::endl;
    }
    if (begin % 8192 == 0) std::cout << "INSERTED " << begin << std::endl;
  }
  if (num("mutated", 0)) {
    std::vector<std::string> deleted;
    for (size_t n = 0; n < rows; n += 17) deleted.push_back(pk(n));
    auto del = col->delete_(deleted);
    check(del.has_value(), "delete rows");
    for (const auto &s : *del) status(s, "delete row");
    std::vector<Doc> updated;
    for (size_t n = 0; n < rows; n += 19)
      if (!removed(n)) {
        Doc d;
        d.set_pk(pk(n));
        if (rich() && null_field(n, "payload"))
          d.set_null("payload");
        else
          d.set<std::string>("payload", payload(n));
        updated.push_back(std::move(d));
      }
    auto up = col->update(updated);
    check(up.has_value(), "update rows");
    for (const auto &s : *up) status(s, "update row");
  }
  status(col->flush(), "flush");
  if (num("optimize", 1)) status(col->optimize(), "optimize");
  check(col->stats().value().doc_count ==
            rows - (num("mutated", 0) ? (rows + 16) / 17 : 0),
        "build document count");
  status(col->close(), "close builder");
  std::cout << "BUILD_COMPLETE " << rows << std::endl;
}
int main(int argc, char **argv) try {
  args.assign(argv + 1, argv + argc);
  if (argc == 1 ||
      std::find(args.begin(), args.end(), "--help") != args.end()) {
    std::cout
        << "Non-vector BP/mmap benchmark. Arguments are --name value pairs.\n"
           "--phase build|validate|query|mixed|lifecycle --path DATA --mmap "
           "0|1\n"
           "--profile scalar|rich|wide --rows N (>2048) --payload-bytes N\n"
           "--array-elements N --null-percent 0..100 --empty-array-percent "
           "0..100\n"
           "--empty-string-percent 0..100 --selectivity-percent 0..100\n"
           "--workers N --budget-mib N --segment-mib N --optimize 0|1 "
           "--mutated 0|1\n"
           "--cases comma,separated --fetch-size N --projection "
           "comma,separated\n"
           "--warmups N --seconds N --first-only 1 (one case, no warmups, "
           "fresh process)\n"
           "--maintenance none|flush|optimize --maintenance-every N "
           "--writer-pause-us N\n"
           "--output FILE (fresh output path)\n";
    return 0;
  }
  check(arg("profile") == "scalar" || rich(), "known profile");
  for (const auto &setting : {"null-percent", "empty-array-percent",
                              "empty-string-percent", "selectivity-percent"})
    check(num(setting, 0) <= 100, "percentage range");
  check(num("rows", 16384) > 2048 && num("workers", 1) > 0 &&
            num("maintenance-every", 32) > 0,
        "positive workload sizes");

  size_t rows = num("rows", 16384), workers = num("workers", 1),
         budget = num("budget-mib", 4096);
  bool mmap = num("mmap", 0);
  auto path = arg("path");
  GlobalConfig::ConfigData cfg;
  cfg.memory_limit_bytes = budget << 20;
  cfg.query_thread_count = workers;
  cfg.optimize_thread_count = 1;
  cfg.query_thread_binding = false;
  cfg.optimize_thread_binding = false;
  cfg.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      GlobalConfig::LogLevel::kWarn);
  status(GlobalConfig::Instance().initialize(cfg), "global config");
  if (arg("phase") == "build") {
    if (!arg("output").empty())
      check(!std::filesystem::exists(arg("output")), "fresh build result");
    build(path, mmap, rows);
    if (!arg("output").empty()) {
      std::ofstream output(arg("output"));
      check(bool(output), "build result file");
      output << "{\"complete\":true,\"phase\":\"build\",\"rows\":" << rows
             << ",\"mmap\":" << mmap << "}\n";
    }
    return 0;
  }
  const bool mixed = arg("phase") == "mixed";
  const bool lifecycle = mixed || arg("phase") == "lifecycle";
  check(!mixed || rows > 4096,
        "mixed needs more than 4096 rows for a stable read partition");
  if (lifecycle) {
    check(!num("mutated", 0), "lifecycle uses a fresh unmutated dataset");
    build(path, mmap, rows);
  }
  auto open_started = Clock::now();
  auto got = Collection::Open(path, CollectionOptions{!lifecycle, mmap});
  check(got.has_value(), got.has_value() ? "" : got.error().message());
  auto col = std::move(got).value();
  const double open_us =
      std::chrono::duration<double, std::micro>(Clock::now() - open_started)
          .count();
  check(col->options()->enable_mmap_ == mmap, "persisted storage mode");
  if (arg("phase") == "validate") {
    check(!std::filesystem::exists(arg("output")), "fresh validation output");
    IteratorOptions options;
    options.include_vector_ = false;
    options.output_fields_ = fields();
    auto opened = col->create_iterator(options);
    check(opened.has_value(), "full validation iterator");
    auto iterator = std::move(opened).value();
    std::vector<uint8_t> seen(rows, 0);
    size_t count = 0;
    uint64_t fingerprint = 0;
    while (true) {
      auto next = iterator->next();
      check(next.has_value(), next.has_value() ? "" : next.error().message());
      if (!*next) break;
      const auto n = id(*next);
      check(n < rows && !removed(n) && !seen[n],
            "complete unique logical dataset");
      seen[n] = 1;
      ++count;
      validate(*next, n, std::nullopt);
      fingerprint ^= (uint64_t(n) + 1) * 0x9e3779b97f4a7c15ULL;
    }
    iterator->close();
    iterator.reset();
    const auto expected = rows - (num("mutated", 0) ? (rows + 16) / 17 : 0);
    check(count == expected, "every live document checked");
    std::ofstream result(arg("output"));
    check(bool(result), "validation output");
    result << "{\"complete\":true,\"validated\":true,\"documents\":" << count
           << ",\"fingerprint\":" << std::quoted(std::to_string(fingerprint))
           << "}\n";
    status(col->close(), "close full validation");
    std::cout << "VALIDATE_COMPLETE " << count << std::endl;
    return 0;
  }
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t warmups = num("warmups", 64);
  const bool first_only = num("first-only", 0);
  const bool first_pass = first_only || num("first-pass", 0),
             full_timed = false;
  check(!first_only || (warmups == 0 && split(arg("cases")).size() == 1),
        "cold first-only requires one case and zero warmups");
  const double minimum_seconds = std::stod(arg("seconds", "1"));
  check(!std::filesystem::exists(arg("output")), "fresh result file required");
  std::ofstream out(arg("output"));
  check(bool(out), "output file");
  out << std::setprecision(17);
  out << "{\"rows\":" << rows << ",\"workers\":" << workers
      << ",\"mmap\":" << mmap << ",\"budget_bytes\":" << (budget << 20)
      << ",\"capacity\":" << pool.capacity()
      << ",\"arrow_cpu_capacity\":" << arrow::GetCpuThreadPoolCapacity()
      << ",\"arrow_io_capacity\":" << arrow::io::GetIOThreadPoolCapacity()
      << ",\"first_only\":" << first_only
      << ",\"request_scheduler\":\"global-prefix64-v1\""
      << ",\"io_metric\":" << std::quoted(io_metric())
      << ",\"open_us\":" << open_us << ",\"cases\":[";
  bool first = true;
  std::map<size_t, std::string> writer_values;
  size_t writer_cycles = 0;
  double maintenance_us = 0;
  auto write_status = [&](const auto &result, const std::string &what) {
    check(result.has_value(),
          result.has_value() ? "" : what + ": " + result.error().message());
    for (const auto &entry : *result) status(entry, what);
  };
  auto restrict_stable = [&](std::vector<Request> &requests,
                             size_t stable_rows) {
    if (!mixed) return;
    for (auto &r : requests)
      if (!r.fetch) {
        auto bound = "key < " + std::to_string(stable_rows);
        r.query.filter_ = r.query.filter_.empty()
                              ? bound
                              : "(" + r.query.filter_ + ") AND " + bound;
      }
  };
  for (auto &kind : split(arg("cases"))) {
    const size_t read_rows = mixed ? rows / 2 : rows;
    auto requests = make_requests(kind, read_rows);
    restrict_stable(requests, read_rows);
    Scratch warm(read_rows);
    uint64_t request_fingerprint = 1469598103934665603ULL;
    auto fingerprint = [&](const std::string &s) {
      for (unsigned char c : s) {
        request_fingerprint ^= c;
        request_fingerprint *= 1099511628211ULL;
      }
      request_fingerprint ^= 255;
      request_fingerprint *= 1099511628211ULL;
    };
    for (const auto &r : requests) {
      fingerprint(r.query.filter_);
      fingerprint(std::to_string(r.query.topk_));
      fingerprint(std::to_string(r.wanted_count));
      for (auto &key : r.keys) fingerprint(key);
      for (auto n : r.expected) fingerprint(std::to_string(n));
      if (r.projection)
        for (auto &f : *r.projection) fingerprint(f);
      else
        fingerprint("*");
    }
    for (size_t i = 0; i < warmups; ++i)
      execute(*col, requests[i % 64], true, warm);
    auto before = pool.stats();
    auto reads = disk_read();
    ProcessUsage cb{}, ca{};
    std::atomic<size_t> ready{0};
    alignas(128) std::atomic<size_t> next_request{workers};
    Clock::time_point started;
    std::atomic<bool> go{false}, stop_writer{false}, readers_done{false};
    std::atomic<size_t> overlap_cycles{0};
    std::exception_ptr writer_error;
    std::thread writer;
    if (mixed) {
      check(split(arg("cases")).size() == 1,
            "mixed runs use one read scenario per fresh dataset");
      writer = std::thread([&] {
        try {
          while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
          while (!stop_writer.load(std::memory_order_acquire)) {
            size_t n = read_rows + writer_cycles % (rows - read_rows);
            std::string value = "writer:" + std::to_string(writer_cycles);
            Doc update;
            update.set_pk(pk(n));
            update.set<std::string>("payload", value);
            std::vector<Doc> updates;
            updates.push_back(std::move(update));
            write_status(col->update(updates), "concurrent update");
            writer_values[n] = value;
            std::vector<Doc> inserts;
            inserts.push_back(make_doc(rows + writer_cycles));
            write_status(col->insert(inserts), "concurrent insert");
            write_status(col->delete_({pk(rows + writer_cycles)}),
                         "concurrent delete");
            ++writer_cycles;
            if (!readers_done.load(std::memory_order_acquire))
              overlap_cycles.fetch_add(1);
            if (writer_cycles % num("maintenance-every", 32) == 0 &&
                arg("maintenance", "none") != "none") {
              auto started = Clock::now();
              status(col->flush(), "concurrent flush");
              if (arg("maintenance") == "optimize")
                status(col->optimize(), "concurrent optimize");
              maintenance_us += std::chrono::duration<double, std::micro>(
                                    Clock::now() - started)
                                    .count();
            }
            if (num("writer-pause-us", 100))
              std::this_thread::sleep_for(
                  std::chrono::microseconds(num("writer-pause-us", 100)));
          }
        } catch (...) {
          writer_error = std::current_exception();
        }
      });
    }
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> latencies(workers);
    std::vector<std::vector<Doc::Ptr>> cold_results(workers);
    std::vector<size_t> count(workers, 0), results(workers, 0);
    std::vector<size_t> expected_results(workers, 0);
    std::vector<std::array<size_t, 64>> request_counts(
        workers, std::array<size_t, 64>{});
    std::vector<std::exception_ptr> errors(workers);
    for (size_t w = 0; w < workers; ++w)
      threads.emplace_back([&, w] {
        bool announced = false;
        try {
          Scratch scratch(read_rows);
          latencies[w].reserve(1 << 18);
          ready.fetch_add(1);
          announced = true;
          while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
          while (true) {
            // Every worker executes its initial request. Remaining requests
            // form one shared deterministic sequence, so even an overloaded
            // run covers all64 variants without requiring 64 per worker.
            // Check the deadline before claiming a ticket. Every claimed
            // timed request must run, even if its worker is descheduled at
            // the deadline; otherwise the global prefix could have holes.
            if (!first_only && count[w] != 0 &&
                next_request.load(std::memory_order_relaxed) >=
                    requests.size() &&
                (first_pass ||
                 std::chrono::duration<double>(Clock::now() - started)
                         .count() >= minimum_seconds))
              break;
            const size_t ticket =
                first_only ? w * 17
                : count[w] == 0
                    ? w
                    : next_request.fetch_add(1, std::memory_order_relaxed);
            if (!first_only && first_pass && ticket >= requests.size()) break;
            const size_t index = ticket % requests.size();
            auto t = Clock::now();
            results[w] += execute(*col, requests[index], full_timed, scratch,
                                  first_only ? &cold_results[w] : nullptr);
            expected_results[w] += requests[index].wanted_count;
            ++request_counts[w][index];
            latencies[w].push_back(
                std::chrono::duration<double, std::micro>(Clock::now() - t)
                    .count());
            ++count[w];
            if (first_only) break;
          }
        } catch (...) {
          errors[w] = std::current_exception();
          if (!announced) ready.fetch_add(1);
        }
      });
    while (ready.load() != workers) std::this_thread::yield();
    cb = process_usage();
    started = Clock::now();
    go.store(true, std::memory_order_release);
    for (auto &t : threads) t.join();
    const double elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    ca = process_usage();
    readers_done.store(true, std::memory_order_release);
    stop_writer.store(true, std::memory_order_release);
    if (writer.joinable()) writer.join();
    const double writer_elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    if (writer_error) std::rethrow_exception(writer_error);
    check(!mixed || overlap_cycles.load() > 0,
          "writer made progress during the read interval");
    for (auto &e : errors)
      if (e) std::rethrow_exception(e);
    auto after = pool.stats();
    size_t total = std::accumulate(count.begin(), count.end(), size_t{});
    size_t expected_per_cycle = 0;
    for (auto &r : requests) expected_per_cycle += r.wanted_count;
    std::array<size_t, 64> histogram{};
    for (size_t w = 0; w < workers; ++w) {
      if (first_only)
        check(count[w] == 1 && results[w] == requests[w * 17 % 64].wanted_count,
              "cold result count");
      else
        check(count[w] > 0 && results[w] == expected_results[w],
              "every timed request returned its expected count");
      for (size_t q = 0; q < histogram.size(); ++q)
        histogram[q] += request_counts[w][q];
    }
    if (!first_only) {
      check(total >= requests.size(), "complete timed request set");
      for (size_t q = 0; q < histogram.size(); ++q)
        check(histogram[q] ==
                  total / requests.size() + (q < total % requests.size()),
              "balanced global request prefix");
    }
    if (first_only)
      for (size_t w = 0; w < workers; ++w)
        for (const auto &doc : cold_results[w])
          validate(doc, id(doc), requests[w * 17 % 64].projection);
    std::vector<double> lat;
    for (auto &v : latencies) lat.insert(lat.end(), v.begin(), v.end());
    double first_request = lat.front();
    std::sort(lat.begin(), lat.end());
    auto quantile = [&](double p) {
      return lat[std::min(lat.size() - 1,
                          size_t(std::ceil(lat.size() * p)) - 1)];
    };
    if (!first) out << ',';
    first = false;
    out << "{\"case\":" << std::quoted(kind) << ",\"request_fingerprint\":"
        << std::quoted(std::to_string(request_fingerprint))
        << ",\"expected_docs_per_cycle\":" << expected_per_cycle
        << ",\"timed_expected_docs\":"
        << std::accumulate(expected_results.begin(), expected_results.end(),
                           size_t{})
        << ",\"request_counts\":[";
    for (size_t q = 0; q < histogram.size(); ++q) {
      if (q) out << ',';
      out << histogram[q];
    }
    out << ']' << ",\"queries\":" << total << ",\"elapsed_s\":" << elapsed
        << ",\"qps\":" << total / elapsed << ",\"p50_us\":" << quantile(.5)
        << ",\"p95_us\":" << quantile(.95) << ",\"p99_us\":" << quantile(.99)
        << ",\"first_request_us\":" << first_request
        << ",\"writer_cycles\":" << writer_cycles
        << ",\"writer_elapsed_s\":" << writer_elapsed
        << ",\"maintenance_us\":" << maintenance_us
        << ",\"cpu_s\":" << ca.cpu_s - cb.cpu_s
        << ",\"rss_bytes\":" << ailego::MemoryHelper::SelfRSS()
        << ",\"pool_used_before\":" << before.used
        << ",\"pool_used_after\":" << after.used
        << ",\"external_used_after\":" << after.external_used
        << ",\"disk_read_bytes\":" << disk_read() - reads
        << ",\"minor_faults\":" << ca.minor_faults - cb.minor_faults
        << ",\"major_faults\":" << ca.major_faults - cb.major_faults
        << ",\"evictions\":"
        << after.bg_evicted_buffers - before.bg_evicted_buffers
        << ",\"high_watermark_hits\":"
        << after.high_watermark_hits - before.high_watermark_hits
        << ",\"capacity_waits\":"
        << after.capacity_waits - before.capacity_waits
        << ",\"capacity_timeouts\":"
        << after.capacity_wait_timeouts - before.capacity_wait_timeouts
        << ",\"ids_validated\":true}";
    out.flush();
    std::cout << "RESULT " << kind << " QPS=" << total / elapsed
              << " calls=" << total << " used=" << after.used << " evicted="
              << after.bg_evicted_buffers - before.bg_evicted_buffers
              << std::endl;
  }
  out << "]";
  if (lifecycle) {
    std::map<std::string, double> timings;
    auto timed = [&](const std::string &name, auto action) {
      auto began = Clock::now();
      action();
      timings[name] =
          std::chrono::duration<double, std::micro>(Clock::now() - began)
              .count();
    };
    auto verify = [&](const std::string &stage) {
      const auto read_rows = mixed ? rows / 2 : rows;
      auto requests = make_requests(split(arg("cases")).front(), read_rows);
      restrict_stable(requests, read_rows);
      Scratch scratch(read_rows);
      auto began = Clock::now();
      for (const auto &request : requests)
        execute(*col, request, false, scratch);
      timings[stage + "_read_qps"] =
          requests.size() /
          std::chrono::duration<double>(Clock::now() - began).count();
      for (const auto &request : requests)
        execute(*col, request, true, scratch);
      check(col->stats().value().doc_count == rows,
            "post-write live document count");
      for (const auto &[n, value] : writer_values) {
        auto docs =
            col->fetch({pk(n)}, std::vector<std::string>{"payload"}, false);
        check(docs.has_value() && docs->at(pk(n)), "updated value retained");
        field<std::string>(docs->at(pk(n)), "payload", value);
      }
    };
    timed("flush_us", [&] { status(col->flush(), "lifecycle flush"); });
    verify("after_flush");
    timed("reopen_us", [&] {
      status(col->close(), "close for reopen");
      col.reset();
      auto opened = Collection::Open(path, CollectionOptions{false, mmap});
      check(opened.has_value(), "lifecycle reopen");
      col = *opened;
    });
    verify("after_reopen");
    timed("optimize_us",
          [&] { status(col->optimize(), "lifecycle optimize"); });
    verify("after_optimize");
    timed("add_column_us", [&] {
      status(col->add_column(std::make_shared<FieldSchema>(
                                 "bench_added", DataType::INT64, true),
                             "7"),
             "add lifecycle column");
    });
    auto added =
        col->fetch({pk(1)}, std::vector<std::string>{"bench_added"}, false);
    check(added.has_value() && added->at(pk(1)), "new field returned");
    field<int64_t>(added->at(pk(1)), "bench_added", 7);
    auto default_added = col->fetch({pk(1)}, std::nullopt, false);
    check(default_added.has_value() && default_added->at(pk(1)),
          "default projection after adding field");
    field<int64_t>(default_added->at(pk(1)), "bench_added", 7);
    verify("after_add_column");
    timed("alter_column_us", [&] {
      status(col->alter_column("bench_added", "bench_renamed"),
             "rename lifecycle column");
    });
    auto renamed =
        col->fetch({pk(1)}, std::vector<std::string>{"bench_renamed"}, false);
    check(renamed.has_value() && renamed->at(pk(1)), "renamed field returned");
    field<int64_t>(renamed->at(pk(1)), "bench_renamed", 7);
    auto default_renamed = col->fetch({pk(1)}, std::nullopt, false);
    check(default_renamed.has_value() && default_renamed->at(pk(1)),
          "default projection after renaming field");
    field<int64_t>(default_renamed->at(pk(1)), "bench_renamed", 7);
    check(!default_renamed->at(pk(1))->has("bench_added"),
          "old field name absent after rename");
    verify("after_alter_column");
    timed("drop_column_us", [&] {
      status(col->drop_column("bench_renamed"), "drop lifecycle column");
    });
    verify("after_drop_column");
    auto default_dropped = col->fetch({pk(1)}, std::nullopt, false);
    check(default_dropped.has_value() && default_dropped->at(pk(1)),
          "default projection after dropping field");
    check(!default_dropped->at(pk(1))->has("bench_added") &&
              !default_dropped->at(pk(1))->has("bench_renamed"),
          "dropped names absent");
    timed("schema_reopen_us", [&] {
      status(col->close(), "close after DDL");
      col.reset();
      auto opened = Collection::Open(path, CollectionOptions{false, mmap});
      check(opened.has_value(), "reopen persisted schema");
      col = *opened;
    });
    verify("after_schema_reopen");
    out << ",\"lifecycle\":{";
    bool item_first = true;
    for (const auto &[name, us] : timings) {
      if (!item_first) out << ',';
      item_first = false;
      out << std::quoted(name) << ':' << us;
    }
    out << ",\"validated\":true}";
  }
  out << ",\"complete\":true}\n";
  out.close();
  status(col->close(), "close reader");
  return 0;
} catch (const std::exception &e) {
  std::cerr << "ERROR " << e.what() << std::endl;
  return 1;
}
