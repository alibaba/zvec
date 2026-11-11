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

#include "python_config.h"
#include <ailego/internal/cpu_features.h>
#include <pybind11/stl.h>
#include <zvec/ailego/io/io_backend.h>
#include "db/sqlengine/scalar_jit/runtime.h"

namespace zvec {

inline bool has_key(py::dict d, const std::string &key) {
  return py::bool_(d.contains(key));
}

template <typename T>
std::optional<T> get_if(py::dict d, const std::string &key) {
  if (has_key(d, key)) {
    try {
      py::object obj = d[py::str(key)];
      return obj.cast<T>();
    } catch (const py::cast_error &) {
      throw py::type_error("Key '" + key + "' is not of expected type.");
    }
  }
  return std::nullopt;
}

inline std::string to_lower(const std::string &s) {
  std::string lower;
  lower.reserve(s.size());
  std::transform(s.begin(), s.end(), std::back_inserter(lower), ::tolower);
  return lower;
}

inline bool iequals(const std::string &a, const std::string &b) {
  return to_lower(a) == to_lower(b);
}

GlobalConfig::LogLevel str_to_loglevel(const std::string &s) {
  if (iequals(s, "debug")) return GlobalConfig::LogLevel::kDebug;
  if (iequals(s, "info")) return GlobalConfig::LogLevel::kInfo;
  if (iequals(s, "warn") || iequals(s, "warning"))
    return GlobalConfig::LogLevel::kWarn;
  if (iequals(s, "error")) return GlobalConfig::LogLevel::kError;
  if (iequals(s, "fatal")) return GlobalConfig::LogLevel::kFatal;
  throw py::value_error("Invalid log level: ");
}


void ZVecPyConfig::Initialize(pybind11::module_ &m) {
  m.def("Initialize", [](py::args args, py::kwargs kwargs) -> py::none {
    py::dict config_dict;
    // parse args
    for (auto &arg : args) {
      if (py::isinstance<py::dict>(arg)) {
        for (auto item : arg.cast<py::dict>()) {
          config_dict[item.first] = item.second;
        }
      } else {
        throw py::type_error("Positional argument must be a dict if provided");
      }
    }

    // parser kwargs
    if (kwargs) {
      for (auto item : kwargs) {
        config_dict[item.first] = item.second;
      }
    }

    if (config_dict.empty()) {
      return py::none();
    }

    GlobalConfig::ConfigData data;
    // config memory_limit_mb
    if (has_key(config_dict, "memory_limit_mb")) {
      auto mb = get_if<int64_t>(config_dict, "memory_limit_mb").value();
      if (mb <= 0) throw py::value_error("memory_limit_mb must be positive");
      data.memory_limit_bytes = static_cast<uint64_t>(mb) * 1024 * 1024;
    }

    // config log
    bool has_log_type = has_key(config_dict, "log_type");
    bool has_log_level = has_key(config_dict, "log_level");
    if (has_log_type || has_log_level) {
      std::string log_type = "console";
      std::string log_level_str = "warn";

      if (has_log_type) {
        log_type = config_dict["log_type"].cast<std::string>();
      }
      if (has_log_level) {
        log_level_str = config_dict["log_level"].cast<std::string>();
      }
      auto log_level = str_to_loglevel(log_level_str);
      if (iequals(log_type, "file")) {
        std::string dir = DEFAULT_LOG_DIR;
        std::string basename = DEFAULT_LOG_BASENAME;
        uint32_t file_size = DEFAULT_LOG_FILE_SIZE;
        uint32_t overdue_days = DEFAULT_LOG_OVERDUE_DAYS;

        if (has_key(config_dict, "log_dir")) {
          dir = get_if<std::string>(config_dict, "log_dir").value();
        }
        if (has_key(config_dict, "log_basename")) {
          basename = get_if<std::string>(config_dict, "log_basename").value();
        }
        if (has_key(config_dict, "log_file_size")) {
          auto s = get_if<int32_t>(config_dict, "log_file_size").value();
          if (s <= 0) {
            throw py::value_error("log_file_size must be positive");
          }
          file_size = static_cast<uint32_t>(s);
        }
        if (has_key(config_dict, "log_overdue_days")) {
          std::cout << " ** log_overdue_days: " << overdue_days << std::endl;
          auto d = get_if<int32_t>(config_dict, "log_overdue_days").value();
          if (d <= 0) {
            throw py::value_error("log_overdue_days must be positive");
          }
          overdue_days = static_cast<uint32_t>(d);
        }

        data.log_config = std::make_shared<GlobalConfig::FileLogConfig>(
            log_level, dir, basename, file_size, overdue_days);

      } else if (iequals(log_type, "console")) {
        data.log_config =
            std::make_shared<GlobalConfig::ConsoleLogConfig>(log_level);
      } else {
        throw py::value_error("log_type must be 'console' or 'file'");
      }
    }

    // set query thread count
    if (has_key(config_dict, "query_threads")) {
      auto q = get_if<int32_t>(config_dict, "query_threads").value();
      if (q <= 0) throw py::value_error("query_threads must be positive");
      data.query_thread_count = static_cast<uint32_t>(q);
    }

    // set optimize thread count
    if (has_key(config_dict, "optimize_threads")) {
      auto o = get_if<int32_t>(config_dict, "optimize_threads").value();
      if (o <= 0) throw py::value_error("optimize_threads must be positive");
      data.optimize_thread_count = static_cast<uint32_t>(o);
    }

    // set invert_to_forward_scan_ratio
    if (has_key(config_dict, "invert_to_forward_scan_ratio")) {
      auto v =
          get_if<double>(config_dict, "invert_to_forward_scan_ratio").value();
      if (v < 0.0 || v > 1.0) {
        throw py::value_error(
            "invert_to_forward_scan_ratio must be in [0.0, 1.0]");
      }
      data.invert_to_forward_scan_ratio = static_cast<float>(v);
    }

    // set brute_force_by_keys_ratio
    if (has_key(config_dict, "brute_force_by_keys_ratio")) {
      auto v = get_if<double>(config_dict, "brute_force_by_keys_ratio").value();
      if (v < 0.0 || v > 1.0) {
        throw py::value_error(
            "brute_force_by_keys_ratio must be in [0.0, 1.0]");
      }
      data.brute_force_by_keys_ratio = static_cast<float>(v);
    }

    // set fts_brute_force_by_keys_ratio
    if (has_key(config_dict, "fts_brute_force_by_keys_ratio")) {
      auto v =
          get_if<double>(config_dict, "fts_brute_force_by_keys_ratio").value();
      if (v < 0.0 || v > 1.0) {
        throw py::value_error(
            "fts_brute_force_by_keys_ratio must be in [0.0, 1.0]");
      }
      data.fts_brute_force_by_keys_ratio = static_cast<float>(v);
    }

    // initialize (contains validate)
    Status status = GlobalConfig::Instance().initialize(data);
    if (!status.ok()) {
      throw std::runtime_error("Initialization failed: " + status.message());
    }
    return py::none();
  });

  m.def(
      "scalar_jit_info",
      [] {
        auto info = sqlengine::scalar_jit::GetRuntimeSnapshot();
        py::dict result, limits, cache, timings, memory, selections, execution;
        result["version"] = 1;
        const auto features = ailego::internal::CpuFeatures::UsableX86();
        py::dict cpu;
        cpu["sse42"] = features.sse42;
        cpu["avx"] = features.avx;
        cpu["avx2"] = features.avx2;
        cpu["avx512f"] = features.avx512f;
        result["cpu_features"] = cpu;
        result["built"] = info.built;
        result["enabled"] = info.enabled;
        result["initialized"] = info.initialized;
        result["disable_reason"] = info.reason;
        result["initialization_error"] = info.initialization_error;
        result["accounting_available"] =
            sqlengine::scalar_jit::InOriginalProcess();
        result["profile"] = info.config.profile;
        result["dump_ir"] = info.config.dump_ir;
        limits["positive_entries"] = 128;
        limits["negative_entries"] = 128;
        limits["compilations"] = info.config.max_compilations;
        limits["live_programs"] = info.config.max_live_programs;
        limits["expression_nodes"] = info.config.limits.nodes;
        limits["expression_depth"] = info.config.limits.depth;
        limits["key_bytes"] = info.config.limits.key_bytes;
        limits["input_bytes"] = info.config.limits.input_bytes;
        limits["in_values"] = info.config.limits.in_values;
        for (const char *name :
             {"hits", "misses", "in_flight_bypasses", "admission_rejections",
              "compilations", "failed_compilations", "backoff_hits",
              "evictions", "entries", "negative_entries"})
          cache[name] = info.counters[std::string("cache_") + name];
        cache["active_compilations"] = info.counters["active_compilations"];
        for (const char *name : {"initialization", "compilation", "lowering",
                                 "binding", "preparation"}) {
          py::dict stage;
          stage["count"] = info.counters[std::string(name) + "_count"];
          stage["nanoseconds"] = info.counters[std::string(name) + "_ns"];
          timings[name] = stage;
        }
        selections["jit"] = info.counters["jit_selections"];
        selections["arrow"] = info.counters["arrow_selections"];
        py::dict fallbacks;
        for (const char *reason :
             {"build_disabled", "runtime_disabled", "invalid_configuration",
              "forked_process", "unsupported_expression",
              "unsupported_snapshot", "expression_limit", "compilation_busy",
              "live_program_limit", "compile_backoff", "initialization_failed",
              "compile_failed", "binding_failed"})
          fallbacks[reason] = info.counters[std::string("fallback_") + reason];
        selections["fallbacks"] = fallbacks;
        selections["meaning"] = "preparation selections, not execution counts";
        for (const char *backend : {"jit", "arrow"}) {
          py::dict sample;
          sample["count"] =
              info.counters[std::string(backend) + "_execution_sample_count"];
          sample["nanoseconds"] =
              info.counters[std::string(backend) + "_execution_sample_ns"];
          execution[backend] = sample;
        }
        execution["sample_interval"] = 1024;
        execution["meaning"] =
            "first and every 1024th evaluation per thread/backend; sampled "
            "only";
        for (const char *kind :
             {"program", "binding", "descriptor", "helper"}) {
          const auto prefix = std::string("live_") + kind;
          memory[py::str(prefix + "s")] = info.counters[prefix + "s"];
          for (const char *suffix :
               {"_object_bytes", "_ir_bytes", "_owned_bytes"})
            memory[py::str(prefix + suffix)] = info.counters[prefix + suffix];
        }
        for (const char *name :
             {"resident_object_bytes", "resident_ir_bytes",
              "resident_owned_bytes", "shared_snapshot_references"})
          memory[name] = info.counters[name];
        memory["object_bytes_meaning"] =
            "emitted object-file size proxy, not executable pages";
        memory["unmeasured"] =
            "LLVM heap, allocator overhead, registry, RE2 internals and "
            "hash-table overhead";
        memory["borrowed_arrow_buffers_included"] = false;
        result["limits"] = limits;
        result["cache"] = cache;
        result["timings"] = timings;
        result["selections"] = selections;
        result["execution_samples"] = execution;
        result["memory"] = memory;
        return result;
      },
      "Returns versioned process-local scalar JIT diagnostics without "
      "initializing LLVM.");

  // Returns the selected DiskAnn I/O backend. Linux tries io_uring, then
  // libaio, then pread; macOS ARM64 uses pread.
  m.def(
      "io_backend_type",
      []() -> ailego::IOBackendType {
        return ailego::current_io_backend_type();
      },
      "Returns the current I/O backend type for DiskAnn disk reads "
      "as an IOBackendType enum (zvec.typing.IOBackendType). "
      "Linux selects IOBackendType.IO_URING, IOBackendType.LIBAIO, or "
      "IOBackendType.PREAD in that order. macOS ARM64 uses "
      "IOBackendType.PREAD. Windows uses "
      "IOBackendType.WINDOWS_OVERLAPPED.");

  // Returns a human-readable description identifying io_uring, libaio, or
  // pread, with asynchronous-backend guidance for Linux pread fallback.
  m.def(
      "io_backend_description",
      []() -> std::string { return ailego::current_io_backend_description(); },
      "Returns a human-readable description of the current I/O backend. "
      "The description identifies io_uring, libaio, or pread. On Linux, the "
      "pread description includes guidance for enabling io_uring or "
      "installing libaio. Windows reports its overlapped-I/O backend.");
}


}  // namespace zvec
