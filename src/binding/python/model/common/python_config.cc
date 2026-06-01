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
#include <pybind11/stl.h>

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

    // jieba_dict_dir: optional override of the SDK-registered default.
    // Empty value is a no-op (Initialize preserves the SDK default).
    if (has_key(config_dict, "jieba_dict_dir")) {
      data.jieba_dict_dir =
          get_if<std::string>(config_dict, "jieba_dict_dir").value();
    }

    // initialize (contains validate)
    Status status = GlobalConfig::Instance().Initialize(data);
    if (!status.ok()) {
      throw std::runtime_error("Initialization failed: " + status.message());
    }
    return py::none();
  });

  // Process-wide setter, independent of Initialize(); called by __init__.py
  // on import to register the wheel-bundled dict path.
  m.def(
      "set_default_jieba_dict_dir",
      [](const std::string &dir) {
        GlobalConfig::Instance().set_default_jieba_dict_dir(dir);
      },
      pybind11::arg("dir"),
      "Register the process-wide default jieba dict directory.");

  m.def(
      "get_default_jieba_dict_dir",
      []() -> std::string { return GlobalConfig::Instance().jieba_dict_dir(); },
      "Read the currently registered default jieba dict directory.");
}


}  // namespace zvec