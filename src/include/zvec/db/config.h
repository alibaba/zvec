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
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <zvec/ailego/pattern/singleton.h>
#include <zvec/db/status.h>

namespace zvec {

const uint32_t MIN_LOG_FILE_SIZE = 128;
const uint32_t DEFAULT_LOG_FILE_SIZE = 2048;
const uint32_t DEFAULT_LOG_OVERDUE_DAYS = 7;
const std::string CONSOLE_LOG_TYPE_NAME = "ConsoleLogger";
const std::string FILE_LOG_TYPE_NAME = "AppendLogger";
const std::string DEFAULT_LOG_DIR = "./logs";
const std::string DEFAULT_LOG_BASENAME = "zvec.log";

class GlobalConfig : public ailego::Singleton<GlobalConfig> {
  friend class ailego::Singleton<GlobalConfig>;

 public:
  enum class LogLevel : uint8_t {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    FATAL,
  };

  struct LogConfig {
    LogLevel level;

    LogConfig(LogLevel level) : level(level) {}
    virtual ~LogConfig() = default;
    virtual std::string GetLoggerType() const = 0;
  };

  // Console log configuration
  struct ConsoleLogConfig : LogConfig {
    ConsoleLogConfig(LogLevel level = LogLevel::WARN) : LogConfig{level} {}

    std::string GetLoggerType() const override {
      return CONSOLE_LOG_TYPE_NAME;
    }
  };

  // File log configuration
  struct FileLogConfig : LogConfig {
    std::string dir;
    std::string basename;
    uint32_t file_size;  // MB
    uint32_t overdue_days;

    FileLogConfig(LogLevel level = LogLevel::WARN,
                  std::string dir = DEFAULT_LOG_DIR,
                  std::string basename = DEFAULT_LOG_BASENAME,
                  uint32_t file_size = DEFAULT_LOG_FILE_SIZE,
                  uint32_t overdue_days = DEFAULT_LOG_OVERDUE_DAYS)
        : LogConfig{level},
          dir{dir},
          basename{basename},
          file_size{file_size},
          overdue_days(overdue_days) {}

    std::string GetLoggerType() const override {
      return FILE_LOG_TYPE_NAME;
    }
  };

  // Configuration data structure
  struct ConfigData {
    uint64_t memory_limit_bytes;

    // log
    std::shared_ptr<LogConfig> log_config;

    // query
    uint32_t query_thread_count;
    float invert_to_forward_scan_ratio;
    float brute_force_by_keys_ratio;

    // optimize
    uint32_t optimize_thread_count;

    ConfigData();
  };

  // Initialize the configuration (can only be called once)
  Status Initialize(const ConfigData &config);

  Status Validate(const ConfigData &config) const;

  // Read-only accessors
  uint64_t memory_limit_bytes() const noexcept;

  const LogConfig &log_config() const noexcept {
    return *config_.log_config;
  }

  std::string log_type() const noexcept {
    return config_.log_config->GetLoggerType();
  }

  LogLevel log_level() const noexcept {
    return config_.log_config->level;
  }

  // File log specific accessors (only valid when using FileLogConfig)
  const std::string &log_dir() const noexcept {
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config_.log_config.get());
    static const std::string empty_string = "";
    return file_config ? file_config->dir : empty_string;
  }

  const std::string &log_file_basename() const noexcept {
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config_.log_config.get());
    static const std::string empty_string = "";
    return file_config ? file_config->basename : empty_string;
  }

  uint32_t log_file_size() const noexcept {
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config_.log_config.get());
    return file_config ? file_config->file_size : 0;
  }

  uint32_t log_overdue_days() const noexcept {
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config_.log_config.get());
    return file_config ? file_config->overdue_days : 0;
  }

  //! Query thread count
  uint32_t query_thread_count() const noexcept {
    return config_.query_thread_count;
  }

  //! Invert to forward scan ratio
  float invert_to_forward_scan_ratio() const noexcept {
    return config_.invert_to_forward_scan_ratio;
  }

  //! Brute force by keys ratio
  float brute_force_by_keys_ratio() const noexcept {
    return config_.brute_force_by_keys_ratio;
  }

  //! Optimize thread count
  uint32_t optimize_thread_count() const noexcept {
    return config_.optimize_thread_count;
  }

 private:
  // Configuration data
  ConfigData config_;

  // Atomic flag to ensure initialization happens only once
  std::atomic<bool> initialized_{false};
};

}  // namespace zvec