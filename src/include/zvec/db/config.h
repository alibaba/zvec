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
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <zvec/ailego/pattern/singleton.h>
#include <zvec/db/status.h>
#include <zvec/export.h>

namespace zvec {

const uint32_t MIN_LOG_FILE_SIZE = 128;
const uint32_t DEFAULT_LOG_FILE_SIZE = 2048;
const uint32_t DEFAULT_LOG_OVERDUE_DAYS = 7;
const std::string CONSOLE_LOG_TYPE_NAME = "ConsoleLogger";
const std::string FILE_LOG_TYPE_NAME = "AppendLogger";
const std::string DEFAULT_LOG_DIR = "./logs";
const std::string DEFAULT_LOG_BASENAME = "zvec.log";

class ZVEC_API GlobalConfig : public ailego::Singleton<GlobalConfig> {
  friend class ailego::Singleton<GlobalConfig>;

 public:
  enum class LogLevel : uint8_t {
    kDebug = 0,
    kInfo,
    kWarn,
    kError,
    kFatal,
  };

  struct LogConfig {
    LogLevel level;

    LogConfig(LogLevel level) : level(level) {}
    virtual ~LogConfig() = default;
    virtual std::string get_logger_type() const = 0;
  };

  // Console log configuration
  struct ConsoleLogConfig : LogConfig {
    ConsoleLogConfig(LogLevel level = LogLevel::kWarn) : LogConfig{level} {}

    std::string get_logger_type() const override {
      return CONSOLE_LOG_TYPE_NAME;
    }
  };

  // File log configuration
  struct FileLogConfig : LogConfig {
    std::string dir;
    std::string basename;
    uint32_t file_size;  // MB
    uint32_t overdue_days;

    FileLogConfig(LogLevel level = LogLevel::kWarn,
                  std::string dir = DEFAULT_LOG_DIR,
                  std::string basename = DEFAULT_LOG_BASENAME,
                  uint32_t file_size = DEFAULT_LOG_FILE_SIZE,
                  uint32_t overdue_days = DEFAULT_LOG_OVERDUE_DAYS)
        : LogConfig{level},
          dir{dir},
          basename{basename},
          file_size{file_size},
          overdue_days(overdue_days) {}

    std::string get_logger_type() const override {
      return FILE_LOG_TYPE_NAME;
    }
  };

  // Configuration data structure
  struct ConfigData {
    // Process-wide managed cache budget. Internally shared by vector storage
    // and RocksDB-backed metadata/index features; it is not a hard RSS limit.
    uint64_t memory_limit_bytes;

    // log
    std::shared_ptr<LogConfig> log_config;

    // query
    uint32_t query_thread_count;
    // CPU binding is opt-in at the DB layer.
    bool query_thread_binding;
    float invert_to_forward_scan_ratio;
    float brute_force_by_keys_ratio;
    // Independent from brute_force_by_keys_ratio: per-candidate FTS cost
    // (phrase phase-2 IO, BM25) is higher, so a tighter default fits.
    float fts_brute_force_by_keys_ratio;

    // optimize
    uint32_t optimize_thread_count;
    // CPU binding is opt-in at the DB layer.
    bool optimize_thread_binding;

    // FTS jieba tokenizer default dict dir (lowest-priority fallback;
    // per-field config > ZVEC_JIEBA_DICT_DIR > this). Empty by default.
    std::string jieba_dict_dir;

    ConfigData();
  };

  // initialize the configuration (can only be called once)
  Status initialize(const ConfigData &config);

  Status validate(const ConfigData &config) const;

  // Set the process-wide default jieba dict dir. Thread-safe and decoupled
  // from initialize() so language SDKs can call it on module load.
  // initialize() with a non-empty config.jieba_dict_dir overrides this.
  void set_default_jieba_dict_dir(const std::string &dir);

  // Read-only accessors
  uint64_t memory_limit_bytes() const noexcept;

  const LogConfig &log_config() const noexcept {
    auto config = config_snapshot();
    return *config->log_config;
  }

  std::string log_type() const noexcept {
    auto config = config_snapshot();
    return config->log_config->get_logger_type();
  }

  LogLevel log_level() const noexcept {
    auto config = config_snapshot();
    return config->log_config->level;
  }

  // File log specific accessors (only valid when using FileLogConfig)
  const std::string &log_dir() const noexcept {
    auto config = config_snapshot();
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config->log_config.get());
    static const std::string empty_string = "";
    return file_config ? file_config->dir : empty_string;
  }

  const std::string &log_file_basename() const noexcept {
    auto config = config_snapshot();
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config->log_config.get());
    static const std::string empty_string = "";
    return file_config ? file_config->basename : empty_string;
  }

  uint32_t log_file_size() const noexcept {
    auto config = config_snapshot();
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config->log_config.get());
    return file_config ? file_config->file_size : 0;
  }

  uint32_t log_overdue_days() const noexcept {
    auto config = config_snapshot();
    const FileLogConfig *file_config =
        dynamic_cast<const FileLogConfig *>(config->log_config.get());
    return file_config ? file_config->overdue_days : 0;
  }

  //! Query thread count
  uint32_t query_thread_count() const noexcept {
    return config_snapshot()->query_thread_count;
  }

  //! Query thread binding
  bool query_thread_binding() const noexcept {
    return config_snapshot()->query_thread_binding;
  }

  //! Invert to forward scan ratio
  float invert_to_forward_scan_ratio() const noexcept {
    return config_snapshot()->invert_to_forward_scan_ratio;
  }

  //! Brute force by keys ratio
  float brute_force_by_keys_ratio() const noexcept {
    return config_snapshot()->brute_force_by_keys_ratio;
  }

  //! FTS brute force by keys ratio (independent from brute_force_by_keys_ratio
  //! because FTS per-candidate cost is higher).
  float fts_brute_force_by_keys_ratio() const noexcept {
    return config_snapshot()->fts_brute_force_by_keys_ratio;
  }

  //! Optimize thread count
  uint32_t optimize_thread_count() const noexcept {
    return config_snapshot()->optimize_thread_count;
  }

  //! Optimize thread binding
  bool optimize_thread_binding() const noexcept {
    return config_snapshot()->optimize_thread_binding;
  }

  //! Effective jieba dict dir. Thread-safe.
  std::string jieba_dict_dir() const;

 private:
  enum class InitializationState : uint8_t {
    kUninitialized,
    kInitializing,
    kInitialized,
    kFailed,
  };

  std::shared_ptr<const ConfigData> config_snapshot() const noexcept {
    return std::atomic_load_explicit(&config_, std::memory_order_acquire);
  }

  // Readers atomically acquire an immutable snapshot, so initialize() and the
  // language-SDK jieba setter can publish whole configurations without data
  // races or mixed-field observations.
  std::shared_ptr<const ConfigData> config_{
      std::make_shared<const ConfigData>()};

  // The legacy logging accessors return references. Keep replaced LogConfig
  // objects alive for this GlobalConfig's lifetime so a reference acquired
  // concurrently with the one-time snapshot publication cannot dangle.
  std::shared_ptr<LogConfig> retained_log_config_;

  // initialize() can be called concurrently by language bindings. Publish a
  // terminal state only after every initialization stage has completed, and
  // make followers observe the same result instead of returning early while
  // the winning thread is still working.
  InitializationState initialization_state_{
      InitializationState::kUninitialized};
  Status initialization_status_{};
  std::condition_variable initialization_cv_;
  std::mutex initialization_mutex_;

  // Serializes immutable snapshot writers and retained_log_config_.
  mutable std::mutex mutex_;
};

}  // namespace zvec
