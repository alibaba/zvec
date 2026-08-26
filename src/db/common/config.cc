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

#include <exception>
#include <iostream>
#include <memory>
#include <zvec/db/config.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"
#include "db/common/global_resource.h"
#include "cgroup_util.h"
#include "global_resource.h"
#include "glogger.h"
#include "logger.h"
#include "typedef.h"

namespace zvec {

static void ExitLogHandler() {
  LogUtil::Shutdown();
}

GlobalConfig::ConfigData::ConfigData()
    : memory_limit_bytes(CgroupUtil::getMemoryLimit() *
                         DEFAULT_MEMORY_LIMIT_RATIO),
      log_config(std::make_shared<ConsoleLogConfig>()),
      query_thread_count(CgroupUtil::getCpuLimit()),
      query_thread_binding(false),
      invert_to_forward_scan_ratio(0.9),
      brute_force_by_keys_ratio(0.1),
      fts_brute_force_by_keys_ratio(0.05),
      optimize_thread_count(query_thread_count),
      optimize_thread_binding(false),
      jieba_dict_dir() {}

Status GlobalConfig::validate(const ConfigData &config) const {
  if (config.memory_limit_bytes < MIN_MEMORY_LIMIT_BYTES) {
    return Status::InvalidArgument("memory_limit_bytes must be greater than ",
                                   MIN_MEMORY_LIMIT_BYTES);
  }

  if (config.memory_limit_bytes > CgroupUtil::getMemoryLimit()) {
    return Status::InvalidArgument("memory_limit_bytes must be less than ",
                                   CgroupUtil::getMemoryLimit());
  }

  // Validate query thread count
  if (config.query_thread_count == 0) {
    return Status::InvalidArgument("query_thread_count must be greater than 0");
  }

  // Validate invert_to_forward_scan_ratio (should be between 0 and 1)
  if (config.invert_to_forward_scan_ratio < 0.0f ||
      config.invert_to_forward_scan_ratio > 1.0f) {
    return Status::InvalidArgument(
        "invert_to_forward_scan_ratio must be between 0 and 1");
  }

  // Validate brute_force_by_keys_ratio (should be between 0 and 1)
  if (config.brute_force_by_keys_ratio < 0.0f ||
      config.brute_force_by_keys_ratio > 1.0f) {
    return Status::InvalidArgument(
        "brute_force_by_keys_ratio must be between 0 and 1");
  }

  // Validate fts_brute_force_by_keys_ratio (should be between 0 and 1)
  if (config.fts_brute_force_by_keys_ratio < 0.0f ||
      config.fts_brute_force_by_keys_ratio > 1.0f) {
    return Status::InvalidArgument(
        "fts_brute_force_by_keys_ratio must be between 0 and 1");
  }

  // Validate optimize thread count
  if (config.optimize_thread_count == 0) {
    return Status::InvalidArgument(
        "optimize_thread_count must be greater than 0");
  }

  // Validate log configuration
  if (!config.log_config) {
    return Status::InvalidArgument("log_config cannot be null");
  }
  const std::string logger_type = config.log_config->get_logger_type();
  if (logger_type == FILE_LOG_TYPE_NAME) {
    auto log_config =
        std::dynamic_pointer_cast<FileLogConfig>(config.log_config);
    if (!log_config) {
      return Status::InvalidArgument(
          "AppendLogger configuration must use FileLogConfig");
    }

    // Validate file log specific configurations
    if (log_config->dir.empty()) {
      return Status::InvalidArgument(
          "log_dir cannot be empty when set to FileLogger");
    }

    if (log_config->basename.empty()) {
      return Status::InvalidArgument(
          "log_file basename cannot be empty when set to FileLogger");
    }

    if (log_config->file_size <= MIN_LOG_FILE_SIZE) {
      return Status::InvalidArgument("log file_size must be greater than ",
                                     MIN_LOG_FILE_SIZE,
                                     " when set to FileLogger");
    }

    if (log_config->overdue_days == 0) {
      return Status::InvalidArgument(
          "log_overdue_days must be greater than 0 when set to FileLogger");
    }
  } else if (logger_type == CONSOLE_LOG_TYPE_NAME) {
    if (!std::dynamic_pointer_cast<ConsoleLogConfig>(config.log_config)) {
      return Status::InvalidArgument(
          "ConsoleLogger configuration must use ConsoleLogConfig");
    }
  } else {
    return Status::InvalidArgument("unsupported logger type: ", logger_type);
  }

  return Status::OK();
}

Status GlobalConfig::initialize(const ConfigData &config) {
  {
    std::unique_lock<std::mutex> lock(initialization_mutex_);
    initialization_cv_.wait(lock, [this] {
      return initialization_state_ != InitializationState::kInitializing;
    });
    if (initialization_state_ == InitializationState::kInitialized ||
        initialization_state_ == InitializationState::kFailed) {
      return initialization_status_;
    }
    initialization_state_ = InitializationState::kInitializing;
  }

  Status result;
  bool validation_failed = false;
  try {
    result = validate(config);
    validation_failed = !result.ok();

    std::unique_lock<std::mutex> config_write_lock;
    ConfigData effective_config;
    std::shared_ptr<const ConfigData> published_config;
    if (result.ok()) {
      // Serialize snapshot writers through the whole one-time initialization.
      // A concurrent SDK jieba setter then runs either before this snapshot is
      // prepared or after it is published, so an empty Initialize() value can
      // never overwrite a newer setter update.
      config_write_lock = std::unique_lock<std::mutex>(mutex_);
      // Preserve the SDK-set jieba_dict_dir when caller didn't specify one.
      // Prepare a complete snapshot locally, but do not publish it until all
      // side-effecting initialization stages have succeeded.
      effective_config = config;
      if (effective_config.jieba_dict_dir.empty()) {
        effective_config.jieba_dict_dir = config_snapshot()->jieba_dict_dir;
      }
      // The public ConfigData remains mutable to preserve API compatibility.
      // Clone the built-in logger configurations so caller-side mutations
      // after Initialize() cannot silently modify the published snapshot.
      if (auto file = std::dynamic_pointer_cast<FileLogConfig>(
              effective_config.log_config)) {
        effective_config.log_config = std::make_shared<FileLogConfig>(*file);
      } else if (auto console = std::dynamic_pointer_cast<ConsoleLogConfig>(
                     effective_config.log_config)) {
        effective_config.log_config =
            std::make_shared<ConsoleLogConfig>(*console);
      }
      // Allocate the immutable publication object before logs, thread pools or
      // the memory pool are changed. The final atomic store cannot fail.
      published_config = std::make_shared<const ConfigData>(effective_config);
    }

    if (result.ok()) {
      static const bool exit_handler_registered =
          std::atexit(ExitLogHandler) == 0;
      if (!exit_handler_registered) {
        std::cerr << "Failed to register exit handler" << std::endl;
        result = Status::InternalError("Failed to register exit handler");
      }
    }

    bool log_setup_attempted = false;
    bool log_initialized = false;
    if (result.ok()) {
      Status log_status;
      const int resource_result =
          GlobalResource::Instance().initialize_with_setup(
              effective_config.memory_limit_bytes,
              effective_config.query_thread_count,
              effective_config.query_thread_binding,
              effective_config.optimize_thread_count,
              effective_config.optimize_thread_binding, [&] {
                log_setup_attempted = true;
                const auto *file_config = dynamic_cast<const FileLogConfig *>(
                    effective_config.log_config.get());
                static const std::string empty;
                log_status = LogUtil::Init(
                    file_config ? file_config->dir : empty,
                    file_config ? file_config->basename : empty,
                    int(effective_config.log_config->level),
                    effective_config.log_config->get_logger_type(),
                    file_config ? file_config->file_size : 0,
                    file_config ? file_config->overdue_days : 0);
                log_initialized = log_status.ok();
                return log_initialized ? 0 : -1;
              });
      if (resource_result != 0 && log_setup_attempted) {
        // LogUtil::Init may have made partial progress even when it returns an
        // error, so normalize every failed setup/resource transaction back to
        // an uninitialized logger.
        LogUtil::Shutdown();
      }
      if (!log_status.ok()) {
        result = log_status;
      } else if (resource_result != 0) {
        // A predictable configuration mismatch is rejected before the setup
        // callback. If a later memory-pool stage fails, undo the newly-created
        // logger so Initialize() does not leave a half-published subsystem.
        result = Status::InternalError(
            "Failed to initialize the process-wide global resources");
      }
    }

    if (result.ok()) {
      auto old_config = config_snapshot();
      if (old_config->log_config != published_config->log_config) {
        retained_log_config_ = old_config->log_config;
      }
      std::atomic_store_explicit(&config_, std::move(published_config),
                                 std::memory_order_release);
    }
  } catch (const std::exception &e) {
    result = Status::InternalError(
        "Global configuration initialization threw: ", e.what());
  } catch (...) {
    result = Status::InternalError(
        "Global configuration initialization threw an unknown exception");
  }

  {
    std::lock_guard<std::mutex> lock(initialization_mutex_);
    initialization_status_ = result;
    // Invalid input has no side effects and may be corrected by a later call.
    // A later failure may already have registered the harmless process-exit
    // callback, so retain and replay that terminal error rather than retrying
    // a one-time process initialization with ambiguous ownership.
    initialization_state_ =
        validation_failed ? InitializationState::kUninitialized
                          : (result.ok() ? InitializationState::kInitialized
                                         : InitializationState::kFailed);
  }
  initialization_cv_.notify_all();
  return result;
}

void GlobalConfig::set_default_jieba_dict_dir(const std::string &dir) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto next = std::make_shared<ConfigData>(*config_snapshot());
  next->jieba_dict_dir = dir;
  std::atomic_store_explicit(&config_,
                             std::shared_ptr<const ConfigData>(std::move(next)),
                             std::memory_order_release);
}

std::string GlobalConfig::jieba_dict_dir() const {
  return config_snapshot()->jieba_dict_dir;
}

uint64_t GlobalConfig::memory_limit_bytes() const noexcept {
  return config_snapshot()->memory_limit_bytes;
}

FACTORY_REGISTER_LOGGER(AppendLogger);

}  // namespace zvec
