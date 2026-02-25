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

#include "zvec/c_api.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>

// =============================================================================
// RAII Helpers and Error Handling Macros
// =============================================================================

namespace {

// RAII guard for malloc-allocated memory
template <typename T>
struct MallocGuard {
  T *ptr;
  explicit MallocGuard(T *p = nullptr) : ptr(p) {}
  ~MallocGuard() {
    if (ptr) std::free(ptr);
  }
  MallocGuard(const MallocGuard &) = delete;
  MallocGuard &operator=(const MallocGuard &) = delete;
  MallocGuard(MallocGuard &&other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  MallocGuard &operator=(MallocGuard &&other) noexcept {
    if (this != &other) {
      if (ptr) std::free(ptr);
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }
  T *get() const {
    return ptr;
  }
  T *release() {
    T *p = ptr;
    ptr = nullptr;
    return p;
  }
  T **ptr_ptr() {
    return &ptr;
  }
};

// RAII guard for C++ objects allocated with new
template <typename T>
struct DeleteGuard {
  T *ptr;
  explicit DeleteGuard(T *p = nullptr) : ptr(p) {}
  ~DeleteGuard() {
    delete ptr;
  }
  DeleteGuard(const DeleteGuard &) = delete;
  DeleteGuard &operator=(const DeleteGuard &) = delete;
  DeleteGuard(DeleteGuard &&other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  T *get() const {
    return ptr;
  }
  T *release() {
    T *p = ptr;
    ptr = nullptr;
    return p;
  }
};

// RAII guard for array allocated with new[]
template <typename T>
struct DeleteArrayGuard {
  T *ptr;
  explicit DeleteArrayGuard(T *p = nullptr) : ptr(p) {}
  ~DeleteArrayGuard() {
    delete[] ptr;
  }
  DeleteArrayGuard(const DeleteArrayGuard &) = delete;
  DeleteArrayGuard &operator=(const DeleteArrayGuard &) = delete;
  DeleteArrayGuard(DeleteArrayGuard &&other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  T *get() const {
    return ptr;
  }
  T *release() {
    T *p = ptr;
    ptr = nullptr;
    return p;
  }
};

}  // namespace

// Error checking macros - these preserve __LINE__ accuracy
#define ZVEC_CHECK_NOTNULL(ptr, error_code, msg)                               \
  if (!(ptr)) {                                                                \
    set_last_error_details(error_code, msg, __FILE__, __LINE__, __FUNCTION__); \
    return nullptr;                                                            \
  }

#define ZVEC_CHECK_NOTNULL_ERRCODE(ptr, error_code, msg)                       \
  if (!(ptr)) {                                                                \
    set_last_error_details(error_code, msg, __FILE__, __LINE__, __FUNCTION__); \
    return (error_code);                                                       \
  }

#define ZVEC_CHECK_COND(cond, error_code, msg)                                 \
  if (cond) {                                                                  \
    set_last_error_details(error_code, msg, __FILE__, __LINE__, __FUNCTION__); \
    return nullptr;                                                            \
  }

#define ZVEC_CHECK_COND_ERRCODE(cond, error_code, msg)                         \
  if (cond) {                                                                  \
    set_last_error_details(error_code, msg, __FILE__, __LINE__, __FUNCTION__); \
    return (error_code);                                                       \
  }

// For void functions (no return value):
#define ZVEC_TRY_BEGIN_VOID try {
#define ZVEC_CATCH_END_VOID                                \
  }                                                        \
  catch (const std::exception &e) {                        \
    set_last_error(std::string("Exception: ") + e.what()); \
  }

// For functions returning pointer - complete try-catch wrapper
// Usage: ZVEC_TRY_RETURN_NULL("error msg", code...)
// Note: Use variadic macro to handle commas in template arguments
#define ZVEC_TRY_RETURN_NULL(msg, ...)                                   \
  try {                                                                  \
    { __VA_ARGS__ }                                                      \
  } catch (const std::bad_alloc &e) {                                    \
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,                \
                           std::string(msg) + ": " + e.what(), __FILE__, \
                           __LINE__, __FUNCTION__);                      \
    return nullptr;                                                      \
  } catch (const std::exception &e) {                                    \
    set_last_error_details(ZVEC_ERROR_INTERNAL_ERROR,                    \
                           std::string(msg) + ": " + e.what(), __FILE__, \
                           __LINE__, __FUNCTION__);                      \
    return nullptr;                                                      \
  }

// For functions returning ErrorCode
// Usage: ZVEC_TRY_RETURN_ERROR("error msg", code...)
// Note: Use variadic macro to handle commas in template arguments
#define ZVEC_TRY_RETURN_ERROR(msg, ...)                                  \
  try {                                                                  \
    { __VA_ARGS__ }                                                      \
  } catch (const std::bad_alloc &e) {                                    \
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,                \
                           std::string(msg) + ": " + e.what(), __FILE__, \
                           __LINE__, __FUNCTION__);                      \
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;                                \
  } catch (const std::exception &e) {                                    \
    set_last_error_details(ZVEC_ERROR_INTERNAL_ERROR,                    \
                           std::string(msg) + ": " + e.what(), __FILE__, \
                           __LINE__, __FUNCTION__);                      \
    return ZVEC_ERROR_INTERNAL_ERROR;                                    \
  }

// For functions returning scalar values (int, float, size_t, etc.)
// Usage: ZVEC_TRY_RETURN_SCALAR("error msg", error_value, code...)
// Note: Use variadic macro to handle commas in template arguments
#define ZVEC_TRY_RETURN_SCALAR(msg, error_val, ...)                      \
  try {                                                                  \
    { __VA_ARGS__ }                                                      \
  } catch (const std::bad_alloc &e) {                                    \
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,                \
                           std::string(msg) + ": " + e.what(), __FILE__, \
                           __LINE__, __FUNCTION__);                      \
    return (error_val);                                                  \
  } catch (const std::exception &e) {                                    \
    set_last_error_details(ZVEC_ERROR_INTERNAL_ERROR,                    \
                           std::string(msg) + ": " + e.what(), __FILE__, \
                           __LINE__, __FUNCTION__);                      \
    return (error_val);                                                  \
  }

// Global status flags
static std::atomic<bool> g_initialized{false};
static std::mutex g_init_mutex;

// Thread-local storage for error information
static thread_local std::string last_error_message;
static thread_local ZVecErrorDetails last_error_details;

// Helper function: set error information
static void set_last_error(const std::string &msg) {
  last_error_message = msg;

  last_error_details.code = ZVEC_ERROR_UNKNOWN;
  last_error_details.message = last_error_message.c_str();
  last_error_details.file = nullptr;
  last_error_details.line = 0;
  last_error_details.function = nullptr;
}

// Error setting function with detailed information
static void set_last_error_details(ZVecErrorCode code, const std::string &msg,
                                   const char *file = nullptr, int line = 0,
                                   const char *function = nullptr) {
  last_error_message = msg;
  last_error_details.code = code;
  last_error_details.message = last_error_message.c_str();
  last_error_details.file = file;
  last_error_details.line = line;
  last_error_details.function = function;
}

// =============================================================================
// Version information interface implementation
// =============================================================================

// Store dynamically generated version information
static std::string g_version_info;
static std::mutex g_version_mutex;

const char *zvec_get_version(void) {
  std::lock_guard<std::mutex> lock(g_version_mutex);

  if (g_version_info.empty()) {
    ZVEC_TRY_BEGIN_VOID
    std::string version = ZVEC_VERSION_STRING;

    // Try to get Git information
    std::string git_info;
#ifdef ZVEC_GIT_DESCRIBE
    git_info = ZVEC_GIT_DESCRIBE;
#elif defined(ZVEC_GIT_COMMIT_HASH)
    git_info = std::string("g") + ZVEC_GIT_COMMIT_HASH;
#endif

    if (!git_info.empty()) {
      version += "-" + git_info;
    }

    version +=
        " (built " + std::string(__DATE__) + " " + std::string(__TIME__) + ")";

    g_version_info = version;
    ZVEC_CATCH_END_VOID
  }

  return g_version_info.c_str();
}

bool zvec_check_version(int major, int minor, int patch) {
  if (major < 0 || minor < 0 || patch < 0) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Version numbers must be non-negative", __FILE__,
                           __LINE__, __FUNCTION__);
    return false;
  }

  if (ZVEC_VERSION_MAJOR > major) return true;
  if (ZVEC_VERSION_MAJOR < major) return false;

  if (ZVEC_VERSION_MINOR > minor) return true;
  if (ZVEC_VERSION_MINOR < minor) return false;

  return ZVEC_VERSION_PATCH >= patch;
}

int zvec_get_version_major(void) {
  return ZVEC_VERSION_MAJOR;
}

int zvec_get_version_minor(void) {
  return ZVEC_VERSION_MINOR;
}

int zvec_get_version_patch(void) {
  return ZVEC_VERSION_PATCH;
}

// =============================================================================
// String management functions implementation
// =============================================================================

ZVecString *zvec_string_create(const char *str) {
  if (!str) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "String pointer cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return nullptr;
  }

  size_t len = strlen(str);
  ZVecString *zstr = static_cast<ZVecString *>(malloc(sizeof(ZVecString)));
  if (!zstr) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecString", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  char *data_buffer = static_cast<char *>(malloc(len + 1));
  if (!data_buffer) {
    free(zstr);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for string data",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  memcpy(data_buffer, str, len + 1);
  zstr->data = data_buffer;
  zstr->length = len;
  zstr->capacity = len + 1;
  return zstr;
}

ZVecString *zvec_string_create_from_view(const ZVecStringView *view) {
  if (!view || !view->data) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "String view or data cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  ZVecString *zstr = static_cast<ZVecString *>(malloc(sizeof(ZVecString)));
  if (!zstr) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecString", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  char *data_buffer = static_cast<char *>(malloc(view->length + 1));
  if (!data_buffer) {
    free(zstr);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for string data",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  memcpy(data_buffer, view->data, view->length);
  data_buffer[view->length] = '\0';
  zstr->data = data_buffer;
  zstr->length = view->length;
  zstr->capacity = view->length + 1;

  return zstr;
}

ZVecString *zvec_bin_create(const uint8_t *data, size_t length) {
  if (!data) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Binary data pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  ZVecString *zstr = static_cast<ZVecString *>(malloc(sizeof(ZVecString)));
  if (!zstr) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecString", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  char *data_buffer = static_cast<char *>(malloc(length + 1));
  if (!data_buffer) {
    free(zstr);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for binary data",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  memcpy(data_buffer, data, length);
  data_buffer[length] = '\0';
  zstr->data = data_buffer;
  zstr->length = length;
  zstr->capacity = length + 1;

  return zstr;
}

ZVecString *zvec_string_copy(const ZVecString *str) {
  if (!str || !str->data) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Source string or data cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  return zvec_string_create(str->data);
}

const char *zvec_string_c_str(const ZVecString *str) {
  if (!str) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "String pointer cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return nullptr;
  }

  return str->data;
}

size_t zvec_string_length(const ZVecString *str) {
  if (!str) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "String pointer cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return 0;
  }

  return str->length;
}

int zvec_string_compare(const ZVecString *str1, const ZVecString *str2) {
  if (!str1 || !str2) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "String pointers cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return -1;
  }

  if (!str1->data || !str2->data) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "String data cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return -1;
  }

  return strcmp(str1->data, str2->data);
}

// =============================================================================
// Configuration-related functions implementation
// =============================================================================

ZVecConsoleLogConfig *zvec_config_console_log_create(ZVecLogLevel level) {
  ZVecConsoleLogConfig *config =
      static_cast<ZVecConsoleLogConfig *>(malloc(sizeof(ZVecConsoleLogConfig)));
  if (!config) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecConsoleLogConfig",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  config->level = level;
  return config;
}

ZVecFileLogConfig *zvec_config_file_log_create(ZVecLogLevel level,
                                               const char *dir,
                                               const char *basename,
                                               uint32_t file_size,
                                               uint32_t overdue_days) {
  if (!dir || !basename) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Directory or basename cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  ZVecFileLogConfig *config =
      static_cast<ZVecFileLogConfig *>(malloc(sizeof(ZVecFileLogConfig)));
  if (!config) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecFileLogConfig",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  config->level = level;
  ZVecString *dir_str = zvec_string_create(dir);
  ZVecString *basename_str = zvec_string_create(basename);

  if (!dir_str || !basename_str) {
    if (dir_str) zvec_free_string(dir_str);
    if (basename_str) zvec_free_string(basename_str);
    free(config);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to create strings for file log config",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  config->dir = *dir_str;
  config->basename = *basename_str;
  config->file_size = file_size;
  config->overdue_days = overdue_days;

  // Free the temporary string wrappers (data is copied by value)
  free(dir_str);
  free(basename_str);

  return config;
}

ZVecConfigData *zvec_config_data_create(void) {
  ZVecConfigData *config =
      static_cast<ZVecConfigData *>(malloc(sizeof(ZVecConfigData)));
  if (!config) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecConfigData",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  ZVecConsoleLogConfig *log_config =
      zvec_config_console_log_create(ZVEC_LOG_LEVEL_WARN);
  if (!log_config) {
    free(config);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to create console log config", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }
  config->log_config = log_config;
  config->log_type = ZVEC_LOG_TYPE_CONSOLE;

  // Set default values from C++ ConfigData
  zvec::GlobalConfig::ConfigData config_data;
  config->memory_limit_bytes = config_data.memory_limit_bytes;
  config->query_thread_count = config_data.query_thread_count;
  config->invert_to_forward_scan_ratio =
      config_data.invert_to_forward_scan_ratio;
  config->brute_force_by_keys_ratio = config_data.brute_force_by_keys_ratio;
  config->optimize_thread_count = config_data.optimize_thread_count;

  return config;
}

void zvec_config_console_log_destroy(ZVecConsoleLogConfig *config) {
  if (config) {
    free(config);
  }
}

void zvec_config_file_log_destroy(ZVecFileLogConfig *config) {
  if (config) {
    if (config->dir.data) free((void *)config->dir.data);
    if (config->basename.data) free((void *)config->basename.data);
    free(config);
  }
}

void zvec_config_data_destroy(ZVecConfigData *config) {
  if (config->log_config) {
    if (config->log_type == ZVEC_LOG_TYPE_CONSOLE) {
      zvec_config_console_log_destroy(
          (ZVecConsoleLogConfig *)config->log_config);
    } else {
      zvec_config_file_log_destroy((ZVecFileLogConfig *)config->log_config);
    }
  }
  free(config);
}

ZVecErrorCode zvec_config_data_set_memory_limit(ZVecConfigData *config,
                                                uint64_t memory_limit_bytes) {
  if (!config) {
    set_last_error("Config data pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  config->memory_limit_bytes = memory_limit_bytes;
  return ZVEC_OK;
}

ZVecErrorCode zvec_config_data_set_log_config(ZVecConfigData *config,
                                              ZVecLogType log_type,
                                              void *log_config) {
  if (!config || !log_config) {
    set_last_error("Config data pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (config->log_config) {
    if (config->log_type == ZVEC_LOG_TYPE_CONSOLE) {
      zvec_config_console_log_destroy(
          (ZVecConsoleLogConfig *)config->log_config);
    } else {
      zvec_config_file_log_destroy((ZVecFileLogConfig *)config->log_config);
    }
  }

  config->log_type = log_type;
  config->log_config = log_config;
  return ZVEC_OK;
}

ZVecErrorCode zvec_config_data_set_query_thread_count(ZVecConfigData *config,
                                                      uint32_t thread_count) {
  if (!config) {
    set_last_error("Config data pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  config->query_thread_count = thread_count;
  return ZVEC_OK;
}

ZVecErrorCode zvec_config_data_set_optimize_thread_count(
    ZVecConfigData *config, uint32_t thread_count) {
  if (!config) {
    set_last_error("Config data pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  config->optimize_thread_count = thread_count;
  return ZVEC_OK;
}


// =============================================================================
// Initialization and cleanup interface implementation
// =============================================================================

ZVecErrorCode zvec_initialize(const ZVecConfigData *config) {
  std::lock_guard<std::mutex> lock(g_init_mutex);

  if (g_initialized.load()) {
    set_last_error_details(ZVEC_ERROR_ALREADY_EXISTS,
                           "Library already initialized");
    return ZVEC_ERROR_ALREADY_EXISTS;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Initialization failed",
      // Convert to C++ configuration object
      if (config) {
        zvec::GlobalConfig::ConfigData cpp_config{};
        cpp_config.memory_limit_bytes = config->memory_limit_bytes;
        cpp_config.query_thread_count = config->query_thread_count;
        cpp_config.invert_to_forward_scan_ratio =
            config->invert_to_forward_scan_ratio;
        cpp_config.brute_force_by_keys_ratio =
            config->brute_force_by_keys_ratio;
        cpp_config.optimize_thread_count = config->optimize_thread_count;

        // Set log configuration
        if (config->log_config) {
          std::shared_ptr<zvec::GlobalConfig::LogConfig> log_config;

          switch (config->log_type) {
            case ZVEC_LOG_TYPE_CONSOLE: {
              ZVecConsoleLogConfig *console_config =
                  (ZVecConsoleLogConfig *)config->log_config;
              auto console_level = static_cast<zvec::GlobalConfig::LogLevel>(
                  console_config->level);
              log_config =
                  std::make_shared<zvec::GlobalConfig::ConsoleLogConfig>(
                      console_level);
              break;
            }
            case ZVEC_LOG_TYPE_FILE: {
              ZVecFileLogConfig *file_config =
                  (ZVecFileLogConfig *)config->log_config;
              auto file_level =
                  static_cast<zvec::GlobalConfig::LogLevel>(file_config->level);
              std::string dir(file_config->dir.data, file_config->dir.length);
              std::string basename(file_config->basename.data,
                                   file_config->basename.length);
              log_config = std::make_shared<zvec::GlobalConfig::FileLogConfig>(
                  file_level, dir, basename);
              break;
            }
            default:
              throw std::runtime_error("Unknown log type");
          }
          cpp_config.log_config = log_config;
        }
        // Initialize global configuration
        auto status = zvec::GlobalConfig::Instance().Initialize(cpp_config);
        if (!status.ok()) {
          set_last_error(status.message());
          return ZVEC_ERROR_INTERNAL_ERROR;
        }
      } else {
        // Initialize with default configuration
        zvec::GlobalConfig::ConfigData default_config;
        auto status = zvec::GlobalConfig::Instance().Initialize(default_config);
        if (!status.ok()) {
          set_last_error(status.message());
          return ZVEC_ERROR_INTERNAL_ERROR;
        }
      } g_initialized.store(true);
      return ZVEC_OK;)
}

ZVecErrorCode zvec_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);

  if (!g_initialized.load()) {
    set_last_error_details(ZVEC_ERROR_FAILED_PRECONDITION,
                           "Library not initialized");
    return ZVEC_ERROR_FAILED_PRECONDITION;
  }

  ZVEC_TRY_RETURN_ERROR("Shutdown failed", g_initialized.store(false);
                        return ZVEC_OK;)
}

ZVecErrorCode zvec_is_initialized(bool *initialized) {
  if (!initialized) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Initialized flag pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *initialized = g_initialized.load();
  return ZVEC_OK;
}

// =============================================================================
// Error handling interface implementation
// =============================================================================

ZVecErrorCode zvec_get_last_error_details(ZVecErrorDetails *error_details) {
  if (!error_details) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Error details pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *error_details = last_error_details;
  return ZVEC_OK;
}

void zvec_clear_error(void) {
  last_error_message.clear();
  last_error_details = {};
}

// Helper functions: convert internal status to error code
static ZVecErrorCode status_to_error_code(const zvec::Status &status) {
  if (status.code() < zvec::StatusCode::OK ||
      status.code() > zvec::StatusCode::UNKNOWN) {
    set_last_error("Unexpected status code: " +
                   std::to_string(static_cast<int>(status.code())));
    return ZVEC_ERROR_UNKNOWN;
  }

  return static_cast<ZVecErrorCode>(status.code());
}

// Helper function: handle Expected results
template <typename T>
static ZVecErrorCode handle_expected_result(
    const tl::expected<T, zvec::Status> &result, T *out_value = nullptr) {
  if (result.has_value()) {
    if (out_value) {
      *out_value = result.value();
    }
    return ZVEC_OK;
  } else {
    set_last_error(result.error().message());
    return status_to_error_code(result.error());
  }
}

// Helper function: copy strings
static char *copy_string(const std::string &str) {
  if (str.empty()) return nullptr;
  char *copy = static_cast<char *>(malloc(str.length() + 1));
  strcpy(copy, str.c_str());
  return copy;
}

static zvec::DataType convert_data_type(ZVecDataType zvec_type) {
  if (zvec_type < ZVEC_DATA_TYPE_UNDEFINED ||
      zvec_type > ZVEC_DATA_TYPE_ARRAY_DOUBLE) {
    return zvec::DataType::UNDEFINED;
  }

  return static_cast<zvec::DataType>(zvec_type);
}

static ZVecDataType convert_zvec_data_type(zvec::DataType cpp_type) {
  if (cpp_type < zvec::DataType::UNDEFINED ||
      cpp_type > zvec::DataType::ARRAY_DOUBLE) {
    return ZVEC_DATA_TYPE_UNDEFINED;
  }

  return static_cast<ZVecDataType>(cpp_type);
}

// Helper function: convert metric type
static zvec::MetricType convert_metric_type(ZVecMetricType metric_type) {
  if (metric_type < ZVEC_METRIC_TYPE_UNDEFINED ||
      metric_type > ZVEC_METRIC_TYPE_MIPSL2) {
    return zvec::MetricType::UNDEFINED;
  }

  return static_cast<zvec::MetricType>(metric_type);
}

// Helper function: convert ZVecIndexType to internal IndexType
static zvec::IndexType convert_index_type(ZVecIndexType zvec_type) {
  if (zvec_type < ZVEC_INDEX_TYPE_UNDEFINED ||
      zvec_type > ZVEC_INDEX_TYPE_INVERT) {
    return zvec::IndexType::UNDEFINED;
  }

  return static_cast<zvec::IndexType>(zvec_type);
}

// Helper function: convert ZVecQuantizeType to internal QuantizeType
static zvec::QuantizeType convert_quantize_type(ZVecQuantizeType zvec_type) {
  if (zvec_type < ZVEC_QUANTIZE_TYPE_UNDEFINED ||
      zvec_type > ZVEC_QUANTIZE_TYPE_INT4) {
    return zvec::QuantizeType::UNDEFINED;
  }

  return static_cast<zvec::QuantizeType>(zvec_type);
}

// Helper function: set field index params
static zvec::Status set_field_index_params(zvec::FieldSchema::Ptr &field_schema,
                                           const ZVecFieldSchema *zvec_field) {
  if (!zvec_field->index_params) {
    return zvec::Status::OK();
  }

  switch (zvec_field->index_params->index_type) {
    case ZVEC_INDEX_TYPE_HNSW: {
      const ZVecHnswIndexParams *params =
          &zvec_field->index_params->params.hnsw_params;
      auto metric = convert_metric_type(params->base.metric_type);
      auto quantize = convert_quantize_type(params->base.quantize_type);
      auto index_params = std::make_shared<zvec::HnswIndexParams>(
          metric, params->m, params->ef_construction, quantize);
      field_schema->set_index_params(index_params);
      break;
    }
    case ZVEC_INDEX_TYPE_FLAT: {
      const ZVecFlatIndexParams *params =
          &zvec_field->index_params->params.flat_params;
      auto metric = convert_metric_type(params->base.metric_type);
      auto quantize = convert_quantize_type(params->base.quantize_type);
      auto index_params =
          std::make_shared<zvec::FlatIndexParams>(metric, quantize);
      field_schema->set_index_params(index_params);
      break;
    }
    case ZVEC_INDEX_TYPE_INVERT: {
      const ZVecInvertIndexParams *params =
          &zvec_field->index_params->params.invert_params;
      auto index_params = std::make_shared<zvec::InvertIndexParams>(
          params->enable_range_optimization, params->enable_extended_wildcard);
      field_schema->set_index_params(index_params);
      break;
    }
    default:
      break;
  }

  return zvec::Status::OK();
}

// =============================================================================
// Memory Management interface implementation
// =============================================================================

void zvec_free_string(ZVecString *str) {
  if (str) {
    if (str->data) {
      free((void *)str->data);
    }
    free(str);
  }
}

ZVecStringArray *zvec_string_array_create(size_t count) {
  ZVecStringArray *array = (ZVecStringArray *)malloc(sizeof(ZVecStringArray));
  array->count = count;
  array->strings = (ZVecString *)malloc(sizeof(ZVecString) * count);
  memset(array->strings, 0, sizeof(ZVecString) * count);
  return array;
}

void zvec_string_array_add(ZVecStringArray *array, size_t idx,
                           const char *str) {
  if (idx >= array->count) return;
  size_t len = strlen(str);
  array->strings[idx].data = (char *)malloc(len + 1);
  memcpy(array->strings[idx].data, str, len + 1);
  array->strings[idx].length = len;
  array->strings[idx].capacity = len + 1;
}

void zvec_string_array_destroy(ZVecStringArray *array) {
  if (!array) return;
  for (size_t i = 0; i < array->count; i++) {
    free((void *)array->strings[i].data);
  }
  free(array->strings);
  free(array);
}


// Byte array helper functions
ZVecMutableByteArray *zvec_byte_array_create(size_t capacity) {
  ZVecMutableByteArray *array =
      (ZVecMutableByteArray *)malloc(sizeof(ZVecMutableByteArray));
  if (!array) return nullptr;

  array->data = (uint8_t *)malloc(capacity);
  if (!array->data) {
    free(array);
    return nullptr;
  }

  array->length = 0;
  array->capacity = capacity;
  memset(array->data, 0, capacity);
  return array;
}

void zvec_byte_array_destroy(ZVecMutableByteArray *array) {
  if (!array) return;
  if (array->data) {
    free(array->data);
  }
  free(array);
}

// Float array helper functions
ZVecFloatArray *zvec_float_array_create(size_t count) {
  ZVecFloatArray *array = (ZVecFloatArray *)malloc(sizeof(ZVecFloatArray));
  if (!array) return nullptr;

  array->data = (const float *)malloc(sizeof(float) * count);
  if (!array->data) {
    free(array);
    return nullptr;
  }

  array->length = count;
  memset((void *)array->data, 0, sizeof(float) * count);
  return array;
}

void zvec_float_array_destroy(ZVecFloatArray *array) {
  if (!array) return;
  if (array->data) {
    free((void *)array->data);
  }
  free(array);
}

// Int64 array helper functions
ZVecInt64Array *zvec_int64_array_create(size_t count) {
  ZVecInt64Array *array = (ZVecInt64Array *)malloc(sizeof(ZVecInt64Array));
  if (!array) return nullptr;

  array->data = (const int64_t *)malloc(sizeof(int64_t) * count);
  if (!array->data) {
    free(array);
    return nullptr;
  }

  array->length = count;
  memset((void *)array->data, 0, sizeof(int64_t) * count);
  return array;
}

void zvec_int64_array_destroy(ZVecInt64Array *array) {
  if (!array) return;
  if (array->data) {
    free((void *)array->data);
  }
  free(array);
}

void zvec_free_float_array(float *array) {
  if (array) {
    free(array);
  }
}

void zvec_free_str_array(char **array, size_t count) {
  if (!array) return;

  // If count is 0, only free the string array itself, don't process internal
  // strings
  if (count == 0) {
    free(array);
    return;
  }

  for (size_t i = 0; i < count; ++i) {
    if (array[i]) {  // Only free when string pointer is not null
      free(array[i]);
    }
  }
  free(array);
}

ZVecErrorCode zvec_get_last_error(char **error_msg) {
  if (!error_msg) {
    set_last_error("Invalid argument: error_msg cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *error_msg = copy_string(last_error_message);
  return ZVEC_OK;
}

void zvec_free_uint8_array(uint8_t *array) {
  if (array) {
    free(array);
  }
}

void zvec_free_field_schema(ZVecFieldSchema *field_schema) {
  if (field_schema) {
    if (field_schema->index_params) {
      zvec_index_params_destroy(field_schema->index_params);
    }
    free(field_schema);
  }
}

// =============================================================================
// Index parameters management interface implementation
// =============================================================================

void zvec_index_params_base_init(ZVecBaseIndexParams *params,
                                 ZVecIndexType index_type) {
  if (params) {
    params->index_type = index_type;
  }
}

void zvec_index_params_invert_init(ZVecInvertIndexParams *params,
                                   bool enable_range_opt,
                                   bool enable_wildcard) {
  if (params) {
    zvec_index_params_base_init(&params->base, ZVEC_INDEX_TYPE_INVERT);
    params->enable_range_optimization = enable_range_opt;
    params->enable_extended_wildcard = enable_wildcard;
  }
}

void zvec_index_params_vector_init(ZVecVectorIndexParams *params,
                                   ZVecIndexType index_type,
                                   ZVecMetricType metric_type,
                                   ZVecQuantizeType quantize_type) {
  if (params) {
    zvec_index_params_base_init(&params->base, index_type);
    params->metric_type = metric_type;
    params->quantize_type = quantize_type;
  }
}

void zvec_index_params_hnsw_init(ZVecHnswIndexParams *params,
                                 ZVecMetricType metric_type, int m,
                                 int ef_construction, int ef_search,
                                 ZVecQuantizeType quantize_type) {
  if (params) {
    zvec_index_params_vector_init(&params->base, ZVEC_INDEX_TYPE_HNSW,
                                  metric_type, quantize_type);
    params->m = m;
    params->ef_construction = ef_construction;
    params->ef_search = ef_search;
  }
}

void zvec_index_params_flat_init(ZVecFlatIndexParams *params,
                                 ZVecMetricType metric_type,
                                 ZVecQuantizeType quantize_type) {
  if (params) {
    zvec_index_params_vector_init(&params->base, ZVEC_INDEX_TYPE_FLAT,
                                  metric_type, quantize_type);
  }
}

void zvec_index_params_ivf_init(ZVecIVFIndexParams *params,
                                ZVecMetricType metric_type, int n_list,
                                int n_iters, bool use_soar, int n_probe,
                                ZVecQuantizeType quantize_type) {
  if (params) {
    zvec_index_params_vector_init(&params->base, ZVEC_INDEX_TYPE_IVF,
                                  metric_type, quantize_type);
    params->n_list = n_list;
    params->n_iters = n_iters;
    params->use_soar = use_soar;
    params->n_probe = n_probe;
  }
}

void zvec_index_params_init_default(ZVecIndexParams *params,
                                    ZVecIndexType index_type,
                                    ZVecMetricType metric_type) {
  if (!params) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Index params pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return;
  }

  params->index_type = index_type;

  switch (index_type) {
    case ZVEC_INDEX_TYPE_INVERT:
      zvec_index_params_invert_init(&params->params.invert_params, false,
                                    false);
      break;

    case ZVEC_INDEX_TYPE_HNSW:
      zvec_index_params_hnsw_init(&params->params.hnsw_params, metric_type, 16,
                                  200, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
      break;

    case ZVEC_INDEX_TYPE_FLAT:
      zvec_index_params_flat_init(&params->params.flat_params, metric_type,
                                  ZVEC_QUANTIZE_TYPE_UNDEFINED);
      break;

    case ZVEC_INDEX_TYPE_IVF:
      zvec_index_params_ivf_init(&params->params.ivf_params, metric_type, 100,
                                 10, false, 10, ZVEC_QUANTIZE_TYPE_UNDEFINED);
      break;

    default:
      set_last_error_details(ZVEC_ERROR_NOT_SUPPORTED, "Unsupported index type",
                             __FILE__, __LINE__, __FUNCTION__);
      break;
  }
}

void zvec_index_params_destroy(ZVecIndexParams *params) {
  if (params) {
    free(params);
  }
}

ZVecInvertIndexParams *zvec_index_params_invert_create(bool enable_range_opt,
                                                       bool enable_wildcard) {
  ZVecInvertIndexParams *params = static_cast<ZVecInvertIndexParams *>(
      malloc(sizeof(ZVecInvertIndexParams)));
  if (!params) {
    set_last_error_details(
        ZVEC_ERROR_RESOURCE_EXHAUSTED,
        "Failed to allocate memory for ZVecInvertIndexParams", __FILE__,
        __LINE__, __FUNCTION__);
    return nullptr;
  }
  zvec_index_params_base_init(&params->base, ZVEC_INDEX_TYPE_INVERT);
  params->enable_range_optimization = enable_range_opt;
  params->enable_extended_wildcard = enable_wildcard;
  return params;
}

ZVecVectorIndexParams *zvec_index_params_vector_create(
    ZVecIndexType index_type, ZVecMetricType metric_type,
    ZVecQuantizeType quantize_type) {
  ZVecVectorIndexParams *params = static_cast<ZVecVectorIndexParams *>(
      malloc(sizeof(ZVecVectorIndexParams)));
  if (!params) {
    set_last_error_details(
        ZVEC_ERROR_RESOURCE_EXHAUSTED,
        "Failed to allocate memory for ZVecVectorIndexParams", __FILE__,
        __LINE__, __FUNCTION__);
    return nullptr;
  }
  zvec_index_params_base_init(&params->base, index_type);
  params->metric_type = metric_type;
  params->quantize_type = quantize_type;
  return params;
}

ZVecHnswIndexParams *zvec_index_params_hnsw_create(
    ZVecMetricType metric_type, ZVecQuantizeType quantize_type, int m,
    int ef_construction, int ef_search) {
  ZVecHnswIndexParams *params =
      static_cast<ZVecHnswIndexParams *>(malloc(sizeof(ZVecHnswIndexParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecHnswIndexParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  zvec_index_params_vector_init(&params->base, ZVEC_INDEX_TYPE_HNSW,
                                metric_type, quantize_type);
  params->m = m;
  params->ef_construction = ef_construction;
  params->ef_search = ef_search;
  return params;
}

ZVecFlatIndexParams *zvec_index_params_flat_create(
    ZVecMetricType metric_type, ZVecQuantizeType quantize_type) {
  ZVecFlatIndexParams *params =
      static_cast<ZVecFlatIndexParams *>(malloc(sizeof(ZVecFlatIndexParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecFlatIndexParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  zvec_index_params_vector_init(&params->base, ZVEC_INDEX_TYPE_FLAT,
                                metric_type, quantize_type);
  return params;
}

ZVecIVFIndexParams *zvec_index_params_ivf_create(ZVecMetricType metric_type,
                                                 ZVecQuantizeType quantize_type,
                                                 int n_list, int n_iters,
                                                 bool use_soar, int n_probe) {
  ZVecIVFIndexParams *params =
      static_cast<ZVecIVFIndexParams *>(malloc(sizeof(ZVecIVFIndexParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecIVFIndexParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  zvec_index_params_vector_init(&params->base, ZVEC_INDEX_TYPE_IVF, metric_type,
                                quantize_type);
  params->n_list = n_list;
  params->n_iters = n_iters;
  params->use_soar = use_soar;
  params->n_probe = n_probe;
  return params;
}

void zvec_index_params_invert_destroy(ZVecInvertIndexParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_index_params_vector_destroy(ZVecVectorIndexParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_index_params_hnsw_destroy(ZVecHnswIndexParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_index_params_flat_destroy(ZVecFlatIndexParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_index_params_ivf_destroy(ZVecIVFIndexParams *params) {
  if (params) {
    free(params);
  }
}

// =============================================================================
// FieldSchema management interface implementation
// =============================================================================

ZVecFieldSchema *zvec_field_schema_create(const char *name,
                                          ZVecDataType data_type, bool nullable,
                                          uint32_t dimension) {
  if (!name) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Field name cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return nullptr;
  }

  ZVecFieldSchema *schema =
      static_cast<ZVecFieldSchema *>(malloc(sizeof(ZVecFieldSchema)));
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecFieldSchema",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  schema->name = zvec_string_create(name);
  if (!schema->name) {
    free(schema);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to create string for field name", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  schema->data_type = data_type;
  schema->nullable = nullable;
  schema->dimension = dimension;
  schema->index_params = nullptr;

  return schema;
}

void zvec_field_schema_destroy(ZVecFieldSchema *schema) {
  if (schema) {
    zvec_free_string(schema->name);
    if (schema->index_params) {
      zvec_index_params_destroy(schema->index_params);
      schema->index_params = nullptr;
    }
    free(schema);
  }
}

ZVecErrorCode zvec_field_schema_set_index_params(
    ZVecFieldSchema *schema, const ZVecIndexParams *index_params) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Field schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!index_params) {
    if (schema->index_params) {
      zvec_index_params_destroy(schema->index_params);
      free(schema->index_params);
      schema->index_params = nullptr;
    }
    return ZVEC_OK;
  }

  if (!schema->index_params) {
    schema->index_params =
        static_cast<ZVecIndexParams *>(malloc(sizeof(ZVecIndexParams)));
    if (!schema->index_params) {
      set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                             "Failed to allocate memory for ZVecIndexParams",
                             __FILE__, __LINE__, __FUNCTION__);
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }
  }

  *schema->index_params = *index_params;

  return ZVEC_OK;
}

void zvec_field_schema_set_invert_index(
    ZVecFieldSchema *field_schema, const ZVecInvertIndexParams *invert_params) {
  if (field_schema && invert_params) {
    if (!field_schema->index_params) {
      field_schema->index_params =
          static_cast<ZVecIndexParams *>(malloc(sizeof(ZVecIndexParams)));
      if (!field_schema->index_params) {
        set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                               "Failed to allocate memory for ZVecIndexParams",
                               __FILE__, __LINE__, __FUNCTION__);
        return;
      }
    }

    field_schema->index_params->index_type = ZVEC_INDEX_TYPE_INVERT;
    field_schema->index_params->params.invert_params = *invert_params;
  }
}

void zvec_field_schema_set_hnsw_index(ZVecFieldSchema *field_schema,
                                      const ZVecHnswIndexParams *hnsw_params) {
  if (field_schema && hnsw_params) {
    if (!field_schema->index_params) {
      field_schema->index_params =
          static_cast<ZVecIndexParams *>(malloc(sizeof(ZVecIndexParams)));
      if (!field_schema->index_params) {
        set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                               "Failed to allocate memory for ZVecIndexParams",
                               __FILE__, __LINE__, __FUNCTION__);
        return;
      }
    }

    field_schema->index_params->index_type = ZVEC_INDEX_TYPE_HNSW;
    field_schema->index_params->params.hnsw_params = *hnsw_params;
  }
}

void zvec_field_schema_set_flat_index(ZVecFieldSchema *field_schema,
                                      const ZVecFlatIndexParams *flat_params) {
  if (field_schema && flat_params) {
    if (!field_schema->index_params) {
      field_schema->index_params =
          static_cast<ZVecIndexParams *>(malloc(sizeof(ZVecIndexParams)));
      if (!field_schema->index_params) {
        set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                               "Failed to allocate memory for ZVecIndexParams",
                               __FILE__, __LINE__, __FUNCTION__);
        return;
      }
    }

    field_schema->index_params->index_type = ZVEC_INDEX_TYPE_FLAT;
    field_schema->index_params->params.flat_params = *flat_params;
  }
}

void zvec_field_schema_set_ivf_index(ZVecFieldSchema *field_schema,
                                     const ZVecIVFIndexParams *ivf_params) {
  if (field_schema && ivf_params) {
    if (!field_schema->index_params) {
      field_schema->index_params =
          static_cast<ZVecIndexParams *>(malloc(sizeof(ZVecIndexParams)));
      if (!field_schema->index_params) {
        set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                               "Failed to allocate memory for ZVecIndexParams",
                               __FILE__, __LINE__, __FUNCTION__);
        return;
      }
    }

    field_schema->index_params->index_type = ZVEC_INDEX_TYPE_IVF;
    field_schema->index_params->params.ivf_params = *ivf_params;
  }
}

static void zvec_field_schema_cleanup(ZVecFieldSchema *field_schema) {
  if (!field_schema) return;

  if (field_schema->index_params) {
    zvec_index_params_destroy(field_schema->index_params);
    free(field_schema->index_params);
    field_schema->index_params = nullptr;
  }

  zvec_free_string(field_schema->name);
  field_schema->name = nullptr;
}

// =============================================================================
// CollectionOptions management interface implementation
// =============================================================================

void zvec_collection_options_init_default(ZVecCollectionOptions *options) {
  if (!options) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection options pointer cannot be null",
                           __FILE__, __LINE__, __FUNCTION__);
    return;
  }

  options->enable_mmap = true;
  options->max_buffer_size = zvec::DEFAULT_MAX_BUFFER_SIZE;
  options->read_only = false;
  options->max_doc_count_per_segment = zvec::MAX_DOC_COUNT_PER_SEGMENT;
}

// =============================================================================
// CollectionSchema management interface implementation
// =============================================================================

ZVecCollectionSchema *zvec_collection_schema_create(const char *name) {
  if (!name) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection name cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return nullptr;
  }

  ZVecCollectionSchema *schema =
      static_cast<ZVecCollectionSchema *>(malloc(sizeof(ZVecCollectionSchema)));
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecCollectionSchema",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  schema->name = zvec_string_create(name);
  if (!schema->name) {
    free(schema);
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to create string for collection name",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }

  schema->fields = nullptr;
  schema->field_count = 0;
  schema->field_capacity = 0;
  schema->max_doc_count_per_segment = zvec::MAX_DOC_COUNT_PER_SEGMENT;

  return schema;
}

void zvec_collection_schema_destroy(ZVecCollectionSchema *schema) {
  if (schema) {
    zvec_free_string(schema->name);

    if (schema->fields) {
      for (size_t i = 0; i < schema->field_count; ++i) {
        zvec_field_schema_destroy(schema->fields[i]);
      }
      free(schema->fields);
    }

    free(schema);
  }
}

ZVecErrorCode zvec_collection_schema_add_field(ZVecCollectionSchema *schema,
                                               ZVecFieldSchema *field) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!field || !field->name) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Field or field name cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    if (schema->fields[i]->name && field->name &&
        zvec_string_compare(schema->fields[i]->name, field->name) == 0) {
      set_last_error_details(
          ZVEC_ERROR_ALREADY_EXISTS,
          std::string("Field '") + field->name->data + "' already exists",
          __FILE__, __LINE__, __FUNCTION__);
      return ZVEC_ERROR_ALREADY_EXISTS;
    }
  }

  if (schema->field_count >= schema->field_capacity) {
    size_t new_capacity =
        schema->field_capacity == 0 ? 8 : schema->field_capacity * 2;
    ZVecFieldSchema **new_fields = static_cast<ZVecFieldSchema **>(
        malloc(new_capacity * sizeof(ZVecFieldSchema *)));
    if (!new_fields) {
      set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                             "Failed to allocate memory for fields", __FILE__,
                             __LINE__, __FUNCTION__);
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }

    for (size_t i = 0; i < schema->field_count; ++i) {
      new_fields[i] = schema->fields[i];
    }

    free(schema->fields);
    schema->fields = new_fields;
    schema->field_capacity = new_capacity;
  }

  schema->fields[schema->field_count] = field;
  schema->field_count++;

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_add_fields(ZVecCollectionSchema *schema,
                                                const ZVecFieldSchema *fields,
                                                size_t field_count) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!fields && field_count > 0) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Fields array cannot be null when field_count > 0",
                           __FILE__, __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (field_count == 0) {
    return ZVEC_OK;
  }

  for (size_t i = 0; i < field_count; ++i) {
    const ZVecFieldSchema &field = fields[i];
    if (!field.name || !field.name->data || field.name->length == 0) {
      set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                             std::string("Field at index ") +
                                 std::to_string(i) + " has invalid name",
                             __FILE__, __LINE__, __FUNCTION__);
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
  }

  size_t total_needed = schema->field_count + field_count;
  if (total_needed > schema->field_capacity) {
    size_t new_capacity = schema->field_capacity;
    while (new_capacity < total_needed) {
      new_capacity = new_capacity == 0 ? 8 : new_capacity * 2;
    }

    ZVecFieldSchema **new_fields = static_cast<ZVecFieldSchema **>(
        malloc(new_capacity * sizeof(ZVecFieldSchema *)));
    if (!new_fields) {
      set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                             "Failed to allocate memory for fields", __FILE__,
                             __LINE__, __FUNCTION__);
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }

    for (size_t i = 0; i < schema->field_count; ++i) {
      new_fields[i] = schema->fields[i];
    }

    free(schema->fields);
    schema->fields = new_fields;
    schema->field_capacity = new_capacity;
  }

  for (size_t i = 0; i < field_count; ++i) {
    const ZVecFieldSchema &src_field = fields[i];

    ZVecFieldSchema *new_field =
        static_cast<ZVecFieldSchema *>(malloc(sizeof(ZVecFieldSchema)));
    if (!new_field) {
      set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                             "Failed to allocate memory for new field",
                             __FILE__, __LINE__, __FUNCTION__);
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }

    new_field->name = zvec_string_copy(src_field.name);
    if (!new_field->name) {
      free(new_field);
      set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                             "Failed to copy field name", __FILE__, __LINE__,
                             __FUNCTION__);
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }

    new_field->data_type = src_field.data_type;
    new_field->nullable = src_field.nullable;
    new_field->dimension = src_field.dimension;

    if (src_field.index_params) {
      new_field->index_params =
          static_cast<ZVecIndexParams *>(malloc(sizeof(ZVecIndexParams)));
      if (!new_field->index_params) {
        zvec_free_string(new_field->name);
        free(new_field);
        set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                               "Failed to allocate memory for index params",
                               __FILE__, __LINE__, __FUNCTION__);
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }
      *(new_field->index_params) = *(src_field.index_params);
    } else {
      new_field->index_params = nullptr;
    }

    schema->fields[schema->field_count] = new_field;
    schema->field_count++;
  }

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_remove_field(ZVecCollectionSchema *schema,
                                                  const char *field_name) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!field_name) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Field name cannot be null", __FILE__, __LINE__,
                           __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    if (schema->fields[i]->name &&
        strcmp(schema->fields[i]->name->data, field_name) == 0) {
      zvec_field_schema_destroy(schema->fields[i]);

      for (size_t j = i; j < schema->field_count - 1; ++j) {
        schema->fields[j] = schema->fields[j + 1];
      }

      schema->field_count--;
      return ZVEC_OK;
    }
  }

  set_last_error_details(ZVEC_ERROR_NOT_FOUND,
                         std::string("Field '") + field_name + "' not found",
                         __FILE__, __LINE__, __FUNCTION__);
  return ZVEC_ERROR_NOT_FOUND;
}

ZVecErrorCode zvec_collection_schema_remove_fields(
    ZVecCollectionSchema *schema, const char *const *field_names,
    size_t field_count) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!field_names && field_count > 0) {
    set_last_error_details(
        ZVEC_ERROR_INVALID_ARGUMENT,
        "Field names array cannot be null when field_count > 0", __FILE__,
        __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (field_count == 0) {
    return ZVEC_OK;
  }

  for (size_t i = 0; i < field_count; ++i) {
    if (!field_names[i]) {
      set_last_error_details(
          ZVEC_ERROR_INVALID_ARGUMENT,
          std::string("Field name at index ") + std::to_string(i) + " is null",
          __FILE__, __LINE__, __FUNCTION__);
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
  }

  std::vector<size_t> remove_indices;
  std::vector<std::string> not_found_fields;

  for (size_t field_idx = 0; field_idx < field_count; ++field_idx) {
    std::string target_name(field_names[field_idx]);
    bool found = false;

    for (size_t i = 0; i < schema->field_count; ++i) {
      if (schema->fields[i]->name &&
          strcmp(schema->fields[i]->name->data, target_name.c_str()) == 0) {
        remove_indices.push_back(i);
        found = true;
        break;
      }
    }

    if (!found) {
      not_found_fields.push_back(target_name);
    }
  }


  if (!not_found_fields.empty()) {
    std::string error_msg = "Fields not found: ";
    for (size_t i = 0; i < not_found_fields.size(); ++i) {
      error_msg += "'" + not_found_fields[i] + "'";
      if (i < not_found_fields.size() - 1) {
        error_msg += ", ";
      }
    }
    set_last_error_details(ZVEC_ERROR_NOT_FOUND, error_msg, __FILE__, __LINE__,
                           __FUNCTION__);
    return ZVEC_ERROR_NOT_FOUND;
  }

  std::sort(remove_indices.begin(), remove_indices.end(),
            std::greater<size_t>());

  for (size_t remove_index : remove_indices) {
    zvec_field_schema_destroy(schema->fields[remove_index]);

    for (size_t j = remove_index; j < schema->field_count - 1; ++j) {
      schema->fields[j] = schema->fields[j + 1];
    }

    schema->field_count--;
  }

  return ZVEC_OK;
}

ZVecFieldSchema *zvec_collection_schema_find_field(
    const ZVecCollectionSchema *schema, const char *field_name) {
  if (!schema || !field_name) {
    return nullptr;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    if (schema->fields[i]->name &&
        strcmp(schema->fields[i]->name->data, field_name) == 0) {
      return schema->fields[i];
    }
  }

  return nullptr;
}

size_t zvec_collection_schema_get_field_count(
    const ZVecCollectionSchema *schema) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return 0;
  }

  return schema->field_count;
}

ZVecFieldSchema *zvec_collection_schema_get_field(
    const ZVecCollectionSchema *schema, size_t index) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return nullptr;
  }

  if (index >= schema->field_count) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Field index out of bounds", __FILE__, __LINE__,
                           __FUNCTION__);
    return nullptr;
  }

  return schema->fields[index];
}

ZVecErrorCode zvec_collection_schema_set_max_doc_count_per_segment(
    ZVecCollectionSchema *schema, uint64_t max_doc_count) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  schema->max_doc_count_per_segment = max_doc_count;
  return ZVEC_OK;
}

uint64_t zvec_collection_schema_get_max_doc_count_per_segment(
    const ZVecCollectionSchema *schema) {
  if (!schema) return 0;
  return schema->max_doc_count_per_segment;
}

ZVecErrorCode zvec_collection_schema_validate(
    const ZVecCollectionSchema *schema, ZVecString **error_msg) {
  if (!schema) {
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection schema pointer cannot be null", __FILE__,
                           __LINE__, __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (error_msg) {
    *error_msg = nullptr;
  }

  if (!schema->name) {
    if (error_msg) {
      *error_msg = zvec_string_create("Collection name is required");
    }
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "Collection name is required", __FILE__, __LINE__,
                           __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (schema->field_count == 0) {
    if (error_msg) {
      *error_msg = zvec_string_create("At least one field is required");
    }
    set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                           "At least one field is required", __FILE__, __LINE__,
                           __FUNCTION__);
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    auto field = schema->fields[i];
    if (!field) {
      if (error_msg) {
        *error_msg = zvec_string_create("Null field found");
      }
      set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT, "Null field found",
                             __FILE__, __LINE__, __FUNCTION__);
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    if (!field->name) {
      if (error_msg) {
        *error_msg = zvec_string_create("Field name is required");
      }
      set_last_error_details(ZVEC_ERROR_INVALID_ARGUMENT,
                             "Field name is required", __FILE__, __LINE__,
                             __FUNCTION__);
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
  }

  return ZVEC_OK;
}

void zvec_collection_schema_cleanup(ZVecCollectionSchema *schema) {
  if (!schema) return;

  ZVEC_TRY_BEGIN_VOID
  if (schema->name) {
    zvec_free_string(schema->name);
  }

  if (schema->fields) {
    for (size_t i = 0; i < schema->field_count; ++i) {
      zvec_field_schema_cleanup(schema->fields[i]);
    }
    delete[] schema->fields;
    schema->fields = nullptr;
    schema->field_count = 0;
  }

  schema->max_doc_count_per_segment = 0;
  ZVEC_CATCH_END_VOID
}

// =============================================================================
// Helper functions
// =============================================================================

const char *zvec_error_code_to_string(ZVecErrorCode error_code) {
  switch (error_code) {
    case ZVEC_OK:
      return "OK";
    case ZVEC_ERROR_NOT_FOUND:
      return "NOT_FOUND";
    case ZVEC_ERROR_ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case ZVEC_ERROR_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case ZVEC_ERROR_PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case ZVEC_ERROR_FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case ZVEC_ERROR_RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case ZVEC_ERROR_UNAVAILABLE:
      return "UNAVAILABLE";
    case ZVEC_ERROR_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case ZVEC_ERROR_NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    case ZVEC_ERROR_UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNKNOWN_ERROR_CODE";
  }
}

const char *zvec_data_type_to_string(ZVecDataType data_type) {
  switch (data_type) {
    case ZVEC_DATA_TYPE_UNDEFINED:
      return "UNDEFINED";
    case ZVEC_DATA_TYPE_BINARY:
      return "BINARY";
    case ZVEC_DATA_TYPE_STRING:
      return "STRING";
    case ZVEC_DATA_TYPE_BOOL:
      return "BOOL";
    case ZVEC_DATA_TYPE_INT32:
      return "INT32";
    case ZVEC_DATA_TYPE_INT64:
      return "INT64";
    case ZVEC_DATA_TYPE_UINT32:
      return "UINT32";
    case ZVEC_DATA_TYPE_UINT64:
      return "UINT64";
    case ZVEC_DATA_TYPE_FLOAT:
      return "FLOAT";
    case ZVEC_DATA_TYPE_DOUBLE:
      return "DOUBLE";
    case ZVEC_DATA_TYPE_VECTOR_BINARY32:
      return "VECTOR_BINARY32";
    case ZVEC_DATA_TYPE_VECTOR_BINARY64:
      return "VECTOR_BINARY64";
    case ZVEC_DATA_TYPE_VECTOR_FP16:
      return "VECTOR_FP16";
    case ZVEC_DATA_TYPE_VECTOR_FP32:
      return "VECTOR_FP32";
    case ZVEC_DATA_TYPE_VECTOR_FP64:
      return "VECTOR_FP64";
    case ZVEC_DATA_TYPE_VECTOR_INT4:
      return "VECTOR_INT4";
    case ZVEC_DATA_TYPE_VECTOR_INT8:
      return "VECTOR_INT8";
    case ZVEC_DATA_TYPE_VECTOR_INT16:
      return "VECTOR_INT16";
    case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16:
      return "SPARSE_VECTOR_FP16";
    case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32:
      return "SPARSE_VECTOR_FP32";
    case ZVEC_DATA_TYPE_ARRAY_BINARY:
      return "ARRAY_BINARY";
    case ZVEC_DATA_TYPE_ARRAY_STRING:
      return "ARRAY_STRING";
    case ZVEC_DATA_TYPE_ARRAY_BOOL:
      return "ARRAY_BOOL";
    case ZVEC_DATA_TYPE_ARRAY_INT32:
      return "ARRAY_INT32";
    case ZVEC_DATA_TYPE_ARRAY_INT64:
      return "ARRAY_INT64";
    case ZVEC_DATA_TYPE_ARRAY_UINT32:
      return "ARRAY_UINT32";
    case ZVEC_DATA_TYPE_ARRAY_UINT64:
      return "ARRAY_UINT64";
    case ZVEC_DATA_TYPE_ARRAY_FLOAT:
      return "ARRAY_FLOAT";
    case ZVEC_DATA_TYPE_ARRAY_DOUBLE:
      return "ARRAY_DOUBLE";
    default:
      return "UNKNOWN_DATA_TYPE";
  }
}

const char *zvec_index_type_to_string(ZVecIndexType index_type) {
  switch (index_type) {
    case ZVEC_INDEX_TYPE_UNDEFINED:
      return "UNDEFINED";
    case ZVEC_INDEX_TYPE_HNSW:
      return "HNSW";
    case ZVEC_INDEX_TYPE_IVF:
      return "IVF";
    case ZVEC_INDEX_TYPE_FLAT:
      return "FLAT";
    case ZVEC_INDEX_TYPE_INVERT:
      return "INVERT";
    default:
      return "UNKNOWN_INDEX_TYPE";
  }
}

const char *zvec_metric_type_to_string(ZVecMetricType metric_type) {
  switch (metric_type) {
    case ZVEC_METRIC_TYPE_UNDEFINED:
      return "UNDEFINED";
    case ZVEC_METRIC_TYPE_L2:
      return "L2";
    case ZVEC_METRIC_TYPE_IP:
      return "IP";
    case ZVEC_METRIC_TYPE_COSINE:
      return "COSINE";
    case ZVEC_METRIC_TYPE_MIPSL2:
      return "MIPSL2";
    default:
      return "UNKNOWN_METRIC_TYPE";
  }
}

bool check_is_vector_field(const ZVecFieldSchema &zvec_field) {
  bool is_vector_field =
      (zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_FP32 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_FP64 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_FP16 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_BINARY32 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_BINARY64 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_INT4 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_INT8 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_VECTOR_INT16 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 ||
       zvec_field.data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16);
  return is_vector_field;
}

// =============================================================================
// Doc functions implementation
// =============================================================================

ZVecDoc *zvec_doc_create(void) {
  ZVEC_TRY_RETURN_NULL("Failed to create document", {
    auto doc_ptr =
        new std::shared_ptr<zvec::Doc>(std::make_shared<zvec::Doc>());
    return reinterpret_cast<ZVecDoc *>(doc_ptr);
  })
}

void zvec_doc_destroy(ZVecDoc *doc) {
  if (doc) {
    delete reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
  }
}

void zvec_doc_clear(ZVecDoc *doc) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
  (*doc_ptr)->clear();
  ZVEC_CATCH_END_VOID
}

void zvec_docs_free(ZVecDoc **docs, size_t count) {
  if (!docs) return;

  for (size_t i = 0; i < count; ++i) {
    zvec_doc_destroy(docs[i]);
  }

  free(docs);
}

void zvec_doc_set_pk(ZVecDoc *doc, const char *pk) {
  if (!doc || !pk) return;

  ZVEC_TRY_BEGIN_VOID
  auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
  (*doc_ptr)->set_pk(std::string(pk));
  ZVEC_CATCH_END_VOID
}

void zvec_doc_set_doc_id(ZVecDoc *doc, uint64_t doc_id) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
  (*doc_ptr)->set_doc_id(doc_id);
  ZVEC_CATCH_END_VOID
}

void zvec_doc_set_score(ZVecDoc *doc, float score) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
  (*doc_ptr)->set_score(score);
  ZVEC_CATCH_END_VOID
}

void zvec_doc_set_operator(ZVecDoc *doc, ZVecDocOperator op) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
  (*doc_ptr)->set_operator(static_cast<zvec::Operator>(op));
  ZVEC_CATCH_END_VOID
}

// =============================================================================
// Document interface implementation
// =============================================================================

// Helper function to extract scalar values from raw data
template <typename T>
T extract_scalar_value(const void *value, size_t value_size,
                       ZVecErrorCode *error_code) {
  if (value_size != sizeof(T)) {
    if (error_code) {
      *error_code = ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return T{};
  }
  return *static_cast<const T *>(value);
}

// Helper function to extract vector values from raw data
template <typename T>
std::vector<T> extract_vector_values(const void *value, size_t value_size,
                                     ZVecErrorCode *error_code) {
  if (value_size % sizeof(T) != 0) {
    if (error_code) {
      *error_code = ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return std::vector<T>();
  }
  size_t count = value_size / sizeof(T);
  const T *vals = static_cast<const T *>(value);
  return std::vector<T>(vals, vals + count);
}

// Helper function to extract array values from raw data
template <typename T>
std::vector<T> extract_array_values(const void *value, size_t value_size,
                                    ZVecErrorCode *error_code) {
  if (value_size % sizeof(T) != 0) {
    if (error_code) {
      *error_code = ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return std::vector<T>();
  }
  size_t count = value_size / sizeof(T);
  const T *vals = static_cast<const T *>(value);
  return std::vector<T>(vals, vals + count);
}

// Helper function to handle sparse vector extraction
template <typename T>
std::pair<std::vector<uint32_t>, std::vector<T>> extract_sparse_vector(
    const void *value, size_t value_size, ZVecErrorCode *error_code) {
  if (value_size < sizeof(uint32_t)) {
    if (error_code) {
      *error_code = ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return std::make_pair(std::vector<uint32_t>(), std::vector<T>());
  }

  const uint32_t *data = static_cast<const uint32_t *>(value);
  uint32_t nnz = data[0];

  size_t required_size =
      sizeof(uint32_t) + nnz * (sizeof(uint32_t) + sizeof(T));
  if (value_size < required_size) {
    if (error_code) {
      *error_code = ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return std::make_pair(std::vector<uint32_t>(), std::vector<T>());
  }

  const uint32_t *indices = data + 1;
  const T *values = reinterpret_cast<const T *>(indices + nnz);

  std::vector<uint32_t> index_vec(indices, indices + nnz);
  std::vector<T> value_vec(values, values + nnz);

  return std::make_pair(std::move(index_vec), std::move(value_vec));
}

// Helper function to extract string array from raw data (C-string array)
std::vector<std::string> extract_string_array(const void *value,
                                              size_t value_size) {
  std::vector<std::string> string_array;
  const char *data = static_cast<const char *>(value);
  size_t pos = 0;

  while (pos < value_size) {
    size_t str_len = strlen(data + pos);
    if (pos + str_len >= value_size) {
      break;
    }
    string_array.emplace_back(data + pos, str_len);
    pos += str_len + 1;
  }
  return string_array;
}

// Helper function to extract string array from ZVecString** array
std::vector<std::string> extract_string_array_from_zvec(
    ZVecString **zvec_strings, size_t count) {
  std::vector<std::string> string_array;
  string_array.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    if (zvec_strings[i] && zvec_strings[i]->data) {
      string_array.emplace_back(zvec_strings[i]->data, zvec_strings[i]->length);
    } else {
      string_array.emplace_back("", 0);
    }
  }

  return string_array;
}

// Helper function to extract binary array from raw data
std::vector<std::string> extract_binary_array(const void *value,
                                              size_t value_size) {
  std::vector<std::string> binary_array;
  const char *data = static_cast<const char *>(value);
  size_t pos = 0;

  while (pos < value_size) {
    if (pos + sizeof(uint32_t) > value_size) {
      break;
    }
    uint32_t bin_len = *reinterpret_cast<const uint32_t *>(data + pos);
    pos += sizeof(uint32_t);

    if (pos + bin_len > value_size) {
      break;
    }
    binary_array.emplace_back(data + pos, bin_len);
    pos += bin_len;
  }
  return binary_array;
}

static std::vector<zvec::Doc> convert_zvec_docs_to_internal(
    const ZVecDoc **zvec_docs, size_t doc_count) {
  std::vector<zvec::Doc> docs;
  docs.reserve(doc_count);

  for (size_t i = 0; i < doc_count; ++i) {
    docs.push_back(
        *(*reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(zvec_docs[i])));
  }

  return docs;
}


static zvec::Status convert_zvec_collection_schema_to_internal(
    const ZVecCollectionSchema *schema,
    zvec::CollectionSchema::Ptr &collection_schema) {
  std::string coll_name(schema->name->data, schema->name->length);
  collection_schema = std::make_shared<zvec::CollectionSchema>(coll_name);
  collection_schema->set_max_doc_count_per_segment(
      schema->max_doc_count_per_segment);

  for (size_t i = 0; i < schema->field_count; ++i) {
    const ZVecFieldSchema &zvec_field = *schema->fields[i];
    zvec::DataType data_type = convert_data_type(zvec_field.data_type);
    std::string field_name =
        std::string(zvec_field.name->data, zvec_field.name->length);
    zvec::FieldSchema::Ptr field_schema;

    bool is_vector_field = check_is_vector_field(zvec_field);

    if (is_vector_field) {
      field_schema = std::make_shared<zvec::FieldSchema>(
          field_name, data_type, zvec_field.dimension, zvec_field.nullable);
    } else {
      field_schema = std::make_shared<zvec::FieldSchema>(field_name, data_type,
                                                         zvec_field.nullable);
    }

    if (zvec_field.index_params != nullptr) {
      zvec::Status status = set_field_index_params(field_schema, &zvec_field);
      if (!status.ok()) {
        return status;
      }
    }

    zvec::Status status = collection_schema->add_field(field_schema);
    if (!status.ok()) {
      return status;
    }
  }

  return zvec::Status::OK();
}

static zvec::Status convert_zvec_field_schema_to_internal(
    const ZVecFieldSchema &zvec_field, zvec::FieldSchema::Ptr &field_schema) {
  // Validate input
  if (!zvec_field.name) {
    return zvec::Status::InvalidArgument("Field name cannot be null");
  }

  zvec::DataType data_type = convert_data_type(zvec_field.data_type);
  if (data_type == zvec::DataType::UNDEFINED) {
    return zvec::Status::InvalidArgument("Invalid data type");
  }

  std::string field_name(zvec_field.name->data, zvec_field.name->length);
  bool is_vector_field = check_is_vector_field(zvec_field);

  if (is_vector_field) {
    field_schema = std::make_shared<zvec::FieldSchema>(
        field_name, data_type, zvec_field.dimension, zvec_field.nullable);

    if (zvec_field.index_params != nullptr) {
      switch (zvec_field.index_params->index_type) {
        case ZVEC_INDEX_TYPE_HNSW: {
          auto *params = &zvec_field.index_params->params.hnsw_params;
          auto metric = convert_metric_type(params->base.metric_type);
          auto quantize = convert_quantize_type(params->base.quantize_type);
          auto index_params = std::make_shared<zvec::HnswIndexParams>(
              metric, params->m, params->ef_construction, quantize);
          field_schema->set_index_params(index_params);
          break;
        }
        case ZVEC_INDEX_TYPE_FLAT: {
          auto *params = &zvec_field.index_params->params.flat_params;
          auto metric = convert_metric_type(params->base.metric_type);
          auto quantize = convert_quantize_type(params->base.quantize_type);
          auto index_params =
              std::make_shared<zvec::FlatIndexParams>(metric, quantize);
          field_schema->set_index_params(index_params);
          break;
        }
        case ZVEC_INDEX_TYPE_IVF: {
          auto *params = &zvec_field.index_params->params.ivf_params;
          auto metric = convert_metric_type(params->base.metric_type);
          auto quantize = convert_quantize_type(params->base.quantize_type);
          auto index_params = std::make_shared<zvec::IVFIndexParams>(
              metric, params->n_list, params->n_iters, params->use_soar,
              quantize);
          field_schema->set_index_params(index_params);
          break;
        }
        default:
          field_schema->set_index_params(
              std::make_shared<zvec::FlatIndexParams>(zvec::MetricType::L2));
          break;
      }
    } else {
      field_schema->set_index_params(
          std::make_shared<zvec::FlatIndexParams>(zvec::MetricType::L2));
    }
  } else {
    field_schema = std::make_shared<zvec::FieldSchema>(field_name, data_type,
                                                       zvec_field.nullable);

    if (zvec_field.index_params != nullptr &&
        zvec_field.index_params->index_type == ZVEC_INDEX_TYPE_INVERT) {
      auto *params = &zvec_field.index_params->params.invert_params;
      auto index_params = std::make_shared<zvec::InvertIndexParams>(
          params->enable_range_optimization, params->enable_extended_wildcard);
      field_schema->set_index_params(index_params);
    }
  }

  return zvec::Status::OK();
}

ZVecErrorCode zvec_doc_add_field_by_value(ZVecDoc *doc, const char *field_name,
                                          ZVecDataType data_type,
                                          const void *value,
                                          size_t value_size) {
  if (!doc || !field_name || !value) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to add field",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      std::string name(field_name); ZVecErrorCode error_code = ZVEC_OK;

      switch (data_type) {
        // Scalar types
        case ZVEC_DATA_TYPE_BINARY:
        case ZVEC_DATA_TYPE_STRING: {
          std::string val(static_cast<const char *>(value), value_size);
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_BOOL: {
          bool val = extract_scalar_value<bool>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for bool type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          int32_t val =
              extract_scalar_value<int32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for int32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          int64_t val =
              extract_scalar_value<int64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for int64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          uint32_t val =
              extract_scalar_value<uint32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for uint32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          uint64_t val =
              extract_scalar_value<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for uint64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          float val =
              extract_scalar_value<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for float type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          double val =
              extract_scalar_value<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for double type");
            return error_code;
          }
          (*doc_ptr)->set(name, val);
          break;
        }

        // Vector types
        case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
          auto vec =
              extract_vector_values<uint32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_binary32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          auto vec =
              extract_vector_values<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_binary64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          auto vec =
              extract_vector_values<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          auto vec = extract_vector_values<zvec::float16_t>(value, value_size,
                                                            &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp16 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          auto vec =
              extract_vector_values<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          auto vec =
              extract_vector_values<int8_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_int8 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          auto vec =
              extract_vector_values<int16_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_int16 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT4: {
          // INT4 vectors are packed - each byte contains 2 int4 values
          size_t count = value_size * 2;
          const int8_t *packed_vals = static_cast<const int8_t *>(value);
          std::vector<int8_t> vec;
          vec.reserve(count);

          // Unpack int4 values
          for (size_t i = 0; i < value_size; ++i) {
            int8_t byte_val = packed_vals[i];
            // Extract lower 4 bits
            vec.push_back(byte_val & 0x0F);
            // Extract upper 4 bits
            vec.push_back((byte_val >> 4) & 0x0F);
          }
          (*doc_ptr)->set(name, vec);
          break;
        }

        // Sparse vector types
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16: {
          auto sparse_vec = extract_sparse_vector<zvec::float16_t>(
              value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid sparse vector data size");
            return error_code;
          }
          (*doc_ptr)->set(name, sparse_vec);
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          auto sparse_vec =
              extract_sparse_vector<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid sparse vector data size");
            return error_code;
          }
          (*doc_ptr)->set(name, sparse_vec);
          break;
        }

        // Array types
        case ZVEC_DATA_TYPE_ARRAY_BINARY: {
          auto binary_array = extract_binary_array(value, value_size);
          (*doc_ptr)->set(name, binary_array);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_STRING: {
          // Check if this is a ZVecString** array or a C-string array
          // ZVecString** array has pointer-sized elements
          constexpr size_t ptr_size = sizeof(void *);
          if (value_size % ptr_size == 0) {
            // Likely a ZVecString** array
            size_t count = value_size / ptr_size;
            ZVecString **zvec_str_array =
                reinterpret_cast<ZVecString **>(const_cast<void *>(value));
            auto string_array =
                extract_string_array_from_zvec(zvec_str_array, count);
            (*doc_ptr)->set(name, string_array);
          } else {
            // C-string array (null-terminated strings)
            auto string_array = extract_string_array(value, value_size);
            (*doc_ptr)->set(name, string_array);
          }
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          auto vec = extract_array_values<bool>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_bool type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          auto vec =
              extract_array_values<int32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_int32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          auto vec =
              extract_array_values<int64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_int64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          auto vec =
              extract_array_values<uint32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_uint32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          auto vec =
              extract_array_values<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_uint64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          auto vec =
              extract_array_values<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_float type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          auto vec =
              extract_array_values<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_double type");
            return error_code;
          }
          (*doc_ptr)->set(name, vec);
          break;
        }

        default:
          set_last_error("Unsupported data type: " + std::to_string(data_type));
          return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      return ZVEC_OK;)
}

ZVecErrorCode zvec_doc_add_field_by_struct(ZVecDoc *doc,
                                           const ZVecDocField *field) {
  if (!doc || !field) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to add field",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);

      std::string name(field->name.data, field->name.length);

      switch (field->data_type) {
        // Scalar types (in ZVecDataType order: BINARY, STRING, BOOL, INT32,
        // INT64, UINT32, UINT64, FLOAT, DOUBLE)
        case ZVEC_DATA_TYPE_BINARY: {
          std::string val(
              reinterpret_cast<const char *>(field->value.binary_value.data),
              field->value.binary_value.length);
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_STRING: {
          std::string val(field->value.string_value.data,
                          field->value.string_value.length);
          (*doc_ptr)->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_BOOL: {
          (*doc_ptr)->set(name, field->value.bool_value);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          (*doc_ptr)->set(name, field->value.int32_value);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          (*doc_ptr)->set(name, field->value.int64_value);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          (*doc_ptr)->set(name, field->value.uint32_value);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          (*doc_ptr)->set(name, field->value.uint64_value);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          (*doc_ptr)->set(name, field->value.float_value);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          (*doc_ptr)->set(name, field->value.double_value);
          break;
        }

        // Vector types (in ZVecDataType order: BINARY32, BINARY64, FP16, FP32,
        // FP64, INT4, INT8, INT16)
        case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
          std::vector<uint32_t> vec(reinterpret_cast<const uint32_t *>(
                                        field->value.vector_value.data),
                                    reinterpret_cast<const uint32_t *>(
                                        field->value.vector_value.data) +
                                        field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          std::vector<uint64_t> vec(reinterpret_cast<const uint64_t *>(
                                        field->value.vector_value.data),
                                    reinterpret_cast<const uint64_t *>(
                                        field->value.vector_value.data) +
                                        field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          std::vector<zvec::float16_t> vec(
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          std::vector<float> vec(field->value.vector_value.data,
                                 field->value.vector_value.data +
                                     field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          std::vector<double> vec(
              reinterpret_cast<const double *>(field->value.vector_value.data),
              reinterpret_cast<const double *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT4: {
          size_t byte_count = (field->value.vector_value.length + 1) / 2;
          const int8_t *packed_data =
              reinterpret_cast<const int8_t *>(field->value.vector_value.data);
          std::vector<int8_t> vec;
          vec.reserve(field->value.vector_value.length);

          for (size_t i = 0;
               i < byte_count && vec.size() < field->value.vector_value.length;
               ++i) {
            int8_t byte_val = packed_data[i];
            // Extract lower 4 bits
            vec.push_back(byte_val & 0x0F);
            // Extract upper 4 bits
            if (vec.size() < field->value.vector_value.length) {
              vec.push_back((byte_val >> 4) & 0x0F);
            }
          }
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          std::vector<int8_t> vec(
              reinterpret_cast<const int8_t *>(field->value.vector_value.data),
              reinterpret_cast<const int8_t *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          std::vector<int16_t> vec(
              reinterpret_cast<const int16_t *>(field->value.vector_value.data),
              reinterpret_cast<const int16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }

        // Sparse vector types (in ZVecDataType order: FP16, FP32)
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16: {
          std::vector<zvec::float16_t> vec(
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          std::vector<float> vec(field->value.vector_value.data,
                                 field->value.vector_value.data +
                                     field->value.vector_value.length);
          (*doc_ptr)->set(name, vec);
          break;
        }

        // Array types (in ZVecDataType order: BINARY, STRING, BOOL, INT32,
        // INT64, UINT32, UINT64, FLOAT, DOUBLE)
        case ZVEC_DATA_TYPE_ARRAY_BINARY: {
          std::vector<std::string> array_values;
          const uint8_t *data_ptr = field->value.binary_value.data;
          size_t total_length = field->value.binary_value.length;
          size_t offset = 0;

          while (offset + sizeof(uint32_t) <= total_length) {
            uint32_t elem_length =
                *reinterpret_cast<const uint32_t *>(data_ptr + offset);
            offset += sizeof(uint32_t);

            if (offset + elem_length <= total_length) {
              std::string elem(
                  reinterpret_cast<const char *>(data_ptr + offset),
                  elem_length);
              array_values.push_back(elem);
              offset += elem_length;
            } else {
              break;
            }
          }
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_STRING: {
          std::vector<std::string> array_values;
          const char *data_ptr = field->value.string_value.data;
          size_t total_length = field->value.string_value.length;
          size_t offset = 0;

          while (offset < total_length) {
            size_t str_len = strlen(data_ptr + offset);
            if (str_len > 0 && offset + str_len <= total_length) {
              array_values.emplace_back(data_ptr + offset, str_len);
              offset += str_len + 1;
            } else {
              break;
            }
          }
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          std::vector<bool> array_values(
              reinterpret_cast<const bool *>(field->value.binary_value.data),
              reinterpret_cast<const bool *>(field->value.binary_value.data) +
                  field->value.binary_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          std::vector<int32_t> array_values(
              reinterpret_cast<const int32_t *>(field->value.vector_value.data),
              reinterpret_cast<const int32_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          std::vector<int64_t> array_values(
              reinterpret_cast<const int64_t *>(field->value.vector_value.data),
              reinterpret_cast<const int64_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          std::vector<uint32_t> array_values(
              reinterpret_cast<const uint32_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const uint32_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          std::vector<uint64_t> array_values(
              reinterpret_cast<const uint64_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const uint64_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          std::vector<float> array_values(field->value.vector_value.data,
                                          field->value.vector_value.data +
                                              field->value.vector_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          std::vector<double> array_values(
              reinterpret_cast<const double *>(field->value.vector_value.data),
              reinterpret_cast<const double *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, array_values);
          break;
        }

        default:
          set_last_error("Unsupported data type: " +
                         std::to_string(field->data_type));
          return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      return ZVEC_OK;)
}

const char *zvec_doc_get_pk_pointer(const ZVecDoc *doc) {
  if (!doc) return nullptr;
  auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
  return (*doc_ptr)->pk_ref().data();
}

const char *zvec_doc_get_pk_copy(const ZVecDoc *doc) {
  if (!doc) return nullptr;
  auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
  const std::string &pk = (*doc_ptr)->pk_ref();
  if (pk.empty()) return nullptr;

  char *result = static_cast<char *>(malloc(pk.length() + 1));
  strcpy(result, pk.c_str());
  return result;
}

uint64_t zvec_doc_get_doc_id(const ZVecDoc *doc) {
  if (!doc) return 0;

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document ID", 0,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->doc_id();)
}

float zvec_doc_get_score(const ZVecDoc *doc) {
  if (!doc) return 0.0f;

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document score", 0.0f,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->score();)
}

ZVecDocOperator zvec_doc_get_operator(const ZVecDoc *doc) {
  if (!doc) return ZVEC_DOC_OP_INSERT;  // default
  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document operator", ZVEC_DOC_OP_INSERT,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      zvec::Operator op = (*doc_ptr)->get_operator();
      return static_cast<ZVecDocOperator>(op);)
}

size_t zvec_doc_get_field_count(const ZVecDoc *doc) {
  if (!doc) return 0;

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get field count", 0,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->field_names().size();)
}

ZVecErrorCode zvec_doc_get_field_value_basic(const ZVecDoc *doc,
                                             const char *field_name,
                                             ZVecDataType field_type,
                                             void *value_buffer,
                                             size_t buffer_size) {
  if (!doc || !field_name || !value_buffer) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field value",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);

      // Check if field exists
      if (!(*doc_ptr)->has(field_name)) {
        set_last_error("Field not found in document");
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      // Handle basic data types that return values directly
      switch (field_type) {
        case ZVEC_DATA_TYPE_BOOL: {
          if (buffer_size < sizeof(bool)) {
            set_last_error("Buffer too small for bool value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const bool val = (*doc_ptr)->get_ref<bool>(field_name);
          *static_cast<bool *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          if (buffer_size < sizeof(int32_t)) {
            set_last_error("Buffer too small for int32 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const int32_t val = (*doc_ptr)->get_ref<int32_t>(field_name);
          *static_cast<int32_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          if (buffer_size < sizeof(int64_t)) {
            set_last_error("Buffer too small for int64 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const int64_t val = (*doc_ptr)->get_ref<int64_t>(field_name);
          *static_cast<int64_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          if (buffer_size < sizeof(uint32_t)) {
            set_last_error("Buffer too small for uint32 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const uint32_t val = (*doc_ptr)->get_ref<uint32_t>(field_name);
          *static_cast<uint32_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          if (buffer_size < sizeof(uint64_t)) {
            set_last_error("Buffer too small for uint64 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const uint64_t val = (*doc_ptr)->get_ref<uint64_t>(field_name);
          *static_cast<uint64_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          if (buffer_size < sizeof(float)) {
            set_last_error("Buffer too small for float value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const float val = (*doc_ptr)->get_ref<float>(field_name);
          *static_cast<float *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          if (buffer_size < sizeof(double)) {
            set_last_error("Buffer too small for double value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const double val = (*doc_ptr)->get_ref<double>(field_name);
          *static_cast<double *>(value_buffer) = val;
          break;
        }
        default: {
          set_last_error("Data type not supported for basic value return");
          return ZVEC_ERROR_INVALID_ARGUMENT;
        }
      }

      return ZVEC_OK;)
}

ZVecErrorCode zvec_doc_get_field_value_copy(const ZVecDoc *doc,
                                            const char *field_name,
                                            ZVecDataType field_type,
                                            void **value, size_t *value_size) {
  if (!doc || !field_name || !value || !value_size) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field value copy",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);

      // Check if field exists
      if (!(*doc_ptr)->has(field_name)) {
        set_last_error("Field not found in document");
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      // Handle copy-returning data types (allocate new memory)
      switch (field_type) {
        // Basic types - copy the actual values
        case ZVEC_DATA_TYPE_BOOL: {
          const bool val = (*doc_ptr)->get_ref<bool>(field_name);
          void *buffer = malloc(sizeof(bool));
          if (!buffer) {
            set_last_error("Memory allocation failed for bool");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<bool *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(bool);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          const int32_t val = (*doc_ptr)->get_ref<int32_t>(field_name);
          void *buffer = malloc(sizeof(int32_t));
          if (!buffer) {
            set_last_error("Memory allocation failed for int32");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<int32_t *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(int32_t);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          const int64_t val = (*doc_ptr)->get_ref<int64_t>(field_name);
          void *buffer = malloc(sizeof(int64_t));
          if (!buffer) {
            set_last_error("Memory allocation failed for int64");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<int64_t *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(int64_t);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          const uint32_t val = (*doc_ptr)->get_ref<uint32_t>(field_name);
          void *buffer = malloc(sizeof(uint32_t));
          if (!buffer) {
            set_last_error("Memory allocation failed for uint32");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<uint32_t *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          const uint64_t val = (*doc_ptr)->get_ref<uint64_t>(field_name);
          void *buffer = malloc(sizeof(uint64_t));
          if (!buffer) {
            set_last_error("Memory allocation failed for uint64");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<uint64_t *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          const float val = (*doc_ptr)->get_ref<float>(field_name);
          void *buffer = malloc(sizeof(float));
          if (!buffer) {
            set_last_error("Memory allocation failed for float");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<float *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          const double val = (*doc_ptr)->get_ref<double>(field_name);
          void *buffer = malloc(sizeof(double));
          if (!buffer) {
            set_last_error("Memory allocation failed for double");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          *static_cast<double *>(buffer) = val;
          *value = buffer;
          *value_size = sizeof(double);
          break;
        }

        // String and binary types - copy the data
        case ZVEC_DATA_TYPE_BINARY:
        case ZVEC_DATA_TYPE_STRING: {
          const std::string &val = (*doc_ptr)->get_ref<std::string>(field_name);
          void *buffer = malloc(val.length());
          if (!buffer) {
            set_last_error("Memory allocation failed for string/binary");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), val.length());
          *value = buffer;
          *value_size = val.length();
          break;
        }

        // Vector types - copy the data
        case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
          const std::vector<uint32_t> &val =
              (*doc_ptr)->get_ref<std::vector<uint32_t>>(field_name);
          size_t total_size = val.size() * sizeof(uint32_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for uint32 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          const std::vector<uint64_t> &val =
              (*doc_ptr)->get_ref<std::vector<uint64_t>>(field_name);
          size_t total_size = val.size() * sizeof(uint64_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for uint64 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          const std::vector<zvec::float16_t> &val =
              (*doc_ptr)->get_ref<std::vector<zvec::float16_t>>(field_name);
          size_t total_size = val.size() * sizeof(zvec::float16_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for fp16 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          const std::vector<float> &val =
              (*doc_ptr)->get_ref<std::vector<float>>(field_name);
          size_t total_size = val.size() * sizeof(float);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for fp32 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          const std::vector<double> &val =
              (*doc_ptr)->get_ref<std::vector<double>>(field_name);
          size_t total_size = val.size() * sizeof(double);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for fp64 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT4:
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          const std::vector<int8_t> &val =
              (*doc_ptr)->get_ref<std::vector<int8_t>>(field_name);
          size_t total_size = val.size() * sizeof(int8_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for int8 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          const std::vector<int16_t> &val =
              (*doc_ptr)->get_ref<std::vector<int16_t>>(field_name);
          size_t total_size = val.size() * sizeof(int16_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for int16 vector");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }
          memcpy(buffer, val.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }

        // Sparse vector types - create flattened representation
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16: {
          using SparseVecFP16 =
              std::pair<std::vector<uint32_t>, std::vector<zvec::float16_t>>;
          const SparseVecFP16 &sparse_vec =
              (*doc_ptr)->get_ref<SparseVecFP16>(field_name);
          size_t nnz = sparse_vec.first.size();
          size_t total_size = sizeof(size_t) + nnz * (sizeof(uint32_t) +
                                                      sizeof(zvec::float16_t));
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for sparse vector FP16");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          char *ptr = static_cast<char *>(buffer);
          *reinterpret_cast<size_t *>(ptr) = nnz;
          ptr += sizeof(size_t);

          for (size_t i = 0; i < nnz; ++i) {
            *reinterpret_cast<uint32_t *>(ptr) = sparse_vec.first[i];
            ptr += sizeof(uint32_t);
          }
          for (size_t i = 0; i < nnz; ++i) {
            *reinterpret_cast<zvec::float16_t *>(ptr) = sparse_vec.second[i];
            ptr += sizeof(zvec::float16_t);
          }

          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          using SparseVecFP32 =
              std::pair<std::vector<uint32_t>, std::vector<float>>;
          const SparseVecFP32 &sparse_vec =
              (*doc_ptr)->get_ref<SparseVecFP32>(field_name);
          size_t nnz = sparse_vec.first.size();
          size_t total_size =
              sizeof(size_t) + nnz * (sizeof(uint32_t) + sizeof(float));
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for sparse vector FP32");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          char *ptr = static_cast<char *>(buffer);
          *reinterpret_cast<size_t *>(ptr) = nnz;
          ptr += sizeof(size_t);

          for (size_t i = 0; i < nnz; ++i) {
            *reinterpret_cast<uint32_t *>(ptr) = sparse_vec.first[i];
            ptr += sizeof(uint32_t);
          }
          for (size_t i = 0; i < nnz; ++i) {
            *reinterpret_cast<float *>(ptr) = sparse_vec.second[i];
            ptr += sizeof(float);
          }

          *value = buffer;
          *value_size = total_size;
          break;
        }

        // Array types - create serialized representations
        case ZVEC_DATA_TYPE_ARRAY_BINARY: {
          using BinaryArray = std::vector<std::string>;
          const BinaryArray &array_vals =
              (*doc_ptr)->get_ref<BinaryArray>(field_name);
          size_t total_size = 0;
          for (const auto &bin_val : array_vals) {
            total_size += bin_val.length();
          }

          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for binary array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          char *ptr = static_cast<char *>(buffer);
          for (const auto &bin_val : array_vals) {
            memcpy(ptr, bin_val.data(), bin_val.length());
            ptr += bin_val.length();
          }

          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_STRING: {
          using StringArray = std::vector<std::string>;
          const StringArray &array_vals =
              (*doc_ptr)->get_ref<StringArray>(field_name);
          size_t total_size = 0;
          for (const auto &str_val : array_vals) {
            total_size += str_val.length() + 1;  // +1 for null terminator
          }

          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for string array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          char *ptr = static_cast<char *>(buffer);
          for (const auto &str_val : array_vals) {
            memcpy(ptr, str_val.c_str(), str_val.length());
            ptr += str_val.length();
            *ptr = '\0';
            ptr++;
          }

          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          using BoolArray = std::vector<bool>;
          const BoolArray &array_vals =
              (*doc_ptr)->get_ref<BoolArray>(field_name);
          size_t byte_count = (array_vals.size() + 7) / 8;
          void *buffer = malloc(byte_count);
          if (!buffer) {
            set_last_error("Memory allocation failed for bool array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          uint8_t *bytes = static_cast<uint8_t *>(buffer);
          memset(bytes, 0, byte_count);

          for (size_t i = 0; i < array_vals.size(); ++i) {
            if (array_vals[i]) {
              bytes[i / 8] |= (1 << (i % 8));
            }
          }

          *value = buffer;
          *value_size = byte_count;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          using Int32Array = std::vector<int32_t>;
          const Int32Array &array_vals =
              (*doc_ptr)->get_ref<Int32Array>(field_name);
          size_t total_size = array_vals.size() * sizeof(int32_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for int32 array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          memcpy(buffer, array_vals.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          using Int64Array = std::vector<int64_t>;
          const Int64Array &array_vals =
              (*doc_ptr)->get_ref<Int64Array>(field_name);
          size_t total_size = array_vals.size() * sizeof(int64_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for int64 array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          memcpy(buffer, array_vals.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          using UInt32Array = std::vector<uint32_t>;
          const UInt32Array &array_vals =
              (*doc_ptr)->get_ref<UInt32Array>(field_name);
          size_t total_size = array_vals.size() * sizeof(uint32_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for uint32 array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          memcpy(buffer, array_vals.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          using UInt64Array = std::vector<uint64_t>;
          const UInt64Array &array_vals =
              (*doc_ptr)->get_ref<UInt64Array>(field_name);
          size_t total_size = array_vals.size() * sizeof(uint64_t);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for uint64 array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          memcpy(buffer, array_vals.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          using FloatArray = std::vector<float>;
          const FloatArray &array_vals =
              (*doc_ptr)->get_ref<FloatArray>(field_name);
          size_t total_size = array_vals.size() * sizeof(float);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for float array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          memcpy(buffer, array_vals.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          using DoubleArray = std::vector<double>;
          const DoubleArray &array_vals =
              (*doc_ptr)->get_ref<DoubleArray>(field_name);
          size_t total_size = array_vals.size() * sizeof(double);
          void *buffer = malloc(total_size);
          if (!buffer) {
            set_last_error("Memory allocation failed for double array");
            return ZVEC_ERROR_INTERNAL_ERROR;
          }

          memcpy(buffer, array_vals.data(), total_size);
          *value = buffer;
          *value_size = total_size;
          break;
        }
        default: {
          set_last_error("Unknown data type");
          return ZVEC_ERROR_INVALID_ARGUMENT;
        }
      }

      return ZVEC_OK;)
}

ZVecErrorCode zvec_doc_get_field_value_pointer(const ZVecDoc *doc,
                                               const char *field_name,
                                               ZVecDataType field_type,
                                               const void **value,
                                               size_t *value_size) {
  if (!doc || !field_name || !value || !value_size) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field value pointer",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);

      // Check if field exists
      if (!(*doc_ptr)->has(field_name)) {
        set_last_error("Field not found in document");
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      // Get field value based on data type
      switch (field_type) {
        case ZVEC_DATA_TYPE_BINARY: {
          const std::string &val = (*doc_ptr)->get_ref<std::string>(field_name);
          *value = val.data();
          *value_size = val.length();
          break;
        }
        case ZVEC_DATA_TYPE_STRING: {
          const std::string &val = (*doc_ptr)->get_ref<std::string>(field_name);
          *value = val.c_str();
          *value_size = val.length();
          break;
        }
        case ZVEC_DATA_TYPE_BOOL: {
          const bool &val = (*doc_ptr)->get_ref<bool>(field_name);
          *value = &val;
          *value_size = sizeof(bool);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          const int32_t &val = (*doc_ptr)->get_ref<int32_t>(field_name);
          *value = &val;
          *value_size = sizeof(int32_t);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          const int64_t &val = (*doc_ptr)->get_ref<int64_t>(field_name);
          *value = &val;
          *value_size = sizeof(int64_t);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          const uint32_t &val = (*doc_ptr)->get_ref<uint32_t>(field_name);
          *value = &val;
          *value_size = sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          const uint64_t &val = (*doc_ptr)->get_ref<uint64_t>(field_name);
          *value = &val;
          *value_size = sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          const float &val = (*doc_ptr)->get_ref<float>(field_name);
          *value = &val;
          *value_size = sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          const double &val = (*doc_ptr)->get_ref<double>(field_name);
          *value = &val;
          *value_size = sizeof(double);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
          const std::vector<uint32_t> &val =
              (*doc_ptr)->get_ref<std::vector<uint32_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          const std::vector<uint64_t> &val =
              (*doc_ptr)->get_ref<std::vector<uint64_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          // FP16 vectors typically stored as uint16_t
          const std::vector<zvec::float16_t> &val =
              (*doc_ptr)->get_ref<std::vector<zvec::float16_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(zvec::float16_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          const std::vector<float> &val =
              (*doc_ptr)->get_ref<std::vector<float>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          const std::vector<double> &val =
              (*doc_ptr)->get_ref<std::vector<double>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(double);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT4: {
          // INT4 vectors typically stored as int8_t with 2 values per byte
          const std::vector<int8_t> &val =
              (*doc_ptr)->get_ref<std::vector<int8_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(int8_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          const std::vector<int8_t> &val =
              (*doc_ptr)->get_ref<std::vector<int8_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(int8_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          const std::vector<int16_t> &val =
              (*doc_ptr)->get_ref<std::vector<int16_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(int16_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          auto &array_vals =
              (*doc_ptr)->get_ref<std::vector<int32_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(int32_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          auto &array_vals =
              (*doc_ptr)->get_ref<std::vector<int64_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(int64_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          auto &array_vals =
              (*doc_ptr)->get_ref<std::vector<uint32_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          auto &array_vals =
              (*doc_ptr)->get_ref<std::vector<uint64_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          auto &array_vals =
              (*doc_ptr)->get_ref<std::vector<float>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          auto &array_vals =
              (*doc_ptr)->get_ref<std::vector<double>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(double);
          break;
        }
        default: {
          set_last_error("Unknown data type");
          return ZVEC_ERROR_INVALID_ARGUMENT;
        }
      }

      return ZVEC_OK;)
}

bool zvec_doc_is_empty(const ZVecDoc *doc) {
  if (!doc) {
    set_last_error("Document pointer is null");
    return true;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check if document is empty", true,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->is_empty();)
}

ZVecErrorCode zvec_doc_remove_field(ZVecDoc *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to remove field",
      auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
      (*doc_ptr)->remove(std::string(field_name)); return ZVEC_OK;)
}


bool zvec_doc_has_field(const ZVecDoc *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return false;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check field existence", false,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->has(std::string(field_name));)
}

bool zvec_doc_has_field_value(const ZVecDoc *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return false;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check field value existence", false,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->has_value(std::string(field_name));)
}

bool zvec_doc_is_field_null(const ZVecDoc *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return false;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check if field is null", false,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->is_null(std::string(field_name));)
}

ZVecErrorCode zvec_doc_get_field_names(const ZVecDoc *doc, char ***field_names,
                                       size_t *count) {
  if (!doc || !field_names || !count) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field names",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      std::vector<std::string> names = (*doc_ptr)->field_names();

      *count = names.size();
      if (*count == 0) {
        *field_names = nullptr;
        return ZVEC_OK;
      }

          *field_names = static_cast<char **>(malloc(*count * sizeof(char *)));
      if (!*field_names) {
        set_last_error("Failed to allocate memory for field names");
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

      for (size_t i = 0; i < *count; ++i) {
        (*field_names)[i] = copy_string(names[i]);
        if (!(*field_names)[i]) {
          for (size_t j = 0; j < i; ++j) {
            free((*field_names)[j]);
          }
          free(*field_names);
          *field_names = nullptr;
          set_last_error("Failed to copy field name");
          return ZVEC_ERROR_INTERNAL_ERROR;
        }
      }

      return ZVEC_OK;)
}

ZVecErrorCode zvec_doc_serialize(const ZVecDoc *doc, uint8_t **data,
                                 size_t *size) {
  if (!doc || !data || !size) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to serialize document",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      std::vector<uint8_t> serialized_data = (*doc_ptr)->serialize();

      *size = serialized_data.size();
      if (*size == 0) {
        *data = nullptr;
        return ZVEC_OK;
      }

          *data = static_cast<uint8_t *>(malloc(*size));
      if (!*data) {
        set_last_error("Failed to allocate memory for serialized data");
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

      memcpy(*data, serialized_data.data(), *size);
      return ZVEC_OK;)
}

ZVecErrorCode zvec_doc_deserialize(const uint8_t *data, size_t size,
                                   ZVecDoc **doc) {
  if (!data || !doc || size == 0) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to deserialize document",
      auto deserialized_doc = zvec::Doc::deserialize(data, size);
      if (!deserialized_doc) {
        set_last_error("Failed to deserialize document");
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

      auto doc_ptr = new std::shared_ptr<zvec::Doc>(deserialized_doc);
      *doc = reinterpret_cast<ZVecDoc *>(doc_ptr); return ZVEC_OK;)
}

void zvec_doc_merge(ZVecDoc *doc, const ZVecDoc *other) {
  if (!doc || !other) {
    set_last_error("Document pointers are null");
    return;
  }

  ZVEC_TRY_BEGIN_VOID
  auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
  auto other_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(other);
  (*doc_ptr)->merge(**other_ptr);
  ZVEC_CATCH_END_VOID
}

size_t zvec_doc_memory_usage(const ZVecDoc *doc) {
  if (!doc) {
    set_last_error("Document pointer is null");
    return 0;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document memory usage", 0,
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      return (*doc_ptr)->memory_usage();)
}

ZVecErrorCode zvec_doc_validate(const ZVecDoc *doc,
                                const ZVecCollectionSchema *schema,
                                bool is_update, char **error_msg) {
  if (!doc || !schema) {
    set_last_error("Document or schema pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to validate document",
      std::shared_ptr<zvec::CollectionSchema> schema_ptr = nullptr;
      auto status =
          convert_zvec_collection_schema_to_internal(schema, schema_ptr);
      if (!status.ok()) {
        if (error_msg) {
          *error_msg = copy_string(status.message());
        }
        return status_to_error_code(status);
      }

      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      status = (*doc_ptr)->validate(schema_ptr, is_update); if (!status.ok()) {
        if (error_msg) {
          *error_msg = copy_string(status.message());
        }
        return status_to_error_code(status);
      }

      if (error_msg) { *error_msg = nullptr; } return ZVEC_OK;)
}

ZVecErrorCode zvec_doc_to_detail_string(const ZVecDoc *doc, char **detail_str) {
  if (!doc || !detail_str) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get document detail string",
      auto doc_ptr = reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(doc);
      std::string detail = (*doc_ptr)->to_detail_string();
      *detail_str = copy_string(detail);

      if (!*detail_str && !detail.empty()) {
        set_last_error("Failed to copy detail string");
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

      return ZVEC_OK;)
}

// =============================================================================
// Collection functions implementation
// =============================================================================

ZVecErrorCode zvec_collection_create_and_open(
    const char *path, const ZVecCollectionSchema *schema,
    const ZVecCollectionOptions *options, ZVecCollection **collection) {
  ZVEC_TRY_RETURN_ERROR(
      "Exception in zvec_collection_create_and_open_with_schema",
      if (!path || !schema || !collection) {
        set_last_error("Path, schema, or collection cannot be null");
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      std::shared_ptr<zvec::CollectionSchema>
          schema_ptr = nullptr;
      auto status =
          convert_zvec_collection_schema_to_internal(schema, schema_ptr);
      if (!status.ok()) {
        set_last_error(status.message());
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      zvec::CollectionOptions collection_options;
      if (options) {
        collection_options.enable_mmap_ = options->enable_mmap;
        collection_options.max_buffer_size_ = options->max_buffer_size;
        collection_options.read_only_ = options->read_only;
      }

      auto result = zvec::Collection::CreateAndOpen(path, *schema_ptr,
                                                    collection_options);
      ZVecErrorCode error_code = handle_expected_result(result);

      if (error_code == ZVEC_OK) {
        *collection = reinterpret_cast<ZVecCollection *>(
            new std::shared_ptr<zvec::Collection>(std::move(result.value())));
      }

      return error_code;)
}

ZVecErrorCode zvec_collection_open(const char *path,
                                   const ZVecCollectionOptions *options,
                                   ZVecCollection **collection) {
  if (!path || !collection) {
    set_last_error("Invalid arguments: path and collection cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred", zvec::CollectionOptions collection_options;
      if (options) {
        collection_options.enable_mmap_ = options->enable_mmap;
        collection_options.max_buffer_size_ = options->max_buffer_size;
        collection_options.read_only_ = options->read_only;
      }

      auto result = zvec::Collection::Open(path, collection_options);
      ZVecErrorCode error_code = handle_expected_result(result);

      if (error_code == ZVEC_OK) {
        *collection = reinterpret_cast<ZVecCollection *>(
            new std::shared_ptr<zvec::Collection>(std::move(result.value())));
      }

      return error_code;)
}

ZVecErrorCode zvec_collection_close(ZVecCollection *collection) {
  if (!collection) {
    set_last_error("Invalid argument: collection cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred",
      delete reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
      return ZVEC_OK;)
}

ZVecErrorCode zvec_collection_destroy(ZVecCollection *collection) {
  if (!collection) {
    set_last_error("Invalid argument: collection cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred",
      auto &coll =
          *reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
      zvec::Status status = coll->Destroy();
      if (!status.ok()) { set_last_error(status.message()); }

      return status_to_error_code(status);)
}

ZVecErrorCode zvec_collection_flush(ZVecCollection *collection) {
  if (!collection) {
    set_last_error("Invalid argument: collection cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred",
      auto &coll =
          *reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
      zvec::Status status = coll->Flush();

      if (!status.ok()) { set_last_error(status.message()); }

      return status_to_error_code(status);)
}

ZVecErrorCode zvec_collection_get_schema(const ZVecCollection *collection,
                                         ZVecCollectionSchema **schema) {
  if (!collection || !schema) {
    set_last_error("Invalid arguments: collection and schema cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred",
      auto &coll = *reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
          collection);
      auto result = coll->Schema();

      ZVecErrorCode error_code = handle_expected_result(result);
      if (error_code == ZVEC_OK) {
        const auto &cpp_schema = result.value();

        // Create new schema structure
        ZVecCollectionSchema *c_schema = static_cast<ZVecCollectionSchema *>(
            malloc(sizeof(ZVecCollectionSchema)));
        if (!c_schema) {
          set_last_error("Failed to allocate memory for schema");
          return ZVEC_ERROR_RESOURCE_EXHAUSTED;
        }

        // Initialize the schema structure
        c_schema->name = nullptr;
        c_schema->fields = nullptr;
        c_schema->field_count = 0;
        c_schema->field_capacity = 0;
        c_schema->max_doc_count_per_segment =
            cpp_schema.max_doc_count_per_segment();

        // Set collection name
        c_schema->name = zvec_string_create(cpp_schema.name().c_str());
        if (!c_schema->name) {
          free(c_schema);
          set_last_error("Failed to allocate memory for collection name");
          return ZVEC_ERROR_RESOURCE_EXHAUSTED;
        }

        // Convert and copy fields
        const auto &cpp_fields = cpp_schema.fields();
        c_schema->field_count = cpp_fields.size();
        c_schema->field_capacity = cpp_fields.size();

        if (c_schema->field_count > 0) {
          // Allocate array of field pointers
          c_schema->fields = static_cast<ZVecFieldSchema **>(
              malloc(c_schema->field_count * sizeof(ZVecFieldSchema *)));
          if (!c_schema->fields) {
            zvec_collection_schema_destroy(c_schema);
            set_last_error("Failed to allocate memory for fields");
            return ZVEC_ERROR_RESOURCE_EXHAUSTED;
          }

          // Initialize all field pointers to nullptr
          for (size_t i = 0; i < c_schema->field_count; ++i) {
            c_schema->fields[i] = nullptr;
          }

          size_t i = 0;
          for (const auto &cpp_field : cpp_fields) {
            try {
              // Create new field schema
              c_schema->fields[i] = static_cast<ZVecFieldSchema *>(
                  malloc(sizeof(ZVecFieldSchema)));
              if (!c_schema->fields[i]) {
                throw std::bad_alloc();
              }

              // Copy field name using zvec_string_create
              c_schema->fields[i]->name =
                  zvec_string_create(cpp_field->name().c_str());
              if (!c_schema->fields[i]->name) {
                throw std::bad_alloc();
              }

              // Convert data type
              c_schema->fields[i]->data_type =
                  convert_zvec_data_type(cpp_field->data_type());

              // Copy dimension for vector fields
              c_schema->fields[i]->dimension = cpp_field->dimension();

              // Copy nullable flag
              c_schema->fields[i]->nullable = cpp_field->nullable();

              // Initialize index parameters
              c_schema->fields[i]->index_params = nullptr;

              // Convert index parameters based on the actual type
              auto index_params = cpp_field->index_params();
              if (index_params) {
                switch (index_params->type()) {
                  case zvec::IndexType::HNSW: {
                    // Cast to HnswIndexParams and convert
                    auto hnsw_params =
                        std::dynamic_pointer_cast<zvec::HnswIndexParams>(
                            index_params);
                    if (hnsw_params) {
                      auto c_hnsw_params = static_cast<ZVecHnswIndexParams *>(
                          malloc(sizeof(ZVecHnswIndexParams)));
                      if (!c_hnsw_params) {
                        throw std::bad_alloc();
                      }

                      // Initialize the base vector index parameters
                      c_hnsw_params->base.base.index_type =
                          ZVEC_INDEX_TYPE_HNSW;
                      c_hnsw_params->base.metric_type =
                          static_cast<ZVecMetricType>(
                              hnsw_params->metric_type());
                      c_hnsw_params->base.quantize_type =
                          static_cast<ZVecQuantizeType>(
                              hnsw_params->quantize_type());

                      // Set HNSW-specific parameters
                      c_hnsw_params->m = hnsw_params->m();
                      c_hnsw_params->ef_construction =
                          hnsw_params->ef_construction();

                      // Assign to field schema (using pointer assignment)
                      c_schema->fields[i]->index_params =
                          reinterpret_cast<ZVecIndexParams *>(c_hnsw_params);
                      c_schema->fields[i]->index_params->index_type =
                          ZVEC_INDEX_TYPE_HNSW;
                    }
                    break;
                  }

                  case zvec::IndexType::IVF: {
                    // Cast to IVFIndexParams and convert
                    auto ivf_params =
                        std::dynamic_pointer_cast<zvec::IVFIndexParams>(
                            index_params);
                    if (ivf_params) {
                      auto c_ivf_params = static_cast<ZVecIVFIndexParams *>(
                          malloc(sizeof(ZVecIVFIndexParams)));
                      if (!c_ivf_params) {
                        throw std::bad_alloc();
                      }

                      // Initialize the base vector index parameters
                      c_ivf_params->base.base.index_type = ZVEC_INDEX_TYPE_IVF;
                      c_ivf_params->base.metric_type =
                          static_cast<ZVecMetricType>(
                              ivf_params->metric_type());
                      c_ivf_params->base.quantize_type =
                          static_cast<ZVecQuantizeType>(
                              ivf_params->quantize_type());

                      // Set IVF-specific parameters
                      c_ivf_params->n_list = ivf_params->n_list();
                      c_ivf_params->n_iters = ivf_params->n_iters();
                      c_ivf_params->use_soar = ivf_params->use_soar();

                      // Assign to field schema (using pointer assignment)
                      c_schema->fields[i]->index_params =
                          reinterpret_cast<ZVecIndexParams *>(c_ivf_params);
                      c_schema->fields[i]->index_params->index_type =
                          ZVEC_INDEX_TYPE_IVF;
                    }
                    break;
                  }

                  case zvec::IndexType::FLAT: {
                    // Cast to FlatIndexParams and convert
                    auto flat_params =
                        std::dynamic_pointer_cast<zvec::FlatIndexParams>(
                            index_params);
                    if (flat_params) {
                      auto c_flat_params = static_cast<ZVecFlatIndexParams *>(
                          malloc(sizeof(ZVecFlatIndexParams)));
                      if (!c_flat_params) {
                        throw std::bad_alloc();
                      }

                      // Initialize the base vector index parameters
                      c_flat_params->base.base.index_type =
                          ZVEC_INDEX_TYPE_FLAT;
                      c_flat_params->base.metric_type =
                          static_cast<ZVecMetricType>(
                              flat_params->metric_type());
                      c_flat_params->base.quantize_type =
                          static_cast<ZVecQuantizeType>(
                              flat_params->quantize_type());

                      // Flat index has no additional parameters

                      // Assign to field schema (using pointer assignment)
                      c_schema->fields[i]->index_params =
                          reinterpret_cast<ZVecIndexParams *>(c_flat_params);
                      c_schema->fields[i]->index_params->index_type =
                          ZVEC_INDEX_TYPE_FLAT;
                    }
                    break;
                  }

                  case zvec::IndexType::INVERT: {
                    // Cast to InvertIndexParams and convert
                    auto invert_params =
                        std::dynamic_pointer_cast<zvec::InvertIndexParams>(
                            index_params);
                    if (invert_params) {
                      auto c_invert_params =
                          static_cast<ZVecInvertIndexParams *>(
                              malloc(sizeof(ZVecInvertIndexParams)));
                      if (!c_invert_params) {
                        throw std::bad_alloc();
                      }

                      // Initialize the base index parameters
                      c_invert_params->base.index_type = ZVEC_INDEX_TYPE_INVERT;

                      // Set Invert-specific parameters
                      c_invert_params->enable_range_optimization =
                          invert_params->enable_range_optimization();
                      c_invert_params->enable_extended_wildcard =
                          invert_params->enable_extended_wildcard();

                      // Assign to field schema (using pointer assignment)
                      c_schema->fields[i]->index_params =
                          reinterpret_cast<ZVecIndexParams *>(c_invert_params);
                      c_schema->fields[i]->index_params->index_type =
                          ZVEC_INDEX_TYPE_INVERT;
                    }
                    break;
                  }

                  default:
                    // For undefined or unsupported index types, set to NULL
                    c_schema->fields[i]->index_params = nullptr;
                    c_schema->fields[i]->index_params->index_type =
                        ZVEC_INDEX_TYPE_UNDEFINED;
                    break;
                }
              } else {
                // No index parameters, set to NULL
                c_schema->fields[i]->index_params = nullptr;
              }
            } catch (const std::bad_alloc &) {
              // Clean up already allocated fields
              for (size_t j = 0; j <= i; ++j) {
                if (c_schema->fields[j]) {
                  zvec_field_schema_destroy(c_schema->fields[j]);
                }
              }
              free(c_schema->fields);
              zvec_free_string(c_schema->name);
              free(c_schema);
              set_last_error("Failed to allocate memory for field");
              return ZVEC_ERROR_RESOURCE_EXHAUSTED;
            }

            ++i;
          }
        }

        *schema = c_schema;
      }

      return error_code;)
}

ZVecErrorCode zvec_collection_get_options(const ZVecCollection *collection,
                                          ZVecCollectionOptions **options) {
  if (!collection || !options) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get collection options",
      auto collection_ptr =
          reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
              collection);
      auto result = (*collection_ptr)->Options();

      if (!result.has_value()) {
        set_last_error("Failed to get collection option: " +
                       result.error().message());
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

          // Create and initialize options structure
          *options = static_cast<ZVecCollectionOptions *>(
              malloc(sizeof(ZVecCollectionOptions)));
      if (!*options) {
        set_last_error("Failed to allocate memory for options");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      (*options)
          ->enable_mmap = result.value().enable_mmap_;
      (*options)->max_buffer_size = result.value().max_buffer_size_;
      (*options)->read_only = result.value().read_only_;
      (*options)->max_doc_count_per_segment = zvec::MAX_DOC_COUNT_PER_SEGMENT;

      return ZVEC_OK;)
}

ZVecErrorCode zvec_collection_get_stats(const ZVecCollection *collection,
                                        ZVecCollectionStats **stats) {
  if (!collection || !stats) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get detailed collection stats",
      auto collection_ptr =
          reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
              collection);
      auto result = (*collection_ptr)->Stats();

      if (!result.has_value()) {
        set_last_error("Failed to get collection stats: " +
                       result.error().message());
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

          *stats = static_cast<ZVecCollectionStats *>(
              malloc(sizeof(ZVecCollectionStats)));
      if (!*stats) {
        set_last_error("Failed to allocate memory for stats");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      ZVecErrorCode error_code = handle_expected_result(result);
      if (error_code == ZVEC_OK) {
        (*stats)->doc_count = result.value().doc_count;
        (*stats)->index_count = result.value().index_completeness.size();
        if ((*stats)->index_count > 0) {
          (*stats)->index_completeness = static_cast<float *>(
              malloc((*stats)->index_count * sizeof(float)));
          (*stats)->index_names = static_cast<ZVecString **>(
              malloc((*stats)->index_count * sizeof(ZVecString *)));
          int i = 0;
          for (auto &[name, completeness] : result.value().index_completeness) {
            (*stats)->index_completeness[i] = completeness;
            (*stats)->index_names[i] = zvec_string_create(name.c_str());
            i++;
          }
        }
      } else {
        (*stats)->index_completeness = nullptr;
        (*stats)->index_names = nullptr;
      }

      return error_code;)
}

ZVecCollectionStats *zvec_collection_stats_create(void) {
  ZVecCollectionStats *stats =
      static_cast<ZVecCollectionStats *>(malloc(sizeof(ZVecCollectionStats)));
  if (!stats) {
    return nullptr;
  }
  stats->doc_count = 0;
  stats->index_count = 0;
  stats->index_completeness = nullptr;
  stats->index_names = nullptr;
  return stats;
}

void zvec_collection_stats_destroy(ZVecCollectionStats *stats) {
  if (stats) {
    if (stats->index_names) {
      for (size_t i = 0; i < stats->index_count; ++i) {
        zvec_free_string(stats->index_names[i]);
      }
      free(stats->index_names);
    }

    if (stats->index_completeness) {
      free(stats->index_completeness);
    }

    free(stats);
  }
}

// =============================================================================
// QueryParams functions implementation
// =============================================================================

ZVecQueryParams *zvec_query_params_create(ZVecIndexType index_type) {
  ZVecQueryParams *params =
      static_cast<ZVecQueryParams *>(malloc(sizeof(ZVecQueryParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecQueryParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  params->index_type = index_type;
  params->radius = 0.0f;
  params->is_linear = false;
  params->is_using_refiner = false;
  return params;
}

ZVecHnswQueryParams *zvec_query_params_hnsw_create(ZVecIndexType index_type,
                                                   int ef, float radius,
                                                   bool is_linear,
                                                   bool is_using_refiner) {
  ZVecHnswQueryParams *params =
      static_cast<ZVecHnswQueryParams *>(malloc(sizeof(ZVecHnswQueryParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecHnswQueryParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  params->base.index_type = index_type;
  params->base.radius = radius;
  params->base.is_linear = is_linear;
  params->base.is_using_refiner = is_using_refiner;
  params->ef = ef;
  return params;
}

ZVecIVFQueryParams *zvec_query_params_ivf_create(ZVecIndexType index_type,
                                                 int nprobe,
                                                 bool is_using_refiner,
                                                 float scale_factor) {
  ZVecIVFQueryParams *params =
      static_cast<ZVecIVFQueryParams *>(malloc(sizeof(ZVecIVFQueryParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecIVFQueryParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  params->base.index_type = index_type;
  params->base.is_using_refiner = is_using_refiner;
  params->nprobe = nprobe;
  params->scale_factor = scale_factor;
  return params;
}

ZVecFlatQueryParams *zvec_query_params_flat_create(ZVecIndexType index_type,
                                                   bool is_using_refiner,
                                                   float scale_factor) {
  ZVecFlatQueryParams *params =
      static_cast<ZVecFlatQueryParams *>(malloc(sizeof(ZVecFlatQueryParams)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecFlatQueryParams",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  params->base.index_type = index_type;
  params->base.is_using_refiner = is_using_refiner;
  params->scale_factor = scale_factor;
  return params;
}

ZVecQueryParamsUnion *zvec_query_params_union_create(ZVecIndexType index_type) {
  ZVecQueryParamsUnion *params =
      static_cast<ZVecQueryParamsUnion *>(malloc(sizeof(ZVecQueryParamsUnion)));
  if (!params) {
    set_last_error_details(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                           "Failed to allocate memory for ZVecQueryParamsUnion",
                           __FILE__, __LINE__, __FUNCTION__);
    return nullptr;
  }
  params->index_type = index_type;

  switch (index_type) {
    case ZVEC_INDEX_TYPE_HNSW:
      params->params.hnsw_params.base.index_type = index_type;
      params->params.hnsw_params.ef =
          zvec::core_interface::kDefaultHnswEfSearch;
      break;
    case ZVEC_INDEX_TYPE_IVF:
      params->params.ivf_params.base.index_type = index_type;
      params->params.ivf_params.nprobe = 10;
      params->params.ivf_params.scale_factor = 10.0f;
      break;
    case ZVEC_INDEX_TYPE_FLAT:
      params->params.flat_params.base.index_type = index_type;
      params->params.flat_params.scale_factor = 10.0f;
      break;
    default:
      params->params.base_params.index_type = index_type;
      break;
  }

  return params;
}

void zvec_query_params_destroy(ZVecQueryParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_query_params_hnsw_destroy(ZVecHnswQueryParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_query_params_ivf_destroy(ZVecIVFQueryParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_query_params_flat_destroy(ZVecFlatQueryParams *params) {
  if (params) {
    free(params);
  }
}

void zvec_query_params_union_destroy(ZVecQueryParamsUnion *params) {
  if (params) {
    free(params);
  }
}

ZVecErrorCode zvec_query_params_set_index_type(ZVecQueryParams *params,
                                               ZVecIndexType index_type) {
  if (!params) {
    set_last_error("Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->index_type = index_type;
  return ZVEC_OK;
}

ZVecErrorCode zvec_query_params_set_radius(ZVecQueryParams *params,
                                           float radius) {
  if (!params) {
    set_last_error("Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->radius = radius;
  return ZVEC_OK;
}

ZVecErrorCode zvec_query_params_set_is_linear(ZVecQueryParams *params,
                                              bool is_linear) {
  if (!params) {
    set_last_error("Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->is_linear = is_linear;
  return ZVEC_OK;
}

ZVecErrorCode zvec_query_params_set_is_using_refiner(ZVecQueryParams *params,
                                                     bool is_using_refiner) {
  if (!params) {
    set_last_error("Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->is_using_refiner = is_using_refiner;
  return ZVEC_OK;
}

ZVecErrorCode zvec_query_params_hnsw_set_ef(ZVecHnswQueryParams *params,
                                            int ef) {
  if (!params) {
    set_last_error("HNSW query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->ef = ef;
  return ZVEC_OK;
}

ZVecErrorCode zvec_query_params_ivf_set_nprobe(ZVecIVFQueryParams *params,
                                               int nprobe) {
  if (!params) {
    set_last_error("IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->nprobe = nprobe;
  return ZVEC_OK;
}

ZVecErrorCode zvec_query_params_ivf_set_scale_factor(ZVecIVFQueryParams *params,
                                                     float scale_factor) {
  if (!params) {
    set_last_error("Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  params->scale_factor = scale_factor;
  return ZVEC_OK;
}


// =============================================================================
// Index Interface Implementation
// =============================================================================

ZVecErrorCode zvec_collection_create_index(
    ZVecCollection *collection, const char *column_name,
    const ZVecIndexParams *index_params) {
  if (!collection || !column_name || !index_params) {
    set_last_error(
        "Invalid arguments: collection, column_name, and index_params cannot "
        "be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR("Exception in zvec_collection_create_index",
    auto coll_ptr =
        reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
    std::string field_name_str(column_name);

    switch (index_params->index_type) {
      case ZVEC_INDEX_TYPE_INVERT: {
        const ZVecInvertIndexParams *invert_params =
            &index_params->params.invert_params;
        auto cpp_params = std::make_shared<zvec::InvertIndexParams>(
            invert_params->enable_range_optimization,
            invert_params->enable_extended_wildcard);
        auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
        return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_HNSW: {
  const ZVecHnswIndexParams *hnsw_params = &index_params->params.hnsw_params;
  auto metric = convert_metric_type(hnsw_params->base.metric_type);
  auto quantize = convert_quantize_type(hnsw_params->base.quantize_type);
  auto cpp_params = std::make_shared<zvec::HnswIndexParams>(
      metric, hnsw_params->m, hnsw_params->ef_construction, quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_FLAT: {
  const ZVecFlatIndexParams *flat_params = &index_params->params.flat_params;
  auto metric = convert_metric_type(flat_params->base.metric_type);
  auto quantize = convert_quantize_type(flat_params->base.quantize_type);
  auto cpp_params = std::make_shared<zvec::FlatIndexParams>(metric, quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_IVF: {
  const ZVecIVFIndexParams *ivf_params = &index_params->params.ivf_params;
  auto metric = convert_metric_type(ivf_params->base.metric_type);
  auto quantize = convert_quantize_type(ivf_params->base.quantize_type);
  auto cpp_params = std::make_shared<zvec::IVFIndexParams>(
      metric, ivf_params->n_list, ivf_params->n_iters, ivf_params->use_soar,
      quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

default: {
  set_last_error("Unsupported index type");
  return ZVEC_ERROR_INVALID_ARGUMENT;
}
  }
  )
  }

  ZVecErrorCode zvec_collection_create_index_with_params(
      ZVecCollection *collection, const char *field_name,
      const void *index_params) {
    if (!collection || !field_name || !index_params) {
      set_last_error("Invalid arguments");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    auto coll_ptr =
        reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
    std::string field_name_str(field_name);

    const ZVecBaseIndexParams *base_params =
        static_cast<const ZVecBaseIndexParams *>(index_params);

  ZVEC_TRY_RETURN_ERROR("Exception occurred",
    switch (base_params->index_type) {
      case ZVEC_INDEX_TYPE_INVERT: {
        const ZVecInvertIndexParams *invert_params =
            static_cast<const ZVecInvertIndexParams *>(index_params);
        auto cpp_params = std::make_shared<zvec::InvertIndexParams>(
            invert_params->enable_range_optimization,
            invert_params->enable_extended_wildcard);
        auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
        return status_to_error_code(status);
  }

case ZVEC_INDEX_TYPE_HNSW: {
  const ZVecHnswIndexParams *hnsw_params =
      static_cast<const ZVecHnswIndexParams *>(index_params);
  auto metric = convert_metric_type(hnsw_params->base.metric_type);
  auto quantize = convert_quantize_type(hnsw_params->base.quantize_type);
  auto cpp_params = std::make_shared<zvec::HnswIndexParams>(
      metric, hnsw_params->m, hnsw_params->ef_construction, quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_FLAT: {
  const ZVecFlatIndexParams *flat_params =
      static_cast<const ZVecFlatIndexParams *>(index_params);
  auto metric = convert_metric_type(flat_params->base.metric_type);
  auto quantize = convert_quantize_type(flat_params->base.quantize_type);
  auto cpp_params = std::make_shared<zvec::FlatIndexParams>(metric, quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_IVF: {
  const ZVecIVFIndexParams *ivf_params =
      static_cast<const ZVecIVFIndexParams *>(index_params);
  auto metric = convert_metric_type(ivf_params->base.metric_type);
  auto quantize = convert_quantize_type(ivf_params->base.quantize_type);
  auto cpp_params = std::make_shared<zvec::IVFIndexParams>(
      metric, ivf_params->n_list, ivf_params->n_iters, ivf_params->use_soar,
      quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

default: {
  set_last_error("Unsupported index type");
  return ZVEC_ERROR_INVALID_ARGUMENT;
}
  }
  )
  }

  ZVecErrorCode zvec_collection_create_hnsw_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecHnswIndexParams *hnsw_params) {
    if (!hnsw_params) {
      set_last_error("Invalid HNSW parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    return zvec_collection_create_index_with_params(collection, field_name,
                                                    hnsw_params);
  }

  ZVecErrorCode zvec_collection_create_flat_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecFlatIndexParams *flat_params) {
    if (!flat_params) {
      set_last_error("Invalid Flat parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    return zvec_collection_create_index_with_params(collection, field_name,
                                                    flat_params);
  }

  ZVecErrorCode zvec_collection_create_ivf_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecIVFIndexParams *ivf_params) {
    if (!ivf_params) {
      set_last_error("Invalid IVF parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    return zvec_collection_create_index_with_params(collection, field_name,
                                                    ivf_params);
  }

  ZVecErrorCode zvec_collection_create_invert_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecInvertIndexParams *invert_params) {
    if (!invert_params) {
      set_last_error("Invalid Invert parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    return zvec_collection_create_index_with_params(collection, field_name,
                                                    invert_params);
  }

  ZVecErrorCode zvec_collection_drop_index(ZVecCollection *collection,
                                           const char *column_name) {
    if (!collection || !column_name) {
      set_last_error(
          "Invalid arguments: collection and column_name cannot be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
        zvec::Status status = (*coll_ptr)->DropIndex(column_name);
        if (!status.ok()) { set_last_error(status.message()); }

        return status_to_error_code(status);)
  }

  ZVecErrorCode zvec_collection_optimize(ZVecCollection *collection) {
    if (!collection) {
      set_last_error("Invalid argument: collection cannot be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
        zvec::Status status = (*coll_ptr)->Optimize();
        if (!status.ok()) { set_last_error(status.message()); }

        return status_to_error_code(status);)
  }


  // =============================================================================
  // Column Interface Implementation
  // =============================================================================

  ZVecErrorCode zvec_collection_add_column(ZVecCollection *collection,
                                           const ZVecFieldSchema *field_schema,
                                           const char *expression) {
    if (!collection || !field_schema) {
      set_last_error(
          "Invalid arguments: collection and field_schema cannot be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

        zvec::DataType data_type = convert_data_type(field_schema->data_type);
        if (data_type == zvec::DataType::UNDEFINED) {
          set_last_error("Invalid data type");
          return ZVEC_ERROR_INVALID_ARGUMENT;
        }

        std::string field_name(field_schema->name->data,
                               field_schema->name->length);
        bool is_vector_field = check_is_vector_field(*field_schema);
        zvec::FieldSchema::Ptr schema;
        if (is_vector_field) {
          schema = std::make_shared<zvec::FieldSchema>(field_name, data_type,
                                                       field_schema->dimension,
                                                       field_schema->nullable);
        } else {
          schema = std::make_shared<zvec::FieldSchema>(field_name, data_type,
                                                       field_schema->nullable);
        }

        std::string expr = expression ? expression : "";
        zvec::Status status = (*coll_ptr)->AddColumn(schema, expr);

        if (!status.ok()) { set_last_error(status.message()); }

        return status_to_error_code(status);)
  }

  ZVecErrorCode zvec_collection_drop_column(ZVecCollection *collection,
                                            const char *column_name) {
    if (!collection || !column_name) {
      set_last_error(
          "Invalid arguments: collection and column_name cannot be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
        zvec::Status status = (*coll_ptr)->DropColumn(column_name);

        if (!status.ok()) { set_last_error(status.message()); }

        return status_to_error_code(status);)
  }

  ZVecErrorCode zvec_collection_alter_column(
      ZVecCollection *collection, const char *column_name, const char *new_name,
      const ZVecFieldSchema *new_schema) {
    if (!collection || !column_name) {
      set_last_error(
          "Invalid arguments: collection and column_name cannot be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
        std::string rename = new_name ? new_name : "";

        zvec::FieldSchema::Ptr schema = nullptr;
        if (new_schema) {
          auto status =
              convert_zvec_field_schema_to_internal(*new_schema, schema);
          if (!status.ok()) {
            set_last_error(status.message());
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
        }

        zvec::Status status =
            (*coll_ptr)->AlterColumn(column_name, rename, schema);
        if (!status.ok()) { set_last_error(status.message()); }

        return status_to_error_code(status);)
  }

  // =============================================================================
  // DML Interface Implementation
  // =============================================================================

  ZVecErrorCode zvec_collection_insert(ZVecCollection *collection,
                                       const ZVecDoc **docs, size_t doc_count,
                                       size_t *success_count,
                                       size_t *error_count) {
    if (!collection || !docs || doc_count == 0 || !success_count ||
        !error_count) {
      set_last_error(
          "Invalid arguments: collection, docs, doc_count, success_count and "
          "error_count cannot be null/zero");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception in zvec_collection_insert_docs",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

        std::vector<zvec::Doc> internal_docs =
            convert_zvec_docs_to_internal(docs, doc_count);

        auto result = (*coll_ptr)->Insert(internal_docs);
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code == ZVEC_OK) {
          *success_count = 0;
          *error_count = 0;
          for (const auto &status : result.value()) {
            if (status.ok()) {
              (*success_count)++;
            } else {
              (*error_count)++;
            }
          }
        } else {
          *success_count = 0;
          *error_count = doc_count;
        }

        return error_code;)
  }

  ZVecErrorCode zvec_collection_update(ZVecCollection *collection,
                                       const ZVecDoc **docs, size_t doc_count,
                                       size_t *success_count,
                                       size_t *error_count) {
    if (!collection || !docs || doc_count == 0 || !success_count ||
        !error_count) {
      set_last_error(
          "Invalid arguments: collection, docs, doc_count, success_count and "
          "error_count cannot be null/zero");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

        std::vector<zvec::Doc> internal_docs =
            convert_zvec_docs_to_internal(docs, doc_count);

        auto result = (*coll_ptr)->Update(internal_docs);
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code == ZVEC_OK) {
          *success_count = 0;
          *error_count = 0;
          for (const auto &status : result.value()) {
            if (status.ok()) {
              (*success_count)++;
            } else {
              (*error_count)++;
            }
          }
        }

        return error_code;)
  }

  ZVecErrorCode zvec_collection_upsert(ZVecCollection *collection,
                                       const ZVecDoc **docs, size_t doc_count,
                                       size_t *success_count,
                                       size_t *error_count) {
    if (!collection || !docs || doc_count == 0 || !success_count ||
        !error_count) {
      set_last_error(
          "Invalid arguments: collection, docs, doc_count, success_count and "
          "error_count cannot be null/zero");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

        std::vector<zvec::Doc> internal_docs =
            convert_zvec_docs_to_internal(docs, doc_count);

        auto result = (*coll_ptr)->Upsert(internal_docs);
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code == ZVEC_OK) {
          *success_count = 0;
          *error_count = 0;
          for (const auto &status : result.value()) {
            if (status.ok()) {
              (*success_count)++;
            } else {
              (*error_count)++;
            }
          }
        }

        return error_code;)
  }

  ZVecErrorCode zvec_collection_delete(ZVecCollection *collection,
                                       const char *const *pks, size_t pk_count,
                                       size_t *success_count,
                                       size_t *error_count) {
    if (!collection || !pks || pk_count == 0 || !success_count ||
        !error_count) {
      set_last_error(
          "Invalid arguments: collection, pks, pk_count, success_count and "
          "error_count cannot be null/zero");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

        std::vector<std::string> primary_keys; primary_keys.reserve(pk_count);
        for (size_t i = 0; i < pk_count; ++i) {
          if (pks[i]) {
            primary_keys.emplace_back(pks[i]);
          }
        }

        auto result = (*coll_ptr)->Delete(primary_keys);
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code == ZVEC_OK) {
          *success_count = 0;
          *error_count = 0;
          for (const auto &status : result.value()) {
            if (status.ok()) {
              (*success_count)++;
            } else {
              (*error_count)++;
            }
          }
        }

        return error_code;)
  }

  ZVecErrorCode zvec_collection_delete_by_filter(ZVecCollection *collection,
                                                 const char *filter) {
    if (!collection || !filter) {
      set_last_error("Invalid arguments: collection,filter cannot be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

        auto status = (*coll_ptr)->DeleteByFilter(filter); if (!status.ok()) {
          set_last_error(status.message());
          return status_to_error_code(status);
        } return ZVEC_OK;)
  }

  // =============================================================================
  // Data query interface implementation
  // =============================================================================

  // Helper function to convert common query parameters
  void convert_common_query_params(zvec::VectorQuery &internal_query,
                                   const ZVecVectorQuery *query) {
    internal_query.topk_ = query->topk;
    internal_query.field_name_ =
        std::string(query->field_name.data, query->field_name.length);
    internal_query.filter_ =
        std::string(query->filter.data, query->filter.length);
    internal_query.include_vector_ = query->include_vector;
    internal_query.include_doc_id_ = query->include_doc_id;

    // Binary data conversion (query_vector)
    if (query->query_vector.data && query->query_vector.length > 0) {
      internal_query.query_vector_.assign(
          reinterpret_cast<const char *>(query->query_vector.data),
          query->query_vector.length);
    }

    // Sparse vector data conversion
    if (query->query_sparse_indices.data &&
        query->query_sparse_indices.length > 0) {
      internal_query.query_sparse_indices_.assign(
          reinterpret_cast<const char *>(query->query_sparse_indices.data),
          query->query_sparse_indices.length);
    }

    if (query->query_sparse_values.data &&
        query->query_sparse_values.length > 0) {
      internal_query.query_sparse_values_.assign(
          reinterpret_cast<const char *>(query->query_sparse_values.data),
          query->query_sparse_values.length);
    }

    // Output fields conversion
    if (query->output_fields.count > 0) {
      internal_query.output_fields_ = std::vector<std::string>();
      for (size_t i = 0; i < query->output_fields.count; ++i) {
        internal_query.output_fields_->emplace_back(
            query->output_fields.strings[i].data,
            query->output_fields.strings[i].length);
      }
    }
  }

  // Helper function to convert query parameters
  void convert_query_params(zvec::VectorQuery &internal_query,
                            const ZVecVectorQuery *query) {
    convert_common_query_params(internal_query, query);

    // QueryParams conversion
    if (query->query_params) {
      auto query_params = std::make_shared<zvec::QueryParams>(
          static_cast<zvec::IndexType>(query->query_params->index_type));

      switch (query->query_params->index_type) {
        case ZVEC_INDEX_TYPE_HNSW: {
          auto hnsw_params = std::make_shared<zvec::HnswQueryParams>(
              query->query_params->params.hnsw_params.ef,
              query->query_params->params.hnsw_params.base.radius,
              query->query_params->params.hnsw_params.base.is_linear,
              query->query_params->params.hnsw_params.base.is_using_refiner);
          internal_query.query_params_ = hnsw_params;
          break;
        }
        case ZVEC_INDEX_TYPE_IVF: {
          auto ivf_params = std::make_shared<zvec::IVFQueryParams>(
              query->query_params->params.ivf_params.nprobe,
              query->query_params->params.ivf_params.base.is_using_refiner,
              query->query_params->params.ivf_params.scale_factor);
          internal_query.query_params_ = ivf_params;
          break;
        }
        case ZVEC_INDEX_TYPE_FLAT: {
          auto flat_params = std::make_shared<zvec::FlatQueryParams>(
              query->query_params->params.flat_params.base.is_using_refiner,
              query->query_params->params.flat_params.scale_factor);
          internal_query.query_params_ = flat_params;
          break;
        }
        default: {
          query_params->set_radius(
              query->query_params->params.base_params.radius);
          query_params->set_is_linear(
              query->query_params->params.base_params.is_linear);
          query_params->set_is_using_refiner(
              query->query_params->params.base_params.is_using_refiner);
          internal_query.query_params_ = query_params;
          break;
        }
      }
    }
  }

  // Helper function to convert group by query parameters
  void convert_groupby_query_params(zvec::GroupByVectorQuery &internal_query,
                                    const ZVecGroupByVectorQuery *query) {
    internal_query.field_name_ =
        std::string(query->field_name.data, query->field_name.length);
    internal_query.filter_ =
        std::string(query->filter.data, query->filter.length);
    internal_query.include_vector_ = query->include_vector;
    internal_query.group_by_field_name_ = std::string(
        query->group_by_field_name.data, query->group_by_field_name.length);
    internal_query.group_count_ = query->group_count;
    internal_query.group_topk_ = query->group_topk;

    if (query->query_vector.data && query->query_vector.length > 0) {
      internal_query.query_vector_.assign(
          reinterpret_cast<const char *>(query->query_vector.data),
          query->query_vector.length);
    }

    if (query->query_sparse_indices.data &&
        query->query_sparse_indices.length > 0) {
      internal_query.query_sparse_indices_.assign(
          reinterpret_cast<const char *>(query->query_sparse_indices.data),
          query->query_sparse_indices.length);
    }

    if (query->query_sparse_values.data &&
        query->query_sparse_values.length > 0) {
      internal_query.query_sparse_values_.assign(
          reinterpret_cast<const char *>(query->query_sparse_values.data),
          query->query_sparse_values.length);
    }

    if (query->output_fields.count > 0) {
      if (!internal_query.output_fields_.has_value()) {
        internal_query.output_fields_ = std::vector<std::string>();
      }
      for (size_t i = 0; i < query->output_fields.count; ++i) {
        internal_query.output_fields_->push_back(
            std::string(query->output_fields.strings[i].data,
                        query->output_fields.strings[i].length));
      }
    }

    if (query->query_params) {
      auto query_params = std::make_shared<zvec::QueryParams>(
          static_cast<zvec::IndexType>(query->query_params->index_type));

      switch (query->query_params->index_type) {
        case ZVEC_INDEX_TYPE_HNSW: {
          auto hnsw_params = std::make_shared<zvec::HnswQueryParams>(
              query->query_params->params.hnsw_params.ef,
              query->query_params->params.hnsw_params.base.radius,
              query->query_params->params.hnsw_params.base.is_linear,
              query->query_params->params.hnsw_params.base.is_using_refiner);
          internal_query.query_params_ = hnsw_params;
          break;
        }
        case ZVEC_INDEX_TYPE_IVF: {
          auto ivf_params = std::make_shared<zvec::IVFQueryParams>(
              query->query_params->params.ivf_params.nprobe,
              query->query_params->params.ivf_params.base.is_using_refiner,
              query->query_params->params.ivf_params.scale_factor);
          internal_query.query_params_ = ivf_params;
          break;
        }
        case ZVEC_INDEX_TYPE_FLAT: {
          auto flat_params = std::make_shared<zvec::FlatQueryParams>(
              query->query_params->params.flat_params.base.is_using_refiner,
              query->query_params->params.flat_params.scale_factor);
          internal_query.query_params_ = flat_params;
          break;
        }
        default: {
          query_params->set_radius(
              query->query_params->params.base_params.radius);
          query_params->set_is_linear(
              query->query_params->params.base_params.is_linear);
          query_params->set_is_using_refiner(
              query->query_params->params.base_params.is_using_refiner);
          internal_query.query_params_ = query_params;
          break;
        }
      }
    }
  }

  // Helper function to convert document results to C API format
  ZVecErrorCode convert_document_results(
      const std::vector<std::shared_ptr<zvec::Doc>> &query_results,
      ZVecDoc ***results, size_t *result_count) {
    *result_count = query_results.size();
    *results =
        static_cast<ZVecDoc **>(malloc(*result_count * sizeof(ZVecDoc *)));

    if (!*results) {
      set_last_error("Failed to allocate memory for query results");
      return ZVEC_ERROR_INTERNAL_ERROR;
    }

    for (size_t i = 0; i < *result_count; ++i) {
      const auto &internal_doc = query_results[i];
      // Create new document wrapper
      ZVecDoc *c_doc = zvec_doc_create();
      if (!c_doc) {
        // Clean up previously allocated documents
        for (size_t j = 0; j < i; ++j) {
          zvec_doc_destroy((*results)[j]);
        }
        free(*results);
        *results = nullptr;
        *result_count = 0;
        set_last_error("Failed to create document wrapper");
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

      // Copy the C++ document to our wrapper
      auto doc_ptr =
          reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(c_doc);
      *(*doc_ptr) = *internal_doc;  // Copy assignment
      (*results)[i] = c_doc;        // Store the pointer, not dereference
    }

    return ZVEC_OK;
  }

  // Helper function to convert grouped document results to C API format
  ZVecErrorCode convert_grouped_document_results(
      const std::vector<zvec::GroupResult> &group_results, ZVecDoc ***results,
      ZVecString ***group_by_values, size_t *result_count) {
    // Calculate total document count across all groups
    size_t total_docs = 0;
    for (const auto &group_result : group_results) {
      total_docs += group_result.docs_.size();
    }

    // Allocate memory for document pointers and group by values
    *result_count = total_docs;
    *results =
        static_cast<ZVecDoc **>(malloc(*result_count * sizeof(ZVecDoc *)));
    *group_by_values = static_cast<ZVecString **>(
        malloc(group_results.size() * sizeof(ZVecString *)));

    if (!*results) {
      set_last_error("Failed to allocate memory for query results");
      return ZVEC_ERROR_INTERNAL_ERROR;
    }

    // Convert C++ grouped results to C API format
    size_t doc_index = 0;
    for (const auto &group_result : group_results) {
      for (const auto &internal_doc : group_result.docs_) {
        if (doc_index >= *result_count) {
          break;
        }

        // Create new document wrapper
        ZVecDoc *c_doc = zvec_doc_create();
        if (!c_doc) {
          // Clean up previously allocated documents
          for (size_t j = 0; j < doc_index; ++j) {
            zvec_doc_destroy((*results)[j]);
          }
          free(*results);
          *results = nullptr;
          *result_count = 0;
          set_last_error("Failed to create document wrapper");
          return ZVEC_ERROR_INTERNAL_ERROR;
        }

        // Copy the C++ document to our wrapper
        auto doc_ptr =
            reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(c_doc);
        *(*doc_ptr) = internal_doc;  // Copy assignment

        ZVecString *c_group_value =
            zvec_string_create(group_result.group_by_value_.c_str());
        if (!c_group_value) {
          for (size_t j = 0; j < doc_index; ++j) {
            zvec_doc_destroy((*results)[j]);
            zvec_free_string((*group_by_values)[doc_index]);
          }
          free(*results);
          *results = nullptr;
          *result_count = 0;
          set_last_error("Failed to create string wrapper");
          return ZVEC_ERROR_INTERNAL_ERROR;
        }

        (*group_by_values)[doc_index] = c_group_value;
        (*results)[doc_index] = c_doc;
        ++doc_index;
      }
    }

    return ZVEC_OK;
  }

  // Helper function to convert fetched document results to C API format
  ZVecErrorCode convert_fetched_document_results(const zvec::DocPtrMap &doc_map,
                                                 ZVecDoc ***results,
                                                 size_t *doc_count) {
    // Calculate actual document count (some PKs might not exist)
    size_t actual_count = 0;
    for (const auto &[pk, doc_ptr] : doc_map) {
      if (doc_ptr) {
        actual_count++;
      }
    }

    // Allocate memory for document pointers
    *doc_count = actual_count;
    if (*doc_count == 0) {
      *results = nullptr;
      return ZVEC_OK;
    }

    *results = static_cast<ZVecDoc **>(malloc(*doc_count * sizeof(ZVecDoc *)));
    if (!*results) {
      set_last_error("Failed to allocate memory for document pointers");
      return ZVEC_ERROR_INTERNAL_ERROR;
    }

    // Convert C++ DocPtrMap to C ZVecDoc pointer array
    size_t index = 0;
    for (const auto &[pk, doc_ptr] : doc_map) {
      if (doc_ptr && index < *doc_count) {
        // Create new document wrapper
        ZVecDoc *c_doc = zvec_doc_create();
        if (!c_doc) {
          // Clean up previously allocated documents
          for (size_t j = 0; j < index; ++j) {
            zvec_doc_destroy((*results)[j]);
          }
          free(*results);
          *results = nullptr;
          *doc_count = 0;
          set_last_error("Failed to create document wrapper");
          return ZVEC_ERROR_INTERNAL_ERROR;
        }

        // Copy the C++ document to our wrapper
        auto cpp_doc_ptr =
            reinterpret_cast<std::shared_ptr<zvec::Doc> *>(c_doc);
        *(*cpp_doc_ptr) = *doc_ptr;  // Copy assignment

        // Set the primary key explicitly
        zvec_doc_set_pk(c_doc, pk.c_str());

        (*results)[index] = c_doc;
        ++index;
      }
    }

    return ZVEC_OK;
  }

  ZVecErrorCode zvec_collection_query(const ZVecCollection *collection,
                                      const ZVecVectorQuery *query,
                                      ZVecDoc ***results,
                                      size_t *result_count) {
    if (!collection || !query || !results || !result_count) {
      set_last_error(
          "Invalid arguments: collection, query, results and result_count "
          "cannot "
          "be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
                collection);

        // Convert query parameters using helper function
        zvec::VectorQuery internal_query;
        convert_query_params(internal_query, query);

        auto result = (*coll_ptr)->Query(internal_query);
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code == ZVEC_OK) {
          const auto &query_results = result.value();
          error_code =
              convert_document_results(query_results, results, result_count);
        } else {
          *results = nullptr;
          *result_count = 0;
        }

        return error_code;)
  }

  ZVecErrorCode zvec_collection_query_by_group(
      const ZVecCollection *collection, const ZVecGroupByVectorQuery *query,
      ZVecDoc ***results, ZVecString ***group_by_values, size_t *result_count) {
    if (!collection || !query || !results || !group_by_values ||
        !result_count) {
      set_last_error(
          "Invalid arguments: collection, query, results, group_by_values and "
          "result_count cannot "
          "be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception occurred",
        auto coll_ptr =
            reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
                collection);

        zvec::GroupByVectorQuery internal_query;
        convert_groupby_query_params(internal_query, query);

        auto result = (*coll_ptr)->GroupByQuery(internal_query);
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code == ZVEC_OK) {
          const auto &group_results = result.value();
          error_code = convert_grouped_document_results(
              group_results, results, group_by_values, result_count);
        } else {
          *results = nullptr;
          *group_by_values = nullptr;
          *result_count = 0;
        }

        return error_code;)
  }

  ZVecErrorCode zvec_collection_fetch(ZVecCollection *collection,
                                      const char *const *pks, size_t pk_count,
                                      ZVecDoc ***results, size_t *doc_count) {
    if (!collection || !pks || !results || !doc_count) {
      set_last_error(
          "Invalid arguments: collection, pks, results and doc_count cannot "
          "be null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    // Handle empty case
    if (pk_count == 0) {
      *results = nullptr;
      *doc_count = 0;
      return ZVEC_OK;
    }

    ZVEC_TRY_RETURN_ERROR(
        "Exception in zvec_collection_fetch",
        auto coll_ptr =
            reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
                collection);

        // Convert C array to C++ vector
        std::vector<std::string> pk_vector; pk_vector.reserve(pk_count);
        for (size_t i = 0; i < pk_count; ++i) {
          if (pks[i]) {
            pk_vector.emplace_back(pks[i]);
          } else {
            set_last_error("Null primary key at index " + std::to_string(i));
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
        }

        // Call C++ fetch method
        auto result = (*coll_ptr)->Fetch(pk_vector);
        if (!result.has_value()) {
          set_last_error("Failed to fetch documents: " +
                         result.error().message());
          return ZVEC_ERROR_INTERNAL_ERROR;
        }

        const auto &doc_map = result.value();
        return convert_fetched_document_results(doc_map, results, doc_count);)
  }
