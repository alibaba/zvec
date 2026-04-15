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

// clang-format off
#include "zvec/c_api.h"
// Include generated version header for build-time
#if defined(__has_include) && __has_include(<zvec_version.h>)
#include <zvec_version.h>
#endif
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include <zvec/db/stats.h>

// Error checking macros - these preserve __LINE__ accuracy
// Simplified macro for setting error with automatic file/line/function info
#define SET_LAST_ERROR(code, msg) \
  set_last_error_details(code, msg, __FILE__, __LINE__, __FUNCTION__)

#define ZVEC_CHECK_NOTNULL(ptr, error_code, msg) \
  if (!(ptr)) {                                  \
    SET_LAST_ERROR(error_code, msg);             \
    return nullptr;                              \
  }

#define ZVEC_CHECK_NOTNULL_ERRCODE(ptr, error_code, msg) \
  if (!(ptr)) {                                          \
    SET_LAST_ERROR(error_code, msg);                     \
    return (error_code);                                 \
  }

#define ZVEC_CHECK_COND(cond, error_code, msg) \
  if (cond) {                                  \
    SET_LAST_ERROR(error_code, msg);           \
    return nullptr;                            \
  }

#define ZVEC_CHECK_COND_ERRCODE(cond, error_code, msg) \
  if (cond) {                                          \
    SET_LAST_ERROR(error_code, msg);                   \
    return (error_code);                               \
  }

// For void functions (no return value):
#define ZVEC_TRY_BEGIN_VOID try {
#define ZVEC_CATCH_END_VOID                                                    \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    SET_LAST_ERROR(ZVEC_ERROR_UNKNOWN, std::string("Exception: ") + e.what()); \
  }

// For functions returning zvec_error_code_t - complete try-catch wrapper
#define ZVEC_TRY_BEGIN_CODE ZVEC_TRY_BEGIN_VOID
#define ZVEC_CATCH_END_CODE(code_on_error)                                     \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    SET_LAST_ERROR(ZVEC_ERROR_UNKNOWN, std::string("Exception: ") + e.what()); \
    return code_on_error;                                                      \
  }                                                                            \
  return ZVEC_OK;

// For functions returning pointer - complete try-catch wrapper
// Usage: ZVEC_TRY_RETURN_NULL("error msg", code...)
// Note: Use variadic macro to handle commas in template arguments
#define ZVEC_TRY_RETURN_NULL(msg, ...)                  \
  try {                                                 \
    {                                                   \
      __VA_ARGS__                                       \
    }                                                   \
  } catch (const std::bad_alloc &e) {                   \
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,       \
                   std::string(msg) + ": " + e.what()); \
    return nullptr;                                     \
  } catch (const std::exception &e) {                   \
    SET_LAST_ERROR(ZVEC_ERROR_INTERNAL_ERROR,           \
                   std::string(msg) + ": " + e.what()); \
    return nullptr;                                     \
  }

// For functions returning ErrorCode
// Usage: ZVEC_TRY_RETURN_ERROR("error msg", code...)
// Note: Use variadic macro to handle commas in template arguments
#define ZVEC_TRY_RETURN_ERROR(msg, ...)                 \
  try {                                                 \
    {                                                   \
      __VA_ARGS__                                       \
    }                                                   \
  } catch (const std::bad_alloc &e) {                   \
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,       \
                   std::string(msg) + ": " + e.what()); \
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;               \
  } catch (const std::exception &e) {                   \
    SET_LAST_ERROR(ZVEC_ERROR_INTERNAL_ERROR,           \
                   std::string(msg) + ": " + e.what()); \
    return ZVEC_ERROR_INTERNAL_ERROR;                   \
  }

// For functions returning scalar values (int, float, size_t, etc.)
// Usage: ZVEC_TRY_RETURN_SCALAR("error msg", error_value, code...)
// Note: Use variadic macro to handle commas in template arguments
#define ZVEC_TRY_RETURN_SCALAR(msg, error_val, ...)     \
  try {                                                 \
    {                                                   \
      __VA_ARGS__                                       \
    }                                                   \
  } catch (const std::bad_alloc &e) {                   \
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,       \
                   std::string(msg) + ": " + e.what()); \
    return (error_val);                                 \
  } catch (const std::exception &e) {                   \
    SET_LAST_ERROR(ZVEC_ERROR_INTERNAL_ERROR,           \
                   std::string(msg) + ": " + e.what()); \
    return (error_val);                                 \
  }

// Global status flags
static std::atomic<bool> g_initialized{false};
static std::mutex g_init_mutex;

// Thread-local storage for error information
static thread_local std::string last_error_message;
static thread_local zvec_error_details_t last_error_details;

// Helper function: set error information (noexcept to avoid exceptions in error handling)
static void set_last_error(const std::string &msg) noexcept {
  try {
    last_error_message = msg;
    last_error_details.code = ZVEC_ERROR_UNKNOWN;
    last_error_details.message = last_error_message.c_str();
    last_error_details.file = nullptr;
    last_error_details.line = 0;
    last_error_details.function = nullptr;
  } catch (...) {
    // If we can't even store the error message, at least set the code
    last_error_details.code = ZVEC_ERROR_RESOURCE_EXHAUSTED;
    last_error_details.message = "Out of memory";
  }
}

// Error setting function with detailed information (noexcept to avoid exceptions in error handling)
static void set_last_error_details(zvec_error_code_t code, const std::string &msg,
                                   const char *file = nullptr, int line = 0,
                                   const char *function = nullptr) noexcept {
  try {
    last_error_message = msg;
    last_error_details.code = code;
    last_error_details.message = last_error_message.c_str();
    last_error_details.file = file;
    last_error_details.line = line;
    last_error_details.function = function;
  } catch (...) {
    // If memory allocation fails, at least set the error code
    last_error_details.code = ZVEC_ERROR_RESOURCE_EXHAUSTED;
    last_error_details.message = "Out of memory";
  }
}

// =============================================================================
// Version information interface implementation
// =============================================================================

// Version string is generated at compile time by CMake
const char *zvec_get_version(void) {
  // ZVEC_VERSION_STRING is a compile-time constant from zvec_version.h
  // Format: "vX.Y.Z-commit-hash" or "g<short-commit-hash>"
  return ZVEC_VERSION_STRING;
}

bool zvec_check_version(int major, int minor, int patch) {
  if (major < 0 || minor < 0 || patch < 0) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Version numbers must be non-negative");
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

zvec_string_t *zvec_string_create(const char *str) {
  if (!str) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointer cannot be null");
    return nullptr;
  }

  size_t len = strlen(str);
  zvec_string_t *zstr = static_cast<zvec_string_t *>(malloc(sizeof(zvec_string_t)));
  if (!zstr) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for zvec_string_t");
    return nullptr;
  }

  char *data_buffer = static_cast<char *>(malloc(len + 1));
  if (!data_buffer) {
    free(zstr);
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for string data");
    return nullptr;
  }

  memcpy(data_buffer, str, len + 1);
  zstr->data = data_buffer;
  zstr->length = len;
  zstr->capacity = len + 1;
  return zstr;
}

zvec_string_t *zvec_string_create_from_view(const zvec_string_view_t *view) {
  if (!view || !view->data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String view or data cannot be null");
    return nullptr;
  }

  zvec_string_t *zstr = static_cast<zvec_string_t *>(malloc(sizeof(zvec_string_t)));
  if (!zstr) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for zvec_string_t");
    return nullptr;
  }

  char *data_buffer = static_cast<char *>(malloc(view->length + 1));
  if (!data_buffer) {
    free(zstr);
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for string data");
    return nullptr;
  }

  memcpy(data_buffer, view->data, view->length);
  data_buffer[view->length] = '\0';
  zstr->data = data_buffer;
  zstr->length = view->length;
  zstr->capacity = view->length + 1;

  return zstr;
}

zvec_string_t *zvec_bin_create(const uint8_t *data, size_t length) {
  if (!data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Binary data pointer cannot be null");
    return nullptr;
  }

  zvec_string_t *zstr = static_cast<zvec_string_t *>(malloc(sizeof(zvec_string_t)));
  if (!zstr) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for zvec_string_t");
    return nullptr;
  }

  char *data_buffer = static_cast<char *>(malloc(length + 1));
  if (!data_buffer) {
    free(zstr);
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for binary data");
    return nullptr;
  }

  memcpy(data_buffer, data, length);
  data_buffer[length] = '\0';
  zstr->data = data_buffer;
  zstr->length = length;
  zstr->capacity = length + 1;

  return zstr;
}

zvec_string_t *zvec_string_copy(const zvec_string_t *str) {
  if (!str || !str->data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Source string or data cannot be null");
    return nullptr;
  }

  return zvec_string_create(str->data);
}

const char *zvec_string_c_str(const zvec_string_t *str) {
  if (!str) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointer cannot be null");
    return nullptr;
  }

  return str->data;
}

size_t zvec_string_length(const zvec_string_t *str) {
  if (!str) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointer cannot be null");
    return 0;
  }

  return str->length;
}

int zvec_string_compare(const zvec_string_t *str1, const zvec_string_t *str2) {
  if (!str1 || !str2) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointers cannot be null");
    return -1;
  }

  if (!str1->data || !str2->data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "String data cannot be null");
    return -1;
  }

  return strcmp(str1->data, str2->data);
}

// =============================================================================
// Configuration-related functions implementation
// =============================================================================

zvec_log_config_t *zvec_config_log_create_console(zvec_log_level_t level) {
  try {
    auto *config = new zvec::GlobalConfig::ConsoleLogConfig(
        static_cast<zvec::GlobalConfig::LogLevel>(level));
    return reinterpret_cast<zvec_log_config_t *>(config);
  } catch (const std::exception &e) {
    SET_LAST_ERROR(ZVEC_ERROR_INTERNAL_ERROR, e.what());
    return nullptr;
  }
}

zvec_log_config_t *zvec_config_log_create_file(zvec_log_level_t level, const char *dir,
                                           const char *basename,
                                           uint32_t file_size,
                                           uint32_t overdue_days) {
  if (!dir || !basename) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Directory or basename cannot be null");
    return nullptr;
  }

  try {
    auto *config = new zvec::GlobalConfig::FileLogConfig(
        static_cast<zvec::GlobalConfig::LogLevel>(level), std::string(dir),
        std::string(basename), file_size, overdue_days);
    return reinterpret_cast<zvec_log_config_t *>(config);
  } catch (const std::exception &e) {
    SET_LAST_ERROR(ZVEC_ERROR_INTERNAL_ERROR, e.what());
    return nullptr;
  }
}

void zvec_config_log_destroy(zvec_log_config_t *config) {
  if (config) {
    delete reinterpret_cast<const zvec::GlobalConfig::LogConfig *>(config);
  }
}

zvec_log_level_t zvec_config_log_get_level(const zvec_log_config_t *config) {
  if (!config) {
    return ZVEC_LOG_LEVEL_WARN;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::LogConfig *>(config);
  return static_cast<zvec_log_level_t>(cpp_config->level);
}

zvec_error_code_t zvec_config_log_set_level(zvec_log_config_t *config,
                                        zvec_log_level_t level) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::LogConfig *>(config);
  cpp_config->level = static_cast<zvec::GlobalConfig::LogLevel>(level);
  return ZVEC_OK;
}

bool zvec_config_log_is_file_type(const zvec_log_config_t *config) {
  if (!config) {
    return false;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::LogConfig *>(config);
  return cpp_config->GetLoggerType() == zvec::FILE_LOG_TYPE_NAME;
}

inline const zvec::GlobalConfig::FileLogConfig* file_config_from_config(const zvec_log_config_t *config) {
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::LogConfig *>(config);
  return dynamic_cast<const zvec::GlobalConfig::FileLogConfig *>(cpp_config);
}

inline zvec::GlobalConfig::FileLogConfig* mutable_file_config_from_config(zvec_log_config_t *config) {
  auto *cpp_config =
      reinterpret_cast<zvec::GlobalConfig::LogConfig *>(config);
  return dynamic_cast<zvec::GlobalConfig::FileLogConfig *>(cpp_config);
}

const char *zvec_config_log_get_dir(const zvec_log_config_t *config) {
  auto* file_config = file_config_from_config(config);
  return file_config->dir.c_str();
}

zvec_error_code_t zvec_config_log_set_dir(zvec_log_config_t *config, const char *dir) {
  auto *file_config = mutable_file_config_from_config(config);
  file_config->dir = dir;
  return ZVEC_OK;
}

const char *zvec_config_log_get_basename(const zvec_log_config_t *config) {
  auto* file_config = file_config_from_config(config);
  return file_config->basename.c_str();
}

zvec_error_code_t zvec_config_log_set_basename(zvec_log_config_t *config,
                                           const char *basename) {
  auto *file_config = mutable_file_config_from_config(config);
  file_config->basename = basename;
  return ZVEC_OK;
}

uint32_t zvec_config_log_get_file_size(const zvec_log_config_t *config) {
  auto* file_config = file_config_from_config(config);
  return file_config->file_size;
}

zvec_error_code_t zvec_config_log_set_file_size(zvec_log_config_t *config,
                                            uint32_t file_size) {
  auto *file_config = mutable_file_config_from_config(config);
  file_config->file_size = file_size;
  return ZVEC_OK;
}

uint32_t zvec_config_log_get_overdue_days(const zvec_log_config_t *config) {
  auto* file_config = file_config_from_config(config);
  return file_config ? file_config->overdue_days : 0;
}

zvec_error_code_t zvec_config_log_set_overdue_days(zvec_log_config_t *config,
                                               uint32_t days) {
  auto *file_config = mutable_file_config_from_config(config);
  file_config->overdue_days = days;
  return ZVEC_OK;
}

// ============================================================================
// Configuration Data Management Functions
// ============================================================================

zvec_config_data_t *zvec_config_data_create(void) {
  try {
    auto *config = new zvec::GlobalConfig::ConfigData();
    return reinterpret_cast<zvec_config_data_t *>(config);
  } catch (const std::exception &e) {
    SET_LAST_ERROR(ZVEC_ERROR_INTERNAL_ERROR, e.what());
    return nullptr;
  }
}

void zvec_config_data_destroy(zvec_config_data_t *config) {
  if (config) {
    delete reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);
  }
}

zvec_error_code_t zvec_config_data_set_memory_limit(zvec_config_data_t *config,
                                                uint64_t memory_limit_bytes) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);
  cpp_config->memory_limit_bytes = memory_limit_bytes;
  return ZVEC_OK;
}

uint64_t zvec_config_data_get_memory_limit(const zvec_config_data_t *config) {
  if (!config) {
    return 0;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
  return cpp_config->memory_limit_bytes;
}

zvec_error_code_t zvec_config_data_set_log_config(zvec_config_data_t *config,
                                              zvec_log_config_t *log_config) {
  if (!config || !log_config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Config or log_config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);

  // Convert raw pointer to shared_ptr for C++ internal use
  auto *log_config_raw =
      reinterpret_cast<zvec::GlobalConfig::LogConfig *>(log_config);
  cpp_config->log_config = std::shared_ptr<zvec::GlobalConfig::LogConfig>(
      log_config_raw, [](zvec::GlobalConfig::LogConfig *ptr) { delete ptr; });

  return ZVEC_OK;
}

zvec_log_type_t zvec_config_data_get_log_type(const zvec_config_data_t *config) {
  if (!config) {
    return ZVEC_LOG_TYPE_CONSOLE;
  }

  const auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
  if (!cpp_config->log_config) {
    return ZVEC_LOG_TYPE_CONSOLE;
  }

  if (cpp_config->log_config->GetLoggerType() == zvec::FILE_LOG_TYPE_NAME) {
    return ZVEC_LOG_TYPE_FILE;
  }
  return ZVEC_LOG_TYPE_CONSOLE;
}

zvec_error_code_t zvec_config_data_set_query_thread_count(zvec_config_data_t *config,
                                                      uint32_t thread_count) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);
  cpp_config->query_thread_count = thread_count;
  return ZVEC_OK;
}

uint32_t zvec_config_data_get_query_thread_count(const zvec_config_data_t *config) {
  if (!config) {
    return 1;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
  return cpp_config->query_thread_count;
}

zvec_error_code_t zvec_config_data_set_invert_to_forward_scan_ratio(
    zvec_config_data_t *config, float ratio) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);
  cpp_config->invert_to_forward_scan_ratio = ratio;
  return ZVEC_OK;
}

float zvec_config_data_get_invert_to_forward_scan_ratio(
    const zvec_config_data_t *config) {
  if (!config) {
    return 0.0f;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
  return cpp_config->invert_to_forward_scan_ratio;
}

zvec_error_code_t zvec_config_data_set_brute_force_by_keys_ratio(
    zvec_config_data_t *config, float ratio) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);
  cpp_config->brute_force_by_keys_ratio = ratio;
  return ZVEC_OK;
}

float zvec_config_data_get_brute_force_by_keys_ratio(
    const zvec_config_data_t *config) {
  if (!config) {
    return 0.0f;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
  return cpp_config->brute_force_by_keys_ratio;
}

zvec_error_code_t zvec_config_data_set_optimize_thread_count(
    zvec_config_data_t *config, uint32_t thread_count) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_config = reinterpret_cast<zvec::GlobalConfig::ConfigData *>(config);
  cpp_config->optimize_thread_count = thread_count;
  return ZVEC_OK;
}

uint32_t zvec_config_data_get_optimize_thread_count(
    const zvec_config_data_t *config) {
  if (!config) {
    return 1;
  }
  auto *cpp_config =
      reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
  return cpp_config->optimize_thread_count;
}


// =============================================================================
// Initialization and cleanup interface implementation
// =============================================================================

zvec_error_code_t zvec_initialize(const zvec_config_data_t *config) {
  std::lock_guard<std::mutex> lock(g_init_mutex);

  if (g_initialized.load()) {
    SET_LAST_ERROR(ZVEC_ERROR_ALREADY_EXISTS, "Library already initialized");
    return ZVEC_ERROR_ALREADY_EXISTS;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Initialization failed",
      // Convert to C++ configuration object
      zvec::GlobalConfig::ConfigData cpp_config{};

      if (config) {
        auto *cpp_config_data =
            reinterpret_cast<const zvec::GlobalConfig::ConfigData *>(config);
        cpp_config = *cpp_config_data;  // Copy the C++ ConfigData
      } else {
        // Initialize with default configuration
        cpp_config = zvec::GlobalConfig::ConfigData{};
      }

      // Initialize global configuration
      auto status = zvec::GlobalConfig::Instance().Initialize(cpp_config);
      if (!status.ok()) {
        set_last_error(status.message());
        return ZVEC_ERROR_INTERNAL_ERROR;
      }

      g_initialized.store(true);
      return ZVEC_OK;)
}

zvec_error_code_t zvec_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);

  if (!g_initialized.load()) {
    SET_LAST_ERROR(ZVEC_ERROR_FAILED_PRECONDITION, "Library not initialized");
    return ZVEC_ERROR_FAILED_PRECONDITION;
  }
  // We're do nothing here for now, 
  // but we might add zvec::GlobalConfig::Finalize in the future.
  ZVEC_TRY_RETURN_ERROR("Shutdown failed", g_initialized.store(false);
                        return ZVEC_OK;)
}

bool zvec_is_initialized(void) {
  return g_initialized.load();
}

// =============================================================================
// Error handling interface implementation
// =============================================================================

zvec_error_code_t zvec_get_last_error_details(zvec_error_details_t *error_details) {
  if (!error_details) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Error details pointer cannot be null");
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
static zvec_error_code_t status_to_error_code(const zvec::Status &status) {
  if (status.code() < zvec::StatusCode::OK ||
      status.code() > zvec::StatusCode::UNKNOWN) {
    set_last_error("Unexpected status code: " +
                   std::to_string(static_cast<int>(status.code())));
    return ZVEC_ERROR_UNKNOWN;
  }

  return static_cast<zvec_error_code_t>(status.code());
}

// Helper function: handle Expected results
template <typename T>
static zvec_error_code_t handle_expected_result(
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

/**
 * @brief Copy a C++ string to C heap-allocated string
 * @param str String to copy
 * @return Newly allocated C string, or NULL on failure
 * @note Caller must free() the returned string
 */
static char *copy_string(const std::string &str) {
  if (str.empty()) return nullptr;
  size_t len = str.length();
  char *copy = static_cast<char *>(malloc(len + 1));
  if (!copy) return nullptr;
  strncpy(copy, str.c_str(), len);
  copy[len] = '\0';  // Ensure null-termination
  return copy;
}

/**
 * @brief Free write results array returned by DML APIs
 * @param results Results array to free
 * @param result_count Number of results
 */
static void free_write_results_internal(zvec_write_result_t *results,
                                        size_t result_count) {
  if (!results) {
    return;
  }
  for (size_t i = 0; i < result_count; ++i) {
    // pk is not stored (ordered style), only free message
    if (results[i].message) {
      free((void *)results[i].message);
      results[i].message = nullptr;
    }
  }
  free(results);
}

// Helper function: convert per-doc statuses to C API write result array.
static zvec_error_code_t build_write_results(
    const std::vector<zvec::Status> &statuses, zvec_write_result_t **results,
    size_t *result_count) {
  if (!results || !result_count) {
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *result_count = statuses.size();
  if (*result_count == 0) {
    *results = nullptr;
    return ZVEC_OK;
  }

  *results = static_cast<zvec_write_result_t *>(
      calloc(*result_count, sizeof(zvec_write_result_t)));
  if (!*results) {
    set_last_error("Failed to allocate memory for write results");
    return ZVEC_ERROR_INTERNAL_ERROR;
  }

  // Use ordered style: result index corresponds to input index.
  // No need to store pk in result, caller can access by index.
  for (size_t i = 0; i < *result_count; ++i) {
    const std::string message = statuses[i].message();
    (*results)[i].message = copy_string(message);
    (*results)[i].code = status_to_error_code(statuses[i]);
  }

  return ZVEC_OK;
}

static std::vector<std::string> collect_doc_pks(const zvec_doc_t **docs,
                                                size_t doc_count) {
  std::vector<std::string> pks;
  pks.reserve(doc_count);
  for (size_t i = 0; i < doc_count; ++i) {
    if (!docs[i]) {
      pks.emplace_back("");
      continue;
    }
    auto *doc_ptr = reinterpret_cast<const zvec::Doc *>(docs[i]);
    pks.emplace_back(doc_ptr->pk_ref());
  }
  return pks;
}

// =============================================================================
// Type conversion helpers
// =============================================================================

/**
 * @brief Convert C index params to C++ shared_ptr
 * @param params C index params handle
 * @return Shared pointer to C++ IndexParams, or nullptr on failure
 * @note Uses switch-based type checking and copy constructor to create 
 *       a deep copy of the underlying C++ IndexParams object
 */
static std::shared_ptr<zvec::IndexParams> convert_c_index_params_to_cpp(
    const zvec_index_params_t *params) {
  if (!params) {
    return nullptr;
  }

  // Get the underlying IndexParams and create a shared_ptr from it
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);

  // Clone the params based on type
  switch (cpp_params->type()) {
    case zvec::IndexType::FLAT: {
      auto *flat_params =
          dynamic_cast<const zvec::FlatIndexParams *>(cpp_params);
      return flat_params ? std::make_shared<zvec::FlatIndexParams>(*flat_params)
                         : nullptr;
    }
    case zvec::IndexType::HNSW: {
      auto *hnsw_params =
          dynamic_cast<const zvec::HnswIndexParams *>(cpp_params);
      return hnsw_params ? std::make_shared<zvec::HnswIndexParams>(*hnsw_params)
                         : nullptr;
    }
    case zvec::IndexType::IVF: {
      auto *ivf_params = dynamic_cast<const zvec::IVFIndexParams *>(cpp_params);
      return ivf_params ? std::make_shared<zvec::IVFIndexParams>(*ivf_params)
                        : nullptr;
    }
    case zvec::IndexType::INVERT: {
      auto *invert_params =
          dynamic_cast<const zvec::InvertIndexParams *>(cpp_params);
      return invert_params
                 ? std::make_shared<zvec::InvertIndexParams>(*invert_params)
                 : nullptr;
    }
    default:
      return nullptr;
  }
}

// =============================================================================
// Memory Management interface implementation
// =============================================================================

/**
 * @brief Allocate memory within the library
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 * 
 * @note Use zvec_malloc instead of malloc to ensure memory is managed
 *       consistently within the library. All memory allocated with zvec_malloc
 *       should be freed with zvec_free.
 */
void* zvec_malloc(size_t size) {
  return malloc(size);
}

/**
 * @brief Free memory allocated by zvec_malloc
 * @param ptr Pointer to memory to free (can be NULL)
 * 
 * @note Use zvec_free instead of free to ensure memory is managed
 *       consistently within the library. This function should be used to free
 *       any memory allocated with zvec_malloc or returned by library functions
 *       that document they return library-allocated memory.
 */
void zvec_free(void *ptr) {
  if (ptr) {
    free(ptr);
  }
}

/**
 * @brief Free a zvec_string_t structure
 * @param str String structure to free (can be NULL)
 */
void zvec_free_string(zvec_string_t *str) {
  if (str) {
    if (str->data) {
      free((void *)str->data);
    }
    free(str);
  }
}

/**
 * @brief Create a string array with given count
 * @param count Number of strings in the array
 * @return New string array, or NULL on failure
 */
zvec_string_array_t *zvec_string_array_create(size_t count) {
  zvec_string_array_t *array = (zvec_string_array_t *)malloc(sizeof(zvec_string_array_t));
  array->count = count;
  array->strings = (zvec_string_t *)malloc(sizeof(zvec_string_t) * count);
  memset(array->strings, 0, sizeof(zvec_string_t) * count);
  return array;
}

/**
 * @brief Create a string array from C-string array
 * @param strings Array of C-strings
 * @param count Number of strings
 * @return New string array, or NULL on failure
 */
zvec_string_array_t *zvec_string_array_create_from_strings(const char **strings,
                                                       size_t count) {
  if (!strings || count == 0) {
    return nullptr;
  }
  zvec_string_array_t *array = zvec_string_array_create(count);
  for (size_t i = 0; i < count; ++i) {
    zvec_string_array_add(array, i, strings[i]);
  }
  return array;
}

/**
 * @brief Add a string to string array at specified index
 * @param array String array to add to
 * @param idx Index to add at
 * @param str String to add
 */
void zvec_string_array_add(zvec_string_array_t *array, size_t idx,
                           const char *str) {
  if (idx >= array->count) return;
  size_t len = strlen(str);
  array->strings[idx].data = (char *)malloc(len + 1);
  memcpy(array->strings[idx].data, str, len + 1);
  array->strings[idx].length = len;
  array->strings[idx].capacity = len + 1;
}

/**
 * @brief Destroy a string array and free all memory
 * @param array String array to destroy (can be NULL)
 */
void zvec_string_array_destroy(zvec_string_array_t *array) {
  if (!array) return;
  for (size_t i = 0; i < array->count; i++) {
    if (array->strings[i].data) {
      free((void *)array->strings[i].data);
    }
  }
  free(array->strings);
  free(array);
}


/**
 * @brief Create a mutable byte array with given capacity
 * @param capacity Initial capacity in bytes
 * @return New byte array, or NULL on failure
 */
// Byte array helper functions
zvec_mutable_byte_array_t *zvec_byte_array_create(size_t capacity) {
  zvec_mutable_byte_array_t *array =
      (zvec_mutable_byte_array_t *)malloc(sizeof(zvec_mutable_byte_array_t));
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

/**
 * @brief Destroy a byte array and free all memory
 * @param array Byte array to destroy (can be NULL)
 */
void zvec_byte_array_destroy(zvec_mutable_byte_array_t *array) {
  if (!array) return;
  if (array->data) {
    free(array->data);
  }
  free(array);
}

/**
 * @brief Create a float array with given count
 * @param count Number of floats in the array
 * @return New float array, or NULL on failure
 */
// Float array helper functions
zvec_float_array_t *zvec_float_array_create(size_t count) {
  zvec_float_array_t *array = (zvec_float_array_t *)malloc(sizeof(zvec_float_array_t));
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

/**
 * @brief Destroy a float array and free all memory
 * @param array Float array to destroy (can be NULL)
 */
void zvec_float_array_destroy(zvec_float_array_t *array) {
  if (!array) return;
  if (array->data) {
    free((void *)array->data);
  }
  free(array);
}

// Int64 array helper functions
zvec_int64_array_t *zvec_int64_array_create(size_t count) {
  zvec_int64_array_t *array = (zvec_int64_array_t *)malloc(sizeof(zvec_int64_array_t));
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

void zvec_int64_array_destroy(zvec_int64_array_t *array) {
  if (!array) return;
  if (array->data) {
    free((void *)array->data);
  }
  free(array);
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

zvec_error_code_t zvec_get_last_error(char **error_msg) {
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

void zvec_free_field_schema(zvec_field_schema_t *field_schema) {
  if (field_schema) {
    // index_params is embedded, no need to free
    free(field_schema);
  }
}

// =============================================================================
// CollectionOptions functions implementation
// =============================================================================

zvec_collection_options_t *zvec_collection_options_create(void) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create zvec_collection_options_t",
      auto *options = new zvec::CollectionOptions();
      return reinterpret_cast<zvec_collection_options_t *>(options);)
  return nullptr;
}

void zvec_collection_options_destroy(zvec_collection_options_t *options) {
  if (options) {
    delete reinterpret_cast<zvec::CollectionOptions *>(options);
  }
}

zvec_error_code_t zvec_collection_options_set_enable_mmap(
    zvec_collection_options_t *options, bool enable) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::CollectionOptions *>(options);
  ptr->enable_mmap_ = enable;
  return ZVEC_OK;
}

bool zvec_collection_options_get_enable_mmap(
    const zvec_collection_options_t *options) {
  if (!options) {
    return true;  // Default
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionOptions *>(options);
  return ptr->enable_mmap_;
}

zvec_error_code_t zvec_collection_options_set_max_buffer_size(
    zvec_collection_options_t *options, size_t size) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::CollectionOptions *>(options);
  ptr->max_buffer_size_ = static_cast<uint32_t>(size);
  return ZVEC_OK;
}

size_t zvec_collection_options_get_max_buffer_size(
    const zvec_collection_options_t *options) {
  if (!options) {
    return zvec::DEFAULT_MAX_BUFFER_SIZE;  // Default
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionOptions *>(options);
  return ptr->max_buffer_size_;
}

zvec_error_code_t zvec_collection_options_set_read_only(
    zvec_collection_options_t *options, bool read_only) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::CollectionOptions *>(options);
  ptr->read_only_ = read_only;
  return ZVEC_OK;
}

bool zvec_collection_options_get_read_only(
    const zvec_collection_options_t *options) {
  if (!options) {
    return false;  // Default
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionOptions *>(options);
  return ptr->read_only_;
}

// =============================================================================
// CollectionStats functions implementation
// =============================================================================

/**
 * @brief Get document count from collection stats
 * @param stats Collection stats handle
 * @return Document count, or 0 if stats is NULL
 */
uint64_t zvec_collection_stats_get_doc_count(const zvec_collection_stats_t *stats) {
  if (!stats) {
    return 0;
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionStats *>(stats);
  return ptr->doc_count;
}

/**
 * @brief Get index count from collection stats
 * @param stats Collection stats handle
 * @return Number of indexes, or 0 if stats is NULL
 */
size_t zvec_collection_stats_get_index_count(const zvec_collection_stats_t *stats) {
  if (!stats) {
    return 0;
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionStats *>(stats);
  return ptr->index_completeness.size();
}

/**
 * @brief Get index name at specified index
 * @param stats Collection stats handle
 * @param index Index position (0-based)
 * @return Index name C-string, or NULL if invalid
 * @note Returned string is owned by stats, do not free
 */
const char *zvec_collection_stats_get_index_name(
    const zvec_collection_stats_t *stats, size_t index) {
  if (!stats) {
    return nullptr;
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionStats *>(stats);
  if (index >= ptr->index_completeness.size()) {
    return nullptr;
  }
  // Return pointer to string data - caller should not free
  auto it = ptr->index_completeness.begin();
  std::advance(it, index);
  return it->first.c_str();
}

/**
 * @brief Get index completeness at specified index
 * @param stats Collection stats handle
 * @param index Index position (0-based)
 * @return Completeness value (0.0-1.0), or 0.0 if invalid
 */
float zvec_collection_stats_get_index_completeness(
    const zvec_collection_stats_t *stats, size_t index) {
  if (!stats) {
    return 0.0f;
  }
  auto *ptr = reinterpret_cast<const zvec::CollectionStats *>(stats);
  if (index >= ptr->index_completeness.size()) {
    return 0.0f;
  }
  auto it = ptr->index_completeness.begin();
  std::advance(it, index);
  return it->second;
}

// =============================================================================
// IndexParams functions implementation
// =============================================================================

/**
 * @brief Create index parameters of specified type
 * @param index_type Type of index to create
 * @return New index params handle, or NULL on failure
 * @note Caller must call zvec_index_params_destroy() to free
 */
zvec_index_params_t *zvec_index_params_create(zvec_index_type_t index_type) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create zvec_index_params_t",
      // Create appropriate C++ IndexParams based on type with default parameters
      zvec::IndexParams *cpp_params = nullptr;

      switch (index_type) {
        case ZVEC_INDEX_TYPE_INVERT:
          cpp_params =
              new zvec::InvertIndexParams(true,    // enable_range_optimization
                                          false);  // enable_extended_wildcard
          break;
        case ZVEC_INDEX_TYPE_HNSW:
          cpp_params =
              new zvec::HnswIndexParams(
                  zvec::MetricType::L2,  // metric_type
                  zvec::core_interface::kDefaultHnswNeighborCnt,  // m
                  zvec::core_interface::kDefaultHnswEfConstruction,  // ef_construction
                  zvec::QuantizeType::UNDEFINED);
          break;
        case ZVEC_INDEX_TYPE_IVF:
          cpp_params =
              new zvec::IVFIndexParams(zvec::MetricType::L2,  // metric_type
                                       1024,                   // n_list (default)
                                       10,                    // n_iters (default)
                                       false,                // use_soar (default)
                                       zvec::QuantizeType::UNDEFINED);
          break;
        case ZVEC_INDEX_TYPE_FLAT:
        default:
          cpp_params =
              new zvec::FlatIndexParams(zvec::MetricType::L2,  // metric_type
                                        zvec::QuantizeType::UNDEFINED);
          break;
      }

      // Return as opaque pointer (raw pointer)
      return reinterpret_cast<zvec_index_params_t *>(cpp_params);)

  return nullptr;
}

/**
 * @brief Destroy index parameters and free memory
 * @param params Index params to destroy (can be NULL)
 */
void zvec_index_params_destroy(zvec_index_params_t *params) {
  if (params) {
    delete reinterpret_cast<zvec::IndexParams *>(params);
  }
}

/**
 * @brief Set metric type for vector index parameters
 * @param params Index parameters (must be vector index type)
 * @param metric_type Metric type to set
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_set_metric_type(zvec_index_params_t *params,
                                                zvec_metric_type_t metric_type) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Index params pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<zvec::IndexParams *>(params);

  // Set metric type in the underlying C++ IndexParams
  if (cpp_params->is_vector_index_type()) {
    auto *vec_params = dynamic_cast<zvec::VectorIndexParams *>(cpp_params);
    if (vec_params) {
      vec_params->set_metric_type(static_cast<zvec::MetricType>(metric_type));
    }
  }
  return ZVEC_OK;
}

/**
 * @brief Get metric type from index parameters
 * @param params Index parameters
 * @return Metric type, or default (L2) if NULL or not vector index
 */
zvec_metric_type_t zvec_index_params_get_metric_type(
    const zvec_index_params_t *params) {
  if (!params) {
    return ZVEC_METRIC_TYPE_L2;  // Default
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);

  if (cpp_params->is_vector_index_type()) {
    auto *vec_params =
        dynamic_cast<const zvec::VectorIndexParams *>(cpp_params);
    if (vec_params) {
      return static_cast<zvec_metric_type_t>(
          static_cast<uint32_t>(vec_params->metric_type()));
    }
  }
  return ZVEC_METRIC_TYPE_L2;
}

/**
 * @brief Set quantization type for vector index parameters
 * @param params Index parameters (must be vector index type)
 * @param quantize_type Quantization type to set
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_set_quantize_type(
    zvec_index_params_t *params, zvec_quantize_type_t quantize_type) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Index params pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<zvec::IndexParams *>(params);

  // Set quantize type in the underlying C++ IndexParams
  if (cpp_params->is_vector_index_type()) {
    auto *vec_params = dynamic_cast<zvec::VectorIndexParams *>(cpp_params);
    if (vec_params) {
      vec_params->set_quantize_type(
          static_cast<zvec::QuantizeType>(quantize_type));
    }
  }
  return ZVEC_OK;
}

/**
 * @brief Get quantization type from index parameters
 * @param params Index parameters
 * @return Quantization type, or UNDEFINED if NULL or not vector index
 */
zvec_quantize_type_t zvec_index_params_get_quantize_type(
    const zvec_index_params_t *params) {
  if (!params) {
    return ZVEC_QUANTIZE_TYPE_UNDEFINED;
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);

  if (cpp_params->is_vector_index_type()) {
    auto *vec_params =
        dynamic_cast<const zvec::VectorIndexParams *>(cpp_params);
    if (vec_params) {
      return static_cast<zvec_quantize_type_t>(
          static_cast<uint32_t>(vec_params->quantize_type()));
    }
  }
  return ZVEC_QUANTIZE_TYPE_UNDEFINED;
}

/**
 * @brief Get index type from index parameters
 * @param params Index parameters
 * @return Index type, or FLAT as default if NULL
 */
zvec_index_type_t zvec_index_params_get_type(const zvec_index_params_t *params) {
  if (!params) {
    return ZVEC_INDEX_TYPE_FLAT;  // Default
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);
  return static_cast<zvec_index_type_t>(
      static_cast<uint32_t>(cpp_params->type()));
}

/**
 * @brief Set HNSW-specific parameters
 * @param params Index parameters (must be HNSW type)
 * @param m Graph connectivity parameter
 * @param ef_construction Construction exploration factor
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_set_hnsw_params(zvec_index_params_t *params, int m,
                                                int ef_construction) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<zvec::IndexParams *>(params);
  auto *hnsw_params = dynamic_cast<zvec::HnswIndexParams *>(cpp_params);
  if (!hnsw_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  hnsw_params->set_m(m);
  hnsw_params->set_ef_construction(ef_construction);
  return ZVEC_OK;
}

/**
 * @brief Get HNSW m parameter
 * @param params Index parameters (must be HNSW type)
 * @return m parameter value, or 0 on error
 */
int zvec_index_params_get_hnsw_m(const zvec_index_params_t *params) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return 0;
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);
  auto *hnsw_params = dynamic_cast<const zvec::HnswIndexParams *>(cpp_params);
  if (!hnsw_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return 0;
  }
  return hnsw_params->m();
}

/**
 * @brief Get HNSW ef_construction parameter
 * @param params Index parameters (must be HNSW type)
 * @return ef_construction parameter value, or 0 on error
 */
int zvec_index_params_get_hnsw_ef_construction(const zvec_index_params_t *params) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return 0;
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);
  auto *hnsw_params = dynamic_cast<const zvec::HnswIndexParams *>(cpp_params);
  if (!hnsw_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return 0;
  }
  return hnsw_params->ef_construction();
}

/**
 * @brief Set IVF-specific parameters
 * @param params Index parameters (must be IVF type)
 * @param n_list Number of clusters
 * @param n_iters Number of k-means iterations
 * @param use_soar Whether to use SOAR optimization
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_set_ivf_params(zvec_index_params_t *params,
                                               int n_list, int n_iters,
                                               bool use_soar) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not IVF index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<zvec::IndexParams *>(params);
  auto *ivf_params = dynamic_cast<zvec::IVFIndexParams *>(cpp_params);
  if (!ivf_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not IVF index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  ivf_params->set_n_list(n_list);
  ivf_params->set_n_iters(n_iters);
  ivf_params->set_use_soar(use_soar);
  return ZVEC_OK;
}

/**
 * @brief Get IVF parameters
 * @param params Index parameters (must be IVF type)
 * @param out_n_list Output for n_list (can be NULL)
 * @param out_n_iters Output for n_iters (can be NULL)
 * @param out_use_soar Output for use_soar (can be NULL)
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_get_ivf_params(const zvec_index_params_t *params,
                                               int *out_n_list,
                                               int *out_n_iters,
                                               bool *out_use_soar) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not IVF index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);
  auto *ivf_params = dynamic_cast<const zvec::IVFIndexParams *>(cpp_params);
  if (!ivf_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not IVF index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (out_n_list) *out_n_list = ivf_params->n_list();
  if (out_n_iters) *out_n_iters = ivf_params->n_iters();
  if (out_use_soar) *out_use_soar = ivf_params->use_soar();
  return ZVEC_OK;
}

/**
 * @brief Set Invert index parameters
 * @param params Index parameters (must be INVERT type)
 * @param enable_range_opt Enable range optimization
 * @param enable_wildcard Enable wildcard search
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_set_invert_params(zvec_index_params_t *params,
                                                  bool enable_range_opt,
                                                  bool enable_wildcard) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not INVERT index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<zvec::IndexParams *>(params);
  auto *invert_params = dynamic_cast<zvec::InvertIndexParams *>(cpp_params);
  if (!invert_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not INVERT index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  invert_params->set_enable_range_optimization(enable_range_opt);
  invert_params->set_enable_extended_wildcard(enable_wildcard);
  return ZVEC_OK;
}

/**
 * @brief Get Invert index parameters
 * @param params Index parameters (must be INVERT type)
 * @param out_enable_range_opt Output for enable_range_optimization (can be NULL)
 * @param out_enable_wildcard Output for enable_extended_wildcard (can be NULL)
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_index_params_get_invert_params(const zvec_index_params_t *params,
                                                  bool *out_enable_range_opt,
                                                  bool *out_enable_wildcard) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not INVERT index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_params = reinterpret_cast<const zvec::IndexParams *>(params);
  auto *invert_params =
      dynamic_cast<const zvec::InvertIndexParams *>(cpp_params);
  if (!invert_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not INVERT index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (out_enable_range_opt)
    *out_enable_range_opt = invert_params->enable_range_optimization();
  if (out_enable_wildcard)
    *out_enable_wildcard = invert_params->enable_extended_wildcard();
  return ZVEC_OK;
}

// =============================================================================
// FieldSchema management interface implementation
// =============================================================================

zvec_field_schema_t *zvec_field_schema_create(const char *name,
                                          zvec_data_type_t data_type, bool nullable,
                                          uint32_t dimension) {
  if (!name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
    return nullptr;
  }

  ZVEC_TRY_RETURN_NULL(
      "Failed to create field schema",
      auto cpp_schema = new zvec::FieldSchema(
          std::string(name), static_cast<zvec::DataType>(data_type), dimension,
          nullable);

      // Return as opaque pointer (raw pointer)
      return reinterpret_cast<zvec_field_schema_t *>(cpp_schema);)

  return nullptr;
}

void zvec_field_schema_destroy(zvec_field_schema_t *schema) {
  if (schema) {
    delete reinterpret_cast<zvec::FieldSchema *>(schema);
  }
}

const char *zvec_field_schema_get_name(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return nullptr;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->name().c_str();
}

zvec_error_code_t zvec_field_schema_set_name(zvec_field_schema_t *schema,
                                         const char *name) {
  if (!schema || !name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema and name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_schema = reinterpret_cast<zvec::FieldSchema *>(schema);
  cpp_schema->set_name(std::string(name));
  return ZVEC_OK;
}

zvec_data_type_t zvec_field_schema_get_data_type(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_DATA_TYPE_UNDEFINED;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return static_cast<zvec_data_type_t>(
      static_cast<uint32_t>(cpp_schema->data_type()));
}

zvec_error_code_t zvec_field_schema_set_data_type(zvec_field_schema_t *schema,
                                              zvec_data_type_t data_type) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_schema = reinterpret_cast<zvec::FieldSchema *>(schema);
  cpp_schema->set_data_type(static_cast<zvec::DataType>(data_type));
  return ZVEC_OK;
}

zvec_data_type_t zvec_field_schema_get_element_data_type(
    const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_DATA_TYPE_UNDEFINED;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return static_cast<zvec_data_type_t>(
      static_cast<uint32_t>(cpp_schema->element_data_type()));
}

size_t zvec_field_schema_get_element_data_size(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return 0;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->element_data_size();
}

bool zvec_field_schema_is_vector_field(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->is_vector_field();
}

bool zvec_field_schema_is_dense_vector(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->is_dense_vector();
}

bool zvec_field_schema_is_sparse_vector(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->is_sparse_vector();
}

bool zvec_field_schema_is_nullable(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->nullable();
}

zvec_error_code_t zvec_field_schema_set_nullable(zvec_field_schema_t *schema,
                                             bool nullable) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_schema = reinterpret_cast<zvec::FieldSchema *>(schema);
  cpp_schema->set_nullable(nullable);
  return ZVEC_OK;
}

bool zvec_field_schema_has_invert_index(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->has_invert_index();
}

bool zvec_field_schema_is_array_type(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->is_array_type();
}

uint32_t zvec_field_schema_get_dimension(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return 0;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->dimension();
}

zvec_error_code_t zvec_field_schema_set_dimension(zvec_field_schema_t *schema,
                                              uint32_t dimension) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_schema = reinterpret_cast<zvec::FieldSchema *>(schema);
  cpp_schema->set_dimension(dimension);
  return ZVEC_OK;
}

zvec_index_type_t zvec_field_schema_get_index_type(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_INDEX_TYPE_UNDEFINED;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  auto cpp_index_params = cpp_schema->index_params();
  if (!cpp_index_params) {
    return ZVEC_INDEX_TYPE_UNDEFINED;
  }
  return static_cast<zvec_index_type_t>(
      static_cast<uint32_t>(cpp_index_params->type()));
}

const zvec_index_params_t *zvec_field_schema_get_index_params(
    const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return nullptr;
  }
  const auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  auto cpp_index_params = cpp_schema->index_params();
  if (!cpp_index_params) {
    return nullptr;
  }
  // Return internal pointer directly - caller does not own and should not free
  // The pointer is valid as long as the schema is not modified or destroyed
  return reinterpret_cast<const zvec_index_params_t *>(cpp_index_params.get());
}

zvec_error_code_t zvec_field_schema_set_index_params(
    zvec_field_schema_t *schema, const zvec_index_params_t *index_params) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *cpp_schema = reinterpret_cast<zvec::FieldSchema *>(schema);

  if (!index_params) {
    cpp_schema->set_index_params(nullptr);
    return ZVEC_OK;
  }

  auto cpp_index_params = convert_c_index_params_to_cpp(index_params);
  cpp_schema->set_index_params(cpp_index_params);
  return ZVEC_OK;
}

zvec_error_code_t zvec_field_schema_validate(const zvec_field_schema_t *schema,
                                         zvec_string_t **error_msg) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (error_msg) {
    *error_msg = nullptr;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to validate field schema",
      auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
      auto status = cpp_schema->validate(); if (!status.ok()) {
        if (error_msg) {
          *error_msg = zvec_string_create(status.message().c_str());
        }
        return status_to_error_code(status);
      })

  return ZVEC_OK;
}

// Internal helper function (forward declared earlier)
bool zvec_field_schema_has_index(const zvec_field_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::FieldSchema *>(schema);
  return cpp_schema->index_params() != nullptr;
}

// =============================================================================
// CollectionSchema management interface implementation
// =============================================================================

zvec_collection_schema_t *zvec_collection_schema_create(const char *name) {
  if (!name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection name cannot be null");
    return nullptr;
  }

  ZVEC_TRY_RETURN_NULL(
      "Failed to create collection schema",
      auto cpp_schema = new zvec::CollectionSchema(std::string(name));

      // Return as opaque pointer (raw pointer)
      return reinterpret_cast<zvec_collection_schema_t *>(cpp_schema);)

  return nullptr;
}

void zvec_collection_schema_destroy(zvec_collection_schema_t *schema) {
  if (schema) {
    delete reinterpret_cast<zvec::CollectionSchema *>(schema);
  }
}

const char *zvec_collection_schema_get_name(
    const zvec_collection_schema_t *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return nullptr;
  }
  auto *cpp_schema = reinterpret_cast<const zvec::CollectionSchema *>(schema);
  // Use strdup to create a persistent copy since name() returns by value
  return strdup(cpp_schema->name().c_str());
}

zvec_error_code_t zvec_collection_schema_set_name(zvec_collection_schema_t *schema,
                                              const char *name) {
  if (!schema || !name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema or name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to set collection name",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      cpp_schema->set_name(std::string(name)); 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_add_field(zvec_collection_schema_t *schema,
                                               const zvec_field_schema_t *field) {
  if (!schema || !field) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema or field pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to add field",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      const auto *cpp_field =
          reinterpret_cast<const zvec::FieldSchema *>(field);

      // Clone the field schema
      auto cloned_field = std::make_shared<zvec::FieldSchema>(*cpp_field);
      auto status = cpp_schema->add_field(cloned_field);
      return status_to_error_code(status);)
}

zvec_error_code_t zvec_collection_schema_alter_field(
    zvec_collection_schema_t *schema, const char *field_name,
    const zvec_field_schema_t *new_field) {
  if (!schema || !field_name || !new_field) {
    SET_LAST_ERROR(
        ZVEC_ERROR_INVALID_ARGUMENT,
        "Collection schema, field name, or new field cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to alter field",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      auto *cpp_new_field =
          reinterpret_cast<const zvec::FieldSchema *>(new_field);
      auto cloned_field = std::make_shared<zvec::FieldSchema>(*cpp_new_field);
      auto status =
          cpp_schema->alter_field(std::string(field_name), cloned_field);
      return status_to_error_code(status);)
}

zvec_error_code_t zvec_collection_schema_drop_field(zvec_collection_schema_t *schema,
                                                const char *field_name) {
  if (!schema || !field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema or field name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to drop field",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      auto status = cpp_schema->drop_field(std::string(field_name));
      return status_to_error_code(status);)
}

bool zvec_collection_schema_has_field(const zvec_collection_schema_t *schema,
                                      const char *field_name) {
  if (!schema || !field_name) {
    return false;
  }

  auto *cpp_schema = reinterpret_cast<const zvec::CollectionSchema *>(schema);
  return cpp_schema->has_field(std::string(field_name));
}

zvec_field_schema_t *zvec_collection_schema_get_field(
    const zvec_collection_schema_t *schema, const char *field_name) {
  if (!schema || !field_name) {
    return nullptr;
  }

  ZVEC_TRY_RETURN_NULL(
      "Failed to get field",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      const zvec::FieldSchema *cpp_field =
          cpp_schema->get_field(std::string(field_name));
      if (!cpp_field) { return nullptr; }
      // Return non-owning pointer - caller should NOT free this
      return reinterpret_cast<zvec_field_schema_t *>(
          const_cast<zvec::FieldSchema *>(cpp_field));)

  return nullptr;
}

zvec_field_schema_t *zvec_collection_schema_get_forward_field(
    const zvec_collection_schema_t *schema, const char *field_name) {
  if (!schema || !field_name) {
    return nullptr;
  }

  ZVEC_TRY_RETURN_NULL(
      "Failed to get forward field",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      const zvec::FieldSchema *cpp_field =
          cpp_schema->get_forward_field(std::string(field_name));
      if (!cpp_field) { return nullptr; }
      // Return non-owning pointer - caller should NOT free this
      return reinterpret_cast<zvec_field_schema_t *>(
          const_cast<zvec::FieldSchema *>(cpp_field));)

  return nullptr;
}

zvec_field_schema_t *zvec_collection_schema_get_vector_field(
    const zvec_collection_schema_t *schema, const char *field_name) {
  if (!schema || !field_name) {
    return nullptr;
  }

  ZVEC_TRY_RETURN_NULL(
      "Failed to get vector field",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      const zvec::FieldSchema *cpp_field =
          cpp_schema->get_vector_field(std::string(field_name));
      if (!cpp_field) { return nullptr; }
      // Return non-owning pointer - caller should NOT free this
      return reinterpret_cast<zvec_field_schema_t *>(
          const_cast<zvec::FieldSchema *>(cpp_field));)

  return nullptr;
}

zvec_error_code_t zvec_collection_schema_get_forward_fields(
    const zvec_collection_schema_t *schema, zvec_field_schema_t ***fields,
    size_t *count) {
  if (!schema || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, fields, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get forward fields",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto forward_fields = cpp_schema->forward_fields();

      *count = forward_fields.size();
      *fields = (zvec_field_schema_t **)zvec_malloc(*count * sizeof(zvec_field_schema_t *));
      if (!*fields) {
        SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                       "Failed to allocate memory");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      for (size_t i = 0; i < *count; ++i) {
        // Return non-owning pointers - caller should NOT free these
        (*fields)[i] =
            reinterpret_cast<zvec_field_schema_t *>(forward_fields[i].get());
      } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_get_forward_fields_with_index(
    const zvec_collection_schema_t *schema, zvec_field_schema_t ***fields,
    size_t *count) {
  if (!schema || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, fields, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get forward fields with index",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto fields_with_index = cpp_schema->forward_fields_with_index();

      *count = fields_with_index.size();
      *fields = (zvec_field_schema_t **)zvec_malloc(*count * sizeof(zvec_field_schema_t *));
      if (!*fields) {
        SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                       "Failed to allocate memory");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      for (size_t i = 0; i < *count; ++i) {
        // Return non-owning pointers - caller should NOT free these
        (*fields)[i] =
            reinterpret_cast<zvec_field_schema_t *>(fields_with_index[i].get());
      } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_get_forward_field_names(
    const zvec_collection_schema_t *schema, const char ***names, size_t *count) {
  if (!schema || !names || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, names, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get forward field names",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto forward_names = cpp_schema->forward_field_names();

      *count = forward_names.size();
      *names = (const char **)malloc(*count * sizeof(const char *));
      if (!*names) {
        SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                       "Failed to allocate memory");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      // Copy strings - caller owns the memory and should free
      for (size_t i = 0; i < *count; ++i) {
        (*names)[i] = strdup(forward_names[i].c_str());
      } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_get_forward_field_names_with_index(
    const zvec_collection_schema_t *schema, const char ***names, size_t *count) {
  if (!schema || !names || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, names, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get forward field names with index",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto forward_names_with_index =
          cpp_schema->forward_field_names_with_index();

      *count = forward_names_with_index.size();
      *names = (const char **)malloc(*count * sizeof(const char *));
      if (!*names) {
        SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                       "Failed to allocate memory");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      // Copy strings - caller owns the memory and should free
      for (size_t i = 0; i < *count; ++i) {
        (*names)[i] = strdup(forward_names_with_index[i].c_str());
      } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_get_all_field_names(
    const zvec_collection_schema_t *schema, const char ***names, size_t *count) {
  if (!schema || !names || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, names, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get all field names",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto all_names = cpp_schema->all_field_names();

      *count = all_names.size();
      *names = (const char **)malloc(*count * sizeof(const char *));
      if (!*names) {
        SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                       "Failed to allocate memory");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      // Copy strings - caller owns the memory and should free
      for (size_t i = 0; i < *count;
           ++i) { (*names)[i] = strdup(all_names[i].c_str()); 
      } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_get_vector_fields(
    const zvec_collection_schema_t *schema, zvec_field_schema_t ***fields,
    size_t *count) {
  if (!schema || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, fields, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get vector fields",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto vector_fields = cpp_schema->vector_fields();

      *count = vector_fields.size();
      *fields = (zvec_field_schema_t **)malloc(*count * sizeof(zvec_field_schema_t *));
      if (!*fields) {
        SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                       "Failed to allocate memory");
        return ZVEC_ERROR_RESOURCE_EXHAUSTED;
      }

      for (size_t i = 0; i < *count; ++i) {
        // Return non-owning pointers - caller should NOT free these
        (*fields)[i] =
            reinterpret_cast<zvec_field_schema_t *>(vector_fields[i].get());
      } 
      return ZVEC_OK;)
}

uint64_t zvec_collection_schema_get_max_doc_count_per_segment(
    const zvec_collection_schema_t *schema) {
  if (!schema) return 0;
  auto *cpp_schema = reinterpret_cast<const zvec::CollectionSchema *>(schema);
  return cpp_schema->max_doc_count_per_segment();
}

zvec_error_code_t zvec_collection_schema_set_max_doc_count_per_segment(
    zvec_collection_schema_t *schema, uint64_t max_doc_count) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to set max doc count per segment",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      cpp_schema->set_max_doc_count_per_segment(max_doc_count); 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_validate(
    const zvec_collection_schema_t *schema, zvec_string_t **error_msg) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (error_msg) {
    *error_msg = nullptr;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to validate schema",
      auto *cpp_schema =
          reinterpret_cast<const zvec::CollectionSchema *>(schema);
      auto status = cpp_schema->validate(); if (!status.ok()) {
        if (error_msg) {
          *error_msg = zvec_string_create(status.message().c_str());
        }
        return status_to_error_code(status);
      } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_schema_add_index(
    zvec_collection_schema_t *schema, const char *field_name,
    const zvec_index_params_t *index_params) {
  if (!schema || !field_name || !index_params) {
    SET_LAST_ERROR(
        ZVEC_ERROR_INVALID_ARGUMENT,
        "Collection schema, field name, or index params cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to add index",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      auto cpp_index_params = convert_c_index_params_to_cpp(index_params);
      auto status =
          cpp_schema->add_index(std::string(field_name), cpp_index_params);
      return status_to_error_code(status);)
}

zvec_error_code_t zvec_collection_schema_drop_index(zvec_collection_schema_t *schema,
                                                const char *field_name) {
  if (!schema || !field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema or field name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to drop index",
      auto *cpp_schema = reinterpret_cast<zvec::CollectionSchema *>(schema);
      // Find the field and clear its index
      auto *field = cpp_schema->get_field(std::string(field_name));
      if (!field) {
        SET_LAST_ERROR(ZVEC_ERROR_NOT_FOUND, "Field not found");
        return ZVEC_ERROR_NOT_FOUND;
      } 
      const_cast<zvec::FieldSchema *>(field)
          ->set_index_params(nullptr);
      return ZVEC_OK;)
}

bool zvec_collection_schema_has_index(const zvec_collection_schema_t *schema,
                                      const char *field_name) {
  if (!schema || !field_name) {
    return false;
  }

  auto *cpp_schema = reinterpret_cast<const zvec::CollectionSchema *>(schema);
  return cpp_schema->has_index(std::string(field_name));
}

// =============================================================================
// Helper functions
// =============================================================================

const char *zvec_error_code_to_string(zvec_error_code_t error_code) {
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

const char *zvec_data_type_to_string(zvec_data_type_t data_type) {
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

const char *zvec_index_type_to_string(zvec_index_type_t index_type) {
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

const char *zvec_metric_type_to_string(zvec_metric_type_t metric_type) {
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

// =============================================================================
// Doc functions implementation
// =============================================================================

zvec_doc_t *zvec_doc_create(void) {
  ZVEC_TRY_RETURN_NULL("Failed to create document", {
    auto *doc_ptr = new zvec::Doc();
    return reinterpret_cast<zvec_doc_t *>(doc_ptr);
  })
}

void zvec_doc_destroy(zvec_doc_t *doc) {
  if (doc) {
    delete reinterpret_cast<zvec::Doc *>(doc);
  }
}

void zvec_doc_clear(zvec_doc_t *doc) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
  doc_ptr->clear();
  ZVEC_CATCH_END_VOID
}

void zvec_docs_free(zvec_doc_t **docs, size_t count) {
  if (!docs) return;

  for (size_t i = 0; i < count; ++i) {
    zvec_doc_destroy(docs[i]);
  }

  free(docs);
}

void zvec_write_results_free(zvec_write_result_t *results, size_t result_count) {
  free_write_results_internal(results, result_count);
}

void zvec_doc_set_pk(zvec_doc_t *doc, const char *pk) {
  if (!doc || !pk) return;

  ZVEC_TRY_BEGIN_VOID
  auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
  doc_ptr->set_pk(std::string(pk));
  ZVEC_CATCH_END_VOID
}

void zvec_doc_set_doc_id(zvec_doc_t *doc, uint64_t doc_id) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
  doc_ptr->set_doc_id(doc_id);
  ZVEC_CATCH_END_VOID
}

void zvec_doc_set_score(zvec_doc_t *doc, float score) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
  doc_ptr->set_score(score);
  ZVEC_CATCH_END_VOID
}

void zvec_doc_set_operator(zvec_doc_t *doc, zvec_doc_operator_t op) {
  if (!doc) return;

  ZVEC_TRY_BEGIN_VOID
  auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
  doc_ptr->set_operator(static_cast<zvec::Operator>(op));
  ZVEC_CATCH_END_VOID
}

zvec_error_code_t zvec_doc_set_field_null(zvec_doc_t *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR("Failed to set null field",
                        auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
                        doc_ptr->set_null(std::string(field_name));
                        return ZVEC_OK;)
}

// =============================================================================
// Document interface implementation
// =============================================================================

// Helper function to extract scalar values from raw data
template <typename T>
T extract_scalar_value(const void *value, size_t value_size,
                       zvec_error_code_t *error_code) {
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
                                     zvec_error_code_t *error_code) {
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
                                    zvec_error_code_t *error_code) {
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
    const void *value, size_t value_size, zvec_error_code_t *error_code) {
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

// Helper function to extract string array from zvec_string_t** array
std::vector<std::string> extract_string_array_from_zvec(
    zvec_string_t **zvec_strings, size_t count) {
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
    const zvec_doc_t **zvec_docs, size_t doc_count) {
  std::vector<zvec::Doc> docs;
  docs.reserve(doc_count);

  for (size_t i = 0; i < doc_count; ++i) {
    auto *doc_ptr = reinterpret_cast<const zvec::Doc *>(zvec_docs[i]);
    // Use copy constructor to create a deep copy
    docs.push_back(*doc_ptr);
  }

  return docs;
}

static zvec::Status convert_zvec_collection_schema_to_internal(
    const zvec_collection_schema_t *schema,
    zvec::CollectionSchema::Ptr &collection_schema) {
  // Get the underlying C++ CollectionSchema
  auto *cpp_schema = reinterpret_cast<const zvec::CollectionSchema *>(schema);

  // Create a copy of the C++ schema as shared_ptr
  collection_schema = std::make_shared<zvec::CollectionSchema>(*cpp_schema);

  return zvec::Status::OK();
}

zvec_error_code_t zvec_doc_add_field_by_value(zvec_doc_t *doc, const char *field_name,
                                          zvec_data_type_t data_type,
                                          const void *value,
                                          size_t value_size) {
  if (!doc || !field_name || !value) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to add field", auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
      std::string name(field_name); zvec_error_code_t error_code = ZVEC_OK;

      switch (data_type) {
        // Scalar types
        case ZVEC_DATA_TYPE_BINARY:
        case ZVEC_DATA_TYPE_STRING: {
          std::string val(static_cast<const char *>(value), value_size);
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_BOOL: {
          bool val = extract_scalar_value<bool>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for bool type");
            return error_code;
          }
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          int32_t val =
              extract_scalar_value<int32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for int32 type");
            return error_code;
          }
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          int64_t val =
              extract_scalar_value<int64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for int64 type");
            return error_code;
          }
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          uint32_t val =
              extract_scalar_value<uint32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for uint32 type");
            return error_code;
          }
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          uint64_t val =
              extract_scalar_value<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for uint64 type");
            return error_code;
          }
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          float val =
              extract_scalar_value<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for float type");
            return error_code;
          }
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          double val =
              extract_scalar_value<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for double type");
            return error_code;
          }
          doc_ptr->set(name, val);
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
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          auto vec =
              extract_vector_values<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_binary64 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          auto vec =
              extract_vector_values<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp32 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          auto vec = extract_vector_values<zvec::float16_t>(value, value_size,
                                                            &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp16 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          auto vec =
              extract_vector_values<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp64 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          auto vec =
              extract_vector_values<int8_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_int8 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          auto vec =
              extract_vector_values<int16_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_int16 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
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
          doc_ptr->set(name, std::move(vec));
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
          doc_ptr->set(name, std::move(sparse_vec));
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          auto sparse_vec =
              extract_sparse_vector<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid sparse vector data size");
            return error_code;
          }
          doc_ptr->set(name, std::move(sparse_vec));
          break;
        }

        // Array types
        case ZVEC_DATA_TYPE_ARRAY_BINARY: {
          auto binary_array = extract_binary_array(value, value_size);
          doc_ptr->set(name, std::move(binary_array));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_STRING: {
          // Check if this is a zvec_string_t** array or a C-string array
          // zvec_string_t** array has pointer-sized elements
          constexpr size_t ptr_size = sizeof(void *);
          if (value_size % ptr_size == 0) {
            // Likely a zvec_string_t** array
            size_t count = value_size / ptr_size;
            zvec_string_t **zvec_str_array =
                reinterpret_cast<zvec_string_t **>(const_cast<void *>(value));
            auto string_array =
                extract_string_array_from_zvec(zvec_str_array, count);
            doc_ptr->set(name, std::move(string_array));
          } else {
            // C-string array (null-terminated strings)
            auto string_array = extract_string_array(value, value_size);
            doc_ptr->set(name, std::move(string_array));
          }
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          auto vec = extract_array_values<bool>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_bool type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          auto vec =
              extract_array_values<int32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_int32 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          auto vec =
              extract_array_values<int64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_int64 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          auto vec =
              extract_array_values<uint32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_uint32 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          auto vec =
              extract_array_values<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_uint64 type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          auto vec =
              extract_array_values<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_float type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          auto vec =
              extract_array_values<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_double type");
            return error_code;
          }
          doc_ptr->set(name, std::move(vec));
          break;
        }

        default:
          set_last_error("Unsupported data type: " + std::to_string(data_type));
          return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      return ZVEC_OK;)
}

zvec_error_code_t zvec_doc_add_field_by_struct(zvec_doc_t *doc,
                                           const zvec_doc_field_t *field) {
  if (!doc || !field) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to add field", auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);

      std::string name(field->name.data, field->name.length);

      switch (field->data_type) {
        // Scalar types (in zvec_data_type_t order: BINARY, STRING, BOOL, INT32,
        // INT64, UINT32, UINT64, FLOAT, DOUBLE)
        case ZVEC_DATA_TYPE_BINARY: {
          std::string val(
              reinterpret_cast<const char *>(field->value.binary_value.data),
              field->value.binary_value.length);
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_STRING: {
          std::string val(field->value.string_value.data,
                          field->value.string_value.length);
          doc_ptr->set(name, val);
          break;
        }
        case ZVEC_DATA_TYPE_BOOL: {
          doc_ptr->set(name, field->value.bool_value);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          doc_ptr->set(name, field->value.int32_value);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          doc_ptr->set(name, field->value.int64_value);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          doc_ptr->set(name, field->value.uint32_value);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          doc_ptr->set(name, field->value.uint64_value);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          doc_ptr->set(name, field->value.float_value);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          doc_ptr->set(name, field->value.double_value);
          break;
        }

        // Vector types (in zvec_data_type_t order: BINARY32, BINARY64, FP16, FP32,
        // FP64, INT4, INT8, INT16)
        case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
          std::vector<uint32_t> vec(reinterpret_cast<const uint32_t *>(
                                        field->value.vector_value.data),
                                    reinterpret_cast<const uint32_t *>(
                                        field->value.vector_value.data) +
                                        field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          std::vector<uint64_t> vec(reinterpret_cast<const uint64_t *>(
                                        field->value.vector_value.data),
                                    reinterpret_cast<const uint64_t *>(
                                        field->value.vector_value.data) +
                                        field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          std::vector<zvec::float16_t> vec(
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          std::vector<float> vec(field->value.vector_value.data,
                                 field->value.vector_value.data +
                                     field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          std::vector<double> vec(
              reinterpret_cast<const double *>(field->value.vector_value.data),
              reinterpret_cast<const double *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
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
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          std::vector<int8_t> vec(
              reinterpret_cast<const int8_t *>(field->value.vector_value.data),
              reinterpret_cast<const int8_t *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          std::vector<int16_t> vec(
              reinterpret_cast<const int16_t *>(field->value.vector_value.data),
              reinterpret_cast<const int16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }

        // Sparse vector types (in zvec_data_type_t order: FP16, FP32)
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16: {
          std::vector<zvec::float16_t> vec(
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          std::vector<float> vec(field->value.vector_value.data,
                                 field->value.vector_value.data +
                                     field->value.vector_value.length);
          doc_ptr->set(name, std::move(vec));
          break;
        }

        // Array types (in zvec_data_type_t order: BINARY, STRING, BOOL, INT32,
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
          doc_ptr->set(name, std::move(array_values));
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
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          std::vector<bool> array_values(
              reinterpret_cast<const bool *>(field->value.binary_value.data),
              reinterpret_cast<const bool *>(field->value.binary_value.data) +
                  field->value.binary_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          std::vector<int32_t> array_values(
              reinterpret_cast<const int32_t *>(field->value.vector_value.data),
              reinterpret_cast<const int32_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          std::vector<int64_t> array_values(
              reinterpret_cast<const int64_t *>(field->value.vector_value.data),
              reinterpret_cast<const int64_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          std::vector<uint32_t> array_values(
              reinterpret_cast<const uint32_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const uint32_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          std::vector<uint64_t> array_values(
              reinterpret_cast<const uint64_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const uint64_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          std::vector<float> array_values(field->value.vector_value.data,
                                          field->value.vector_value.data +
                                              field->value.vector_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          std::vector<double> array_values(
              reinterpret_cast<const double *>(field->value.vector_value.data),
              reinterpret_cast<const double *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          doc_ptr->set(name, std::move(array_values));
          break;
        }

        default:
          set_last_error("Unsupported data type: " +
                         std::to_string(field->data_type));
          return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      return ZVEC_OK;)
}

const char *zvec_doc_get_pk_pointer(const zvec_doc_t *doc) {
  if (!doc) return nullptr;
  auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
  return doc_ptr->pk_ref().data();
}

const char *zvec_doc_get_pk_copy(const zvec_doc_t *doc) {
  if (!doc) return nullptr;
  auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
  const std::string &pk = doc_ptr->pk_ref();
  if (pk.empty()) return nullptr;

  char *result = static_cast<char *>(malloc(pk.length() + 1));
  strcpy(result, pk.c_str());
  return result;
}

uint64_t zvec_doc_get_doc_id(const zvec_doc_t *doc) {
  if (!doc) return 0;

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document ID", 0,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->doc_id();)
}

float zvec_doc_get_score(const zvec_doc_t *doc) {
  if (!doc) return 0.0f;

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document score", 0.0f,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->score();)
}

zvec_doc_operator_t zvec_doc_get_operator(const zvec_doc_t *doc) {
  if (!doc) return ZVEC_DOC_OP_INSERT;  // default
  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document operator", ZVEC_DOC_OP_INSERT,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      zvec::Operator op = doc_ptr->get_operator();
      return static_cast<zvec_doc_operator_t>(op);)
}

size_t zvec_doc_get_field_count(const zvec_doc_t *doc) {
  if (!doc) return 0;

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get field count", 0,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->field_names().size();)
}

zvec_error_code_t zvec_doc_get_field_value_basic(const zvec_doc_t *doc,
                                             const char *field_name,
                                             zvec_data_type_t field_type,
                                             void *value_buffer,
                                             size_t buffer_size) {
  if (!doc || !field_name || !value_buffer) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field value",
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);

      // Check if field exists
      if (!doc_ptr->has(field_name)) {
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
          const bool val = doc_ptr->get_ref<bool>(field_name);
          *static_cast<bool *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          if (buffer_size < sizeof(int32_t)) {
            set_last_error("Buffer too small for int32 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const int32_t val = doc_ptr->get_ref<int32_t>(field_name);
          *static_cast<int32_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          if (buffer_size < sizeof(int64_t)) {
            set_last_error("Buffer too small for int64 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const int64_t val = doc_ptr->get_ref<int64_t>(field_name);
          *static_cast<int64_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          if (buffer_size < sizeof(uint32_t)) {
            set_last_error("Buffer too small for uint32 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const uint32_t val = doc_ptr->get_ref<uint32_t>(field_name);
          *static_cast<uint32_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          if (buffer_size < sizeof(uint64_t)) {
            set_last_error("Buffer too small for uint64 value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const uint64_t val = doc_ptr->get_ref<uint64_t>(field_name);
          *static_cast<uint64_t *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          if (buffer_size < sizeof(float)) {
            set_last_error("Buffer too small for float value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const float val = doc_ptr->get_ref<float>(field_name);
          *static_cast<float *>(value_buffer) = val;
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          if (buffer_size < sizeof(double)) {
            set_last_error("Buffer too small for double value");
            return ZVEC_ERROR_INVALID_ARGUMENT;
          }
          const double val = doc_ptr->get_ref<double>(field_name);
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

zvec_error_code_t zvec_doc_get_field_value_copy(const zvec_doc_t *doc,
                                            const char *field_name,
                                            zvec_data_type_t field_type,
                                            void **value, size_t *value_size) {
  if (!doc || !field_name || !value || !value_size) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field value copy",
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);

      // Check if field exists
      if (!doc_ptr->has(field_name)) {
        set_last_error("Field not found in document");
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      // Handle copy-returning data types (allocate new memory)
      switch (field_type) {
        // Basic types - copy the actual values
        case ZVEC_DATA_TYPE_BOOL: {
          const bool val = doc_ptr->get_ref<bool>(field_name);
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
          const int32_t val = doc_ptr->get_ref<int32_t>(field_name);
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
          const int64_t val = doc_ptr->get_ref<int64_t>(field_name);
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
          const uint32_t val = doc_ptr->get_ref<uint32_t>(field_name);
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
          const uint64_t val = doc_ptr->get_ref<uint64_t>(field_name);
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
          const float val = doc_ptr->get_ref<float>(field_name);
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
          const double val = doc_ptr->get_ref<double>(field_name);
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
          const std::string &val = doc_ptr->get_ref<std::string>(field_name);
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
              doc_ptr->get_ref<std::vector<uint32_t>>(field_name);
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
              doc_ptr->get_ref<std::vector<uint64_t>>(field_name);
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
              doc_ptr->get_ref<std::vector<zvec::float16_t>>(field_name);
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
              doc_ptr->get_ref<std::vector<float>>(field_name);
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
              doc_ptr->get_ref<std::vector<double>>(field_name);
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
              doc_ptr->get_ref<std::vector<int8_t>>(field_name);
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
              doc_ptr->get_ref<std::vector<int16_t>>(field_name);
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
              doc_ptr->get_ref<SparseVecFP16>(field_name);
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
              doc_ptr->get_ref<SparseVecFP32>(field_name);
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
              doc_ptr->get_ref<BinaryArray>(field_name);
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
              doc_ptr->get_ref<StringArray>(field_name);
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
          const BoolArray &array_vals = doc_ptr->get_ref<BoolArray>(field_name);
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
              doc_ptr->get_ref<Int32Array>(field_name);
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
              doc_ptr->get_ref<Int64Array>(field_name);
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
              doc_ptr->get_ref<UInt32Array>(field_name);
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
              doc_ptr->get_ref<UInt64Array>(field_name);
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
              doc_ptr->get_ref<FloatArray>(field_name);
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
              doc_ptr->get_ref<DoubleArray>(field_name);
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

zvec_error_code_t zvec_doc_get_field_value_pointer(const zvec_doc_t *doc,
                                               const char *field_name,
                                               zvec_data_type_t field_type,
                                               const void **value,
                                               size_t *value_size) {
  if (!doc || !field_name || !value || !value_size) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field value pointer",
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);

      // Check if field exists
      if (!doc_ptr->has(field_name)) {
        set_last_error("Field not found in document");
        return ZVEC_ERROR_INVALID_ARGUMENT;
      }

      // Get field value based on data type
      switch (field_type) {
        case ZVEC_DATA_TYPE_BINARY: {
          const std::string &val = doc_ptr->get_ref<std::string>(field_name);
          *value = val.data();
          *value_size = val.length();
          break;
        }
        case ZVEC_DATA_TYPE_STRING: {
          const std::string &val = doc_ptr->get_ref<std::string>(field_name);
          *value = val.c_str();
          *value_size = val.length();
          break;
        }
        case ZVEC_DATA_TYPE_BOOL: {
          const bool &val = doc_ptr->get_ref<bool>(field_name);
          *value = &val;
          *value_size = sizeof(bool);
          break;
        }
        case ZVEC_DATA_TYPE_INT32: {
          const int32_t &val = doc_ptr->get_ref<int32_t>(field_name);
          *value = &val;
          *value_size = sizeof(int32_t);
          break;
        }
        case ZVEC_DATA_TYPE_INT64: {
          const int64_t &val = doc_ptr->get_ref<int64_t>(field_name);
          *value = &val;
          *value_size = sizeof(int64_t);
          break;
        }
        case ZVEC_DATA_TYPE_UINT32: {
          const uint32_t &val = doc_ptr->get_ref<uint32_t>(field_name);
          *value = &val;
          *value_size = sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_UINT64: {
          const uint64_t &val = doc_ptr->get_ref<uint64_t>(field_name);
          *value = &val;
          *value_size = sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_FLOAT: {
          const float &val = doc_ptr->get_ref<float>(field_name);
          *value = &val;
          *value_size = sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_DOUBLE: {
          const double &val = doc_ptr->get_ref<double>(field_name);
          *value = &val;
          *value_size = sizeof(double);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
          const std::vector<uint32_t> &val =
              doc_ptr->get_ref<std::vector<uint32_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          const std::vector<uint64_t> &val =
              doc_ptr->get_ref<std::vector<uint64_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          // FP16 vectors typically stored as uint16_t
          const std::vector<zvec::float16_t> &val =
              doc_ptr->get_ref<std::vector<zvec::float16_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(zvec::float16_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          const std::vector<float> &val =
              doc_ptr->get_ref<std::vector<float>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          const std::vector<double> &val =
              doc_ptr->get_ref<std::vector<double>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(double);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT4: {
          // INT4 vectors typically stored as int8_t with 2 values per byte
          const std::vector<int8_t> &val =
              doc_ptr->get_ref<std::vector<int8_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(int8_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          const std::vector<int8_t> &val =
              doc_ptr->get_ref<std::vector<int8_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(int8_t);
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          const std::vector<int16_t> &val =
              doc_ptr->get_ref<std::vector<int16_t>>(field_name);
          *value = val.data();
          *value_size = val.size() * sizeof(int16_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          auto &array_vals = doc_ptr->get_ref<std::vector<int32_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(int32_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          auto &array_vals = doc_ptr->get_ref<std::vector<int64_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(int64_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          auto &array_vals =
              doc_ptr->get_ref<std::vector<uint32_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(uint32_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          auto &array_vals =
              doc_ptr->get_ref<std::vector<uint64_t>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(uint64_t);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          auto &array_vals = doc_ptr->get_ref<std::vector<float>>(field_name);
          *value = array_vals.data();
          *value_size = array_vals.size() * sizeof(float);
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          auto &array_vals = doc_ptr->get_ref<std::vector<double>>(field_name);
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

bool zvec_doc_is_empty(const zvec_doc_t *doc) {
  if (!doc) {
    set_last_error("Document pointer is null");
    return true;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check if document is empty", true,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->is_empty();)
}

zvec_error_code_t zvec_doc_remove_field(zvec_doc_t *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR("Failed to remove field",
                        auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
                        doc_ptr->remove(std::string(field_name));
                        return ZVEC_OK;)
}


bool zvec_doc_has_field(const zvec_doc_t *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return false;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check field existence", false,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->has(std::string(field_name));)
}

bool zvec_doc_has_field_value(const zvec_doc_t *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return false;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check field value existence", false,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->has_value(std::string(field_name));)
}

bool zvec_doc_is_field_null(const zvec_doc_t *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Document pointer or field name is null");
    return false;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to check if field is null", false,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->is_null(std::string(field_name));)
}

zvec_error_code_t zvec_doc_get_field_names(const zvec_doc_t *doc, char ***field_names,
                                       size_t *count) {
  if (!doc || !field_names || !count) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get field names",
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      std::vector<std::string> names = doc_ptr->field_names();

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

zvec_error_code_t zvec_doc_serialize(const zvec_doc_t *doc, uint8_t **data,
                                 size_t *size) {
  if (!doc || !data || !size) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to serialize document",
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      std::vector<uint8_t> serialized_data = doc_ptr->serialize();

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

zvec_error_code_t zvec_doc_deserialize(const uint8_t *data, size_t size,
                                   zvec_doc_t **doc) {
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

      // Create a new Doc by copying the deserialized content
      auto *new_doc = new zvec::Doc(*deserialized_doc);
      *doc = reinterpret_cast<zvec_doc_t *>(new_doc); 
      return ZVEC_OK;)
}

void zvec_doc_merge(zvec_doc_t *doc, const zvec_doc_t *other) {
  if (!doc || !other) {
    set_last_error("Document pointers are null");
    return;
  }

  ZVEC_TRY_BEGIN_VOID
  auto *doc_ptr = reinterpret_cast<zvec::Doc *>(doc);
  auto *other_ptr = reinterpret_cast<const zvec::Doc *>(other);
  doc_ptr->merge(*other_ptr);
  ZVEC_CATCH_END_VOID
}

size_t zvec_doc_memory_usage(const zvec_doc_t *doc) {
  if (!doc) {
    set_last_error("Document pointer is null");
    return 0;
  }

  ZVEC_TRY_RETURN_SCALAR(
      "Failed to get document memory usage", 0,
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      return doc_ptr->memory_usage();)
}

zvec_error_code_t zvec_doc_validate(const zvec_doc_t *doc,
                                const zvec_collection_schema_t *schema,
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

      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      status = doc_ptr->validate(schema_ptr, is_update); if (!status.ok()) {
        if (error_msg) {
          *error_msg = copy_string(status.message());
        }
        return status_to_error_code(status);
      }

      if (error_msg) { *error_msg = nullptr; } 
      return ZVEC_OK;)
}

zvec_error_code_t zvec_doc_to_detail_string(const zvec_doc_t *doc, char **detail_str) {
  if (!doc || !detail_str) {
    set_last_error("Invalid arguments");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to get document detail string",
      auto doc_ptr = reinterpret_cast<const zvec::Doc *>(doc);
      std::string detail = doc_ptr->to_detail_string();
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

zvec_error_code_t zvec_collection_create_and_open(
    const char *path, const zvec_collection_schema_t *schema,
    const zvec_collection_options_t *options, zvec_collection_t **collection) {
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
        auto *opts = reinterpret_cast<const zvec::CollectionOptions *>(options);
        collection_options.enable_mmap_ = opts->enable_mmap_;
        collection_options.max_buffer_size_ = opts->max_buffer_size_;
        collection_options.read_only_ = opts->read_only_;
      }

      auto result = zvec::Collection::CreateAndOpen(path, *schema_ptr,
                                                    collection_options);
      zvec_error_code_t error_code = handle_expected_result(result);

      if (error_code == ZVEC_OK) {
        *collection = reinterpret_cast<zvec_collection_t *>(
            new std::shared_ptr<zvec::Collection>(std::move(result.value())));
      }

      return error_code;)
}

zvec_error_code_t zvec_collection_open(const char *path,
                                   const zvec_collection_options_t *options,
                                   zvec_collection_t **collection) {
  if (!path || !collection) {
    set_last_error("Invalid arguments: path and collection cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred", zvec::CollectionOptions collection_options;
      if (options) {
        auto *opts = reinterpret_cast<const zvec::CollectionOptions *>(options);
        collection_options.enable_mmap_ = opts->enable_mmap_;
        collection_options.max_buffer_size_ = opts->max_buffer_size_;
        collection_options.read_only_ = opts->read_only_;
      }

      auto result = zvec::Collection::Open(path, collection_options);
      zvec_error_code_t error_code = handle_expected_result(result);

      if (error_code == ZVEC_OK) {
        *collection = reinterpret_cast<zvec_collection_t *>(
            new std::shared_ptr<zvec::Collection>(std::move(result.value())));
      }

      return error_code;)
}

zvec_error_code_t zvec_collection_close(zvec_collection_t *collection) {
  if (!collection) {
    set_last_error("Invalid argument: collection cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred",
      delete reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_destroy(zvec_collection_t *collection) {
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

zvec_error_code_t zvec_collection_flush(zvec_collection_t *collection) {
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

zvec_error_code_t zvec_collection_get_schema(const zvec_collection_t *collection,
                                         zvec_collection_schema_t **schema) {
  if (!collection || !schema) {
    set_last_error("Invalid arguments: collection and schema cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Exception occurred",
      auto &coll = *reinterpret_cast<const std::shared_ptr<zvec::Collection> *>(
          collection);
      auto result = coll->Schema();

      zvec_error_code_t error_code = handle_expected_result(result);
      if (error_code == ZVEC_OK) {
        const auto &cpp_schema = result.value();

        // Create a copy of the schema and return as raw pointer
        auto *copied_schema = new zvec::CollectionSchema(cpp_schema);
        *schema = reinterpret_cast<zvec_collection_schema_t *>(copied_schema);
      }

      return error_code;)
}

zvec_error_code_t zvec_collection_get_options(const zvec_collection_t *collection,
                                          zvec_collection_options_t **options) {
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

          // Create and initialize options using new
          *options = reinterpret_cast<zvec_collection_options_t *>(
              new zvec::CollectionOptions(result.value()));

      return ZVEC_OK;)
}

zvec_error_code_t zvec_collection_get_stats(const zvec_collection_t *collection,
                                        zvec_collection_stats_t **stats) {
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

      // Create a new CollectionStats object and return as opaque pointer
      *stats = reinterpret_cast<zvec_collection_stats_t *>(
          new zvec::CollectionStats(result.value()));

      return ZVEC_OK;)
}

void zvec_collection_stats_destroy(zvec_collection_stats_t *stats) {
  if (stats) {
    delete reinterpret_cast<zvec::CollectionStats *>(stats);
  }
}

// =============================================================================
// QueryParams implementation
// =============================================================================
// Users should create type-specific query params:
// - HnswQueryParams via zvec_query_params_hnsw_create()
// - IVFQueryParams via zvec_query_params_ivf_create()
// - FlatQueryParams via zvec_query_params_flat_create()
//
// Each type-specific instance has its own destroy function.
// Common parameters (radius, is_linear, is_using_refiner) are set via the
// type-specific create functions.

// =============================================================================
// HnswQueryParams implementation - wrapper around zvec::HnswQueryParams
// =============================================================================

zvec_hnsw_query_params_t *zvec_query_params_hnsw_create(int ef, float radius,
                                                   bool is_linear,
                                                   bool is_using_refiner) {
  ZVEC_TRY_RETURN_NULL("Failed to create HnswQueryParams",
                       auto *params = new zvec::HnswQueryParams(
                           ef, radius, is_linear, is_using_refiner);
                       return reinterpret_cast<zvec_hnsw_query_params_t *>(params);)
  return nullptr;
}

void zvec_query_params_hnsw_destroy(zvec_hnsw_query_params_t *params) {
  if (params) {
    delete reinterpret_cast<zvec::HnswQueryParams *>(params);
  }
}

zvec_error_code_t zvec_query_params_hnsw_set_ef(zvec_hnsw_query_params_t *params,
                                            int ef) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "HNSW query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::HnswQueryParams *>(params);
  ptr->set_ef(ef);
  return ZVEC_OK;
}

int zvec_query_params_hnsw_get_ef(const zvec_hnsw_query_params_t *params) {
  if (!params) return zvec::core_interface::kDefaultHnswEfSearch;
  auto *ptr = reinterpret_cast<const zvec::HnswQueryParams *>(params);
  return ptr->ef();
}

zvec_error_code_t zvec_query_params_hnsw_set_radius(zvec_hnsw_query_params_t *params,
                                                float radius) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "HNSW query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::HnswQueryParams *>(params);
  ptr->set_radius(radius);
  return ZVEC_OK;
}

float zvec_query_params_hnsw_get_radius(const zvec_hnsw_query_params_t *params) {
  if (!params) return 0.0f;
  auto *ptr = reinterpret_cast<const zvec::HnswQueryParams *>(params);
  return ptr->radius();
}

zvec_error_code_t zvec_query_params_hnsw_set_is_linear(zvec_hnsw_query_params_t *params,
                                                   bool is_linear) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "HNSW query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::HnswQueryParams *>(params);
  ptr->set_is_linear(is_linear);
  return ZVEC_OK;
}

bool zvec_query_params_hnsw_get_is_linear(const zvec_hnsw_query_params_t *params) {
  if (!params) return false;
  auto *ptr = reinterpret_cast<const zvec::HnswQueryParams *>(params);
  return ptr->is_linear();
}

zvec_error_code_t zvec_query_params_hnsw_set_is_using_refiner(
    zvec_hnsw_query_params_t *params, bool is_using_refiner) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "HNSW query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::HnswQueryParams *>(params);
  ptr->set_is_using_refiner(is_using_refiner);
  return ZVEC_OK;
}

bool zvec_query_params_hnsw_get_is_using_refiner(
    const zvec_hnsw_query_params_t *params) {
  if (!params) return false;
  auto *ptr = reinterpret_cast<const zvec::HnswQueryParams *>(params);
  return ptr->is_using_refiner();
}

// =============================================================================
// IVFQueryParams implementation - wrapper around zvec::IVFQueryParams
// =============================================================================

zvec_ivf_query_params_t *zvec_query_params_ivf_create(int nprobe,
                                                 bool is_using_refiner,
                                                 float scale_factor) {
  ZVEC_TRY_RETURN_NULL("Failed to create IVFQueryParams",
                       auto *params = new zvec::IVFQueryParams(
                           nprobe, is_using_refiner, scale_factor);
                       return reinterpret_cast<zvec_ivf_query_params_t *>(params);)
  return nullptr;
}

void zvec_query_params_ivf_destroy(zvec_ivf_query_params_t *params) {
  if (params) {
    delete reinterpret_cast<zvec::IVFQueryParams *>(params);
  }
}

zvec_error_code_t zvec_query_params_ivf_set_nprobe(zvec_ivf_query_params_t *params,
                                               int nprobe) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::IVFQueryParams *>(params);
  ptr->set_nprobe(nprobe);
  return ZVEC_OK;
}

int zvec_query_params_ivf_get_nprobe(const zvec_ivf_query_params_t *params) {
  if (!params) return 10;
  auto *ptr = reinterpret_cast<const zvec::IVFQueryParams *>(params);
  return ptr->nprobe();
}

zvec_error_code_t zvec_query_params_ivf_set_scale_factor(zvec_ivf_query_params_t *params,
                                                     float scale_factor) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::IVFQueryParams *>(params);
  ptr->set_scale_factor(scale_factor);
  return ZVEC_OK;
}

float zvec_query_params_ivf_get_scale_factor(const zvec_ivf_query_params_t *params) {
  if (!params) return 10.0f;
  auto *ptr = reinterpret_cast<const zvec::IVFQueryParams *>(params);
  return ptr->scale_factor();
}

zvec_error_code_t zvec_query_params_ivf_set_radius(zvec_ivf_query_params_t *params,
                                               float radius) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::IVFQueryParams *>(params);
  ptr->set_radius(radius);
  return ZVEC_OK;
}

float zvec_query_params_ivf_get_radius(const zvec_ivf_query_params_t *params) {
  if (!params) return 0.0f;
  auto *ptr = reinterpret_cast<const zvec::IVFQueryParams *>(params);
  return ptr->radius();
}

zvec_error_code_t zvec_query_params_ivf_set_is_linear(zvec_ivf_query_params_t *params,
                                                  bool is_linear) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::IVFQueryParams *>(params);
  ptr->set_is_linear(is_linear);
  return ZVEC_OK;
}

bool zvec_query_params_ivf_get_is_linear(const zvec_ivf_query_params_t *params) {
  if (!params) return false;
  auto *ptr = reinterpret_cast<const zvec::IVFQueryParams *>(params);
  return ptr->is_linear();
}

zvec_error_code_t zvec_query_params_ivf_set_is_using_refiner(
    zvec_ivf_query_params_t *params, bool is_using_refiner) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::IVFQueryParams *>(params);
  ptr->set_is_using_refiner(is_using_refiner);
  return ZVEC_OK;
}

bool zvec_query_params_ivf_get_is_using_refiner(
    const zvec_ivf_query_params_t *params) {
  if (!params) return false;
  auto *ptr = reinterpret_cast<const zvec::IVFQueryParams *>(params);
  return ptr->is_using_refiner();
}

// =============================================================================
// FlatQueryParams implementation - wrapper around zvec::FlatQueryParams
// =============================================================================

zvec_flat_query_params_t *zvec_query_params_flat_create(bool is_using_refiner,
                                                   float scale_factor) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create FlatQueryParams",
      auto *params = new zvec::FlatQueryParams(is_using_refiner, scale_factor);
      return reinterpret_cast<zvec_flat_query_params_t *>(params);)
  return nullptr;
}

void zvec_query_params_flat_destroy(zvec_flat_query_params_t *params) {
  if (params) {
    delete reinterpret_cast<zvec::FlatQueryParams *>(params);
  }
}

zvec_error_code_t zvec_query_params_flat_set_scale_factor(
    zvec_flat_query_params_t *params, float scale_factor) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Flat query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::FlatQueryParams *>(params);
  ptr->set_scale_factor(scale_factor);
  return ZVEC_OK;
}

float zvec_query_params_flat_get_scale_factor(
    const zvec_flat_query_params_t *params) {
  if (!params) return 10.0f;
  auto *ptr = reinterpret_cast<const zvec::FlatQueryParams *>(params);
  return ptr->scale_factor();
}

zvec_error_code_t zvec_query_params_flat_set_radius(zvec_flat_query_params_t *params,
                                                float radius) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Flat query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::FlatQueryParams *>(params);
  ptr->set_radius(radius);
  return ZVEC_OK;
}

float zvec_query_params_flat_get_radius(const zvec_flat_query_params_t *params) {
  if (!params) return 0.0f;
  auto *ptr = reinterpret_cast<const zvec::FlatQueryParams *>(params);
  return ptr->radius();
}

zvec_error_code_t zvec_query_params_flat_set_is_linear(zvec_flat_query_params_t *params,
                                                   bool is_linear) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Flat query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::FlatQueryParams *>(params);
  ptr->set_is_linear(is_linear);
  return ZVEC_OK;
}

bool zvec_query_params_flat_get_is_linear(const zvec_flat_query_params_t *params) {
  if (!params) return false;
  auto *ptr = reinterpret_cast<const zvec::FlatQueryParams *>(params);
  return ptr->is_linear();
}

zvec_error_code_t zvec_query_params_flat_set_is_using_refiner(
    zvec_flat_query_params_t *params, bool is_using_refiner) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Flat query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::FlatQueryParams *>(params);
  ptr->set_is_using_refiner(is_using_refiner);
  return ZVEC_OK;
}

bool zvec_query_params_flat_get_is_using_refiner(
    const zvec_flat_query_params_t *params) {
  if (!params) return false;
  auto *ptr = reinterpret_cast<const zvec::FlatQueryParams *>(params);
  return ptr->is_using_refiner();
}

// =============================================================================
// VectorQuery implementation - owns zvec::VectorQuery via raw pointer
// =============================================================================

zvec_vector_query_t *zvec_vector_query_create(void) {
  ZVEC_TRY_RETURN_NULL("Failed to create VectorQuery",
                       auto *query = new zvec::VectorQuery();
                       query->topk_ = 10; query->include_doc_id_ = true;
                       query->include_vector_ = false;
                       return reinterpret_cast<zvec_vector_query_t *>(query);)
  return nullptr;
}

void zvec_vector_query_destroy(zvec_vector_query_t *query) {
  if (query) {
    delete reinterpret_cast<zvec::VectorQuery *>(query);
  }
}

zvec_error_code_t zvec_vector_query_set_topk(zvec_vector_query_t *query, int topk) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  ptr->topk_ = topk;
  return ZVEC_OK;
}

int zvec_vector_query_get_topk(const zvec_vector_query_t *query) {
  if (!query) return 10;
  auto *ptr = reinterpret_cast<const zvec::VectorQuery *>(query);
  return ptr->topk_;
}

zvec_error_code_t zvec_vector_query_set_field_name(zvec_vector_query_t *query,
                                               const char *field_name) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  ptr->field_name_ = field_name ? field_name : "";
  return ZVEC_OK;
}

const char *zvec_vector_query_get_field_name(const zvec_vector_query_t *query) {
  if (!query) return nullptr;
  auto *ptr = reinterpret_cast<const zvec::VectorQuery *>(query);
  return ptr->field_name_.empty() ? nullptr : ptr->field_name_.c_str();
}

zvec_error_code_t zvec_vector_query_set_query_vector(zvec_vector_query_t *query,
                                                 const void *data,
                                                 size_t size) {
  if (!query || !data || size == 0) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Vector query pointer or data is null/empty");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  ptr->query_vector_.assign(static_cast<const char *>(data), size);
  return ZVEC_OK;
}

zvec_error_code_t zvec_vector_query_set_filter(zvec_vector_query_t *query,
                                           const char *filter) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  ptr->filter_ = filter ? filter : "";
  return ZVEC_OK;
}

const char *zvec_vector_query_get_filter(const zvec_vector_query_t *query) {
  if (!query) return nullptr;
  auto *ptr = reinterpret_cast<const zvec::VectorQuery *>(query);
  return ptr->filter_.empty() ? nullptr : ptr->filter_.c_str();
}

zvec_error_code_t zvec_vector_query_set_include_vector(zvec_vector_query_t *query,
                                                   bool include) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  ptr->include_vector_ = include;
  return ZVEC_OK;
}

bool zvec_vector_query_get_include_vector(const zvec_vector_query_t *query) {
  if (!query) return false;
  auto *ptr = reinterpret_cast<const zvec::VectorQuery *>(query);
  return ptr->include_vector_;
}

zvec_error_code_t zvec_vector_query_set_include_doc_id(zvec_vector_query_t *query,
                                                   bool include) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  ptr->include_doc_id_ = include;
  return ZVEC_OK;
}

bool zvec_vector_query_get_include_doc_id(const zvec_vector_query_t *query) {
  if (!query) return false;
  auto *ptr = reinterpret_cast<const zvec::VectorQuery *>(query);
  return ptr->include_doc_id_;
}

zvec_error_code_t zvec_vector_query_set_output_fields(zvec_vector_query_t *query,
                                                  const char **fields,
                                                  size_t count) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  if (!fields || count == 0) {
    ptr->output_fields_ = std::nullopt;
  } else {
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      result.emplace_back(fields[i]);
    }
    ptr->output_fields_ = std::move(result);
  }
  return ZVEC_OK;
}

zvec_error_code_t zvec_vector_query_get_output_fields(const zvec_vector_query_t *query,
                                                  const char ***fields,
                                                  size_t *count) {
  if (!query || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query, fields, or count pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<const zvec::VectorQuery *>(query);

  if (!ptr->output_fields_.has_value()) {
    *fields = nullptr;
    *count = 0;
  } else {
    const auto &output_fields = ptr->output_fields_.value();
    *count = output_fields.size();
    *fields = (const char **)malloc(*count * sizeof(const char *));
    if (!*fields) {
      SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                     "Failed to allocate memory");
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }
    for (size_t i = 0; i < *count; ++i) {
      (*fields)[i] = strdup(output_fields[i].c_str());
    }
  }
  return ZVEC_OK;
}

// =============================================================================
// Type-safe query params attachment functions (transfer ownership to
// VectorQuery)
// =============================================================================

zvec_error_code_t zvec_vector_query_set_query_params(zvec_vector_query_t *query,
                                                 void *params) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::VectorQuery *>(query);

  // Cast to QueryParams* and transfer ownership via shared_ptr
  // The params pointer is now owned by VectorQuery's shared_ptr
  auto *params_ptr = reinterpret_cast<zvec::QueryParams *>(params);
  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

// Type-specific setters for cleaner ownership transfer
zvec_error_code_t zvec_vector_query_set_hnsw_params(
    zvec_vector_query_t *query, zvec_hnsw_query_params_t *hnsw_params) {
  if (!query || !hnsw_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or HNSW params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::HnswQueryParams *>(hnsw_params);

  // Transfer ownership via shared_ptr (polymorphic conversion)
  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

zvec_error_code_t zvec_vector_query_set_ivf_params(zvec_vector_query_t *query,
                                               zvec_ivf_query_params_t *ivf_params) {
  if (!query || !ivf_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or IVF params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::IVFQueryParams *>(ivf_params);

  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

zvec_error_code_t zvec_vector_query_set_flat_params(
    zvec_vector_query_t *query, zvec_flat_query_params_t *flat_params) {
  if (!query || !flat_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or Flat params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::VectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::FlatQueryParams *>(flat_params);

  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

// =============================================================================
// GroupByVectorQuery implementation - owns zvec::GroupByVectorQuery via raw
// pointer
// =============================================================================

zvec_group_by_vector_query_t *zvec_group_by_vector_query_create(void) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create GroupByVectorQuery",
      auto *query = new zvec::GroupByVectorQuery();
      query->group_count_ = 2; query->group_topk_ = 3;
      return reinterpret_cast<zvec_group_by_vector_query_t *>(query);)
  return nullptr;
}

void zvec_group_by_vector_query_destroy(zvec_group_by_vector_query_t *query) {
  if (query) {
    delete reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  }
}

zvec_error_code_t zvec_group_by_vector_query_set_field_name(
    zvec_group_by_vector_query_t *query, const char *field_name) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->field_name_ = field_name ? field_name : "";
  return ZVEC_OK;
}

const char *zvec_group_by_vector_query_get_field_name(
    const zvec_group_by_vector_query_t *query) {
  if (!query) return nullptr;
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);
  return ptr->field_name_.empty() ? nullptr : ptr->field_name_.c_str();
}

zvec_error_code_t zvec_group_by_vector_query_set_group_by_field_name(
    zvec_group_by_vector_query_t *query, const char *field_name) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->group_by_field_name_ = field_name ? field_name : "";
  return ZVEC_OK;
}

const char *zvec_group_by_vector_query_get_group_by_field_name(
    const zvec_group_by_vector_query_t *query) {
  if (!query) return nullptr;
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);
  return ptr->group_by_field_name_.empty() ? nullptr
                                           : ptr->group_by_field_name_.c_str();
}

zvec_error_code_t zvec_group_by_vector_query_set_group_count(
    zvec_group_by_vector_query_t *query, uint32_t count) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->group_count_ = count;
  return ZVEC_OK;
}

uint32_t zvec_group_by_vector_query_get_group_count(
    const zvec_group_by_vector_query_t *query) {
  if (!query) return 2;
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);
  return ptr->group_count_;
}

zvec_error_code_t zvec_group_by_vector_query_set_group_topk(
    zvec_group_by_vector_query_t *query, uint32_t topk) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->group_topk_ = topk;
  return ZVEC_OK;
}

uint32_t zvec_group_by_vector_query_get_group_topk(
    const zvec_group_by_vector_query_t *query) {
  if (!query) return 3;
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);
  return ptr->group_topk_;
}

zvec_error_code_t zvec_group_by_vector_query_set_query_vector(
    zvec_group_by_vector_query_t *query, const void *data, size_t size) {
  if (!query || !data || size == 0) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer or data is null/empty");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->query_vector_.assign(static_cast<const char *>(data), size);
  return ZVEC_OK;
}

zvec_error_code_t zvec_group_by_vector_query_set_filter(
    zvec_group_by_vector_query_t *query, const char *filter) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->filter_ = filter ? filter : "";
  return ZVEC_OK;
}

const char *zvec_group_by_vector_query_get_filter(
    const zvec_group_by_vector_query_t *query) {
  if (!query) return nullptr;
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);
  return ptr->filter_.empty() ? nullptr : ptr->filter_.c_str();
}

zvec_error_code_t zvec_group_by_vector_query_set_include_vector(
    zvec_group_by_vector_query_t *query, bool include) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  ptr->include_vector_ = include;
  return ZVEC_OK;
}

bool zvec_group_by_vector_query_get_include_vector(
    const zvec_group_by_vector_query_t *query) {
  if (!query) return false;
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);
  return ptr->include_vector_;
}

zvec_error_code_t zvec_group_by_vector_query_set_output_fields(
    zvec_group_by_vector_query_t *query, const char **fields, size_t count) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "GroupByVectorQuery pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  if (!fields || count == 0) {
    ptr->output_fields_ = std::nullopt;
  } else {
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      result.emplace_back(fields[i]);
    }
    ptr->output_fields_ = std::move(result);
  }
  return ZVEC_OK;
}

zvec_error_code_t zvec_group_by_vector_query_get_output_fields(
    zvec_group_by_vector_query_t *query, const char ***fields, size_t *count) {
  if (!query || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query, fields, or count pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  auto *ptr = reinterpret_cast<const zvec::GroupByVectorQuery *>(query);

  if (!ptr->output_fields_.has_value()) {
    *fields = nullptr;
    *count = 0;
  } else {
    const auto &output_fields = ptr->output_fields_.value();
    *count = output_fields.size();
    *fields = (const char **)malloc(*count * sizeof(const char *));
    if (!*fields) {
      SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                     "Failed to allocate memory");
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }
    for (size_t i = 0; i < *count; ++i) {
      (*fields)[i] = strdup(output_fields[i].c_str());
    }
  }
  return ZVEC_OK;
}

zvec_error_code_t zvec_group_by_vector_query_set_query_params(
    zvec_group_by_vector_query_t *query, void *params) {
  if (!query || !params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::QueryParams *>(params);

  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

// Type-specific setters for GroupByVectorQuery
zvec_error_code_t zvec_group_by_vector_query_set_hnsw_params(
    zvec_group_by_vector_query_t *query, zvec_hnsw_query_params_t *hnsw_params) {
  if (!query || !hnsw_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or HNSW params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::HnswQueryParams *>(hnsw_params);

  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

zvec_error_code_t zvec_group_by_vector_query_set_ivf_params(
    zvec_group_by_vector_query_t *query, zvec_ivf_query_params_t *ivf_params) {
  if (!query || !ivf_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or IVF params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::IVFQueryParams *>(ivf_params);

  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

zvec_error_code_t zvec_group_by_vector_query_set_flat_params(
    zvec_group_by_vector_query_t *query, zvec_flat_query_params_t *flat_params) {
  if (!query || !flat_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Query or Flat params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  auto *query_ptr = reinterpret_cast<zvec::GroupByVectorQuery *>(query);
  auto *params_ptr = reinterpret_cast<zvec::FlatQueryParams *>(flat_params);

  query_ptr->query_params_.reset(params_ptr);

  return ZVEC_OK;
}

// =============================================================================
// Index Interface Implementation
// =============================================================================

/**
 * @brief Create index on a collection column
 * @param collection Collection handle
 * @param column_name Column name to create index on
 * @param index_params Index parameters
 * @return ZVEC_OK on success, error code on failure
 * @note index_params is cloned internally, caller should still call
 *       zvec_index_params_destroy() to free the original
 */
zvec_error_code_t zvec_collection_create_index(
    zvec_collection_t *collection, const char *column_name,
    const zvec_index_params_t *index_params) {
  if (!collection || !column_name || !index_params) {
    set_last_error(
        "Invalid arguments: collection, column_name, and index_params cannot "
        "be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
    "Exception in zvec_collection_create_index",
    auto coll_ptr =
        reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);
    std::string field_name_str(column_name);

    auto *cpp_params =
        reinterpret_cast<const zvec::IndexParams *>(index_params);
    auto index_params_ptr = cpp_params->clone();
    auto status = (*coll_ptr)->CreateIndex(field_name_str, index_params_ptr);
    return status_to_error_code(status);)
}

/**
 * @brief Drop index from a collection column
 * @param collection Collection handle
 * @param column_name Column name to drop index from
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_collection_drop_index(zvec_collection_t *collection,
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

/**
 * @brief Optimize collection (rebuild indexes, merge segments)
 * @param collection Collection handle
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_collection_optimize(zvec_collection_t *collection) {
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

/**
 * @brief Add a column to collection
 * @param collection Collection handle
 * @param field_schema Field schema (deep-copied, caller retains ownership)
 * @param expression Default value expression (can be NULL for no default)
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_collection_add_column(zvec_collection_t *collection,
                                          const zvec_field_schema_t *field_schema,
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

      // Deep copy the schema - caller retains ownership
      auto *cpp_schema =
          reinterpret_cast<const zvec::FieldSchema *>(field_schema);
      zvec::FieldSchema::Ptr schema =
          std::make_shared<zvec::FieldSchema>(*cpp_schema);

      std::string expr = expression ? expression : "";
      zvec::Status status = (*coll_ptr)->AddColumn(schema, expr);

      if (!status.ok()) { set_last_error(status.message()); }

      return status_to_error_code(status);)
}

/**
 * @brief Drop a column from collection
 * @param collection Collection handle
 * @param column_name Column name to drop
 * @return ZVEC_OK on success, error code on failure
 */
zvec_error_code_t zvec_collection_drop_column(zvec_collection_t *collection,
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

zvec_error_code_t zvec_collection_alter_column(
    zvec_collection_t *collection, const char *column_name, const char *new_name,
    const zvec_field_schema_t *new_schema) {
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

      // Deep copy the schema - caller retains ownership and must call
      // zvec_field_schema_destroy after the call
      zvec::FieldSchema::Ptr schema = nullptr;
      if (new_schema) {
        auto *cpp_schema =
            reinterpret_cast<const zvec::FieldSchema *>(new_schema);
        // Use copy constructor to create a deep copy
        schema = std::make_shared<zvec::FieldSchema>(*cpp_schema);
      }

      zvec::Status status =
          (*coll_ptr)->AlterColumn(column_name, rename, schema);
      if (!status.ok()) { set_last_error(status.message()); }

      return status_to_error_code(status);)
}

// =============================================================================
// DML Interface Implementation
// =============================================================================

zvec_error_code_t zvec_collection_insert(zvec_collection_t *collection,
                                      const zvec_doc_t **docs, size_t doc_count,
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
      zvec_error_code_t error_code = handle_expected_result(result);

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

zvec_error_code_t zvec_collection_insert_with_results(zvec_collection_t *collection,
                                                  const zvec_doc_t **docs,
                                                  size_t doc_count,
                                                  zvec_write_result_t **results,
                                                  size_t *result_count) {
  if (!collection || !docs || doc_count == 0 || !results || !result_count) {
    set_last_error(
        "Invalid arguments: collection, docs, doc_count, results and "
        "result_count cannot be null/zero");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *results = nullptr;
  *result_count = 0;

  ZVEC_TRY_RETURN_ERROR(
      "Exception in zvec_collection_insert_with_results",
      auto coll_ptr =
          reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

      std::vector<zvec::Doc> internal_docs =
          convert_zvec_docs_to_internal(docs, doc_count);
      std::vector<std::string> pks = collect_doc_pks(docs, doc_count);

      auto result = (*coll_ptr)->Insert(internal_docs);
      zvec_error_code_t error_code = handle_expected_result(result);

      if (error_code != ZVEC_OK) { return error_code; }

      return build_write_results(result.value(), results, result_count);)
}

zvec_error_code_t zvec_collection_update(zvec_collection_t *collection,
                                      const zvec_doc_t **docs, size_t doc_count,
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
      zvec_error_code_t error_code = handle_expected_result(result);

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

zvec_error_code_t zvec_collection_update_with_results(zvec_collection_t *collection,
                                                  const zvec_doc_t **docs,
                                                  size_t doc_count,
                                                  zvec_write_result_t **results,
                                                  size_t *result_count) {
  if (!collection || !docs || doc_count == 0 || !results || !result_count) {
    set_last_error(
        "Invalid arguments: collection, docs, doc_count, results and "
        "result_count cannot be null/zero");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *results = nullptr;
  *result_count = 0;

  ZVEC_TRY_RETURN_ERROR(
      "Exception in zvec_collection_update_with_results",
      auto coll_ptr =
          reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

      std::vector<zvec::Doc> internal_docs =
          convert_zvec_docs_to_internal(docs, doc_count);
      std::vector<std::string> pks = collect_doc_pks(docs, doc_count);

      auto result = (*coll_ptr)->Update(internal_docs);
      zvec_error_code_t error_code = handle_expected_result(result);

      if (error_code != ZVEC_OK) { return error_code; }

      return build_write_results(result.value(), results, result_count);)
}

zvec_error_code_t zvec_collection_upsert(zvec_collection_t *collection,
                                      const zvec_doc_t **docs, size_t doc_count,
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
      zvec_error_code_t error_code = handle_expected_result(result);

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

zvec_error_code_t zvec_collection_upsert_with_results(zvec_collection_t *collection,
                                                  const zvec_doc_t **docs,
                                                  size_t doc_count,
                                                  zvec_write_result_t **results,
                                                  size_t *result_count) {
  if (!collection || !docs || doc_count == 0 || !results || !result_count) {
    set_last_error(
        "Invalid arguments: collection, docs, doc_count, results and "
        "result_count cannot be null/zero");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *results = nullptr;
  *result_count = 0;

  ZVEC_TRY_RETURN_ERROR(
      "Exception in zvec_collection_upsert_with_results",
      auto coll_ptr =
          reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

      std::vector<zvec::Doc> internal_docs =
          convert_zvec_docs_to_internal(docs, doc_count);
      std::vector<std::string> pks = collect_doc_pks(docs, doc_count);

      auto result = (*coll_ptr)->Upsert(internal_docs);
      zvec_error_code_t error_code = handle_expected_result(result);

      if (error_code != ZVEC_OK) { return error_code; }

      return build_write_results(result.value(), results, result_count);)
}

zvec_error_code_t zvec_collection_delete(zvec_collection_t *collection,
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
      zvec_error_code_t error_code = handle_expected_result(result);

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

zvec_error_code_t zvec_collection_delete_with_results(zvec_collection_t *collection,
                                                  const char *const *pks,
                                                  size_t pk_count,
                                                  zvec_write_result_t **results,
                                                  size_t *result_count) {
  if (!collection || !pks || pk_count == 0 || !results || !result_count) {
    set_last_error(
        "Invalid arguments: collection, pks, pk_count, results and "
        "result_count cannot be null/zero");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *results = nullptr;
  *result_count = 0;

  ZVEC_TRY_RETURN_ERROR(
      "Exception in zvec_collection_delete_with_results",
      auto coll_ptr =
          reinterpret_cast<std::shared_ptr<zvec::Collection> *>(collection);

      std::vector<std::string> primary_keys; primary_keys.reserve(pk_count);
      for (size_t i = 0; i < pk_count; ++i) {
        if (pks[i]) {
          primary_keys.emplace_back(pks[i]);
        } else {
          primary_keys.emplace_back("");
        }
      }

      auto result = (*coll_ptr)->Delete(primary_keys);
      zvec_error_code_t error_code = handle_expected_result(result);

      if (error_code != ZVEC_OK) { return error_code; }

      return build_write_results(result.value(), results,
                                  result_count);)
}

zvec_error_code_t zvec_collection_delete_by_filter(zvec_collection_t *collection,
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
      } 
      return ZVEC_OK;)
}

// =============================================================================
// Data query interface implementation
// =============================================================================

// Helper function to convert document results to C API format
zvec_error_code_t convert_document_results(
    const std::vector<std::shared_ptr<zvec::Doc>> &query_results,
    zvec_doc_t ***results, size_t *result_count) {
  *result_count = query_results.size();
  *results =
      static_cast<zvec_doc_t **>(malloc(*result_count * sizeof(zvec_doc_t *)));

  if (!*results) {
    set_last_error("Failed to allocate memory for query results");
    return ZVEC_ERROR_INTERNAL_ERROR;
  }

  for (size_t i = 0; i < *result_count; ++i) {
    const auto &internal_doc = query_results[i];
    // Create new document wrapper
    zvec_doc_t *c_doc = zvec_doc_create();
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
    auto *doc_ptr = reinterpret_cast<zvec::Doc *>(c_doc);
    *doc_ptr = *internal_doc;  // Copy assignment
    (*results)[i] = c_doc;     // Store the pointer, not dereference
  }

  return ZVEC_OK;
}

// Helper function to convert fetched document results to C API format
static void normalize_nullable_fields_for_fetch(
    const zvec::CollectionSchema &schema, zvec::DocPtrMap &doc_map) {
  std::vector<std::string> nullable_fields;
  nullable_fields.reserve(schema.fields().size());

  for (const auto &field : schema.fields()) {
    if (field && field->nullable()) {
      nullable_fields.push_back(field->name());
    }
  }

  if (nullable_fields.empty()) {
    return;
  }

  for (auto &[_, doc_ptr] : doc_map) {
    if (!doc_ptr) {
      continue;
    }

    for (const auto &field_name : nullable_fields) {
      if (!doc_ptr->has(field_name)) {
        doc_ptr->set_null(field_name);
      }
    }
  }
}

zvec_error_code_t convert_fetched_document_results(const zvec::DocPtrMap &doc_map,
                                                zvec_doc_t ***results,
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

  *results = static_cast<zvec_doc_t **>(malloc(*doc_count * sizeof(zvec_doc_t *)));
  if (!*results) {
    set_last_error("Failed to allocate memory for document pointers");
    return ZVEC_ERROR_INTERNAL_ERROR;
  }

  // Convert C++ DocPtrMap to C zvec_doc_t pointer array
  size_t index = 0;
  for (const auto &[pk, doc_ptr] : doc_map) {
    if (doc_ptr && index < *doc_count) {
      // Create new document wrapper
      zvec_doc_t *c_doc = zvec_doc_create();
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

      // Copy the C++ document to our wrapper using copy assignment
      auto *cpp_doc_ptr = reinterpret_cast<zvec::Doc *>(c_doc);
      *cpp_doc_ptr = *doc_ptr;  // Copy assignment from shared_ptr

      // Set the primary key explicitly
      zvec_doc_set_pk(c_doc, pk.c_str());

      (*results)[index] = c_doc;
      ++index;
    }
  }

  return ZVEC_OK;
}

zvec_error_code_t zvec_collection_query(const zvec_collection_t *collection,
                                    const zvec_vector_query_t *query,
                                    zvec_doc_t ***results,
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

      // Cast zvec_vector_query_t* to zvec::VectorQuery* directly
      auto *internal_query =
          reinterpret_cast<const zvec::VectorQuery *>(query);

      auto result = (*coll_ptr)->Query(*internal_query);
      zvec_error_code_t error_code = handle_expected_result(result);

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

zvec_error_code_t zvec_collection_fetch(zvec_collection_t *collection,
                                    const char *const *pks, size_t pk_count,
                                    zvec_doc_t ***results, size_t *doc_count) {
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

      auto doc_map = result.value();
      auto schema_result = (*coll_ptr)->Schema();
      if (schema_result.has_value()) {
        normalize_nullable_fields_for_fetch(schema_result.value(), doc_map);
      } 
      return convert_fetched_document_results(doc_map, results, doc_count);)
}
