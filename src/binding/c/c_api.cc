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

// For functions returning ZVecErrorCode - complete try-catch wrapper
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
    { __VA_ARGS__ }                                     \
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
    { __VA_ARGS__ }                                     \
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
    { __VA_ARGS__ }                                     \
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

ZVecString *zvec_string_create(const char *str) {
  if (!str) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointer cannot be null");
    return nullptr;
  }

  size_t len = strlen(str);
  ZVecString *zstr = static_cast<ZVecString *>(malloc(sizeof(ZVecString)));
  if (!zstr) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecString");
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

ZVecString *zvec_string_create_from_view(const ZVecStringView *view) {
  if (!view || !view->data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String view or data cannot be null");
    return nullptr;
  }

  ZVecString *zstr = static_cast<ZVecString *>(malloc(sizeof(ZVecString)));
  if (!zstr) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecString");
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

ZVecString *zvec_bin_create(const uint8_t *data, size_t length) {
  if (!data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Binary data pointer cannot be null");
    return nullptr;
  }

  ZVecString *zstr = static_cast<ZVecString *>(malloc(sizeof(ZVecString)));
  if (!zstr) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecString");
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

ZVecString *zvec_string_copy(const ZVecString *str) {
  if (!str || !str->data) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Source string or data cannot be null");
    return nullptr;
  }

  return zvec_string_create(str->data);
}

const char *zvec_string_c_str(const ZVecString *str) {
  if (!str) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointer cannot be null");
    return nullptr;
  }

  return str->data;
}

size_t zvec_string_length(const ZVecString *str) {
  if (!str) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "String pointer cannot be null");
    return 0;
  }

  return str->length;
}

int zvec_string_compare(const ZVecString *str1, const ZVecString *str2) {
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

// Internal structure - Console log configuration
struct ZVecConsoleLogConfig {
  ZVecLogLevel level;
};

// Internal structure - File log configuration
struct ZVecFileLogConfig {
  ZVecLogLevel level;
  ZVecString *dir;
  ZVecString *basename;
  uint32_t file_size;
  uint32_t overdue_days;
};

// Internal structure - Configuration data
struct ZVecConfigData {
  uint64_t memory_limit_bytes;

  // log
  ZVecLogType log_type;
  void *log_config;  // ZVecConsoleLogConfig* or ZVecFileLogConfig*

  // query
  uint32_t query_thread_count;
  float invert_to_forward_scan_ratio;
  float brute_force_by_keys_ratio;

  // optimize
  uint32_t optimize_thread_count;
};

ZVecConsoleLogConfig *zvec_config_console_log_create(ZVecLogLevel level) {
  ZVecConsoleLogConfig *config =
      static_cast<ZVecConsoleLogConfig *>(malloc(sizeof(ZVecConsoleLogConfig)));
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecConsoleLogConfig");
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
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Directory or basename cannot be null");
    return nullptr;
  }

  ZVecFileLogConfig *config =
      static_cast<ZVecFileLogConfig *>(malloc(sizeof(ZVecFileLogConfig)));
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecFileLogConfig");
    return nullptr;
  }

  config->level = level;
  config->dir = zvec_string_create(dir);
  config->basename = zvec_string_create(basename);

  if (!config->dir || !config->basename) {
    if (config->dir) zvec_free_string(config->dir);
    if (config->basename) zvec_free_string(config->basename);
    free(config);
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to create strings for file log config");
    return nullptr;
  }

  config->file_size = file_size;
  config->overdue_days = overdue_days;

  return config;
}

void zvec_config_console_log_destroy(ZVecConsoleLogConfig *config) {
  free(const_cast<ZVecConsoleLogConfig *>(config));
}

void zvec_config_file_log_destroy(ZVecFileLogConfig *config) {
  if (config) {
    if (config->dir) zvec_free_string(config->dir);
    if (config->basename) zvec_free_string(config->basename);
    free(const_cast<ZVecFileLogConfig *>(config));
  }
}

ZVecLogLevel zvec_config_console_log_get_level(
    const ZVecConsoleLogConfig *config) {
  if (!config) {
    return ZVEC_LOG_LEVEL_WARN;
  }
  return config->level;
}

ZVecErrorCode zvec_config_console_log_set_level(ZVecConsoleLogConfig *config,
                                                ZVecLogLevel level) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->level = level;
  return ZVEC_OK;
}

ZVecLogLevel zvec_config_file_log_get_level(const ZVecFileLogConfig *config) {
  if (!config) {
    return ZVEC_LOG_LEVEL_WARN;
  }
  return config->level;
}

ZVecErrorCode zvec_config_file_log_set_level(ZVecFileLogConfig *config,
                                             ZVecLogLevel level) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->level = level;
  return ZVEC_OK;
}

const char *zvec_config_file_log_get_dir(const ZVecFileLogConfig *config) {
  if (!config || !config->dir) {
    return nullptr;
  }
  return config->dir->data;
}

ZVecErrorCode zvec_config_file_log_set_dir(ZVecFileLogConfig *config,
                                           const char *dir) {
  if (!config || !dir) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Config or dir pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (config->dir) {
    zvec_free_string(config->dir);
  }
  config->dir = zvec_string_create(dir);
  if (!config->dir) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to create dir string");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }
  return ZVEC_OK;
}

const char *zvec_config_file_log_get_basename(const ZVecFileLogConfig *config) {
  if (!config || !config->basename) {
    return nullptr;
  }
  return config->basename->data;
}

ZVecErrorCode zvec_config_file_log_set_basename(ZVecFileLogConfig *config,
                                                const char *basename) {
  if (!config || !basename) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Config or basename pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (config->basename) {
    zvec_free_string(config->basename);
  }
  config->basename = zvec_string_create(basename);
  if (!config->basename) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to create basename string");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }
  return ZVEC_OK;
}

uint32_t zvec_config_file_log_get_file_size(const ZVecFileLogConfig *config) {
  if (!config) {
    return 0;
  }
  return config->file_size;
}

ZVecErrorCode zvec_config_file_log_set_file_size(ZVecFileLogConfig *config,
                                                 uint32_t file_size) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->file_size = file_size;
  return ZVEC_OK;
}

uint32_t zvec_config_file_log_get_overdue_days(
    const ZVecFileLogConfig *config) {
  if (!config) {
    return 0;
  }
  return config->overdue_days;
}

ZVecErrorCode zvec_config_file_log_set_overdue_days(ZVecFileLogConfig *config,
                                                    uint32_t days) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->overdue_days = days;
  return ZVEC_OK;
}

ZVecConfigData *zvec_config_data_create(void) {
  ZVecConfigData *config =
      static_cast<ZVecConfigData *>(malloc(sizeof(ZVecConfigData)));
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecConfigData");
    return nullptr;
  }

  // Create default console log config
  ZVecConsoleLogConfig *log_config =
      zvec_config_console_log_create(ZVEC_LOG_LEVEL_WARN);
  if (!log_config) {
    free(config);
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to create console log config");
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

void zvec_config_data_destroy(ZVecConfigData *config) {
  if (config) {
    if (config->log_config) {
      if (config->log_type == ZVEC_LOG_TYPE_CONSOLE) {
        zvec_config_console_log_destroy(
            static_cast<ZVecConsoleLogConfig *>(config->log_config));
      } else {
        zvec_config_file_log_destroy(
            static_cast<ZVecFileLogConfig *>(config->log_config));
      }
    }
    free(config);
  }
}

ZVecErrorCode zvec_config_data_set_memory_limit(ZVecConfigData *config,
                                                uint64_t memory_limit_bytes) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->memory_limit_bytes = memory_limit_bytes;
  return ZVEC_OK;
}

uint64_t zvec_config_data_get_memory_limit(const ZVecConfigData *config) {
  if (!config) {
    return 0;
  }
  return config->memory_limit_bytes;
}

ZVecErrorCode zvec_config_data_set_log_config(ZVecConfigData *config,
                                              ZVecLogType log_type,
                                              void *log_config) {
  if (!config || !log_config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Config or log_config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (config->log_config) {
    if (config->log_type == ZVEC_LOG_TYPE_CONSOLE) {
      zvec_config_console_log_destroy(
          static_cast<ZVecConsoleLogConfig *>(config->log_config));
    } else {
      zvec_config_file_log_destroy(
          static_cast<ZVecFileLogConfig *>(config->log_config));
    }
  }

  config->log_type = log_type;
  config->log_config = log_config;
  return ZVEC_OK;
}

ZVecLogType zvec_config_data_get_log_type(const ZVecConfigData *config) {
  if (!config) {
    return ZVEC_LOG_TYPE_CONSOLE;
  }
  return config->log_type;
}

ZVecConsoleLogConfig *zvec_config_data_get_console_log_config(
    const ZVecConfigData *config) {
  if (!config || config->log_type != ZVEC_LOG_TYPE_CONSOLE) {
    return nullptr;
  }
  return static_cast<ZVecConsoleLogConfig *>(config->log_config);
}

ZVecFileLogConfig *zvec_config_data_get_file_log_config(
    const ZVecConfigData *config) {
  if (!config || config->log_type != ZVEC_LOG_TYPE_FILE) {
    return nullptr;
  }
  return static_cast<ZVecFileLogConfig *>(config->log_config);
}

ZVecErrorCode zvec_config_data_set_query_thread_count(ZVecConfigData *config,
                                                      uint32_t thread_count) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->query_thread_count = thread_count;
  return ZVEC_OK;
}

uint32_t zvec_config_data_get_query_thread_count(const ZVecConfigData *config) {
  if (!config) {
    return 1;
  }
  return config->query_thread_count;
}

ZVecErrorCode zvec_config_data_set_invert_to_forward_scan_ratio(
    ZVecConfigData *config, float ratio) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->invert_to_forward_scan_ratio = ratio;
  return ZVEC_OK;
}

float zvec_config_data_get_invert_to_forward_scan_ratio(
    const ZVecConfigData *config) {
  if (!config) {
    return 0.0f;
  }
  return config->invert_to_forward_scan_ratio;
}

ZVecErrorCode zvec_config_data_set_brute_force_by_keys_ratio(
    ZVecConfigData *config, float ratio) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->brute_force_by_keys_ratio = ratio;
  return ZVEC_OK;
}

float zvec_config_data_get_brute_force_by_keys_ratio(
    const ZVecConfigData *config) {
  if (!config) {
    return 0.0f;
  }
  return config->brute_force_by_keys_ratio;
}

ZVecErrorCode zvec_config_data_set_optimize_thread_count(
    ZVecConfigData *config, uint32_t thread_count) {
  if (!config) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Config pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  config->optimize_thread_count = thread_count;
  return ZVEC_OK;
}

uint32_t zvec_config_data_get_optimize_thread_count(
    const ZVecConfigData *config) {
  if (!config) {
    return 1;
  }
  return config->optimize_thread_count;
}


// =============================================================================
// Initialization and cleanup interface implementation
// =============================================================================

ZVecErrorCode zvec_initialize(const ZVecConfigData *config) {
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
        cpp_config.memory_limit_bytes =
            zvec_config_data_get_memory_limit(config);
        cpp_config.query_thread_count =
            zvec_config_data_get_query_thread_count(config);
        cpp_config.invert_to_forward_scan_ratio =
            zvec_config_data_get_invert_to_forward_scan_ratio(config);
        cpp_config.brute_force_by_keys_ratio =
            zvec_config_data_get_brute_force_by_keys_ratio(config);
        cpp_config.optimize_thread_count =
            zvec_config_data_get_optimize_thread_count(config);

        // Set log configuration
        void *log_config = zvec_config_data_get_console_log_config(config);
        if (!log_config) {
          log_config = zvec_config_data_get_file_log_config(config);
        }

        if (log_config) {
          std::shared_ptr<zvec::GlobalConfig::LogConfig> cpp_log_config;

          switch (zvec_config_data_get_log_type(config)) {
            case ZVEC_LOG_TYPE_CONSOLE: {
              ZVecConsoleLogConfig *console_config =
                  static_cast<ZVecConsoleLogConfig *>(log_config);
              auto console_level = static_cast<zvec::GlobalConfig::LogLevel>(
                  zvec_config_console_log_get_level(console_config));
              cpp_log_config =
                  std::make_shared<zvec::GlobalConfig::ConsoleLogConfig>(
                      console_level);
              break;
            }
            case ZVEC_LOG_TYPE_FILE: {
              ZVecFileLogConfig *file_config =
                  static_cast<ZVecFileLogConfig *>(log_config);
              auto file_level = static_cast<zvec::GlobalConfig::LogLevel>(
                  zvec_config_file_log_get_level(file_config));
              std::string dir(zvec_config_file_log_get_dir(file_config));
              std::string basename(
                  zvec_config_file_log_get_basename(file_config));
              cpp_log_config =
                  std::make_shared<zvec::GlobalConfig::FileLogConfig>(
                      file_level, dir, basename,
                      zvec_config_file_log_get_file_size(file_config),
                      zvec_config_file_log_get_overdue_days(file_config));
              break;
            }
            default:
              throw std::runtime_error("Unknown log type");
          }
          cpp_config.log_config = cpp_log_config;
        }
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

ZVecErrorCode zvec_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);

  if (!g_initialized.load()) {
    SET_LAST_ERROR(ZVEC_ERROR_FAILED_PRECONDITION, "Library not initialized");
    return ZVEC_ERROR_FAILED_PRECONDITION;
  }

  ZVEC_TRY_RETURN_ERROR("Shutdown failed", g_initialized.store(false);
                        return ZVEC_OK;)
}

bool zvec_is_initialized(void) {
  return g_initialized.load();
}

// =============================================================================
// Error handling interface implementation
// =============================================================================

ZVecErrorCode zvec_get_last_error_details(ZVecErrorDetails *error_details) {
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
  size_t len = str.length();
  char *copy = static_cast<char *>(malloc(len + 1));
  if (!copy) return nullptr;
  strncpy(copy, str.c_str(), len);
  copy[len] = '\0';  // Ensure null-termination
  return copy;
}

// Helper function: free write results returned by detailed DML APIs.
static void free_write_results_internal(ZVecWriteResult *results,
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
static ZVecErrorCode build_write_results(
    const std::vector<zvec::Status> &statuses,
    const std::vector<std::string> &pks, ZVecWriteResult **results,
    size_t *result_count) {
  if (!results || !result_count) {
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  *result_count = statuses.size();
  if (*result_count == 0) {
    *results = nullptr;
    return ZVEC_OK;
  }

  *results = static_cast<ZVecWriteResult *>(
      calloc(*result_count, sizeof(ZVecWriteResult)));
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

static std::vector<std::string> collect_doc_pks(const ZVecDoc **docs,
                                                size_t doc_count) {
  std::vector<std::string> pks;
  pks.reserve(doc_count);
  for (size_t i = 0; i < doc_count; ++i) {
    if (!docs[i]) {
      pks.emplace_back("");
      continue;
    }
    auto doc_ptr =
        reinterpret_cast<const std::shared_ptr<zvec::Doc> *>(docs[i]);
    pks.emplace_back((*doc_ptr)->pk_ref());
  }
  return pks;
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

// Forward declaration: convert C index params to C++
static std::shared_ptr<zvec::IndexParams> convert_c_index_params_to_cpp(
    const ZVecIndexParams *params);

// Helper function: set field index params
static zvec::Status set_field_index_params(zvec::FieldSchema::Ptr &field_schema,
                                           const ZVecFieldSchema *zvec_field) {
  if (!zvec_field_schema_has_index(zvec_field)) {
    return zvec::Status::OK();
  }

  // Get the index params using getter - we need to access internal struct
  // For this internal function, we can access the struct members since it's in
  // the implementation We'll add a friend-like internal getter
  ZVecIndexParams *index_params = nullptr;
  // Use a hack to get the index_params - cast to access internal member
  // This is safe because we're in the implementation file
  struct InternalFieldSchema {
    ZVecString *name;
    ZVecDataType data_type;
    bool nullable;
    uint32_t dimension;
    ZVecIndexParams *index_params;
    bool has_index;
  };
  index_params =
      reinterpret_cast<const InternalFieldSchema *>(zvec_field)->index_params;

  if (!index_params) {
    return zvec::Status::OK();
  }

  // Use the conversion helper function
  auto cpp_params = convert_c_index_params_to_cpp(index_params);
  if (cpp_params) {
    field_schema->set_index_params(cpp_params);
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

ZVecStringArray *zvec_string_array_create_from_strings(const char **strings,
                                                       size_t count) {
  if (!strings || count == 0) {
    return nullptr;
  }
  ZVecStringArray *array = zvec_string_array_create(count);
  for (size_t i = 0; i < count; ++i) {
    zvec_string_array_add(array, i, strings[i]);
  }
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

void zvec_free_ptr(void *ptr) {
  if (ptr) {
    free(ptr);
  }
}

void zvec_free_field_schema(ZVecFieldSchema *field_schema) {
  if (field_schema) {
    // index_params is embedded, no need to free
    free(field_schema);
  }
}

// =============================================================================
// Index parameters management interface implementation (deprecated)
// These are deprecated in favor of the opaque pointer API
// =============================================================================

// Deprecated: Use zvec_index_params_create() instead
void zvec_index_params_init(ZVecIndexParams *params, ZVecIndexType index_type,
                            ZVecMetricType metric_type) {
  // This function is deprecated and should not be used
  // Use zvec_index_params_create() instead
  SET_LAST_ERROR(
      ZVEC_ERROR_NOT_SUPPORTED,
      "zvec_index_params_init is deprecated. Use zvec_index_params_create()");
}

// Deprecated: Use zvec_index_params_set_hnsw_params() instead
void zvec_index_params_set_hnsw(ZVecIndexParams *params, int m,
                                int ef_construction, int ef_search) {
  SET_LAST_ERROR(ZVEC_ERROR_NOT_SUPPORTED,
                 "zvec_index_params_set_hnsw is deprecated. Use "
                 "zvec_index_params_set_hnsw_params()");
}

// Deprecated: Use zvec_index_params_set_ivf_params() instead
void zvec_index_params_set_ivf(ZVecIndexParams *params, int n_list, int n_iters,
                               bool use_soar, int n_probe) {
  SET_LAST_ERROR(ZVEC_ERROR_NOT_SUPPORTED,
                 "zvec_index_params_set_ivf is deprecated. Use "
                 "zvec_index_params_set_ivf_params()");
}

// Deprecated: Use zvec_index_params_set_invert_params() instead
void zvec_index_params_set_invert(ZVecIndexParams *params,
                                  bool enable_range_opt, bool enable_wildcard) {
  SET_LAST_ERROR(ZVEC_ERROR_NOT_SUPPORTED,
                 "zvec_index_params_set_invert is deprecated. Use "
                 "zvec_index_params_set_invert_params()");
}

// =============================================================================
// ZVecIndexParams opaque pointer implementation
// =============================================================================

// Internal structure - holds C++ shared_ptr<IndexParams>
struct ZVecIndexParams {
  std::shared_ptr<zvec::IndexParams> cpp_params;
  ZVecIndexType index_type;
  ZVecMetricType metric_type;
  ZVecQuantizeType quantize_type;

  // Type-specific storage (only one is active based on index_type)
  struct {
    bool enable_range_optimization;
    bool enable_extended_wildcard;
  } invert;

  struct {
    int m;
    int ef_construction;
  } hnsw;

  struct {
    int n_list;
    int n_iters;
    bool use_soar;
  } ivf;
};

// =============================================================================
// ZVecFieldSchema opaque pointer implementation
// =============================================================================

// Internal structure - field schema with private members
struct ZVecFieldSchema {
  ZVecString *name;
  ZVecDataType data_type;
  bool nullable;
  uint32_t dimension;
  ZVecIndexParams *index_params;  // Owned by field schema
  bool has_index;
};

// Internal structure - collection schema with private members
struct ZVecCollectionSchema {
  ZVecString *name;
  ZVecFieldSchema **fields;
  size_t field_count;
  size_t field_capacity;
  uint64_t max_doc_count_per_segment;
};

// =============================================================================
// Configuration structures opaque pointer implementation
// =============================================================================

// Internal structure - QueryParams (base)
struct ZVecQueryParams {
  ZVecIndexType index_type;
  float radius;
  bool is_linear;
  bool is_using_refiner;
};

// Internal structure - HnswQueryParams
struct ZVecHnswQueryParams {
  ZVecQueryParams base;
  int ef;
};

// Internal structure - IVFQueryParams
struct ZVecIVFQueryParams {
  ZVecQueryParams base;
  int nprobe;
  float scale_factor;
};

// Internal structure - FlatQueryParams
struct ZVecFlatQueryParams {
  ZVecQueryParams base;
  float scale_factor;
};

// Internal structure - VectorQuery
struct ZVecVectorQuery {
  int topk;
  ZVecString *field_name;
  ZVecByteArray query_vector;
  ZVecByteArray query_sparse_indices;
  ZVecByteArray query_sparse_values;
  ZVecString *filter;
  bool include_vector;
  bool include_doc_id;
  ZVecStringArray *output_fields;
  void *query_params;         // Type-specific params (HnswQueryParams*,
                              // IVFQueryParams*, etc.)
  ZVecIndexType params_type;  // To track the type of query_params
};

// Internal structure - GroupByVectorQuery
struct ZVecGroupByVectorQuery {
  ZVecString *field_name;
  ZVecByteArray query_vector;
  ZVecByteArray query_sparse_indices;
  ZVecByteArray query_sparse_values;
  ZVecString *filter;
  bool include_vector;
  ZVecStringArray *output_fields;
  ZVecString *group_by_field_name;
  uint32_t group_count;
  uint32_t group_topk;
  void *query_params;         // Type-specific params
  ZVecIndexType params_type;  // To track the type of query_params
};

// Internal structure - CollectionOptions
struct ZVecCollectionOptions {
  bool enable_mmap;
  size_t max_buffer_size;
  bool read_only;
  uint64_t max_doc_count_per_segment;
};

// Internal structure - CollectionStats
struct ZVecCollectionStats {
  uint64_t doc_count;
  ZVecString **index_names;
  float *index_completeness;
  size_t index_count;
};

ZVecIndexParams *zvec_index_params_create(ZVecIndexType index_type) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create ZVecIndexParams",
      ZVecIndexParams *params = new ZVecIndexParams();
      params->index_type = index_type;
      params->metric_type = ZVEC_METRIC_TYPE_L2;  // Default
      params->quantize_type = ZVEC_QUANTIZE_TYPE_UNDEFINED;

      // Initialize type-specific params with defaults
      memset(&params->invert, 0, sizeof(params->invert));
      memset(&params->hnsw, 0, sizeof(params->hnsw));
      memset(&params->ivf, 0, sizeof(params->ivf));

      // Set defaults based on index type
      switch (index_type) {
        case ZVEC_INDEX_TYPE_INVERT:
          params->invert.enable_range_optimization = true;
          params->invert.enable_extended_wildcard = false;
          break;
        case ZVEC_INDEX_TYPE_HNSW:
          params->hnsw.m = 16;
          params->hnsw.ef_construction = 200;
          break;
        case ZVEC_INDEX_TYPE_IVF:
          params->ivf.n_list = 100;
          params->ivf.n_iters = 10;
          params->ivf.use_soar = false;
          break;
        case ZVEC_INDEX_TYPE_FLAT:
        default:
          break;
      }

      return params;)

  return nullptr;
}

void zvec_index_params_destroy(ZVecIndexParams *params) {
  if (params) {
    delete params;
  }
}

ZVecErrorCode zvec_index_params_set_metric_type(ZVecIndexParams *params,
                                                ZVecMetricType metric_type) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Index params pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->metric_type = metric_type;
  return ZVEC_OK;
}

ZVecMetricType zvec_index_params_get_metric_type(
    const ZVecIndexParams *params) {
  if (!params) {
    return ZVEC_METRIC_TYPE_L2;  // Default
  }
  return params->metric_type;
}

ZVecErrorCode zvec_index_params_set_quantize_type(
    ZVecIndexParams *params, ZVecQuantizeType quantize_type) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Index params pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->quantize_type = quantize_type;
  return ZVEC_OK;
}

ZVecQuantizeType zvec_index_params_get_quantize_type(
    const ZVecIndexParams *params) {
  if (!params) {
    return ZVEC_QUANTIZE_TYPE_UNDEFINED;
  }
  return params->quantize_type;
}

ZVecIndexType zvec_index_params_get_type(const ZVecIndexParams *params) {
  if (!params) {
    return ZVEC_INDEX_TYPE_FLAT;  // Default
  }
  return params->index_type;
}

ZVecErrorCode zvec_index_params_set_hnsw_params(ZVecIndexParams *params, int m,
                                                int ef_construction) {
  if (!params || params->index_type != ZVEC_INDEX_TYPE_HNSW) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->hnsw.m = m;
  params->hnsw.ef_construction = ef_construction;
  return ZVEC_OK;
}

ZVecErrorCode zvec_index_params_get_hnsw_params(const ZVecIndexParams *params,
                                                int *out_m,
                                                int *out_ef_construction) {
  if (!params || params->index_type != ZVEC_INDEX_TYPE_HNSW) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not HNSW index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (out_m) *out_m = params->hnsw.m;
  if (out_ef_construction) *out_ef_construction = params->hnsw.ef_construction;
  return ZVEC_OK;
}

ZVecErrorCode zvec_index_params_set_ivf_params(ZVecIndexParams *params,
                                               int n_list, int n_iters,
                                               bool use_soar) {
  if (!params || params->index_type != ZVEC_INDEX_TYPE_IVF) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not IVF index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->ivf.n_list = n_list;
  params->ivf.n_iters = n_iters;
  params->ivf.use_soar = use_soar;
  return ZVEC_OK;
}

ZVecErrorCode zvec_index_params_get_ivf_params(const ZVecIndexParams *params,
                                               int *out_n_list,
                                               int *out_n_iters,
                                               bool *out_use_soar) {
  if (!params || params->index_type != ZVEC_INDEX_TYPE_IVF) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not IVF index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (out_n_list) *out_n_list = params->ivf.n_list;
  if (out_n_iters) *out_n_iters = params->ivf.n_iters;
  if (out_use_soar) *out_use_soar = params->ivf.use_soar;
  return ZVEC_OK;
}

ZVecErrorCode zvec_index_params_set_invert_params(ZVecIndexParams *params,
                                                  bool enable_range_opt,
                                                  bool enable_wildcard) {
  if (!params || params->index_type != ZVEC_INDEX_TYPE_INVERT) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not INVERT index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->invert.enable_range_optimization = enable_range_opt;
  params->invert.enable_extended_wildcard = enable_wildcard;
  return ZVEC_OK;
}

ZVecErrorCode zvec_index_params_get_invert_params(const ZVecIndexParams *params,
                                                  bool *out_enable_range_opt,
                                                  bool *out_enable_wildcard) {
  if (!params || params->index_type != ZVEC_INDEX_TYPE_INVERT) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Invalid params or not INVERT index type");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (out_enable_range_opt)
    *out_enable_range_opt = params->invert.enable_range_optimization;
  if (out_enable_wildcard)
    *out_enable_wildcard = params->invert.enable_extended_wildcard;
  return ZVEC_OK;
}

// Helper function to convert C++ IndexParams to C ZVecIndexParams
static ZVecIndexParams *convert_cpp_index_params_to_c(
    const std::shared_ptr<zvec::IndexParams> &cpp_params) {
  if (!cpp_params) {
    return nullptr;
  }

  ZVecIndexType c_type;
  switch (cpp_params->type()) {
    case zvec::IndexType::HNSW:
      c_type = ZVEC_INDEX_TYPE_HNSW;
      break;
    case zvec::IndexType::IVF:
      c_type = ZVEC_INDEX_TYPE_IVF;
      break;
    case zvec::IndexType::FLAT:
      c_type = ZVEC_INDEX_TYPE_FLAT;
      break;
    case zvec::IndexType::INVERT:
      c_type = ZVEC_INDEX_TYPE_INVERT;
      break;
    default:
      c_type = ZVEC_INDEX_TYPE_FLAT;
      break;
  }

  ZVecIndexParams *params = zvec_index_params_create(c_type);
  if (!params) return nullptr;

  params->cpp_params = cpp_params;

  // Extract metric and quantize types from VectorIndexParams if applicable
  if (cpp_params->is_vector_index_type()) {
    auto *vec_params =
        dynamic_cast<const zvec::VectorIndexParams *>(cpp_params.get());
    if (vec_params) {
      switch (vec_params->metric_type()) {
        case zvec::MetricType::L2:
          params->metric_type = ZVEC_METRIC_TYPE_L2;
          break;
        case zvec::MetricType::IP:
          params->metric_type = ZVEC_METRIC_TYPE_IP;
          break;
        case zvec::MetricType::COSINE:
          params->metric_type = ZVEC_METRIC_TYPE_COSINE;
          break;
        default:
          params->metric_type = ZVEC_METRIC_TYPE_L2;
          break;
      }
      // Note: quantize_type would need similar mapping if used
    }
  }

  // Extract type-specific parameters
  switch (c_type) {
    case ZVEC_INDEX_TYPE_HNSW: {
      auto *hnsw =
          dynamic_cast<const zvec::HnswIndexParams *>(cpp_params.get());
      if (hnsw) {
        params->hnsw.m = hnsw->m();
        params->hnsw.ef_construction = hnsw->ef_construction();
      }
      break;
    }
    case ZVEC_INDEX_TYPE_IVF: {
      auto *ivf = dynamic_cast<const zvec::IVFIndexParams *>(cpp_params.get());
      if (ivf) {
        params->ivf.n_list = ivf->n_list();
        params->ivf.n_iters = ivf->n_iters();
        params->ivf.use_soar = ivf->use_soar();
      }
      break;
    }
    case ZVEC_INDEX_TYPE_INVERT: {
      auto *invert =
          dynamic_cast<const zvec::InvertIndexParams *>(cpp_params.get());
      if (invert) {
        params->invert.enable_range_optimization =
            invert->enable_range_optimization();
        params->invert.enable_extended_wildcard =
            invert->enable_extended_wildcard();
      }
      break;
    }
    default:
      break;
  }

  return params;
}

// Helper function to convert C ZVecIndexParams to C++ IndexParams
static std::shared_ptr<zvec::IndexParams> convert_c_index_params_to_cpp(
    const ZVecIndexParams *params) {
  if (!params) {
    return nullptr;
  }

  zvec::MetricType metric = zvec::MetricType::L2;
  switch (params->metric_type) {
    case ZVEC_METRIC_TYPE_L2:
      metric = zvec::MetricType::L2;
      break;
    case ZVEC_METRIC_TYPE_IP:
      metric = zvec::MetricType::IP;
      break;
    case ZVEC_METRIC_TYPE_COSINE:
      metric = zvec::MetricType::COSINE;
      break;
    default:
      metric = zvec::MetricType::L2;
      break;
  }

  zvec::QuantizeType quantize = zvec::QuantizeType::UNDEFINED;
  // Add quantize type mapping if needed

  switch (params->index_type) {
    case ZVEC_INDEX_TYPE_HNSW:
      return std::make_shared<zvec::HnswIndexParams>(
          metric, params->hnsw.m, params->hnsw.ef_construction, quantize);
    case ZVEC_INDEX_TYPE_IVF:
      return std::make_shared<zvec::IVFIndexParams>(
          metric, params->ivf.n_list, params->ivf.n_iters, params->ivf.use_soar,
          quantize);
    case ZVEC_INDEX_TYPE_FLAT:
      return std::make_shared<zvec::FlatIndexParams>(metric, quantize);
    case ZVEC_INDEX_TYPE_INVERT:
      return std::make_shared<zvec::InvertIndexParams>(
          params->invert.enable_range_optimization,
          params->invert.enable_extended_wildcard);
    default:
      return std::make_shared<zvec::FlatIndexParams>(zvec::MetricType::L2);
  }
}

// =============================================================================
// FieldSchema management interface implementation
// =============================================================================

ZVecFieldSchema *zvec_field_schema_create(const char *name,
                                          ZVecDataType data_type, bool nullable,
                                          uint32_t dimension) {
  if (!name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
    return nullptr;
  }

  ZVecFieldSchema *schema = new ZVecFieldSchema();
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecFieldSchema");
    return nullptr;
  }

  schema->name = zvec_string_create(name);
  if (!schema->name) {
    delete schema;
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to create string for field name");
    return nullptr;
  }

  schema->data_type = data_type;
  schema->nullable = nullable;
  schema->dimension = dimension;
  schema->index_params = nullptr;
  schema->has_index = false;

  return schema;
}

void zvec_field_schema_destroy(ZVecFieldSchema *schema) {
  if (schema) {
    zvec_free_string(schema->name);
    if (schema->index_params) {
      zvec_index_params_destroy(schema->index_params);
    }
    delete schema;
  }
}

// Getter functions
const char *zvec_field_schema_get_name(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return nullptr;
  }
  return zvec_string_c_str(schema->name);
}

ZVecDataType zvec_field_schema_get_data_type(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_DATA_TYPE_UNDEFINED;
  }
  return schema->data_type;
}

bool zvec_field_schema_is_nullable(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  return schema->nullable;
}

ZVecErrorCode zvec_field_schema_set_nullable(ZVecFieldSchema *schema,
                                             bool nullable) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  schema->nullable = nullable;
  return ZVEC_OK;
}

uint32_t zvec_field_schema_get_dimension(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return 0;
  }
  return schema->dimension;
}

ZVecErrorCode zvec_field_schema_set_dimension(ZVecFieldSchema *schema,
                                              uint32_t dimension) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  schema->dimension = dimension;
  return ZVEC_OK;
}

bool zvec_field_schema_has_index(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  return schema->has_index;
}

ZVecIndexType zvec_field_schema_get_index_type(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_INDEX_TYPE_UNDEFINED;
  }
  if (!schema->index_params) {
    return ZVEC_INDEX_TYPE_UNDEFINED;
  }
  return schema->index_params->index_type;
}

const ZVecIndexParams *zvec_field_schema_get_index_params(
    const ZVecFieldSchema *schema) {
  if (!schema) {
    return nullptr;
  }
  return schema->index_params;
}

bool zvec_field_schema_is_vector_field(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  ZVecDataType data_type = schema->data_type;
  return (data_type == ZVEC_DATA_TYPE_VECTOR_FP32 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_FP64 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_FP16 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY32 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY64 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT4 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT8 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT16 ||
          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 ||
          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16);
}

bool zvec_field_schema_is_dense_vector(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  ZVecDataType data_type = schema->data_type;
  return (data_type == ZVEC_DATA_TYPE_VECTOR_FP32 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_FP64 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_FP16 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY32 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY64 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT4 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT8 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT16);
}

bool zvec_field_schema_is_sparse_vector(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  ZVecDataType data_type = schema->data_type;
  return (data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 ||
          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16);
}

bool zvec_field_schema_is_array_type(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  ZVecDataType data_type = schema->data_type;
  return (data_type == ZVEC_DATA_TYPE_ARRAY_BINARY ||
          data_type == ZVEC_DATA_TYPE_ARRAY_STRING ||
          data_type == ZVEC_DATA_TYPE_ARRAY_BOOL ||
          data_type == ZVEC_DATA_TYPE_ARRAY_INT32 ||
          data_type == ZVEC_DATA_TYPE_ARRAY_INT64 ||
          data_type == ZVEC_DATA_TYPE_ARRAY_UINT32 ||
          data_type == ZVEC_DATA_TYPE_ARRAY_UINT64 ||
          data_type == ZVEC_DATA_TYPE_ARRAY_FLOAT ||
          data_type == ZVEC_DATA_TYPE_ARRAY_DOUBLE);
}

ZVecDataType zvec_field_schema_get_element_data_type(
    const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_DATA_TYPE_UNDEFINED;
  }
  ZVecDataType data_type = schema->data_type;
  switch (data_type) {
    case ZVEC_DATA_TYPE_ARRAY_BINARY:
      return ZVEC_DATA_TYPE_BINARY;
    case ZVEC_DATA_TYPE_ARRAY_STRING:
      return ZVEC_DATA_TYPE_STRING;
    case ZVEC_DATA_TYPE_ARRAY_BOOL:
      return ZVEC_DATA_TYPE_BOOL;
    case ZVEC_DATA_TYPE_ARRAY_INT32:
      return ZVEC_DATA_TYPE_INT32;
    case ZVEC_DATA_TYPE_ARRAY_INT64:
      return ZVEC_DATA_TYPE_INT64;
    case ZVEC_DATA_TYPE_ARRAY_UINT32:
      return ZVEC_DATA_TYPE_UINT32;
    case ZVEC_DATA_TYPE_ARRAY_UINT64:
      return ZVEC_DATA_TYPE_UINT64;
    case ZVEC_DATA_TYPE_ARRAY_FLOAT:
      return ZVEC_DATA_TYPE_FLOAT;
    case ZVEC_DATA_TYPE_ARRAY_DOUBLE:
      return ZVEC_DATA_TYPE_DOUBLE;
    default:
      return data_type;
  }
}

bool zvec_field_schema_has_invert_index(const ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return false;
  }
  // Invert index is for non-vector fields with index
  if (zvec_field_schema_is_vector_field(schema)) {
    return false;
  }
  return schema->has_index && schema->index_params &&
         schema->index_params->index_type == ZVEC_INDEX_TYPE_INVERT;
}

// Helper function to check if a data type is a vector type
bool zvec_is_vector_data_type(ZVecDataType data_type) {
  return (data_type == ZVEC_DATA_TYPE_VECTOR_FP32 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_FP64 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_FP16 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY32 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY64 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT4 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT8 ||
          data_type == ZVEC_DATA_TYPE_VECTOR_INT16 ||
          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 ||
          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16);
}

ZVecErrorCode zvec_field_schema_clear_index(ZVecFieldSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (schema->index_params) {
    zvec_index_params_destroy(schema->index_params);
    schema->index_params = nullptr;
  }
  schema->has_index = false;
  return ZVEC_OK;
}

ZVecErrorCode zvec_field_schema_set_index_params(
    ZVecFieldSchema *schema, const ZVecIndexParams *index_params) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!index_params) {
    if (schema->index_params) {
      zvec_index_params_destroy(schema->index_params);
      schema->index_params = nullptr;
    }
    schema->has_index = false;
    return ZVEC_OK;
  }

  // Clone the index_params (create a new copy)
  if (schema->index_params) {
    zvec_index_params_destroy(schema->index_params);
  }
  schema->index_params = zvec_index_params_create(index_params->index_type);
  if (!schema->index_params) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to clone index params");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }

  // Copy all fields using getter/setter API
  ZVecErrorCode err = ZVEC_OK;
  err = zvec_index_params_set_metric_type(schema->index_params,
                                          index_params->metric_type);
  if (err != ZVEC_OK) return err;

  err = zvec_index_params_set_quantize_type(schema->index_params,
                                            index_params->quantize_type);
  if (err != ZVEC_OK) return err;

  // Copy type-specific params
  switch (index_params->index_type) {
    case ZVEC_INDEX_TYPE_INVERT:
      err = zvec_index_params_set_invert_params(
          schema->index_params, index_params->invert.enable_range_optimization,
          index_params->invert.enable_extended_wildcard);
      break;
    case ZVEC_INDEX_TYPE_HNSW:
      err = zvec_index_params_set_hnsw_params(
          schema->index_params, index_params->hnsw.m,
          index_params->hnsw.ef_construction);
      break;
    case ZVEC_INDEX_TYPE_IVF:
      err = zvec_index_params_set_ivf_params(
          schema->index_params, index_params->ivf.n_list,
          index_params->ivf.n_iters, index_params->ivf.use_soar);
      break;
    case ZVEC_INDEX_TYPE_FLAT:
    default:
      break;
  }

  if (err != ZVEC_OK) return err;

  schema->has_index = true;
  return ZVEC_OK;
}

void zvec_field_schema_set_invert_index(ZVecFieldSchema *field_schema,
                                        const ZVecIndexParams *invert_params) {
  if (field_schema && invert_params) {
    if (field_schema->index_params) {
      zvec_index_params_destroy(field_schema->index_params);
    }
    field_schema->index_params =
        zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
    if (field_schema->index_params) {
      field_schema->index_params->index_type = ZVEC_INDEX_TYPE_INVERT;
      field_schema->index_params->metric_type = invert_params->metric_type;
      field_schema->index_params->quantize_type = invert_params->quantize_type;
      field_schema->index_params->invert.enable_range_optimization =
          invert_params->invert.enable_range_optimization;
      field_schema->index_params->invert.enable_extended_wildcard =
          invert_params->invert.enable_extended_wildcard;
      field_schema->has_index = true;
    }
  }
}

void zvec_field_schema_set_hnsw_index(ZVecFieldSchema *field_schema,
                                      const ZVecIndexParams *hnsw_params) {
  if (field_schema && hnsw_params) {
    if (field_schema->index_params) {
      zvec_index_params_destroy(field_schema->index_params);
    }
    field_schema->index_params = zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
    if (field_schema->index_params) {
      field_schema->index_params->index_type = ZVEC_INDEX_TYPE_HNSW;
      field_schema->index_params->metric_type = hnsw_params->metric_type;
      field_schema->index_params->quantize_type = hnsw_params->quantize_type;
      field_schema->index_params->hnsw.m = hnsw_params->hnsw.m;
      field_schema->index_params->hnsw.ef_construction =
          hnsw_params->hnsw.ef_construction;
      field_schema->has_index = true;
    }
  }
}

void zvec_field_schema_set_flat_index(ZVecFieldSchema *field_schema,
                                      const ZVecIndexParams *flat_params) {
  if (field_schema && flat_params) {
    if (field_schema->index_params) {
      zvec_index_params_destroy(field_schema->index_params);
    }
    field_schema->index_params = zvec_index_params_create(ZVEC_INDEX_TYPE_FLAT);
    if (field_schema->index_params) {
      field_schema->index_params->index_type = ZVEC_INDEX_TYPE_FLAT;
      field_schema->index_params->metric_type = flat_params->metric_type;
      field_schema->index_params->quantize_type = flat_params->quantize_type;
      field_schema->has_index = true;
    }
  }
}

void zvec_field_schema_set_ivf_index(ZVecFieldSchema *field_schema,
                                     const ZVecIndexParams *ivf_params) {
  if (field_schema && ivf_params) {
    if (field_schema->index_params) {
      zvec_index_params_destroy(field_schema->index_params);
    }
    field_schema->index_params = zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
    if (field_schema->index_params) {
      field_schema->index_params->index_type = ZVEC_INDEX_TYPE_IVF;
      field_schema->index_params->metric_type = ivf_params->metric_type;
      field_schema->index_params->quantize_type = ivf_params->quantize_type;
      field_schema->index_params->ivf.n_list = ivf_params->ivf.n_list;
      field_schema->index_params->ivf.n_iters = ivf_params->ivf.n_iters;
      field_schema->index_params->ivf.use_soar = ivf_params->ivf.use_soar;
      field_schema->has_index = true;
    }
  }
}

static void zvec_field_schema_cleanup(ZVecFieldSchema *field_schema) {
  if (!field_schema) return;

  zvec_free_string(field_schema->name);
  field_schema->name = nullptr;
  if (field_schema->index_params) {
    zvec_index_params_destroy(field_schema->index_params);
    field_schema->index_params = nullptr;
  }
}

// =============================================================================
// CollectionOptions management interface implementation
// =============================================================================

// =============================================================================
// CollectionOptions functions implementation
// =============================================================================

ZVecCollectionOptions *zvec_collection_options_create(void) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create ZVecCollectionOptions",
      ZVecCollectionOptions *options = new ZVecCollectionOptions();
      options->enable_mmap = true;
      options->max_buffer_size = zvec::DEFAULT_MAX_BUFFER_SIZE;
      options->read_only = false;
      options->max_doc_count_per_segment = zvec::MAX_DOC_COUNT_PER_SEGMENT;
      return options;)
  return nullptr;
}

void zvec_collection_options_destroy(ZVecCollectionOptions *options) {
  if (options) {
    delete options;
  }
}

ZVecErrorCode zvec_collection_options_set_enable_mmap(
    ZVecCollectionOptions *options, bool enable) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  options->enable_mmap = enable;
  return ZVEC_OK;
}

bool zvec_collection_options_get_enable_mmap(
    const ZVecCollectionOptions *options) {
  if (!options) {
    return true;  // Default
  }
  return options->enable_mmap;
}

ZVecErrorCode zvec_collection_options_set_max_buffer_size(
    ZVecCollectionOptions *options, size_t size) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  options->max_buffer_size = size;
  return ZVEC_OK;
}

size_t zvec_collection_options_get_max_buffer_size(
    const ZVecCollectionOptions *options) {
  if (!options) {
    return zvec::DEFAULT_MAX_BUFFER_SIZE;  // Default
  }
  return options->max_buffer_size;
}

ZVecErrorCode zvec_collection_options_set_read_only(
    ZVecCollectionOptions *options, bool read_only) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  options->read_only = read_only;
  return ZVEC_OK;
}

bool zvec_collection_options_get_read_only(
    const ZVecCollectionOptions *options) {
  if (!options) {
    return false;  // Default
  }
  return options->read_only;
}

ZVecErrorCode zvec_collection_options_set_max_doc_count_per_segment(
    ZVecCollectionOptions *options, uint64_t count) {
  if (!options) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection options pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  options->max_doc_count_per_segment = count;
  return ZVEC_OK;
}

uint64_t zvec_collection_options_get_max_doc_count_per_segment(
    const ZVecCollectionOptions *options) {
  if (!options) {
    return zvec::MAX_DOC_COUNT_PER_SEGMENT;  // Default
  }
  return options->max_doc_count_per_segment;
}

// =============================================================================
// CollectionStats functions implementation
// =============================================================================

uint64_t zvec_collection_stats_get_doc_count(const ZVecCollectionStats *stats) {
  if (!stats) {
    return 0;
  }
  return stats->doc_count;
}

size_t zvec_collection_stats_get_index_count(const ZVecCollectionStats *stats) {
  if (!stats) {
    return 0;
  }
  return stats->index_count;
}

const char *zvec_collection_stats_get_index_name(
    const ZVecCollectionStats *stats, size_t index) {
  if (!stats || !stats->index_names || index >= stats->index_count) {
    return nullptr;
  }
  return stats->index_names[index]->data;
}

float zvec_collection_stats_get_index_completeness(
    const ZVecCollectionStats *stats, size_t index) {
  if (!stats || !stats->index_completeness || index >= stats->index_count) {
    return 0.0f;
  }
  return stats->index_completeness[index];
}

// =============================================================================
// CollectionSchema management interface implementation
// =============================================================================

ZVecCollectionSchema *zvec_collection_schema_create(const char *name) {
  if (!name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection name cannot be null");
    return nullptr;
  }

  ZVecCollectionSchema *schema = new ZVecCollectionSchema();
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to allocate memory for ZVecCollectionSchema");
    return nullptr;
  }

  schema->name = zvec_string_create(name);
  if (!schema->name) {
    delete schema;
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                   "Failed to create string for collection name");
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

    delete schema;
  }
}

const char *zvec_collection_schema_get_name(
    const ZVecCollectionSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return nullptr;
  }
  return zvec_string_c_str(schema->name);
}

ZVecErrorCode zvec_collection_schema_add_field(ZVecCollectionSchema *schema,
                                               ZVecFieldSchema *field) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!field) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  const char *field_name = zvec_field_schema_get_name(field);
  if (!field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    const char *existing_name = zvec_field_schema_get_name(schema->fields[i]);
    if (existing_name && strcmp(existing_name, field_name) == 0) {
      SET_LAST_ERROR(ZVEC_ERROR_ALREADY_EXISTS,
                     std::string("Field '") + field_name + "' already exists");
      return ZVEC_ERROR_ALREADY_EXISTS;
    }
  }

  if (schema->field_count >= schema->field_capacity) {
    size_t new_capacity =
        schema->field_capacity == 0 ? 8 : schema->field_capacity * 2;
    ZVecFieldSchema **new_fields = static_cast<ZVecFieldSchema **>(
        malloc(new_capacity * sizeof(ZVecFieldSchema *)));
    if (!new_fields) {
      SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                     "Failed to allocate memory for fields");
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

ZVecErrorCode zvec_collection_schema_add_fields(
    ZVecCollectionSchema *schema, const ZVecFieldSchema *const *fields,
    size_t field_count) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!fields && field_count > 0) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Fields array cannot be null when field_count > 0");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (field_count == 0) {
    return ZVEC_OK;
  }

  // Validate all fields first
  for (size_t i = 0; i < field_count; ++i) {
    if (!fields[i]) {
      SET_LAST_ERROR(
          ZVEC_ERROR_INVALID_ARGUMENT,
          std::string("Field at index ") + std::to_string(i) + " is null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
    const char *field_name = zvec_field_schema_get_name(fields[i]);
    if (!field_name || strlen(field_name) == 0) {
      SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                     std::string("Field at index ") + std::to_string(i) +
                         " has invalid name");
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
      SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                     "Failed to allocate memory for fields");
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }

    for (size_t i = 0; i < schema->field_count; ++i) {
      new_fields[i] = schema->fields[i];
    }

    free(schema->fields);
    schema->fields = new_fields;
    schema->field_capacity = new_capacity;
  }

  // Clone each field and add to schema
  for (size_t i = 0; i < field_count; ++i) {
    const ZVecFieldSchema *src_field = fields[i];
    const char *field_name = zvec_field_schema_get_name(src_field);
    ZVecDataType data_type = zvec_field_schema_get_data_type(src_field);
    bool nullable = zvec_field_schema_is_nullable(src_field);
    uint32_t dimension = zvec_field_schema_get_dimension(src_field);

    // Create a new field with the same properties
    ZVecFieldSchema *new_field =
        zvec_field_schema_create(field_name, data_type, nullable, dimension);
    if (!new_field) {
      // Clean up previously created fields
      for (size_t j = 0; j < i; ++j) {
        zvec_field_schema_destroy(
            schema->fields[schema->field_count - (i - j)]);
      }
      SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED,
                     "Failed to create new field");
      return ZVEC_ERROR_RESOURCE_EXHAUSTED;
    }

    // Copy index params if present
    if (zvec_field_schema_has_index(src_field)) {
      // Internal access: we need to get the index_params pointer
      // Use the same hack as in set_field_index_params
      struct InternalFieldSchema {
        ZVecString *name;
        ZVecDataType data_type;
        bool nullable;
        uint32_t dimension;
        ZVecIndexParams *index_params;
        bool has_index;
      };
      const ZVecIndexParams *src_index_params =
          reinterpret_cast<const InternalFieldSchema *>(src_field)
              ->index_params;
      if (src_index_params) {
        zvec_field_schema_set_index_params(new_field, src_index_params);
      }
    }

    schema->fields[schema->field_count] = new_field;
    schema->field_count++;
  }

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_remove_field(ZVecCollectionSchema *schema,
                                                  const char *field_name) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
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

  SET_LAST_ERROR(ZVEC_ERROR_NOT_FOUND,
                 std::string("Field '") + field_name + "' not found");
  return ZVEC_ERROR_NOT_FOUND;
}

ZVecErrorCode zvec_collection_schema_remove_fields(
    ZVecCollectionSchema *schema, const char *const *field_names,
    size_t field_count) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (!field_names && field_count > 0) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Field names array cannot be null when field_count > 0");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (field_count == 0) {
    return ZVEC_OK;
  }

  for (size_t i = 0; i < field_count; ++i) {
    if (!field_names[i]) {
      SET_LAST_ERROR(
          ZVEC_ERROR_INVALID_ARGUMENT,
          std::string("Field name at index ") + std::to_string(i) + " is null");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
  }

  std::vector<size_t> remove_indices;
  std::vector<std::string> not_found_fields;

  for (size_t field_idx = 0; field_idx < field_count; ++field_idx) {
    std::string target_name(field_names[field_idx]);
    bool found = false;

    for (size_t i = 0; i < schema->field_count; ++i) {
      const char *current_name = zvec_field_schema_get_name(schema->fields[i]);
      if (current_name && strcmp(current_name, target_name.c_str()) == 0) {
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
    SET_LAST_ERROR(ZVEC_ERROR_NOT_FOUND, error_msg);
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
    const char *current_name = zvec_field_schema_get_name(schema->fields[i]);
    if (current_name && strcmp(current_name, field_name) == 0) {
      return schema->fields[i];
    }
  }

  return nullptr;
}

size_t zvec_collection_schema_get_field_count(
    const ZVecCollectionSchema *schema) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return 0;
  }

  return schema->field_count;
}

ZVecFieldSchema *zvec_collection_schema_get_field(
    const ZVecCollectionSchema *schema, size_t index) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return nullptr;
  }

  if (index >= schema->field_count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field index out of bounds");
    return nullptr;
  }

  return schema->fields[index];
}

ZVecErrorCode zvec_collection_schema_set_max_doc_count_per_segment(
    ZVecCollectionSchema *schema, uint64_t max_doc_count) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
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
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (error_msg) {
    *error_msg = nullptr;
  }

  if (!schema->name) {
    if (error_msg) {
      *error_msg = zvec_string_create("Collection name is required");
    }
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Collection name is required");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  if (schema->field_count == 0) {
    if (error_msg) {
      *error_msg = zvec_string_create("At least one field is required");
    }
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "At least one field is required");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    auto field = schema->fields[i];
    if (!field) {
      if (error_msg) {
        *error_msg = zvec_string_create("Null field found");
      }
      SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Null field found");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }

    const char *field_name = zvec_field_schema_get_name(field);
    if (!field_name || strlen(field_name) == 0) {
      if (error_msg) {
        *error_msg = zvec_string_create("Field name is required");
      }
      SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name is required");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
  }

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_set_name(ZVecCollectionSchema *schema,
                                              const char *name) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_VOID
  if (schema->name) {
    zvec_free_string(schema->name);
  }
  schema->name = zvec_string_create(name);
  ZVEC_CATCH_END_VOID

  return ZVEC_OK;
}

bool zvec_collection_schema_has_field(const ZVecCollectionSchema *schema,
                                      const char *field_name) {
  if (!schema || !field_name) {
    return false;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    const char *name = zvec_field_schema_get_name(schema->fields[i]);
    if (name && strcmp(name, field_name) == 0) {
      return true;
    }
  }
  return false;
}

ZVecErrorCode zvec_collection_schema_alter_field(
    ZVecCollectionSchema *schema, const char *field_name,
    const ZVecFieldSchema *new_field) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!new_field) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "New field cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_CODE
  // Find the field
  for (size_t i = 0; i < schema->field_count; ++i) {
    const char *name = zvec_field_schema_get_name(schema->fields[i]);
    if (name && strcmp(name, field_name) == 0) {
      // Clone the new field
      ZVecFieldSchema *cloned =
          zvec_field_schema_create(zvec_field_schema_get_name(new_field),
                                   zvec_field_schema_get_data_type(new_field),
                                   zvec_field_schema_is_nullable(new_field),
                                   zvec_field_schema_get_dimension(new_field));

      if (zvec_field_schema_has_index(new_field)) {
        ZVecIndexType idx_type = zvec_field_schema_get_index_type(new_field);
        ZVecIndexParams *cloned_params = zvec_index_params_create(idx_type);
        const ZVecIndexParams *src_params =
            zvec_field_schema_get_index_params(new_field);

        // Copy index parameters
        switch (idx_type) {
          case ZVEC_INDEX_TYPE_INVERT: {
            bool enable_opt;
            bool enable_wildcard;
            zvec_index_params_get_invert_params(src_params, &enable_opt,
                                                &enable_wildcard);
            zvec_index_params_set_invert_params(cloned_params, enable_opt,
                                                enable_wildcard);
            break;
          }
          case ZVEC_INDEX_TYPE_HNSW: {
            int m, ef_const;
            zvec_index_params_get_hnsw_params(src_params, &m, &ef_const);
            zvec_index_params_set_hnsw_params(cloned_params, m, ef_const);
            break;
          }
          case ZVEC_INDEX_TYPE_IVF: {
            int n_list, n_iters;
            bool use_soar;
            zvec_index_params_get_ivf_params(src_params, &n_list, &n_iters,
                                             &use_soar);
            zvec_index_params_set_ivf_params(cloned_params, n_list, n_iters,
                                             use_soar);
            break;
          }
          default:
            break;
        }

        zvec_field_schema_set_index_params(cloned, cloned_params);
        zvec_index_params_destroy(cloned_params);
      }

      // Destroy old field and replace with new one
      zvec_field_schema_destroy(schema->fields[i]);
      schema->fields[i] = cloned;
      return ZVEC_OK;
    }
  }

  SET_LAST_ERROR(ZVEC_ERROR_NOT_FOUND, "Field not found");
  return ZVEC_ERROR_NOT_FOUND;
  ZVEC_CATCH_END_CODE(ZVEC_ERROR_UNKNOWN)
}

ZVecFieldSchema *zvec_collection_schema_get_forward_field(
    const ZVecCollectionSchema *schema, const char *field_name) {
  if (!schema || !field_name) {
    return nullptr;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    const char *name = zvec_field_schema_get_name(field);
    if (name && strcmp(name, field_name) == 0) {
      // Check if it's a scalar field (not vector)
      ZVecDataType data_type = zvec_field_schema_get_data_type(field);
      if (!zvec_is_vector_data_type(data_type)) {
        return field;
      }
    }
  }
  return nullptr;
}

ZVecFieldSchema *zvec_collection_schema_get_vector_field(
    const ZVecCollectionSchema *schema, const char *field_name) {
  if (!schema || !field_name) {
    return nullptr;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    const char *name = zvec_field_schema_get_name(field);
    if (name && strcmp(name, field_name) == 0) {
      // Check if it's a vector field
      ZVecDataType data_type = zvec_field_schema_get_data_type(field);
      if (zvec_is_vector_data_type(data_type)) {
        return field;
      }
    }
  }
  return nullptr;
}

ZVecErrorCode zvec_collection_schema_get_forward_fields(
    const ZVecCollectionSchema *schema, ZVecFieldSchema ***fields,
    size_t *count) {
  if (!schema || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, fields, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_VOID
  // Count scalar fields
  size_t scalar_count = 0;
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecDataType data_type = zvec_field_schema_get_data_type(schema->fields[i]);
    if (!zvec_is_vector_data_type(data_type)) {
      scalar_count++;
    }
  }

  *fields =
      (ZVecFieldSchema **)malloc(scalar_count * sizeof(ZVecFieldSchema *));
  if (!*fields) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED, "Failed to allocate memory");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }

  // Fill the array
  size_t idx = 0;
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecDataType data_type = zvec_field_schema_get_data_type(schema->fields[i]);
    if (!zvec_is_vector_data_type(data_type)) {
      (*fields)[idx++] = schema->fields[i];
    }
  }

  *count = scalar_count;
  ZVEC_CATCH_END_VOID

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_get_forward_fields_with_index(
    const ZVecCollectionSchema *schema, ZVecFieldSchema ***fields,
    size_t *count) {
  if (!schema || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, fields, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_VOID
  // Count scalar fields with index
  size_t indexed_count = 0;
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    ZVecDataType data_type = zvec_field_schema_get_data_type(field);
    if (!zvec_is_vector_data_type(data_type) &&
        zvec_field_schema_has_index(field)) {
      indexed_count++;
    }
  }

  *fields =
      (ZVecFieldSchema **)malloc(indexed_count * sizeof(ZVecFieldSchema *));
  if (!*fields) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED, "Failed to allocate memory");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }

  // Fill the array
  size_t idx = 0;
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    ZVecDataType data_type = zvec_field_schema_get_data_type(field);
    if (!zvec_is_vector_data_type(data_type) &&
        zvec_field_schema_has_index(field)) {
      (*fields)[idx++] = field;
    }
  }

  *count = indexed_count;
  ZVEC_CATCH_END_VOID

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_get_all_field_names(
    const ZVecCollectionSchema *schema, const char ***names, size_t *count) {
  if (!schema || !names || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, names, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_VOID
  *count = schema->field_count;
  *names = (const char **)malloc(schema->field_count * sizeof(const char *));
  if (!*names) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED, "Failed to allocate memory");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    (*names)[i] = zvec_field_schema_get_name(schema->fields[i]);
  }

  ZVEC_CATCH_END_VOID

  return ZVEC_OK;
}

ZVecErrorCode zvec_collection_schema_get_vector_fields(
    const ZVecCollectionSchema *schema, ZVecFieldSchema ***fields,
    size_t *count) {
  if (!schema || !fields || !count) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Schema, fields, and count cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_VOID
  // Count vector fields
  size_t vector_count = 0;
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecDataType data_type = zvec_field_schema_get_data_type(schema->fields[i]);
    if (zvec_is_vector_data_type(data_type)) {
      vector_count++;
    }
  }

  *fields =
      (ZVecFieldSchema **)malloc(vector_count * sizeof(ZVecFieldSchema *));
  if (!*fields) {
    SET_LAST_ERROR(ZVEC_ERROR_RESOURCE_EXHAUSTED, "Failed to allocate memory");
    return ZVEC_ERROR_RESOURCE_EXHAUSTED;
  }

  // Fill the array
  size_t idx = 0;
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecDataType data_type = zvec_field_schema_get_data_type(schema->fields[i]);
    if (zvec_is_vector_data_type(data_type)) {
      (*fields)[idx++] = schema->fields[i];
    }
  }

  *count = vector_count;
  ZVEC_CATCH_END_VOID

  return ZVEC_OK;
}

bool zvec_collection_schema_has_index(const ZVecCollectionSchema *schema,
                                      const char *field_name) {
  if (!schema || !field_name) {
    return false;
  }

  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    const char *name = zvec_field_schema_get_name(field);
    if (name && strcmp(name, field_name) == 0) {
      return zvec_field_schema_has_index(field);
    }
  }
  return false;
}

ZVecErrorCode zvec_collection_schema_add_index(
    ZVecCollectionSchema *schema, const char *field_name,
    const ZVecIndexParams *index_params) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!index_params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Index params cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_CODE
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    const char *name = zvec_field_schema_get_name(field);
    if (name && strcmp(name, field_name) == 0) {
      // Clone the index params
      ZVecIndexType idx_type = zvec_index_params_get_type(index_params);
      ZVecIndexParams *cloned_params = zvec_index_params_create(idx_type);

      // Copy parameters based on type
      switch (idx_type) {
        case ZVEC_INDEX_TYPE_INVERT: {
          bool enable_opt, enable_wildcard;
          zvec_index_params_get_invert_params(index_params, &enable_opt,
                                              &enable_wildcard);
          zvec_index_params_set_invert_params(cloned_params, enable_opt,
                                              enable_wildcard);
          break;
        }
        case ZVEC_INDEX_TYPE_HNSW: {
          int m, ef_const;
          zvec_index_params_get_hnsw_params(index_params, &m, &ef_const);
          zvec_index_params_set_hnsw_params(cloned_params, m, ef_const);
          break;
        }
        case ZVEC_INDEX_TYPE_IVF: {
          int n_list, n_iters;
          bool use_soar;
          zvec_index_params_get_ivf_params(index_params, &n_list, &n_iters,
                                           &use_soar);
          zvec_index_params_set_ivf_params(cloned_params, n_list, n_iters,
                                           use_soar);
          break;
        }
        default:
          break;
      }

      zvec_field_schema_set_index_params(field, cloned_params);
      zvec_index_params_destroy(cloned_params);
      return ZVEC_OK;
    }
  }

  SET_LAST_ERROR(ZVEC_ERROR_NOT_FOUND, "Field not found");
  return ZVEC_ERROR_NOT_FOUND;
  ZVEC_CATCH_END_CODE(ZVEC_ERROR_UNKNOWN)
}

ZVecErrorCode zvec_collection_schema_drop_index(ZVecCollectionSchema *schema,
                                                const char *field_name) {
  if (!schema) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Collection schema pointer cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (!field_name) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Field name cannot be null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_BEGIN_CODE
  for (size_t i = 0; i < schema->field_count; ++i) {
    ZVecFieldSchema *field = schema->fields[i];
    const char *name = zvec_field_schema_get_name(field);
    if (name && strcmp(name, field_name) == 0) {
      // Clear the index
      zvec_field_schema_clear_index(field);
      return ZVEC_OK;
    }
  }

  SET_LAST_ERROR(ZVEC_ERROR_NOT_FOUND, "Field not found");
  return ZVEC_ERROR_NOT_FOUND;
  ZVEC_CATCH_END_CODE(ZVEC_ERROR_UNKNOWN)
}

void zvec_collection_schema_cleanup(ZVecCollectionSchema *schema) {
  if (!schema) return;

  ZVEC_TRY_BEGIN_VOID
  if (schema->name) {
    zvec_free_string(schema->name);
  }

  if (schema->fields) {
    for (size_t i = 0; i < schema->field_count; ++i) {
      zvec_field_schema_destroy(schema->fields[i]);
    }
    free(schema->fields);
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
  ZVecDataType data_type = zvec_field_schema_get_data_type(&zvec_field);
  bool is_vector_field = (data_type == ZVEC_DATA_TYPE_VECTOR_FP32 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_FP64 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_FP16 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY32 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_BINARY64 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_INT4 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_INT8 ||
                          data_type == ZVEC_DATA_TYPE_VECTOR_INT16 ||
                          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 ||
                          data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16);
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

void zvec_write_results_free(ZVecWriteResult *results, size_t result_count) {
  free_write_results_internal(results, result_count);
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

ZVecErrorCode zvec_doc_set_field_null(ZVecDoc *doc, const char *field_name) {
  if (!doc || !field_name) {
    set_last_error("Invalid arguments: null pointer");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }

  ZVEC_TRY_RETURN_ERROR(
      "Failed to set null field",
      auto doc_ptr = reinterpret_cast<std::shared_ptr<zvec::Doc> *>(doc);
      (*doc_ptr)->set_null(std::string(field_name)); return ZVEC_OK;)
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
  std::string coll_name(zvec_string_c_str(schema->name),
                        zvec_string_length(schema->name));
  collection_schema = std::make_shared<zvec::CollectionSchema>(coll_name);
  collection_schema->set_max_doc_count_per_segment(
      schema->max_doc_count_per_segment);

  for (size_t i = 0; i < schema->field_count; ++i) {
    const ZVecFieldSchema *zvec_field = schema->fields[i];
    ZVecDataType field_data_type = zvec_field_schema_get_data_type(zvec_field);
    zvec::DataType data_type = convert_data_type(field_data_type);
    std::string field_name = zvec_field_schema_get_name(zvec_field);
    bool nullable = zvec_field_schema_is_nullable(zvec_field);
    uint32_t dimension = zvec_field_schema_get_dimension(zvec_field);
    zvec::FieldSchema::Ptr field_schema;

    bool is_vector_field = check_is_vector_field(*zvec_field);

    if (is_vector_field) {
      field_schema = std::make_shared<zvec::FieldSchema>(field_name, data_type,
                                                         dimension, nullable);
    } else {
      field_schema =
          std::make_shared<zvec::FieldSchema>(field_name, data_type, nullable);
    }

    if (zvec_field_schema_has_index(zvec_field)) {
      zvec::Status status = set_field_index_params(field_schema, zvec_field);
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
    const ZVecFieldSchema *zvec_field, zvec::FieldSchema::Ptr &field_schema) {
  // Validate input
  if (!zvec_field) {
    return zvec::Status::InvalidArgument("Field schema cannot be null");
  }

  const char *field_name_cstr = zvec_field_schema_get_name(zvec_field);
  if (!field_name_cstr) {
    return zvec::Status::InvalidArgument("Field name cannot be null");
  }

  ZVecDataType data_type = zvec_field_schema_get_data_type(zvec_field);
  zvec::DataType data_type_internal = convert_data_type(data_type);
  if (data_type_internal == zvec::DataType::UNDEFINED) {
    return zvec::Status::InvalidArgument("Invalid data type");
  }

  std::string field_name(field_name_cstr);
  bool nullable = zvec_field_schema_is_nullable(zvec_field);
  uint32_t dimension = zvec_field_schema_get_dimension(zvec_field);
  bool is_vector_field = check_is_vector_field(*zvec_field);

  if (is_vector_field) {
    field_schema = std::make_shared<zvec::FieldSchema>(
        field_name, data_type_internal, dimension, nullable);

    if (zvec_field_schema_has_index(zvec_field)) {
      // Internal access to index_params
      struct InternalFieldSchema {
        ZVecString *name;
        ZVecDataType data_type;
        bool nullable;
        uint32_t dimension;
        ZVecIndexParams *index_params;
        bool has_index;
      };
      const ZVecIndexParams *index_params =
          reinterpret_cast<const InternalFieldSchema *>(zvec_field)
              ->index_params;

      if (index_params) {
        ZVecIndexType index_type = zvec_index_params_get_type(index_params);
        ZVecMetricType metric_type =
            zvec_index_params_get_metric_type(index_params);
        ZVecQuantizeType quantize_type =
            zvec_index_params_get_quantize_type(index_params);

        auto metric = convert_metric_type(metric_type);
        auto quantize = convert_quantize_type(quantize_type);

        switch (index_type) {
          case ZVEC_INDEX_TYPE_HNSW: {
            int m, ef_construction;
            zvec_index_params_get_hnsw_params(index_params, &m,
                                              &ef_construction);
            auto hnsw_params = std::make_shared<zvec::HnswIndexParams>(
                metric, m, ef_construction, quantize);
            field_schema->set_index_params(hnsw_params);
            break;
          }
          case ZVEC_INDEX_TYPE_FLAT: {
            auto flat_params =
                std::make_shared<zvec::FlatIndexParams>(metric, quantize);
            field_schema->set_index_params(flat_params);
            break;
          }
          case ZVEC_INDEX_TYPE_IVF: {
            int n_list, n_iters;
            bool use_soar;
            zvec_index_params_get_ivf_params(index_params, &n_list, &n_iters,
                                             &use_soar);
            auto ivf_params = std::make_shared<zvec::IVFIndexParams>(
                metric, n_list, n_iters, use_soar, quantize);
            field_schema->set_index_params(ivf_params);
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
      field_schema->set_index_params(
          std::make_shared<zvec::FlatIndexParams>(zvec::MetricType::L2));
    }
  } else {
    field_schema = std::make_shared<zvec::FieldSchema>(
        field_name, data_type_internal, nullable);

    if (zvec_field_schema_has_index(zvec_field)) {
      struct InternalFieldSchema {
        ZVecString *name;
        ZVecDataType data_type;
        bool nullable;
        uint32_t dimension;
        ZVecIndexParams *index_params;
        bool has_index;
      };
      const ZVecIndexParams *index_params =
          reinterpret_cast<const InternalFieldSchema *>(zvec_field)
              ->index_params;

      if (index_params &&
          zvec_index_params_get_type(index_params) == ZVEC_INDEX_TYPE_INVERT) {
        bool enable_range_opt, enable_wildcard;
        zvec_index_params_get_invert_params(index_params, &enable_range_opt,
                                            &enable_wildcard);
        auto invert_params = std::make_shared<zvec::InvertIndexParams>(
            enable_range_opt, enable_wildcard);
        field_schema->set_index_params(invert_params);
      }
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
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          auto vec =
              extract_vector_values<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_binary64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          auto vec =
              extract_vector_values<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          auto vec = extract_vector_values<zvec::float16_t>(value, value_size,
                                                            &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp16 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          auto vec =
              extract_vector_values<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_fp64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          auto vec =
              extract_vector_values<int8_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_int8 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          auto vec =
              extract_vector_values<int16_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for vector_int16 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
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
          (*doc_ptr)->set(name, std::move(vec));
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
          (*doc_ptr)->set(name, std::move(sparse_vec));
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          auto sparse_vec =
              extract_sparse_vector<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid sparse vector data size");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(sparse_vec));
          break;
        }

        // Array types
        case ZVEC_DATA_TYPE_ARRAY_BINARY: {
          auto binary_array = extract_binary_array(value, value_size);
          (*doc_ptr)->set(name, std::move(binary_array));
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
            (*doc_ptr)->set(name, std::move(string_array));
          } else {
            // C-string array (null-terminated strings)
            auto string_array = extract_string_array(value, value_size);
            (*doc_ptr)->set(name, std::move(string_array));
          }
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          auto vec = extract_array_values<bool>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_bool type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          auto vec =
              extract_array_values<int32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_int32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          auto vec =
              extract_array_values<int64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_int64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          auto vec =
              extract_array_values<uint32_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_uint32 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          auto vec =
              extract_array_values<uint64_t>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_uint64 type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          auto vec =
              extract_array_values<float>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_float type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          auto vec =
              extract_array_values<double>(value, value_size, &error_code);
          if (error_code != ZVEC_OK) {
            set_last_error("Invalid value size for array_double type");
            return error_code;
          }
          (*doc_ptr)->set(name, std::move(vec));
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
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
          std::vector<uint64_t> vec(reinterpret_cast<const uint64_t *>(
                                        field->value.vector_value.data),
                                    reinterpret_cast<const uint64_t *>(
                                        field->value.vector_value.data) +
                                        field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP16: {
          std::vector<zvec::float16_t> vec(
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const zvec::float16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP32: {
          std::vector<float> vec(field->value.vector_value.data,
                                 field->value.vector_value.data +
                                     field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_FP64: {
          std::vector<double> vec(
              reinterpret_cast<const double *>(field->value.vector_value.data),
              reinterpret_cast<const double *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
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
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT8: {
          std::vector<int8_t> vec(
              reinterpret_cast<const int8_t *>(field->value.vector_value.data),
              reinterpret_cast<const int8_t *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_VECTOR_INT16: {
          std::vector<int16_t> vec(
              reinterpret_cast<const int16_t *>(field->value.vector_value.data),
              reinterpret_cast<const int16_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
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
          (*doc_ptr)->set(name, std::move(vec));
          break;
        }
        case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
          std::vector<float> vec(field->value.vector_value.data,
                                 field->value.vector_value.data +
                                     field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(vec));
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
          (*doc_ptr)->set(name, std::move(array_values));
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
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_BOOL: {
          std::vector<bool> array_values(
              reinterpret_cast<const bool *>(field->value.binary_value.data),
              reinterpret_cast<const bool *>(field->value.binary_value.data) +
                  field->value.binary_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT32: {
          std::vector<int32_t> array_values(
              reinterpret_cast<const int32_t *>(field->value.vector_value.data),
              reinterpret_cast<const int32_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_INT64: {
          std::vector<int64_t> array_values(
              reinterpret_cast<const int64_t *>(field->value.vector_value.data),
              reinterpret_cast<const int64_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT32: {
          std::vector<uint32_t> array_values(
              reinterpret_cast<const uint32_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const uint32_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_UINT64: {
          std::vector<uint64_t> array_values(
              reinterpret_cast<const uint64_t *>(
                  field->value.vector_value.data),
              reinterpret_cast<const uint64_t *>(
                  field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
          std::vector<float> array_values(field->value.vector_value.data,
                                          field->value.vector_value.data +
                                              field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
          break;
        }
        case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
          std::vector<double> array_values(
              reinterpret_cast<const double *>(field->value.vector_value.data),
              reinterpret_cast<const double *>(field->value.vector_value.data) +
                  field->value.vector_value.length);
          (*doc_ptr)->set(name, std::move(array_values));
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

              // Initialize index parameters to nullptr
              c_schema->fields[i]->index_params = nullptr;
              c_schema->fields[i]->has_index = false;

              // Convert index parameters based on the actual type
              auto index_params = cpp_field->index_params();
              if (index_params) {
                // Use helper function to convert C++ index params to C
                c_schema->fields[i]->index_params =
                    convert_cpp_index_params_to_c(index_params);
                if (c_schema->fields[i]->index_params) {
                  c_schema->fields[i]->has_index = true;
                }
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

          // Create and initialize options using new
          *options = new ZVecCollectionOptions();
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

          *stats = new ZVecCollectionStats();
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
  ZVEC_TRY_RETURN_NULL("Failed to create ZVecQueryParams",
                       ZVecQueryParams *params = new ZVecQueryParams();
                       params->index_type = index_type; params->radius = 0.0f;
                       params->is_linear = false;
                       params->is_using_refiner = false; return params;)
  return nullptr;
}

void zvec_query_params_destroy(ZVecQueryParams *params) {
  if (params) {
    delete params;
  }
}

ZVecErrorCode zvec_query_params_set_index_type(ZVecQueryParams *params,
                                               ZVecIndexType index_type) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->index_type = index_type;
  return ZVEC_OK;
}

ZVecIndexType zvec_query_params_get_index_type(const ZVecQueryParams *params) {
  if (!params) {
    return ZVEC_INDEX_TYPE_UNDEFINED;
  }
  return params->index_type;
}

ZVecErrorCode zvec_query_params_set_radius(ZVecQueryParams *params,
                                           float radius) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->radius = radius;
  return ZVEC_OK;
}

float zvec_query_params_get_radius(const ZVecQueryParams *params) {
  if (!params) {
    return 0.0f;
  }
  return params->radius;
}

ZVecErrorCode zvec_query_params_set_is_linear(ZVecQueryParams *params,
                                              bool is_linear) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->is_linear = is_linear;
  return ZVEC_OK;
}

bool zvec_query_params_get_is_linear(const ZVecQueryParams *params) {
  if (!params) {
    return false;
  }
  return params->is_linear;
}

ZVecErrorCode zvec_query_params_set_is_using_refiner(ZVecQueryParams *params,
                                                     bool is_using_refiner) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->is_using_refiner = is_using_refiner;
  return ZVEC_OK;
}

bool zvec_query_params_get_is_using_refiner(const ZVecQueryParams *params) {
  if (!params) {
    return false;
  }
  return params->is_using_refiner;
}

// =============================================================================
// HnswQueryParams functions implementation
// =============================================================================

ZVecHnswQueryParams *zvec_query_params_hnsw_create(int ef, float radius,
                                                   bool is_linear,
                                                   bool is_using_refiner) {
  ZVEC_TRY_RETURN_NULL("Failed to create ZVecHnswQueryParams",
                       ZVecHnswQueryParams *params = new ZVecHnswQueryParams();
                       params->base.index_type = ZVEC_INDEX_TYPE_HNSW;
                       params->base.radius = radius;
                       params->base.is_linear = is_linear;
                       params->base.is_using_refiner = is_using_refiner;
                       params->ef = ef; return params;)
  return nullptr;
}

void zvec_query_params_hnsw_destroy(ZVecHnswQueryParams *params) {
  if (params) {
    delete params;
  }
}

ZVecErrorCode zvec_query_params_hnsw_set_ef(ZVecHnswQueryParams *params,
                                            int ef) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "HNSW query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->ef = ef;
  return ZVEC_OK;
}

int zvec_query_params_hnsw_get_ef(const ZVecHnswQueryParams *params) {
  if (!params) {
    return zvec::core_interface::kDefaultHnswEfSearch;
  }
  return params->ef;
}

// =============================================================================
// IVFQueryParams functions implementation
// =============================================================================

ZVecIVFQueryParams *zvec_query_params_ivf_create(int nprobe,
                                                 bool is_using_refiner,
                                                 float scale_factor) {
  ZVEC_TRY_RETURN_NULL("Failed to create ZVecIVFQueryParams",
                       ZVecIVFQueryParams *params = new ZVecIVFQueryParams();
                       params->base.index_type = ZVEC_INDEX_TYPE_IVF;
                       params->base.is_using_refiner = is_using_refiner;
                       params->nprobe = nprobe;
                       params->scale_factor = scale_factor; return params;)
  return nullptr;
}

void zvec_query_params_ivf_destroy(ZVecIVFQueryParams *params) {
  if (params) {
    delete params;
  }
}

ZVecErrorCode zvec_query_params_ivf_set_nprobe(ZVecIVFQueryParams *params,
                                               int nprobe) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->nprobe = nprobe;
  return ZVEC_OK;
}

int zvec_query_params_ivf_get_nprobe(const ZVecIVFQueryParams *params) {
  if (!params) {
    return 10;
  }
  return params->nprobe;
}

ZVecErrorCode zvec_query_params_ivf_set_scale_factor(ZVecIVFQueryParams *params,
                                                     float scale_factor) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "IVF query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->scale_factor = scale_factor;
  return ZVEC_OK;
}

float zvec_query_params_ivf_get_scale_factor(const ZVecIVFQueryParams *params) {
  if (!params) {
    return 10.0f;
  }
  return params->scale_factor;
}

// =============================================================================
// FlatQueryParams functions implementation
// =============================================================================

ZVecFlatQueryParams *zvec_query_params_flat_create(bool is_using_refiner,
                                                   float scale_factor) {
  ZVEC_TRY_RETURN_NULL("Failed to create ZVecFlatQueryParams",
                       ZVecFlatQueryParams *params = new ZVecFlatQueryParams();
                       params->base.index_type = ZVEC_INDEX_TYPE_FLAT;
                       params->base.is_using_refiner = is_using_refiner;
                       params->scale_factor = scale_factor; return params;)
  return nullptr;
}

void zvec_query_params_flat_destroy(ZVecFlatQueryParams *params) {
  if (params) {
    delete params;
  }
}

ZVecErrorCode zvec_query_params_flat_set_scale_factor(
    ZVecFlatQueryParams *params, float scale_factor) {
  if (!params) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Flat query params pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  params->scale_factor = scale_factor;
  return ZVEC_OK;
}

float zvec_query_params_flat_get_scale_factor(
    const ZVecFlatQueryParams *params) {
  if (!params) {
    return 10.0f;
  }
  return params->scale_factor;
}

// =============================================================================
// VectorQuery and GroupByVectorQuery functions implementation
// =============================================================================

ZVecVectorQuery *zvec_vector_query_create(void) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create ZVecVectorQuery",
      ZVecVectorQuery *query = new ZVecVectorQuery();
      query->topk = 10; query->field_name = nullptr;
      query->query_vector.data = nullptr; query->query_vector.length = 0;
      query->query_sparse_indices.data = nullptr;
      query->query_sparse_indices.length = 0;
      query->query_sparse_values.data = nullptr;
      query->query_sparse_values.length = 0; query->filter = nullptr;
      query->include_vector = false; query->include_doc_id = true;
      query->output_fields = nullptr; query->query_params = nullptr;
      query->params_type = ZVEC_INDEX_TYPE_UNDEFINED; return query;)
  return nullptr;
}

void zvec_vector_query_destroy(ZVecVectorQuery *query) {
  if (query) {
    if (query->field_name) {
      zvec_free_string(query->field_name);
    }
    if (query->filter) {
      zvec_free_string(query->filter);
    }
    if (query->output_fields) {
      zvec_string_array_destroy(query->output_fields);
    }
    if (query->query_params) {
      // Delete type-specific params based on params_type
      switch (query->params_type) {
        case ZVEC_INDEX_TYPE_HNSW:
          delete static_cast<ZVecHnswQueryParams *>(query->query_params);
          break;
        case ZVEC_INDEX_TYPE_IVF:
          delete static_cast<ZVecIVFQueryParams *>(query->query_params);
          break;
        case ZVEC_INDEX_TYPE_FLAT:
          delete static_cast<ZVecFlatQueryParams *>(query->query_params);
          break;
        default:
          delete static_cast<ZVecQueryParams *>(query->query_params);
          break;
      }
    }
    delete query;
  }
}

ZVecErrorCode zvec_vector_query_set_topk(ZVecVectorQuery *query, int topk) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->topk = topk;
  return ZVEC_OK;
}

int zvec_vector_query_get_topk(const ZVecVectorQuery *query) {
  if (!query) {
    return 10;
  }
  return query->topk;
}

ZVecErrorCode zvec_vector_query_set_field_name(ZVecVectorQuery *query,
                                               const char *field_name) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->field_name) {
    zvec_free_string(query->field_name);
  }
  query->field_name = zvec_string_create(field_name);
  return ZVEC_OK;
}

const char *zvec_vector_query_get_field_name(const ZVecVectorQuery *query) {
  if (!query || !query->field_name) {
    return nullptr;
  }
  return query->field_name->data;
}

ZVecErrorCode zvec_vector_query_set_query_vector(ZVecVectorQuery *query,
                                                 const void *data,
                                                 size_t size) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->query_vector.data = (const uint8_t *)data;
  query->query_vector.length = size;
  return ZVEC_OK;
}

ZVecErrorCode zvec_vector_query_set_filter(ZVecVectorQuery *query,
                                           const char *filter) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->filter) {
    zvec_free_string(query->filter);
  }
  if (filter && strlen(filter) > 0) {
    query->filter = zvec_string_create(filter);
  } else {
    query->filter = nullptr;
  }
  return ZVEC_OK;
}

const char *zvec_vector_query_get_filter(const ZVecVectorQuery *query) {
  if (!query || !query->filter) {
    return nullptr;
  }
  return query->filter->data;
}

ZVecErrorCode zvec_vector_query_set_include_vector(ZVecVectorQuery *query,
                                                   bool include) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->include_vector = include;
  return ZVEC_OK;
}

ZVecErrorCode zvec_vector_query_set_include_doc_id(ZVecVectorQuery *query,
                                                   bool include) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->include_doc_id = include;
  return ZVEC_OK;
}

ZVecErrorCode zvec_vector_query_set_output_fields(ZVecVectorQuery *query,
                                                  const char **fields,
                                                  size_t count) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->output_fields) {
    zvec_string_array_destroy(query->output_fields);
  }
  if (fields && count > 0) {
    query->output_fields = zvec_string_array_create_from_strings(fields, count);
  } else {
    query->output_fields = nullptr;
  }
  return ZVEC_OK;
}

ZVecErrorCode zvec_vector_query_set_query_params(ZVecVectorQuery *query,
                                                 void *params) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT, "Vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  // Note: We don't delete old params here, caller should manage lifetime
  query->query_params = params;
  // Set params_type based on the type of params (caller should ensure
  // consistency) For now, we assume params is one of the known types
  if (params) {
    // We can't automatically determine the type, so we'll need to trust the
    // caller to set the correct type via a separate call if needed
    query->params_type = ZVEC_INDEX_TYPE_UNDEFINED;
  }
  return ZVEC_OK;
}

// GroupByVectorQuery functions

ZVecGroupByVectorQuery *zvec_group_by_vector_query_create(void) {
  ZVEC_TRY_RETURN_NULL(
      "Failed to create ZVecGroupByVectorQuery",
      ZVecGroupByVectorQuery *query = new ZVecGroupByVectorQuery();
      query->field_name = nullptr; query->query_vector.data = nullptr;
      query->query_vector.length = 0;
      query->query_sparse_indices.data = nullptr;
      query->query_sparse_indices.length = 0;
      query->query_sparse_values.data = nullptr;
      query->query_sparse_values.length = 0; query->filter = nullptr;
      query->include_vector = false; query->output_fields = nullptr;
      query->group_by_field_name = nullptr; query->group_count = 0;
      query->group_topk = 0; query->query_params = nullptr;
      query->params_type = ZVEC_INDEX_TYPE_UNDEFINED; return query;)
  return nullptr;
}

void zvec_group_by_vector_query_destroy(ZVecGroupByVectorQuery *query) {
  if (query) {
    if (query->field_name) {
      zvec_free_string(query->field_name);
    }
    if (query->filter) {
      zvec_free_string(query->filter);
    }
    if (query->output_fields) {
      zvec_string_array_destroy(query->output_fields);
    }
    if (query->group_by_field_name) {
      zvec_free_string(query->group_by_field_name);
    }
    if (query->query_params) {
      switch (query->params_type) {
        case ZVEC_INDEX_TYPE_HNSW:
          delete static_cast<ZVecHnswQueryParams *>(query->query_params);
          break;
        case ZVEC_INDEX_TYPE_IVF:
          delete static_cast<ZVecIVFQueryParams *>(query->query_params);
          break;
        case ZVEC_INDEX_TYPE_FLAT:
          delete static_cast<ZVecFlatQueryParams *>(query->query_params);
          break;
        default:
          delete static_cast<ZVecQueryParams *>(query->query_params);
          break;
      }
    }
    delete query;
  }
}

ZVecErrorCode zvec_group_by_vector_query_set_field_name(
    ZVecGroupByVectorQuery *query, const char *field_name) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->field_name) {
    zvec_free_string(query->field_name);
  }
  query->field_name = zvec_string_create(field_name);
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_group_by_field_name(
    ZVecGroupByVectorQuery *query, const char *field_name) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->group_by_field_name) {
    zvec_free_string(query->group_by_field_name);
  }
  query->group_by_field_name = zvec_string_create(field_name);
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_group_count(
    ZVecGroupByVectorQuery *query, uint32_t count) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->group_count = count;
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_group_topk(
    ZVecGroupByVectorQuery *query, uint32_t topk) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->group_topk = topk;
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_query_vector(
    ZVecGroupByVectorQuery *query, const void *data, size_t size) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->query_vector.data = (const uint8_t *)data;
  query->query_vector.length = size;
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_filter(
    ZVecGroupByVectorQuery *query, const char *filter) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->filter) {
    zvec_free_string(query->filter);
  }
  if (filter && strlen(filter) > 0) {
    query->filter = zvec_string_create(filter);
  } else {
    query->filter = nullptr;
  }
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_include_vector(
    ZVecGroupByVectorQuery *query, bool include) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->include_vector = include;
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_output_fields(
    ZVecGroupByVectorQuery *query, const char **fields, size_t count) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  if (query->output_fields) {
    zvec_string_array_destroy(query->output_fields);
  }
  if (fields && count > 0) {
    query->output_fields = zvec_string_array_create_from_strings(fields, count);
  } else {
    query->output_fields = nullptr;
  }
  return ZVEC_OK;
}

ZVecErrorCode zvec_group_by_vector_query_set_query_params(
    ZVecGroupByVectorQuery *query, void *params) {
  if (!query) {
    SET_LAST_ERROR(ZVEC_ERROR_INVALID_ARGUMENT,
                   "Group by vector query pointer is null");
    return ZVEC_ERROR_INVALID_ARGUMENT;
  }
  query->query_params = params;
  query->params_type = ZVEC_INDEX_TYPE_UNDEFINED;
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
        auto cpp_params = std::make_shared<zvec::InvertIndexParams>(
            index_params->invert.enable_range_optimization,
            index_params->invert.enable_extended_wildcard);
        auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
        return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_HNSW: {
  auto metric = convert_metric_type(index_params->metric_type);
  auto quantize = convert_quantize_type(index_params->quantize_type);
  auto cpp_params = std::make_shared<zvec::HnswIndexParams>(
      metric, index_params->hnsw.m, index_params->hnsw.ef_construction,
      quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_FLAT: {
  auto metric = convert_metric_type(index_params->metric_type);
  auto quantize = convert_quantize_type(index_params->quantize_type);
  auto cpp_params = std::make_shared<zvec::FlatIndexParams>(metric, quantize);
  auto status = (*coll_ptr)->CreateIndex(field_name_str, cpp_params);
  return status_to_error_code(status);
}

case ZVEC_INDEX_TYPE_IVF: {
  auto metric = convert_metric_type(index_params->metric_type);
  auto quantize = convert_quantize_type(index_params->quantize_type);
  auto cpp_params = std::make_shared<zvec::IVFIndexParams>(
      metric, index_params->ivf.n_list, index_params->ivf.n_iters,
      index_params->ivf.use_soar, quantize);
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

  // Legacy function - kept for backward compatibility, just calls
  // zvec_collection_create_index
  ZVecErrorCode zvec_collection_create_hnsw_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecIndexParams *hnsw_params) {
    if (!hnsw_params) {
      set_last_error("Invalid HNSW parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return zvec_collection_create_index(collection, field_name, hnsw_params);
  }

  ZVecErrorCode zvec_collection_create_flat_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecIndexParams *flat_params) {
    if (!flat_params) {
      set_last_error("Invalid Flat parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return zvec_collection_create_index(collection, field_name, flat_params);
  }

  ZVecErrorCode zvec_collection_create_ivf_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecIndexParams *ivf_params) {
    if (!ivf_params) {
      set_last_error("Invalid IVF parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return zvec_collection_create_index(collection, field_name, ivf_params);
  }

  ZVecErrorCode zvec_collection_create_invert_index(
      ZVecCollection *collection, const char *field_name,
      const ZVecIndexParams *invert_params) {
    if (!invert_params) {
      set_last_error("Invalid Invert parameters");
      return ZVEC_ERROR_INVALID_ARGUMENT;
    }
    return zvec_collection_create_index(collection, field_name, invert_params);
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

        zvec::DataType data_type =
            convert_data_type(zvec_field_schema_get_data_type(field_schema));
        if (data_type == zvec::DataType::UNDEFINED) {
          set_last_error("Invalid data type");
          return ZVEC_ERROR_INVALID_ARGUMENT;
        }

        std::string field_name(zvec_field_schema_get_name(field_schema));
        bool is_vector_field = check_is_vector_field(*field_schema);
        zvec::FieldSchema::Ptr schema;
        if (is_vector_field) {
          schema = std::make_shared<zvec::FieldSchema>(
              field_name, data_type,
              zvec_field_schema_get_dimension(field_schema),
              zvec_field_schema_is_nullable(field_schema));
        } else {
          schema = std::make_shared<zvec::FieldSchema>(
              field_name, data_type,
              zvec_field_schema_is_nullable(field_schema));
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
              convert_zvec_field_schema_to_internal(new_schema, schema);
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

  ZVecErrorCode zvec_collection_insert_with_results(ZVecCollection *collection,
                                                    const ZVecDoc **docs,
                                                    size_t doc_count,
                                                    ZVecWriteResult **results,
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
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code != ZVEC_OK) { return error_code; }

        return build_write_results(result.value(), pks, results, result_count);)
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

  ZVecErrorCode zvec_collection_update_with_results(ZVecCollection *collection,
                                                    const ZVecDoc **docs,
                                                    size_t doc_count,
                                                    ZVecWriteResult **results,
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
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code != ZVEC_OK) { return error_code; }

        return build_write_results(result.value(), pks, results, result_count);)
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

  ZVecErrorCode zvec_collection_upsert_with_results(ZVecCollection *collection,
                                                    const ZVecDoc **docs,
                                                    size_t doc_count,
                                                    ZVecWriteResult **results,
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
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code != ZVEC_OK) { return error_code; }

        return build_write_results(result.value(), pks, results, result_count);)
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

  ZVecErrorCode zvec_collection_delete_with_results(ZVecCollection *collection,
                                                    const char *const *pks,
                                                    size_t pk_count,
                                                    ZVecWriteResult **results,
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
        ZVecErrorCode error_code = handle_expected_result(result);

        if (error_code != ZVEC_OK) { return error_code; }

        return build_write_results(result.value(), primary_keys, results,
                                   result_count);)
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
        query->field_name
            ? std::string(query->field_name->data, query->field_name->length)
            : "";
    internal_query.filter_ =
        query->filter ? std::string(query->filter->data, query->filter->length)
                      : "";
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
    if (query->output_fields && query->output_fields->count > 0) {
      internal_query.output_fields_ = std::vector<std::string>();
      for (size_t i = 0; i < query->output_fields->count; ++i) {
        internal_query.output_fields_->emplace_back(
            query->output_fields->strings[i].data,
            query->output_fields->strings[i].length);
      }
    }
  }

  // Helper function to convert query parameters
  void convert_query_params(zvec::VectorQuery &internal_query,
                            const ZVecVectorQuery *query) {
    convert_common_query_params(internal_query, query);

    // QueryParams conversion
    if (query->query_params) {
      switch (query->params_type) {
        case ZVEC_INDEX_TYPE_HNSW: {
          auto hnsw_params =
              static_cast<ZVecHnswQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::HnswQueryParams>(
              hnsw_params->ef, hnsw_params->base.radius,
              hnsw_params->base.is_linear, hnsw_params->base.is_using_refiner);
          internal_query.query_params_ = internal_params;
          break;
        }
        case ZVEC_INDEX_TYPE_IVF: {
          auto ivf_params =
              static_cast<ZVecIVFQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::IVFQueryParams>(
              ivf_params->nprobe, ivf_params->base.is_using_refiner,
              ivf_params->scale_factor);
          internal_query.query_params_ = internal_params;
          break;
        }
        case ZVEC_INDEX_TYPE_FLAT: {
          auto flat_params =
              static_cast<ZVecFlatQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::FlatQueryParams>(
              flat_params->base.is_using_refiner, flat_params->scale_factor);
          internal_query.query_params_ = internal_params;
          break;
        }
        default: {
          auto base_params =
              static_cast<ZVecQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::QueryParams>(
              static_cast<zvec::IndexType>(base_params->index_type));
          internal_params->set_radius(base_params->radius);
          internal_params->set_is_linear(base_params->is_linear);
          internal_params->set_is_using_refiner(base_params->is_using_refiner);
          internal_query.query_params_ = internal_params;
          break;
        }
      }
    }
  }

  // Helper function to convert group by query parameters
  void convert_groupby_query_params(zvec::GroupByVectorQuery &internal_query,
                                    const ZVecGroupByVectorQuery *query) {
    internal_query.field_name_ =
        query->field_name
            ? std::string(query->field_name->data, query->field_name->length)
            : "";
    internal_query.filter_ =
        query->filter ? std::string(query->filter->data, query->filter->length)
                      : "";
    internal_query.include_vector_ = query->include_vector;
    internal_query.group_by_field_name_ =
        query->group_by_field_name
            ? std::string(query->group_by_field_name->data,
                          query->group_by_field_name->length)
            : "";
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

    if (query->output_fields && query->output_fields->count > 0) {
      if (!internal_query.output_fields_.has_value()) {
        internal_query.output_fields_ = std::vector<std::string>();
      }
      for (size_t i = 0; i < query->output_fields->count; ++i) {
        internal_query.output_fields_->push_back(
            std::string(query->output_fields->strings[i].data,
                        query->output_fields->strings[i].length));
      }
    }

    if (query->query_params) {
      switch (query->params_type) {
        case ZVEC_INDEX_TYPE_HNSW: {
          auto hnsw_params =
              static_cast<ZVecHnswQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::HnswQueryParams>(
              hnsw_params->ef, hnsw_params->base.radius,
              hnsw_params->base.is_linear, hnsw_params->base.is_using_refiner);
          internal_query.query_params_ = internal_params;
          break;
        }
        case ZVEC_INDEX_TYPE_IVF: {
          auto ivf_params =
              static_cast<ZVecIVFQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::IVFQueryParams>(
              ivf_params->nprobe, ivf_params->base.is_using_refiner,
              ivf_params->scale_factor);
          internal_query.query_params_ = internal_params;
          break;
        }
        case ZVEC_INDEX_TYPE_FLAT: {
          auto flat_params =
              static_cast<ZVecFlatQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::FlatQueryParams>(
              flat_params->base.is_using_refiner, flat_params->scale_factor);
          internal_query.query_params_ = internal_params;
          break;
        }
        default: {
          auto base_params =
              static_cast<ZVecQueryParams *>(query->query_params);
          auto internal_params = std::make_shared<zvec::QueryParams>(
              static_cast<zvec::IndexType>(base_params->index_type));
          internal_params->set_radius(base_params->radius);
          internal_params->set_is_linear(base_params->is_linear);
          internal_params->set_is_using_refiner(base_params->is_using_refiner);
          internal_query.query_params_ = internal_params;
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

        auto doc_map = result.value();
        auto schema_result = (*coll_ptr)->Schema();
        if (schema_result.has_value()) {
          normalize_nullable_fields_for_fetch(schema_result.value(), doc_map);
        } return convert_fetched_document_results(doc_map, results, doc_count);)
  }
