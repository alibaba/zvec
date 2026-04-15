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

#ifndef ZVEC_C_API_H
#define ZVEC_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// =============================================================================
// API Export Control
// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef ZVEC_BUILD_SHARED
#define ZVEC_EXPORT __declspec(dllexport)
#elif defined(ZVEC_USE_SHARED)
#define ZVEC_EXPORT __declspec(dllimport)
#else
#define ZVEC_EXPORT
#endif
#define ZVEC_CALL __cdecl
#else
#if __GNUC__ >= 4
#define ZVEC_EXPORT __attribute__((visibility("default")))
#else
#define ZVEC_EXPORT
#endif
#define ZVEC_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif


// =============================================================================
// Version Information
// =============================================================================

/**
 * @brief Get library version information
 *
 * Return format: "{base_version}[-{git_info}] (built {build_time})"
 * Example: "0.3.0-g3f8a2b1 (built 2025-05-13 10:30:45)"
 *
 * @return const char* Version string, managed internally by the library, caller
 * should not free
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_get_version(void);

/**
 * @brief Check API version compatibility
 *
 * Check if the current library version meets the specified minimum version
 * requirements Following semantic versioning specification: MAJOR.MINOR.PATCH
 *
 * @param major Required major version number
 * @param minor Required minor version number
 * @param patch Required patch version number
 * @return bool Returns true if compatible, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_check_version(int major, int minor, int patch);

/**
 * @brief Get major version number
 *
 * @return int Major version number
 */
ZVEC_EXPORT int ZVEC_CALL zvec_get_version_major(void);

/**
 * @brief Get minor version number
 *
 * @return int Minor version number
 */
ZVEC_EXPORT int ZVEC_CALL zvec_get_version_minor(void);


/**
 * @brief Get patch version number
 *
 * @return int Patch version number
 */
ZVEC_EXPORT int ZVEC_CALL zvec_get_version_patch(void);


// =============================================================================
// Error Code Definitions
// =============================================================================

/**
 * @brief ZVec C API error code enumeration
 */
typedef enum {
  ZVEC_OK = 0,                        /**< Success */
  ZVEC_ERROR_NOT_FOUND = 1,           /**< Resource not found */
  ZVEC_ERROR_ALREADY_EXISTS = 2,      /**< Resource already exists */
  ZVEC_ERROR_INVALID_ARGUMENT = 3,    /**< Invalid argument */
  ZVEC_ERROR_PERMISSION_DENIED = 4,   /**< Permission denied */
  ZVEC_ERROR_FAILED_PRECONDITION = 5, /**< Failed precondition */
  ZVEC_ERROR_RESOURCE_EXHAUSTED = 6,  /**< Resource exhausted */
  ZVEC_ERROR_UNAVAILABLE = 7,         /**< Unavailable */
  ZVEC_ERROR_INTERNAL_ERROR = 8,      /**< Internal error */
  ZVEC_ERROR_NOT_SUPPORTED = 9,       /**< Unsupported operation */
  ZVEC_ERROR_UNKNOWN = 10             /**< Unknown error */
} zvec_error_code_t;

/**
 * @brief Error details structure
 */
typedef struct {
  zvec_error_code_t code; /**< Error code */
  const char *message;    /**< Error message */
  const char *file;       /**< File where error occurred */
  int line;               /**< Line number where error occurred */
  const char *function;   /**< Function where error occurred */
} zvec_error_details_t;

/**
 * @brief Get detailed information of the last error
 * @param[out] error_details Pointer to error details structure
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_get_last_error_details(zvec_error_details_t *error_details);

/**
 * @brief Get last error message
 * @param[out] error_msg Returned error message string (needs to be freed by
 * calling free)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_get_last_error(char **error_msg);

/**
 * @brief Clear error status
 */
ZVEC_EXPORT void ZVEC_CALL zvec_clear_error(void);


// =============================================================================
// Basic Data Structures
// =============================================================================

/**
 * @brief String view structure (does not own memory)
 */
typedef struct {
  const char *data; /**< String data pointer */
  size_t length;    /**< String length */
} zvec_string_view_t;

/**
 * @brief Mutable string structure (owns memory)
 */
typedef struct {
  char *data;      /**< String data pointer */
  size_t length;   /**< String length */
  size_t capacity; /**< Allocated capacity */
} zvec_string_t;

/**
 * @brief String array structure
 */
typedef struct {
  zvec_string_t *strings; /**< String array */
  size_t count;           /**< String count */
} zvec_string_array_t;

/**
 * @brief Float array structure
 */
typedef struct {
  const float *data;
  size_t length;
} zvec_float_array_t;

/**
 * @brief Integer array structure
 */
typedef struct {
  const int64_t *data;
  size_t length;
} zvec_int64_array_t;

/**
 * @brief Byte array structure
 */
typedef struct {
  const uint8_t *data; /**< Byte data pointer */
  size_t length;       /**< Array length */
} zvec_byte_array_t;

/**
 * @brief Mutable byte array structure
 */
typedef struct {
  uint8_t *data;   /**< Byte data pointer */
  size_t length;   /**< Current length */
  size_t capacity; /**< Allocated capacity */
} zvec_mutable_byte_array_t;

// =============================================================================
// String management functions
// =============================================================================

/**
 * @brief Create string from C string
 * @param str C string
 * @return zvec_string_t* Pointer to the newly created string
 */
ZVEC_EXPORT zvec_string_t *ZVEC_CALL zvec_string_create(const char *str);

/**
 * @brief Create string from string view
 *
 * Creates a new zvec_string_t by copying data from a zvec_string_view_t.
 * The created string owns its memory and must be freed with zvec_free_string().
 *
 * @param view Pointer to source string view (must not be NULL)
 * @return zvec_string_t* New string instance on success, NULL on error
 * @note Caller is responsible for freeing the returned string
 */
ZVEC_EXPORT zvec_string_t *ZVEC_CALL
zvec_string_create_from_view(const zvec_string_view_t *view);

/**
 * @brief Create binary-safe string from raw data
 *
 * Creates a new zvec_string_t from raw binary data that may contain null bytes.
 * Unlike zvec_string_create(), this function takes explicit length parameter
 * and doesn't rely on null-termination.
 * The created string owns its memory and must be freed with zvec_free_string().
 *
 * @param data Raw binary data pointer (must not be NULL)
 * @param length Length of data in bytes
 * @return zvec_string_t* New string instance on success, NULL on error
 * @note Caller is responsible for freeing the returned string
 * @note This function is suitable for binary data containing null bytes
 */
ZVEC_EXPORT zvec_string_t *ZVEC_CALL zvec_bin_create(const uint8_t *data,
                                                     size_t length);

/**
 * @brief Copy string
 *
 * Creates a new zvec_string_t by copying an existing string.
 * The created string owns its memory and must be freed with zvec_free_string().
 *
 * @param str Pointer to source string (must not be NULL)
 * @return zvec_string_t* New string instance on success, NULL on error
 * @note Caller is responsible for freeing the returned string
 */
ZVEC_EXPORT zvec_string_t *ZVEC_CALL zvec_string_copy(const zvec_string_t *str);

/**
 * @brief Get C string from zvec_string_t
 * @param str zvec_string_t pointer
 * @return const char* C string
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_string_c_str(const zvec_string_t *str);

/**
 * @brief Get string length
 * @param str zvec_string_t pointer
 * @return size_t String length
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_string_length(const zvec_string_t *str);

/**
 * @brief Compare two strings
 * @param str1 First string
 * @param str2 Second string
 * @return int Comparison result (-1, 0, or 1)
 */
ZVEC_EXPORT int ZVEC_CALL zvec_string_compare(const zvec_string_t *str1,
                                              const zvec_string_t *str2);

/**
 * @brief Free string memory
 * @param str String pointer to free
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free_string(zvec_string_t *str);


// =============================================================================
// Array Memory management functions
// =============================================================================

/**
 * @brief Create a new string array
 * @param count Initial number of strings to allocate space for
 * @return Pointer to the newly created string array, or NULL on failure
 */
ZVEC_EXPORT zvec_string_array_t *ZVEC_CALL
zvec_string_array_create(size_t count);

/**
 * @brief Add a string to the string array at specified index
 * @param array String array pointer
 * @param idx Index position where the string should be added
 * @param str Null-terminated C string to add
 */
ZVEC_EXPORT void ZVEC_CALL zvec_string_array_add(zvec_string_array_t *array,
                                                 size_t idx, const char *str);

/**
 * @brief Destroy string array and free all associated memory
 * @param array String array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_string_array_destroy(zvec_string_array_t *array);

/**
 * @brief Create a new mutable byte array
 * @param capacity Initial capacity in bytes
 * @return Pointer to the newly created byte array, or NULL on failure
 */
ZVEC_EXPORT zvec_mutable_byte_array_t *ZVEC_CALL
zvec_byte_array_create(size_t capacity);


/**
 * @brief Destroy byte array and free all associated memory
 * @param array Byte array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_byte_array_destroy(zvec_mutable_byte_array_t *array);

/**
 * @brief Create a new float array
 * @param count Number of floats to allocate space for
 * @return Pointer to the newly created float array, or NULL on failure
 */
ZVEC_EXPORT zvec_float_array_t *ZVEC_CALL zvec_float_array_create(size_t count);

/**
 * @brief Destroy float array and free all associated memory
 * @param array Float array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_float_array_destroy(zvec_float_array_t *array);

/**
 * @brief Create a new int64 array
 * @param count Number of int64 values to allocate space for
 * @return Pointer to the newly created int64 array, or NULL on failure
 */
ZVEC_EXPORT zvec_int64_array_t *ZVEC_CALL zvec_int64_array_create(size_t count);

/**
 * @brief Destroy int64 array and free all associated memory
 * @param array Int64 array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_int64_array_destroy(zvec_int64_array_t *array);

/**
 * @brief Release uint8_t array memory
 *
 * @param array uint8_t array pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free_uint8_array(uint8_t *array);

/**
 * @brief Allocate memory within the zvec library
 *
 * Use this function instead of malloc to ensure memory is managed consistently
 * within the library. All memory allocated with zvec_malloc should be freed
 * with zvec_free.
 *
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 *
 * @see zvec_free
 */
ZVEC_EXPORT void *ZVEC_CALL zvec_malloc(size_t size);

/**
 * @brief Free memory allocated by zvec_malloc or zvec library functions
 *
 * Use this function instead of free to ensure memory is managed consistently
 * within the library. This should be used to free any memory allocated with
 * zvec_malloc or returned by library functions that document they return
 * library-allocated memory.
 *
 * @param ptr Pointer to memory to free (can be NULL)
 *
 * @see zvec_malloc
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free(void *ptr);


// =============================================================================
// Configuration and Options Structures
// =============================================================================

/**
 * @brief Log level enumeration
 */
typedef enum {
  ZVEC_LOG_LEVEL_DEBUG = 0,
  ZVEC_LOG_LEVEL_INFO = 1,
  ZVEC_LOG_LEVEL_WARN = 2,
  ZVEC_LOG_LEVEL_ERROR = 3,
  ZVEC_LOG_LEVEL_FATAL = 4
} zvec_log_level_t;

/**
 * @brief Log type enumeration
 */
typedef enum {
  ZVEC_LOG_TYPE_CONSOLE = 0,
  ZVEC_LOG_TYPE_FILE = 1
} zvec_log_type_t;

// =============================================================================
// Configuration Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Log configuration base type (opaque pointer)
 * Corresponds to zvec::GlobalConfig::LogConfig
 * Use factory functions to create specific log configurations:
 * - zvec_config_log_create_console() for console logging
 * - zvec_config_log_create_file() for file logging
 */
typedef struct zvec_log_config_t zvec_log_config_t;

/**
 * @brief Configuration data (opaque pointer)
 * Corresponds to zvec::GlobalConfig::ConfigData
 * Use zvec_config_data_create() to create and
 * zvec_config_data_destroy() to destroy
 */
typedef struct zvec_config_data_t zvec_config_data_t;

// =============================================================================
// Log Configuration Management Functions
// =============================================================================

/**
 * @brief Create console log configuration
 * @param level Log level
 * @return zvec_log_config_t* Pointer to the newly created log configuration
 */
ZVEC_EXPORT zvec_log_config_t *ZVEC_CALL
zvec_config_log_create_console(zvec_log_level_t level);

/**
 * @brief Create file log configuration
 * @param level Log level
 * @param dir Log directory
 * @param basename Log file base name
 * @param file_size Log file size (MB)
 * @param overdue_days Log expiration days
 * @return zvec_log_config_t* Pointer to the newly created log configuration
 */
ZVEC_EXPORT zvec_log_config_t *ZVEC_CALL zvec_config_log_create_file(
    zvec_log_level_t level, const char *dir, const char *basename,
    uint32_t file_size, uint32_t overdue_days);

/**
 * @brief Destroy log configuration
 * @param config Log configuration pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_config_log_destroy(zvec_log_config_t *config);

/**
 * @brief Get log level from log config
 * @param config Log configuration pointer
 * @return zvec_log_level_t Log level
 */
ZVEC_EXPORT zvec_log_level_t ZVEC_CALL
zvec_config_log_get_level(const zvec_log_config_t *config);

/**
 * @brief Set log level in log config
 * @param config Log configuration pointer
 * @param level Log level
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_log_set_level(zvec_log_config_t *config, zvec_log_level_t level);

/**
 * @brief Check if log config is file type
 * @param config Log configuration pointer
 * @return true if file type, false if console type
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_config_log_is_file_type(const zvec_log_config_t *config);

/**
 * @brief Get log directory from file log config
 * @param config Log configuration pointer (must be file type)
 * @return const char* Log directory (owned by config, do not free)
 * @note Only valid for file log config, returns NULL for console config
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_config_log_get_dir(const zvec_log_config_t *config);

/**
 * @brief Set log directory in file log config
 * @param config Log configuration pointer (must be file type)
 * @param dir Log directory
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_log_set_dir(zvec_log_config_t *config, const char *dir);

/**
 * @brief Get log file basename from file log config
 * @param config Log configuration pointer (must be file type)
 * @return const char* Log file basename (owned by config, do not free)
 * @note Only valid for file log config, returns NULL for console config
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_config_log_get_basename(const zvec_log_config_t *config);

/**
 * @brief Set log file basename in file log config
 * @param config Log configuration pointer (must be file type)
 * @param basename Log file basename
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_log_set_basename(zvec_log_config_t *config, const char *basename);

/**
 * @brief Get log file size from file log config
 * @param config Log configuration pointer (must be file type)
 * @return uint32_t Log file size in MB
 * @note Only valid for file log config, returns 0 for console config
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_log_get_file_size(const zvec_log_config_t *config);

/**
 * @brief Set log file size in file log config
 * @param config Log configuration pointer (must be file type)
 * @param file_size Log file size in MB
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_log_set_file_size(zvec_log_config_t *config, uint32_t file_size);

/**
 * @brief Get log overdue days from file log config
 * @param config Log configuration pointer (must be file type)
 * @return uint32_t Log overdue days
 * @note Only valid for file log config, returns 0 for console config
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_log_get_overdue_days(const zvec_log_config_t *config);

/**
 * @brief Set log overdue days in file log config
 * @param config Log configuration pointer (must be file type)
 * @param days Log overdue days
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_log_set_overdue_days(zvec_log_config_t *config, uint32_t days);

// =============================================================================
// Configuration Data Management Functions
// =============================================================================

/**
 * @brief Create configuration data
 * @return zvec_config_data_t* Pointer to the newly created configuration data
 */
ZVEC_EXPORT zvec_config_data_t *ZVEC_CALL zvec_config_data_create(void);

/**
 * @brief Destroy configuration data
 * @param config Configuration data pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_config_data_destroy(zvec_config_data_t *config);

/**
 * @brief Set memory limit in configuration data
 * @param config Configuration data pointer
 * @param memory_limit_bytes Memory limit in bytes
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_config_data_set_memory_limit(
    zvec_config_data_t *config, uint64_t memory_limit_bytes);

/**
 * @brief Get memory limit from configuration data
 * @param config Configuration data pointer
 * @return uint64_t Memory limit in bytes
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_config_data_get_memory_limit(const zvec_config_data_t *config);

/**
 * @brief Set log configuration in configuration data
 * @param config Configuration data pointer
 * @param log_config Log configuration pointer (ownership is transferred to
 * config, do not free separately)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_config_data_set_log_config(
    zvec_config_data_t *config, zvec_log_config_t *log_config);

/**
 * @brief Get log type from configuration data
 * @param config Configuration data pointer
 * @return zvec_log_type_t Log type
 */
ZVEC_EXPORT zvec_log_type_t ZVEC_CALL
zvec_config_data_get_log_type(const zvec_config_data_t *config);

/**
 * @brief Set query thread count in configuration data
 * @param config Configuration data pointer
 * @param thread_count Query thread count
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_config_data_set_query_thread_count(
    zvec_config_data_t *config, uint32_t thread_count);

/**
 * @brief Get query thread count from configuration data
 * @param config Configuration data pointer
 * @return uint32_t Query thread count
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_data_get_query_thread_count(const zvec_config_data_t *config);

/**
 * @brief Set invert to forward scan ratio in configuration data
 * @param config Configuration data pointer
 * @param ratio Invert to forward scan ratio
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_data_set_invert_to_forward_scan_ratio(zvec_config_data_t *config,
                                                  float ratio);

/**
 * @brief Get invert to forward scan ratio from configuration data
 * @param config Configuration data pointer
 * @return float Invert to forward scan ratio
 */
ZVEC_EXPORT float ZVEC_CALL zvec_config_data_get_invert_to_forward_scan_ratio(
    const zvec_config_data_t *config);

/**
 * @brief Set brute force by keys ratio in configuration data
 * @param config Configuration data pointer
 * @param ratio Brute force by keys ratio
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_data_set_brute_force_by_keys_ratio(zvec_config_data_t *config,
                                               float ratio);

/**
 * @brief Get brute force by keys ratio from configuration data
 * @param config Configuration data pointer
 * @return float Brute force by keys ratio
 */
ZVEC_EXPORT float ZVEC_CALL zvec_config_data_get_brute_force_by_keys_ratio(
    const zvec_config_data_t *config);

/**
 * @brief Set optimize thread count in configuration data
 * @param config Configuration data pointer
 * @param thread_count Optimize thread count
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_config_data_set_optimize_thread_count(zvec_config_data_t *config,
                                           uint32_t thread_count);

/**
 * @brief Get optimize thread count from configuration data
 * @param config Configuration data pointer
 * @return uint32_t Optimize thread count
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_data_get_optimize_thread_count(const zvec_config_data_t *config);

// =============================================================================
// Initialization and Cleanup Interface
// =============================================================================

/**
 * @brief Initialize ZVec library
 * @param config Configuration data (optional, NULL means using default
 * configuration)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_initialize(const zvec_config_data_t *config);

/**
 * @brief Clean up ZVec library resources
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_shutdown(void);

/**
 * @brief Check if library is initialized
 * @return true if initialized, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_is_initialized(void);

// =============================================================================
// Data Type Enumerations
// =============================================================================

/**
 * @brief Data type codes (must match zvec::DataType in zvec/db/type.h)
 *
 * These are defined as uint32_t constants instead of C enums to ensure
 * consistent binary representation across C and C++ boundaries.
 */
typedef uint32_t zvec_data_type_t;
#define ZVEC_DATA_TYPE_UNDEFINED 0
#define ZVEC_DATA_TYPE_BINARY 1
#define ZVEC_DATA_TYPE_STRING 2
#define ZVEC_DATA_TYPE_BOOL 3
#define ZVEC_DATA_TYPE_INT32 4
#define ZVEC_DATA_TYPE_INT64 5
#define ZVEC_DATA_TYPE_UINT32 6
#define ZVEC_DATA_TYPE_UINT64 7
#define ZVEC_DATA_TYPE_FLOAT 8
#define ZVEC_DATA_TYPE_DOUBLE 9
#define ZVEC_DATA_TYPE_VECTOR_BINARY32 20
#define ZVEC_DATA_TYPE_VECTOR_BINARY64 21
#define ZVEC_DATA_TYPE_VECTOR_FP16 22
#define ZVEC_DATA_TYPE_VECTOR_FP32 23
#define ZVEC_DATA_TYPE_VECTOR_FP64 24
#define ZVEC_DATA_TYPE_VECTOR_INT4 25
#define ZVEC_DATA_TYPE_VECTOR_INT8 26
#define ZVEC_DATA_TYPE_VECTOR_INT16 27
#define ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 30
#define ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 31
#define ZVEC_DATA_TYPE_ARRAY_BINARY 40
#define ZVEC_DATA_TYPE_ARRAY_STRING 41
#define ZVEC_DATA_TYPE_ARRAY_BOOL 42
#define ZVEC_DATA_TYPE_ARRAY_INT32 43
#define ZVEC_DATA_TYPE_ARRAY_INT64 44
#define ZVEC_DATA_TYPE_ARRAY_UINT32 45
#define ZVEC_DATA_TYPE_ARRAY_UINT64 46
#define ZVEC_DATA_TYPE_ARRAY_FLOAT 47
#define ZVEC_DATA_TYPE_ARRAY_DOUBLE 48

/**
 * @brief Index type codes (must match zvec::IndexType in zvec/db/type.h)
 *
 * These are defined as uint32_t constants instead of C enums to ensure
 * consistent binary representation across C and C++ boundaries.
 */
typedef uint32_t zvec_index_type_t;
#define ZVEC_INDEX_TYPE_UNDEFINED 0
#define ZVEC_INDEX_TYPE_HNSW 1
#define ZVEC_INDEX_TYPE_IVF 2
#define ZVEC_INDEX_TYPE_FLAT 3
#define ZVEC_INDEX_TYPE_INVERT 10

/**
 * @brief Distance metric type codes (must match zvec::MetricType in
 * zvec/db/type.h)
 *
 * These are defined as uint32_t constants instead of C enums to ensure
 * consistent binary representation across C and C++ boundaries.
 */
typedef uint32_t zvec_metric_type_t;
#define ZVEC_METRIC_TYPE_UNDEFINED 0
#define ZVEC_METRIC_TYPE_L2 1
#define ZVEC_METRIC_TYPE_IP 2
#define ZVEC_METRIC_TYPE_COSINE 3
#define ZVEC_METRIC_TYPE_MIPSL2 4

/**
 * @brief Quantization type codes (must match zvec::QuantizeType in
 * zvec/db/type.h)
 *
 * These are defined as uint32_t constants instead of C enums to ensure
 * consistent binary representation across C and C++ boundaries.
 */
typedef uint32_t zvec_quantize_type_t;
#define ZVEC_QUANTIZE_TYPE_UNDEFINED 0
#define ZVEC_QUANTIZE_TYPE_FP16 1
#define ZVEC_QUANTIZE_TYPE_INT8 2
#define ZVEC_QUANTIZE_TYPE_INT4 3

// =============================================================================
// Collection Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Collection handle (opaque pointer)
 *
 * Internally maps to std::shared_ptr<zvec::Collection>*.
 * Managed by zvec_collection_create/open() and zvec_collection_close().
 */
typedef struct zvec_collection_t zvec_collection_t;

// =============================================================================
// Index Parameters Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Index parameters (opaque pointer)
 *
 * Use zvec_index_params_create() to create and zvec_index_params_destroy() to
 * destroy. Specific parameters are set via type-specific setter functions.
 */
typedef struct zvec_index_params_t zvec_index_params_t;

// =============================================================================
// Field Schema Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Field schema handle (opaque pointer)
 *
 * Internally maps to zvec::FieldSchema* (raw pointer).
 * Created by zvec_field_schema_create() and destroyed by
 * zvec_field_schema_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct zvec_field_schema_t zvec_field_schema_t;


// =============================================================================
// Index Parameters Functions
// =============================================================================

/**
 * @brief Create index parameters
 * @param index_type Index type
 * @return Pointer to newly created zvec_index_params_t, or NULL on error
 */
ZVEC_EXPORT zvec_index_params_t *ZVEC_CALL
zvec_index_params_create(zvec_index_type_t index_type);

/**
 * @brief Destroy index parameters
 * @param params Index parameters to destroy
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_index_params_destroy(zvec_index_params_t *params);

// =============================================================================
// Collection Schema Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Get index type
 * @param params Index parameters (must not be NULL)
 * @return Index type
 */
ZVEC_EXPORT zvec_index_type_t ZVEC_CALL
zvec_index_params_get_type(const zvec_index_params_t *params);

/**
 * @brief Set metric type (for vector indexes)
 * @param params Index parameters (must be vector index type)
 * @param metric_type Metric type
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_set_metric_type(
    zvec_index_params_t *params, zvec_metric_type_t metric_type);

/**
 * @brief Get metric type
 * @param params Index parameters (must not be NULL)
 * @return Metric type
 */
ZVEC_EXPORT zvec_metric_type_t ZVEC_CALL
zvec_index_params_get_metric_type(const zvec_index_params_t *params);

/**
 * @brief Set quantize type (for vector indexes)
 * @param params Index parameters (must be vector index type)
 * @param quantize_type Quantize type
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_set_quantize_type(
    zvec_index_params_t *params, zvec_quantize_type_t quantize_type);

/**
 * @brief Get quantize type
 * @param params Index parameters (must not be NULL)
 * @return Quantize type
 */
ZVEC_EXPORT zvec_quantize_type_t ZVEC_CALL
zvec_index_params_get_quantize_type(const zvec_index_params_t *params);

/**
 * @brief Set HNSW specific parameters
 * @param params Index parameters (must be HNSW type)
 * @param m Graph connectivity parameter
 * @param ef_construction Construction exploration factor
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_set_hnsw_params(
    zvec_index_params_t *params, int m, int ef_construction);

/**
 * @brief Get HNSW m parameter
 * @param params Index parameters (must not be NULL)
 * @return m parameter
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_index_params_get_hnsw_m(const zvec_index_params_t *params);

/**
 * @brief Get HNSW ef_construction parameter
 * @param params Index parameters (must not be NULL)
 * @return ef_construction parameter
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_index_params_get_hnsw_ef_construction(const zvec_index_params_t *params);

/**
 * @brief Set IVF specific parameters
 * @param params Index parameters (must be IVF type)
 * @param n_list Number of cluster centers
 * @param n_iters Number of iterations
 * @param use_soar Whether to use SOAR algorithm
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_set_ivf_params(
    zvec_index_params_t *params, int n_list, int n_iters, bool use_soar);

/**
 * @brief Get IVF parameters (all at once)
 * @param params Index parameters (must not be NULL)
 * @param out_n_list Output parameter for n_list
 * @param out_n_iters Output parameter for n_iters
 * @param out_use_soar Output parameter for use_soar
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_get_ivf_params(
    const zvec_index_params_t *params, int *out_n_list, int *out_n_iters,
    bool *out_use_soar);

/**
 * @brief Get invert index parameters (all at once)
 * @param params Index parameters (must not be NULL)
 * @param out_enable_range_opt Output parameter for enable_range_optimization
 * @param out_enable_wildcard Output parameter for enable_extended_wildcard
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_get_invert_params(
    const zvec_index_params_t *params, bool *out_enable_range_opt,
    bool *out_enable_wildcard);

/**
 * @brief Set invert index specific parameters
 * @param params Index parameters (must be INVERT type)
 * @param enable_range_opt Whether to enable range optimization
 * @param enable_wildcard Whether to enable extended wildcard
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_index_params_set_invert_params(
    zvec_index_params_t *params, bool enable_range_opt, bool enable_wildcard);

// =============================================================================
// Query Parameters Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief HNSW query parameters handle (opaque pointer)
 *
 * Internally maps to zvec::HnswQueryParams* (raw pointer).
 * Created by zvec_query_params_hnsw_create() and destroyed by
 * zvec_query_params_hnsw_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct zvec_hnsw_query_params_t zvec_hnsw_query_params_t;

/**
 * @brief IVF query parameters handle (opaque pointer)
 *
 * Internally maps to zvec::IVFQueryParams* (raw pointer).
 * Created by zvec_query_params_ivf_create() and destroyed by
 * zvec_query_params_ivf_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct zvec_ivf_query_params_t zvec_ivf_query_params_t;

/**
 * @brief Flat query parameters handle (opaque pointer)
 *
 * Internally maps to zvec::FlatQueryParams* (raw pointer).
 * Created by zvec_query_params_flat_create() and destroyed by
 * zvec_query_params_flat_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct zvec_flat_query_params_t zvec_flat_query_params_t;


// =============================================================================
// Query Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Vector query structure (opaque pointer)
 * Aligned with zvec::VectorQuery
 * Use zvec_vector_query_create() to create and zvec_vector_query_destroy() to
 * destroy
 */
typedef struct zvec_vector_query_t zvec_vector_query_t;

/**
 * @brief Grouped vector query structure (opaque pointer)
 * Aligned with zvec::GroupByVectorQuery
 * Use zvec_group_by_vector_query_create() to create and
 * zvec_group_by_vector_query_destroy() to destroy
 */
typedef struct zvec_group_by_vector_query_t zvec_group_by_vector_query_t;


// =============================================================================
// Query Parameters Management Functions
// =============================================================================

// -----------------------------------------------------------------------------
// zvec_hnsw_query_params_t (HNSW Query Parameters)
// -----------------------------------------------------------------------------

/**
 * @brief Create HNSW query parameters
 * @param ef Exploration factor during search (default: 40)
 * @param radius Search radius (default: 0.0)
 * @param is_linear Whether linear search (default: false)
 * @param is_using_refiner Whether using refiner (default: false)
 * @return zvec_hnsw_query_params_t* Pointer to the newly created HNSW query
 * parameters
 */
ZVEC_EXPORT zvec_hnsw_query_params_t *ZVEC_CALL zvec_query_params_hnsw_create(
    int ef, float radius, bool is_linear, bool is_using_refiner);

/**
 * @brief Destroy HNSW query parameters
 * @param params HNSW query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_hnsw_destroy(zvec_hnsw_query_params_t *params);

/**
 * @brief Set exploration factor
 * @param params HNSW query parameters pointer
 * @param ef Exploration factor
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_query_params_hnsw_set_ef(zvec_hnsw_query_params_t *params, int ef);

/**
 * @brief Get exploration factor
 * @param params HNSW query parameters pointer
 * @return int Exploration factor
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_query_params_hnsw_get_ef(const zvec_hnsw_query_params_t *params);

/**
 * @brief Set search radius (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @param radius Search radius
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_hnsw_set_radius(
    zvec_hnsw_query_params_t *params, float radius);

/**
 * @brief Get search radius (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @return float Search radius
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_hnsw_get_radius(const zvec_hnsw_query_params_t *params);

/**
 * @brief Set linear search mode (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @param is_linear Whether linear search
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_hnsw_set_is_linear(
    zvec_hnsw_query_params_t *params, bool is_linear);

/**
 * @brief Get linear search mode (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @return bool Whether linear search
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_hnsw_get_is_linear(const zvec_hnsw_query_params_t *params);

/**
 * @brief Set whether to use refiner (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_query_params_hnsw_set_is_using_refiner(zvec_hnsw_query_params_t *params,
                                            bool is_using_refiner);

/**
 * @brief Get whether to use refiner (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @return bool Whether to use refiner
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_query_params_hnsw_get_is_using_refiner(
    const zvec_hnsw_query_params_t *params);

// -----------------------------------------------------------------------------
// zvec_ivf_query_params_t (IVF Query Parameters)
// -----------------------------------------------------------------------------

/**
 * @brief Create IVF query parameters
 * @param nprobe Number of clusters to probe (default: 10)
 * @param is_using_refiner Whether using refiner (default: false)
 * @param scale_factor Scale factor (default: 10.0)
 * @return zvec_ivf_query_params_t* Pointer to the newly created IVF query
 * parameters
 */
ZVEC_EXPORT zvec_ivf_query_params_t *ZVEC_CALL zvec_query_params_ivf_create(
    int nprobe, bool is_using_refiner, float scale_factor);

/**
 * @brief Destroy IVF query parameters
 * @param params IVF query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_ivf_destroy(zvec_ivf_query_params_t *params);

/**
 * @brief Set number of probe clusters
 * @param params IVF query parameters pointer
 * @param nprobe Number of probe clusters
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_query_params_ivf_set_nprobe(zvec_ivf_query_params_t *params, int nprobe);

/**
 * @brief Get number of probe clusters
 * @param params IVF query parameters pointer
 * @return int Number of probe clusters
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_query_params_ivf_get_nprobe(const zvec_ivf_query_params_t *params);

/**
 * @brief Set scale factor
 * @param params IVF query parameters pointer
 * @param scale_factor Scale factor
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_ivf_set_scale_factor(
    zvec_ivf_query_params_t *params, float scale_factor);

/**
 * @brief Get scale factor
 * @param params IVF query parameters pointer
 * @return float Scale factor
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_ivf_get_scale_factor(const zvec_ivf_query_params_t *params);

/**
 * @brief Set search radius (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @param radius Search radius
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_query_params_ivf_set_radius(zvec_ivf_query_params_t *params, float radius);

/**
 * @brief Get search radius (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @return float Search radius
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_ivf_get_radius(const zvec_ivf_query_params_t *params);

/**
 * @brief Set linear search mode (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @param is_linear Whether linear search
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_ivf_set_is_linear(
    zvec_ivf_query_params_t *params, bool is_linear);

/**
 * @brief Get linear search mode (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @return bool Whether linear search
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_ivf_get_is_linear(const zvec_ivf_query_params_t *params);

/**
 * @brief Set whether to use refiner (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_query_params_ivf_set_is_using_refiner(zvec_ivf_query_params_t *params,
                                           bool is_using_refiner);

/**
 * @brief Get whether to use refiner (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @return bool Whether to use refiner
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_query_params_ivf_get_is_using_refiner(
    const zvec_ivf_query_params_t *params);

// -----------------------------------------------------------------------------
// zvec_flat_query_params_t (Flat Query Parameters)
// -----------------------------------------------------------------------------

/**
 * @brief Create Flat query parameters
 * @param is_using_refiner Whether using refiner (default: false)
 * @param scale_factor Scale factor (default: 10.0)
 * @return zvec_flat_query_params_t* Pointer to the newly created Flat query
 * parameters
 */
ZVEC_EXPORT zvec_flat_query_params_t *ZVEC_CALL
zvec_query_params_flat_create(bool is_using_refiner, float scale_factor);

/**
 * @brief Destroy Flat query parameters
 * @param params Flat query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_flat_destroy(zvec_flat_query_params_t *params);

/**
 * @brief Set scale factor
 * @param params Flat query parameters pointer
 * @param scale_factor Scale factor
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_flat_set_scale_factor(
    zvec_flat_query_params_t *params, float scale_factor);

/**
 * @brief Get scale factor
 * @param params Flat query parameters pointer
 * @return float Scale factor
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_flat_get_scale_factor(const zvec_flat_query_params_t *params);

/**
 * @brief Set search radius (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @param radius Search radius
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_flat_set_radius(
    zvec_flat_query_params_t *params, float radius);

/**
 * @brief Get search radius (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @return float Search radius
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_flat_get_radius(const zvec_flat_query_params_t *params);

/**
 * @brief Set linear search mode (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @param is_linear Whether linear search
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_query_params_flat_set_is_linear(
    zvec_flat_query_params_t *params, bool is_linear);

/**
 * @brief Get linear search mode (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @return bool Whether linear search
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_flat_get_is_linear(const zvec_flat_query_params_t *params);

/**
 * @brief Set whether to use refiner (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_query_params_flat_set_is_using_refiner(zvec_flat_query_params_t *params,
                                            bool is_using_refiner);

/**
 * @brief Get whether to use refiner (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @return bool Whether to use refiner
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_query_params_flat_get_is_using_refiner(
    const zvec_flat_query_params_t *params);

// -----------------------------------------------------------------------------
// zvec_vector_query_t (Vector Query)
// -----------------------------------------------------------------------------

/**
 * @brief Create vector query
 * @return zvec_vector_query_t* Pointer to the newly created vector query
 */
ZVEC_EXPORT zvec_vector_query_t *ZVEC_CALL zvec_vector_query_create(void);

/**
 * @brief Destroy vector query
 * @param query Vector query pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_vector_query_destroy(zvec_vector_query_t *query);

/**
 * @brief Set topk (number of results to return)
 * @param query Vector query pointer
 * @param topk Number of results
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_vector_query_set_topk(zvec_vector_query_t *query, int topk);

/**
 * @brief Get topk
 * @param query Vector query pointer
 * @return int Number of results
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_vector_query_get_topk(const zvec_vector_query_t *query);

/**
 * @brief Set field name
 * @param query Vector query pointer
 * @param field_name Field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_set_field_name(
    zvec_vector_query_t *query, const char *field_name);

/**
 * @brief Get field name
 * @param query Vector query pointer
 * @return const char* Field name (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_vector_query_get_field_name(const zvec_vector_query_t *query);

/**
 * @brief Set query vector data
 * @param query Vector query pointer
 * @param data Vector data pointer
 * @param size Data size in bytes
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_set_query_vector(
    zvec_vector_query_t *query, const void *data, size_t size);

/**
 * @brief Set filter expression
 * @param query Vector query pointer
 * @param filter Filter expression string
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_vector_query_set_filter(zvec_vector_query_t *query, const char *filter);

/**
 * @brief Get filter expression
 * @param query Vector query pointer
 * @return const char* Filter expression (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_vector_query_get_filter(const zvec_vector_query_t *query);

/**
 * @brief Set whether to include vector data in results
 * @param query Vector query pointer
 * @param include Whether to include vector
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_vector_query_set_include_vector(zvec_vector_query_t *query, bool include);

/**
 * @brief Get whether to include vector data in results
 * @param query Vector query pointer
 * @return bool Whether to include vector
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_vector_query_get_include_vector(const zvec_vector_query_t *query);

/**
 * @brief Set whether to include doc ID in results
 * @param query Vector query pointer
 * @param include Whether to include doc ID
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_vector_query_set_include_doc_id(zvec_vector_query_t *query, bool include);

/**
 * @brief Get whether to include doc ID in results
 * @param query Vector query pointer
 * @return bool Whether to include doc ID
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_vector_query_get_include_doc_id(const zvec_vector_query_t *query);

/**
 * @brief Set output fields
 * @param query Vector query pointer
 * @param fields Array of field names
 * @param count Number of fields
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_set_output_fields(
    zvec_vector_query_t *query, const char **fields, size_t count);

/**
 * @brief Get output fields
 * @param query Vector query pointer
 * @param[out] fields Output array of field names (allocated by library)
 * @param[out] count Number of fields
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual string pointers
 *       are owned by the query and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_get_output_fields(
    const zvec_vector_query_t *query, const char ***fields, size_t *count);

/**
 * @brief Set query parameters (takes ownership)
 * @param query Vector query pointer
 * @param params Query parameters pointer (type-specific:
 * zvec_hnsw_query_params_t*, etc.)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_vector_query_set_query_params(zvec_vector_query_t *query, void *params);

/**
 * @brief Set HNSW query parameters (takes ownership)
 * @param query Vector query pointer
 * @param hnsw_params HNSW query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_set_hnsw_params(
    zvec_vector_query_t *query, zvec_hnsw_query_params_t *hnsw_params);

/**
 * @brief Set IVF query parameters (takes ownership)
 * @param query Vector query pointer
 * @param ivf_params IVF query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_set_ivf_params(
    zvec_vector_query_t *query, zvec_ivf_query_params_t *ivf_params);

/**
 * @brief Set Flat query parameters (takes ownership)
 * @param query Vector query pointer
 * @param flat_params Flat query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_vector_query_set_flat_params(
    zvec_vector_query_t *query, zvec_flat_query_params_t *flat_params);

// -----------------------------------------------------------------------------
// zvec_group_by_vector_query_t (Group By Vector Query)
// -----------------------------------------------------------------------------

/**
 * @brief Create group by vector query
 * @return zvec_group_by_vector_query_t* Pointer to the newly created group by
 * vector query
 */
ZVEC_EXPORT zvec_group_by_vector_query_t *ZVEC_CALL
zvec_group_by_vector_query_create(void);

/**
 * @brief Destroy group by vector query
 * @param query Group by vector query pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_group_by_vector_query_destroy(zvec_group_by_vector_query_t *query);

/**
 * @brief Set field name
 * @param query Group by vector query pointer
 * @param field_name Field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_field_name(zvec_group_by_vector_query_t *query,
                                          const char *field_name);

/**
 * @brief Get field name
 * @param query Group by vector query pointer
 * @return const char* Field name (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_group_by_vector_query_get_field_name(
    const zvec_group_by_vector_query_t *query);

/**
 * @brief Set group by field name
 * @param query Group by vector query pointer
 * @param field_name Group by field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_group_by_field_name(
    zvec_group_by_vector_query_t *query, const char *field_name);

/**
 * @brief Get group by field name
 * @param query Group by vector query pointer
 * @return const char* Group by field name (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_group_by_vector_query_get_group_by_field_name(
    const zvec_group_by_vector_query_t *query);

/**
 * @brief Set group count
 * @param query Group by vector query pointer
 * @param count Number of groups
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_group_count(zvec_group_by_vector_query_t *query,
                                           uint32_t count);

/**
 * @brief Get group count
 * @param query Group by vector query pointer
 * @return uint32_t Number of groups
 */
ZVEC_EXPORT uint32_t ZVEC_CALL zvec_group_by_vector_query_get_group_count(
    const zvec_group_by_vector_query_t *query);

/**
 * @brief Set group topk
 * @param query Group by vector query pointer
 * @param topk Number of results per group
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_group_topk(zvec_group_by_vector_query_t *query,
                                          uint32_t topk);

/**
 * @brief Get group topk
 * @param query Group by vector query pointer
 * @return uint32_t Number of results per group
 */
ZVEC_EXPORT uint32_t ZVEC_CALL zvec_group_by_vector_query_get_group_topk(
    const zvec_group_by_vector_query_t *query);

/**
 * @brief Set query vector data
 * @param query Group by vector query pointer
 * @param data Vector data pointer
 * @param size Data size in bytes
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_query_vector(zvec_group_by_vector_query_t *query,
                                            const void *data, size_t size);

/**
 * @brief Set filter expression
 * @param query Group by vector query pointer
 * @param filter Filter expression string
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_group_by_vector_query_set_filter(
    zvec_group_by_vector_query_t *query, const char *filter);

/**
 * @brief Get filter expression
 * @param query Group by vector query pointer
 * @return const char* Filter expression (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_group_by_vector_query_get_filter(
    const zvec_group_by_vector_query_t *query);

/**
 * @brief Set whether to include vector data in results
 * @param query Group by vector query pointer
 * @param include Whether to include vectors
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_include_vector(
    zvec_group_by_vector_query_t *query, bool include);

/**
 * @brief Get whether to include vector data in results
 * @param query Group by vector query pointer
 * @return bool Whether to include vectors
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_group_by_vector_query_get_include_vector(
    const zvec_group_by_vector_query_t *query);

/**
 * @brief Set output fields
 * @param query Group by vector query pointer
 * @param fields Array of field names
 * @param count Number of fields
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_output_fields(
    zvec_group_by_vector_query_t *query, const char **fields, size_t count);

/**
 * @brief Get output fields
 * @param query Group by vector query pointer
 * @param[out] fields Output array of field names (allocated by library)
 * @param[out] count Number of fields
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual string pointers
 *       are owned by the query and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_get_output_fields(
    zvec_group_by_vector_query_t *query, const char ***fields, size_t *count);

/**
 * @brief Set query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param params Query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_query_params(zvec_group_by_vector_query_t *query,
                                            void *params);

/**
 * @brief Set HNSW query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param hnsw_params HNSW query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_hnsw_params(
    zvec_group_by_vector_query_t *query, zvec_hnsw_query_params_t *hnsw_params);

/**
 * @brief Set IVF query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param ivf_params IVF query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_ivf_params(zvec_group_by_vector_query_t *query,
                                          zvec_ivf_query_params_t *ivf_params);

/**
 * @brief Set Flat query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param flat_params Flat query parameters pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_group_by_vector_query_set_flat_params(
    zvec_group_by_vector_query_t *query, zvec_flat_query_params_t *flat_params);

// =============================================================================
// Collection Options and Statistics (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Collection options (opaque pointer)
 * Use zvec_collection_options_create() to create and
 * zvec_collection_options_destroy() to destroy
 */
typedef struct zvec_collection_options_t zvec_collection_options_t;

/**
 * @brief Collection statistics (opaque pointer)
 * Use zvec_collection_stats_get functions to access fields
 */
typedef struct zvec_collection_stats_t zvec_collection_stats_t;

// =============================================================================
// Collection Options Management Functions
// =============================================================================

/**
 * @brief Create collection options
 * @return zvec_collection_options_t* Pointer to the newly created collection
 * options
 */
ZVEC_EXPORT zvec_collection_options_t *ZVEC_CALL
zvec_collection_options_create(void);

/**
 * @brief Destroy collection options
 * @param options Collection options pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_options_destroy(zvec_collection_options_t *options);

/**
 * @brief Set whether to enable memory mapping
 * @param options Collection options pointer
 * @param enable Whether to enable mmap
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_options_set_enable_mmap(
    zvec_collection_options_t *options, bool enable);

/**
 * @brief Get whether to enable memory mapping
 * @param options Collection options pointer
 * @return bool Whether mmap is enabled
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_collection_options_get_enable_mmap(
    const zvec_collection_options_t *options);

/**
 * @brief Set maximum buffer size
 * @param options Collection options pointer
 * @param size Maximum buffer size in bytes
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_options_set_max_buffer_size(zvec_collection_options_t *options,
                                            size_t size);

/**
 * @brief Get maximum buffer size
 * @param options Collection options pointer
 * @return size_t Maximum buffer size in bytes
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_collection_options_get_max_buffer_size(
    const zvec_collection_options_t *options);

/**
 * @brief Set whether read-only mode
 * @param options Collection options pointer
 * @param read_only Whether read-only
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_options_set_read_only(
    zvec_collection_options_t *options, bool read_only);

/**
 * @brief Get whether read-only mode
 * @param options Collection options pointer
 * @return bool Whether read-only mode
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_collection_options_get_read_only(const zvec_collection_options_t *options);

// =============================================================================
// Collection Statistics Management Functions
// =============================================================================

/**
 * @brief Get document count from collection stats
 * @param stats Collection statistics pointer
 * @return uint64_t Document count
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_collection_stats_get_doc_count(const zvec_collection_stats_t *stats);

/**
 * @brief Get index count from collection stats
 * @param stats Collection statistics pointer
 * @return size_t Number of indexes
 */
ZVEC_EXPORT size_t ZVEC_CALL
zvec_collection_stats_get_index_count(const zvec_collection_stats_t *stats);

/**
 * @brief Get index name at specified index
 * @param stats Collection statistics pointer
 * @param index Index of the index name
 * @return const char* Index name (owned by stats, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_collection_stats_get_index_name(
    const zvec_collection_stats_t *stats, size_t index);

/**
 * @brief Get index completeness at specified index
 * @param stats Collection statistics pointer
 * @param index Index of the completeness value
 * @return float Index completeness
 */
ZVEC_EXPORT float ZVEC_CALL zvec_collection_stats_get_index_completeness(
    const zvec_collection_stats_t *stats, size_t index);

/**
 * @brief Create field schema
 * @param name Field name
 * @param data_type Data type
 * @param nullable Whether nullable
 * @param dimension Vector dimension
 * @return zvec_field_schema_t* Pointer to the newly created field schema
 */
ZVEC_EXPORT zvec_field_schema_t *ZVEC_CALL
zvec_field_schema_create(const char *name, zvec_data_type_t data_type,
                         bool nullable, uint32_t dimension);

/**
 * @brief Destroy field schema
 * @param schema Field schema pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_field_schema_destroy(zvec_field_schema_t *schema);

/**
 * @brief Set field name
 * @param schema Field schema pointer
 * @param name New field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_field_schema_set_name(zvec_field_schema_t *schema, const char *name);

/**
 * @brief Get field name
 * @param schema Field schema pointer (must not be NULL)
 * @return const char* Field name (owned by schema, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_field_schema_get_name(const zvec_field_schema_t *schema);

/**
 * @brief Get field data type
 * @param schema Field schema pointer (must not be NULL)
 * @return zvec_data_type_t Data type
 */
ZVEC_EXPORT zvec_data_type_t ZVEC_CALL
zvec_field_schema_get_data_type(const zvec_field_schema_t *schema);

/**
 * @brief Set field data type
 * @param schema Field schema pointer
 * @param data_type New data type
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_field_schema_set_data_type(
    zvec_field_schema_t *schema, zvec_data_type_t data_type);

/**
 * @brief Get element data type for array fields
 * @param schema Field schema pointer (must not be NULL)
 * @return zvec_data_type_t Element data type, or original type if not array
 */
ZVEC_EXPORT zvec_data_type_t ZVEC_CALL
zvec_field_schema_get_element_data_type(const zvec_field_schema_t *schema);

/**
 * @brief Get element data size for the field
 * @param schema Field schema pointer (must not be NULL)
 * @return size_t Element data size in bytes, or 0 for variable-size types
 */
ZVEC_EXPORT size_t ZVEC_CALL
zvec_field_schema_get_element_data_size(const zvec_field_schema_t *schema);

/**
 * @brief Check if field is a vector field (dense or sparse)
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if vector field, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_vector_field(const zvec_field_schema_t *schema);

/**
 * @brief Check if field is a dense vector field
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if dense vector field, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_dense_vector(const zvec_field_schema_t *schema);

/**
 * @brief Check if field is a sparse vector field
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if sparse vector field, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_sparse_vector(const zvec_field_schema_t *schema);

/**
 * @brief Check if field is nullable
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if nullable, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_nullable(const zvec_field_schema_t *schema);

/**
 * @brief Set field nullable
 * @param schema Field schema pointer
 * @param nullable Whether nullable
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_field_schema_set_nullable(zvec_field_schema_t *schema, bool nullable);

/**
 * @brief Check if field has inverted index (for scalar fields)
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if has inverted index, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_has_invert_index(const zvec_field_schema_t *schema);

/**
 * @brief Check if field is an array type
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if array type, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_array_type(const zvec_field_schema_t *schema);

/**
 * @brief Get field dimension (for vector fields)
 * @param schema Field schema pointer (must not be NULL)
 * @return uint32_t Dimension value
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_field_schema_get_dimension(const zvec_field_schema_t *schema);

/**
 * @brief Set field dimension (for vector fields)
 * @param schema Field schema pointer
 * @param dimension Dimension value
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_field_schema_set_dimension(
    zvec_field_schema_t *schema, uint32_t dimension);

/**
 * @brief Get index type of the field
 * @param schema Field schema pointer (must not be NULL)
 * @return zvec_index_type_t Index type, ZVEC_INDEX_TYPE_UNDEFINED if no index
 */
ZVEC_EXPORT zvec_index_type_t ZVEC_CALL
zvec_field_schema_get_index_type(const zvec_field_schema_t *schema);

/**
 * @brief Check if field has index
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if field has index, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_has_index(const zvec_field_schema_t *schema);

/**
 * @brief Get index params of the field (returns pointer owned by the field
 * schema, do not destroy. Pointer becomes invalid if schema is modified or
 * destroyed)
 * @param schema Field schema pointer (must not be NULL)
 * @return zvec_index_params_t* Index params pointer, NULL if no index
 */
ZVEC_EXPORT const zvec_index_params_t *ZVEC_CALL
zvec_field_schema_get_index_params(const zvec_field_schema_t *schema);

/**
 * @brief Set index parameters for field
 * @param schema Field schema pointer
 * @param index_params Index parameters pointer (deep-copied internally, caller
 *                     retains ownership and must call zvec_index_params_destroy
 *                     after the call)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_field_schema_set_index_params(
    zvec_field_schema_t *schema, const zvec_index_params_t *index_params);

/**
 * @brief Validate field schema
 * @param schema Field schema pointer
 * @param[out] error_msg Error message (needs to be freed by calling
 * zvec_free_string)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_field_schema_validate(
    const zvec_field_schema_t *schema, zvec_string_t **error_msg);

// =============================================================================
// Collection Schema Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Collection schema handle (opaque pointer)
 *
 * Internally maps to zvec::CollectionSchema* (raw pointer).
 * Created by zvec_collection_schema_create() and destroyed by
 * zvec_collection_schema_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct zvec_collection_schema_t zvec_collection_schema_t;

/**
 * @brief Create collection schema
 * @param name Collection name
 * @return zvec_collection_schema_t* Pointer to the newly created collection
 * schema
 */
ZVEC_EXPORT zvec_collection_schema_t *ZVEC_CALL
zvec_collection_schema_create(const char *name);

/**
 * @brief Destroy collection schema
 * @param schema Collection schema pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_schema_destroy(zvec_collection_schema_t *schema);

/**
 * @brief Get collection schema name
 * @param schema Collection schema pointer (must not be NULL)
 * @return const char* Collection name string pointer
 *
 * @note Returns a pointer to internal memory. Caller does NOT own the memory
 *       and should NOT free it. The pointer is valid as long as the schema
 *       exists.
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_collection_schema_get_name(const zvec_collection_schema_t *schema);

/**
 * @brief Set collection schema name
 * @param schema Collection schema pointer
 * @param name New collection name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_set_name(
    zvec_collection_schema_t *schema, const char *name);

/**
 * @brief Add field to collection schema
 * @param schema Collection schema pointer
 * @param field Field schema pointer (will be cloned, caller retains ownership)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_add_field(
    zvec_collection_schema_t *schema, const zvec_field_schema_t *field);

/**
 * @brief Alter field schema
 * @param schema Collection schema pointer
 * @param field_name Name of field to alter
 * @param new_field New field schema with updated properties
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_alter_field(
    zvec_collection_schema_t *schema, const char *field_name,
    const zvec_field_schema_t *new_field);

/**
 * @brief Drop field from schema
 * @param schema Collection schema pointer
 * @param field_name Field name to drop
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_drop_field(
    zvec_collection_schema_t *schema, const char *field_name);

/**
 * @brief Check if field exists in schema
 * @param schema Collection schema pointer
 * @param field_name Field name to check
 * @return true if field exists, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_collection_schema_has_field(
    const zvec_collection_schema_t *schema, const char *field_name);

/**
 * @brief Get field by name
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return zvec_field_schema_t* Field schema pointer (non-owning, do not
 * destroy), NULL if not found
 */
ZVEC_EXPORT zvec_field_schema_t *ZVEC_CALL zvec_collection_schema_get_field(
    const zvec_collection_schema_t *schema, const char *field_name);

/**
 * @brief Get forward (scalar) field by name
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return zvec_field_schema_t* Field schema pointer (non-owning, do not
 * destroy), NULL if not found or not scalar
 */
ZVEC_EXPORT zvec_field_schema_t *ZVEC_CALL
zvec_collection_schema_get_forward_field(const zvec_collection_schema_t *schema,
                                         const char *field_name);

/**
 * @brief Get vector field by name
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return zvec_field_schema_t* Field schema pointer (non-owning, do not
 * destroy), NULL if not found or not vector
 */
ZVEC_EXPORT zvec_field_schema_t *ZVEC_CALL
zvec_collection_schema_get_vector_field(const zvec_collection_schema_t *schema,
                                        const char *field_name);

/**
 * @brief Get all forward (scalar) fields
 * @param schema Collection schema pointer
 * @param[out] fields Receives a newly allocated array of pointers to field
 *             schemas. The array is allocated by the library and should be
 *             freed using zvec_free() when no longer needed.
 * @param[out] count Number of fields
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual field pointers
 *       are owned by the schema and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_get_forward_fields(
    const zvec_collection_schema_t *schema, zvec_field_schema_t ***fields,
    size_t *count);

/**
 * @brief Get all forward fields with index
 * @param schema Collection schema pointer
 * @param[out] fields Receives a newly allocated array of pointers to field
 *             schemas. The array is allocated by the library and should be
 *             freed using zvec_free() when no longer needed.
 * @param[out] count Number of fields
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual field pointers
 *       are owned by the schema and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_get_forward_fields_with_index(
    const zvec_collection_schema_t *schema, zvec_field_schema_t ***fields,
    size_t *count);

/**
 * @brief Get all forward (scalar) field names
 * @param schema Collection schema pointer
 * @param[out] names Output array of field names (allocated by library)
 * @param[out] count Number of field names
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual string pointers
 *       are owned by the schema and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_get_forward_field_names(
    const zvec_collection_schema_t *schema, const char ***names, size_t *count);

/**
 * @brief Get all forward field names with index
 * @param schema Collection schema pointer
 * @param[out] names Output array of field names (allocated by library)
 * @param[out] count Number of field names
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual string pointers
 *       are owned by the schema and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_get_forward_field_names_with_index(
    const zvec_collection_schema_t *schema, const char ***names, size_t *count);

/**
 * @brief Get all field names
 * @param schema Collection schema pointer
 * @param[out] names Output array of field names (allocated by library)
 * @param[out] count Number of field names
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual string pointers
 *       are owned by the schema and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_get_all_field_names(
    const zvec_collection_schema_t *schema, const char ***names, size_t *count);

/**
 * @brief Get all vector fields
 * @param schema Collection schema pointer
 * @param[out] fields Receives a newly allocated array of pointers to field
 *             schemas. The array is allocated by the library and should be
 *             freed using zvec_free() when no longer needed.
 * @param[out] count Number of fields
 * @return zvec_error_code_t Error code
 *
 * @note The returned array is allocated by the library and should be freed
 *       using zvec_free() when no longer needed. The individual field pointers
 *       are owned by the schema and must NOT be freed.
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_get_vector_fields(const zvec_collection_schema_t *schema,
                                         zvec_field_schema_t ***fields,
                                         size_t *count);

/**
 * @brief Get maximum document count per segment of collection schema
 *
 * @param schema Collection schema pointer
 * @return uint64_t Maximum document count per segment
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_collection_schema_get_max_doc_count_per_segment(
    const zvec_collection_schema_t *schema);

/**
 * @brief Set maximum document count per segment
 * @param schema Collection schema pointer
 * @param max_doc_count Maximum document count
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_schema_set_max_doc_count_per_segment(
    zvec_collection_schema_t *schema, uint64_t max_doc_count);

/**
 * @brief Validate collection schema
 * @param schema Collection schema pointer
 * @param[out] error_msg Error message (needs to be freed by calling
 * zvec_free_string)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_validate(
    const zvec_collection_schema_t *schema, zvec_string_t **error_msg);

/**
 * @brief Add index to field
 * @param schema Collection schema pointer
 * @param field_name Field name to add index to
 * @param index_params Index parameters
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_add_index(
    zvec_collection_schema_t *schema, const char *field_name,
    const zvec_index_params_t *index_params);

/**
 * @brief Drop index from field
 * @param schema Collection schema pointer
 * @param field_name Field name to drop index from
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_schema_drop_index(
    zvec_collection_schema_t *schema, const char *field_name);

/**
 * @brief Check if field has index
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return true if field has index, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_collection_schema_has_index(
    const zvec_collection_schema_t *schema, const char *field_name);

// =============================================================================
// Collection Management Functions
// =============================================================================

/**
 * @brief Create and open collection
 * @param path Collection path
 * @param schema Collection schema pointer
 * @param options Collection options pointer (NULL uses default options)
 * @param[out] collection Returned collection handle
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_create_and_open(
    const char *path, const zvec_collection_schema_t *schema,
    const zvec_collection_options_t *options, zvec_collection_t **collection);


/**
 * @brief Open existing collection
 * @param path Collection path
 * @param options Collection options pointer (NULL uses default options)
 * @param[out] collection Returned collection handle
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_open(const char *path, const zvec_collection_options_t *options,
                     zvec_collection_t **collection);

/**
 * @brief Close collection
 * @param collection Collection handle
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_close(zvec_collection_t *collection);

/**
 * @brief Destroy collection
 *
 * @param collection Collection handle
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_destroy(zvec_collection_t *collection);

/**
 * @brief Flush collection data to disk
 * @param collection Collection handle
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_flush(zvec_collection_t *collection);

/**
 * @brief Get collection schema
 * @param collection Collection handle
 * @param[out] schema
 * Returned collection schema pointer (needs to be freed by calling
 * zvec_collection_schema_destroy)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_get_schema(
    const zvec_collection_t *collection, zvec_collection_schema_t **schema);

/**
 * @brief Get collection options
 * @param collection Collection handle
 * @param[out] options
 * Returned collection options pointer (needs to be freed by calling
 * zvec_collection_options_destroy)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_get_options(
    const zvec_collection_t *collection, zvec_collection_options_t **options);

/**
 * @brief Get collection statistics
 * @param collection Collection handle
 * @param[out] stats
 * Returned statistics pointer (needs to be freed by calling
 * zvec_collection_stats_destroy)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_get_stats(
    const zvec_collection_t *collection, zvec_collection_stats_t **stats);

/**
 * @brief Destroy collection statistics
 * @param stats Statistics pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_stats_destroy(zvec_collection_stats_t *stats);

/**
 * @brief Free field schema memory
 *
 * @param field_schema Field schema pointer to be freed
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_free_field_schema(zvec_field_schema_t *field_schema);

// =============================================================================
// Index Management Interface
// =============================================================================

/**
 * @brief Create index for a collection field
 *
 * @param collection Collection handle
 * @param field_name Field name to create index on
 * @param index_params Index parameters. The function will make an internal copy
 *                     of the parameters, so the caller retains ownership and
 *                     should call zvec_index_params_destroy() after the call.
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_create_index(
    zvec_collection_t *collection, const char *field_name,
    const zvec_index_params_t *index_params);

/**
 * @brief Drop index
 * @param collection Collection handle
 * @param field_name Field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_drop_index(
    zvec_collection_t *collection, const char *field_name);

/**
 * @brief Optimize collection (rebuild indexes, merge segments, etc.)
 * @param collection Collection handle
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_collection_optimize(zvec_collection_t *collection);

// =============================================================================
// Column Management Interface (DDL)
// =============================================================================

/**
 * @brief Add column
 * @param collection Collection handle
 * @param field_schema Field schema pointer (deep-copied internally, caller
 *                     retains ownership and must call zvec_field_schema_destroy
 *                     after the call)
 * @param expression Default value expression (can be NULL)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_add_column(
    zvec_collection_t *collection, const zvec_field_schema_t *field_schema,
    const char *expression);

/**
 * @brief Drop column
 * @param collection Collection handle
 * @param column_name Field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_drop_column(
    zvec_collection_t *collection, const char *column_name);

/**
 * @brief Alter column
 * @param collection Collection handle
 * @param column_name Column/field name to alter
 * @param new_name New field name (can be NULL to indicate no renaming)
 * @param new_schema New field schema (can be NULL to indicate no schema
 * modification). The schema is deep-copied internally, caller retains
 *               ownership and must call zvec_field_schema_destroy after the
 * call.
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_alter_column(
    zvec_collection_t *collection, const char *column_name,
    const char *new_name, const zvec_field_schema_t *new_schema);

/**
 * @brief Document structure (opaque pointer mode)
 * Internal implementation details are not visible to the outside, and
 * operations are performed through API functions
 */
typedef struct zvec_doc_t zvec_doc_t;

/**
 * @brief Per-document status returned by detailed DML APIs.
 * @note Uses ordered style: result index corresponds to input document index.
 *       Caller should access pk by index from the original input array.
 */
typedef struct {
  zvec_error_code_t code; /**< Per-document status code */
  const char *message;    /**< Per-document status message (allocated by API) */
} zvec_write_result_t;

// =============================================================================
// Data Manipulation Interface (DML)
// =============================================================================

/**
 * @brief Insert documents into collection
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] success_count Number of successfully inserted documents
 * @param[out] error_count Number of failed insertions
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_insert(
    zvec_collection_t *collection, const zvec_doc_t **docs, size_t doc_count,
    size_t *success_count, size_t *error_count);

/**
 * @brief Insert documents and return per-document statuses.
 *
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] results Per-document result array (free with
 * zvec_write_results_free)
 * @param[out] result_count Number of result entries
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_insert_with_results(
    zvec_collection_t *collection, const zvec_doc_t **docs, size_t doc_count,
    zvec_write_result_t **results, size_t *result_count);

/**
 * @brief Update documents in collection
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] success_count Number of successfully updated documents
 * @param[out] error_count Number of failed updates
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_update(
    zvec_collection_t *collection, const zvec_doc_t **docs, size_t doc_count,
    size_t *success_count, size_t *error_count);

/**
 * @brief Update documents and return per-document statuses.
 *
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] results Per-document result array (free with
 * zvec_write_results_free)
 * @param[out] result_count Number of result entries
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_update_with_results(
    zvec_collection_t *collection, const zvec_doc_t **docs, size_t doc_count,
    zvec_write_result_t **results, size_t *result_count);

/**
 * @brief Insert or update documents in collection (upsert operation)
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] success_count Number of successful operations
 * @param[out] error_count Number of failed operations
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_upsert(
    zvec_collection_t *collection, const zvec_doc_t **docs, size_t doc_count,
    size_t *success_count, size_t *error_count);

/**
 * @brief Upsert documents and return per-document statuses.
 *
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] results Per-document result array (free with
 * zvec_write_results_free)
 * @param[out] result_count Number of result entries
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_upsert_with_results(
    zvec_collection_t *collection, const zvec_doc_t **docs, size_t doc_count,
    zvec_write_result_t **results, size_t *result_count);

/**
 * @brief Delete documents from collection
 * @param collection Collection handle
 * @param pks Primary key array
 * @param pk_count Primary key count
 * @param[out] success_count Number of successfully deleted documents
 * @param[out] error_count Number of failed deletions
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_delete(
    zvec_collection_t *collection, const char *const *pks, size_t pk_count,
    size_t *success_count, size_t *error_count);

/**
 * @brief Delete documents by PK and return per-document statuses.
 *
 * @param collection Collection handle
 * @param pks Primary key array
 * @param pk_count Primary key count
 * @param[out] results Per-document result array (free with
 * zvec_write_results_free)
 * @param[out] result_count Number of result entries
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_delete_with_results(
    zvec_collection_t *collection, const char *const *pks, size_t pk_count,
    zvec_write_result_t **results, size_t *result_count);

/**
 * @brief Free result arrays returned by detailed DML APIs.
 *
 * @param results Result array pointer
 * @param result_count Number of entries in result array
 */
ZVEC_EXPORT void ZVEC_CALL zvec_write_results_free(zvec_write_result_t *results,
                                                   size_t result_count);

/**
 * @brief Delete documents by filter condition
 * @param collection Collection handle
 * @param filter Filter expression
 * @param[out] deleted_count Number of deleted documents
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_delete_by_filter(
    zvec_collection_t *collection, const char *filter);

// =============================================================================
// Data Query Interface (DQL)
// =============================================================================

/**
 * @brief Vector similarity search
 * @param collection Collection handle
 * @param query Query parameters pointer
 * @param[out] results Returned document array (needs to be freed by calling
 * zvec_docs_free)
 * @param[out] result_count Number of returned results
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_query(
    const zvec_collection_t *collection, const zvec_vector_query_t *query,
    zvec_doc_t ***results, size_t *result_count);

/**
 * @brief Fetch documents by primary keys
 * @param collection Collection handle
 * @param primary_keys Primary key array
 * @param count Number of primary keys
 * @param[out] documents Returned document array (needs to be freed by calling
 * zvec_docs_free)
 * @param[out] found_count Number of found documents
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_collection_fetch(
    zvec_collection_t *collection, const char *const *primary_keys,
    size_t count, zvec_doc_t ***documents, size_t *found_count);

// =============================================================================
// Document Related Structures
// =============================================================================

/**
 * @brief Document field value union
 */
typedef union {
  bool bool_value;
  int32_t int32_value;
  int64_t int64_value;
  uint32_t uint32_value;
  uint64_t uint64_value;
  float float_value;
  double double_value;
  zvec_string_t string_value;
  zvec_float_array_t vector_value;
  zvec_byte_array_t binary_value; /**< Binary data value */
} zvec_field_value_t;

/**
 * @brief Document field structure
 */
typedef struct {
  zvec_string_t name;          ///< Field name
  zvec_data_type_t data_type;  ///< Data type
  zvec_field_value_t value;    ///< Field value
} zvec_doc_field_t;

/**
 * @brief Document operator enumeration
 */
typedef enum {
  ZVEC_DOC_OP_INSERT = 0,  ///< Insert operation
  ZVEC_DOC_OP_UPDATE = 1,  ///< Update operation
  ZVEC_DOC_OP_UPSERT = 2,  ///< Insert or update operation
  ZVEC_DOC_OP_DELETE = 3   ///< Delete operation
} zvec_doc_operator_t;

// =============================================================================
// Data Manipulation Interface (DML)
// =============================================================================

/**
 * @brief Create a new document object
 *
 * @return zvec_doc_t* Pointer to the newly created document object, returns
 * NULL on failure
 */
ZVEC_EXPORT zvec_doc_t *ZVEC_CALL zvec_doc_create(void);

/**
 * @brief Destroy the document object and release all resources
 *
 * @param doc Pointer to the document object
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_destroy(zvec_doc_t *doc);

/**
 * @brief Clear the document object
 *
 * @param doc Pointer to the document object
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_clear(zvec_doc_t *doc);

/**
 * @brief Add field to document by value
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @param data_type Data type
 * @param value Value pointer
 * @param value_size Value size
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_doc_add_field_by_value(
    zvec_doc_t *doc, const char *field_name, zvec_data_type_t data_type,
    const void *value, size_t value_size);

/**
 * @brief Add field to document by structure
 *
 * @param doc Document object pointer
 * @param field Field structure pointer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_add_field_by_struct(zvec_doc_t *doc, const zvec_doc_field_t *field);

/**
 * @brief Remove field from document
 *
 * @param doc Document structure pointer
 * @param field_name Field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_remove_field(zvec_doc_t *doc, const char *field_name);

/**
 * @brief Batch release document array
 *
 * @param documents Document pointer array
 * @param count Document count
 */
ZVEC_EXPORT void ZVEC_CALL zvec_docs_free(zvec_doc_t **documents, size_t count);

/**
 * @brief Set document primary key
 *
 * @param doc Pointer to the document structure
 * @param pk Primary key string
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_pk(zvec_doc_t *doc, const char *pk);

/**
 * @brief Set document ID
 *
 * @param doc Document structure pointer
 * @param doc_id Document ID
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_doc_id(zvec_doc_t *doc,
                                               uint64_t doc_id);

/**
 * @brief Set document score
 *
 * @param doc Document structure pointer
 * @param score Score value
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_score(zvec_doc_t *doc, float score);

/**
 * @brief Set document operator
 *
 * @param doc Document structure pointer
 * @param op Operator
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_operator(zvec_doc_t *doc,
                                                 zvec_doc_operator_t op);

/**
 * @brief Explicitly mark a document field as null.
 *
 * @param doc Document structure pointer
 * @param field_name Field name
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_set_field_null(zvec_doc_t *doc, const char *field_name);

/**
 * @brief Get document ID
 *
 * @param doc Document structure pointer
 * @return uint64_t Document ID
 */
ZVEC_EXPORT uint64_t ZVEC_CALL zvec_doc_get_doc_id(const zvec_doc_t *doc);

/**
 * @brief Get document score
 *
 * @param doc Document structure pointer
 * @return float Score value
 */
ZVEC_EXPORT float ZVEC_CALL zvec_doc_get_score(const zvec_doc_t *doc);

/**
 * @brief Get document operator
 *
 * @param doc Document structure pointer
 * @return zvec_doc_operator_t Operator
 */
ZVEC_EXPORT zvec_doc_operator_t ZVEC_CALL
zvec_doc_get_operator(const zvec_doc_t *doc);

/**
 * @brief Get document field count
 *
 * @param doc Document structure pointer
 * @return size_t Field count
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_doc_get_field_count(const zvec_doc_t *doc);

/**
 * @brief Get document primary key pointer (no copy)
 *
 * @param doc Document object pointer
 * @return const char* Primary key string pointer, returns NULL if not set
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_doc_get_pk_pointer(const zvec_doc_t *doc);

/**
 * @brief Get document primary key copy (needs manual release)
 *
 * @param doc Document object pointer
 * @return const char* Primary key string copy, needs to call zvec_free() to
 *         release, returns NULL if not set
 *
 * @note The returned string is allocated by the library and must be freed
 *       using zvec_free() when no longer needed.
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_doc_get_pk_copy(const zvec_doc_t *doc);

/**
 * @brief Get field value (basic type returned directly)
 *
 * Supports basic numeric data types: BOOL, INT32, INT64, UINT32, UINT64,
 * FLOAT, DOUBLE. The value is copied directly into the provided buffer.
 * For STRING, BINARY, and VECTOR types, use zvec_doc_get_field_value_copy
 * or zvec_doc_get_field_value_pointer instead.
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @param field_type Field type (must be a basic numeric type)
 * @param value_buffer Output buffer to receive the value
 * @param buffer_size Size of the output buffer
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_doc_get_field_value_basic(
    const zvec_doc_t *doc, const char *field_name, zvec_data_type_t field_type,
    void *value_buffer, size_t buffer_size);

/**
 * @brief Get field value copy (allocate new memory)
 *
 * Supports all data types including:
 * - Basic types: BOOL, INT32, INT64, UINT32, UINT64, FLOAT, DOUBLE
 * - String types: STRING, BINARY
 * - Vector types: VECTOR_FP32, VECTOR_FP64, VECTOR_FP16, VECTOR_INT4,
 *   VECTOR_INT8, VECTOR_INT16, VECTOR_BINARY32, VECTOR_BINARY64
 * - Sparse vector types: SPARSE_VECTOR_FP32, SPARSE_VECTOR_FP16
 * - Array types: ARRAY_STRING, ARRAY_BINARY, ARRAY_BOOL, ARRAY_INT32,
 *   ARRAY_INT64, ARRAY_UINT32, ARRAY_UINT64, ARRAY_FLOAT, ARRAY_DOUBLE
 *
 * The returned value pointer must be manually freed using appropriate
 * deallocation functions (zvec_free() for basic types and strings,
 * zvec_free_uint8_array() for binary data).
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @param field_type Field type
 * @param[out] value Returned value pointer (needs manual release)
 * @param[out] value_size Returned value size
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_doc_get_field_value_copy(
    const zvec_doc_t *doc, const char *field_name, zvec_data_type_t field_type,
    void **value, size_t *value_size);

/**
 * @brief Get field value pointer (data remains in document)
 *
 * Supports data types where direct pointer access is safe:
 * - Basic types: BOOL, INT32, INT64, UINT32, UINT64, FLOAT, DOUBLE
 * - String types: STRING (returns null-terminated C string), BINARY
 * - Vector types: VECTOR_FP32, VECTOR_FP64, VECTOR_FP16, VECTOR_INT4,
 *   VECTOR_INT8, VECTOR_INT16, VECTOR_BINARY32, VECTOR_BINARY64
 * - Array types: ARRAY_INT32, ARRAY_INT64, ARRAY_UINT32, ARRAY_UINT64,
 *   ARRAY_FLOAT, ARRAY_DOUBLE
 *
 * The returned pointer points to data within the document object and
 * does not require manual memory management. The pointer remains valid
 * as long as the document exists.
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @param field_type Field type
 * @param[out] value Returned value pointer (points to document-internal data)
 * @param[out] value_size Returned value size
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_doc_get_field_value_pointer(
    const zvec_doc_t *doc, const char *field_name, zvec_data_type_t field_type,
    const void **value, size_t *value_size);

/**
 * @brief Check if document is empty
 *
 * @param doc Document object pointer
 * @return bool Returns true if document is empty, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_is_empty(const zvec_doc_t *doc);

/**
 * @brief Check if document contains specified field
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @return bool Returns true if field exists, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_has_field(const zvec_doc_t *doc,
                                              const char *field_name);

/**
 * @brief Check if document field has value
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @return bool Returns true if field has value, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_has_field_value(const zvec_doc_t *doc,
                                                    const char *field_name);

/**
 * @brief Check if document field is null
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @return bool Returns true if field is null, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_is_field_null(const zvec_doc_t *doc,
                                                  const char *field_name);

/**
 * @brief Get all field names of document
 *
 * @param doc Document object pointer
 * @param[out] field_names
 * Returned field name array (needs to call zvec_free_str_array to release)
 * @param[out] count Returned field count
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL zvec_doc_get_field_names(
    const zvec_doc_t *doc, char ***field_names, size_t *count);

/**
 * @brief Release string array memory
 *
 * @param array String array pointer
 * @param count Array element count
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free_str_array(char **array, size_t count);

/**
 * @brief Serialize document
 *
 * @param doc Document object pointer
 * @param[out] data Returned serialized data (needs to call
 * zvec_free_uint8_array to release)
 * @param[out] size Returned data size
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_serialize(const zvec_doc_t *doc, uint8_t **data, size_t *size);

/**
 * @brief Deserialize document
 *
 * @param data Serialized data
 * @param size Data size
 * @param[out] doc Returned document object pointer (needs to call
 * zvec_doc_destroy to release)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_deserialize(const uint8_t *data, size_t size, zvec_doc_t **doc);

/**
 * @brief Merge two documents
 *
 * @param doc Target document object pointer
 * @param other Source document object pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_merge(zvec_doc_t *doc,
                                          const zvec_doc_t *other);

/**
 * @brief Get document memory usage
 *
 * @param doc Document object pointer
 * @return size_t Memory usage (bytes)
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_doc_memory_usage(const zvec_doc_t *doc);

/**
 * @brief Validate document against Schema
 *
 * @param doc Document object pointer
 * @param schema Schema object pointer
 * @param is_update Whether it's an update operation
 * @param[out] error_msg Error message (needs manual release)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_validate(const zvec_doc_t *doc, const zvec_collection_schema_t *schema,
                  bool is_update, char **error_msg);

/**
 * @brief Get detailed string representation of document
 *
 * @param doc Document object pointer
 * @param[out] detail_str Returned detailed string (needs manual release)
 * @return zvec_error_code_t Error code
 */
ZVEC_EXPORT zvec_error_code_t ZVEC_CALL
zvec_doc_to_detail_string(const zvec_doc_t *doc, char **detail_str);

/**
 * @brief Free docs array memory
 * @param docs Document array pointer
 * @param count Document count
 */
ZVEC_EXPORT void ZVEC_CALL zvec_docs_free(zvec_doc_t **docs, size_t count);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Convert error code to description string
 * @param error_code Error code
 * @return const char* Error description string
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_error_code_to_string(zvec_error_code_t error_code);

/**
 * @brief Convert data type to string
 * @param data_type Data type
 * @return const char* Data type string
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_data_type_to_string(zvec_data_type_t data_type);

/**
 * @brief Convert index type to string
 * @param index_type Index type
 * @return const char* Index type string
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_index_type_to_string(zvec_index_type_t index_type);

/**
 * @brief Convert metric type to string
 * @param metric_type Metric type
 * @return const char* Metric type string
 */
const char *zvec_metric_type_to_string(zvec_metric_type_t metric_type);

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Simplified HNSW index parameters initialization macro
 * @param _metric Distance metric type
 * @param _m Connectivity parameter
 * @param _ef_construction Exploration factor during construction
 * @param _ef_search Exploration factor during search
 * @param _quant Quantization type
 *
 * Usage example:
 * @code
 * zvec_index_params_t params = ZVEC_HNSW_PARAMS(
 *     ZVEC_METRIC_TYPE_COSINE, 16, 200, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
 * @endcode
 */
// clang-format off
#define ZVEC_HNSW_PARAMS(_metric, _m, _ef_construction, _ef_search, _quant) \
  ((zvec_index_params_t){                                                       \
    .index_type = ZVEC_INDEX_TYPE_HNSW,                                     \
    .metric_type = (_metric),                                               \
    .quantize_type = (_quant),                                              \
    .hnsw.m = (_m),                                                         \
    .hnsw.ef_construction = (_ef_construction),                             \
    .hnsw.ef_search = (_ef_search) })
// clang-format on

/**
 * @brief Simplified inverted index parameters initialization macro
 * @param range_opt Whether to enable range optimization
 * @param wildcard Whether to enable wildcard expansion
 *
 * Usage example:
 * zvec_index_params_t params = ZVEC_INVERT_PARAMS(true, false);
 */
// clang-format off
#define ZVEC_INVERT_PARAMS(_range_opt, _wildcard) \
  ((zvec_index_params_t){                               \
    .index_type = ZVEC_INDEX_TYPE_INVERT,           \
    .invert.enable_range_optimization = (_range_opt), \
    .invert.enable_extended_wildcard = (_wildcard) })
// clang-format on

/**
 * @brief Simplified Flat index parameters initialization macro
 * @param metric Distance metric type
 * @param quant Quantization type
 */
// clang-format off
#define ZVEC_FLAT_PARAMS(_metric, _quant) \
  ((zvec_index_params_t){                     \
    .index_type = ZVEC_INDEX_TYPE_FLAT,   \
    .metric_type = (_metric),             \
    .quantize_type = (_quant) })
// clang-format on

/**
 * @brief Simplified IVF index parameters initialization macro
 * @param metric Distance metric type
 * @param nlist Number of cluster centers
 * @param niters Number of iterations
 * @param soar Whether to use SOAR algorithm
 * @param nprobe Number of clusters to probe during search
 * @param quant Quantization type
 */
// clang-format off
#define ZVEC_IVF_PARAMS(_metric, _nlist, _niters, _soar, _nprobe, _quant)  \
  ((zvec_index_params_t){                                                      \
    .index_type = ZVEC_INDEX_TYPE_IVF,                                     \
    .metric_type = (_metric),                                              \
    .quantize_type = (_quant),                                             \
    .ivf.n_list = (_nlist),                                                \
    .ivf.n_iters = (_niters),                                              \
    .ivf.use_soar = (_soar),                                               \
    .ivf.n_probe = (_nprobe) })
// clang-format on

/**
 * @brief Simplified string initialization macro
 * @param str String content
 *
 * Usage example:
 * zvec_string_t name = ZVEC_STRING("my_collection");
 */
#define ZVEC_STRING(str)               \
  (zvec_string_t) {                    \
    .data = str, .length = strlen(str) \
  }

/**
 * @brief Simplified string view initialization macro
 * @param str String content
 *
 * Usage example:
 * zvec_string_view_t name = ZVEC_STRING_VIEW("my_collection");
 */
#define ZVEC_STRING_VIEW(str)          \
  (zvec_string_view_t) {               \
    .data = str, .length = strlen(str) \
  }

// Has been replaced by the new ZVEC_STRING_VIEW macro

/**
 * @brief Simplified float array initialization macro
 * @param data_ptr Float array pointer
 * @param len Array length
 *
 * Usage example:
 * float vectors[] = {0.1f, 0.2f, 0.3f};
 * zvec_float_array_t vec_array = ZVEC_FLOAT_ARRAY(vectors, 3);
 */
#define ZVEC_FLOAT_ARRAY(data_ptr, len) \
  (zvec_float_array_t) {                \
    .data = data_ptr, .length = len     \
  }

/**
 * @brief Simplified integer array initialization macro
 * @param data_ptr Integer array pointer
 * @param len Array length
 */
#define ZVEC_INT64_ARRAY(data_ptr, len) \
  (zvec_int64_array_t) {                \
    .data = data_ptr, .length = len     \
  }

/**
 * @brief Simplified document field initialization macro
 * @param name_str Field name
 * @param type Data type
 * @param value_union Field value union
 *
 * Usage example:
 * zvec_doc_field_t field = ZVEC_DOC_FIELD("id", ZVEC_DATA_TYPE_STRING,
 *     {.string_value = ZVEC_STRING("doc1")});
 */
#define ZVEC_DOC_FIELD(name_str, type, value_union)                        \
  (zvec_doc_field_t) {                                                     \
    .name = ZVEC_STRING(name_str), .data_type = type, .value = value_union \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ZVEC_C_API_H
