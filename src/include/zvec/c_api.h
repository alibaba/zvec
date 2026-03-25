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
} ZVecErrorCode;

/**
 * @brief Error details structure
 */
typedef struct {
  ZVecErrorCode code;   /**< Error code */
  const char *message;  /**< Error message */
  const char *file;     /**< File where error occurred */
  int line;             /**< Line number where error occurred */
  const char *function; /**< Function where error occurred */
} ZVecErrorDetails;

/**
 * @brief Get detailed information of the last error
 * @param[out] error_details Pointer to error details structure
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_get_last_error_details(ZVecErrorDetails *error_details);

/**
 * @brief Get last error message
 * @param[out] error_msg Returned error message string (needs to be freed by
 * calling free)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_get_last_error(char **error_msg);

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
} ZVecStringView;

/**
 * @brief Mutable string structure (owns memory)
 */
typedef struct {
  char *data;      /**< String data pointer */
  size_t length;   /**< String length */
  size_t capacity; /**< Allocated capacity */
} ZVecString;

/**
 * @brief String array structure
 */
typedef struct {
  ZVecString *strings; /**< String array */
  size_t count;        /**< String count */
} ZVecStringArray;

/**
 * @brief Float array structure
 */
typedef struct {
  const float *data;
  size_t length;
} ZVecFloatArray;

/**
 * @brief Integer array structure
 */
typedef struct {
  const int64_t *data;
  size_t length;
} ZVecInt64Array;

/**
 * @brief Byte array structure
 */
typedef struct {
  const uint8_t *data; /**< Byte data pointer */
  size_t length;       /**< Array length */
} ZVecByteArray;

/**
 * @brief Mutable byte array structure
 */
typedef struct {
  uint8_t *data;   /**< Byte data pointer */
  size_t length;   /**< Current length */
  size_t capacity; /**< Allocated capacity */
} ZVecMutableByteArray;

// =============================================================================
// String management functions
// =============================================================================

/**
 * @brief Create string from C string
 * @param str C string
 * @return ZVecString* Pointer to the newly created string
 */
ZVEC_EXPORT ZVecString *ZVEC_CALL zvec_string_create(const char *str);

/**
 * @brief Create string from string view
 *
 * Creates a new ZVecString by copying data from a ZVecStringView.
 * The created string owns its memory and must be freed with zvec_free_string().
 *
 * @param view Pointer to source string view (must not be NULL)
 * @return ZVecString* New string instance on success, NULL on error
 * @note Caller is responsible for freeing the returned string
 */
ZVEC_EXPORT ZVecString *ZVEC_CALL
zvec_string_create_from_view(const ZVecStringView *view);

/**
 * @brief Create binary-safe string from raw data
 *
 * Creates a new ZVecString from raw binary data that may contain null bytes.
 * Unlike zvec_string_create(), this function takes explicit length parameter
 * and doesn't rely on null-termination.
 * The created string owns its memory and must be freed with zvec_free_string().
 *
 * @param data Raw binary data pointer (must not be NULL)
 * @param length Length of data in bytes
 * @return ZVecString* New string instance on success, NULL on error
 * @note Caller is responsible for freeing the returned string
 * @note This function is suitable for binary data containing null bytes
 */
ZVEC_EXPORT ZVecString *ZVEC_CALL zvec_bin_create(const uint8_t *data,
                                                  size_t length);

/**
 * @brief Copy string
 *
 * Creates a new ZVecString by copying an existing string.
 * The created string owns its memory and must be freed with zvec_free_string().
 *
 * @param str Pointer to source string (must not be NULL)
 * @return ZVecString* New string instance on success, NULL on error
 * @note Caller is responsible for freeing the returned string
 */
ZVEC_EXPORT ZVecString *ZVEC_CALL zvec_string_copy(const ZVecString *str);

/**
 * @brief Get C string from ZVecString
 * @param str ZVecString pointer
 * @return const char* C string
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_string_c_str(const ZVecString *str);

/**
 * @brief Get string length
 * @param str ZVecString pointer
 * @return size_t String length
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_string_length(const ZVecString *str);

/**
 * @brief Compare two strings
 * @param str1 First string
 * @param str2 Second string
 * @return int Comparison result (-1, 0, or 1)
 */
ZVEC_EXPORT int ZVEC_CALL zvec_string_compare(const ZVecString *str1,
                                              const ZVecString *str2);

/**
 * @brief Free string memory
 * @param str String pointer to free
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free_string(ZVecString *str);


// =============================================================================
// Array Memory management functions
// =============================================================================

/**
 * @brief Create a new string array
 * @param count Initial number of strings to allocate space for
 * @return Pointer to the newly created string array, or NULL on failure
 */
ZVEC_EXPORT ZVecStringArray *ZVEC_CALL zvec_string_array_create(size_t count);

/**
 * @brief Add a string to the string array at specified index
 * @param array String array pointer
 * @param idx Index position where the string should be added
 * @param str Null-terminated C string to add
 */
ZVEC_EXPORT void ZVEC_CALL zvec_string_array_add(ZVecStringArray *array,
                                                 size_t idx, const char *str);

/**
 * @brief Destroy string array and free all associated memory
 * @param array String array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_string_array_destroy(ZVecStringArray *array);

/**
 * @brief Create a new mutable byte array
 * @param capacity Initial capacity in bytes
 * @return Pointer to the newly created byte array, or NULL on failure
 */
ZVEC_EXPORT ZVecMutableByteArray *ZVEC_CALL
zvec_byte_array_create(size_t capacity);


/**
 * @brief Destroy byte array and free all associated memory
 * @param array Byte array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_byte_array_destroy(ZVecMutableByteArray *array);

/**
 * @brief Create a new float array
 * @param count Number of floats to allocate space for
 * @return Pointer to the newly created float array, or NULL on failure
 */
ZVEC_EXPORT ZVecFloatArray *ZVEC_CALL zvec_float_array_create(size_t count);

/**
 * @brief Destroy float array and free all associated memory
 * @param array Float array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_float_array_destroy(ZVecFloatArray *array);

/**
 * @brief Create a new int64 array
 * @param count Number of int64 values to allocate space for
 * @return Pointer to the newly created int64 array, or NULL on failure
 */
ZVEC_EXPORT ZVecInt64Array *ZVEC_CALL zvec_int64_array_create(size_t count);

/**
 * @brief Destroy int64 array and free all associated memory
 * @param array Int64 array pointer to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_int64_array_destroy(ZVecInt64Array *array);

/**
 * @brief Release uint8_t array memory
 *
 * @param array uint8_t array pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free_uint8_array(uint8_t *array);

/**
 * @brief Free heap memory allocated by zvec C API.
 *
 * Use this helper for pointer-returning APIs that document malloc-allocated
 * buffers. This avoids allocator mismatch across DLL boundaries.
 *
 * @param ptr Memory pointer returned by zvec C API
 */
ZVEC_EXPORT void ZVEC_CALL zvec_free_ptr(void *ptr);


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
} ZVecLogLevel;

/**
 * @brief Log type enumeration
 */
typedef enum { ZVEC_LOG_TYPE_CONSOLE = 0, ZVEC_LOG_TYPE_FILE = 1 } ZVecLogType;

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
typedef struct ZVecLogConfig ZVecLogConfig;

/**
 * @brief Configuration data (opaque pointer)
 * Corresponds to zvec::GlobalConfig::ConfigData
 * Use zvec_config_data_create() to create and
 * zvec_config_data_destroy() to destroy
 */
typedef struct ZVecConfigData ZVecConfigData;

// =============================================================================
// Log Configuration Management Functions
// =============================================================================

/**
 * @brief Create console log configuration
 * @param level Log level
 * @return ZVecLogConfig* Pointer to the newly created log configuration
 */
ZVEC_EXPORT ZVecLogConfig *ZVEC_CALL
zvec_config_log_create_console(ZVecLogLevel level);

/**
 * @brief Create file log configuration
 * @param level Log level
 * @param dir Log directory
 * @param basename Log file base name
 * @param file_size Log file size (MB)
 * @param overdue_days Log expiration days
 * @return ZVecLogConfig* Pointer to the newly created log configuration
 */
ZVEC_EXPORT ZVecLogConfig *ZVEC_CALL zvec_config_log_create_file(
    ZVecLogLevel level, const char *dir, const char *basename,
    uint32_t file_size, uint32_t overdue_days);

/**
 * @brief Destroy log configuration
 * @param config Log configuration pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_config_log_destroy(ZVecLogConfig *config);

/**
 * @brief Get log level from log config
 * @param config Log configuration pointer
 * @return ZVecLogLevel Log level
 */
ZVEC_EXPORT ZVecLogLevel ZVEC_CALL
zvec_config_log_get_level(const ZVecLogConfig *config);

/**
 * @brief Set log level in log config
 * @param config Log configuration pointer
 * @param level Log level
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_log_set_level(ZVecLogConfig *config, ZVecLogLevel level);

/**
 * @brief Check if log config is file type
 * @param config Log configuration pointer
 * @return true if file type, false if console type
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_config_log_is_file_type(const ZVecLogConfig *config);

/**
 * @brief Get log directory from file log config
 * @param config Log configuration pointer (must be file type)
 * @return const char* Log directory (owned by config, do not free)
 * @note Only valid for file log config, returns NULL for console config
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_config_log_get_dir(const ZVecLogConfig *config);

/**
 * @brief Set log directory in file log config
 * @param config Log configuration pointer (must be file type)
 * @param dir Log directory
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_log_set_dir(ZVecLogConfig *config, const char *dir);

/**
 * @brief Get log file basename from file log config
 * @param config Log configuration pointer (must be file type)
 * @return const char* Log file basename (owned by config, do not free)
 * @note Only valid for file log config, returns NULL for console config
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_config_log_get_basename(const ZVecLogConfig *config);

/**
 * @brief Set log file basename in file log config
 * @param config Log configuration pointer (must be file type)
 * @param basename Log file basename
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_log_set_basename(ZVecLogConfig *config, const char *basename);

/**
 * @brief Get log file size from file log config
 * @param config Log configuration pointer (must be file type)
 * @return uint32_t Log file size in MB
 * @note Only valid for file log config, returns 0 for console config
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_log_get_file_size(const ZVecLogConfig *config);

/**
 * @brief Set log file size in file log config
 * @param config Log configuration pointer (must be file type)
 * @param file_size Log file size in MB
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_log_set_file_size(ZVecLogConfig *config, uint32_t file_size);

/**
 * @brief Get log overdue days from file log config
 * @param config Log configuration pointer (must be file type)
 * @return uint32_t Log overdue days
 * @note Only valid for file log config, returns 0 for console config
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_log_get_overdue_days(const ZVecLogConfig *config);

/**
 * @brief Set log overdue days in file log config
 * @param config Log configuration pointer (must be file type)
 * @param days Log overdue days
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_log_set_overdue_days(ZVecLogConfig *config, uint32_t days);

// =============================================================================
// Configuration Data Management Functions
// =============================================================================

/**
 * @brief Create configuration data
 * @return ZVecConfigData* Pointer to the newly created configuration data
 */
ZVEC_EXPORT ZVecConfigData *ZVEC_CALL zvec_config_data_create(void);

/**
 * @brief Destroy configuration data
 * @param config Configuration data pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_config_data_destroy(ZVecConfigData *config);

/**
 * @brief Set memory limit in configuration data
 * @param config Configuration data pointer
 * @param memory_limit_bytes Memory limit in bytes
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_memory_limit(
    ZVecConfigData *config, uint64_t memory_limit_bytes);

/**
 * @brief Get memory limit from configuration data
 * @param config Configuration data pointer
 * @return uint64_t Memory limit in bytes
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_config_data_get_memory_limit(const ZVecConfigData *config);

/**
 * @brief Set log configuration in configuration data
 * @param config Configuration data pointer
 * @param log_type Log type (console or file)
 * @param log_config Log configuration pointer (ownership is transferred to
 * config, do not free separately)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_log_config(
    ZVecConfigData *config, ZVecLogType log_type, void *log_config);

/**
 * @brief Get log type from configuration data
 * @param config Configuration data pointer
 * @return ZVecLogType Log type
 */
ZVEC_EXPORT ZVecLogType ZVEC_CALL
zvec_config_data_get_log_type(const ZVecConfigData *config);

/**
 * @brief Set query thread count in configuration data
 * @param config Configuration data pointer
 * @param thread_count Query thread count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_query_thread_count(
    ZVecConfigData *config, uint32_t thread_count);

/**
 * @brief Get query thread count from configuration data
 * @param config Configuration data pointer
 * @return uint32_t Query thread count
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_data_get_query_thread_count(const ZVecConfigData *config);

/**
 * @brief Set invert to forward scan ratio in configuration data
 * @param config Configuration data pointer
 * @param ratio Invert to forward scan ratio
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_data_set_invert_to_forward_scan_ratio(ZVecConfigData *config,
                                                  float ratio);

/**
 * @brief Get invert to forward scan ratio from configuration data
 * @param config Configuration data pointer
 * @return float Invert to forward scan ratio
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_config_data_get_invert_to_forward_scan_ratio(const ZVecConfigData *config);

/**
 * @brief Set brute force by keys ratio in configuration data
 * @param config Configuration data pointer
 * @param ratio Brute force by keys ratio
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_config_data_set_brute_force_by_keys_ratio(ZVecConfigData *config,
                                               float ratio);

/**
 * @brief Get brute force by keys ratio from configuration data
 * @param config Configuration data pointer
 * @return float Brute force by keys ratio
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_config_data_get_brute_force_by_keys_ratio(const ZVecConfigData *config);

/**
 * @brief Set optimize thread count in configuration data
 * @param config Configuration data pointer
 * @param thread_count Optimize thread count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_optimize_thread_count(
    ZVecConfigData *config, uint32_t thread_count);

/**
 * @brief Get optimize thread count from configuration data
 * @param config Configuration data pointer
 * @return uint32_t Optimize thread count
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_config_data_get_optimize_thread_count(const ZVecConfigData *config);

// =============================================================================
// Initialization and Cleanup Interface
// =============================================================================

/**
 * @brief Initialize ZVec library
 * @param config Configuration data (optional, NULL means using default
 * configuration)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_initialize(const ZVecConfigData *config);

/**
 * @brief Clean up ZVec library resources
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_shutdown(void);

/**
 * @brief Check if library is initialized
 * @return true if initialized, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_is_initialized(void);

// =============================================================================
// Data Type Enumerations
// =============================================================================

/**
 * @brief Data type enumeration
 */
typedef enum {
  ZVEC_DATA_TYPE_UNDEFINED = 0,

  ZVEC_DATA_TYPE_BINARY = 1,
  ZVEC_DATA_TYPE_STRING = 2,
  ZVEC_DATA_TYPE_BOOL = 3,
  ZVEC_DATA_TYPE_INT32 = 4,
  ZVEC_DATA_TYPE_INT64 = 5,
  ZVEC_DATA_TYPE_UINT32 = 6,
  ZVEC_DATA_TYPE_UINT64 = 7,
  ZVEC_DATA_TYPE_FLOAT = 8,
  ZVEC_DATA_TYPE_DOUBLE = 9,

  ZVEC_DATA_TYPE_VECTOR_BINARY32 = 20,
  ZVEC_DATA_TYPE_VECTOR_BINARY64 = 21,
  ZVEC_DATA_TYPE_VECTOR_FP16 = 22,
  ZVEC_DATA_TYPE_VECTOR_FP32 = 23,
  ZVEC_DATA_TYPE_VECTOR_FP64 = 24,
  ZVEC_DATA_TYPE_VECTOR_INT4 = 25,
  ZVEC_DATA_TYPE_VECTOR_INT8 = 26,
  ZVEC_DATA_TYPE_VECTOR_INT16 = 27,

  ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 = 30,
  ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 = 31,

  ZVEC_DATA_TYPE_ARRAY_BINARY = 40,
  ZVEC_DATA_TYPE_ARRAY_STRING = 41,
  ZVEC_DATA_TYPE_ARRAY_BOOL = 42,
  ZVEC_DATA_TYPE_ARRAY_INT32 = 43,
  ZVEC_DATA_TYPE_ARRAY_INT64 = 44,
  ZVEC_DATA_TYPE_ARRAY_UINT32 = 45,
  ZVEC_DATA_TYPE_ARRAY_UINT64 = 46,
  ZVEC_DATA_TYPE_ARRAY_FLOAT = 47,
  ZVEC_DATA_TYPE_ARRAY_DOUBLE = 48
} ZVecDataType;

/**
 * @brief Index type enumeration
 */
typedef enum {
  ZVEC_INDEX_TYPE_UNDEFINED = 0,
  ZVEC_INDEX_TYPE_HNSW = 1,
  ZVEC_INDEX_TYPE_IVF = 3,
  ZVEC_INDEX_TYPE_FLAT = 4,
  ZVEC_INDEX_TYPE_INVERT = 10
} ZVecIndexType;

/**
 * @brief Distance metric type enumeration
 */
typedef enum {
  ZVEC_METRIC_TYPE_UNDEFINED = 0,
  ZVEC_METRIC_TYPE_L2 = 1,
  ZVEC_METRIC_TYPE_IP = 2,
  ZVEC_METRIC_TYPE_COSINE = 3,
  ZVEC_METRIC_TYPE_MIPSL2 = 4
} ZVecMetricType;

/**
 * @brief Quantization type enumeration
 */
typedef enum {
  ZVEC_QUANTIZE_TYPE_UNDEFINED = 0,
  ZVEC_QUANTIZE_TYPE_FP16 = 1,
  ZVEC_QUANTIZE_TYPE_INT8 = 2,
  ZVEC_QUANTIZE_TYPE_INT4 = 3
} ZVecQuantizeType;

// =============================================================================
// Collection Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Collection handle (opaque pointer)
 *
 * Internally maps to std::shared_ptr<zvec::Collection>*.
 * Managed by zvec_collection_create/open() and zvec_collection_close().
 */
typedef struct ZVecCollection ZVecCollection;

// =============================================================================
// Index Parameters Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Index parameters (opaque pointer)
 *
 * Use zvec_index_params_create() to create and zvec_index_params_destroy() to
 * destroy. Specific parameters are set via type-specific setter functions.
 */
typedef struct ZVecIndexParams ZVecIndexParams;

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
typedef struct ZVecFieldSchema ZVecFieldSchema;


// =============================================================================
// Index Parameters Functions
// =============================================================================

/**
 * @brief Create index parameters
 * @param index_type Index type
 * @return Pointer to newly created ZVecIndexParams, or NULL on error
 */
ZVEC_EXPORT ZVecIndexParams *ZVEC_CALL
zvec_index_params_create(ZVecIndexType index_type);

/**
 * @brief Destroy index parameters
 * @param params Index parameters to destroy
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_destroy(ZVecIndexParams *params);

// =============================================================================
// Collection Schema Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Get index type
 * @param params Index parameters (must not be NULL)
 * @return Index type
 */
ZVEC_EXPORT ZVecIndexType ZVEC_CALL
zvec_index_params_get_type(const ZVecIndexParams *params);

/**
 * @brief Set metric type (for vector indexes)
 * @param params Index parameters (must be vector index type)
 * @param metric_type Metric type
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_set_metric_type(
    ZVecIndexParams *params, ZVecMetricType metric_type);

/**
 * @brief Get metric type
 * @param params Index parameters (must not be NULL)
 * @return Metric type
 */
ZVEC_EXPORT ZVecMetricType ZVEC_CALL
zvec_index_params_get_metric_type(const ZVecIndexParams *params);

/**
 * @brief Set quantize type (for vector indexes)
 * @param params Index parameters (must be vector index type)
 * @param quantize_type Quantize type
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_set_quantize_type(
    ZVecIndexParams *params, ZVecQuantizeType quantize_type);

/**
 * @brief Get quantize type
 * @param params Index parameters (must not be NULL)
 * @return Quantize type
 */
ZVEC_EXPORT ZVecQuantizeType ZVEC_CALL
zvec_index_params_get_quantize_type(const ZVecIndexParams *params);

/**
 * @brief Set HNSW specific parameters
 * @param params Index parameters (must be HNSW type)
 * @param m Graph connectivity parameter
 * @param ef_construction Construction exploration factor
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_set_hnsw_params(
    ZVecIndexParams *params, int m, int ef_construction);

/**
 * @brief Get HNSW m parameter
 * @param params Index parameters (must not be NULL)
 * @return m parameter
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_index_params_get_hnsw_m(const ZVecIndexParams *params);

/**
 * @brief Get HNSW ef_construction parameter
 * @param params Index parameters (must not be NULL)
 * @return ef_construction parameter
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_index_params_get_hnsw_ef_construction(const ZVecIndexParams *params);

/**
 * @brief Get HNSW parameters (all at once)
 * @param params Index parameters (must not be NULL)
 * @param out_m Output parameter for m
 * @param out_ef_construction Output parameter for ef_construction
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_get_hnsw_params(
    const ZVecIndexParams *params, int *out_m, int *out_ef_construction);

/**
 * @brief Set IVF specific parameters
 * @param params Index parameters (must be IVF type)
 * @param n_list Number of cluster centers
 * @param n_iters Number of iterations
 * @param use_soar Whether to use SOAR algorithm
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_set_ivf_params(
    ZVecIndexParams *params, int n_list, int n_iters, bool use_soar);

/**
 * @brief Get IVF parameters (all at once)
 * @param params Index parameters (must not be NULL)
 * @param out_n_list Output parameter for n_list
 * @param out_n_iters Output parameter for n_iters
 * @param out_use_soar Output parameter for use_soar
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_index_params_get_ivf_params(const ZVecIndexParams *params, int *out_n_list,
                                 int *out_n_iters, bool *out_use_soar);

/**
 * @brief Get invert index parameters (all at once)
 * @param params Index parameters (must not be NULL)
 * @param out_enable_range_opt Output parameter for enable_range_optimization
 * @param out_enable_wildcard Output parameter for enable_extended_wildcard
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_get_invert_params(
    const ZVecIndexParams *params, bool *out_enable_range_opt,
    bool *out_enable_wildcard);

/**
 * @brief Set invert index specific parameters
 * @param params Index parameters (must be INVERT type)
 * @param enable_range_opt Whether to enable range optimization
 * @param enable_wildcard Whether to enable extended wildcard
 * @return ZVEC_OK on success, error code on failure
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_index_params_set_invert_params(
    ZVecIndexParams *params, bool enable_range_opt, bool enable_wildcard);

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
typedef struct ZVecHnswQueryParams ZVecHnswQueryParams;

/**
 * @brief IVF query parameters handle (opaque pointer)
 *
 * Internally maps to zvec::IVFQueryParams* (raw pointer).
 * Created by zvec_query_params_ivf_create() and destroyed by
 * zvec_query_params_ivf_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct ZVecIVFQueryParams ZVecIVFQueryParams;

/**
 * @brief Flat query parameters handle (opaque pointer)
 *
 * Internally maps to zvec::FlatQueryParams* (raw pointer).
 * Created by zvec_query_params_flat_create() and destroyed by
 * zvec_query_params_flat_destroy(). Caller owns the pointer and must explicitly
 * destroy it.
 */
typedef struct ZVecFlatQueryParams ZVecFlatQueryParams;


// =============================================================================
// Query Structures (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Vector query structure (opaque pointer)
 * Aligned with zvec::VectorQuery
 * Use zvec_vector_query_create() to create and zvec_vector_query_destroy() to
 * destroy
 */
typedef struct ZVecVectorQuery ZVecVectorQuery;

/**
 * @brief Grouped vector query structure (opaque pointer)
 * Aligned with zvec::GroupByVectorQuery
 * Use zvec_group_by_vector_query_create() to create and
 * zvec_group_by_vector_query_destroy() to destroy
 */
typedef struct ZVecGroupByVectorQuery ZVecGroupByVectorQuery;


// =============================================================================
// Query Parameters Management Functions
// =============================================================================

// -----------------------------------------------------------------------------
// ZVecHnswQueryParams (HNSW Query Parameters)
// -----------------------------------------------------------------------------

/**
 * @brief Create HNSW query parameters
 * @param ef Exploration factor during search (default: 40)
 * @param radius Search radius (default: 0.0)
 * @param is_linear Whether linear search (default: false)
 * @param is_using_refiner Whether using refiner (default: false)
 * @return ZVecHnswQueryParams* Pointer to the newly created HNSW query
 * parameters
 */
ZVEC_EXPORT ZVecHnswQueryParams *ZVEC_CALL zvec_query_params_hnsw_create(
    int ef, float radius, bool is_linear, bool is_using_refiner);

/**
 * @brief Destroy HNSW query parameters
 * @param params HNSW query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_hnsw_destroy(ZVecHnswQueryParams *params);

/**
 * @brief Set exploration factor
 * @param params HNSW query parameters pointer
 * @param ef Exploration factor
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_hnsw_set_ef(ZVecHnswQueryParams *params, int ef);

/**
 * @brief Get exploration factor
 * @param params HNSW query parameters pointer
 * @return int Exploration factor
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_query_params_hnsw_get_ef(const ZVecHnswQueryParams *params);

/**
 * @brief Set search radius (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @param radius Search radius
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_hnsw_set_radius(ZVecHnswQueryParams *params, float radius);

/**
 * @brief Get search radius (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @return float Search radius
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_hnsw_get_radius(const ZVecHnswQueryParams *params);

/**
 * @brief Set linear search mode (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @param is_linear Whether linear search
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_hnsw_set_is_linear(
    ZVecHnswQueryParams *params, bool is_linear);

/**
 * @brief Get linear search mode (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @return bool Whether linear search
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_hnsw_get_is_linear(const ZVecHnswQueryParams *params);

/**
 * @brief Set whether to use refiner (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_hnsw_set_is_using_refiner(
    ZVecHnswQueryParams *params, bool is_using_refiner);

/**
 * @brief Get whether to use refiner (common parameter from QueryParams base)
 * @param params HNSW query parameters pointer
 * @return bool Whether to use refiner
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_hnsw_get_is_using_refiner(const ZVecHnswQueryParams *params);

// -----------------------------------------------------------------------------
// ZVecIVFQueryParams (IVF Query Parameters)
// -----------------------------------------------------------------------------

/**
 * @brief Create IVF query parameters
 * @param nprobe Number of clusters to probe (default: 10)
 * @param is_using_refiner Whether using refiner (default: false)
 * @param scale_factor Scale factor (default: 10.0)
 * @return ZVecIVFQueryParams* Pointer to the newly created IVF query parameters
 */
ZVEC_EXPORT ZVecIVFQueryParams *ZVEC_CALL zvec_query_params_ivf_create(
    int nprobe, bool is_using_refiner, float scale_factor);

/**
 * @brief Destroy IVF query parameters
 * @param params IVF query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_ivf_destroy(ZVecIVFQueryParams *params);

/**
 * @brief Set number of probe clusters
 * @param params IVF query parameters pointer
 * @param nprobe Number of probe clusters
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_ivf_set_nprobe(ZVecIVFQueryParams *params, int nprobe);

/**
 * @brief Get number of probe clusters
 * @param params IVF query parameters pointer
 * @return int Number of probe clusters
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_query_params_ivf_get_nprobe(const ZVecIVFQueryParams *params);

/**
 * @brief Set scale factor
 * @param params IVF query parameters pointer
 * @param scale_factor Scale factor
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_ivf_set_scale_factor(
    ZVecIVFQueryParams *params, float scale_factor);

/**
 * @brief Get scale factor
 * @param params IVF query parameters pointer
 * @return float Scale factor
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_ivf_get_scale_factor(const ZVecIVFQueryParams *params);

/**
 * @brief Set search radius (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @param radius Search radius
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_ivf_set_radius(ZVecIVFQueryParams *params, float radius);

/**
 * @brief Get search radius (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @return float Search radius
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_ivf_get_radius(const ZVecIVFQueryParams *params);

/**
 * @brief Set linear search mode (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @param is_linear Whether linear search
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_ivf_set_is_linear(ZVecIVFQueryParams *params, bool is_linear);

/**
 * @brief Get linear search mode (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @return bool Whether linear search
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_ivf_get_is_linear(const ZVecIVFQueryParams *params);

/**
 * @brief Set whether to use refiner (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_ivf_set_is_using_refiner(
    ZVecIVFQueryParams *params, bool is_using_refiner);

/**
 * @brief Get whether to use refiner (common parameter from QueryParams base)
 * @param params IVF query parameters pointer
 * @return bool Whether to use refiner
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_ivf_get_is_using_refiner(const ZVecIVFQueryParams *params);

// -----------------------------------------------------------------------------
// ZVecFlatQueryParams (Flat Query Parameters)
// -----------------------------------------------------------------------------

/**
 * @brief Create Flat query parameters
 * @param is_using_refiner Whether using refiner (default: false)
 * @param scale_factor Scale factor (default: 10.0)
 * @return ZVecFlatQueryParams* Pointer to the newly created Flat query
 * parameters
 */
ZVEC_EXPORT ZVecFlatQueryParams *ZVEC_CALL
zvec_query_params_flat_create(bool is_using_refiner, float scale_factor);

/**
 * @brief Destroy Flat query parameters
 * @param params Flat query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_flat_destroy(ZVecFlatQueryParams *params);

/**
 * @brief Set scale factor
 * @param params Flat query parameters pointer
 * @param scale_factor Scale factor
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_flat_set_scale_factor(
    ZVecFlatQueryParams *params, float scale_factor);

/**
 * @brief Get scale factor
 * @param params Flat query parameters pointer
 * @return float Scale factor
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_flat_get_scale_factor(const ZVecFlatQueryParams *params);

/**
 * @brief Set search radius (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @param radius Search radius
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_flat_set_radius(ZVecFlatQueryParams *params, float radius);

/**
 * @brief Get search radius (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @return float Search radius
 */
ZVEC_EXPORT float ZVEC_CALL
zvec_query_params_flat_get_radius(const ZVecFlatQueryParams *params);

/**
 * @brief Set linear search mode (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @param is_linear Whether linear search
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_flat_set_is_linear(
    ZVecFlatQueryParams *params, bool is_linear);

/**
 * @brief Get linear search mode (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @return bool Whether linear search
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_flat_get_is_linear(const ZVecFlatQueryParams *params);

/**
 * @brief Set whether to use refiner (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_flat_set_is_using_refiner(
    ZVecFlatQueryParams *params, bool is_using_refiner);

/**
 * @brief Get whether to use refiner (common parameter from QueryParams base)
 * @param params Flat query parameters pointer
 * @return bool Whether to use refiner
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_query_params_flat_get_is_using_refiner(const ZVecFlatQueryParams *params);

// -----------------------------------------------------------------------------
// ZVecVectorQuery (Vector Query)
// -----------------------------------------------------------------------------

/**
 * @brief Create vector query
 * @return ZVecVectorQuery* Pointer to the newly created vector query
 */
ZVEC_EXPORT ZVecVectorQuery *ZVEC_CALL zvec_vector_query_create(void);

/**
 * @brief Destroy vector query
 * @param query Vector query pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_vector_query_destroy(ZVecVectorQuery *query);

/**
 * @brief Set topk (number of results to return)
 * @param query Vector query pointer
 * @param topk Number of results
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_vector_query_set_topk(ZVecVectorQuery *query, int topk);

/**
 * @brief Get topk
 * @param query Vector query pointer
 * @return int Number of results
 */
ZVEC_EXPORT int ZVEC_CALL
zvec_vector_query_get_topk(const ZVecVectorQuery *query);

/**
 * @brief Set field name
 * @param query Vector query pointer
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_set_field_name(
    ZVecVectorQuery *query, const char *field_name);

/**
 * @brief Get field name
 * @param query Vector query pointer
 * @return const char* Field name (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_vector_query_get_field_name(const ZVecVectorQuery *query);

/**
 * @brief Set query vector data
 * @param query Vector query pointer
 * @param data Vector data pointer
 * @param size Data size in bytes
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_set_query_vector(
    ZVecVectorQuery *query, const void *data, size_t size);

/**
 * @brief Set filter expression
 * @param query Vector query pointer
 * @param filter Filter expression string
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_vector_query_set_filter(ZVecVectorQuery *query, const char *filter);

/**
 * @brief Get filter expression
 * @param query Vector query pointer
 * @return const char* Filter expression (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_vector_query_get_filter(const ZVecVectorQuery *query);

/**
 * @brief Set whether to include vector data in results
 * @param query Vector query pointer
 * @param include Whether to include vector
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_vector_query_set_include_vector(ZVecVectorQuery *query, bool include);

/**
 * @brief Get whether to include vector data in results
 * @param query Vector query pointer
 * @return bool Whether to include vector
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_vector_query_get_include_vector(const ZVecVectorQuery *query);

/**
 * @brief Set whether to include doc ID in results
 * @param query Vector query pointer
 * @param include Whether to include doc ID
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_vector_query_set_include_doc_id(ZVecVectorQuery *query, bool include);

/**
 * @brief Get whether to include doc ID in results
 * @param query Vector query pointer
 * @return bool Whether to include doc ID
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_vector_query_get_include_doc_id(const ZVecVectorQuery *query);

/**
 * @brief Set output fields
 * @param query Vector query pointer
 * @param fields Array of field names
 * @param count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_set_output_fields(
    ZVecVectorQuery *query, const char **fields, size_t count);

/**
 * @brief Get output fields
 * @param query Vector query pointer
 * @param[out] fields Output array of field names (caller owns the memory,
 * should free with free())
 * @param[out] count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_get_output_fields(
    const ZVecVectorQuery *query, const char ***fields, size_t *count);

/**
 * @brief Set query parameters (takes ownership)
 * @param query Vector query pointer
 * @param params Query parameters pointer (type-specific: ZVecHnswQueryParams*,
 * etc.)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_vector_query_set_query_params(ZVecVectorQuery *query, void *params);

/**
 * @brief Set HNSW query parameters (takes ownership)
 * @param query Vector query pointer
 * @param hnsw_params HNSW query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_set_hnsw_params(
    ZVecVectorQuery *query, ZVecHnswQueryParams *hnsw_params);

/**
 * @brief Set IVF query parameters (takes ownership)
 * @param query Vector query pointer
 * @param ivf_params IVF query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_set_ivf_params(
    ZVecVectorQuery *query, ZVecIVFQueryParams *ivf_params);

/**
 * @brief Set Flat query parameters (takes ownership)
 * @param query Vector query pointer
 * @param flat_params Flat query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_vector_query_set_flat_params(
    ZVecVectorQuery *query, ZVecFlatQueryParams *flat_params);

// -----------------------------------------------------------------------------
// ZVecGroupByVectorQuery (Group By Vector Query)
// -----------------------------------------------------------------------------

/**
 * @brief Create group by vector query
 * @return ZVecGroupByVectorQuery* Pointer to the newly created group by vector
 * query
 */
ZVEC_EXPORT ZVecGroupByVectorQuery *ZVEC_CALL
zvec_group_by_vector_query_create(void);

/**
 * @brief Destroy group by vector query
 * @param query Group by vector query pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_group_by_vector_query_destroy(ZVecGroupByVectorQuery *query);

/**
 * @brief Set field name
 * @param query Group by vector query pointer
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_field_name(
    ZVecGroupByVectorQuery *query, const char *field_name);

/**
 * @brief Get field name
 * @param query Group by vector query pointer
 * @return const char* Field name (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_group_by_vector_query_get_field_name(const ZVecGroupByVectorQuery *query);

/**
 * @brief Set group by field name
 * @param query Group by vector query pointer
 * @param field_name Group by field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_group_by_vector_query_set_group_by_field_name(
    ZVecGroupByVectorQuery *query, const char *field_name);

/**
 * @brief Get group by field name
 * @param query Group by vector query pointer
 * @return const char* Group by field name (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_group_by_vector_query_get_group_by_field_name(
    const ZVecGroupByVectorQuery *query);

/**
 * @brief Set group count
 * @param query Group by vector query pointer
 * @param count Number of groups
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_group_count(
    ZVecGroupByVectorQuery *query, uint32_t count);

/**
 * @brief Get group count
 * @param query Group by vector query pointer
 * @return uint32_t Number of groups
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_group_by_vector_query_get_group_count(const ZVecGroupByVectorQuery *query);

/**
 * @brief Set group topk
 * @param query Group by vector query pointer
 * @param topk Number of results per group
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_group_topk(
    ZVecGroupByVectorQuery *query, uint32_t topk);

/**
 * @brief Get group topk
 * @param query Group by vector query pointer
 * @return uint32_t Number of results per group
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_group_by_vector_query_get_group_topk(const ZVecGroupByVectorQuery *query);

/**
 * @brief Set query vector data
 * @param query Group by vector query pointer
 * @param data Vector data pointer
 * @param size Data size in bytes
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_query_vector(
    ZVecGroupByVectorQuery *query, const void *data, size_t size);

/**
 * @brief Set filter expression
 * @param query Group by vector query pointer
 * @param filter Filter expression string
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_filter(
    ZVecGroupByVectorQuery *query, const char *filter);

/**
 * @brief Get filter expression
 * @param query Group by vector query pointer
 * @return const char* Filter expression (owned by query, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_group_by_vector_query_get_filter(const ZVecGroupByVectorQuery *query);

/**
 * @brief Set whether to include vector data in results
 * @param query Group by vector query pointer
 * @param include Whether to include vectors
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_group_by_vector_query_set_include_vector(ZVecGroupByVectorQuery *query,
                                              bool include);

/**
 * @brief Get whether to include vector data in results
 * @param query Group by vector query pointer
 * @return bool Whether to include vectors
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_group_by_vector_query_get_include_vector(
    const ZVecGroupByVectorQuery *query);

/**
 * @brief Set output fields
 * @param query Group by vector query pointer
 * @param fields Array of field names
 * @param count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_group_by_vector_query_set_output_fields(ZVecGroupByVectorQuery *query,
                                             const char **fields, size_t count);

/**
 * @brief Get output fields
 * @param query Group by vector query pointer
 * @param[out] fields Output array of field names (caller owns the memory,
 * should free with free())
 * @param[out] count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_group_by_vector_query_get_output_fields(ZVecGroupByVectorQuery *query,
                                             const char ***fields,
                                             size_t *count);

/**
 * @brief Set query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param params Query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_query_params(
    ZVecGroupByVectorQuery *query, void *params);

/**
 * @brief Set HNSW query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param hnsw_params HNSW query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_hnsw_params(
    ZVecGroupByVectorQuery *query, ZVecHnswQueryParams *hnsw_params);

/**
 * @brief Set IVF query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param ivf_params IVF query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_ivf_params(
    ZVecGroupByVectorQuery *query, ZVecIVFQueryParams *ivf_params);

/**
 * @brief Set Flat query parameters (takes ownership)
 * @param query Group by vector query pointer
 * @param flat_params Flat query parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_group_by_vector_query_set_flat_params(
    ZVecGroupByVectorQuery *query, ZVecFlatQueryParams *flat_params);

// =============================================================================
// Collection Options and Statistics (Opaque Pointer Pattern)
// =============================================================================

/**
 * @brief Collection options (opaque pointer)
 * Use zvec_collection_options_create() to create and
 * zvec_collection_options_destroy() to destroy
 */
typedef struct ZVecCollectionOptions ZVecCollectionOptions;

/**
 * @brief Collection statistics (opaque pointer)
 * Use zvec_collection_stats_get functions to access fields
 */
typedef struct ZVecCollectionStats ZVecCollectionStats;

// =============================================================================
// Collection Options Management Functions
// =============================================================================

/**
 * @brief Create collection options
 * @return ZVecCollectionOptions* Pointer to the newly created collection
 * options
 */
ZVEC_EXPORT ZVecCollectionOptions *ZVEC_CALL
zvec_collection_options_create(void);

/**
 * @brief Destroy collection options
 * @param options Collection options pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_options_destroy(ZVecCollectionOptions *options);

/**
 * @brief Set whether to enable memory mapping
 * @param options Collection options pointer
 * @param enable Whether to enable mmap
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_options_set_enable_mmap(
    ZVecCollectionOptions *options, bool enable);

/**
 * @brief Get whether to enable memory mapping
 * @param options Collection options pointer
 * @return bool Whether mmap is enabled
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_collection_options_get_enable_mmap(const ZVecCollectionOptions *options);

/**
 * @brief Set maximum buffer size
 * @param options Collection options pointer
 * @param size Maximum buffer size in bytes
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_options_set_max_buffer_size(
    ZVecCollectionOptions *options, size_t size);

/**
 * @brief Get maximum buffer size
 * @param options Collection options pointer
 * @return size_t Maximum buffer size in bytes
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_collection_options_get_max_buffer_size(
    const ZVecCollectionOptions *options);

/**
 * @brief Set whether read-only mode
 * @param options Collection options pointer
 * @param read_only Whether read-only
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_options_set_read_only(
    ZVecCollectionOptions *options, bool read_only);

/**
 * @brief Get whether read-only mode
 * @param options Collection options pointer
 * @return bool Whether read-only mode
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_collection_options_get_read_only(const ZVecCollectionOptions *options);

/**
 * @brief Set maximum document count per segment
 * @param options Collection options pointer
 * @param count Maximum document count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_options_set_max_doc_count_per_segment(
    ZVecCollectionOptions *options, uint64_t count);

/**
 * @brief Get maximum document count per segment
 * @param options Collection options pointer
 * @return uint64_t Maximum document count per segment
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_collection_options_get_max_doc_count_per_segment(
    const ZVecCollectionOptions *options);

// =============================================================================
// Collection Statistics Management Functions
// =============================================================================

/**
 * @brief Get document count from collection stats
 * @param stats Collection statistics pointer
 * @return uint64_t Document count
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_collection_stats_get_doc_count(const ZVecCollectionStats *stats);

/**
 * @brief Get index count from collection stats
 * @param stats Collection statistics pointer
 * @return size_t Number of indexes
 */
ZVEC_EXPORT size_t ZVEC_CALL
zvec_collection_stats_get_index_count(const ZVecCollectionStats *stats);

/**
 * @brief Get index name at specified index
 * @param stats Collection statistics pointer
 * @param index Index of the index name
 * @return const char* Index name (owned by stats, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_collection_stats_get_index_name(
    const ZVecCollectionStats *stats, size_t index);

/**
 * @brief Get index completeness at specified index
 * @param stats Collection statistics pointer
 * @param index Index of the completeness value
 * @return float Index completeness
 */
ZVEC_EXPORT float ZVEC_CALL zvec_collection_stats_get_index_completeness(
    const ZVecCollectionStats *stats, size_t index);

/**
 * @brief Create field schema
 * @param name Field name
 * @param data_type Data type
 * @param nullable Whether nullable
 * @param dimension Vector dimension
 * @return ZVecFieldSchema* Pointer to the newly created field schema
 */
ZVEC_EXPORT ZVecFieldSchema *ZVEC_CALL
zvec_field_schema_create(const char *name, ZVecDataType data_type,
                         bool nullable, uint32_t dimension);

/**
 * @brief Destroy field schema
 * @param schema Field schema pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_field_schema_destroy(ZVecFieldSchema *schema);

/**
 * @brief Set field name
 * @param schema Field schema pointer
 * @param name New field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_field_schema_set_name(ZVecFieldSchema *schema, const char *name);

/**
 * @brief Get field name
 * @param schema Field schema pointer (must not be NULL)
 * @return const char* Field name (owned by schema, do not free)
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_field_schema_get_name(const ZVecFieldSchema *schema);

/**
 * @brief Get field data type
 * @param schema Field schema pointer (must not be NULL)
 * @return ZVecDataType Data type
 */
ZVEC_EXPORT ZVecDataType ZVEC_CALL
zvec_field_schema_get_data_type(const ZVecFieldSchema *schema);

/**
 * @brief Set field data type
 * @param schema Field schema pointer
 * @param data_type New data type
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_field_schema_set_data_type(
    ZVecFieldSchema *schema, ZVecDataType data_type);

/**
 * @brief Get element data type for array fields
 * @param schema Field schema pointer (must not be NULL)
 * @return ZVecDataType Element data type, or original type if not array
 */
ZVEC_EXPORT ZVecDataType ZVEC_CALL
zvec_field_schema_get_element_data_type(const ZVecFieldSchema *schema);

/**
 * @brief Get element data size for the field
 * @param schema Field schema pointer (must not be NULL)
 * @return size_t Element data size in bytes, or 0 for variable-size types
 */
ZVEC_EXPORT size_t ZVEC_CALL
zvec_field_schema_get_element_data_size(const ZVecFieldSchema *schema);

/**
 * @brief Check if field is a vector field (dense or sparse)
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if vector field, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_vector_field(const ZVecFieldSchema *schema);

/**
 * @brief Check if field is a dense vector field
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if dense vector field, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_dense_vector(const ZVecFieldSchema *schema);

/**
 * @brief Check if field is a sparse vector field
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if sparse vector field, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_sparse_vector(const ZVecFieldSchema *schema);

/**
 * @brief Check if field is nullable
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if nullable, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_nullable(const ZVecFieldSchema *schema);

/**
 * @brief Set field nullable
 * @param schema Field schema pointer
 * @param nullable Whether nullable
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_field_schema_set_nullable(ZVecFieldSchema *schema, bool nullable);

/**
 * @brief Check if field has inverted index (for scalar fields)
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if has inverted index, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_has_invert_index(const ZVecFieldSchema *schema);

/**
 * @brief Check if field is an array type
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if array type, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_is_array_type(const ZVecFieldSchema *schema);

/**
 * @brief Get field dimension (for vector fields)
 * @param schema Field schema pointer (must not be NULL)
 * @return uint32_t Dimension value
 */
ZVEC_EXPORT uint32_t ZVEC_CALL
zvec_field_schema_get_dimension(const ZVecFieldSchema *schema);

/**
 * @brief Set field dimension (for vector fields)
 * @param schema Field schema pointer
 * @param dimension Dimension value
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_field_schema_set_dimension(ZVecFieldSchema *schema, uint32_t dimension);

/**
 * @brief Get index type of the field
 * @param schema Field schema pointer (must not be NULL)
 * @return ZVecIndexType Index type, ZVEC_INDEX_TYPE_UNDEFINED if no index
 */
ZVEC_EXPORT ZVecIndexType ZVEC_CALL
zvec_field_schema_get_index_type(const ZVecFieldSchema *schema);

/**
 * @brief Check if field has index
 * @param schema Field schema pointer (must not be NULL)
 * @return bool true if field has index, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL
zvec_field_schema_has_index(const ZVecFieldSchema *schema);

/**
 * @brief Get index params of the field (returns pointer owned by the field
 * schema, do not destroy. Pointer becomes invalid if schema is modified or
 * destroyed)
 * @param schema Field schema pointer (must not be NULL)
 * @return ZVecIndexParams* Index params pointer, NULL if no index
 */
ZVEC_EXPORT const ZVecIndexParams *ZVEC_CALL
zvec_field_schema_get_index_params(const ZVecFieldSchema *schema);

/**
 * @brief Set index parameters for field
 * @param schema Field schema pointer
 * @param index_params Index parameters pointer (deep-copied internally, caller
 *                     retains ownership and must call zvec_index_params_destroy
 *                     after the call)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_field_schema_set_index_params(
    ZVecFieldSchema *schema, const ZVecIndexParams *index_params);

/**
 * @brief Validate field schema
 * @param schema Field schema pointer
 * @param[out] error_msg Error message (needs to be freed by calling
 * zvec_free_string)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_field_schema_validate(
    const ZVecFieldSchema *schema, ZVecString **error_msg);

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
typedef struct ZVecCollectionSchema ZVecCollectionSchema;

/**
 * @brief Create collection schema
 * @param name Collection name
 * @return ZVecCollectionSchema* Pointer to the newly created collection schema
 */
ZVEC_EXPORT ZVecCollectionSchema *ZVEC_CALL
zvec_collection_schema_create(const char *name);

/**
 * @brief Destroy collection schema
 * @param schema Collection schema pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_schema_destroy(ZVecCollectionSchema *schema);

/**
 * @brief Get collection schema name
 * @param schema Collection schema pointer (must not be NULL)
 * @return const char* Collection name (caller owns the memory, should free with
 * free())
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_collection_schema_get_name(const ZVecCollectionSchema *schema);

/**
 * @brief Set collection schema name
 * @param schema Collection schema pointer
 * @param name New collection name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_schema_set_name(ZVecCollectionSchema *schema, const char *name);

/**
 * @brief Add field to collection schema
 * @param schema Collection schema pointer
 * @param field Field schema pointer (will be cloned, caller retains ownership)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_add_field(
    ZVecCollectionSchema *schema, const ZVecFieldSchema *field);

/**
 * @brief Alter field schema
 * @param schema Collection schema pointer
 * @param field_name Name of field to alter
 * @param new_field New field schema with updated properties
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_alter_field(
    ZVecCollectionSchema *schema, const char *field_name,
    const ZVecFieldSchema *new_field);

/**
 * @brief Drop field from schema
 * @param schema Collection schema pointer
 * @param field_name Field name to drop
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_drop_field(
    ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Check if field exists in schema
 * @param schema Collection schema pointer
 * @param field_name Field name to check
 * @return true if field exists, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_collection_schema_has_field(
    const ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Get field by name
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return ZVecFieldSchema* Field schema pointer (non-owning, do not destroy),
 * NULL if not found
 */
ZVEC_EXPORT ZVecFieldSchema *ZVEC_CALL zvec_collection_schema_get_field(
    const ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Get forward (scalar) field by name
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return ZVecFieldSchema* Field schema pointer (non-owning, do not destroy),
 * NULL if not found or not scalar
 */
ZVEC_EXPORT ZVecFieldSchema *ZVEC_CALL zvec_collection_schema_get_forward_field(
    const ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Get vector field by name
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return ZVecFieldSchema* Field schema pointer (non-owning, do not destroy),
 * NULL if not found or not vector
 */
ZVEC_EXPORT ZVecFieldSchema *ZVEC_CALL zvec_collection_schema_get_vector_field(
    const ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Get all forward (scalar) fields
 * @param schema Collection schema pointer
 * @param[out] fields Output array of field pointers (owned by caller, do not
 * destroy)
 * @param[out] count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_get_forward_fields(
    const ZVecCollectionSchema *schema, ZVecFieldSchema ***fields,
    size_t *count);

/**
 * @brief Get all forward fields with index
 * @param schema Collection schema pointer
 * @param[out] fields Output array of field pointers
 * @param[out] count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_schema_get_forward_fields_with_index(
    const ZVecCollectionSchema *schema, ZVecFieldSchema ***fields,
    size_t *count);

/**
 * @brief Get all forward (scalar) field names
 * @param schema Collection schema pointer
 * @param[out] names Output array of field names (caller owns the memory, should
 * free with free())
 * @param[out] count Number of field names
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_schema_get_forward_field_names(
    const ZVecCollectionSchema *schema, const char ***names, size_t *count);

/**
 * @brief Get all forward field names with index
 * @param schema Collection schema pointer
 * @param[out] names Output array of field names (caller owns the memory, should
 * free with free())
 * @param[out] count Number of field names
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_schema_get_forward_field_names_with_index(
    const ZVecCollectionSchema *schema, const char ***names, size_t *count);

/**
 * @brief Get all field names
 * @param schema Collection schema pointer
 * @param[out] names Output array of field names (caller owns the memory, should
 * free with free())
 * @param[out] count Number of field names
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_get_all_field_names(
    const ZVecCollectionSchema *schema, const char ***names, size_t *count);

/**
 * @brief Get all vector fields
 * @param schema Collection schema pointer
 * @param[out] fields Output array of field pointers
 * @param[out] count Number of fields
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_get_vector_fields(
    const ZVecCollectionSchema *schema, ZVecFieldSchema ***fields,
    size_t *count);

/**
 * @brief Get maximum document count per segment of collection schema
 *
 * @param schema Collection schema pointer
 * @return uint64_t Maximum document count per segment
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_collection_schema_get_max_doc_count_per_segment(
    const ZVecCollectionSchema *schema);

/**
 * @brief Set maximum document count per segment
 * @param schema Collection schema pointer
 * @param max_doc_count Maximum document count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_schema_set_max_doc_count_per_segment(
    ZVecCollectionSchema *schema, uint64_t max_doc_count);

/**
 * @brief Validate collection schema
 * @param schema Collection schema pointer
 * @param[out] error_msg Error message (needs to be freed by calling
 * zvec_free_string)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_validate(
    const ZVecCollectionSchema *schema, ZVecString **error_msg);

/**
 * @brief Add index to field
 * @param schema Collection schema pointer
 * @param field_name Field name to add index to
 * @param index_params Index parameters
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_add_index(
    ZVecCollectionSchema *schema, const char *field_name,
    const ZVecIndexParams *index_params);

/**
 * @brief Drop index from field
 * @param schema Collection schema pointer
 * @param field_name Field name to drop index from
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_drop_index(
    ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Check if field has index
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return true if field has index, false otherwise
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_collection_schema_has_index(
    const ZVecCollectionSchema *schema, const char *field_name);

// =============================================================================
// Collection Management Functions
// =============================================================================

/**
 * @brief Create and open collection
 * @param path Collection path
 * @param schema Collection schema pointer
 * @param options Collection options pointer (NULL uses default options)
 * @param[out] collection Returned collection handle
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_and_open(
    const char *path, const ZVecCollectionSchema *schema,
    const ZVecCollectionOptions *options, ZVecCollection **collection);


/**
 * @brief Open existing collection
 * @param path Collection path
 * @param options Collection options pointer (NULL uses default options)
 * @param[out] collection Returned collection handle
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_open(const char *path, const ZVecCollectionOptions *options,
                     ZVecCollection **collection);

/**
 * @brief Close collection
 * @param collection Collection handle
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_close(ZVecCollection *collection);

/**
 * @brief Destroy collection
 *
 * @param collection Collection handle
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_destroy(ZVecCollection *collection);

/**
 * @brief Flush collection data to disk
 * @param collection Collection handle
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_flush(ZVecCollection *collection);

/**
 * @brief Get collection schema
 * @param collection Collection handle
 * @param[out] schema
 * Returned collection schema pointer (needs to be freed by calling
 * zvec_collection_schema_destroy)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_get_schema(
    const ZVecCollection *collection, ZVecCollectionSchema **schema);

/**
 * @brief Get collection options
 * @param collection Collection handle
 * @param[out] options
 * Returned collection options pointer (needs to be freed by calling
 * zvec_collection_options_destroy)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_get_options(
    const ZVecCollection *collection, ZVecCollectionOptions **options);

/**
 * @brief Get collection statistics
 * @param collection Collection handle
 * @param[out] stats
 * Returned statistics pointer (needs to be freed by calling
 * zvec_collection_stats_destroy)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_get_stats(
    const ZVecCollection *collection, ZVecCollectionStats **stats);

/**
 * @brief Destroy collection statistics
 * @param stats Statistics pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_stats_destroy(ZVecCollectionStats *stats);

/**
 * @brief Free field schema memory
 *
 * @param field_schema Field schema pointer to be freed
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_free_field_schema(ZVecFieldSchema *field_schema);

// =============================================================================
// Index Management Interface
// =============================================================================

/**
 * @brief Create index
 *
 * @param collection Collection handle
 * @param field_name Field name
 * @param index_params Index parameters
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_create_index(ZVecCollection *collection, const char *field_name,
                             const ZVecIndexParams *index_params);

/**
 * @brief Create HNSW index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param hnsw_params HNSW index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_hnsw_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecIndexParams *hnsw_params);

/**
 * @brief Create Flat index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param flat_params Flat index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_flat_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecIndexParams *flat_params);

/**
 * @brief Create IVF index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param ivf_params IVF index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_ivf_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecIndexParams *ivf_params);

/**
 * @brief Create scalar index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param invert_params Scalar index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_invert_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecIndexParams *invert_params);

/**
 * @brief Drop index
 * @param collection Collection handle
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_drop_index(ZVecCollection *collection, const char *field_name);

/**
 * @brief Optimize collection (rebuild indexes, merge segments, etc.)
 * @param collection Collection handle
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_optimize(ZVecCollection *collection);

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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_add_column(
    ZVecCollection *collection, const ZVecFieldSchema *field_schema,
    const char *expression);

/**
 * @brief Drop column
 * @param collection Collection handle
 * @param column_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_drop_column(
    ZVecCollection *collection, const char *column_name);

/**
 * @brief Alter column
 * @param collection Collection handle
 * @param column_name Column/field name to alter
 * @param new_name New field name (can be NULL to indicate no renaming)
 * @param new_schema New field schema (can be NULL to indicate no schema
 * modification). The schema is deep-copied internally, caller retains
 *               ownership and must call zvec_field_schema_destroy after the
 * call.
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_alter_column(
    ZVecCollection *collection, const char *column_name, const char *new_name,
    const ZVecFieldSchema *new_schema);

/**
 * @brief Document structure (opaque pointer mode)
 * Internal implementation details are not visible to the outside, and
 * operations are performed through API functions
 */
typedef struct ZVecDoc ZVecDoc;

/**
 * @brief Per-document status returned by detailed DML APIs.
 * @note Uses ordered style: result index corresponds to input document index.
 *       Caller should access pk by index from the original input array.
 */
typedef struct {
  ZVecErrorCode code;  /**< Per-document status code */
  const char *message; /**< Per-document status message (allocated by API) */
} ZVecWriteResult;

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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_insert(
    ZVecCollection *collection, const ZVecDoc **docs, size_t doc_count,
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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_insert_with_results(
    ZVecCollection *collection, const ZVecDoc **docs, size_t doc_count,
    ZVecWriteResult **results, size_t *result_count);

/**
 * @brief Update documents in collection
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] success_count Number of successfully updated documents
 * @param[out] error_count Number of failed updates
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_update(
    ZVecCollection *collection, const ZVecDoc **docs, size_t doc_count,
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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_update_with_results(
    ZVecCollection *collection, const ZVecDoc **docs, size_t doc_count,
    ZVecWriteResult **results, size_t *result_count);

/**
 * @brief Insert or update documents in collection (upsert operation)
 * @param collection Collection handle
 * @param docs Document array
 * @param doc_count Document count
 * @param[out] success_count Number of successful operations
 * @param[out] error_count Number of failed operations
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_upsert(
    ZVecCollection *collection, const ZVecDoc **docs, size_t doc_count,
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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_upsert_with_results(
    ZVecCollection *collection, const ZVecDoc **docs, size_t doc_count,
    ZVecWriteResult **results, size_t *result_count);

/**
 * @brief Delete documents from collection
 * @param collection Collection handle
 * @param pks Primary key array
 * @param pk_count Primary key count
 * @param[out] success_count Number of successfully deleted documents
 * @param[out] error_count Number of failed deletions
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_delete(
    ZVecCollection *collection, const char *const *pks, size_t pk_count,
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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_delete_with_results(
    ZVecCollection *collection, const char *const *pks, size_t pk_count,
    ZVecWriteResult **results, size_t *result_count);

/**
 * @brief Free result arrays returned by detailed DML APIs.
 *
 * @param results Result array pointer
 * @param result_count Number of entries in result array
 */
ZVEC_EXPORT void ZVEC_CALL zvec_write_results_free(ZVecWriteResult *results,
                                                   size_t result_count);

/**
 * @brief Delete documents by filter condition
 * @param collection Collection handle
 * @param filter Filter expression
 * @param[out] deleted_count Number of deleted documents
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_delete_by_filter(
    ZVecCollection *collection, const char *filter);

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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_query(
    const ZVecCollection *collection, const ZVecVectorQuery *query,
    ZVecDoc ***results, size_t *result_count);

/**
 * @brief Grouped vector similarity search
 * @param collection Collection handle
 * @param query Grouped query parameters pointer
 * @param[out] results Returned document array (needs to be freed by calling
 * zvec_docs_free)
 * @param[out] group_by_values Returned group by field values array (needs to be
 * freed by calling zvec_string_array_destroy)
 * @param[out] result_count Number of returned results
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_query_by_group(
    const ZVecCollection *collection, const ZVecGroupByVectorQuery *query,
    ZVecDoc ***results, ZVecString ***group_by_values, size_t *result_count);

/**
 * @brief Fetch documents by primary keys
 * @param collection Collection handle
 * @param primary_keys Primary key array
 * @param count Number of primary keys
 * @param[out] documents Returned document array (needs to be freed by calling
 * zvec_docs_free)
 * @param[out] found_count Number of found documents
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_fetch(
    ZVecCollection *collection, const char *const *primary_keys, size_t count,
    ZVecDoc ***documents, size_t *found_count);

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
  ZVecString string_value;
  ZVecFloatArray vector_value;
  ZVecByteArray binary_value; /**< Binary data value */
} ZVecFieldValue;

/**
 * @brief Document field structure
 */
typedef struct {
  ZVecString name;         ///< Field name
  ZVecDataType data_type;  ///< Data type
  ZVecFieldValue value;    ///< Field value
} ZVecDocField;

/**
 * @brief Document operator enumeration
 */
typedef enum {
  ZVEC_DOC_OP_INSERT = 0,  ///< Insert operation
  ZVEC_DOC_OP_UPDATE = 1,  ///< Update operation
  ZVEC_DOC_OP_UPSERT = 2,  ///< Insert or update operation
  ZVEC_DOC_OP_DELETE = 3   ///< Delete operation
} ZVecDocOperator;

// =============================================================================
// Data Manipulation Interface (DML)
// =============================================================================

/**
 * @brief Create a new document object
 *
 * @return ZVecDoc* Pointer to the newly created document object, returns NULL
 * on failure
 */
ZVEC_EXPORT ZVecDoc *ZVEC_CALL zvec_doc_create(void);

/**
 * @brief Destroy the document object and release all resources
 *
 * @param doc Pointer to the document object
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_destroy(ZVecDoc *doc);

/**
 * @brief Clear the document object
 *
 * @param doc Pointer to the document object
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_clear(ZVecDoc *doc);

/**
 * @brief Add field to document by value
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @param data_type Data type
 * @param value Value pointer
 * @param value_size Value size
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_add_field_by_value(
    ZVecDoc *doc, const char *field_name, ZVecDataType data_type,
    const void *value, size_t value_size);

/**
 * @brief Add field to document by structure
 *
 * @param doc Document object pointer
 * @param field Field structure pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_doc_add_field_by_struct(ZVecDoc *doc, const ZVecDocField *field);

/**
 * @brief Remove field from document
 *
 * @param doc Document structure pointer
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_doc_remove_field(ZVecDoc *doc, const char *field_name);

/**
 * @brief Batch release document array
 *
 * @param documents Document pointer array
 * @param count Document count
 */
ZVEC_EXPORT void ZVEC_CALL zvec_docs_free(ZVecDoc **documents, size_t count);

/**
 * @brief Set document primary key
 *
 * @param doc Pointer to the document structure
 * @param pk Primary key string
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_pk(ZVecDoc *doc, const char *pk);

/**
 * @brief Set document ID
 *
 * @param doc Document structure pointer
 * @param doc_id Document ID
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_doc_id(ZVecDoc *doc, uint64_t doc_id);

/**
 * @brief Set document score
 *
 * @param doc Document structure pointer
 * @param score Score value
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_score(ZVecDoc *doc, float score);

/**
 * @brief Set document operator
 *
 * @param doc Document structure pointer
 * @param op Operator
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_set_operator(ZVecDoc *doc,
                                                 ZVecDocOperator op);

/**
 * @brief Explicitly mark a document field as null.
 *
 * @param doc Document structure pointer
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_doc_set_field_null(ZVecDoc *doc, const char *field_name);

/**
 * @brief Get document ID
 *
 * @param doc Document structure pointer
 * @return uint64_t Document ID
 */
ZVEC_EXPORT uint64_t ZVEC_CALL zvec_doc_get_doc_id(const ZVecDoc *doc);

/**
 * @brief Get document score
 *
 * @param doc Document structure pointer
 * @return float Score value
 */
ZVEC_EXPORT float ZVEC_CALL zvec_doc_get_score(const ZVecDoc *doc);

/**
 * @brief Get document operator
 *
 * @param doc Document structure pointer
 * @return ZVecDocOperator Operator
 */
ZVEC_EXPORT ZVecDocOperator ZVEC_CALL zvec_doc_get_operator(const ZVecDoc *doc);

/**
 * @brief Get document field count
 *
 * @param doc Document structure pointer
 * @return size_t Field count
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_doc_get_field_count(const ZVecDoc *doc);

/**
 * @brief Get document primary key pointer (no copy)
 *
 * @param doc Document object pointer
 * @return const char* Primary key string pointer, returns NULL if not set
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_doc_get_pk_pointer(const ZVecDoc *doc);

/**
 * @brief Get document primary key copy (needs manual release)
 *
 * @param doc Document object pointer
 * @return const char* Primary key string copy, needs to call free() to release,
 * returns NULL if not set
 */
ZVEC_EXPORT const char *ZVEC_CALL zvec_doc_get_pk_copy(const ZVecDoc *doc);

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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_get_field_value_basic(
    const ZVecDoc *doc, const char *field_name, ZVecDataType field_type,
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
 * deallocation functions (free() for basic types and strings,
 * zvec_free_uint8_array() for binary data).
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @param field_type Field type
 * @param[out] value Returned value pointer (needs manual release)
 * @param[out] value_size Returned value size
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_get_field_value_copy(
    const ZVecDoc *doc, const char *field_name, ZVecDataType field_type,
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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_get_field_value_pointer(
    const ZVecDoc *doc, const char *field_name, ZVecDataType field_type,
    const void **value, size_t *value_size);

/**
 * @brief Check if document is empty
 *
 * @param doc Document object pointer
 * @return bool Returns true if document is empty, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_is_empty(const ZVecDoc *doc);

/**
 * @brief Check if document contains specified field
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @return bool Returns true if field exists, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_has_field(const ZVecDoc *doc,
                                              const char *field_name);

/**
 * @brief Check if document field has value
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @return bool Returns true if field has value, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_has_field_value(const ZVecDoc *doc,
                                                    const char *field_name);

/**
 * @brief Check if document field is null
 *
 * @param doc Document object pointer
 * @param field_name Field name
 * @return bool Returns true if field is null, otherwise returns false
 */
ZVEC_EXPORT bool ZVEC_CALL zvec_doc_is_field_null(const ZVecDoc *doc,
                                                  const char *field_name);

/**
 * @brief Get all field names of document
 *
 * @param doc Document object pointer
 * @param[out] field_names
 * Returned field name array (needs to call zvec_free_str_array to release)
 * @param[out] count Returned field count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_get_field_names(
    const ZVecDoc *doc, char ***field_names, size_t *count);

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
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_serialize(const ZVecDoc *doc,
                                                       uint8_t **data,
                                                       size_t *size);

/**
 * @brief Deserialize document
 *
 * @param data Serialized data
 * @param size Data size
 * @param[out] doc Returned document object pointer (needs to call
 * zvec_doc_destroy to release)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_doc_deserialize(const uint8_t *data,
                                                         size_t size,
                                                         ZVecDoc **doc);

/**
 * @brief Merge two documents
 *
 * @param doc Target document object pointer
 * @param other Source document object pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_doc_merge(ZVecDoc *doc, const ZVecDoc *other);

/**
 * @brief Get document memory usage
 *
 * @param doc Document object pointer
 * @return size_t Memory usage (bytes)
 */
ZVEC_EXPORT size_t ZVEC_CALL zvec_doc_memory_usage(const ZVecDoc *doc);

/**
 * @brief Validate document against Schema
 *
 * @param doc Document object pointer
 * @param schema Schema object pointer
 * @param is_update Whether it's an update operation
 * @param[out] error_msg Error message (needs manual release)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_doc_validate(const ZVecDoc *doc, const ZVecCollectionSchema *schema,
                  bool is_update, char **error_msg);

/**
 * @brief Get detailed string representation of document
 *
 * @param doc Document object pointer
 * @param[out] detail_str Returned detailed string (needs manual release)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_doc_to_detail_string(const ZVecDoc *doc, char **detail_str);

/**
 * @brief Free docs array memory
 * @param docs Document array pointer
 * @param count Document count
 */
ZVEC_EXPORT void ZVEC_CALL zvec_docs_free(ZVecDoc **docs, size_t count);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Convert error code to description string
 * @param error_code Error code
 * @return const char* Error description string
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_error_code_to_string(ZVecErrorCode error_code);

/**
 * @brief Convert data type to string
 * @param data_type Data type
 * @return const char* Data type string
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_data_type_to_string(ZVecDataType data_type);

/**
 * @brief Convert index type to string
 * @param index_type Index type
 * @return const char* Index type string
 */
ZVEC_EXPORT const char *ZVEC_CALL
zvec_index_type_to_string(ZVecIndexType index_type);

/**
 * @brief Convert metric type to string
 * @param metric_type Metric type
 * @return const char* Metric type string
 */
const char *zvec_metric_type_to_string(ZVecMetricType metric_type);

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
 * ZVecIndexParams params = ZVEC_HNSW_PARAMS(
 *     ZVEC_METRIC_TYPE_COSINE, 16, 200, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
 * @endcode
 */
// clang-format off
#define ZVEC_HNSW_PARAMS(_metric, _m, _ef_construction, _ef_search, _quant) \
  ((ZVecIndexParams){                                                       \
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
 * ZVecIndexParams params = ZVEC_INVERT_PARAMS(true, false);
 */
// clang-format off
#define ZVEC_INVERT_PARAMS(_range_opt, _wildcard) \
  ((ZVecIndexParams){                               \
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
  ((ZVecIndexParams){                     \
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
  ((ZVecIndexParams){                                                      \
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
 * ZVecString name = ZVEC_STRING("my_collection");
 */
#define ZVEC_STRING(str)               \
  (ZVecString) {                       \
    .data = str, .length = strlen(str) \
  }

/**
 * @brief Simplified string view initialization macro
 * @param str String content
 *
 * Usage example:
 * ZVecStringView name = ZVEC_STRING_VIEW("my_collection");
 */
#define ZVEC_STRING_VIEW(str)          \
  (ZVecStringView) {                   \
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
 * ZVecFloatArray vec_array = ZVEC_FLOAT_ARRAY(vectors, 3);
 */
#define ZVEC_FLOAT_ARRAY(data_ptr, len) \
  (ZVecFloatArray) {                    \
    .data = data_ptr, .length = len     \
  }

/**
 * @brief Simplified integer array initialization macro
 * @param data_ptr Integer array pointer
 * @param len Array length
 */
#define ZVEC_INT64_ARRAY(data_ptr, len) \
  (ZVecInt64Array) {                    \
    .data = data_ptr, .length = len     \
  }

/**
 * @brief Simplified document field initialization macro
 * @param name_str Field name
 * @param type Data type
 * @param value_union Field value union
 *
 * Usage example:
 * ZVecDocField field = ZVEC_DOC_FIELD("id", ZVEC_DATA_TYPE_STRING,
 *     {.string_value = ZVEC_STRING("doc1")});
 */
#define ZVEC_DOC_FIELD(name_str, type, value_union)                        \
  (ZVecDocField) {                                                         \
    .name = ZVEC_STRING(name_str), .data_type = type, .value = value_union \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ZVEC_C_API_H
