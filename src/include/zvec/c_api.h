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

/** @brief Major version number */
#define ZVEC_VERSION_MAJOR 0

/** @brief Minor version number */
#define ZVEC_VERSION_MINOR 3

/** @brief Patch version number */
#define ZVEC_VERSION_PATCH 0

/** @brief Full version string */
#define ZVEC_VERSION_STRING "0.3.0"

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

/**
 * @brief Console log configuration structure
 */
typedef struct {
  ZVecLogLevel level; /**< Log level */
} ZVecConsoleLogConfig;

/**
 * @brief File log configuration structure
 */
typedef struct {
  ZVecLogLevel level;    /**< Log level */
  ZVecString dir;        /**< Log directory */
  ZVecString basename;   /**< Log file base name */
  uint32_t file_size;    /**< Log file size (MB) */
  uint32_t overdue_days; /**< Log expiration days */
} ZVecFileLogConfig;

/**
 * @brief ZVec configuration data structure (corresponds to zvec::ConfigData)
 */
typedef struct {
  uint64_t memory_limit_bytes; /**< Memory limit in bytes */

  // log
  ZVecLogType log_type;
  void *log_config; /**< Log configuration (ZVecConsoleLogConfig or
                       ZVecFileLogConfig) */

  // query
  uint32_t query_thread_count;        /**< Query thread count */
  float invert_to_forward_scan_ratio; /**< Inverted to forward scan ratio */
  float brute_force_by_keys_ratio;    /**< Brute force by keys ratio */

  // optimize
  uint32_t optimize_thread_count; /**< Optimize thread count */
} ZVecConfigData;

/**
 * @brief Create console log configuration
 * @param level Log level
 * @return ZVecConsoleLogConfig* Pointer to the newly created console log
 * configuration
 */
ZVEC_EXPORT ZVecConsoleLogConfig *ZVEC_CALL
zvec_config_console_log_create(ZVecLogLevel level);

/**
 * @brief Create file log configuration
 * @param level Log level
 * @param dir Log directory
 * @param basename Log file base name
 * @param file_size Log file size (MB)
 * @param overdue_days Log expiration days
 * @return ZVecFileLogConfig* Pointer to the newly created file log
 * configuration
 */
ZVEC_EXPORT ZVecFileLogConfig *ZVEC_CALL zvec_config_file_log_create(
    ZVecLogLevel level, const char *dir, const char *basename,
    uint32_t file_size, uint32_t overdue_days);

/**
 * @brief Destroy console log configuration
 * @param config Console log configuration pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_config_console_log_destroy(ZVecConsoleLogConfig *config);

/**
 * @brief Destroy file log configuration
 * @param config File log configuration pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_config_file_log_destroy(ZVecFileLogConfig *config);


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
 * @brief Set log configuration in configuration data
 * @param config Configuration data pointer
 * @param log_config Log configuration pointer (ownership is transferred to
 * config, do not free separately)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_log_config(
    ZVecConfigData *config, ZVecLogType log_type, void *log_config);

/**
 * @brief Set query thread count in configuration data
 * @param config Configuration data pointer
 * @param thread_count Query thread count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_query_thread_count(
    ZVecConfigData *config, uint32_t thread_count);

/**
 * @brief Set optimize thread count in configuration data
 * @param config Configuration data pointer
 * @param thread_count Optimize thread count
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_config_data_set_optimize_thread_count(
    ZVecConfigData *config, uint32_t thread_count);

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
 * @param[out] initialized Whether initialized
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_is_initialized(bool *initialized);

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
// Forward Declarations
// =============================================================================

typedef struct ZVecCollection ZVecCollection;

// =============================================================================
// Index Parameters Structures
// =============================================================================

/**
 * @brief Base index parameters structure
 */
typedef struct {
  ZVecIndexType index_type; /**< Index type */
} ZVecBaseIndexParams;

/**
 * @brief Scalar index parameters structure
 */
typedef struct {
  ZVecBaseIndexParams base;       /**< Inherit base parameters */
  bool enable_range_optimization; /**< Whether to enable range optimization */
  bool enable_extended_wildcard;  /**< Whether to enable extended wildcard */
} ZVecInvertIndexParams;

/**
 * @brief Vector index base parameters structure
 */
typedef struct {
  ZVecBaseIndexParams base;       /**< Inherit base parameters */
  ZVecMetricType metric_type;     /**< Distance metric type */
  ZVecQuantizeType quantize_type; /**< Quantization type */
} ZVecVectorIndexParams;

/**
 * @brief HNSW index parameters structure
 */
typedef struct {
  ZVecVectorIndexParams base; /**< Inherit vector index parameters */
  int m;                      /**< Graph connectivity parameter */
  int ef_construction;        /**< Exploration factor during construction */
  int ef_search;              /**< Exploration factor during search */
} ZVecHnswIndexParams;

/**
 * @brief Flat index parameters structure
 */
typedef struct {
  ZVecVectorIndexParams base; /**< Inherit vector index parameters */
                              // Flat index has no additional parameters
} ZVecFlatIndexParams;

/**
 * @brief IVF index parameters structure
 */
typedef struct {
  ZVecVectorIndexParams base; /**< Inherit vector index parameters */
  int n_list;                 /**< Number of cluster centers */
  int n_iters;                /**< Number of iterations */
  bool use_soar;              /**< Whether to use SOAR algorithm */
  int n_probe;                /**< Number of clusters to probe during search */
} ZVecIVFIndexParams;

/**
 * @brief Generic index parameters union
 */
typedef struct {
  ZVecIndexType index_type; /**< Index type */
  union {
    ZVecInvertIndexParams invert_params; /**< Scalar index parameters */
    ZVecHnswIndexParams hnsw_params;     /**< HNSW index parameters */
    ZVecFlatIndexParams flat_params;     /**< Flat index parameters */
    ZVecIVFIndexParams ivf_params;       /**< IVF index parameters */
  } params;
} ZVecIndexParams;

// =============================================================================
// Field Schema Structures
// =============================================================================

/**
 * @brief Field schema structure
 */
typedef struct {
  ZVecString *name;       /**< Field name */
  ZVecDataType data_type; /**< Data type */
  bool nullable;          /**< Whether nullable */
  uint32_t dimension;     /**< Vector dimension (only used for vector fields) */
  ZVecIndexParams *index_params; /**< Index parameters, NULL means no index */
} ZVecFieldSchema;


// =============================================================================
// Index Parameters Creation and Destruction Interface
// =============================================================================

/**
 * @brief Initialize base index parameters
 * @param params Base index parameters structure pointer
 * @param index_type Index type
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_base_init(
    ZVecBaseIndexParams *params, ZVecIndexType index_type);

/**
 * @brief Initialize scalar index parameters
 * @param params Scalar index parameters structure pointer
 * @param enable_range_opt Whether to enable range optimization
 * @param enable_wildcard Whether to enable wildcard expansion
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_invert_init(
    ZVecInvertIndexParams *params, bool enable_range_opt, bool enable_wildcard);

/**
 * @brief Initialize vector index parameters
 * @param params Vector index parameters structure pointer
 * @param index_type Index type
 * @param metric_type Metric type
 * @param quantize_type Quantization type
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_vector_init(
    ZVecVectorIndexParams *params, ZVecIndexType index_type,
    ZVecMetricType metric_type, ZVecQuantizeType quantize_type);

/**
 * @brief Initialize HNSW index parameters
 * @param params HNSW index parameters structure pointer
 * @param metric_type Metric type
 * @param m Connectivity parameter
 * @param ef_construction Construction exploration factor
 * @param ef_search Search exploration factor
 * @param quantize_type Quantization type
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_hnsw_init(
    ZVecHnswIndexParams *params, ZVecMetricType metric_type, int m,
    int ef_construction, int ef_search, ZVecQuantizeType quantize_type);

/**
 * @brief Initialize Flat index parameters
 * @param params Flat index parameters structure pointer
 * @param metric_type Metric type
 * @param quantize_type Quantization type
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_flat_init(
    ZVecFlatIndexParams *params, ZVecMetricType metric_type,
    ZVecQuantizeType quantize_type);

/**
 * @brief Initialize IVF index parameters
 * @param params IVF index parameters structure pointer
 * @param metric_type Metric type
 * @param n_list Number of cluster centers
 * @param n_iters Number of iterations
 * @param use_soar Whether to use SOAR algorithm
 * @param n_probe Search probe count
 * @param quantize_type Quantization type
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_ivf_init(
    ZVecIVFIndexParams *params, ZVecMetricType metric_type, int n_list,
    int n_iters, bool use_soar, int n_probe, ZVecQuantizeType quantize_type);

/**
 * @brief Initialize generic index parameters
 * @param params Generic index parameters structure pointer
 * @param index_type Index type
 * @param metric_type Metric type (only valid for vector indexes)
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_init_default(
    ZVecIndexParams *params, ZVecIndexType index_type,
    ZVecMetricType metric_type);

/**
 * @brief Destroy index parameters (free internal dynamically allocated memory)
 * @param params Index parameters structure pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_index_params_destroy(ZVecIndexParams *params);


/**
 * @brief Create inverted index parameters
 * @param enable_range_opt Whether to enable range optimization
 * @param enable_wildcard Whether to enable extended wildcard
 * @return ZVecInvertIndexParams* Pointer to the newly created index parameters
 */
ZVEC_EXPORT ZVecInvertIndexParams *ZVEC_CALL
zvec_index_params_invert_create(bool enable_range_opt, bool enable_wildcard);

/**
 * @brief Create vector index base parameters
 * @param index_type Index type
 * @param metric_type Metric type
 * @param quantize_type Quantization type
 * @return ZVecVectorIndexParams* Pointer to the newly created index parameters
 */
ZVEC_EXPORT ZVecVectorIndexParams *ZVEC_CALL zvec_index_params_vector_create(
    ZVecIndexType index_type, ZVecMetricType metric_type,
    ZVecQuantizeType quantize_type);

/**
 * @brief Create HNSW index parameters
 * @param metric_type Metric type
 * @param quantize_type Quantization type
 * @param m Graph degree parameter
 * @param ef_construction Exploration factor during construction
 * @param ef_search Exploration factor during search

 * @return ZVecHnswIndexParams* Pointer to the newly created index parameters
 */
ZVEC_EXPORT ZVecHnswIndexParams *ZVEC_CALL zvec_index_params_hnsw_create(
    ZVecMetricType metric_type, ZVecQuantizeType quantize_type, int m,
    int ef_construction, int ef_search);

/**
 * @brief Create Flat index parameters
 * @param metric_type Metric type
 * @param quantize_type Quantization type
 * @return ZVecFlatIndexParams* Pointer to the newly created index parameters
 */
ZVEC_EXPORT ZVecFlatIndexParams *ZVEC_CALL zvec_index_params_flat_create(
    ZVecMetricType metric_type, ZVecQuantizeType quantize_type);

/**
 * @brief Create IVF index parameters
 * @param metric_type Metric type
 * @param n_list Number of cluster centers
 * @param n_iters Number of iterations
 * @param use_soar Whether to use SOAR algorithm
 * @param n_probe Number of clusters to probe during search
 * @param quantize_type Quantization type
 * @return ZVecIVFIndexParams* Pointer to the newly created index parameters
 */
ZVEC_EXPORT ZVecIVFIndexParams *ZVEC_CALL zvec_index_params_ivf_create(
    ZVecMetricType metric_type, ZVecQuantizeType quantize_type, int n_list,
    int n_iters, bool use_soar, int n_probe);


/**
 * @brief Destroy inverted index parameters
 * @param params Index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_index_params_invert_destroy(ZVecInvertIndexParams *params);

/**
 * @brief Destroy vector index parameters
 * @param params Index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_index_params_vector_destroy(ZVecVectorIndexParams *params);

/**
 * @brief Destroy HNSW index parameters
 * @param params Index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_index_params_hnsw_destroy(ZVecHnswIndexParams *params);

/**
 * @brief Destroy Flat index parameters
 * @param params Index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_index_params_flat_destroy(ZVecFlatIndexParams *params);

/**
 * @brief Destroy IVF index parameters
 * @param params Index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_index_params_ivf_destroy(ZVecIVFIndexParams *params);


// =============================================================================
// Query Parameters Structures
// =============================================================================

/**
 * @brief Base query parameters structure (corresponds to zvec::QueryParams)
 */
typedef struct {
  ZVecIndexType index_type; /**< Index type */
  float radius;             /**< Search radius */
  bool is_linear;           /**< Whether linear search */
  bool is_using_refiner;    /**< Whether using refiner */
} ZVecQueryParams;

/**
 * @brief HNSW query parameters structure (corresponds to zvec::HnswQueryParams)
 */
typedef struct {
  ZVecQueryParams base; /**< Inherit base query parameters */
  int ef;               /**< Exploration factor during search */
} ZVecHnswQueryParams;

/**
 * @brief IVF query parameters structure (corresponds to zvec::IVFQueryParams)
 */
typedef struct {
  ZVecQueryParams base; /**< Inherit base query parameters */
  int nprobe;           /**< Number of clusters to probe during search */
  float scale_factor;   /**< Scale factor */
} ZVecIVFQueryParams;

/**
 * @brief Flat query parameters structure (corresponds to zvec::FlatQueryParams)
 */
typedef struct {
  ZVecQueryParams base; /**< Inherit base query parameters */
  float scale_factor;   /**< Scale factor */
} ZVecFlatQueryParams;

/**
 * @brief Query parameters union (supports query parameters for different index
 * types)
 */
typedef struct {
  ZVecIndexType index_type; /**< Index type, used to distinguish the parameter
                               type stored in the union */
  union {
    ZVecQueryParams base_params;     /**< Base query parameters */
    ZVecHnswQueryParams hnsw_params; /**< HNSW query parameters */
    ZVecIVFQueryParams ivf_params;   /**< IVF query parameters */
    ZVecFlatQueryParams flat_params; /**< Flat query parameters */
  } params;
} ZVecQueryParamsUnion;

// =============================================================================
// Query Structures (Updated Version, Including QueryParams)
// =============================================================================

/**
 * @brief Vector query structure (aligned with zvec::VectorQuery, includes
 * QueryParams)
 */
typedef struct {
  int topk;                   /**< Number of results to return */
  ZVecString field_name;      /**< Query field name */
  ZVecByteArray query_vector; /**< Query vector (binary data) */
  ZVecByteArray
      query_sparse_indices;          /**< Sparse vector indices (binary data) */
  ZVecByteArray query_sparse_values; /**< Sparse vector values (binary data) */
  ZVecString filter;                 /**< Filter expression */
  bool include_vector;               /**< Whether to include vector data */
  bool include_doc_id;               /**< Whether to include document ID */
  ZVecStringArray output_fields;     /**< Output field list (NULL means all) */
  ZVecQueryParamsUnion *query_params; /**< Query parameters (optional, NULL
                                         means using default parameters) */
} ZVecVectorQuery;

/**
 * @brief Grouped vector query structure (aligned with zvec::GroupByVectorQuery,
 * includes QueryParams)
 */
typedef struct {
  ZVecString field_name;      /**< Query field name */
  ZVecByteArray query_vector; /**< Query vector (binary data) */
  ZVecByteArray
      query_sparse_indices;          /**< Sparse vector indices (binary data) */
  ZVecByteArray query_sparse_values; /**< Sparse vector values (binary data) */
  ZVecString filter;                 /**< Filter expression */
  bool include_vector;               /**< Whether to include vector data */
  ZVecStringArray output_fields;     /**< Output field list */
  ZVecString group_by_field_name;    /**< Group by field name */
  uint32_t group_count;              /**< Number of groups */
  uint32_t group_topk; /**< Number of results to return per group */
  ZVecQueryParamsUnion *query_params; /**< Query parameters (optional, NULL
                                         means using default parameters) */
} ZVecGroupByVectorQuery;


// =============================================================================
// Query Parameters Management Functions
// =============================================================================

/**
 * @brief Create base query parameters
 * @param index_type Index type
 * @return ZVecQueryParams* Pointer to the newly created query parameters
 */
ZVEC_EXPORT ZVecQueryParams *ZVEC_CALL
zvec_query_params_create(ZVecIndexType index_type);

/**
 * @brief Create HNSW query parameters
 * @param index_type Index type (should be ZVEC_INDEX_TYPE_HNSW)
 * @param ef Exploration factor during search
 * @param radius Search radius
 * @param is_linear Whether linear search
 * @param is_using_refiner Whether using refiner
 * @return ZVecHnswQueryParams* Pointer to the newly created HNSW query
 * parameters
 */
ZVEC_EXPORT ZVecHnswQueryParams *ZVEC_CALL
zvec_query_params_hnsw_create(ZVecIndexType index_type, int ef, float radius,
                              bool is_linear, bool is_using_refiner);

/**
 * @brief Create IVF query parameters
 * @param index_type Index type (should be ZVEC_INDEX_TYPE_IVF)
 * @param nprobe Number of clusters to probe during search
 * @param is_using_refiner Whether using refiner
 * @param scale_factor Scale factor
 * @return ZVecIVFQueryParams* Pointer to the newly created IVF query parameters
 */
ZVEC_EXPORT ZVecIVFQueryParams *ZVEC_CALL
zvec_query_params_ivf_create(ZVecIndexType index_type, int nprobe,
                             bool is_using_refiner, float scale_factor);

/**
 * @brief Create Flat query parameters
 * @param index_type Index type (should be ZVEC_INDEX_TYPE_FLAT)
 * @param is_using_refiner Whether using refiner
 * @param scale_factor Scale factor
 * @return ZVecFlatQueryParams* Pointer to the newly created Flat query
 * parameters
 */
ZVEC_EXPORT ZVecFlatQueryParams *ZVEC_CALL zvec_query_params_flat_create(
    ZVecIndexType index_type, bool is_using_refiner, float scale_factor);

/**
 * @brief Create query parameters union
 * @param index_type Index type
 * @return ZVecQueryParamsUnion* Pointer to the newly created query parameters
 * union
 */
ZVEC_EXPORT ZVecQueryParamsUnion *ZVEC_CALL
zvec_query_params_union_create(ZVecIndexType index_type);


/**
 * @brief Destroy base query parameters
 * @param params query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_query_params_destroy(ZVecQueryParams *params);

/**
 * @brief Destroy HNSW query parameters
 * @param params HNSW query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_hnsw_destroy(ZVecHnswQueryParams *params);

/**
 * @brief Destroy IVF query parameters
 * @param params IVF query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_ivf_destroy(ZVecIVFQueryParams *params);

/**
 * @brief Destroy Flat query parameters
 * @param params Flat query parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_flat_destroy(ZVecFlatQueryParams *params);

/**
 * @brief Destroy query parameters union
 * @param params Query parameters union pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_query_params_union_destroy(ZVecQueryParamsUnion *params);

/**
 * @brief Set query parameters index type
 * @param params Query parameters pointer
 * @param index_type Index type
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_set_index_type(
    ZVecQueryParams *params, ZVecIndexType index_type);

/**
 * @brief Set search radius for query parameters
 * @param params Query parameters pointer
 * @param radius Search radius
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_set_radius(ZVecQueryParams *params, float radius);

/**
 * @brief Set scale factor for query parameters
 * @param params Query parameters pointer
 * @param scale_factor Scale factor
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_set_is_linear(ZVecQueryParams *params, bool is_linear);

/**
 * @brief Set whether to use refiner for query parameters
 * @param params Query parameters pointer
 * @param is_using_refiner Whether to use refiner
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_set_is_using_refiner(
    ZVecQueryParams *params, bool is_using_refiner);

/**
 * @brief Set exploration factor for HNSW query parameters
 * @param params HNSW query parameters pointer
 * @param ef Exploration factor
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_hnsw_set_ef(ZVecHnswQueryParams *params, int ef);

/**
 * @brief Set number of probe clusters for IVF query parameters
 * @param params IVF query parameters pointer
 * @param nprobe Number of probe clusters
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_query_params_ivf_set_nprobe(ZVecIVFQueryParams *params, int nprobe);

/**
 * @brief Set scale factor for IVF/Flat query parameters
 * @param params IVF or Flat query parameters pointer
 * @param scale_factor Scale factor
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_query_params_ivf_set_scale_factor(
    ZVecIVFQueryParams *params, float scale_factor);

/**
 * @brief Collection options structure
 */
typedef struct {
  bool enable_mmap;                   /**< Whether to enable memory mapping */
  size_t max_buffer_size;             /**< Maximum buffer size */
  bool read_only;                     /**< Whether read-only mode */
  uint64_t max_doc_count_per_segment; /**< Maximum document count per segment */
} ZVecCollectionOptions;


/**
 * @brief Collection statistics structure
 */
typedef struct {
  uint64_t doc_count;        /**< Total document count */
  ZVecString **index_names;  /**< Index name array */
  float *index_completeness; /**< Index completeness array */
  size_t index_count;        /**< Index name count */
} ZVecCollectionStats;


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
 * @brief Set index parameters for field
 * @param schema Field schema pointer
 * @param index_params Index parameters pointer
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_field_schema_set_index_params(
    ZVecFieldSchema *schema, const ZVecIndexParams *index_params);


/**
 * @brief Set inverted index parameters for field schema
 * @param field_schema Field schema pointer
 * @param invert_params Inverted index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_field_schema_set_invert_index(
    ZVecFieldSchema *field_schema, const ZVecInvertIndexParams *invert_params);

/**
 * @brief Set HNSW index parameters for field schema
 * @param field_schema Field schema pointer
 * @param hnsw_params HNSW index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_field_schema_set_hnsw_index(
    ZVecFieldSchema *field_schema, const ZVecHnswIndexParams *hnsw_params);

/**
 * @brief Set Flat index parameters for field schema
 * @param field_schema Field schema pointer
 * @param flat_params Flat index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_field_schema_set_flat_index(
    ZVecFieldSchema *field_schema, const ZVecFlatIndexParams *flat_params);

/**
 * @brief Set IVF index parameters for field schema
 * @param field_schema Field schema pointer
 * @param ivf_params IVF index parameters pointer
 */
ZVEC_EXPORT void ZVEC_CALL zvec_field_schema_set_ivf_index(
    ZVecFieldSchema *field_schema, const ZVecIVFIndexParams *ivf_params);


// =============================================================================
// Collection Schema Structures
// =============================================================================

/**
 * @brief Collection schema structure
 */
typedef struct {
  ZVecString *name;                   /**< Collection name */
  ZVecFieldSchema **fields;           /**< Field array */
  size_t field_count;                 /**< Field count */
  size_t field_capacity;              /**< Field array capacity */
  uint64_t max_doc_count_per_segment; /**< Maximum document count per segment */
} ZVecCollectionSchema;

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
 * @brief Add field to collection schema
 * @param schema Collection schema pointer
 * @param field Field schema pointer (function takes ownership)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_add_field(
    ZVecCollectionSchema *schema, ZVecFieldSchema *field);

/**
 * @brief Add multiple fields to collection schema at once
 *
 * @param schema Collection schema pointer
 * @param fields Array of fields to add
 * @param field_count Number of fields to add
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_add_fields(
    ZVecCollectionSchema *schema, const ZVecFieldSchema *fields,
    size_t field_count);

/**
 * @brief Remove field
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_remove_field(
    ZVecCollectionSchema *schema, const char *field_name);

/**
 * @brief Remove multiple fields from collection schema at once
 *
 * @param schema Collection schema pointer
 * @param field_names Array of field names to remove
 * @param field_count Number of fields to remove
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_schema_remove_fields(
    ZVecCollectionSchema *schema, const char *const *field_names,
    size_t field_count);

/**
 * @brief Get field count
 *
 * @param schema Collection schema pointer
 * @return size_t Field count
 */
ZVEC_EXPORT size_t ZVEC_CALL
zvec_collection_schema_get_field_count(const ZVecCollectionSchema *schema);

/**
 * @brief Find field
 * @param schema Collection schema pointer
 * @param field_name Field name
 * @return ZVecFieldSchema* Field schema pointer, returns NULL if not found
 */
ZVEC_EXPORT ZVecFieldSchema *ZVEC_CALL zvec_collection_schema_find_field(
    const ZVecCollectionSchema *schema, const char *field_name);

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
 * @brief Get field by index
 * @param schema Collection schema pointer
 * @param index Field index
 * @return ZVecFieldSchema* Field schema pointer
 */
ZVEC_EXPORT ZVecFieldSchema *ZVEC_CALL zvec_collection_schema_get_field(
    const ZVecCollectionSchema *schema, size_t index);

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
 * @brief Get maximum document count per segment of collection schema
 *
 * @param schema Collection schema pointer
 * @return uint64_t Maximum document count per segment
 */
ZVEC_EXPORT uint64_t ZVEC_CALL
zvec_collection_schema_get_max_doc_count_per_segment(
    const ZVecCollectionSchema *schema);


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
 * @brief Initialize default collection options
 * @param options Collection options structure pointer
 */
ZVEC_EXPORT void ZVEC_CALL
zvec_collection_options_init_default(ZVecCollectionOptions *options);

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
 * @brief Create index for collection field (using specific type parameters)
 * @param collection Collection handle
 * @param field_name Field name
 * @param index_params Index parameters (select appropriate structure based on
 * index type)
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_index_with_params(
    ZVecCollection *collection, const char *field_name,
    const void
        *index_params);  // Determine specific type based on index_type field

/**
 * @brief Create HNSW index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param hnsw_params HNSW index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_hnsw_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecHnswIndexParams *hnsw_params);

/**
 * @brief Create Flat index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param flat_params Flat index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_flat_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecFlatIndexParams *flat_params);

/**
 * @brief Create IVF index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param ivf_params IVF index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_ivf_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecIVFIndexParams *ivf_params);

/**
 * @brief Create scalar index for collection field
 * @param collection Collection handle
 * @param field_name Field name
 * @param invert_params Scalar index parameters
 * @return Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_create_invert_index(
    ZVecCollection *collection, const char *field_name,
    const ZVecInvertIndexParams *invert_params);

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
 * @param field_schema Field schema pointer
 * @param default_expression Default value expression (can be NULL)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_add_column(
    ZVecCollection *collection, const ZVecFieldSchema *field_schema,
    const char *default_expression);

/**
 * @brief Drop column
 * @param collection Collection handle
 * @param field_name Field name
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL
zvec_collection_drop_column(ZVecCollection *collection, const char *field_name);

/**
 * @brief Alter column
 * @param collection Collection handle
 * @param old_name Original field name
 * @param new_name New field name (can be NULL to indicate no renaming)
 * @param new_schema New field schema (can be NULL to indicate no schema
 * modification)
 * @return ZVecErrorCode Error code
 */
ZVEC_EXPORT ZVecErrorCode ZVEC_CALL zvec_collection_alter_column(
    ZVecCollection *collection, const char *old_name, const char *new_name,
    const ZVecFieldSchema *new_schema);


/**
 * @brief Document structure (opaque pointer mode)
 * Internal implementation details are not visible to the outside, and
 * operations are performed through API functions
 */
typedef struct ZVecDoc ZVecDoc;

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
 * @param metric Distance metric type
 * @param m_ Connectivity parameter
 * @param ef_construction Exploration factor during construction
 * @param ef_search Exploration factor during search
 * @param quant Quantization type
 *
 * Usage example:
 * ZVecHnswIndexParams params = ZVEC_HNSW_PARAMS(ZVEC_METRIC_TYPE_COSINE, 16,
 * 200, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
 */
#define ZVEC_HNSW_PARAMS(metric, m_, ef_construction, ef_search, quant)       \
  (ZVecHnswIndexParams) {                                                     \
    .base.base.index_type = ZVEC_INDEX_TYPE_HNSW, .base.metric_type = metric, \
    .base.quantize_type = quant, .m = m_, .ef_construction = ef_construction, \
    .ef_search = ef_search                                                    \
  }

/**
 * @brief Simplified inverted index parameters initialization macro
 * @param range_opt Whether to enable range optimization
 * @param wildcard Whether to enable wildcard expansion
 *
 * Usage example:
 * ZVecInvertIndexParams params = ZVEC_INVERT_PARAMS(true, false);
 */
#define ZVEC_INVERT_PARAMS(range_opt, wildcard) \
  (ZVecInvertIndexParams) {                     \
    .base.index_type = ZVEC_INDEX_TYPE_INVERT,  \
    .enable_range_optimization = range_opt,     \
    .enable_extended_wildcard = wildcard        \
  }

/**
 * @brief Simplified Flat index parameters initialization macro
 * @param metric Distance metric type
 * @param quant Quantization type
 */
#define ZVEC_FLAT_PARAMS(metric, quant)                                  \
  (ZVecFlatIndexParams) {                                                \
    .base.index_type = ZVEC_INDEX_TYPE_FLAT, .base.metric_type = metric, \
    .base.quantize_type = quant                                          \
  }

/**
 * @brief Simplified IVF index parameters initialization macro
 * @param metric Distance metric type
 * @param nlist Number of cluster centers
 * @param niters Number of iterations
 * @param soar Whether to use SOAR algorithm
 * @param nprobe Number of clusters to probe during search
 * @param quant Quantization type
 */
#define ZVEC_IVF_PARAMS(metric, nlist, niters, soar, nprobe, quant)     \
  (ZVecIVFIndexParams) {                                                \
    .base.index_type = ZVEC_INDEX_TYPE_IVF, .base.metric_type = metric, \
    .base.quantize_type = quant, .n_list = nlist, .n_iters = niters,    \
    .use_soar = soar, .n_probe = nprobe                                 \
  }

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
 * @brief Simplified collection options initialization macro (using default
 * values)
 *
 * Usage example:
 * ZVecCollectionOptions opts = ZVEC_DEFAULT_OPTIONS();
 */
#define ZVEC_DEFAULT_OPTIONS()                                           \
  (ZVecCollectionOptions) {                                              \
    .enable_mmap = true, .max_buffer_size = 1048576, .read_only = false, \
    .max_doc_count_per_segment = 1000000                                 \
  }

/**
 * @brief Simplified vector query initialization macro
 * @param field_name_str Query field name
 * @param query_vec Query vector array
 * @param top_k Number of results to return
 * @param filter_str Filter condition string
 *
 * Usage example:
 * ZVecVectorQuery query = ZVEC_VECTOR_QUERY("embedding", query_vectors, 10,
 * "");
 */
#define ZVEC_VECTOR_QUERY(field_name_str, query_vec, top_k, filter_str)    \
  (ZVecVectorQuery) {                                                      \
    .field_name = ZVEC_STRING(field_name_str), .query_vector = query_vec,  \
    .topk = top_k, .filter = ZVEC_STRING(filter_str), .include_vector = 1, \
    .include_doc_id = 1                                                    \
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
