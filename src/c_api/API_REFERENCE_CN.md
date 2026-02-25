# ZVec C API 参考文档

**版本**: 0.3.0
**许可**: Apache License 2.0

---

## 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [版本管理](#版本管理)
4. [错误处理](#错误处理)
5. [初始化与关闭](#初始化与关闭)
6. [配置管理](#配置管理)
7. [数据结构](#数据结构)
8. [Schema 管理](#schema-管理)
9. [Collection 管理](#collection-管理)
10. [索引管理](#索引管理)
11. [文档操作](#文档操作)
12. [数据增删改](#数据增删改)
13. [数据查询](#数据查询)
14. [工具函数](#工具函数)
15. [完整示例](#完整示例)

---

## 概述

ZVec C API 是 ZVec 向量数据库的 C 语言接口，提供了完整的向量存储、索引和检索功能。本接口采用 C ABI，可与 C、C++、Rust、Go 等语言互操作。

### 核心概念

| 概念 | 说明 |
|------|------|
| **Collection** | 数据集合，类似数据库中的表 |
| **Schema** | 集合的结构定义，包含字段信息 |
| **Document** | 单条数据记录 |
| **Index** | 字段索引，加速查询 |
| **Field** | 字段，支持标量和向量类型 |

---

## 快速开始

### 最小可用示例

```c
#include "zvec/c_api.h"
#include <stdio.h>

int main() {
    // 1. 初始化库
    zvec_initialize(NULL);

    // 2. 创建集合 Schema
    ZVecCollectionSchema *schema = zvec_collection_schema_create("my_collection");
    ZVecFieldSchema *field = zvec_field_schema_create(
        "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 3);
    zvec_collection_schema_add_field(schema, field);

    // 3. 创建并打开集合
    ZVecCollection *collection = NULL;
    ZVecErrorCode rc = zvec_collection_create_and_open(
        "./my_data", schema, NULL, &collection);

    if (rc != ZVEC_OK) {
        char *err_msg;
        zvec_get_last_error(&err_msg);
        printf("Error: %s\n", err_msg);
        return 1;
    }

    // 4. 创建索引
    ZVecHnswIndexParams *params = zvec_index_params_hnsw_create(
        ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 200, 50);
    zvec_collection_create_hnsw_index(collection, "embedding", params);

    // 5. 插入数据
    ZVecDoc *doc = zvec_doc_create();
    zvec_doc_set_pk(doc, "doc_001");
    float vec[] = {0.1f, 0.2f, 0.3f};
    zvec_doc_add_field_by_value(doc, "embedding",
        ZVEC_DATA_TYPE_VECTOR_FP32, vec, sizeof(vec));

    size_t success, errors;
    zvec_collection_insert(collection, &doc, 1, &success, &errors);
    zvec_doc_destroy(doc);

    // 6. 查询
    ZVecVectorQuery query = ZVEC_VECTOR_QUERY(
        "embedding", ZVEC_FLOAT_ARRAY(vec, 3), 10, "");
    ZVecDoc **results;
    size_t count;
    zvec_collection_query(collection, &query, &results, &count);

    // 7. 清理
    zvec_docs_free(results, count);
    zvec_index_params_hnsw_destroy(params);
    zvec_collection_close(collection);
    zvec_collection_destroy(collection);
    zvec_collection_schema_destroy(schema);
    zvec_shutdown();

    return 0;
}
```

---

## 版本管理

### 获取版本信息

```c
// 获取完整版本字符串
const char *version = zvec_get_version();
// 输出示例："0.3.0-g3f8a2b1 (built 2025-05-13 10:30:45)"

// 获取各部分版本号
int major = zvec_get_version_major();  // 0
int minor = zvec_get_version_minor();  // 3
int patch = zvec_get_version_patch();  // 0
```

### 版本兼容性检查

```c
// 检查当前库版本是否满足最低要求
bool compatible = zvec_check_version(0, 2, 0);
if (!compatible) {
    printf("Library version too old!\n");
}
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_get_version()` | 无 | `const char*` | 获取完整版本字符串 |
| `zvec_get_version_major()` | 无 | `int` | 获取主版本号 |
| `zvec_get_version_minor()` | 无 | `int` | 获取次版本号 |
| `zvec_get_version_patch()` | 无 | `int` | 获取补丁版本号 |
| `zvec_check_version()` | `major, minor, patch` | `bool` | 检查版本兼容性 |

---

## 错误处理

### 错误码枚举

```c
typedef enum {
  ZVEC_OK = 0,                        // 成功
  ZVEC_ERROR_NOT_FOUND = 1,           // 资源未找到
  ZVEC_ERROR_ALREADY_EXISTS = 2,      // 资源已存在
  ZVEC_ERROR_INVALID_ARGUMENT = 3,    // 无效参数
  ZVEC_ERROR_PERMISSION_DENIED = 4,   // 权限拒绝
  ZVEC_ERROR_FAILED_PRECONDITION = 5, // 前置条件失败
  ZVEC_ERROR_RESOURCE_EXHAUSTED = 6,  // 资源耗尽
  ZVEC_ERROR_UNAVAILABLE = 7,         // 服务不可用
  ZVEC_ERROR_INTERNAL_ERROR = 8,      // 内部错误
  ZVEC_ERROR_NOT_SUPPORTED = 9,       // 不支持的操作
  ZVEC_ERROR_UNKNOWN = 10             // 未知错误
} ZVecErrorCode;
```

### 获取错误信息

```c
// 获取详细错误信息
ZVecErrorDetails details;
zvec_get_last_error_details(&details);
printf("Error %d: %s\n", details.code, details.message);
printf("  at %s:%d in %s()\n", details.file, details.line, details.function);

// 获取错误消息字符串
char *error_msg;
ZVecErrorCode code = zvec_get_last_error(&error_msg);
if (code != ZVEC_OK) {
    printf("Error: %s\n", error_msg);
    free(error_msg);  // 需要调用者释放
}

// 清除错误状态
zvec_clear_error();

// 错误码转字符串
const char *err_str = zvec_error_code_to_string(ZVEC_ERROR_INVALID_ARGUMENT);
// 返回："Invalid argument"
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_get_last_error(&msg)` | `char **msg` | `ZVecErrorCode` | 获取最后错误消息 |
| `zvec_get_last_error_details(&details)` | `ZVecErrorDetails*` | `ZVecErrorCode` | 获取详细错误信息 |
| `zvec_clear_error()` | 无 | void | 清除错误状态 |
| `zvec_error_code_to_string(code)` | `ZVecErrorCode` | `const char*` | 错误码转字符串 |

---

## 初始化与关闭

### 初始化库

```c
// 使用默认配置初始化
ZVecErrorCode rc = zvec_initialize(NULL);

// 使用自定义配置初始化
ZVecConfigData *config = zvec_config_data_create();
zvec_config_data_set_memory_limit(config, 2UL * 1024 * 1024 * 1024); // 2GB
zvec_config_data_set_query_thread_count(config, 4);
rc = zvec_initialize(config);
zvec_config_data_destroy(config);

if (rc != ZVEC_OK) {
    // 处理初始化失败
}
```

### 关闭库

```c
// 关闭前确保所有 Collection 已关闭
zvec_collection_close(collection);
zvec_collection_destroy(collection);

// 关闭库，释放所有资源
ZVecErrorCode rc = zvec_shutdown();
```

### 检查初始化状态

```c
bool initialized;
zvec_is_initialized(&initialized);
if (!initialized) {
    zvec_initialize(NULL);
}
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_initialize(config)` | `const ZVecConfigData*` | `ZVecErrorCode` | 初始化库 |
| `zvec_shutdown()` | 无 | `ZVecErrorCode` | 关闭库 |
| `zvec_is_initialized(&initialized)` | `bool*` | `ZVecErrorCode` | 检查是否已初始化 |

---

## 配置管理

### 配置数据结构

```c
typedef struct {
  uint64_t memory_limit_bytes;    // 内存限制（字节）

  // 日志配置
  ZVecLogType log_type;
  void *log_config;               // ZVecConsoleLogConfig 或 ZVecFileLogConfig

  // 查询配置
  uint32_t query_thread_count;            // 查询线程数
  float invert_to_forward_scan_ratio;     // 倒排转正扫比例
  float brute_force_by_keys_ratio;        // 暴力检索比例

  // 优化配置
  uint32_t optimize_thread_count;         // 优化线程数
} ZVecConfigData;
```

### 日志配置

```c
// 控制台日志配置
typedef struct {
  ZVecLogLevel level;  // 日志级别
} ZVecConsoleLogConfig;

// 文件日志配置
typedef struct {
  ZVecLogLevel level;      // 日志级别
  ZVecString dir;          // 日志目录
  ZVecString basename;     // 日志文件基础名
  uint32_t file_size;      // 文件大小 (MB)
  uint32_t overdue_days;   // 过期天数
} ZVecFileLogConfig;
```

### 日志级别

```c
typedef enum {
  ZVEC_LOG_LEVEL_DEBUG = 0,
  ZVEC_LOG_LEVEL_INFO = 1,
  ZVEC_LOG_LEVEL_WARN = 2,
  ZVEC_LOG_LEVEL_ERROR = 3,
  ZVEC_LOG_LEVEL_FATAL = 4
} ZVecLogLevel;
```

### 配置创建与销毁

```c
// 创建配置
ZVecConfigData *config = zvec_config_data_create();

// 创建控制台日志配置
ZVecConsoleLogConfig *console_log = zvec_config_console_log_create(
    ZVEC_LOG_LEVEL_INFO);

// 创建文件日志配置
ZVecFileLogConfig *file_log = zvec_config_file_log_create(
    ZVEC_LOG_LEVEL_DEBUG,
    "/var/log/zvec",    // 日志目录
    "zvec",             // 基础文件名
    100,                // 文件大小 100MB
    30                  // 保留 30 天
);

// 设置配置
zvec_config_data_set_memory_limit(config, 1024 * 1024 * 1024);
zvec_config_data_set_log_config(config, ZVEC_LOG_TYPE_CONSOLE, console_log);
zvec_config_data_set_query_thread_count(config, 8);
zvec_config_data_set_optimize_thread_count(config, 4);

// 销毁配置
zvec_config_console_log_destroy(console_log);
zvec_config_file_log_destroy(file_log);
zvec_config_data_destroy(config);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_config_data_create()` | 无 | `ZVecConfigData*` | 创建配置数据 |
| `zvec_config_data_destroy(config)` | `ZVecConfigData*` | void | 销毁配置数据 |
| `zvec_config_data_set_memory_limit(config, bytes)` | config, 字节数 | `ZVecErrorCode` | 设置内存限制 |
| `zvec_config_data_set_log_config(config, type, cfg)` | config, 类型，配置 | `ZVecErrorCode` | 设置日志配置 |
| `zvec_config_data_set_query_thread_count(config, count)` | config, 线程数 | `ZVecErrorCode` | 设置查询线程数 |
| `zvec_config_data_set_optimize_thread_count(config, count)` | config, 线程数 | `ZVecErrorCode` | 设置优化线程数 |
| `zvec_config_console_log_create(level)` | 日志级别 | `ZVecConsoleLogConfig*` | 创建控制台日志配置 |
| `zvec_config_console_log_destroy(cfg)` | 配置指针 | void | 销毁控制台日志配置 |
| `zvec_config_file_log_create(...)` | 级别，目录，文件名，大小，天数 | `ZVecFileLogConfig*` | 创建文件日志配置 |
| `zvec_config_file_log_destroy(cfg)` | 配置指针 | void | 销毁文件日志配置 |

---

## 数据结构

### 字符串类型

```c
// 字符串视图（不拥有内存）
typedef struct {
  const char *data;
  size_t length;
} ZVecStringView;

// 可变字符串（拥有内存）
typedef struct {
  char *data;
  size_t length;
  size_t capacity;
} ZVecString;

// 字符串数组
typedef struct {
  ZVecString *strings;
  size_t count;
} ZVecStringArray;
```

### 数组类型

```c
// Float 数组
typedef struct {
  const float *data;
  size_t length;
} ZVecFloatArray;

// Int64 数组
typedef struct {
  const int64_t *data;
  size_t length;
} ZVecInt64Array;

// 字节数组
typedef struct {
  const uint8_t *data;
  size_t length;
} ZVecByteArray;

// 可变字节数组
typedef struct {
  uint8_t *data;
  size_t length;
  size_t capacity;
} ZVecMutableByteArray;
```

### 字符串操作

```c
// 从 C 字符串创建
ZVecString *str = zvec_string_create("Hello, World!");

// 从字符串视图创建
ZVecStringView view = {"Hello", 5};
ZVecString *str2 = zvec_string_create_from_view(&view);

// 创建二进制安全字符串（可包含 null 字节）
uint8_t data[] = {0x00, 0x01, 0x02, 0x03};
ZVecString *bin_str = zvec_bin_create(data, sizeof(data));

// 复制字符串
ZVecString *copy = zvec_string_copy(str);

// 获取 C 字符串
const char *c_str = zvec_string_c_str(str);

// 获取长度
size_t len = zvec_string_length(str);

// 比较字符串
int cmp = zvec_string_compare(str1, str2);  // 返回 -1, 0, 1

// 释放字符串
zvec_free_string(str);
```

### 数组操作

```c
// 创建字符串数组
ZVecStringArray *arr = zvec_string_array_create(10);

// 添加字符串
zvec_string_array_add(arr, 0, "first");
zvec_string_array_add(arr, 1, "second");

// 销毁字符串数组
zvec_string_array_destroy(arr);

// 创建字节数组
ZVecMutableByteArray *byte_arr = zvec_byte_array_create(1024);
zvec_byte_array_destroy(byte_arr);

// 创建 float 数组
ZVecFloatArray *float_arr = zvec_float_array_create(100);
zvec_float_array_destroy(float_arr);

// 创建 int64 数组
ZVecInt64Array *int_arr = zvec_int64_array_create(50);
zvec_int64_array_destroy(int_arr);

// 释放 uint8 数组
zvec_free_uint8_array(uint8_t *array);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_string_create(str)` | `const char*` | `ZVecString*` | 从 C 字符串创建 |
| `zvec_string_create_from_view(view)` | `ZVecStringView*` | `ZVecString*` | 从视图创建字符串 |
| `zvec_bin_create(data, length)` | `uint8_t*`, size_t | `ZVecString*` | 创建二进制字符串 |
| `zvec_string_copy(str)` | `ZVecString*` | `ZVecString*` | 复制字符串 |
| `zvec_string_c_str(str)` | `ZVecString*` | `const char*` | 获取 C 字符串 |
| `zvec_string_length(str)` | `ZVecString*` | size_t | 获取长度 |
| `zvec_string_compare(s1, s2)` | 两个字符串 | int | 比较字符串 |
| `zvec_free_string(str)` | `ZVecString*` | void | 释放字符串 |
| `zvec_string_array_create(count)` | size_t | `ZVecStringArray*` | 创建字符串数组 |
| `zvec_string_array_add(arr, idx, str)` | arr, 索引，字符串 | void | 添加字符串 |
| `zvec_string_array_destroy(arr)` | `ZVecStringArray*` | void | 销毁字符串数组 |
| `zvec_byte_array_create(capacity)` | size_t | `ZVecMutableByteArray*` | 创建字节数组 |
| `zvec_byte_array_destroy(arr)` | `ZVecMutableByteArray*` | void | 销毁字节数组 |
| `zvec_float_array_create(count)` | size_t | `ZVecFloatArray*` | 创建 float 数组 |
| `zvec_float_array_destroy(arr)` | `ZVecFloatArray*` | void | 销毁 float 数组 |
| `zvec_int64_array_create(count)` | size_t | `ZVecInt64Array*` | 创建 int64 数组 |
| `zvec_int64_array_destroy(arr)` | `ZVecInt64Array*` | void | 销毁 int64 数组 |
| `zvec_free_uint8_array(arr)` | `uint8_t*` | void | 释放 uint8 数组 |

---

## Schema 管理

### 数据类型

```c
typedef enum {
  // 标量类型
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

  // 向量类型
  ZVEC_DATA_TYPE_VECTOR_BINARY32 = 20,
  ZVEC_DATA_TYPE_VECTOR_BINARY64 = 21,
  ZVEC_DATA_TYPE_VECTOR_FP16 = 22,
  ZVEC_DATA_TYPE_VECTOR_FP32 = 23,
  ZVEC_DATA_TYPE_VECTOR_FP64 = 24,
  ZVEC_DATA_TYPE_VECTOR_INT4 = 25,
  ZVEC_DATA_TYPE_VECTOR_INT8 = 26,
  ZVEC_DATA_TYPE_VECTOR_INT16 = 27,

  // 稀疏向量类型
  ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 = 30,
  ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 = 31,

  // 数组类型
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
```

### 字段 Schema

```c
typedef struct {
  ZVecString *name;           // 字段名
  ZVecDataType data_type;     // 数据类型
  bool nullable;              // 是否可空
  uint32_t dimension;         // 向量维度（仅向量类型使用）
  ZVecIndexParams *index_params;  // 索引参数
} ZVecFieldSchema;
```

### 创建字段 Schema

```c
// 创建标量字段
ZVecFieldSchema *id_field = zvec_field_schema_create(
    "id", ZVEC_DATA_TYPE_STRING, false, 0);

// 创建向量字段（768 维）
ZVecFieldSchema *embedding_field = zvec_field_schema_create(
    "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 768);

// 创建带索引的字段
ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
    ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 200, 50);
zvec_field_schema_set_hnsw_index(embedding_field, hnsw_params);

// 或者使用专用函数
zvec_field_schema_set_invert_index(field, invert_params);
zvec_field_schema_set_hnsw_index(field, hnsw_params);
zvec_field_schema_set_flat_index(field, flat_params);
zvec_field_schema_set_ivf_index(field, ivf_params);

// 设置索引参数
zvec_field_schema_set_index_params(field, index_params);

// 销毁字段 Schema
zvec_field_schema_destroy(field);
zvec_free_field_schema(field);
```

### Collection Schema

```c
typedef struct {
  ZVecString *name;                   // 集合名
  ZVecFieldSchema **fields;           // 字段数组
  size_t field_count;                 // 字段数量
  size_t field_capacity;              // 字段容量
  uint64_t max_doc_count_per_segment; // 每段最大文档数
} ZVecCollectionSchema;
```

### 创建 Collection Schema

```c
// 创建 Schema
ZVecCollectionSchema *schema = zvec_collection_schema_create("my_collection");

// 添加单个字段
ZVecFieldSchema *field = zvec_field_schema_create(
    "title", ZVEC_DATA_TYPE_STRING, false, 0);
zvec_collection_schema_add_field(schema, field);

// 批量添加字段
ZVecFieldSchema fields[3] = {
    *zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0),
    *zvec_field_schema_create("embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 768),
    *zvec_field_schema_create("timestamp", ZVEC_DATA_TYPE_INT64, true, 0)
};
zvec_collection_schema_add_fields(schema, fields, 3);

// 获取字段数量
size_t count = zvec_collection_schema_get_field_count(schema);

// 按索引获取字段
ZVecFieldSchema *f = zvec_collection_schema_get_field(schema, 0);

// 按名称查找字段
ZVecFieldSchema *f = zvec_collection_schema_find_field(schema, "embedding");

// 删除字段
zvec_collection_schema_remove_field(schema, "title");

// 批量删除字段
const char *field_names[] = {"field1", "field2"};
zvec_collection_schema_remove_fields(schema, field_names, 2);

// 设置每段最大文档数
zvec_collection_schema_set_max_doc_count_per_segment(schema, 500000);

// 获取每段最大文档数
uint64_t max_docs = zvec_collection_schema_get_max_doc_count_per_segment(schema);

// 验证 Schema
ZVecString *error_msg;
ZVecErrorCode rc = zvec_collection_schema_validate(schema, &error_msg);
if (rc != ZVEC_OK) {
    printf("Invalid schema: %s\n", error_msg->data);
    zvec_free_string(error_msg);
}

// 销毁 Schema
zvec_collection_schema_destroy(schema);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_field_schema_create(name, type, nullable, dim)` | 名，类型，是否可空，维度 | `ZVecFieldSchema*` | 创建字段 Schema |
| `zvec_field_schema_destroy(schema)` | `ZVecFieldSchema*` | void | 销毁字段 Schema |
| `zvec_field_schema_set_index_params(schema, params)` | schema, 索引参数 | `ZVecErrorCode` | 设置索引参数 |
| `zvec_field_schema_set_invert_index(schema, params)` | schema, 倒排参数 | void | 设置倒排索引 |
| `zvec_field_schema_set_hnsw_index(schema, params)` | schema, HNSW 参数 | void | 设置 HNSW 索引 |
| `zvec_field_schema_set_flat_index(schema, params)` | schema, Flat 参数 | void | 设置 Flat 索引 |
| `zvec_field_schema_set_ivf_index(schema, params)` | schema, IVF 参数 | void | 设置 IVF 索引 |
| `zvec_free_field_schema(schema)` | `ZVecFieldSchema*` | void | 释放字段 Schema |
| `zvec_collection_schema_create(name)` | 集合名 | `ZVecCollectionSchema*` | 创建集合 Schema |
| `zvec_collection_schema_destroy(schema)` | `ZVecCollectionSchema*` | void | 销毁集合 Schema |
| `zvec_collection_schema_add_field(schema, field)` | schema, 字段 | `ZVecErrorCode` | 添加字段 |
| `zvec_collection_schema_add_fields(schema, fields, count)` | schema, 字段数组，数量 | `ZVecErrorCode` | 批量添加字段 |
| `zvec_collection_schema_remove_field(schema, name)` | schema, 字段名 | `ZVecErrorCode` | 删除字段 |
| `zvec_collection_schema_remove_fields(schema, names, count)` | schema, 字段名数组，数量 | `ZVecErrorCode` | 批量删除字段 |
| `zvec_collection_schema_get_field_count(schema)` | `ZVecCollectionSchema*` | size_t | 获取字段数量 |
| `zvec_collection_schema_get_field(schema, index)` | schema, 索引 | `ZVecFieldSchema*` | 按索引获取字段 |
| `zvec_collection_schema_find_field(schema, name)` | schema, 字段名 | `ZVecFieldSchema*` | 按名查找字段 |
| `zvec_collection_schema_validate(schema, &error)` | schema, 错误输出 | `ZVecErrorCode` | 验证 Schema |
| `zvec_collection_schema_set_max_doc_count_per_segment(schema, count)` | schema, 数量 | `ZVecErrorCode` | 设置段最大文档数 |
| `zvec_collection_schema_get_max_doc_count_per_segment(schema)` | `ZVecCollectionSchema*` | uint64_t | 获取段最大文档数 |

---

## Collection 管理

### Collection 选项

```c
typedef struct {
  bool enable_mmap;                   // 是否启用内存映射
  size_t max_buffer_size;             // 最大缓冲区大小
  bool read_only;                     // 是否只读模式
  uint64_t max_doc_count_per_segment; // 每段最大文档数
} ZVecCollectionOptions;
```

### 创建和打开 Collection

```c
// 初始化默认选项
ZVecCollectionOptions options;
zvec_collection_options_init_default(&options);

// 或使用宏
ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();

// 自定义选项
options.enable_mmap = true;
options.max_buffer_size = 2 * 1024 * 1024;  // 2MB
options.read_only = false;
options.max_doc_count_per_segment = 500000;

// 创建并打开
ZVecCollection *collection;
ZVecErrorCode rc = zvec_collection_create_and_open(
    "/path/to/data", schema, &options, &collection);

// 打开已有集合
rc = zvec_collection_open("/path/to/data", &options, &collection);
```

### Collection 操作

```c
// 关闭集合
rc = zvec_collection_close(collection);

// 销毁集合
rc = zvec_collection_destroy(collection);

// 刷盘数据
rc = zvec_collection_flush(collection);

// 获取 Schema
ZVecCollectionSchema *schema;
rc = zvec_collection_get_schema(collection, &schema);
// 使用后销毁
zvec_collection_schema_destroy(schema);

// 获取选项
ZVecCollectionOptions *options;
rc = zvec_collection_get_options(collection, &options);
// 使用后销毁
free(options);

// 获取统计信息
typedef struct {
  uint64_t doc_count;         // 文档总数
  ZVecString **index_names;   // 索引名数组
  float *index_completeness;  // 索引完成度数组
  size_t index_count;         // 索引数量
} ZVecCollectionStats;

ZVecCollectionStats *stats;
rc = zvec_collection_get_stats(collection, &stats);
printf("Documents: %lu\n", stats->doc_count);
printf("Indexes: %zu\n", stats->index_count);
zvec_collection_stats_destroy(stats);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_collection_options_init_default(&opts)` | `ZVecCollectionOptions*` | void | 初始化默认选项 |
| `zvec_collection_create_and_open(path, schema, opts, &coll)` | 路径，Schema, 选项，输出 | `ZVecErrorCode` | 创建并打开集合 |
| `zvec_collection_open(path, opts, &coll)` | 路径，选项，输出 | `ZVecErrorCode` | 打开已有集合 |
| `zvec_collection_close(coll)` | `ZVecCollection*` | `ZVecErrorCode` | 关闭集合 |
| `zvec_collection_destroy(coll)` | `ZVecCollection*` | `ZVecErrorCode` | 销毁集合 |
| `zvec_collection_flush(coll)` | `ZVecCollection*` | `ZVecErrorCode` | 刷盘数据 |
| `zvec_collection_get_schema(coll, &schema)` | 集合，输出 | `ZVecErrorCode` | 获取 Schema |
| `zvec_collection_get_options(coll, &opts)` | 集合，输出 | `ZVecErrorCode` | 获取选项 |
| `zvec_collection_get_stats(coll, &stats)` | 集合，输出 | `ZVecErrorCode` | 获取统计信息 |
| `zvec_collection_stats_destroy(stats)` | `ZVecCollectionStats*` | void | 销毁统计信息 |

---

## 索引管理

### 索引类型

```c
typedef enum {
  ZVEC_INDEX_TYPE_UNDEFINED = 0,
  ZVEC_INDEX_TYPE_HNSW = 1,    // HNSW 图索引
  ZVEC_INDEX_TYPE_IVF = 3,     // 倒排文件索引
  ZVEC_INDEX_TYPE_FLAT = 4,    // 暴力检索
  ZVEC_INDEX_TYPE_INVERT = 10  // 标量倒排索引
} ZVecIndexType;
```

### 距离度量类型

```c
typedef enum {
  ZVEC_METRIC_TYPE_UNDEFINED = 0,
  ZVEC_METRIC_TYPE_L2 = 1,         // L2 距离
  ZVEC_METRIC_TYPE_IP = 2,         // 内积
  ZVEC_METRIC_TYPE_COSINE = 3,     // 余弦相似度
  ZVEC_METRIC_TYPE_MIPSL2 = 4      // L2 内积
} ZVecMetricType;
```

### 量化类型

```c
typedef enum {
  ZVEC_QUANTIZE_TYPE_UNDEFINED = 0,
  ZVEC_QUANTIZE_TYPE_FP16 = 1,    // FP16 量化
  ZVEC_QUANTIZE_TYPE_INT8 = 2,    // INT8 量化
  ZVEC_QUANTIZE_TYPE_INT4 = 3     // INT4 量化
} ZVecQuantizeType;
```

### HNSW 索引参数

```c
typedef struct {
  ZVecVectorIndexParams base;  // 基类参数
  int m;                        // 图连接度参数
  int ef_construction;          // 构建时探索因子
  int ef_search;                // 搜索时探索因子
} ZVecHnswIndexParams;

// 创建 HNSW 参数
ZVecHnswIndexParams *params = zvec_index_params_hnsw_create(
    ZVEC_METRIC_TYPE_COSINE,   // 距离类型
    ZVEC_QUANTIZE_TYPE_UNDEFINED,  // 量化类型
    16,    // m: 图连接度
    200,   // ef_construction: 构建探索因子
    50     // ef_search: 搜索探索因子
);

// 或使用初始化函数
ZVecHnswIndexParams params;
zvec_index_params_hnsw_init(&params,
    ZVEC_METRIC_TYPE_COSINE, 16, 200, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);

// 或使用宏
ZVecHnswIndexParams params = ZVEC_HNSW_PARAMS(
    ZVEC_METRIC_TYPE_COSINE, 16, 200, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);

zvec_index_params_hnsw_destroy(params);
```

### IVF 索引参数

```c
typedef struct {
  ZVecVectorIndexParams base;  // 基类参数
  int n_list;                   // 聚类中心数量
  int n_iters;                  // 迭代次数
  bool use_soar;                // 是否使用 SOAR 算法
  int n_probe;                  // 搜索时探测的聚类数
} ZVecIVFIndexParams;

// 创建 IVF 参数
ZVecIVFIndexParams *params = zvec_index_params_ivf_create(
    ZVEC_METRIC_TYPE_L2,     // 距离类型
    ZVEC_QUANTIZE_TYPE_INT8, // 量化类型
    1024,   // n_list: 聚类中心数
    25,     // n_iters: 迭代次数
    true,   // use_soar: 使用 SOAR
    20      // n_probe: 探测聚类数
);

// 或使用宏
ZVecIVFIndexParams params = ZVEC_IVF_PARAMS(
    ZVEC_METRIC_TYPE_L2, 1024, 25, true, 20, ZVEC_QUANTIZE_TYPE_INT8);

zvec_index_params_ivf_destroy(params);
```

### Flat 索引参数

```c
typedef struct {
  ZVecVectorIndexParams base;  // 基类参数
} ZVecFlatIndexParams;

// 创建 Flat 参数
ZVecFlatIndexParams *params = zvec_index_params_flat_create(
    ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED);

// 或使用宏
ZVecFlatIndexParams params = ZVEC_FLAT_PARAMS(
    ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED);

zvec_index_params_flat_destroy(params);
```

### 标量倒排索引参数

```c
typedef struct {
  ZVecBaseIndexParams base;         // 基类参数
  bool enable_range_optimization;   // 是否启用范围优化
  bool enable_extended_wildcard;    // 是否启用通配符
} ZVecInvertIndexParams;

// 创建倒排索引参数
ZVecInvertIndexParams *params = zvec_index_params_invert_create(
    true,   // enable_range_optimization
    false   // enable_extended_wildcard
);

// 或使用宏
ZVecInvertIndexParams params = ZVEC_INVERT_PARAMS(true, false);

// 或使用初始化函数
ZVecInvertIndexParams params;
zvec_index_params_invert_init(&params, true, false);

zvec_index_params_invert_destroy(params);
```

### 创建索引

```c
// 通用创建索引函数
zvec_collection_create_index(collection, "embedding", index_params);

// 类型安全的创建索引函数
zvec_collection_create_hnsw_index(collection, "embedding", hnsw_params);
zvec_collection_create_ivf_index(collection, "embedding", ivf_params);
zvec_collection_create_flat_index(collection, "embedding", flat_params);
zvec_collection_create_invert_index(collection, "title", invert_params);

// 删除索引
zvec_collection_drop_index(collection, "embedding");

// 优化集合（重建索引、合并段）
zvec_collection_optimize(collection);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_index_params_base_init(params, type)` | 参数，类型 | void | 初始化基础参数 |
| `zvec_index_params_invert_init(params, range_opt, wildcard)` | 参数，范围优化，通配符 | void | 初始化倒排参数 |
| `zvec_index_params_vector_init(params, idx, metric, quant)` | 参数，索引类型，度量，量化 | void | 初始化向量索引参数 |
| `zvec_index_params_hnsw_init(params, metric, m, ef_c, ef_s, quant)` | 参数，度量，m, ef_construction, ef_search, 量化 | void | 初始化 HNSW 参数 |
| `zvec_index_params_ivf_init(params, metric, nlist, niters, soar, nprobe, quant)` | 参数，度量，nlist, niters, soar, nprobe, 量化 | void | 初始化 IVF 参数 |
| `zvec_index_params_flat_init(params, metric, quant)` | 参数，度量，量化 | void | 初始化 Flat 参数 |
| `zvec_index_params_invert_create(range_opt, wildcard)` | 范围优化，通配符 | `ZVecInvertIndexParams*` | 创建倒排参数 |
| `zvec_index_params_vector_create(type, metric, quant)` | 类型，度量，量化 | `ZVecVectorIndexParams*` | 创建向量索引参数 |
| `zvec_index_params_hnsw_create(metric, quant, m, ef_c, ef_s)` | 度量，量化，m, ef_construction, ef_search | `ZVecHnswIndexParams*` | 创建 HNSW 参数 |
| `zvec_index_params_ivf_create(metric, quant, nlist, niters, soar, nprobe)` | 度量，量化，nlist, niters, soar, nprobe | `ZVecIVFIndexParams*` | 创建 IVF 参数 |
| `zvec_index_params_flat_create(metric, quant)` | 度量，量化 | `ZVecFlatIndexParams*` | 创建 Flat 参数 |
| `zvec_index_params_invert_destroy(params)` | 参数 | void | 销毁倒排参数 |
| `zvec_index_params_vector_destroy(params)` | 参数 | void | 销毁向量索引参数 |
| `zvec_index_params_hnsw_destroy(params)` | 参数 | void | 销毁 HNSW 参数 |
| `zvec_index_params_ivf_destroy(params)` | 参数 | void | 销毁 IVF 参数 |
| `zvec_index_params_flat_destroy(params)` | 参数 | void | 销毁 Flat 参数 |
| `zvec_collection_create_index(coll, field, params)` | 集合，字段，参数 | `ZVecErrorCode` | 创建索引 |
| `zvec_collection_create_hnsw_index(...)` | 集合，字段，HNSW 参数 | `ZVecErrorCode` | 创建 HNSW 索引 |
| `zvec_collection_create_ivf_index(...)` | 集合，字段，IVF 参数 | `ZVecErrorCode` | 创建 IVF 索引 |
| `zvec_collection_create_flat_index(...)` | 集合，字段，Flat 参数 | `ZVecErrorCode` | 创建 Flat 索引 |
| `zvec_collection_create_invert_index(...)` | 集合，字段，倒排参数 | `ZVecErrorCode` | 创建倒排索引 |
| `zvec_collection_drop_index(coll, field)` | 集合，字段名 | `ZVecErrorCode` | 删除索引 |
| `zvec_collection_optimize(coll)` | 集合 | `ZVecErrorCode` | 优化集合 |

---

## 文档操作

### 文档结构

```c
typedef struct ZVecDoc ZVecDoc;  // 不透明指针

// 字段值联合
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
  ZVecByteArray binary_value;
} ZVecFieldValue;

// 文档字段
typedef struct {
  ZVecString name;
  ZVecDataType data_type;
  ZVecFieldValue value;
} ZVecDocField;
```

### 创建和销毁文档

```c
// 创建文档
ZVecDoc *doc = zvec_doc_create();

// 清空文档
zvec_doc_clear(doc);

// 销毁文档
zvec_doc_destroy(doc);
```

### 设置文档属性

```c
// 设置主键
zvec_doc_set_pk(doc, "doc_001");

// 设置文档 ID
zvec_doc_set_doc_id(doc, 12345);

// 设置分数
zvec_doc_set_score(doc, 0.95f);

// 设置操作类型
typedef enum {
  ZVEC_DOC_OP_INSERT = 0,  // 插入
  ZVEC_DOC_OP_UPDATE = 1,  // 更新
  ZVEC_DOC_OP_UPSERT = 2,  // 插入或更新
  ZVEC_DOC_OP_DELETE = 3   // 删除
} ZVecDocOperator;

zvec_doc_set_operator(doc, ZVEC_DOC_OP_INSERT);
```

### 获取文档属性

```c
// 获取文档 ID
uint64_t id = zvec_doc_get_doc_id(doc);

// 获取分数
float score = zvec_doc_get_score(doc);

// 获取操作类型
ZVecDocOperator op = zvec_doc_get_operator(doc);

// 获取主键指针（不复制）
const char *pk = zvec_doc_get_pk_pointer(doc);

// 获取主键副本（需手动释放）
const char *pk = zvec_doc_get_pk_copy(doc);
free((void*)pk);

// 获取字段数量
size_t count = zvec_doc_get_field_count(doc);

// 检查文档是否为空
bool empty = zvec_doc_is_empty(doc);

// 检查是否包含字段
bool has = zvec_doc_has_field(doc, "embedding");

// 检查字段是否有值
bool has_value = zvec_doc_has_field_value(doc, "embedding");

// 检查字段是否为 null
bool is_null = zvec_doc_is_field_null(doc, "optional_field");
```

### 添加字段

```c
// 按值添加字段
float embedding[768] = {0.1f, 0.2f, ...};
zvec_doc_add_field_by_value(doc, "embedding",
    ZVEC_DATA_TYPE_VECTOR_FP32, embedding, sizeof(embedding));

// 添加字符串字段
const char *title = "Hello World";
zvec_doc_add_field_by_value(doc, "title",
    ZVEC_DATA_TYPE_STRING, title, strlen(title) + 1);

// 添加整数字段
int64_t timestamp = 1234567890;
zvec_doc_add_field_by_value(doc, "timestamp",
    ZVEC_DATA_TYPE_INT64, &timestamp, sizeof(timestamp));

// 按结构添加字段
ZVecDocField field;
field.name = ZVEC_STRING("score");
field.data_type = ZVEC_DATA_TYPE_FLOAT;
field.value.float_value = 0.95f;
zvec_doc_add_field_by_struct(doc, &field);

// 删除字段
zvec_doc_remove_field(doc, "title");
```

### 获取字段值

```c
// 获取基本类型值
float float_val;
zvec_doc_get_field_value_basic(doc, "score",
    ZVEC_DATA_TYPE_FLOAT, &float_val, sizeof(float_val));

int64_t int_val;
zvec_doc_get_field_value_basic(doc, "timestamp",
    ZVEC_DATA_TYPE_INT64, &int_val, sizeof(int_val));

// 获取字段值副本（需手动释放）
void *value;
size_t value_size;

// 获取字符串
zvec_doc_get_field_value_copy(doc, "title", ZVEC_DATA_TYPE_STRING, &value, &value_size);
printf("Title: %s\n", (char*)value);
free(value);

// 获取向量
zvec_doc_get_field_value_copy(doc, "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, &value, &value_size);
float *vec = (float*)value;
// 使用...
free(value);

// 获取二进制数据
zvec_doc_get_field_value_copy(doc, "data", ZVEC_DATA_TYPE_BINARY, &value, &value_size);
zvec_free_uint8_array((uint8_t*)value);

// 获取字段值指针（无需释放，数据在文档内）
const void *value;
size_t value_size;
zvec_doc_get_field_value_pointer(doc, "score", ZVEC_DATA_TYPE_FLOAT, &value, &value_size);
float score = *(float*)value;
```

### 获取所有字段名

```c
char **field_names;
size_t count;
zvec_doc_get_field_names(doc, &field_names, &count);

for (size_t i = 0; i < count; i++) {
    printf("Field %zu: %s\n", i, field_names[i]);
}

// 释放
zvec_free_str_array(field_names, count);
```

### 序列化/反序列化

```c
// 序列化
uint8_t *data;
size_t size;
ZVecErrorCode rc = zvec_doc_serialize(doc, &data, &size);

// 保存到文件
FILE *f = fopen("doc.bin", "wb");
fwrite(data, 1, size, f);
fclose(f);
zvec_free_uint8_array(data);

// 从文件读取
FILE *f = fopen("doc.bin", "rb");
fseek(f, 0, SEEK_END);
size_t file_size = ftell(f);
fseek(f, 0, SEEK_SET);
uint8_t *buffer = malloc(file_size);
fread(buffer, 1, file_size, f);
fclose(f);

// 反序列化
ZVecDoc *new_doc;
rc = zvec_doc_deserialize(buffer, file_size, &new_doc);
free(buffer);

// 使用...
zvec_doc_destroy(new_doc);
```

### 文档合并

```c
// 合并两个文档
ZVecDoc *doc1 = zvec_doc_create();
ZVecDoc *doc2 = zvec_doc_create();

// 设置字段...
zvec_doc_merge(doc1, doc2);  // 将 doc2 的字段合并到 doc1
```

### 内存使用

```c
size_t bytes = zvec_doc_memory_usage(doc);
printf("Document uses %zu bytes\n", bytes);
```

### 验证文档

```c
char *error_msg;
ZVecErrorCode rc = zvec_doc_validate(doc, schema, false, &error_msg);
if (rc != ZVEC_OK) {
    printf("Invalid document: %s\n", error_msg);
    free(error_msg);
}
```

### 文档详细信息

```c
char *detail_str;
zvec_doc_to_detail_string(doc, &detail_str);
printf("Document: %s\n", detail_str);
free(detail_str);
```

### 批量释放文档

```c
ZVecDoc **docs = malloc(count * sizeof(ZVecDoc*));
// 填充 docs...

// 批量释放
zvec_docs_free(docs, count);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_doc_create()` | 无 | `ZVecDoc*` | 创建文档 |
| `zvec_doc_destroy(doc)` | `ZVecDoc*` | void | 销毁文档 |
| `zvec_doc_clear(doc)` | `ZVecDoc*` | void | 清空文档 |
| `zvec_doc_set_pk(doc, pk)` | doc, 主键 | void | 设置主键 |
| `zvec_doc_set_doc_id(doc, id)` | doc, ID | void | 设置文档 ID |
| `zvec_doc_set_score(doc, score)` | doc, 分数 | void | 设置分数 |
| `zvec_doc_set_operator(doc, op)` | doc, 操作类型 | void | 设置操作类型 |
| `zvec_doc_get_doc_id(doc)` | `ZVecDoc*` | uint64_t | 获取文档 ID |
| `zvec_doc_get_score(doc)` | `ZVecDoc*` | float | 获取分数 |
| `zvec_doc_get_operator(doc)` | `ZVecDoc*` | `ZVecDocOperator` | 获取操作类型 |
| `zvec_doc_get_pk_pointer(doc)` | `ZVecDoc*` | `const char*` | 获取主键指针 |
| `zvec_doc_get_pk_copy(doc)` | `ZVecDoc*` | `const char*` | 获取主键副本 |
| `zvec_doc_get_field_count(doc)` | `ZVecDoc*` | size_t | 获取字段数量 |
| `zvec_doc_is_empty(doc)` | `ZVecDoc*` | bool | 检查是否为空 |
| `zvec_doc_has_field(doc, name)` | doc, 字段名 | bool | 检查是否包含字段 |
| `zvec_doc_has_field_value(doc, name)` | doc, 字段名 | bool | 检查字段是否有值 |
| `zvec_doc_is_field_null(doc, name)` | doc, 字段名 | bool | 检查字段是否为 null |
| `zvec_doc_add_field_by_value(doc, name, type, value, size)` | doc, 名，类型，值，大小 | `ZVecErrorCode` | 添加字段 |
| `zvec_doc_add_field_by_struct(doc, field)` | doc, 字段结构 | `ZVecErrorCode` | 按结构添加字段 |
| `zvec_doc_remove_field(doc, name)` | doc, 字段名 | `ZVecErrorCode` | 删除字段 |
| `zvec_doc_get_field_value_basic(doc, name, type, buf, size)` | doc, 名，类型，缓冲区，大小 | `ZVecErrorCode` | 获取基本类型值 |
| `zvec_doc_get_field_value_copy(doc, name, type, &val, &size)` | doc, 名，类型，值输出，大小输出 | `ZVecErrorCode` | 获取字段值副本 |
| `zvec_doc_get_field_value_pointer(doc, name, type, &val, &size)` | doc, 名，类型，值输出，大小输出 | `ZVecErrorCode` | 获取字段值指针 |
| `zvec_doc_get_field_names(doc, &names, &count)` | doc, 名称输出，数量输出 | `ZVecErrorCode` | 获取所有字段名 |
| `zvec_doc_serialize(doc, &data, &size)` | doc, 数据输出，大小输出 | `ZVecErrorCode` | 序列化 |
| `zvec_doc_deserialize(data, size, &doc)` | 数据，大小，文档输出 | `ZVecErrorCode` | 反序列化 |
| `zvec_doc_merge(doc, other)` | doc, 源文档 | void | 合并文档 |
| `zvec_doc_memory_usage(doc)` | `ZVecDoc*` | size_t | 获取内存使用 |
| `zvec_doc_validate(doc, schema, is_update, &err)` | doc, schema, 是否更新，错误输出 | `ZVecErrorCode` | 验证文档 |
| `zvec_doc_to_detail_string(doc, &str)` | doc, 字符串输出 | `ZVecErrorCode` | 获取详细信息字符串 |
| `zvec_docs_free(docs, count)` | 文档数组，数量 | void | 批量释放文档 |
| `zvec_free_str_array(arr, count)` | 字符串数组，数量 | void | 释放字符串数组 |

---

## 数据增删改

### 插入文档

```c
ZVecDoc *docs[3];
docs[0] = zvec_doc_create();
docs[1] = zvec_doc_create();
docs[2] = zvec_doc_create();

zvec_doc_set_pk(docs[0], "doc_001");
zvec_doc_set_pk(docs[1], "doc_002");
zvec_doc_set_pk(docs[2], "doc_003");

// 添加字段...

size_t success_count, error_count;
ZVecErrorCode rc = zvec_collection_insert(collection,
    (const ZVecDoc**)docs, 3, &success_count, &error_count);

printf("Inserted: %zu, Failed: %zu\n", success_count, error_count);

// 清理
zvec_docs_free(docs, 3);
```

### 更新文档

```c
ZVecDoc *doc = zvec_doc_create();
zvec_doc_set_pk(doc, "doc_001");

// 设置要更新的字段
float new_embedding[768] = {0.2f, 0.3f, ...};
zvec_doc_add_field_by_value(doc, "embedding",
    ZVEC_DATA_TYPE_VECTOR_FP32, new_embedding, sizeof(new_embedding));

size_t success_count, error_count;
ZVecErrorCode rc = zvec_collection_update(collection,
    (const ZVecDoc**)&doc, 1, &success_count, &error_count);

zvec_doc_destroy(doc);
```

### 插入或更新（Upsert）

```c
ZVecDoc *doc = zvec_doc_create();
zvec_doc_set_pk(doc, "doc_001");
// 设置字段...

size_t success_count, error_count;
ZVecErrorCode rc = zvec_collection_upsert(collection,
    (const ZVecDoc**)&doc, 1, &success_count, &error_count);

zvec_doc_destroy(doc);
```

### 删除文档

```c
// 按主键删除
const char *pks[] = {"doc_001", "doc_002", "doc_003"};
size_t success_count, error_count;
ZVecErrorCode rc = zvec_collection_delete(collection,
    pks, 3, &success_count, &error_count);

// 按过滤条件删除
rc = zvec_collection_delete_by_filter(collection, "category='spam'");
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_collection_insert(coll, docs, count, &success, &error)` | 集合，文档数组，数量，成功数输出，错误数输出 | `ZVecErrorCode` | 插入文档 |
| `zvec_collection_update(coll, docs, count, &success, &error)` | 集合，文档数组，数量，成功数输出，错误数输出 | `ZVecErrorCode` | 更新文档 |
| `zvec_collection_upsert(coll, docs, count, &success, &error)` | 集合，文档数组，数量，成功数输出，错误数输出 | `ZVecErrorCode` | 插入或更新 |
| `zvec_collection_delete(coll, pks, count, &success, &error)` | 集合，主键数组，数量，成功数输出，错误数输出 | `ZVecErrorCode` | 按主键删除 |
| `zvec_collection_delete_by_filter(coll, filter)` | 集合，过滤表达式 | `ZVecErrorCode` | 按条件删除 |

---

## 数据查询

### 向量查询参数

```c
typedef struct {
  ZVecIndexType index_type;  // 索引类型
  float radius;               // 搜索半径
  bool is_linear;             // 是否线性搜索
  bool is_using_refiner;      // 是否使用优化器
} ZVecQueryParams;
```

### HNSW 查询参数

```c
typedef struct {
  ZVecQueryParams base;
  int ef;  // 搜索时探索因子
} ZVecHnswQueryParams;

// 创建
ZVecHnswQueryParams *params = zvec_query_params_hnsw_create(
    ZVEC_INDEX_TYPE_HNSW,
    100,    // ef
    0.0f,   // radius
    false,  // is_linear
    true    // is_using_refiner
);

zvec_query_params_hnsw_set_ef(params, 200);
zvec_query_params_hnsw_destroy(params);
```

### IVF 查询参数

```c
typedef struct {
  ZVecQueryParams base;
  int nprobe;         // 探测聚类数
  float scale_factor; // 缩放因子
} ZVecIVFQueryParams;

// 创建
ZVecIVFQueryParams *params = zvec_query_params_ivf_create(
    ZVEC_INDEX_TYPE_IVF,
    20,     // nprobe
    true,   // is_using_refiner
    1.0f    // scale_factor
);

zvec_query_params_ivf_set_nprobe(params, 50);
zvec_query_params_ivf_set_scale_factor(params, 1.5f);
zvec_query_params_ivf_destroy(params);
```

### Flat 查询参数

```c
typedef struct {
  ZVecQueryParams base;
  float scale_factor;  // 缩放因子
} ZVecFlatQueryParams;

ZVecFlatQueryParams *params = zvec_query_params_flat_create(
    ZVEC_INDEX_TYPE_FLAT,
    false,  // is_using_refiner
    1.0f    // scale_factor
);

zvec_query_params_flat_destroy(params);
```

### 基础查询参数

```c
// 创建基础参数
ZVecQueryParams *params = zvec_query_params_create(ZVEC_INDEX_TYPE_HNSW);

// 设置属性
zvec_query_params_set_index_type(params, ZVEC_INDEX_TYPE_HNSW);
zvec_query_params_set_radius(params, 0.5f);
zvec_query_params_set_is_linear(params, true);
zvec_query_params_set_is_using_refiner(params, true);

zvec_query_params_destroy(params);
```

### 向量查询

```c
typedef struct {
  int topk;                        // 返回结果数
  ZVecString field_name;           // 查询字段名
  ZVecByteArray query_vector;      // 查询向量
  ZVecByteArray query_sparse_indices;  // 稀疏向量索引
  ZVecByteArray query_sparse_values;   // 稀疏向量值
  ZVecString filter;               // 过滤表达式
  bool include_vector;             // 是否返回向量
  bool include_doc_id;             // 是否返回文档 ID
  ZVecStringArray output_fields;   // 输出字段列表
  ZVecQueryParamsUnion *query_params;  // 查询参数
} ZVecVectorQuery;

// 使用宏快速创建
float query_vec[768] = {0.1f, 0.2f, ...};
ZVecVectorQuery query = ZVEC_VECTOR_QUERY(
    "embedding",              // 字段名
    ZVEC_FLOAT_ARRAY(query_vec, 768),
    10,                       // topK
    "category='news'"         // 过滤条件
);

// 手动创建
ZVecVectorQuery query = {
    .topk = 10,
    .field_name = ZVEC_STRING("embedding"),
    .query_vector = ZVEC_FLOAT_ARRAY(query_vec, 768),
    .filter = ZVEC_STRING(""),
    .include_vector = true,
    .include_doc_id = true,
    .output_fields.strings = NULL,
    .output_fields.count = 0,
    .query_params = NULL
};

// 执行查询
ZVecDoc **results;
size_t result_count;
ZVecErrorCode rc = zvec_collection_query(collection, &query, &results, &result_count);

if (rc == ZVEC_OK) {
    for (size_t i = 0; i < result_count; i++) {
        const char *pk = zvec_doc_get_pk_pointer(results[i]);
        float score = zvec_doc_get_score(results[i]);
        printf("Result %zu: pk=%s, score=%f\n", i, pk, score);
    }
}

// 释放结果
zvec_docs_free(results, result_count);
```

### 分组向量查询

```c
typedef struct {
  ZVecString field_name;           // 查询字段名
  ZVecByteArray query_vector;      // 查询向量
  ZVecByteArray query_sparse_indices;  // 稀疏向量索引
  ZVecByteArray query_sparse_values;   // 稀疏向量值
  ZVecString filter;               // 过滤表达式
  bool include_vector;             // 是否返回向量
  ZVecStringArray output_fields;   // 输出字段列表
  ZVecString group_by_field_name;  // 分组字段名
  uint32_t group_count;            // 分组数量
  uint32_t group_topk;             // 每组返回结果数
  ZVecQueryParamsUnion *query_params;  // 查询参数
} ZVecGroupByVectorQuery;

// 创建分组查询
ZVecGroupByVectorQuery query = {
    .field_name = ZVEC_STRING("embedding"),
    .query_vector = ZVEC_FLOAT_ARRAY(query_vec, 768),
    .filter = ZVEC_STRING(""),
    .include_vector = false,
    .group_by_field_name = ZVEC_STRING("category"),
    .group_count = 5,
    .group_topk = 3,
    .query_params = NULL
};

// 执行查询
ZVecDoc **results;
ZVecString **group_values;
size_t result_count;

ZVecErrorCode rc = zvec_collection_query_by_group(
    collection, &query, &results, &group_values, &result_count);

if (rc == ZVEC_OK) {
    for (size_t i = 0; i < result_count; i++) {
        printf("Group: %s\n", group_values[i]->data);
        // 处理结果...
    }
}

// 释放结果
zvec_docs_free(results, result_count);
zvec_string_array_destroy((ZVecStringArray*)group_values);
```

### 按主键获取

```c
const char *pks[] = {"doc_001", "doc_002", "doc_003"};
ZVecDoc **documents;
size_t found_count;

ZVecErrorCode rc = zvec_collection_fetch(collection,
    pks, 3, &documents, &found_count);

printf("Found %zu documents\n", found_count);

// 使用...
zvec_docs_free(documents, found_count);
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_query_params_create(type)` | 索引类型 | `ZVecQueryParams*` | 创建查询参数 |
| `zvec_query_params_hnsw_create(type, ef, radius, linear, refiner)` | 类型，ef, 半径，线性，优化器 | `ZVecHnswQueryParams*` | 创建 HNSW 查询参数 |
| `zvec_query_params_ivf_create(type, nprobe, refiner, scale)` | 类型，nprobe, 优化器，缩放因子 | `ZVecIVFQueryParams*` | 创建 IVF 查询参数 |
| `zvec_query_params_flat_create(type, refiner, scale)` | 类型，优化器，缩放因子 | `ZVecFlatQueryParams*` | 创建 Flat 查询参数 |
| `zvec_query_params_union_create(type)` | 索引类型 | `ZVecQueryParamsUnion*` | 创建查询参数联合 |
| `zvec_query_params_destroy(params)` | 参数 | void | 销毁查询参数 |
| `zvec_query_params_hnsw_destroy(params)` | 参数 | void | 销毁 HNSW 查询参数 |
| `zvec_query_params_ivf_destroy(params)` | 参数 | void | 销毁 IVF 查询参数 |
| `zvec_query_params_flat_destroy(params)` | 参数 | void | 销毁 Flat 查询参数 |
| `zvec_query_params_union_destroy(params)` | 参数 | void | 销毁查询参数联合 |
| `zvec_query_params_set_index_type(params, type)` | 参数，类型 | `ZVecErrorCode` | 设置索引类型 |
| `zvec_query_params_set_radius(params, radius)` | 参数，半径 | `ZVecErrorCode` | 设置搜索半径 |
| `zvec_query_params_set_is_linear(params, linear)` | 参数，是否线性 | `ZVecErrorCode` | 设置线性搜索 |
| `zvec_query_params_set_is_using_refiner(params, refiner)` | 参数，是否优化器 | `ZVecErrorCode` | 设置优化器 |
| `zvec_query_params_hnsw_set_ef(params, ef)` | 参数，ef | `ZVecErrorCode` | 设置 ef |
| `zvec_query_params_ivf_set_nprobe(params, nprobe)` | 参数，nprobe | `ZVecErrorCode` | 设置 nprobe |
| `zvec_query_params_ivf_set_scale_factor(params, scale)` | 参数，缩放因子 | `ZVecErrorCode` | 设置缩放因子 |
| `zvec_collection_query(coll, query, &results, &count)` | 集合，查询，结果输出，数量输出 | `ZVecErrorCode` | 向量查询 |
| `zvec_collection_query_by_group(coll, query, &results, &groups, &count)` | 集合，分组查询，结果输出，分组值输出，数量输出 | `ZVecErrorCode` | 分组向量查询 |
| `zvec_collection_fetch(coll, pks, count, &docs, &found)` | 集合，主键数组，数量，文档输出，找到数量 | `ZVecErrorCode` | 按主键获取 |

---

## 工具函数

### 类型转字符串

```c
// 数据类型转字符串
const char *type_str = zvec_data_type_to_string(ZVEC_DATA_TYPE_VECTOR_FP32);
// 返回："VECTOR_FP32"

// 索引类型转字符串
const char *idx_str = zvec_index_type_to_string(ZVEC_INDEX_TYPE_HNSW);
// 返回："HNSW"

// 距离类型转字符串
const char *metric_str = zvec_metric_type_to_string(ZVEC_METRIC_TYPE_COSINE);
// 返回："COSINE"

// 错误码转字符串
const char *err_str = zvec_error_code_to_string(ZVEC_ERROR_INVALID_ARGUMENT);
// 返回："Invalid argument"
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `zvec_data_type_to_string(type)` | `ZVecDataType` | `const char*` | 数据类型转字符串 |
| `zvec_index_type_to_string(type)` | `ZVecIndexType` | `const char*` | 索引类型转字符串 |
| `zvec_metric_type_to_string(type)` | `ZVecMetricType` | `const char*` | 距离类型转字符串 |
| `zvec_error_code_to_string(code)` | `ZVecErrorCode` | `const char*` | 错误码转字符串 |

---

## 完整示例

### 构建可搜索的向量数据库

```c
#include "zvec/c_api.h"
#include <stdio.h>
#include <stdlib.h>

#define DIM 768
#define DOC_COUNT 1000

// 生成随机向量
void generate_vector(float *vec, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    ZVecErrorCode rc;

    // ========== 1. 初始化 ==========
    printf("Initializing ZVec...\n");
    rc = zvec_initialize(NULL);
    if (rc != ZVEC_OK) {
        fprintf(stderr, "Failed to initialize: %s\n",
                zvec_error_code_to_string(rc));
        return 1;
    }
    printf("Version: %s\n", zvec_get_version());

    // ========== 2. 创建 Schema ==========
    printf("Creating schema...\n");
    ZVecCollectionSchema *schema = zvec_collection_schema_create("documents");

    // ID 字段
    ZVecFieldSchema *id_field = zvec_field_schema_create(
        "id", ZVEC_DATA_TYPE_STRING, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    // 向量字段
    ZVecFieldSchema *embedding_field = zvec_field_schema_create(
        "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, DIM);
    zvec_collection_schema_add_field(schema, embedding_field);

    // 标题字段
    ZVecFieldSchema *title_field = zvec_field_schema_create(
        "title", ZVEC_DATA_TYPE_STRING, true, 0);
    ZVecInvertIndexParams *invert_params = zvec_index_params_invert_create(
        true, true);  // 启用范围优化和通配符
    zvec_field_schema_set_invert_index(title_field, invert_params);
    zvec_collection_schema_add_field(schema, title_field);

    // 时间戳字段
    ZVecFieldSchema *ts_field = zvec_field_schema_create(
        "timestamp", ZVEC_DATA_TYPE_INT64, true, 0);
    zvec_collection_schema_add_field(schema, ts_field);

    // 验证 Schema
    ZVecString *error_msg;
    rc = zvec_collection_schema_validate(schema, &error_msg);
    if (rc != ZVEC_OK) {
        fprintf(stderr, "Invalid schema: %s\n", error_msg->data);
        zvec_free_string(error_msg);
        return 1;
    }

    // ========== 3. 创建 Collection ==========
    printf("Creating collection...\n");
    ZVecCollection *collection;
    ZVecCollectionOptions options = ZVEC_DEFAULT_OPTIONS();

    rc = zvec_collection_create_and_open(
        "./my_vector_db", schema, &options, &collection);
    if (rc != ZVEC_OK) {
        fprintf(stderr, "Failed to create collection: %s\n",
                zvec_error_code_to_string(rc));
        return 1;
    }

    // ========== 4. 创建索引 ==========
    printf("Creating HNSW index...\n");
    ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
        ZVEC_METRIC_TYPE_COSINE,
        ZVEC_QUANTIZE_TYPE_UNDEFINED,
        16,     // m
        200,    // ef_construction
        50      // ef_search
    );
    rc = zvec_collection_create_hnsw_index(collection, "embedding", hnsw_params);
    zvec_index_params_hnsw_destroy(hnsw_params);

    // ========== 5. 批量插入数据 ==========
    printf("Inserting %d documents...\n", DOC_COUNT);

    ZVecDoc **docs = malloc(DOC_COUNT * sizeof(ZVecDoc*));
    float vectors[DOC_COUNT][DIM];

    for (int i = 0; i < DOC_COUNT; i++) {
        docs[i] = zvec_doc_create();

        // 设置主键
        char pk[32];
        snprintf(pk, sizeof(pk), "doc_%06d", i);
        zvec_doc_set_pk(docs[i], pk);

        // 生成随机向量
        generate_vector(vectors[i], DIM);
        zvec_doc_add_field_by_value(docs[i], "embedding",
            ZVEC_DATA_TYPE_VECTOR_FP32, vectors[i], sizeof(float) * DIM);

        // 添加标题
        char title[64];
        snprintf(title, sizeof(title), "Document Title %d", i);
        zvec_doc_add_field_by_value(docs[i], "title",
            ZVEC_DATA_TYPE_STRING, title, strlen(title) + 1);

        // 添加时间戳
        int64_t ts = 1700000000 + i * 1000;
        zvec_doc_add_field_by_value(docs[i], "timestamp",
            ZVEC_DATA_TYPE_INT64, &ts, sizeof(ts));
    }

    size_t success_count, error_count;
    rc = zvec_collection_insert(collection,
        (const ZVecDoc**)docs, DOC_COUNT, &success_count, &error_count);
    printf("Inserted: %zu, Failed: %zu\n", success_count, error_count);

    // 清理文档
    zvec_docs_free(docs, DOC_COUNT);
    free(docs);

    // 刷盘
    zvec_collection_flush(collection);

    // ========== 6. 查询 ==========
    printf("\nPerforming vector search...\n");

    // 生成查询向量
    float query_vec[DIM];
    generate_vector(query_vec, DIM);

    // 创建查询
    ZVecVectorQuery query = ZVEC_VECTOR_QUERY(
        "embedding",
        ZVEC_FLOAT_ARRAY(query_vec, DIM),
        10,     // topK
        "timestamp > 1700500000"  // 过滤条件
    );

    // 执行查询
    ZVecDoc **results;
    size_t result_count;
    rc = zvec_collection_query(collection, &query, &results, &result_count);

    if (rc == ZVEC_OK) {
        printf("Found %zu results:\n", result_count);
        for (size_t i = 0; i < result_count; i++) {
            const char *pk = zvec_doc_get_pk_pointer(results[i]);
            float score = zvec_doc_get_score(results[i]);

            // 获取标题
            const char *title;
            size_t title_size;
            zvec_doc_get_field_value_copy(results[i], "title",
                ZVEC_DATA_TYPE_STRING, (void**)&title, &title_size);

            printf("  [%zu] %s - score: %.4f - title: %s\n",
                   i, pk, score, title);
            free((void*)title);
        }
    }

    // 释放结果
    zvec_docs_free(results, result_count);

    // ========== 7. 获取统计信息 ==========
    printf("\nCollection statistics:\n");
    ZVecCollectionStats *stats;
    rc = zvec_collection_get_stats(collection, &stats);
    if (rc == ZVEC_OK) {
        printf("  Total documents: %lu\n", stats->doc_count);
        printf("  Index count: %zu\n", stats->index_count);
        for (size_t i = 0; i < stats->index_count; i++) {
            printf("    Index %zu: %s (%.1f%% complete)\n",
                   i, stats->index_names[i]->data,
                   stats->index_completeness[i] * 100);
        }
        zvec_collection_stats_destroy(stats);
    }

    // ========== 8. 清理 ==========
    printf("\nCleaning up...\n");
    zvec_collection_close(collection);
    zvec_collection_destroy(collection);
    zvec_collection_schema_destroy(schema);
    zvec_shutdown();

    printf("Done!\n");
    return 0;
}
```

### 编译示例

```bash
gcc -o example example.c -lzvec -I./include -L./lib
./example
```

---

## 附录

### 内存管理约定

| 创建函数 | 释放函数 | 说明 |
|----------|----------|------|
| `zvec_*_create()` | `zvec_*_destroy()` | 需要成对调用 |
| `zvec_collection_create_and_open()` | `zvec_collection_close()` + `zvec_collection_destroy()` | Collection 生命周期 |
| `zvec_doc_create()` | `zvec_doc_destroy()` | 文档生命周期 |
| `zvec_get_last_error(&msg)` | `free(msg)` | 错误消息需手动释放 |
| `zvec_doc_get_field_value_copy()` | `free()` 或 `zvec_free_uint8_array()` | 字段值副本需释放 |
| 查询返回的 `results` | `zvec_docs_free()` | 查询结果批量释放 |

### 宏定义速查

```c
// 索引参数宏
ZVEC_HNSW_PARAMS(metric, m, ef_construction, ef_search, quant)
ZVEC_IVF_PARAMS(metric, nlist, niters, soar, nprobe, quant)
ZVEC_FLAT_PARAMS(metric, quant)
ZVEC_INVERT_PARAMS(range_opt, wildcard)

// 数据结构宏
ZVEC_STRING(str)
ZVEC_STRING_VIEW(str)
ZVEC_FLOAT_ARRAY(data_ptr, len)
ZVEC_INT64_ARRAY(data_ptr, len)

// 选项宏
ZVEC_DEFAULT_OPTIONS()

// 查询宏
ZVEC_VECTOR_QUERY(field_name, query_vec, top_k, filter)

// 文档字段宏
ZVEC_DOC_FIELD(name, type, value_union)
```

### 最佳实践

1. **初始化检查**: 总是检查 `zvec_initialize()` 的返回值
2. **错误处理**: 每次 API 调用后检查返回值，使用 `zvec_get_last_error()` 获取详情
3. **资源释放**: 确保所有创建的资源都被正确释放
4. **批量操作**: 使用批量插入/更新/删除提高性能
5. **索引选择**:
   - 小规模数据 (< 10 万): 使用 Flat 索引
   - 中等规模 (10 万 -1000 万): 使用 HNSW 索引
   - 大规模 (> 1000 万): 使用 IVF 索引
6. **查询优化**: 合理使用过滤条件减少扫描范围
