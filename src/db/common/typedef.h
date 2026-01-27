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

#include <zvec/ailego/logger/logger.h>
#include "error_code.h"

using idx_t = uint64_t;

#define PROXIMA_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &) = delete;             \
  TypeName &operator=(const TypeName &) = delete;


#define COLLECTION_FORMAT " collection[%s] "

#define CLOG_DEBUG(format, ...) \
  LOG_DEBUG(format COLLECTION_FORMAT, ##__VA_ARGS__, collection_name().c_str())

#define CLOG_INFO(format, ...) \
  LOG_INFO(format COLLECTION_FORMAT, ##__VA_ARGS__, collection_name().c_str())

#define CLOG_WARN(format, ...) \
  LOG_WARN(format COLLECTION_FORMAT, ##__VA_ARGS__, collection_name().c_str())

#define CLOG_ERROR(format, ...) \
  LOG_ERROR(format COLLECTION_FORMAT, ##__VA_ARGS__, collection_name().c_str())

#define CLOG_FATAL(format, ...) \
  LOG_FATAL(format COLLECTION_FORMAT, ##__VA_ARGS__, collection_name().c_str())

#define ELOG_ERROR(format, ...) \
  LOG_ERROR(format " errno[%s] ", ##__VA_ARGS__, std::strerror(errno))

#define WAL_FORMAT " wal_path_[%s] "

#define WLOG_DEBUG(format, ...) \
  LOG_DEBUG(format WAL_FORMAT, ##__VA_ARGS__, wal_path_.c_str())

#define WLOG_INFO(format, ...) \
  LOG_INFO(format WAL_FORMAT, ##__VA_ARGS__, wal_path_.c_str())


#define WLOG_WARN(format, ...) \
  LOG_WARN(format WAL_FORMAT, ##__VA_ARGS__, wal_path_.c_str())

#define WLOG_ERROR(format, ...) \
  LOG_ERROR(format WAL_FORMAT, ##__VA_ARGS__, wal_path_.c_str())

#define WLOG_FATAL(format, ...) \
  LOG_FATAL(format WAL_FORMAT, ##__VA_ARGS__, wal_path_.c_str())

#define SEGMENT_FORMAT " segment[%zu] collection[%s] "

#define SLOG_DEBUG(format, ...)                                         \
  LOG_DEBUG(format SEGMENT_FORMAT, ##__VA_ARGS__, (size_t)segment_id(), \
            collection_name().c_str())

#define SLOG_INFO(format, ...)                                         \
  LOG_INFO(format SEGMENT_FORMAT, ##__VA_ARGS__, (size_t)segment_id(), \
           collection_name().c_str())

#define SLOG_WARN(format, ...)                                         \
  LOG_WARN(format SEGMENT_FORMAT, ##__VA_ARGS__, (size_t)segment_id(), \
           collection_name().c_str())

#define SLOG_ERROR(format, ...)                                         \
  LOG_ERROR(format SEGMENT_FORMAT, ##__VA_ARGS__, (size_t)segment_id(), \
            collection_name().c_str())

#define SLOG_FATAL(format, ...)                                         \
  LOG_FATAL(format SEGMENT_FORMAT, ##__VA_ARGS__, (size_t)segment_id(), \
            collection_name().c_str())

#define COLUMN_FORMAT " column[%s] segment[%zu] collection[%s] "

#define LLOG_DEBUG(format, ...)                                         \
  LOG_DEBUG(format COLUMN_FORMAT, ##__VA_ARGS__, column_name().c_str(), \
            (size_t)segment_id(), collection_name().c_str())

#define LLOG_INFO(format, ...)                                         \
  LOG_INFO(format COLUMN_FORMAT, ##__VA_ARGS__, column_name().c_str(), \
           (size_t)segment_id(), collection_name().c_str())

#define LLOG_WARN(format, ...)                                         \
  LOG_WARN(format COLUMN_FORMAT, ##__VA_ARGS__, column_name().c_str(), \
           (size_t)segment_id(), collection_name().c_str())

#define LLOG_ERROR(format, ...)                                         \
  LOG_ERROR(format COLUMN_FORMAT, ##__VA_ARGS__, column_name().c_str(), \
            (size_t)segment_id(), collection_name().c_str())

#define LLOG_FATAL(format, ...)                                         \
  LOG_FATAL(format COLUMN_FORMAT, ##__VA_ARGS__, column_name().c_str(), \
            (size_t)segment_id(), collection_name().c_str())

#define CHECK_STATUS(status, expect)                                         \
  if (status != expect) {                                                    \
    LOG_ERROR("Check status failed. status[%d] expect[%d]", status, expect); \
    return PROXIMA_ZVEC_ERROR_CODE(StatusError);                             \
  }

#define CHECK_STATUS_CLOSURE(status, expect)                                 \
  if (status != expect) {                                                    \
    LOG_ERROR("Check status failed. status[%d] expect[%d]", status, expect); \
    done->set_code(PROXIMA_ZVEC_ERROR_CODE(StatusError));                    \
    return;                                                                  \
  }

#define CHECK_RETURN(ret, expect_ret) \
  if (ret != expect_ret) {            \
    return ret;                       \
  }

#define CHECK_RETURN_WITH_LOG(ret, expect_ret, format, ...) \
  if (ret != expect_ret) {                                  \
    LOG_ERROR(format, ##__VA_ARGS__);                       \
    return ret;                                             \
  }

#define CHECK_RETURN_WITH_CLOG(ret, expect_ret, format, ...) \
  if (ret != expect_ret) {                                   \
    CLOG_ERROR(format, ##__VA_ARGS__);                       \
    return ret;                                              \
  }

#define CHECK_RETURN_WITH_SLOG(ret, expect_ret, format, ...) \
  if (ret != expect_ret) {                                   \
    SLOG_ERROR(format, ##__VA_ARGS__);                       \
    return ret;                                              \
  }

#define CHECK_RETURN_WITH_LLOG(ret, expect_ret, format, ...) \
  if (ret != expect_ret) {                                   \
    LLOG_ERROR(format, ##__VA_ARGS__);                       \
    return ret;                                              \
  }

#define CHECK_DESTROY_RETURN_STATUS(status, expect)                     \
  if (status != expect) {                                               \
    LOG_ERROR("Collection[%s] is already destroyed.",                   \
              schema_->name().c_str());                                 \
    return Status::InvalidArgument("collection is already destroyed."); \
  }

#define CHECK_DESTROY_RETURN_STATUS_EXPECTED(status, expect)          \
  if (status != expect) {                                             \
    LOG_ERROR("Collection[%s] is already destroyed.",                 \
              schema_->name().c_str());                               \
    return tl::make_unexpected(                                       \
        Status::InvalidArgument("collection is already destroyed.")); \
  }

#define CHECK_RETURN_STATUS(status) \
  if (!status.ok()) {               \
    return status;                  \
  }

#define CHECK_RETURN_STATUS_EXPECTED(status) \
  if (!status.ok()) {                        \
    return tl::make_unexpected(status);      \
  }

#define CHECK_COLLECTION_READONLY_RETURN_STATUS \
  CHECK_READONLY_RETURN_STATUS(Collection)

#define CHECK_SEGMENT_READONLY_RETURN_STATUS \
  CHECK_READONLY_RETURN_STATUS(Segment)

#define CHECK_READONLY_RETURN_STATUS(type)                \
  if (options_.read_only_) {                              \
    return Status::InvalidArgument(#type                  \
                                   " is "                 \
                                   "opened in read-only " \
                                   "mode");               \
  }

#define CHECK_COLLECTION_READONLY_RETURN_STATUS_EXPECTED \
  CHECK_READONLY_RETURN_STATUS_EXPECTED(Collection)

#define CHECK_SEGMENT_READONLY_RETURN_STATUS_EXPECTED \
  CHECK_READONLY_RETURN_STATUS_EXPECTED(Segment)

#define CHECK_READONLY_RETURN_STATUS_EXPECTED(type)                           \
  if (options_.read_only_) {                                                  \
    return tl::make_unexpected(Status::InvalidArgument(#type                  \
                                                       " is "                 \
                                                       "opened in read-only " \
                                                       "mode"));              \
  }
