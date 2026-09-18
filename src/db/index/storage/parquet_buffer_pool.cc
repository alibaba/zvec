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

#include "parquet_buffer_pool.h"
#include <limits>
#include <unordered_set>
#include <arrow/array/array_binary.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>

namespace zvec {

namespace {

void RetainAndDetachArrowBuffers(
    const std::shared_ptr<arrow::ArrayData> &data,
    detail::ParquetBufferPayload *payload,
    std::unordered_set<const arrow::Buffer *> *seen, size_t *size) {
  if (!data) {
    return;
  }
  for (auto &buffer : data->buffers) {
    if (!buffer) {
      continue;
    }
    std::shared_ptr<arrow::Buffer> retained = buffer;
    if (seen->insert(retained.get()).second) {
      payload->arrow_refs.emplace_back(retained);
      const int64_t arrow_capacity = retained->capacity();
      if (arrow_capacity < 0 || static_cast<uint64_t>(arrow_capacity) >
                                    std::numeric_limits<size_t>::max()) {
        *size = std::numeric_limits<size_t>::max();
      } else {
        const size_t capacity = static_cast<size_t>(arrow_capacity);
        *size = capacity > std::numeric_limits<size_t>::max() - *size
                    ? std::numeric_limits<size_t>::max()
                    : *size + capacity;
      }
    }
    buffer =
        std::shared_ptr<arrow::Buffer>(retained.get(), [](arrow::Buffer *) {});
  }
  for (const auto &child : data->child_data) {
    RetainAndDetachArrowBuffers(child, payload, seen, size);
  }
  RetainAndDetachArrowBuffers(data->dictionary, payload, seen, size);
}

std::shared_ptr<arrow::ArrayData> CloneWithPinnedBuffers(
    const std::shared_ptr<arrow::ArrayData> &data,
    const std::shared_ptr<ParquetBufferContextHandle> &pin) {
  if (!data) {
    return nullptr;
  }
  auto clone = data->Copy();
  for (auto &buffer : clone->buffers) {
    if (buffer) {
      buffer = std::shared_ptr<arrow::Buffer>(pin, buffer.get());
    }
  }
  for (auto &child : clone->child_data) {
    child = CloneWithPinnedBuffers(child, pin);
  }
  clone->dictionary = CloneWithPinnedBuffers(clone->dictionary, pin);
  return clone;
}

}  // namespace

ParquetBufferID::ParquetBufferID(const std::string &filename, int column,
                                 int row_group)
    : filename(filename), column(column), row_group(row_group) {
  const auto path = ailego::FileHelper::PathFromUtf8(filename);
  bool has_stat_mtime = false;
  int64_t stat_mtime = 0;
#if defined(_WIN32) || defined(_WIN64)
  struct _stat64 file_stat;
  if (!path.empty() && _wstat64(path.c_str(), &file_stat) == 0) {
#else
  struct stat file_stat;
  if (stat(path.c_str(), &file_stat) == 0) {
#endif
    file_id = file_stat.st_ino;
    has_stat_mtime = true;
    stat_mtime = static_cast<int64_t>(file_stat.st_mtime);
  }

  std::error_code ec;
  const auto ftime = std::filesystem::last_write_time(path, ec);
  if (!ec) {
    mtime = static_cast<int64_t>(ftime.time_since_epoch().count());
  } else if (has_stat_mtime) {
    mtime = stat_mtime;
  }
}

const std::string ParquetBufferID::to_string() const {
  std::string msg{"Buffer["};
  msg += "parquet: " + filename + "[" + std::to_string(file_id) + "]" +
         ", column: " + std::to_string(column) +
         ", row_group: " + std::to_string(row_group);
  msg += ", mtime: " + std::to_string(mtime);
  msg += "]";
  return msg;
}

ParquetBufferContextHandle::ParquetBufferContextHandle(
    const ParquetBufferContextHandle &handle)
    : buffer_id_(handle.buffer_id_) {
  if (handle.arrow_) {
    arrow_ = ParquetBufferPool::get_instance().retain(buffer_id_);
  }
}

ParquetBufferContextHandle::~ParquetBufferContextHandle() {
  if (arrow_) {
    ParquetBufferPool::get_instance().release(buffer_id_);
  }
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferContextHandle::data() const {
  if (!arrow_) {
    return nullptr;
  }
  auto pin = std::make_shared<ParquetBufferContextHandle>(*this);
  if (!pin->arrow_) {
    return nullptr;
  }
  // Alias every Arrow buffer to the cache pin while sharing payload bytes.
  arrow::ArrayVector chunks;
  chunks.reserve(arrow_->num_chunks());
  for (const auto &chunk : arrow_->chunks()) {
    chunks.emplace_back(
        arrow::MakeArray(CloneWithPinnedBuffers(chunk->data(), pin)));
  }
  return std::make_shared<arrow::ChunkedArray>(std::move(chunks),
                                               arrow_->type());
}

bool detail::ParquetBufferLoader::load(const ParquetBufferID &buffer_id,
                                       ParquetBufferPayload &payload,
                                       size_t &size) {
  arrow::MemoryPool *mem_pool = arrow::default_memory_pool();

  std::shared_ptr<arrow::io::RandomAccessFile> input;
  const auto &file_name = buffer_id.filename;
  auto input_result = arrow::io::ReadableFile::Open(file_name);
  if (!input_result.ok()) {
    LOG_ERROR("Failed to open parquet file[%s]: %s", file_name.c_str(),
              input_result.status().ToString().c_str());
    return false;
  }
  input = *input_result;

  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto reader_result = parquet::arrow::OpenFile(input, mem_pool);
  if (!reader_result.ok()) {
    LOG_ERROR("Failed to create parquet reader[%s]: %s", file_name.c_str(),
              reader_result.status().ToString().c_str());
    return false;
  }
  reader = std::move(*reader_result);

  int row_group = buffer_id.row_group;
  int column = buffer_id.column;
  auto s = reader->RowGroup(row_group)->Column(column)->Read(&payload.arrow);
  if (!s.ok()) {
    LOG_ERROR("Failed to read parquet file[%s]: %s", file_name.c_str(),
              s.ToString().c_str());
    payload.arrow = nullptr;
    return false;
  }

  size = 0;
  payload.arrow_refs.clear();
  std::unordered_set<const arrow::Buffer *> seen;
  for (auto &array : payload.arrow->chunks()) {
    RetainAndDetachArrowBuffers(array->data(), &payload, &seen, &size);
  }

  return true;
}

void detail::ParquetBufferLoader::clear(ParquetBufferPayload &payload) const {
  payload.arrow = nullptr;
  payload.arrow_refs.clear();
}

ParquetBufferContextHandle ParquetBufferPool::acquire_buffer(
    ParquetBufferID buffer_id) {
  auto arrow = cache_.acquire(buffer_id);
  if (!arrow) {
    LOG_ERROR("Failed to acquire parquet buffer: %s",
              buffer_id.to_string().c_str());
    return ParquetBufferContextHandle();
  }
  return ParquetBufferContextHandle(buffer_id, arrow);
}

void ParquetBufferPool::release(ParquetBufferID buffer_id) {
  cache_.release(buffer_id);
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::retain(
    ParquetBufferID buffer_id) {
  return cache_.retain(buffer_id);
}

}  // namespace zvec
