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
#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {

ParquetBufferID::ParquetBufferID(const std::string &filename, int column,
                                 int row_group)
    : filename(filename), column(column), row_group(row_group) {
  struct stat file_stat;
  if (stat(filename.c_str(), &file_stat) == 0) {
    file_id = file_stat.st_ino;
    std::filesystem::path p(filename);
    auto ftime = std::filesystem::last_write_time(p);
    mtime = static_cast<std::uint64_t>(ftime.time_since_epoch().count());
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
  for (auto &array : payload.arrow->chunks()) {
    auto &buffers = array->data()->buffers;
    for (size_t buf_idx = 0; buf_idx < buffers.size(); ++buf_idx) {
      if (buffers[buf_idx] == nullptr) {
        continue;
      }
      payload.arrow_refs.emplace_back(buffers[buf_idx]);
      size += buffers[buf_idx]->capacity();
      std::shared_ptr<arrow::Buffer> hijacked_buffer(buffers[buf_idx].get(),
                                                     [](arrow::Buffer *) {});
      buffers[buf_idx] = hijacked_buffer;
    }
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
