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

#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/buffer/parquet_hash_table.h>

namespace zvec {
namespace ailego {

ParquetBufferID::ParquetBufferID(std::string &filename, int column,
                                 int row_group)
    : filename(filename), column(column), row_group(row_group) {
  struct stat file_stat;
  if (stat(filename.c_str(), &file_stat) == 0) {
    // file_stat.st_ino contains the inode number
    // file_stat.st_dev contains the device ID
    // Together they uniquely identify a file
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
    const ParquetBufferContextHandle &handle_)
    : buffer_id_(handle_.buffer_id_), arrow_(handle_.arrow_) {
  if (arrow_) {
    ParquetBufferPool::get_instance().acquire_locked(buffer_id_);
  }
}

ParquetBufferContextHandle::~ParquetBufferContextHandle() {
  if (arrow_) {
    ParquetBufferPool::get_instance().release(buffer_id_);
  }
}

arrow::Status ParquetBufferPool::acquire(ParquetBufferID buffer_id,
                                         ParquetBufferContext &context) {
  // TODO: file handler and memory pool can be optimized
  arrow::MemoryPool *mem_pool = arrow::default_memory_pool();

  // Open file
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  const auto &file_name = buffer_id.filename;
  ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(file_name));

  // Open reader
  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(input, mem_pool));

  // Perform read
  int row_group = buffer_id.row_group;
  int column = buffer_id.column;
  auto s = reader->RowGroup(row_group)->Column(column)->Read(&context.arrow);
  if (!s.ok()) {
    LOG_ERROR("Failed to read parquet file[%s]", file_name.c_str());
    context.arrow = nullptr;
    return s;
  }

  context.size = 0;
  context.arrow_refs.clear();
  // Compute the memory usage and hijack Arrow's buffers with our
  // implementation
  for (auto &array : context.arrow->chunks()) {
    auto &buffers = array->data()->buffers;
    for (size_t buf_idx = 0; buf_idx < buffers.size(); ++buf_idx) {
      if (buffers[buf_idx] == nullptr) {
        continue;
      }
      // Keep references to original buffers to prevent premature deletion
      context.arrow_refs.emplace_back(buffers[buf_idx]);
      context.size += buffers[buf_idx]->capacity();
      // Create hijacked buffer with custom deleter that notifies us when
      // Arrow is finished with the buffer
      std::shared_ptr<arrow::Buffer> hijacked_buffer(
          buffers[buf_idx].get(), ArrowBufferDeleter(this, buffer_id));
      buffers[buf_idx] = hijacked_buffer;
    }
  }

  return arrow::Status::OK();
}

ParquetBufferContextHandle ParquetBufferPool::acquire_buffer(
    ParquetBufferID buffer_id) {
  std::shared_ptr<arrow::ChunkedArray> arrow{nullptr};
  {
    std::shared_lock<std::shared_mutex> lock(table_mutex_);
    auto iter = table_.find(buffer_id);
    if (iter != table_.end()) {
      arrow = acquire(buffer_id);
      if (arrow != nullptr) {
        return ParquetBufferContextHandle(buffer_id, arrow);
      }
    }
  }
  {
    bool found = !MemoryLimitPool::get_instance().is_full();
    if (!found) {
      for (int i = 0; i < 5; i++) {
        BlockEvictionQueue::get_instance().recycle();
        found = !MemoryLimitPool::get_instance().is_full();
        if (found) {
          break;
        }
      }
    }
    if (!found) {
      LOG_ERROR("Failed to acquire parquet buffer: %s",
                buffer_id.to_string().c_str());
      return ParquetBufferContextHandle();
    }
    std::unique_lock<std::shared_mutex> lock(table_mutex_);
    if (acquire(buffer_id, table_[buffer_id]).ok()) {
      MemoryLimitPool::get_instance().acquire_parquet(table_[buffer_id].size);
      arrow = set_block_acquired(buffer_id);
      return ParquetBufferContextHandle(buffer_id, arrow);
    } else {
      LOG_ERROR("Failed to acquire parquet buffer: %s",
                buffer_id.to_string().c_str());
      return ParquetBufferContextHandle();
    }
  }
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::set_block_acquired(
    ParquetBufferID buffer_id) {
  ParquetBufferContext &context = table_[buffer_id];
  while (true) {
    int current_count = context.ref_count.load(std::memory_order_relaxed);
    if (current_count >= 0) {
      if (context.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        return context.arrow;
      }
    } else {
      if (context.ref_count.compare_exchange_weak(current_count, 1,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
        context.load_count.fetch_add(1, std::memory_order_relaxed);
        return context.arrow;
      }
    }
  }
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::acquire(
    ParquetBufferID buffer_id) {
  auto iter = table_.find(buffer_id);
  if (iter == table_.end()) {
    return nullptr;
  }
  ParquetBufferContext &context = table_[buffer_id];
  while (true) {
    int current_count = context.ref_count.load(std::memory_order_acquire);
    if (current_count < 0) {
      return nullptr;
    }
    if (context.ref_count.compare_exchange_weak(
            current_count, current_count + 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      if (current_count == 0) {
        context.load_count.fetch_add(1, std::memory_order_relaxed);
      }
      return context.arrow;
    }
  }
  return nullptr;
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::acquire_locked(
    ParquetBufferID buffer_id) {
  std::shared_lock<std::shared_mutex> lock(table_mutex_);
  return acquire(buffer_id);
}

void ParquetBufferPool::release(ParquetBufferID buffer_id) {
  std::shared_lock<std::shared_mutex> lock(table_mutex_);
  auto iter = table_.find(buffer_id);
  if (iter == table_.end()) {
    return;
  }
  ParquetBufferContext &context = table_[buffer_id];
  if (context.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    BlockEvictionQueue::BlockType block;
    block.parquet_buffer_block.first = buffer_id;
    block.parquet_buffer_block.second = context.load_count.load();
    BlockEvictionQueue::get_instance().add_single_block(block, 0);
  }
}

void ParquetBufferPool::evict(ParquetBufferID buffer_id) {
  std::unique_lock<std::shared_mutex> lock(table_mutex_);
  auto iter = table_.find(buffer_id);
  if (iter == table_.end()) {
    return;
  }
  ParquetBufferContext &context = table_[buffer_id];
  int expected = 0;
  if (context.ref_count.compare_exchange_strong(
          expected, std::numeric_limits<int>::min())) {
    MemoryLimitPool::get_instance().release_parquet(context.size);
    context.arrow = nullptr;
    context.arrow_refs.clear();
  }
}

bool ParquetBufferPool::is_dead_node(BlockEvictionQueue::BlockType &block) {
  std::shared_lock<std::shared_mutex> lock(table_mutex_);
  auto iter = table_.find(block.parquet_buffer_block.first);
  if (iter == table_.end()) {
    return true;
  }
  return iter->second.load_count.load() != block.parquet_buffer_block.second;
}

}  // namespace ailego
}  // namespace zvec