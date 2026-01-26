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

#include "buffer_manager.h"
#include <atomic>
#include <mutex>
#include <ailego/logger/logger.h>
#include <ailego/pattern/defer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include "ailego/internal/platform.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <arrow/api.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


namespace zvec {


namespace ailego {


namespace {


struct IDHash {
  size_t operator()(const BufferID &buffer_id) const {
    size_t hash = std::hash<int>{}(static_cast<int>(buffer_id.type));
    hash = hash ^ (std::hash<uint64_t>{}(buffer_id.file_id));
    if (buffer_id.type == BufferID::TYPE::PARQUET) {
      hash = hash * 31 + std::hash<int>{}(buffer_id.parquet().column);
      hash = hash * 31 + std::hash<int>{}(buffer_id.parquet().row_group);
    } else if (buffer_id.type == BufferID::TYPE::VECTOR) {
      hash = hash * 31 + std::hash<uint32_t>{}(buffer_id.vector().offset);
    }
    return hash;
  }
};


struct IDEqual {
  bool operator()(const BufferID &a, const BufferID &b) const {
    if (a.type != b.type) {
      return false;
    }
    if (a.file_name != b.file_name) {
      return false;
    }
    if (a.file_id != b.file_id) {
      return false;
    }
    if (a.mtime != b.mtime) {
      return false;
    }
    if (a.type == BufferID::TYPE::PARQUET) {
      return a.parquet().column == b.parquet().column &&
             a.parquet().row_group == b.parquet().row_group;
    } else if (a.type == BufferID::TYPE::VECTOR) {
      return a.vector().offset == b.vector().offset;
    } else {
      return false;
    }
  }
};


}  // namespace


struct BufferManager::BufferContext {
  BufferContext(const BufferID &id, BufferPool *p) : id(id), pool(p) {};
  BufferContext(const BufferContext &) = delete;
  BufferContext(BufferContext &&) = delete;
  BufferContext &operator=(const BufferContext &) = delete;
  BufferContext &operator=(BufferContext &&) = delete;


  ~BufferContext() {
    if (vector) {
      free(vector);
    }
  }


  typedef std::unique_ptr<BufferManager::BufferContext> Pointer;


  enum State : uint32_t {
    IDLE = 0,      // Empty and not held by any users, not in LRU
    RESERVED = 1,  // Pinned by a user but no data yet, not in LRU
    IN_USE = 2,    // Pinned by a user and data is present, not in LRU
    CACHED = 3,    // Data is present but not held by any users, in LRU
    ERROR = 4      // Something went wrong, not in LRU
  };


  // Identifier for the buffer
  BufferID id;

  // Current state
  State state{IDLE};

  // The size of the buffer
  uint32_t size{0};

  // Handle of the file backing this buffer
  File file;

  // The number of external references to this buffer (via pin/unpin)
  std::atomic<uint32_t> refs_buf{0};

  // The number of external references to this context (via BufferHandle)
  std::atomic<uint32_t> refs_context{0};

  BufferPool *pool{nullptr};

  // A shared pointer to the buffers allocated for arrow parquet data
  std::shared_ptr<arrow::ChunkedArray> arrow{nullptr};

  // Guard original arrow buffers to prevent premature deletion
  std::vector<std::shared_ptr<arrow::Buffer>> arrow_refs{};

  // A pointer to the buffer allocated for vector data
  void *vector{nullptr};

  // Doubly linked LRU list
  BufferContext *next{nullptr};
  BufferContext *prev{nullptr};


  // Return a string representation of the status
  const std::string status_string() const;

  // Populate the buffer with parquet data
  arrow::Status read_arrow_parquet();

  // Populate the buffer with vector data
  bool read_vector();
};


const std::string BufferManager::BufferContext::status_string() const {
  std::string msg{id.to_string() + ": "};
  switch (state) {
    case State::IDLE: {
      msg += "Idle";
      break;
    }
    case State::RESERVED: {
      msg += "Reserved";
      break;
    }
    case State::IN_USE: {
      msg += "In use";
      break;
    }
    case State::CACHED: {
      msg += "Cached";
      break;
    }
    case State::ERROR: {
      msg += "Error";
      break;
    }
  }
  return msg;
}


arrow::Status BufferManager::BufferContext::read_arrow_parquet() {
  // TODO: file handler and memory pool can be optimized
  arrow::MemoryPool *mem_pool = arrow::default_memory_pool();

  // Open file
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  const auto &file_name = id.file_name;
  ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(file_name));

  // Open reader
  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(input, mem_pool));

  // Perform read
  int row_group = id.parquet().row_group;
  int column = id.parquet().column;
  auto s = reader->RowGroup(row_group)->Column(column)->Read(&arrow);
  if (!s.ok()) {
    LOG_ERROR("Failed to read parquet file[%s]", file_name.c_str());
    arrow = nullptr;
    return s;
  }

  // Compute the memory usage and hijack Arrow's buffers with our implementation
  for (auto &array : arrow->chunks()) {
    auto &buffers = array->data()->buffers;
    for (size_t buf_idx = 0; buf_idx < buffers.size(); ++buf_idx) {
      if (buffers[buf_idx] == nullptr) {
        continue;
      }
      // Keep references to original buffers to prevent premature deletion
      arrow_refs.emplace_back(buffers[buf_idx]);
      size += buffers[buf_idx]->capacity();
      // Create hijacked buffer with custom deleter that notifies us when Arrow
      // is finished with the buffer
      std::shared_ptr<arrow::Buffer> hijacked_buffer(
          buffers[buf_idx].get(), BufferManager::ArrowBufferDeleter(this));
      buffers[buf_idx] = hijacked_buffer;
    }
  }

  return arrow::Status::OK();
}


bool BufferManager::BufferContext::read_vector() {
  const auto &file_name = id.file_name;
  if (!file.is_valid()) {
    if (!File::IsExist(file_name)) {
      LOG_ERROR("File[%s] does not exist", file_name.c_str());
      return false;
    }
    if (!File::IsRegular(file_name)) {
      LOG_ERROR("[%s] is not a regular file", file_name.c_str());
      return false;
    }
    if (!file.open(file_name.c_str(), true, false)) {
      LOG_ERROR("Failed to open file[%s]", file_name.c_str());
      return false;
    }
  }
  AILEGO_DEFER([this] { file.close(); });
  uint32_t len = id.vector().length;
  auto ret = posix_memalign((void **)&vector, 64, len);  // 64-byte alignment
  if (ret != 0 || vector == nullptr) {
    LOG_ERROR("Failed to allocate buffer for file[%s]", file_name.c_str());
    return false;
  }
  uint32_t offset = id.vector().offset;
  if (file.read(offset, vector, len) != len) {
    LOG_ERROR("Failed to read file[%s]", file_name.c_str());
    free(vector);
    vector = nullptr;
    return false;
  }
  size = len;
  return true;
}


// Thread-safe buffer pool implementation.
//
// BufferContext states:
// 1. Must exist in the lookup (hash) table.
// 2. LRU list presence:
//    - In LRU: holds memory but not pinned by any users
//    - Not in LRU: either holds memory pinned by users, or doesn't hold memory
// 3. External references: when an external user acquires a context and pins the
//    memory, that context is removed from LRU list; when they unpins the
//    memory, that context is moved to LRU list if it was the last reference.
//
// Any operation on the hash table is protected by mutex_table_.
// Any change to context state and LRU list is protected by mutex_context_.
//
class BufferManager::BufferPool {
 public:
  explicit BufferPool(uint64_t limit) : limit_(limit) {
    sentinel_.next = &sentinel_;
    sentinel_.prev = &sentinel_;
  }


  BufferContext *acquire_locked(BufferID &id) {
    std::lock_guard<std::mutex> lock(mutex_context_);
    if (auto iter = table_.find(id); iter != table_.end()) {
      return iter->second.get();
    }
    auto [iter, _] =
        table_.emplace(id, std::make_unique<BufferContext>(id, this));
    return iter->second.get();
  }


  void try_release_context_locked(BufferContext *context) {
    if (context->refs_context.load() != 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_table_);
    if (context->refs_context.load() != 0) {
      return;
    }
    if (context->state == BufferContext::State::IDLE) {
      table_.erase(context->id);
    }
  }


  void pin_locked(BufferContext *ctx) {
    std::lock_guard<std::mutex> lock(mutex_context_);
    if (ctx->state == BufferContext::State::IDLE) {
      return pin_at_IDLE(ctx);
    }
    if (ctx->state == BufferContext::State::IN_USE) {
      return pin_at_IN_USE(ctx);
    }
    if (ctx->state == BufferContext::State::CACHED) {
      return pin_at_CACHED(ctx);
    }
    if (ctx->state == BufferContext::State::ERROR) {
      return;
    }
  }


  bool unpin_locked(BufferContext *ctx) {
    uint32_t prev_refs = ctx->refs_buf.fetch_sub(1);
    if (prev_refs > 1) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_context_);
    if (ctx->refs_buf.load() == 0 &&
        ctx->state != BufferContext::State::CACHED) {
      ctx->state = BufferContext::State::CACHED;
      LRU_insert(ctx);
      return true;
    } else {
      return false;
    }
  }


  void LRU_insert_locked(BufferContext *context) {
    std::lock_guard<std::mutex> lock(mutex_context_);
    LRU_insert(context);
  }


  void LRU_remove_locked(BufferContext *context) {
    std::lock_guard<std::mutex> lock(mutex_context_);
    LRU_remove(context);
  }


  uint64_t usage() const {
    return usage_;
  }


 private:
  void pin_at_IDLE(BufferContext *ctx) {
    ctx->state = BufferContext::State::RESERVED;

    while (usage_ >= limit_) {
      // The tail of LRU list is the least recently used context
      BufferContext *victim = sentinel_.prev;
      if (victim == &sentinel_) {  // No victim could be found
        ctx->state = BufferContext::State::ERROR;
        return;
      }
      if (victim->state == BufferContext::State::ERROR) {
        LRU_remove(victim);
        try_release_context_locked(ctx);
        continue;
      }
      if (victim->id.type == BufferID::TYPE::PARQUET) {
        victim->arrow_refs.clear();
      } else {
        free(victim->vector);
        victim->vector = nullptr;
      }
      victim->state = BufferContext::State::IDLE;
      LRU_remove(victim);
      try_release_context_locked(ctx);
      usage_ -= victim->size;
    }

    if (ctx->id.type == BufferID::TYPE::PARQUET) {
      if (ctx->read_arrow_parquet().ok()) {
        ctx->state = BufferContext::State::IN_USE;
        ctx->refs_buf.fetch_add(ctx->arrow_refs.size());
        usage_ += ctx->size;
      } else {
        LOG_ERROR("Failed to read to %s", ctx->id.to_string().c_str());
        ctx->state = BufferContext::State::ERROR;
      }
    } else {
      if (ctx->read_vector()) {
        ctx->state = BufferContext::State::IN_USE;
        ctx->refs_buf.fetch_add(1);
        usage_ += ctx->size;
      } else {
        LOG_ERROR("Failed to read to %s", ctx->id.to_string().c_str());
        ctx->state = BufferContext::State::ERROR;
      }
    }
  }


  void pin_at_IN_USE(BufferContext *ctx) {
    if (ctx->id.type == BufferID::TYPE::PARQUET) {
      ctx->refs_buf.fetch_add(ctx->arrow_refs.size());
    } else {
      ctx->refs_buf.fetch_add(1);
    }
  }


  void pin_at_CACHED(BufferContext *ctx) {
    if (ctx->id.type == BufferID::TYPE::PARQUET) {
      ctx->refs_buf.fetch_add(ctx->arrow_refs.size());
    } else {
      ctx->refs_buf.fetch_add(1);
    }
    LRU_remove(ctx);
    ctx->state = BufferContext::State::IN_USE;
  }


  void LRU_insert(BufferContext *context) {
    if (context->refs_buf > 0) {
      return;  // Already pinned, should not be evicted
    }
    if (context->next != nullptr || context->prev != nullptr) {
      return;
    }
    // Insert the context to the head of LRU list
    context->next = sentinel_.next;
    context->prev = &sentinel_;
    sentinel_.next = context;
    context->next->prev = context;
    inactive_ += context->size;
  }


  void LRU_remove(BufferContext *context) {
    if (context->next == nullptr) {
      return;  // Not in LRU list
    }
    context->next->prev = context->prev;
    context->prev->next = context->next;
    context->next = nullptr;
    context->prev = nullptr;
    inactive_ -= context->size;
  }

 private:
  using Table =
      std::unordered_map<BufferID, BufferContext::Pointer, IDHash, IDEqual>;

  uint64_t limit_;
  std::atomic<uint64_t> usage_{0};
  std::atomic<uint64_t> inactive_{0};

  Table table_{};
  std::mutex mutex_table_{};
  BufferContext sentinel_{BufferID{}, this};  // LRU list sentinel
  std::mutex mutex_context_{};
};


BufferManager::ArrowBufferDeleter::ArrowBufferDeleter(BufferContext *c)
    : context(c) {}


void BufferManager::ArrowBufferDeleter::operator()(arrow::Buffer *) {
  context->pool->unpin_locked(context);
}


BufferHandle::BufferHandle(BufferContext *context) : context_(context) {
  if (context_ != nullptr) {
    pool_ = context_->pool;
    context_->refs_context.fetch_add(1);
  }
}


BufferHandle::~BufferHandle() {
  if (context_ != nullptr) {
    uint32_t prev_refs = context_->refs_context.fetch_sub(1);
    if (prev_refs > 1) {
      return;
    }
    if (context_->state == BufferContext::State::IDLE) {
      pool_->try_release_context_locked(context_);
    }
  }
}


std::shared_ptr<arrow::ChunkedArray> BufferHandle::pin_parquet_data() {
  pool_->pin_locked(context_);
  return context_->arrow;
}


void *BufferHandle::pin_vector_data() {
  if (!context_) {
    return nullptr;
  }
  pool_->pin_locked(context_);
  return context_->vector;
}


bool BufferHandle::unpin_vector_data() {
  if (!context_) {
    return true;
  }
  return pool_->unpin_locked(context_);
}


uint32_t BufferHandle::references() const {
  return context_->refs_buf.load();
}


uint32_t BufferHandle::size() const {
  return context_->size;
}


void BufferManager::init(uint64_t limit, uint32_t num_shards) {
  pools_.clear();
  uint64_t limit_per_shard = ailego_align(limit / num_shards, 4096);
  for (uint32_t i = 0; i < num_shards; ++i) {
    auto pool = new BufferPool(limit_per_shard);
    pools_.push_back(pool);
  }
  LOG_INFO(
      "BufferManager initialized with [%u] buffer pools, [%zu] bytes memory "
      "limit per pool, total memory limit [%zu] bytes",
      num_shards, (size_t)limit_per_shard, (size_t)limit);
}


BufferHandle BufferManager::acquire(BufferID &buffer_id) {
  static IDHash id_hash{};
  auto hash_val = id_hash(buffer_id);
  auto ctx = pools_[hash_val % pools_.size()]->acquire_locked(buffer_id);
  return BufferHandle(ctx);
}


std::unique_ptr<BufferHandle> BufferManager::acquire_ptr(BufferID &buffer_id) {
  static IDHash id_hash{};
  auto hash_val = id_hash(buffer_id);
  auto ctx = pools_[hash_val % pools_.size()]->acquire_locked(buffer_id);
  return std::make_unique<BufferHandle>(ctx);
}


uint64_t BufferManager::total_size_in_bytes() const {
  uint64_t total_usage = 0;
  for (auto pool : pools_) {
    total_usage += pool->usage();
  }
  return total_usage;
}


BufferManager::~BufferManager() {
  for (auto pool : pools_) {
    delete pool;
  }
}


}  // namespace ailego


}  // namespace zvec