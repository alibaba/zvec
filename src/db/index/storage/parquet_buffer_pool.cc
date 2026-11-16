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
#include <atomic>
#include <limits>
#include <optional>
#include <unordered_set>
#include <arrow/array/array_binary.h>
#include <arrow/array/concatenate.h>
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

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace zvec {

arrow::Result<std::shared_ptr<const ParquetFileContext>>
ParquetFileContext::make(
    const std::string &filename,
    std::shared_ptr<arrow::io::RandomAccessFile> opened_file) {
  if (auto *mapped =
          dynamic_cast<arrow::io::MemoryMappedFile *>(opened_file.get())) {
    // Keep the existing pread-based cache loader without reopening a path
    // that may now refer to another file.
#if defined(_WIN32) || defined(_WIN64)
    int fd = _dup(mapped->file_descriptor());
#else
    int fd = fcntl(mapped->file_descriptor(), F_DUPFD_CLOEXEC, 0);
#endif
    if (fd < 0) {
      return arrow::Status::IOError("Failed to duplicate opened parquet file: ",
                                    filename);
    }
    auto readable = arrow::io::ReadableFile::Open(fd);
    if (!readable.ok()) {
#if defined(_WIN32) || defined(_WIN64)
      _close(fd);
#else
      close(fd);
#endif
      return readable.status();
    }
    opened_file = std::move(readable).ValueOrDie();
  }
  return std::shared_ptr<const ParquetFileContext>(
      std::make_shared<ParquetFileContext>(filename, std::move(opened_file)));
}

ParquetBufferID::ParquetBufferID(
    const std::string &filename,
    const std::shared_ptr<arrow::io::RandomAccessFile> &file, int column,
    int row_group)
    : column(column),
      row_group(row_group),
      input(file),
      filename_(std::make_shared<const std::string>(filename)) {
  int fd = -1;
  if (auto *readable = dynamic_cast<arrow::io::ReadableFile *>(file.get())) {
    fd = readable->file_descriptor();
  } else if (auto *mapped =
                 dynamic_cast<arrow::io::MemoryMappedFile *>(file.get())) {
    fd = mapped->file_descriptor();
  }
  // Identify the opened file, so a pathname replacement cannot mix the
  // cache identity with bytes from a different file.
#if defined(_WIN32) || defined(_WIN64)
  BY_HANDLE_FILE_INFORMATION info;
  if (fd >= 0 && GetFileInformationByHandle(
                     reinterpret_cast<HANDLE>(_get_osfhandle(fd)), &info)) {
    file_id =
        (static_cast<uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    device_id = info.dwVolumeSerialNumber;
    mtime = static_cast<int64_t>(
        (static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
        info.ftLastWriteTime.dwLowDateTime);
    return;
  }
#else
  struct stat file_stat;
  if (fd >= 0 && fstat(fd, &file_stat) == 0) {
    file_id = file_stat.st_ino;
    device_id = file_stat.st_dev;
#if defined(__APPLE__)
    mtime = static_cast<int64_t>(file_stat.st_mtimespec.tv_sec) * 1000000000 +
            file_stat.st_mtimespec.tv_nsec;
#else
    mtime = static_cast<int64_t>(file_stat.st_mtim.tv_sec) * 1000000000 +
            file_stat.st_mtim.tv_nsec;
#endif
    return;
  }
#endif
  // Backends without a local descriptor cannot safely share a pathname
  // identity across opens. Isolate their caches for this open lifetime.
  static std::atomic<uint64_t> next_generation{1};
  generation = next_generation.fetch_add(1, std::memory_order_relaxed);
}

const std::string ParquetBufferID::to_string() const {
  std::string msg{"Buffer["};
  msg += "parquet: " + std::string(filename()) + "[" + std::to_string(file_id) +
         "]" + ", column: " + std::to_string(column) +
         ", row_group: " + std::to_string(row_group);
  msg += ", mtime: " + std::to_string(mtime);
  msg += ", device: " + std::to_string(device_id) +
         ", generation: " + std::to_string(generation);
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

namespace {
// Bound outstanding borrowed scan buffers across readers. Leave most of the
// managed pool available for loads; retained batches fall back to owned copies.
std::atomic<size_t> scan_lease_bytes{0};

bool ScanBufferBytes(const std::shared_ptr<arrow::ArrayData> &data,
                     size_t limit, size_t *bytes) {
  for (const auto &buffer : data->buffers) {
    if (!buffer) continue;
    const auto size = static_cast<size_t>(buffer->capacity());
    if (size > limit - *bytes) return false;
    *bytes += size;
  }
  for (const auto &child : data->child_data) {
    if (!ScanBufferBytes(child, limit, bytes)) return false;
  }
  return !data->dictionary || ScanBufferBytes(data->dictionary, limit, bytes);
}

}  // namespace

namespace detail {
struct ParquetScanBufferLease {
  ~ParquetScanBufferLease() {
    // Publish budget availability only after releasing the cache pin.
    handle.reset();
    scan_lease_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  }
  std::optional<ParquetBufferContextHandle> handle;
  size_t bytes{0};
};
}  // namespace detail

namespace {
using ScanBufferLease = detail::ParquetScanBufferLease;

std::shared_ptr<arrow::ArrayData> LeaseArrayData(
    const std::shared_ptr<arrow::ArrayData> &source,
    const std::shared_ptr<ScanBufferLease> &lease) {
  auto data = source->Copy();
  for (auto &buffer : data->buffers) {
    if (buffer) buffer = std::shared_ptr<arrow::Buffer>(lease, buffer.get());
  }
  for (auto &child : data->child_data) child = LeaseArrayData(child, lease);
  if (data->dictionary) {
    data->dictionary = LeaseArrayData(data->dictionary, lease);
  }
  return data;
}

std::shared_ptr<ScanBufferLease> MakeScanLease(
    ParquetBufferContextHandle &handle) {
  const auto *source = handle.borrowed_data();
  if (!source) return nullptr;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t capacity = pool.capacity();
  // Borrowing pins already-accounted cache buffers; it allocates no data.
  // Use idle headroom to avoid copying an otherwise resident scan. Keep at
  // least three quarters reclaimable, and restore the smaller bound when
  // live consumers use more than half the pool. Existing leases stay valid.
  const size_t limit = capacity / (pool.used() <= capacity / 2 ? 4 : 8);
  size_t bytes = 0;
  for (const auto &chunk : source->chunks()) {
    if (!ScanBufferBytes(chunk->data(), limit, &bytes)) return nullptr;
  }
  auto lease = std::make_shared<ScanBufferLease>();
  size_t used = scan_lease_bytes.load(std::memory_order_relaxed);
  do {
    if (used > limit || bytes > limit - used) return nullptr;
  } while (!scan_lease_bytes.compare_exchange_weak(used, used + bytes,
                                                   std::memory_order_relaxed));
  lease->bytes = bytes;
  lease->handle.emplace(std::move(handle));
  return lease;
}

std::shared_ptr<arrow::ChunkedArray> MakeLeasedColumn(
    const std::shared_ptr<ScanBufferLease> &lease) {
  if (!lease) return nullptr;
  const auto *source = lease->handle->borrowed_data();
  arrow::ArrayVector chunks;
  chunks.reserve(source->num_chunks());
  for (const auto &chunk : source->chunks()) {
    chunks.push_back(arrow::MakeArray(LeaseArrayData(chunk->data(), lease)));
  }
  return std::make_shared<arrow::ChunkedArray>(std::move(chunks),
                                               source->type());
}

bool RetainParquetColumnBuffers(
    const std::shared_ptr<arrow::ArrayData> &data,
    detail::ParquetBufferPayload &payload, size_t &size,
    std::unordered_set<const arrow::Buffer *> &seen) {
  for (auto &buffer : data->buffers) {
    if (!buffer) continue;
    if (seen.insert(buffer.get()).second) {
      const auto bytes = static_cast<size_t>(buffer->capacity());
      if (bytes > (std::numeric_limits<size_t>::max)() - size) return false;
      payload.arrow_refs.push_back(buffer);
      size += bytes;
    }
    // Cache ownership belongs to arrow_refs, including nested values and
    // dictionaries. Consumers either hold a pin or copy the selected data.
    buffer =
        std::shared_ptr<arrow::Buffer>(buffer.get(), [](arrow::Buffer *) {});
  }
  for (const auto &child : data->child_data) {
    if (!RetainParquetColumnBuffers(child, payload, size, seen)) return false;
  }
  return !data->dictionary ||
         RetainParquetColumnBuffers(data->dictionary, payload, size, seen);
}
}  // namespace

std::shared_ptr<arrow::ChunkedArray>
ParquetBufferContextHandle::borrow_for_scan() && {
  return MakeLeasedColumn(MakeScanLease(*this));
}

std::shared_ptr<arrow::ChunkedArray>
ParquetBufferContextHandle::borrow_for_scan(
    const ParquetFileContext &context) && {
  if (!arrow_) return nullptr;
  std::shared_ptr<ScanBufferLease> lease;
  {
    std::lock_guard<std::mutex> lock(context.scan_lease_mutex_);
    auto &slot =
        context.scan_leases_[{buffer_id_.column, buffer_id_.row_group}];
    lease = slot.lock();
    if (lease && !ParquetBufferIDEqual{}(lease->handle->buffer_id_, buffer_id_))
      lease.reset();
    if (!lease) {
      lease = MakeScanLease(*this);
      if (lease) slot = lease;
    }
  }
  if (!lease) return nullptr;
  // A reused lease already owns its pin. Consume this redundant handle only
  // after successful lookup; admission failure must leave it usable for copy.
  ParquetBufferContextHandle redundant(std::move(*this));
  return MakeLeasedColumn(lease);
}

namespace {
// Keep hot cache values contiguous across scan batches, with a bounded copy
// workspace. Tight budgets retain the original small decoded chunks.
constexpr int64_t kCachedStringChunkBytes = 32 * 1024 * 1024;
constexpr size_t kCoalesceHeadroom = 2 * kCachedStringChunkBytes;

int64_t CachedStringChunkBytes(const arrow::Array &array) {
  int64_t bytes = 0;
  for (const auto &buffer : array.data()->buffers) {
    if (buffer) {
      bytes += std::min(buffer->size(), kCachedStringChunkBytes + 1 - bytes);
      if (bytes > kCachedStringChunkBytes) break;
    }
  }
  return bytes;
}

arrow::Result<std::shared_ptr<arrow::ChunkedArray>> CoalesceCachedStringChunks(
    std::shared_ptr<arrow::ChunkedArray> column,
    arrow::MemoryPool *memory_pool) {
  const auto id = column->type()->id();
  auto &budget = ailego::MemoryLimitPool::get_instance();
  if ((id != arrow::Type::STRING && id != arrow::Type::LARGE_STRING) ||
      column->num_chunks() < 2 || budget.available() < kCoalesceHeadroom)
    return column;

  bool mergeable = false;
  auto previous = CachedStringChunkBytes(*column->chunk(0));
  for (int i = 1; i < column->num_chunks(); ++i) {
    const auto bytes = CachedStringChunkBytes(*column->chunk(i));
    if (previous + bytes <= kCachedStringChunkBytes) {
      mergeable = true;
      break;
    }
    previous = bytes;
  }
  if (!mergeable) return column;

  const auto type = column->type();
  const size_t count = static_cast<size_t>(column->num_chunks());
  // Standard Arrow containers keep their normal allocator. Admit temporary
  // pointer vectors and one extra Array wrapper before allocating them.
  arrow::ArrayVector chunks, output, group;
  try {
    chunks = column->chunks();
    output.reserve(count);
    group.reserve(count);
  } catch (const std::bad_alloc &) {
    return column;
  }
  column.reset();

  for (size_t begin = 0; begin < count;) {
    if (budget.available() < kCoalesceHeadroom) {
      for (size_t i = begin; i < count; ++i)
        output.push_back(std::move(chunks[i]));
      break;
    }
    size_t end = begin;
    int64_t bytes = 0;
    while (end < count) {
      const auto next = CachedStringChunkBytes(*chunks[end]);
      if (end != begin && bytes + next > kCachedStringChunkBytes) break;
      bytes += next;
      ++end;
      if (bytes >= kCachedStringChunkBytes) break;
    }
    if (end == begin + 1) {
      output.push_back(std::move(chunks[begin]));
      begin = end;
      continue;
    }
    group.clear();
    for (size_t i = begin; i < end; ++i) group.push_back(std::move(chunks[i]));
    auto joined = [&]() -> arrow::Result<std::shared_ptr<arrow::Array>> {
      try {
        return arrow::Concatenate(group, memory_pool);
      } catch (const std::bad_alloc &) {
        return arrow::Status::OutOfMemory("Optional cached string coalescing");
      }
    }();
    if (!joined.ok()) {
      if (!joined.status().IsOutOfMemory()) return joined.status();
      // Coalescing is optional. Under concurrent pressure publish the valid
      // decoded chunks, including any prefix that was already coalesced.
      for (auto &chunk : group) output.push_back(std::move(chunk));
      for (size_t i = end; i < count; ++i)
        output.push_back(std::move(chunks[i]));
      break;
    }
    output.push_back(std::move(joined).ValueOrDie());
    // Release this source group before allocating the next destination. Data
    // copy workspace stays bounded by one group and uses the ordinary Arrow
    // pool.
    group.clear();
    begin = end;
  }
  return arrow::ChunkedArray::Make(std::move(output), type);
}
}  // namespace

bool detail::ParquetBufferLoader::load(const ParquetBufferID &buffer_id,
                                       ParquetBufferPayload &payload,
                                       size_t &size) {
  arrow::MemoryPool *mem_pool = arrow::default_memory_pool();

  auto input = buffer_id.opened_file();
  const auto file_name = buffer_id.filename();
  if (!input) {
    LOG_ERROR("Opened parquet file is no longer available: %s",
              file_name.data());
    return false;
  }

  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto reader_result = parquet::arrow::OpenFile(input, mem_pool);
  if (!reader_result.ok()) {
    LOG_ERROR("Failed to create parquet reader[%s]: %s", file_name.data(),
              reader_result.status().ToString().c_str());
    return false;
  }
  reader = std::move(*reader_result);

  int row_group = buffer_id.row_group;
  int column = buffer_id.column;
  auto s = reader->RowGroup(row_group)->Column(column)->Read(&payload.arrow);
  if (!s.ok()) {
    LOG_ERROR("Failed to read parquet file[%s]: %s", file_name.data(),
              s.ToString().c_str());
    payload.arrow = nullptr;
    return false;
  }

  auto coalesced =
      CoalesceCachedStringChunks(std::move(payload.arrow), mem_pool);
  if (!coalesced.ok()) return false;
  payload.arrow = std::move(coalesced).ValueOrDie();
  size = 0;
  payload.arrow_refs.clear();
  std::unordered_set<const arrow::Buffer *> seen;
  for (auto &array : payload.arrow->chunks()) {
    if (!RetainParquetColumnBuffers(array->data(), payload, size, seen)) {
      LOG_ERROR("Parquet column buffer size overflow: %s", file_name.data());
      return false;
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
  auto &cache = cache_for(buffer_id);
  auto arrow = cache.retain(buffer_id);
  if (!arrow) {
    // Hold the shared load slot through decoding, charging and publication.
    // acquire() rechecks the entry and coalesces concurrent loads of one key.
    LoadSlot slot(*this);
    arrow = cache.acquire(buffer_id);
  }
  if (!arrow) {
    LOG_ERROR("Failed to acquire parquet buffer: %s",
              buffer_id.to_string().c_str());
    return ParquetBufferContextHandle();
  }
  return ParquetBufferContextHandle(std::move(buffer_id), std::move(arrow));
}

ParquetBufferContextHandle ParquetBufferPool::try_acquire_buffer(
    ParquetBufferID buffer_id) {
  auto value = cache_for(buffer_id).retain(buffer_id);
  return ParquetBufferContextHandle(std::move(buffer_id), std::move(value));
}

bool ParquetBufferPool::small_owned_row(const arrow::ChunkedArray &column,
                                        int64_t row) {
  for (const auto &chunk : column.chunks()) {
    if (row >= chunk->length()) {
      row -= chunk->length();
      continue;
    }
    if (chunk->IsNull(row)) return true;
    switch (chunk->type_id()) {
      case arrow::Type::BOOL:
      case arrow::Type::INT32:
      case arrow::Type::INT64:
      case arrow::Type::UINT32:
      case arrow::Type::UINT64:
      case arrow::Type::FLOAT:
      case arrow::Type::DOUBLE:
        return true;
      case arrow::Type::STRING:
        return static_cast<const arrow::StringArray &>(*chunk).value_length(
                   row) <= 4096;
      case arrow::Type::BINARY:
        return static_cast<const arrow::BinaryArray &>(*chunk).value_length(
                   row) <= 4096;
      default:
        return false;
    }
  }
  return false;
}

void ParquetBufferPool::release(const ParquetBufferID &buffer_id) {
  cache_for(buffer_id).release(buffer_id);
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::retain(
    const ParquetBufferID &buffer_id) {
  return cache_for(buffer_id).retain(buffer_id);
}

ParquetBufferPool::Cache &ParquetBufferPool::cache_for(
    const ParquetBufferID &buffer_id) {
  // Stable identity components select the lock shard; the cache still checks
  // the full key, including the pathname, for equality.
  const uint64_t hash = buffer_id.file_id ^ buffer_id.generation ^
                        (static_cast<uint64_t>(buffer_id.column) * 31) ^
                        static_cast<uint64_t>(buffer_id.row_group);
  return caches_[hash % kCacheShards].cache;
}

ParquetBufferPool::LoadSlot::LoadSlot(ParquetBufferPool &pool) : pool_(pool) {
  std::unique_lock<std::mutex> lock(pool_.load_mutex_);
  pool_.load_cv_.wait(lock, [&] {
    return pool_.active_loads_ < Cache::kDefaultMaxConcurrentLoads;
  });
  ++pool_.active_loads_;
}

ParquetBufferPool::LoadSlot::~LoadSlot() {
  {
    std::lock_guard<std::mutex> lock(pool_.load_mutex_);
    --pool_.active_loads_;
  }
  pool_.load_cv_.notify_one();
}

}  // namespace zvec
