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

#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>

#if defined(__linux__)
#include <ailego/io/io_backend_def.h>
#include <ailego/io/iouring_loader.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
static ssize_t zvec_pread(int fd, void *buf, size_t count, size_t offset) {
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov = {};
  ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytes_read = 0;
  if (!ReadFile(handle, buf, static_cast<DWORD>(count), &bytes_read, &ov)) {
    return -1;
  }
  return static_cast<ssize_t>(bytes_read);
}
static ssize_t zvec_pwrite(int fd, const void *buf, size_t count,
                           size_t offset) {
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov = {};
  ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytes_written = 0;
  if (!WriteFile(handle, buf, static_cast<DWORD>(count), &bytes_written, &ov)) {
    return -1;
  }
  return static_cast<ssize_t>(bytes_written);
}
#else
#include <unistd.h>
static inline ssize_t zvec_pread(int fd, void *buf, size_t count,
                                 size_t offset) {
  return ::pread(fd, buf, count, static_cast<off_t>(offset));
}
static inline ssize_t zvec_pwrite(int fd, const void *buf, size_t count,
                                  size_t offset) {
  return ::pwrite(fd, buf, count, static_cast<off_t>(offset));
}
#endif

namespace zvec {
namespace ailego {

const size_t kVectorPageSize = MemoryHelper::PageSize();

namespace {
constexpr size_t kBlockingAioBatchSize = VecBufferPool::kWritebackBatchPages;
}

VecBufferPool::~VecBufferPool() {
  // Finish queued writes before page buffers, latches, and descriptors go
  // away. The synchronous pass below catches any prior writeback error.
  stop_writeback();
  // Flush dirty pages before releasing memory and descriptors.
  (void)this->flush_all();
  // Preserve final cache and writeback statistics after both persistence
  // paths have drained.
  log_stats();
  page_table_.force_evict_all_loaded();
  const size_t writable_metadata_bytes = block_mutex_metadata_bytes() +
                                         writeback_staging_size_ +
                                         writeback_io_staging_charge_;
  block_mutexes_.reset();
  if (writeback_staging_ != nullptr) {
    ailego_free(writeback_staging_);
    writeback_staging_ = nullptr;
  }
#if defined(__linux__)
  writeback_io_uring_.reset();
#endif
  MemoryLimitPool::get_instance().release_metadata(writable_metadata_bytes);
  block_mutex_count_ = 0;
  writeback_staging_size_ = 0;
  writeback_io_staging_charge_ = 0;
  initialized_ = false;
#if defined(_MSC_VER)
  _close(fd_);
  _close(meta_fd_);
#else
  close(fd_);
  close(meta_fd_);
#endif
}

version_t VectorPageTable::next_owner_version() {
  return BlockEvictionQueue::get_instance().next_version();
}

size_t VectorPageTable::metadata_bytes_for_entries(size_t entry_num) {
  if (entry_num > kMaxEntries) {
    return std::numeric_limits<size_t>::max();
  }
  const size_t segment_count =
      entry_num == 0 ? 0 : (entry_num - 1) / kSegmentSize + 1;
  if (segment_count == 0) {
    return 0;
  }
  if (segment_count >
      (std::numeric_limits<size_t>::max() - kSegmentDirectoryBytes) /
          kSegmentMetadataBytes) {
    return std::numeric_limits<size_t>::max();
  }
  return kSegmentDirectoryBytes + segment_count * kSegmentMetadataBytes;
}

void VectorPageTable::initialize_segment(Entry *entries,
                                         MetadataEntry *metadata_entries) {
  for (size_t i = 0; i < kSegmentSize; ++i) {
    entries[i].buffer.store(nullptr, std::memory_order_relaxed);
    entries[i].ref_count.store(kUnloadedRefCount, std::memory_order_relaxed);
    entries[i].in_evict_queue.store(false, std::memory_order_relaxed);
    entries[i].referenced.store(false, std::memory_order_relaxed);
    entries[i].evict_priority.store(0, std::memory_order_relaxed);
    entries[i].ghost_state.store(kNoGhostHistory, std::memory_order_relaxed);
    metadata_entries[i].next_loaded = kInvalidLoadedBlock;
    metadata_entries[i].file_offset = 0;
    metadata_entries[i].admission_state.store(0, std::memory_order_relaxed);
    metadata_entries[i].is_dirty.store(false, std::memory_order_relaxed);
    metadata_entries[i].writeback_pending.store(false,
                                                std::memory_order_relaxed);
    metadata_entries[i].ever_loaded.store(false, std::memory_order_relaxed);
  }
}

bool VectorPageTable::should_admit_miss(block_id_t block_id, uint32_t epoch) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  Entry &entry = entry_at(block_id);

  // Joining an existing residency transition preserves single-flight. A
  // protected hint or hot ghost is also stronger evidence than miss count.
  if (entry.ref_count.load(std::memory_order_acquire) != kUnloadedRefCount ||
      entry.evict_priority.load(std::memory_order_relaxed) >= kNormalPriority ||
      entry.ghost_state.load(std::memory_order_relaxed) == kEvictedHot) {
    return true;
  }

  MetadataEntry &metadata = metadata_entry_at(block_id);
  static constexpr uint32_t kEpochMask = (uint32_t{1} << 24) - 1;
  static constexpr uint8_t kAdmissionThreshold = 2;
  epoch &= kEpochMask;

  uint32_t state = metadata.admission_state.load(std::memory_order_relaxed);
  while (true) {
    const uint32_t previous_epoch = state >> 8;
    const uint8_t previous_count = static_cast<uint8_t>(state);
    const uint32_t age = (epoch - previous_epoch) & kEpochMask;

    uint8_t count = 1;
    if (age == 0) {
      count = previous_count == std::numeric_limits<uint8_t>::max()
                  ? previous_count
                  : static_cast<uint8_t>(previous_count + 1);
    } else if (age == 1) {
      count = static_cast<uint8_t>(previous_count / 2 + 1);
    }
    const uint32_t updated = (epoch << 8) | count;
    if (metadata.admission_state.compare_exchange_weak(
            state, updated, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      // Close the race with a loader that claimed the page while its miss was
      // being recorded; waiting for that load is preferable to duplicate I/O.
      return count >= kAdmissionThreshold ||
             entry.ref_count.load(std::memory_order_acquire) !=
                 kUnloadedRefCount;
    }
  }
}

bool VectorPageTable::init(size_t entry_num) {
  if (entry_num > kMaxEntries) {
    LOG_ERROR(
        "VectorPageTable::init: entry_num=%zu exceeds capacity "
        "(kMaxEntries=%zu, kMaxSegments=%zu); "
        "refusing to init.",
        entry_num, kMaxEntries, kMaxSegments);
    return false;
  }
  const size_t old_entry_num = entry_num_.load(std::memory_order_relaxed);
  const size_t old_count = segment_count_.load(std::memory_order_relaxed);
  if (old_count != 0) {
    if (old_entry_num == entry_num) {
      return true;
    }
    LOG_ERROR(
        "VectorPageTable::init: refusing to replace an initialized table "
        "(old_entries=%zu, requested_entries=%zu)",
        old_entry_num, entry_num);
    return false;
  }
  const size_t need_segments =
      entry_num == 0 ? 0 : (entry_num - 1) / kSegmentSize + 1;
  const size_t charge = metadata_bytes_for_entries(entry_num);
  if (charge == std::numeric_limits<size_t>::max()) {
    LOG_ERROR("VectorPageTable::init: metadata size overflow for %zu entries",
              entry_num);
    return false;
  }
  if (!MemoryLimitPool::get_instance().try_charge_metadata(charge)) {
    LOG_ERROR(
        "VectorPageTable::init: shared memory budget cannot reserve %zu "
        "metadata bytes for %zu entries",
        charge, entry_num);
    return false;
  }

  std::vector<std::unique_ptr<Entry[]>> new_segments;
  std::vector<std::unique_ptr<MetadataEntry[]>> new_metadata_segments;
  std::unique_ptr<Entry *[]> new_segment_directory;
  std::unique_ptr<MetadataEntry *[]> new_metadata_segment_directory;
  try {
    if (need_segments != 0) {
      new_segment_directory = std::make_unique<Entry *[]>(kMaxSegments);
      new_metadata_segment_directory =
          std::make_unique<MetadataEntry *[]>(kMaxSegments);
    }
    new_segments.reserve(need_segments);
    new_metadata_segments.reserve(need_segments);
    for (size_t s = 0; s < need_segments; ++s) {
      auto entries = std::make_unique<Entry[]>(kSegmentSize);
      auto metadata_entries = std::make_unique<MetadataEntry[]>(kSegmentSize);
      initialize_segment(entries.get(), metadata_entries.get());
      new_segments.push_back(std::move(entries));
      new_metadata_segments.push_back(std::move(metadata_entries));
    }
  } catch (const std::bad_alloc &) {
    MemoryLimitPool::get_instance().release_metadata(charge);
    LOG_ERROR(
        "VectorPageTable::init: allocation failed for %zu entries (%zu "
        "metadata bytes)",
        entry_num, charge);
    return false;
  }
  for (size_t s = 0; s < need_segments; ++s) {
    new_segment_directory[s] = new_segments[s].release();
    new_metadata_segment_directory[s] = new_metadata_segments[s].release();
  }
  segments_ = std::move(new_segment_directory);
  metadata_segments_ = std::move(new_metadata_segment_directory);
  // Publish segments before the externally visible entry count.
  segment_count_.store(need_segments, std::memory_order_release);
  entry_num_.store(entry_num, std::memory_order_release);
  return true;
}

bool VectorPageTable::extend(size_t new_entry_num) {
  // The caller serializes page-table extension.
  if (new_entry_num <= entry_num_.load(std::memory_order_relaxed)) {
    return true;
  }
  if (new_entry_num > kMaxEntries) {
    LOG_ERROR(
        "VectorPageTable::extend: new_entry_num=%zu exceeds capacity "
        "(kMaxEntries=%zu, kMaxSegments=%zu); "
        "refusing to extend.",
        new_entry_num, kMaxEntries, kMaxSegments);
    return false;
  }
  const size_t new_segment_count =
      new_entry_num == 0 ? 0 : (new_entry_num - 1) / kSegmentSize + 1;
  const size_t old_count = segment_count_.load(std::memory_order_relaxed);
  const size_t added_segments = new_segment_count - old_count;
  const bool needs_directory = old_count == 0 && new_segment_count != 0;
  if (added_segments > (std::numeric_limits<size_t>::max() -
                        (needs_directory ? kSegmentDirectoryBytes : 0)) /
                           kSegmentMetadataBytes) {
    LOG_ERROR(
        "VectorPageTable::extend: metadata size overflow for %zu new "
        "segments",
        added_segments);
    return false;
  }
  const size_t added_charge = added_segments * kSegmentMetadataBytes +
                              (needs_directory ? kSegmentDirectoryBytes : 0);
  if (!MemoryLimitPool::get_instance().try_charge_metadata(added_charge)) {
    LOG_ERROR(
        "VectorPageTable::extend: shared memory budget cannot reserve %zu "
        "additional metadata bytes (old_entries=%zu, new_entries=%zu)",
        added_charge, entry_num_.load(std::memory_order_relaxed),
        new_entry_num);
    return false;
  }

  std::vector<std::unique_ptr<Entry[]>> new_segments;
  std::vector<std::unique_ptr<MetadataEntry[]>> new_metadata_segments;
  std::unique_ptr<Entry *[]> new_segment_directory;
  std::unique_ptr<MetadataEntry *[]> new_metadata_segment_directory;
  try {
    if (needs_directory) {
      new_segment_directory = std::make_unique<Entry *[]>(kMaxSegments);
      new_metadata_segment_directory =
          std::make_unique<MetadataEntry *[]>(kMaxSegments);
    }
    new_segments.reserve(new_segment_count - old_count);
    new_metadata_segments.reserve(new_segment_count - old_count);
    for (size_t s = old_count; s < new_segment_count; ++s) {
      auto entries = std::make_unique<Entry[]>(kSegmentSize);
      auto metadata_entries = std::make_unique<MetadataEntry[]>(kSegmentSize);
      initialize_segment(entries.get(), metadata_entries.get());
      new_segments.push_back(std::move(entries));
      new_metadata_segments.push_back(std::move(metadata_entries));
    }
  } catch (const std::bad_alloc &) {
    MemoryLimitPool::get_instance().release_metadata(added_charge);
    LOG_ERROR(
        "VectorPageTable::extend: allocation failed for %zu new entries "
        "(%zu additional metadata bytes)",
        new_entry_num, added_charge);
    return false;
  }
  Entry **segment_directory =
      needs_directory ? new_segment_directory.get() : segments_.get();
  MetadataEntry **metadata_segment_directory =
      needs_directory ? new_metadata_segment_directory.get()
                      : metadata_segments_.get();
  for (size_t s = old_count; s < new_segment_count; ++s) {
    const size_t idx = s - old_count;
    segment_directory[s] = new_segments[idx].release();
    metadata_segment_directory[s] = new_metadata_segments[idx].release();
  }
  if (needs_directory) {
    segments_ = std::move(new_segment_directory);
    metadata_segments_ = std::move(new_metadata_segment_directory);
  }
  // Match init() publication order: segments first, entry count last.
  segment_count_.store(new_segment_count, std::memory_order_release);
  entry_num_.store(new_entry_num, std::memory_order_release);
  return true;
}

bool VectorPageTable::rollback_extend(size_t old_entry_num) {
  const size_t current_entry_num = entry_num_.load(std::memory_order_relaxed);
  if (old_entry_num > current_entry_num) {
    return false;
  }
  if (old_entry_num == current_entry_num) {
    return true;
  }
  for (size_t i = old_entry_num; i < current_entry_num; ++i) {
    if (entry_at(i).buffer.load(std::memory_order_relaxed) != nullptr ||
        entry_at(i).ref_count.load(std::memory_order_relaxed) !=
            std::numeric_limits<int>::min() ||
        metadata_entry_at(i).ever_loaded.load(std::memory_order_relaxed)) {
      LOG_ERROR(
          "VectorPageTable::rollback_extend: new entry %zu is already in "
          "use; refusing rollback",
          i);
      return false;
    }
  }

  const size_t old_segment_count =
      old_entry_num == 0 ? 0 : (old_entry_num - 1) / kSegmentSize + 1;
  const size_t current_segment_count =
      segment_count_.load(std::memory_order_relaxed);
  entry_num_.store(old_entry_num, std::memory_order_release);
  segment_count_.store(old_segment_count, std::memory_order_release);
  for (size_t s = old_segment_count; s < current_segment_count; ++s) {
    delete[] segments_[s];
    segments_[s] = nullptr;
    delete[] metadata_segments_[s];
    metadata_segments_[s] = nullptr;
  }
  size_t released_charge =
      (current_segment_count - old_segment_count) * kSegmentMetadataBytes;
  if (old_segment_count == 0) {
    segments_.reset();
    metadata_segments_.reset();
    released_charge += kSegmentDirectoryBytes;
  }
  MemoryLimitPool::get_instance().release_metadata(released_charge);
  return true;
}

char *VectorPageTable::acquire_block(block_id_t block_id, bool record_reuse) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);
  // Pin only resident pages; negative values are transition sentinels.
  int count = e.ref_count.load(std::memory_order_acquire);
  while (ailego_likely(count >= 0)) {
    if (e.ref_count.compare_exchange_weak(count, count + 1,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
      if (record_reuse) {
        const uint32_t sample = next_hit_sample();
        // Reuse policy is approximate: a genuinely hot page is sampled
        // quickly, while the common hit path avoids repeated atomic metadata
        // updates. Also stop policy work after pressure has subsided.
        if ((sample & (kReusePolicySampleRate - 1)) == 0 &&
            adaptive_priority_enabled_ &&
            has_evicted_.load(std::memory_order_relaxed)) {
          const uint8_t ghost_state =
              e.ghost_state.load(std::memory_order_relaxed);
          const uint8_t priority =
              e.evict_priority.load(std::memory_order_relaxed);
          // Most HNSW hits are already protected by the one-time hot-set hint.
          // Avoid global pressure checks and no-op promotion attempts for
          // those pages. A ghost-admitted protected page remains eligible so
          // one sampled reuse can validate its renewed hot history.
          const bool needs_policy_update =
              ghost_state == kGhostAdmitted || priority < kNormalPriority;
          if (needs_policy_update &&
              MemoryLimitPool::get_instance().under_cache_pressure()) {
            // A sampled reuse after ghost admission validates that the page is
            // still hot. Its next protected residency may leave another ghost.
            if (ghost_state == kGhostAdmitted) {
              uint8_t ghost_admitted = kGhostAdmitted;
              (void)e.ghost_state.compare_exchange_strong(
                  ghost_admitted, kNoGhostHistory, std::memory_order_relaxed,
                  std::memory_order_relaxed);
            }
            if (priority < kNormalPriority) {
              (void)promote_evict_priority(block_id, kNormalPriority);
            }
            if (!e.referenced.load(std::memory_order_relaxed)) {
              e.referenced.store(true, std::memory_order_relaxed);
            }
          }
        }
        // Sample the observability counter and CLOCK reference bit together.
        if ((sample & (kHitSampleRate - 1)) == 0) {
          if (!e.referenced.load(std::memory_order_relaxed)) {
            e.referenced.store(true, std::memory_order_relaxed);
          }
          inc_sampled_hit();
        }
      }
      return e.buffer.load(std::memory_order_acquire);
    }
  }
  return nullptr;
}

VectorPageTable::LoadClaimResult VectorPageTable::try_claim_block_load(
    block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  Entry &entry = entry_at(block_id);
  int state = entry.ref_count.load(std::memory_order_acquire);
  while (true) {
    if (state >= 0) {
      return LoadClaimResult::kResident;
    }
    if (state == kLoadingRefCount) {
      return LoadClaimResult::kLoading;
    }
    if (state != kUnloadedRefCount) {
      return LoadClaimResult::kEvicting;
    }
    if (entry.ref_count.compare_exchange_weak(state, kLoadingRefCount,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return LoadClaimResult::kClaimed;
    }
  }
}

bool VectorPageTable::wait_for_block_transition(block_id_t block_id) const {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  const Entry &entry = entry_at(block_id);
  using clock = std::chrono::steady_clock;
  const auto wait_start = clock::now();
  auto last_log = wait_start;
  unsigned spin_count = 0;
  bool warned = false;
  static constexpr auto kHardTimeout = std::chrono::seconds(30);
  while (true) {
    const int state = entry.ref_count.load(std::memory_order_acquire);
    if (state != kLoadingRefCount && state != kEvictingRefCount) {
      return true;
    }

    ++spin_count;
    if (spin_count < 64) {
    } else if (spin_count < 1024) {
      std::this_thread::yield();
    } else if (spin_count < 8192) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto now = clock::now();
    const auto elapsed = now - wait_start;
    if (!warned && elapsed >= std::chrono::milliseconds(100)) {
      LOG_WARN(
          "wait_for_block_transition: long wait on block_id=%zu state=%d "
          "(>=100ms)",
          static_cast<size_t>(block_id), state);
      warned = true;
    }
    if (elapsed >= kHardTimeout) {
      LOG_ERROR(
          "wait_for_block_transition: hard timeout (%lld s) on block_id=%zu "
          "state=%d",
          static_cast<long long>(
              std::chrono::duration_cast<std::chrono::seconds>(elapsed)
                  .count()),
          static_cast<size_t>(block_id), state);
      return false;
    }
    if (elapsed >= std::chrono::seconds(1) &&
        (now - last_log) >= std::chrono::seconds(1)) {
      const auto secs =
          std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
      LOG_ERROR(
          "wait_for_block_transition: block_id=%zu state=%d still busy after "
          "%lld s",
          static_cast<size_t>(block_id), state, static_cast<long long>(secs));
      last_log = now;
    }
  }
}

char *VectorPageTable::publish_claimed_block(block_id_t block_id, char *buffer,
                                             size_t file_offset) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  assert(buffer != nullptr);
  Entry &entry = entry_at(block_id);
  MetadataEntry &metadata = metadata_entry_at(block_id);
  if (entry.ref_count.load(std::memory_order_acquire) != kLoadingRefCount) {
    LOG_ERROR(
        "publish_claimed_block: block_id=%zu is not owned by a loader, "
        "state=%d",
        static_cast<size_t>(block_id),
        entry.ref_count.load(std::memory_order_relaxed));
    MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
    return nullptr;
  }

  metadata.file_offset = file_offset;
  metadata.is_dirty.store(false, std::memory_order_relaxed);
  entry.referenced.store(false, std::memory_order_relaxed);
  metadata.admission_state.store(0, std::memory_order_relaxed);
  if (adaptive_priority_enabled_) {
    uint8_t evicted_hot = kEvictedHot;
    if (entry.ghost_state.compare_exchange_strong(evicted_hot, kGhostAdmitted,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
      // A ghost hit is admitted directly to protected. It must be reused
      // while resident before it is allowed to leave another ghost.
      (void)promote_evict_priority(block_id, kNormalPriority);
      ghost_hot_hits_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (!metadata.ever_loaded.exchange(true, std::memory_order_acq_rel)) {
    size_t head = loaded_head_.load(std::memory_order_relaxed);
    do {
      metadata.next_loaded = head;
    } while (!loaded_head_.compare_exchange_weak(
        head, block_id, std::memory_order_release, std::memory_order_relaxed));
  }
  entry.buffer.store(buffer, std::memory_order_release);
  entry.in_evict_queue.store(true, std::memory_order_relaxed);
  entry.ref_count.store(1, std::memory_order_release);

  BlockEvictionQueue::BlockType block;
  block.owner = this;
  block.owner_key = block_id;
  block.version = owner_version_;
  if (!BlockEvictionQueue::get_instance().add_single_block(
          block, static_cast<int>(
                     entry.evict_priority.load(std::memory_order_relaxed)))) {
    // The final release will take the rare fallback registration path.
    eviction_requeue_failed(block_id, owner_version_);
  }
  return buffer;
}

bool VectorPageTable::cancel_block_load(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  Entry &entry = entry_at(block_id);
  int expected = kLoadingRefCount;
  return entry.ref_count.compare_exchange_strong(expected, kUnloadedRefCount,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed);
}

void VectorPageTable::release_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);

  // Installation normally registers the page; retry only after queue failure.
  if (e.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
    if (e.in_evict_queue.load(std::memory_order_relaxed)) {
      return;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    bool expected = false;
    if (e.in_evict_queue.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed)) {
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = block_id;
      block.version = owner_version_;
      if (!BlockEvictionQueue::get_instance().add_single_block(
              block, static_cast<int>(
                         e.evict_priority.load(std::memory_order_relaxed)))) {
        eviction_requeue_failed(block_id, owner_version_);
      }
    }
  }
}

bool VectorPageTable::evict_block(block_id_t block_id) {
  return do_evict_block(block_id, /*force=*/false);
}

bool VectorPageTable::force_evict_block(block_id_t block_id) {
  return do_evict_block(block_id, /*force=*/true);
}

bool VectorPageTable::reclaim_clean_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &entry = entry_at(block_id);
  int expected = 0;
  if (!entry.ref_count.compare_exchange_strong(expected, kEvictingRefCount)) {
    return false;
  }
  if (metadata_entry_at(block_id).is_dirty.load(std::memory_order_acquire)) {
    entry.ref_count.store(0, std::memory_order_release);
    return false;
  }

  char *buffer = entry.buffer.exchange(nullptr, std::memory_order_acq_rel);
  if (buffer != nullptr) {
    MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
  }
  inc_evict(entry.evict_priority.load(std::memory_order_relaxed));
  entry.in_evict_queue.store(false, std::memory_order_relaxed);
  entry.ref_count.store(kUnloadedRefCount, std::memory_order_release);
  return true;
}

void VectorPageTable::force_evict_all_loaded() {
  size_t block_id = loaded_head_.load(std::memory_order_acquire);
  while (block_id != kInvalidLoadedBlock) {
    const size_t next = metadata_entry_at(block_id).next_loaded;
    assert(is_released(block_id));
    (void)force_evict_block(block_id);
    block_id = next;
  }
}

size_t VectorPageTable::recover_eviction_queue() {
  if (!eviction_recovery_needed_.exchange(false, std::memory_order_acq_rel)) {
    return 0;
  }
  size_t recovered = 0;
  size_t block_id = loaded_head_.load(std::memory_order_acquire);
  while (block_id != kInvalidLoadedBlock) {
    Entry &entry = entry_at(block_id);
    const size_t next = metadata_entry_at(block_id).next_loaded;
    if (entry.buffer.load(std::memory_order_acquire) != nullptr &&
        entry.ref_count.load(std::memory_order_acquire) == 0) {
      bool expected = false;
      if (entry.in_evict_queue.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        BlockEvictionQueue::BlockType block;
        block.owner = this;
        block.owner_key = block_id;
        block.version = owner_version_;
        if (BlockEvictionQueue::get_instance().add_single_block(
                block, static_cast<int>(entry.evict_priority.load(
                           std::memory_order_relaxed)))) {
          ++recovered;
        } else {
          entry.in_evict_queue.store(false, std::memory_order_release);
          eviction_recovery_needed_.store(true, std::memory_order_release);
        }
      }
    }
    block_id = next;
  }
  return recovered;
}

std::array<size_t, VectorPageTable::kPriorityCount>
VectorPageTable::resident_pages_by_priority() const {
  std::array<size_t, kPriorityCount> resident{};
  size_t block_id = loaded_head_.load(std::memory_order_acquire);
  while (block_id != kInvalidLoadedBlock) {
    const Entry &entry = entry_at(block_id);
    const size_t next = metadata_entry_at(block_id).next_loaded;
    if (entry.buffer.load(std::memory_order_acquire) != nullptr) {
      const uint8_t priority =
          entry.evict_priority.load(std::memory_order_relaxed);
      if (priority < kPriorityCount) {
        ++resident[priority];
      }
    }
    block_id = next;
  }
  return resident;
}

bool VectorPageTable::do_evict_block(block_id_t block_id, bool force) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);
  int expected = 0;
  if (e.ref_count.compare_exchange_strong(expected, kEvictingRefCount)) {
    // CLOCK gives recently referenced pages one more queue turn.
    if (!force && e.referenced.load(std::memory_order_relaxed)) {
      e.referenced.store(false, std::memory_order_relaxed);
      inc_second_chance();
      // Preserve logical membership while moving the page to the tail.
      e.ref_count.store(0, std::memory_order_release);
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = block_id;
      block.version = owner_version_;
      if (!BlockEvictionQueue::get_instance().add_single_block(
              block, static_cast<int>(
                         e.evict_priority.load(std::memory_order_relaxed)))) {
        eviction_requeue_failed(block_id, owner_version_);
      }
      return false;  // spared, not reclaimed
    }
    if (!force && adaptive_priority_enabled_) {
      uint8_t expected_priority = kNormalPriority;
      if (e.evict_priority.compare_exchange_strong(
              expected_priority, kLowPriority, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        inc_priority_demotion(kLowPriority);
        uint8_t ghost_state = e.ghost_state.load(std::memory_order_relaxed);
        if (ghost_state == kGhostAdmitted) {
          // A ghost-admitted page that was not reused is stale. Do not let it
          // renew itself indefinitely through repeated reloads.
          e.ghost_state.store(kNoGhostHistory, std::memory_order_relaxed);
        } else if (ghost_state != kEvictedHot) {
          e.ghost_state.store(kEvictedHot, std::memory_order_relaxed);
          ghost_hot_marks_.fetch_add(1, std::memory_order_relaxed);
        }
        // Demotion already gives the page another queue turn. A real reuse can
        // set CLOCK again; an unconditional second chance only amplifies CPU
        // work during sustained pressure.
        e.referenced.store(false, std::memory_order_relaxed);
        e.ref_count.store(0, std::memory_order_release);
        BlockEvictionQueue::BlockType block;
        block.owner = this;
        block.owner_key = block_id;
        block.version = owner_version_;
        if (!BlockEvictionQueue::get_instance().add_single_block(
                block, static_cast<int>(kLowPriority))) {
          eviction_requeue_failed(block_id, owner_version_);
        }
        return false;
      }
    }
    MetadataEntry &metadata = metadata_entry_at(block_id);
    char *buffer = e.buffer.load(std::memory_order_acquire);
    if (buffer && metadata.is_dirty.load(std::memory_order_relaxed)) {
      if (!force && writeback_callback_) {
        bool scheduled = false;
        try {
          scheduled = writeback_callback_(block_id);
        } catch (...) {
          LOG_ERROR(
              "VectorPageTable::evict_block: writeback callback threw for "
              "block_id=%zu",
              static_cast<size_t>(block_id));
        }
        if (scheduled) {
          // Persistence belongs to the pool's writeback worker. Keep this
          // page resident and queued until the worker makes it clean.
          e.ref_count.store(0, std::memory_order_release);
          BlockEvictionQueue::BlockType block;
          block.owner = this;
          block.owner_key = block_id;
          block.version = owner_version_;
          if (!BlockEvictionQueue::get_instance().add_single_block(
                  block, static_cast<int>(e.evict_priority.load(
                             std::memory_order_relaxed)))) {
            e.in_evict_queue.store(false, std::memory_order_relaxed);
            eviction_recovery_needed_.store(true, std::memory_order_release);
          }
          return false;
        }
      }
      int flush_rc = -1;
      if (flush_callback_) {
        try {
          flush_rc = flush_callback_(block_id, buffer, kVectorPageSize,
                                     metadata.file_offset);
        } catch (...) {
          LOG_ERROR(
              "VectorPageTable::evict_block: flush callback threw for "
              "block_id=%zu",
              static_cast<size_t>(block_id));
        }
      } else {
        LOG_ERROR(
            "VectorPageTable::evict_block: dirty block %zu has no flush "
            "callback",
            static_cast<size_t>(block_id));
      }
      if (flush_rc != 0 && !force) {
        // Keep a dirty page resident when writeback fails.
        e.ref_count.store(0, std::memory_order_release);
        BlockEvictionQueue::BlockType block;
        block.owner = this;
        block.owner_key = block_id;
        block.version = owner_version_;
        if (!BlockEvictionQueue::get_instance().add_single_block(
                block, static_cast<int>(
                           e.evict_priority.load(std::memory_order_relaxed)))) {
          e.in_evict_queue.store(false, std::memory_order_relaxed);
          eviction_recovery_needed_.store(true, std::memory_order_release);
        }
        return false;
      }
      if (flush_rc == 0) {
        metadata.is_dirty.store(false, std::memory_order_relaxed);
        inc_dirty_flush();
      } else {
        LOG_ERROR(
            "VectorPageTable::force_evict_block: discarding dirty block %zu "
            "after flush failure during teardown",
            static_cast<size_t>(block_id));
      }
    }
    buffer = e.buffer.exchange(nullptr, std::memory_order_acq_rel);
    if (buffer) {
      MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
    }
    inc_evict(e.evict_priority.load(std::memory_order_relaxed));
    // Clear old membership before publishing the unloaded sentinel.
    e.in_evict_queue.store(false, std::memory_order_relaxed);
    e.ref_count.store(kUnloadedRefCount, std::memory_order_release);
    return true;
  }

  // Do not queue unloaded or transitioning entries.
  if (expected < 0) {
    return false;
  }

  // Move pinned pages to the tail without duplicating membership.
  BlockEvictionQueue::BlockType block;
  block.owner = this;
  block.owner_key = block_id;
  block.version = owner_version_;
  if (!BlockEvictionQueue::get_instance().add_single_block(
          block,
          static_cast<int>(e.evict_priority.load(std::memory_order_relaxed)))) {
    // Let release_block() retry registration when the last pin is dropped.
    eviction_requeue_failed(block_id, owner_version_);
  }
  return false;
}

char *VectorPageTable::set_block_acquired(block_id_t block_id, char *buffer,
                                          size_t file_offset) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  while (true) {
    switch (try_claim_block_load(block_id)) {
      case LoadClaimResult::kClaimed:
        return publish_claimed_block(block_id, buffer, file_offset);
      case LoadClaimResult::kResident: {
        char *resident = acquire_block(block_id, /*record_reuse=*/false);
        if (resident != nullptr) {
          MemoryLimitPool::get_instance().release_buffer(buffer,
                                                         kVectorPageSize);
          return resident;
        }
        break;
      }
      case LoadClaimResult::kLoading:
      case LoadClaimResult::kEvicting:
        if (!wait_for_block_transition(block_id)) {
          MemoryLimitPool::get_instance().release_buffer(buffer,
                                                         kVectorPageSize);
          return nullptr;
        }
        break;
    }
  }
}

VecBufferPool::VecBufferPool(const std::string &filename, bool writable) {
  file_name_ = filename;
  writable_ = writable;
  page_table_.set_adaptive_priority(!writable_);
#if defined(_MSC_VER)
  int flags = writable_ ? (O_RDWR | _O_BINARY) : (O_RDONLY | _O_BINARY);
  fd_ = _open(filename.c_str(), flags, 0644);
  meta_fd_ = _open(filename.c_str(), flags, 0644);
#else
  int base_flags = writable_ ? O_RDWR : O_RDONLY;
  // Buffered channel for unaligned metadata I/O.
  meta_fd_ = ::open(filename.c_str(), base_flags, 0644);
  // Keep metadata buffered, but bypass the kernel page cache for page data.
  // Linux uses O_DIRECT; Darwin provides the equivalent through F_NOCACHE.
  int data_flags = base_flags;
#ifdef O_DIRECT
  data_flags |= O_DIRECT;
#endif
  fd_ = ::open(filename.c_str(), data_flags, 0644);
#ifdef O_DIRECT
  if (fd_ < 0) {
    LOG_WARN(
        "VecBufferPool: open with O_DIRECT failed for file[%s] (errno=%d), "
        "falling back to buffered IO",
        filename.c_str(), errno);
    fd_ = ::open(filename.c_str(), base_flags, 0644);
    direct_io_enabled_ = false;
  } else {
    direct_io_enabled_ = true;
  }
#elif defined(F_NOCACHE)
  if (fd_ >= 0) {
    if (::fcntl(fd_, F_NOCACHE, 1) != 0) {
      const int error = errno;
      LOG_ERROR(
          "VecBufferPool: failed to enable F_NOCACHE for file[%s] "
          "(errno=%d)",
          filename.c_str(), error);
      ::close(fd_);
      fd_ = -1;
      errno = error;
    } else {
      direct_io_enabled_ = true;
    }
  }
#else
  direct_io_enabled_ = false;
#endif
#endif
  if (fd_ < 0 || meta_fd_ < 0) {
    if (fd_ >= 0) {
#if defined(_MSC_VER)
      _close(fd_);
#else
      ::close(fd_);
#endif
    }
    if (meta_fd_ >= 0) {
#if defined(_MSC_VER)
      _close(meta_fd_);
#else
      ::close(meta_fd_);
#endif
    }
    throw std::runtime_error("Failed to open file: " + filename);
  }
#if defined(_MSC_VER)
  struct _stat64 st;
  if (_fstat64(fd_, &st) < 0) {
    _close(fd_);
    _close(meta_fd_);
#else
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    ::close(fd_);
    ::close(meta_fd_);
#endif
    throw std::runtime_error("Failed to stat file: " + filename);
  }
  file_size_ = st.st_size;
  initial_file_size_ = file_size_;
#if defined(__linux__)
  // Select the process-wide backend; thread-local contexts are created lazily.
  io_backend_type_ = direct_io_enabled_ ? IOBackend::Instance().available()
                                        : IOBackendType::kPread;
  aio_enabled_ = io_backend_type_ != IOBackendType::kPread;
#endif
}

size_t VecBufferPool::metadata_bytes_for_page_count(size_t page_count,
                                                    bool writable) {
  const size_t page_table_bytes =
      VectorPageTable::metadata_bytes_for_entries(page_count);
  if (page_table_bytes == std::numeric_limits<size_t>::max()) {
    return page_table_bytes;
  }
  // Writable files can grow after the pool opens. The mutex array cannot be
  // replaced while readers and writers hold stripes, so allocate the stable
  // maximum up front. Besides avoiding cross-page write contention, this
  // lets a 128-page writeback batch hold distinct stripes after extend_file().
  const size_t mutex_count = writable ? kMutexBucketCount : 0;
  const size_t mutex_bytes = mutex_count * sizeof(std::shared_mutex);
  const size_t staging_bytes =
      writable ? kBlockingAioBatchSize * kVectorPageSize : 0;
  size_t io_staging_bytes = 0;
#if defined(__linux__)
  if (writable &&
      IOBackend::Instance().available() == IOBackendType::kIoUring) {
    io_staging_bytes = kBlockingAioBatchSize * kVectorPageSize;
  }
#endif
  if (mutex_bytes > std::numeric_limits<size_t>::max() - staging_bytes ||
      mutex_bytes + staging_bytes >
          std::numeric_limits<size_t>::max() - io_staging_bytes) {
    return std::numeric_limits<size_t>::max();
  }
  const size_t writable_bytes = mutex_bytes + staging_bytes + io_staging_bytes;
  if (page_table_bytes > std::numeric_limits<size_t>::max() - writable_bytes) {
    return std::numeric_limits<size_t>::max();
  }
  return page_table_bytes + writable_bytes;
}

int VecBufferPool::init() {
  if (initialized_) {
    return 0;
  }
  if (writable_) {
    // Configure the potentially allocating callback before reserving metadata.
    try {
      int fd = fd_;
      page_table_.set_flush_callback(
          [fd, &fn = file_name_](block_id_t /*block_id*/, char *buf, size_t sz,
                                 size_t off) -> int {
            ssize_t w = zvec_pwrite(fd, buf, sz, off);
            if (w != static_cast<ssize_t>(sz)) {
              LOG_ERROR(
                  "Buffer pool flush failed: file[%s], offset[%zu], "
                  "expected[%zu], got[%zd]",
                  fn.c_str(), off, sz, w);
              return -1;
            }
            return 0;
          });
      page_table_.set_writeback_callback(
          [this](block_id_t block_id) { return enqueue_writeback(block_id); });
    } catch (const std::bad_alloc &) {
      LOG_ERROR(
          "VecBufferPool::init: failed to allocate flush callback for file[%s]",
          file_name_.c_str());
      return -1;
    }
  }

  const size_t block_num =
      file_size_ == 0 ? 0 : (file_size_ - 1) / kVectorPageSize + 1;
  if (block_num > VectorPageTable::kMaxEntries) {
    LOG_ERROR(
        "VecBufferPool::init: file[%s] needs %zu entries, exceeding "
        "VectorPageTable::kMaxEntries=%zu",
        file_name_.c_str(), block_num, VectorPageTable::kMaxEntries);
    return -1;
  }
  // Writable files grow in place. Keep stripe addresses stable for the pool's
  // lifetime and avoid collapsing future writeback batches onto the few pages
  // present when the file was opened.
  const size_t mutex_count = writable_ ? kMutexBucketCount : 0;
  const size_t mutex_charge = mutex_count * sizeof(std::shared_mutex);
  const size_t staging_charge =
      writable_ ? kBlockingAioBatchSize * kVectorPageSize : 0;
  size_t io_staging_charge = 0;
#if defined(__linux__)
  std::unique_ptr<IoUringRing> writeback_io_uring;
  if (writable_ && io_backend_type_ == IOBackendType::kIoUring) {
    // Keep the estimator and actual reservation stable even if creating this
    // pool's ring fails and it must fall back to pwrite.
    io_staging_charge = kBlockingAioBatchSize * kVectorPageSize;
    try {
      writeback_io_uring = std::make_unique<IoUringRing>();
      if (!writeback_io_uring->setup(kBlockingAioBatchSize)) {
        writeback_io_uring.reset();
        LOG_WARN(
            "VecBufferPool::init: io_uring writeback setup failed for "
            "file[%s], falling back to pwrite",
            file_name_.c_str());
      }
    } catch (const std::bad_alloc &) {
      writeback_io_uring.reset();
      LOG_WARN(
          "VecBufferPool::init: cannot allocate io_uring writeback context "
          "for file[%s], falling back to pwrite",
          file_name_.c_str());
    }
  }
#endif
  const size_t writable_metadata_charge =
      mutex_charge + staging_charge + io_staging_charge;
  if (writable_metadata_charge != 0 &&
      !MemoryLimitPool::get_instance().try_charge_metadata(
          writable_metadata_charge)) {
    LOG_ERROR(
        "VecBufferPool::init: shared memory budget cannot reserve %zu bytes "
        "for %zu page-lock stripes and writeback staging (file=%s)",
        writable_metadata_charge, mutex_count, file_name_.c_str());
    return -1;
  }
  std::unique_ptr<std::shared_mutex[]> mutexes;
  if (mutex_count != 0) {
    try {
      mutexes = std::make_unique<std::shared_mutex[]>(mutex_count);
    } catch (const std::bad_alloc &) {
      MemoryLimitPool::get_instance().release_metadata(
          writable_metadata_charge);
      LOG_ERROR(
          "VecBufferPool::init: failed to allocate %zu page-lock stripes "
          "(file=%s)",
          mutex_count, file_name_.c_str());
      return -1;
    }
  }
  char *writeback_staging = nullptr;
  if (staging_charge != 0) {
    writeback_staging = static_cast<char *>(
        ailego_aligned_malloc(staging_charge, kVectorPageSize));
    if (writeback_staging == nullptr) {
      MemoryLimitPool::get_instance().release_metadata(
          writable_metadata_charge);
      LOG_ERROR(
          "VecBufferPool::init: failed to allocate %zu bytes of writeback "
          "staging (file=%s)",
          staging_charge, file_name_.c_str());
      return -1;
    }
  }
  if (!page_table_.init(block_num)) {
    if (writeback_staging != nullptr) {
      ailego_free(writeback_staging);
    }
    MemoryLimitPool::get_instance().release_metadata(writable_metadata_charge);
    LOG_ERROR(
        "VecBufferPool::init: page_table_ init failed for file[%s], "
        "file_size=%zu, block_num=%zu, required_metadata=%zu",
        file_name_.c_str(), file_size_, block_num,
        metadata_bytes_for_page_count(block_num, writable_));
    return -1;
  }
  block_mutexes_ = std::move(mutexes);
  block_mutex_count_ = mutex_count;
  writeback_staging_ = writeback_staging;
  writeback_staging_size_ = staging_charge;
  writeback_io_staging_charge_ = io_staging_charge;
#if defined(__linux__)
  writeback_io_uring_ = std::move(writeback_io_uring);
#endif
  LOG_DEBUG("entry num: %zu, file_size: %zu", page_table_.entry_num(),
            file_size_);

  initialized_ = true;
  if (writable_) {
    try {
      start_writeback();
    } catch (const std::exception &e) {
      page_table_.set_writeback_callback({});
      LOG_WARN(
          "VecBufferPool::init: failed to start background writeback for "
          "file[%s], falling back to synchronous dirty eviction: %s",
          file_name_.c_str(), e.what());
    } catch (...) {
      page_table_.set_writeback_callback({});
      LOG_WARN(
          "VecBufferPool::init: failed to start background writeback for "
          "file[%s], falling back to synchronous dirty eviction",
          file_name_.c_str());
    }
  }
  return 0;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
  return VecBufferPoolHandle(*this);
}

bool VecBufferPool::enqueue_writeback(block_id_t page_id) {
  if (writeback_error() != 0) {
    return false;
  }
  if (!page_table_.try_mark_writeback_pending(page_id)) {
    return true;
  }

  try {
    {
      std::lock_guard<std::mutex> lock(writeback_mutex_);
      if (writeback_stopping_) {
        page_table_.clear_writeback_pending(page_id);
        return false;
      }
      writeback_queue_.push_back(page_id);
      writeback_requests_.fetch_add(1, std::memory_order_relaxed);
      const uint64_t pending =
          writeback_pending_.fetch_add(1, std::memory_order_relaxed) + 1;
      uint64_t peak = writeback_peak_pending_.load(std::memory_order_relaxed);
      while (peak < pending && !writeback_peak_pending_.compare_exchange_weak(
                                   peak, pending, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      }
    }
  } catch (...) {
    page_table_.clear_writeback_pending(page_id);
    return false;
  }
  writeback_cv_.notify_one();
  return true;
}

void VecBufferPool::start_writeback() {
  std::lock_guard<std::mutex> lock(writeback_mutex_);
  if (writeback_thread_.joinable()) {
    return;
  }
  writeback_stopping_ = false;
  writeback_error_.store(0, std::memory_order_release);
  writeback_thread_ = std::thread([this] { writeback_loop(); });
}

void VecBufferPool::stop_writeback() {
  {
    std::lock_guard<std::mutex> lock(writeback_mutex_);
    if (!writeback_thread_.joinable()) {
      return;
    }
    writeback_stopping_ = true;
  }
  writeback_cv_.notify_all();
  writeback_thread_.join();
}

void VecBufferPool::drain_writeback() {
  std::unique_lock<std::mutex> lock(writeback_mutex_);
  if (!writeback_thread_.joinable()) {
    return;
  }
  writeback_drained_cv_.wait(lock, [this] {
    return writeback_queue_.empty() && writeback_inflight_ == 0;
  });
}

void VecBufferPool::writeback_loop() {
  std::vector<block_id_t> page_ids;
  page_ids.reserve(kBlockingAioBatchSize);
  while (true) {
    {
      std::unique_lock<std::mutex> lock(writeback_mutex_);
      writeback_cv_.wait(lock, [this] {
        return writeback_stopping_ || !writeback_queue_.empty();
      });
      if (writeback_queue_.empty()) {
        if (writeback_stopping_) {
          break;
        }
        continue;
      }
      page_ids.clear();
      while (!writeback_queue_.empty() &&
             page_ids.size() < kBlockingAioBatchSize) {
        page_ids.push_back(writeback_queue_.front());
        writeback_queue_.pop_front();
      }
      writeback_inflight_ += page_ids.size();
    }

    try {
      std::lock_guard<std::mutex> flush_lock(writeback_flush_mutex_);
      flush_writeback_batch(page_ids, writeback_staging_);
    } catch (...) {
      writeback_failures_.fetch_add(page_ids.size(), std::memory_order_relaxed);
      int expected = 0;
      (void)writeback_error_.compare_exchange_strong(
          expected, EIO, std::memory_order_release, std::memory_order_relaxed);
      LOG_ERROR("VecBufferPool writeback threw: file[%s], pages[%zu]",
                file_name_.c_str(), page_ids.size());
    }

    for (block_id_t page_id : page_ids) {
      page_table_.clear_writeback_pending(page_id);
    }
    writeback_pending_.fetch_sub(page_ids.size(), std::memory_order_relaxed);
    for (block_id_t page_id : page_ids) {
      (void)page_table_.reclaim_clean_block(page_id);
    }

    {
      std::lock_guard<std::mutex> lock(writeback_mutex_);
      writeback_inflight_ -= page_ids.size();
      if (writeback_queue_.empty() && writeback_inflight_ == 0) {
        writeback_drained_cv_.notify_all();
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(writeback_mutex_);
    if (writeback_queue_.empty() && writeback_inflight_ == 0) {
      writeback_drained_cv_.notify_all();
    }
  }
}

bool VecBufferPool::flush_writeback_batch(std::vector<block_id_t> &page_ids,
                                          char *staging) {
  if (page_ids.empty()) {
    return true;
  }
  std::sort(page_ids.begin(), page_ids.end());
  page_ids.erase(std::unique(page_ids.begin(), page_ids.end()), page_ids.end());

#if defined(__linux__)
  if (writeback_io_uring_ && writeback_io_uring_->is_valid()) {
    bool all_ok = true;
    size_t pos = 0;
    while (pos < page_ids.size()) {
      std::array<block_id_t, kBlockingAioBatchSize> selected_pages{};
      std::array<char *, kBlockingAioBatchSize> buffers{};
      std::array<size_t, kBlockingAioBatchSize> locked_stripes{};
      std::array<std::shared_lock<std::shared_mutex>, kBlockingAioBatchSize>
          locks;
      size_t selected = 0;

      while (pos < page_ids.size() && selected < kBlockingAioBatchSize) {
        const block_id_t page_id = page_ids[pos];
        if (!page_table_.is_block_dirty(page_id)) {
          ++pos;
          continue;
        }

        const size_t stripe = page_id % block_mutex_count_;
        bool stripe_already_locked = false;
        for (size_t i = 0; i < selected; ++i) {
          if (locked_stripes[i] == stripe) {
            stripe_already_locked = true;
            break;
          }
        }
        // std::shared_mutex does not guarantee recursive shared ownership.
        // Submit the current group before taking the same stripe again.
        if (stripe_already_locked) {
          break;
        }

        char *buffer = page_table_.acquire_block(page_id,
                                                 /*record_reuse=*/false);
        if (buffer == nullptr) {
          ++pos;
          continue;
        }
        locks[selected] =
            std::shared_lock<std::shared_mutex>(block_mutexes_[stripe]);
        if (!page_table_.is_block_dirty(page_id)) {
          locks[selected].unlock();
          page_table_.release_block(page_id);
          ++pos;
          continue;
        }
        selected_pages[selected] = page_id;
        buffers[selected] = buffer;
        locked_stripes[selected] = stripe;
        ++selected;
        ++pos;
      }

      if (selected == 0) {
        continue;
      }

      std::array<IoUringWrite, kBlockingAioBatchSize> requests{};
      for (size_t i = 0; i < selected; ++i) {
        requests[i] = IoUringWrite(selected_pages[i] * kVectorPageSize,
                                   kVectorPageSize, buffers[i]);
      }

      writeback_batches_.fetch_add(1, std::memory_order_relaxed);
      writeback_aio_batches_.fetch_add(1, std::memory_order_relaxed);
      writeback_aio_pages_.fetch_add(selected, std::memory_order_relaxed);
      bool aio_ok = writeback_io_uring_->execute_writes(fd_, requests.data(),
                                                        selected) == 0;
      if (!aio_ok) {
        writeback_aio_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      }

      size_t flushed = 0;
      size_t failed = 0;
      int batch_error = 0;
      for (size_t i = 0; i < selected; ++i) {
        bool page_ok = aio_ok;
        if (!page_ok) {
          errno = 0;
          const ssize_t written =
              zvec_pwrite(fd_, buffers[i], kVectorPageSize,
                          selected_pages[i] * kVectorPageSize);
          page_ok = written == static_cast<ssize_t>(kVectorPageSize);
          if (!page_ok) {
            const int error = errno != 0 ? errno : EIO;
            if (batch_error == 0) {
              batch_error = error;
            }
            LOG_ERROR(
                "VecBufferPool writeback fallback failed: file[%s], "
                "page[%zu], expected[%zu], got[%zd], errno[%d]",
                file_name_.c_str(), static_cast<size_t>(selected_pages[i]),
                kVectorPageSize, written, error);
          }
        }
        if (page_ok) {
          page_table_.clear_dirty(selected_pages[i]);
          ++flushed;
        } else {
          ++failed;
        }
      }
      if (flushed != 0) {
        page_table_.record_dirty_flush(flushed);
        writeback_pages_.fetch_add(flushed, std::memory_order_relaxed);
      }
      if (failed != 0) {
        writeback_failures_.fetch_add(failed, std::memory_order_relaxed);
        int expected = 0;
        (void)writeback_error_.compare_exchange_strong(
            expected, batch_error != 0 ? batch_error : EIO,
            std::memory_order_release, std::memory_order_relaxed);
        all_ok = false;
      }

      for (size_t i = 0; i < selected; ++i) {
        locks[i].unlock();
        page_table_.release_block(selected_pages[i]);
      }
    }
    return all_ok;
  }
#endif

  bool all_ok = true;
  const size_t max_run =
      std::max<size_t>(1, std::min(kBlockingAioBatchSize, block_mutex_count_));
  const size_t run_limit = staging != nullptr ? max_run : 1;
  std::array<char *, kBlockingAioBatchSize> buffers{};
  std::array<std::shared_lock<std::shared_mutex>, kBlockingAioBatchSize> locks;

  size_t pos = 0;
  while (pos < page_ids.size()) {
    const block_id_t run_start = page_ids[pos];
    size_t run_count = 0;
    while (pos + run_count < page_ids.size() && run_count < run_limit &&
           page_ids[pos + run_count] == run_start + run_count) {
      const block_id_t page_id = page_ids[pos + run_count];
      if (!page_table_.is_block_dirty(page_id)) {
        break;
      }
      char *buffer = page_table_.acquire_block(page_id,
                                               /*record_reuse=*/false);
      if (buffer == nullptr) {
        break;
      }
      locks[run_count] = std::shared_lock<std::shared_mutex>(
          block_mutexes_[page_id % block_mutex_count_]);
      if (!page_table_.is_block_dirty(page_id)) {
        locks[run_count].unlock();
        page_table_.release_block(page_id);
        break;
      }
      buffers[run_count] = buffer;
      if (staging != nullptr) {
        std::memcpy(staging + run_count * kVectorPageSize, buffer,
                    kVectorPageSize);
      }
      ++run_count;
    }

    if (run_count == 0) {
      ++pos;
      continue;
    }

    const char *write_buffer = staging != nullptr ? staging : buffers[0];
    // Without staging, resident pages are not contiguous; preserve correctness
    // by submitting one page at a time.
    const size_t submitted_pages = staging != nullptr ? run_count : 1;
    const size_t submitted_size = submitted_pages * kVectorPageSize;
    writeback_batches_.fetch_add(1, std::memory_order_relaxed);
    const ssize_t written = zvec_pwrite(fd_, write_buffer, submitted_size,
                                        run_start * kVectorPageSize);
    const bool ok = written == static_cast<ssize_t>(submitted_size);
    if (ok) {
      for (size_t i = 0; i < submitted_pages; ++i) {
        page_table_.clear_dirty(run_start + i);
      }
      page_table_.record_dirty_flush(submitted_pages);
      writeback_pages_.fetch_add(submitted_pages, std::memory_order_relaxed);
    } else {
      writeback_failures_.fetch_add(submitted_pages, std::memory_order_relaxed);
      int expected = 0;
      const int error = errno != 0 ? errno : EIO;
      (void)writeback_error_.compare_exchange_strong(expected, error,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed);
      LOG_ERROR(
          "VecBufferPool writeback failed: file[%s], offset[%zu], "
          "expected[%zu], got[%zd], errno[%d]",
          file_name_.c_str(), run_start * kVectorPageSize, submitted_size,
          written, error);
      all_ok = false;
    }

    for (size_t i = 0; i < run_count; ++i) {
      locks[i].unlock();
      page_table_.release_block(run_start + i);
    }
    pos += staging != nullptr ? run_count : submitted_pages;
  }
  return all_ok;
}

char *VecBufferPool::acquire_buffer(block_id_t page_id, int retry,
                                    bool record_reuse) {
  assert(page_id < page_table_.entry_num());
  while (true) {
    char *buffer = page_table_.acquire_block(page_id, record_reuse);
    if (buffer) {
      return buffer;
    }

    const auto claim = page_table_.try_claim_block_load(page_id);
    if (claim != VectorPageTable::LoadClaimResult::kClaimed) {
      if (claim == VectorPageTable::LoadClaimResult::kResident) {
        continue;
      }
      if (claim == VectorPageTable::LoadClaimResult::kLoading) {
        singleflight_waits_.fetch_add(1, std::memory_order_relaxed);
      }
      // Recheck the stable state from the beginning after it completes.
      if (!page_table_.wait_for_block_transition(page_id)) {
        return nullptr;
      }
      continue;
    }

    bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
        kVectorPageSize, buffer);
    if (!found && writable_ && retry > 0) {
      int no_progress_waits = 0;
      uint64_t completed = writeback_pages_.load(std::memory_order_relaxed);
      while (!found && no_progress_waits < retry) {
        // Bound foreground queue scanning. Dirty candidates are only queued;
        // disk I/O belongs to the writeback worker.
        (void)BlockEvictionQueue::get_instance().batch_recycle(64);
        found = MemoryLimitPool::get_instance().try_acquire_buffer(
            kVectorPageSize, buffer);
        if (found || writeback_error() != 0) {
          break;
        }

        writeback_waits_.fetch_add(1, std::memory_order_relaxed);
        const auto wait_start = std::chrono::steady_clock::now();
        const bool capacity_released =
            MemoryLimitPool::get_instance().wait_for_available(
                kVectorPageSize, std::chrono::milliseconds(100));
        const auto wait_end = std::chrono::steady_clock::now();
        writeback_wait_us_.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    wait_end - wait_start)
                    .count()),
            std::memory_order_relaxed);
        const uint64_t now = writeback_pages_.load(std::memory_order_relaxed);
        if (capacity_released || now != completed) {
          no_progress_waits = 0;
          completed = now;
        } else {
          ++no_progress_waits;
        }
      }
    } else if (!found) {
      for (int i = 0; i < retry; i++) {
        BlockEvictionQueue::get_instance().recycle();
        found = MemoryLimitPool::get_instance().try_acquire_buffer(
            kVectorPageSize, buffer);
        if (found) {
          break;
        }
      }
    }
    if (!found) {
      const auto memory_stats = MemoryLimitPool::get_instance().stats();
      const auto page_stats = page_table_.stats();
      const int error = writeback_error();
      if (error != 0) {
        LOG_ERROR(
            "Buffer pool allocation stopped after writeback failure: "
            "file[%s], page_id[%zu], error[%d], used[%zu], "
            "committed[%zu], free_buffers[%zu]",
            file_name_.c_str(), page_id, error, memory_stats.used,
            memory_stats.committed, memory_stats.free_buffers);
      } else if (writable_) {
        LOG_WARN(
            "Buffer pool allocation made no progress: file[%s], "
            "page_id[%zu], used[%zu], committed[%zu], free_buffers[%zu], "
            "evict[%llu], second_chance[%llu]",
            file_name_.c_str(), page_id, memory_stats.used,
            memory_stats.committed, memory_stats.free_buffers,
            static_cast<unsigned long long>(page_stats.evict),
            static_cast<unsigned long long>(page_stats.second_chance));
      } else {
        LOG_DEBUG(
            "Buffer pool failed to get free buffer: file[%s], page_id[%zu], "
            "used[%zu], committed[%zu], free_buffers[%zu], evict[%llu], "
            "second_chance[%llu]",
            file_name_.c_str(), page_id, memory_stats.used,
            memory_stats.committed, memory_stats.free_buffers,
            static_cast<unsigned long long>(page_stats.evict),
            static_cast<unsigned long long>(page_stats.second_chance));
      }
      (void)page_table_.cancel_block_load(page_id);
      return nullptr;
    }

    const size_t page_offset = page_id * kVectorPageSize;
    // Count one miss per page for which this thread won the load claim.
    miss_count_.fetch_add(1, std::memory_order_relaxed);
    // Newly extended pages start zeroed; reload evicted pages from disk.
    if (writable_ && page_offset >= initial_file_size_ &&
        !page_table_.is_ever_loaded(page_id)) {
      std::memset(buffer, 0, kVectorPageSize);
    } else {
      // Accept and zero-pad an unaligned final page.
      const size_t read_len =
          direct_io_enabled_
              ? kVectorPageSize
              : std::min(kVectorPageSize, file_size_ - page_offset);
      if (read_len < kVectorPageSize) {
        std::memset(buffer + read_len, 0, kVectorPageSize - read_len);
      }
      const ssize_t read_bytes = zvec_pread(fd_, buffer, read_len, page_offset);
      if (read_bytes != static_cast<ssize_t>(read_len)) {
        // Accept short read at EOF: last page may not be full kVectorPageSize.
        if (read_bytes > 0 &&
            (page_offset + static_cast<size_t>(read_bytes) >= file_size_)) {
          std::memset(buffer + read_bytes, 0, kVectorPageSize - read_bytes);
        } else {
          LOG_ERROR(
              "Buffer pool failed to read file at offset: file[%s], "
              "page_id[%zu], offset[%zu], expected[%zu], got[%zd]",
              file_name_.c_str(), page_id, page_offset, read_len, read_bytes);
          MemoryLimitPool::get_instance().release_buffer(buffer,
                                                         kVectorPageSize);
          (void)page_table_.cancel_block_load(page_id);
          return nullptr;
        }
      }
    }
    return page_table_.publish_claimed_block(page_id, buffer, page_offset);
  }
}

bool VecBufferPool::try_acquire_resident_pages(const block_id_t *page_ids,
                                               size_t count, char **pages) {
  if (count == 0) return true;
  if (!page_ids || !pages) return false;

  std::fill_n(pages, count, nullptr);
  for (size_t i = 0; i < count; ++i) {
    if (page_ids[i] >= page_table_.entry_num()) {
      release_pages(page_ids, i);
      std::fill_n(pages, i, nullptr);
      return false;
    }
    pages[i] = try_acquire_buffer(page_ids[i]);
    if (!pages[i]) {
      release_pages(page_ids, i);
      std::fill_n(pages, i, nullptr);
      return false;
    }
  }
  return true;
}

bool VecBufferPool::acquire_pages(const block_id_t *page_ids, size_t count,
                                  char **pages) {
  if (count == 0) return true;
  if (!page_ids || !pages) return false;

  std::fill_n(pages, count, nullptr);
  std::array<block_id_t, kBlockingAioBatchSize> miss_batch{};
  size_t miss_count = 0;

  // Pin hits before I/O so eviction cannot reclaim them before delivery.
  for (size_t i = 0; i < count; ++i) {
    if (page_ids[i] >= page_table_.entry_num()) {
      for (size_t j = 0; j < i; ++j) {
        if (pages[j]) {
          page_table_.release_block(page_ids[j]);
          pages[j] = nullptr;
        }
      }
      return false;
    }
    pages[i] = try_acquire_buffer(page_ids[i]);
    if (!pages[i]) {
      miss_batch[miss_count++] = page_ids[i];
      if (miss_count == miss_batch.size()) {
        (void)load_pages_aio(miss_batch.data(), miss_count, kLowPriority);
        miss_count = 0;
      }
    }
  }
  if (miss_count != 0) {
    (void)load_pages_aio(miss_batch.data(), miss_count, kLowPriority);
  }

  // Resolve and pin every output, including duplicates, after population.
  for (size_t i = 0; i < count; ++i) {
    if (pages[i]) continue;
    // Population releases its installation pin before this resolution pass.
    // Acquiring it here completes the original miss; it is not evidence of a
    // later reuse and must leave the page in probation.
    // A batch caller can roll back and use its direct-I/O fallback. Keep only
    // one bounded foreground reclaim attempt here so a capacity miss does not
    // leave the page in kLoadingRefCount while scanning the global queue.
    pages[i] = acquire_buffer(page_ids[i], 1, /*record_reuse=*/false);
    if (!pages[i]) {
      for (size_t j = 0; j < count; ++j) {
        if (pages[j]) {
          page_table_.release_block(page_ids[j]);
          pages[j] = nullptr;
        }
      }
      return false;
    }
  }
  return true;
}

void VecBufferPool::release_pages(const block_id_t *page_ids, size_t count) {
  if (!page_ids) return;
  for (size_t i = 0; i < count; ++i) {
    if (page_ids[i] < page_table_.entry_num()) {
      page_table_.release_block(page_ids[i]);
    }
  }
}

bool VecBufferPool::should_admit_page(block_id_t page_id) {
  if (page_id >= page_table_.entry_num()) {
    return false;
  }
  if (writable_ || !MemoryLimitPool::get_instance().under_cache_pressure()) {
    return true;
  }

  // Move the aging epoch every 64K evaluated cold misses. Exact per-page
  // counters live in existing page-table padding, so this adds no side hash.
  static constexpr uint64_t kObservationsPerEpoch = uint64_t{1} << 16;
  const uint64_t observation =
      admission_observations_.fetch_add(1, std::memory_order_relaxed);
  const uint32_t epoch =
      static_cast<uint32_t>(observation / kObservationsPerEpoch);
  const bool admitted = page_table_.should_admit_miss(page_id, epoch);
  if (admitted) {
    admission_admitted_.fetch_add(1, std::memory_order_relaxed);
  } else {
    admission_rejected_.fetch_add(1, std::memory_order_relaxed);
  }
  return admitted;
}

int VecBufferPool::get_meta(size_t offset, size_t length, char *buffer) {
  if (length == 0) {
    return 0;
  }
  if (buffer == nullptr || offset > file_size_ ||
      length > file_size_ - offset) {
    return -1;
  }
  ssize_t read_bytes = zvec_pread(meta_fd_, buffer, length, offset);
  if (read_bytes != static_cast<ssize_t>(length)) {
    LOG_ERROR(
        "Buffer pool failed to read file at offset: file[%s], offset[%zu], "
        "length[%zu]",
        file_name_.c_str(), offset, length);
    return -1;
  }
  return 0;
}

bool VecBufferPool::read_range_bypass(size_t file_offset, size_t length,
                                      char *buffer) {
  if (length == 0) {
    return true;
  }
  if (buffer == nullptr || file_offset > file_size_ ||
      length > file_size_ - file_offset) {
    return false;
  }

  struct BypassScratch {
    ~BypassScratch() {
      if (page != nullptr) {
        ailego_free(page);
      }
    }
    char *page{nullptr};
  };
  static thread_local BypassScratch scratch;
  if (scratch.page == nullptr) {
    scratch.page = static_cast<char *>(
        ailego_aligned_malloc(kVectorPageSize, kVectorPageSize));
  }
  if (scratch.page == nullptr) {
    return false;
  }
  char *page = scratch.page;

  size_t copied = 0;
  size_t io_requests = 0;
  bool ok = true;
  while (copied < length) {
    const size_t absolute = file_offset + copied;
    const size_t page_offset = (absolute / kVectorPageSize) * kVectorPageSize;
    const size_t within_page = absolute - page_offset;
    const size_t copy_size =
        std::min(length - copied, kVectorPageSize - within_page);
    const size_t available = file_size_ - page_offset;
    const size_t read_size = direct_io_enabled_
                                 ? kVectorPageSize
                                 : std::min(kVectorPageSize, available);

    ++io_requests;
    const ssize_t read_bytes = zvec_pread(fd_, page, read_size, page_offset);
    if (read_bytes <= 0 ||
        within_page + copy_size > static_cast<size_t>(read_bytes)) {
      ok = false;
      break;
    }
    std::memcpy(buffer + copied, page + within_page, copy_size);
    copied += copy_size;
  }
  if (ok) {
    record_bypass_read(length, io_requests);
  }
  return ok;
}

int VecBufferPool::write_range(size_t file_offset, size_t length,
                               const char *src) {
  if (!writable_) {
    LOG_ERROR("write_range called on read-only pool: file[%s]",
              file_name_.c_str());
    return -1;
  }
  if (length == 0) {
    return 0;
  }
  if (src == nullptr || file_offset > file_size_ ||
      length > file_size_ - file_offset) {
    LOG_ERROR(
        "write_range exceeds file bounds: file[%s], offset[%zu], "
        "length[%zu], file_size[%zu]",
        file_name_.c_str(), file_offset, length, file_size_);
    return -1;
  }
  size_t first_page = file_offset / kVectorPageSize;
  size_t last_page = (file_offset + length - 1) / kVectorPageSize;
  size_t remaining = length;
  size_t src_cursor = 0;
  for (size_t pg = first_page; pg <= last_page; ++pg) {
    // Load partial pages before modifying them.
    char *page = this->acquire_buffer(pg, 50);
    if (!page) {
      LOG_ERROR("write_range acquire failed: file[%s], page[%zu]",
                file_name_.c_str(), pg);
      return -1;
    }
    std::unique_lock<std::shared_mutex> page_lock(
        block_mutexes_[pg % block_mutex_count_]);
    size_t page_start = pg * kVectorPageSize;
    size_t intra_offset = (pg == first_page) ? (file_offset - page_start) : 0;
    size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
    std::memcpy(page + intra_offset, src + src_cursor, chunk);
    page_table_.mark_dirty(pg);
    page_table_.release_block(pg);
    src_cursor += chunk;
    remaining -= chunk;
  }
  return 0;
}

int VecBufferPool::write_fragments(const VecBufferWriteFragment *fragments,
                                   size_t count) {
  if (!writable_ || (count != 0 && fragments == nullptr)) {
    return -1;
  }
  if (count == 0) {
    return 0;
  }

  size_t page_id = std::numeric_limits<size_t>::max();
  bool has_data = false;
  for (size_t i = 0; i < count; ++i) {
    const auto &fragment = fragments[i];
    if (fragment.length == 0) {
      continue;
    }
    if (fragment.src == nullptr || fragment.file_offset > file_size_ ||
        fragment.length > file_size_ - fragment.file_offset) {
      return -1;
    }
    const size_t fragment_page = fragment.file_offset / kVectorPageSize;
    const size_t offset_in_page = fragment.file_offset % kVectorPageSize;
    if (fragment.length > kVectorPageSize - offset_in_page ||
        (has_data && fragment_page != page_id)) {
      return -1;
    }
    page_id = fragment_page;
    has_data = true;
  }
  if (!has_data) {
    return 0;
  }

  char *page = acquire_buffer(static_cast<block_id_t>(page_id), 50);
  if (page == nullptr) {
    return -1;
  }
  {
    std::unique_lock<std::shared_mutex> page_lock(
        block_mutexes_[page_id % block_mutex_count_]);
    for (size_t i = 0; i < count; ++i) {
      const auto &fragment = fragments[i];
      if (fragment.length != 0) {
        std::memcpy(page + fragment.file_offset % kVectorPageSize, fragment.src,
                    fragment.length);
      }
    }
    page_table_.mark_dirty(page_id);
  }
  page_table_.release_block(page_id);
  return 0;
}

int VecBufferPool::write_meta(size_t offset, size_t length,
                              const char *buffer) {
  if (!writable_) {
    LOG_ERROR("write_meta called on read-only pool: file[%s]",
              file_name_.c_str());
    return -1;
  }
  if (length == 0) {
    return 0;
  }
  if (buffer == nullptr || offset > file_size_ ||
      length > file_size_ - offset) {
    return -1;
  }
  ssize_t w = zvec_pwrite(meta_fd_, buffer, length, offset);
  if (w != static_cast<ssize_t>(length)) {
    LOG_ERROR(
        "Buffer pool failed to write meta: file[%s], offset[%zu], "
        "length[%zu], got[%zd]",
        file_name_.c_str(), offset, length, w);
    return -1;
  }
  return 0;
}

int VecBufferPool::flush_all() {
  if (!writable_) {
    return 0;
  }
  // Establish one persistence owner before the full scan. This also gives
  // callers a deterministic drain point for all previously queued pages.
  drain_writeback();
  std::unique_lock<std::mutex> flush_lock(writeback_flush_mutex_);
  const size_t total = page_table_.entry_num();
  if (total == 0) {
    return 0;
  }

  int rc = 0;
  size_t total_dirty = 0;
  size_t failed_batches = 0;
  std::vector<block_id_t> dirty_pages;
  dirty_pages.reserve(kBlockingAioBatchSize);
  for (size_t page_id = 0; page_id < total; ++page_id) {
    if (page_table_.is_block_dirty(page_id)) {
      dirty_pages.push_back(page_id);
    }
    if (dirty_pages.size() == kBlockingAioBatchSize ||
        (page_id + 1 == total && !dirty_pages.empty())) {
      total_dirty += dirty_pages.size();
      if (!flush_writeback_batch(dirty_pages, writeback_staging_)) {
        rc = -1;
        ++failed_batches;
      }
      dirty_pages.clear();
    }
  }

  if (failed_batches != 0) {
    LOG_ERROR(
        "VecBufferPool::flush_all: %zu writeback batch(es) covering %zu dirty "
        "page(s) failed, file[%s] last_rc=%d -- on-disk data may be stale.",
        failed_batches, total_dirty, file_name_.c_str(), rc);
  } else {
    writeback_error_.store(0, std::memory_order_release);
  }
  flush_lock.unlock();
  drain_writeback();
  if (writeback_error() != 0) {
    rc = -1;
  }
  return rc;
}

bool VecBufferPool::extend_file(size_t new_size) {
  if (!writable_) {
    LOG_ERROR("extend_file called on read-only pool: file[%s]",
              file_name_.c_str());
    return false;
  }
  if (new_size <= file_size_) {
    return true;
  }
  // O_DIRECT requires page-aligned backing-file growth.
  if (new_size % kVectorPageSize != 0) {
    LOG_ERROR(
        "extend_file target must be page-aligned: file[%s], new_size[%zu], "
        "page_size[%zu]",
        file_name_.c_str(), new_size, kVectorPageSize);
    return false;
  }
  // Validate page-table capacity before changing the file.
  const size_t new_entry_num = (new_size - 1) / kVectorPageSize + 1;
  if (new_entry_num > VectorPageTable::kMaxEntries) {
    LOG_ERROR(
        "extend_file: requested new_size=%zu would require %zu page entries, "
        "exceeding VectorPageTable::kMaxEntries=%zu (file=%s).",
        new_size, new_entry_num, VectorPageTable::kMaxEntries,
        file_name_.c_str());
    return false;
  }
  const size_t old_entry_num = page_table_.entry_num();
  if (new_entry_num > old_entry_num && !page_table_.extend(new_entry_num)) {
    LOG_ERROR(
        "extend_file: page_table_.extend(%zu) failed before resizing "
        "file=%s to %zu bytes",
        new_entry_num, file_name_.c_str(), new_size);
    return false;
  }

#if defined(_MSC_VER)
  if (_chsize_s(fd_, static_cast<int64_t>(new_size)) != 0) {
    LOG_ERROR("extend_file _chsize_s failed: file[%s], new_size[%zu]",
              file_name_.c_str(), new_size);
    if (!page_table_.rollback_extend(old_entry_num)) {
      LOG_ERROR("extend_file: failed to roll back page table for file[%s]",
                file_name_.c_str());
    }
    return false;
  }
#else
  if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
    LOG_ERROR("extend_file ftruncate failed: file[%s], new_size[%zu]",
              file_name_.c_str(), new_size);
    if (!page_table_.rollback_extend(old_entry_num)) {
      LOG_ERROR("extend_file: failed to roll back page table for file[%s]",
                file_name_.c_str());
    }
    return false;
  }
#endif
  file_size_ = new_size;
  return true;
}

char *VecBufferPoolHandle::get_single_page(size_t file_offset, size_t len,
                                           size_t &out_page_id) {
  if (file_offset >= pool_.file_size_ || len > pool_.file_size_ - file_offset) {
    return nullptr;
  }
  size_t first_page = file_offset / kVectorPageSize;
  assert(len == 0 || len <= kVectorPageSize - (file_offset % kVectorPageSize));
  out_page_id = first_page;
  char *page = pool_.acquire_buffer(first_page, 50);
  if (!page) {
    LOG_DEBUG(
        "VecBufferPoolHandle::get_single_page: acquire_buffer failed, "
        "file_offset=%zu, len=%zu, page=%zu, page_size=%zu",
        file_offset, len, first_page, kVectorPageSize);
    return nullptr;
  }
  return page + (file_offset - first_page * kVectorPageSize);
}

bool VecBufferPoolHandle::acquire_pages(const block_id_t *page_ids,
                                        size_t count, char **pages) {
  return pool_.acquire_pages(page_ids, count, pages);
}

bool VecBufferPoolHandle::try_acquire_resident_pages(
    const block_id_t *page_ids, size_t count, char **pages) {
  return pool_.try_acquire_resident_pages(page_ids, count, pages);
}

void VecBufferPoolHandle::release_pages(const block_id_t *page_ids,
                                        size_t count) {
  pool_.release_pages(page_ids, count);
}

bool VecBufferPoolHandle::read_range(size_t file_offset, size_t len,
                                     char *out) {
  if (len == 0) {
    return true;
  }
  if (out == nullptr || file_offset > pool_.file_size_ ||
      len > pool_.file_size_ - file_offset) {
    return false;
  }
  size_t first_page = file_offset / kVectorPageSize;
  size_t last_page = (file_offset + len - 1) / kVectorPageSize;
  size_t remaining = len;
  size_t dst_cursor = 0;

  // Protect payload copies only for writable pools.
  if (pool_.writable_) {
    for (size_t pg = first_page; pg <= last_page; ++pg) {
      char *page = pool_.acquire_buffer(static_cast<block_id_t>(pg), 50);
      if (page == nullptr) {
        return false;
      }
      std::shared_lock<std::shared_mutex> page_lock(
          pool_.block_mutexes_[pg % pool_.block_mutex_count_]);
      const size_t page_start = pg * kVectorPageSize;
      const size_t intra_offset =
          (pg == first_page) ? (file_offset - page_start) : 0;
      const size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
      std::memcpy(out + dst_cursor, page + intra_offset, chunk);
      page_lock.unlock();
      pool_.page_table_.release_block(static_cast<block_id_t>(pg));
      dst_cursor += chunk;
      remaining -= chunk;
    }
    return true;
  }

  static constexpr size_t kMaxRunPages = 1024;  // 4MB max per bulk read

  for (size_t pg = first_page; pg <= last_page; ++pg) {
    char *page = pool_.page_table_.acquire_block(pg);
    if (page) {
      size_t page_start = pg * kVectorPageSize;
      size_t intra_offset = (pg == first_page) ? (file_offset - page_start) : 0;
      size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
      std::memcpy(out + dst_cursor, page + intra_offset, chunk);
      pool_.page_table_.release_block(pg);
      dst_cursor += chunk;
      remaining -= chunk;
      continue;
    }

    size_t run_start = pg;
    size_t run_end = pg + 1;
    while (run_end <= last_page && !pool_.page_table_.is_loaded(run_end) &&
           (run_end - run_start) < kMaxRunPages) {
      ++run_end;
    }
    size_t run_pages = run_end - run_start;

    if (run_pages <= 3) {
      for (size_t j = 0; j < run_pages; ++j) {
        block_id_t pid = static_cast<block_id_t>(run_start + j);
        // Once a sparse resident set breaks a cold range into short holes,
        // apply the same frequency admission policy as the bulk path. Without
        // this check, one- to three-page first touches continuously evict the
        // useful working set even though long first-touch runs are bypassed.
        const bool use_admission =
            MemoryLimitPool::get_instance().page_admission_reserve() != 0;
        const bool admit = !use_admission || pool_.should_admit_page(pid);
        if (admit) {
          page = pool_.acquire_buffer(pid, 50);
        } else {
          pool_.miss_count_.fetch_add(1, std::memory_order_relaxed);
          page = nullptr;
        }
        size_t page_start = pid * kVectorPageSize;
        size_t intra_offset =
            (pid == first_page) ? (file_offset - page_start) : 0;
        size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
        if (page != nullptr) {
          std::memcpy(out + dst_cursor, page + intra_offset, chunk);
          pool_.page_table_.release_block(pid);
        } else if (!pool_.read_range_bypass(page_start + intra_offset, chunk,
                                            out + dst_cursor)) {
          return false;
        }
        dst_cursor += chunk;
        remaining -= chunk;
      }
      pg = run_end - 1;
      continue;
    }

    size_t run_bytes = run_pages * kVectorPageSize;
    size_t run_file_off = run_start * kVectorPageSize;

    char *bulk_buf =
        static_cast<char *>(ailego_aligned_malloc(run_bytes, 4096));
    if (!bulk_buf) {
      page = pool_.acquire_buffer(static_cast<block_id_t>(pg), 50);
      size_t page_start = pg * kVectorPageSize;
      size_t intra_offset = (pg == first_page) ? (file_offset - page_start) : 0;
      size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
      if (page != nullptr) {
        std::memcpy(out + dst_cursor, page + intra_offset, chunk);
        pool_.page_table_.release_block(static_cast<block_id_t>(pg));
      } else if (!pool_.read_range_bypass(page_start + intra_offset, chunk,
                                          out + dst_cursor)) {
        return false;
      }
      dst_cursor += chunk;
      remaining -= chunk;
      continue;
    }

    ssize_t got = zvec_pread(pool_.fd_, bulk_buf, run_bytes, run_file_off);
    // read_range validated file_offset + len against file_size_ above.
    size_t needed_bytes = (file_offset + len) - run_file_off;
    if (needed_bytes > run_bytes) needed_bytes = run_bytes;
    if (got < 0 || static_cast<size_t>(got) < needed_bytes) {
      ailego_free(bulk_buf);
      LOG_ERROR(
          "read_range bulk pread failed: off=%zu len=%zu got=%zd needed=%zu",
          run_file_off, run_bytes, got, needed_bytes);
      return false;
    }
    size_t actually_read = static_cast<size_t>(got);
    // Account for pages populated outside acquire_buffer().
    size_t pages_read = (actually_read + kVectorPageSize - 1) / kVectorPageSize;
    pool_.miss_count_.fetch_add(pages_read, std::memory_order_relaxed);

    for (size_t j = 0; j < run_pages; ++j) {
      block_id_t pid = static_cast<block_id_t>(run_start + j);
      size_t page_start = pid * kVectorPageSize;
      size_t intra_offset =
          (pid == first_page) ? (file_offset - page_start) : 0;
      size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
      std::memcpy(out + dst_cursor,
                  bulk_buf + j * kVectorPageSize + intra_offset, chunk);
      dst_cursor += chunk;
      remaining -= chunk;

      size_t page_end_in_buf = (j + 1) * kVectorPageSize;
      // Large sequential reads (for example IVF posting-list scans) must not
      // populate every cold page once the shared cache is under pressure.
      // Reuse the same compact frequency admission policy as batched random
      // reads so first-touch scan pages bypass while repeated pages can enter.
      if (page_end_in_buf <= actually_read &&
          !pool_.page_table_.is_loaded(pid) && pool_.should_admit_page(pid)) {
        char *page_buf = nullptr;
        bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
            kVectorPageSize, page_buf);
        if (!found) {
          BlockEvictionQueue::get_instance().recycle();
          found = MemoryLimitPool::get_instance().try_acquire_buffer(
              kVectorPageSize, page_buf);
        }
        if (found) {
          std::memcpy(page_buf, bulk_buf + j * kVectorPageSize,
                      kVectorPageSize);
          char *installed = pool_.page_table_.set_block_acquired(
              pid, page_buf, run_file_off + j * kVectorPageSize);
          if (installed != nullptr) {
            pool_.page_table_.release_block(pid);
          }
        }
      }
    }
    ailego_free(bulk_buf);
    pg = run_end - 1;
  }
  return true;
}

bool VecBufferPoolHandle::read_range_immutable(size_t file_offset, size_t len,
                                               char *out) {
  if (len == 0) {
    return true;
  }
  if (out == nullptr || file_offset > pool_.file_size_ ||
      len > pool_.file_size_ - file_offset) {
    return false;
  }

  const size_t first_page = file_offset / kVectorPageSize;
  const size_t last_page = (file_offset + len - 1) / kVectorPageSize;
  size_t remaining = len;
  size_t dst_cursor = 0;
  for (size_t pg = first_page; pg <= last_page; ++pg) {
    char *page = pool_.acquire_buffer(static_cast<block_id_t>(pg), 50);
    if (page == nullptr) {
      return false;
    }
    const size_t page_start = pg * kVectorPageSize;
    const size_t intra_offset = pg == first_page ? file_offset - page_start : 0;
    const size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
    std::memcpy(out + dst_cursor, page + intra_offset, chunk);
    pool_.page_table_.release_block(static_cast<block_id_t>(pg));
    dst_cursor += chunk;
    remaining -= chunk;
  }
  return true;
}

bool VecBufferPoolHandle::read_range_bypass(size_t file_offset, size_t len,
                                            char *out) {
  return pool_.read_range_bypass(file_offset, len, out);
}

int VecBufferPoolHandle::get_meta(size_t offset, size_t length, char *buffer) {
  return pool_.get_meta(offset, length, buffer);
}

int VecBufferPoolHandle::write_range(size_t file_offset, size_t len,
                                     const char *src) {
  return pool_.write_range(file_offset, len, src);
}

int VecBufferPoolHandle::write_fragments(
    const VecBufferWriteFragment *fragments, size_t count) {
  return pool_.write_fragments(fragments, count);
}

int VecBufferPoolHandle::write_meta(size_t offset, size_t length,
                                    const char *buffer) {
  return pool_.write_meta(offset, length, buffer);
}

int VecBufferPoolHandle::flush_all() {
  return pool_.flush_all();
}

bool VecBufferPoolHandle::writable() const {
  return pool_.writable();
}

void VecBufferPoolHandle::release_one(block_id_t block_id) {
  pool_.page_table_.release_block(block_id);
}

void VecBufferPoolHandle::acquire_one(block_id_t block_id) {
  // Caller guarantees the page is resident.
  pool_.page_table_.acquire_block(block_id);
}

void VecBufferPool::warmup() {
  const size_t total_pages = page_table_.entry_num();
  // Read sequentially in 4 MB chunks.
  static constexpr size_t kChunkPages = 1024;
  const size_t kChunkSize = kChunkPages * kVectorPageSize;

  // Aligned buffer for bulk read (O_DIRECT requires alignment).
  char *chunk_buf =
      static_cast<char *>(ailego_aligned_malloc(kChunkSize, 4096));
  if (!chunk_buf) return;

  size_t loaded = 0;
  bool pool_full = false;
  for (size_t base = 0; base < total_pages && !pool_full; base += kChunkPages) {
    const size_t pages_in_chunk = std::min(kChunkPages, total_pages - base);
    const size_t read_bytes = pages_in_chunk * kVectorPageSize;
    const size_t file_offset = base * kVectorPageSize;
    const size_t expected_bytes =
        std::min(read_bytes, file_size_ - file_offset);

    // One large sequential pread instead of N individual ones.
    ssize_t got = zvec_pread(fd_, chunk_buf, read_bytes, file_offset);
    if (got != static_cast<ssize_t>(expected_bytes)) break;
    // The final page may extend past EOF. Keep its unread tail deterministic,
    // matching the regular single-page load path.
    if (expected_bytes < read_bytes) {
      std::memset(chunk_buf + expected_bytes, 0, read_bytes - expected_bytes);
    }

    // Distribute chunk data into individual page buffers.
    for (size_t j = 0; j < pages_in_chunk; ++j) {
      auto page_id = static_cast<block_id_t>(base + j);
      // Skip if already loaded.
      char *existing = page_table_.acquire_block(page_id);
      if (existing) {
        page_table_.release_block(page_id);
        ++loaded;
        continue;
      }
      // Allocate page buffer from pool (no retry - stop if full).
      char *buf = nullptr;
      bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
          kVectorPageSize, buf);
      if (!found) {
        pool_full = true;
        break;
      }
      std::memcpy(buf, chunk_buf + j * kVectorPageSize, kVectorPageSize);
      char *installed = page_table_.set_block_acquired(
          page_id, buf, file_offset + j * kVectorPageSize);
      if (installed != nullptr) {
        page_table_.release_block(page_id);
        ++loaded;
      }
    }
  }
  ailego_free(chunk_buf);
  LOG_DEBUG("VecBufferPool::warmup: preloaded %zu/%zu pages for file[%s]",
            loaded, total_pages, file_name_.c_str());
}

void VecBufferPool::prefetch_pages(block_id_t first_page, size_t page_count,
                                   uint8_t priority) {
  const size_t total_pages = page_table_.entry_num();
  if (priority > kHighPriority || page_count == 0 ||
      first_page >= total_pages) {
    return;
  }
  page_count = std::min(page_count, total_pages - first_page);

  bool all_loaded = true;
  for (size_t page = first_page; page < first_page + page_count; ++page) {
    if (page_table_.is_loaded(page)) {
      page_table_.promote_evict_priority(page, priority);
    } else {
      all_loaded = false;
    }
  }
  if (all_loaded) {
    return;
  }

#if defined(__linux__)
  if (aio_enabled_) {
    prefetch_pages_aio(first_page, page_count, priority);
    return;
  }
#endif

  prefetch_pages_sync(first_page, page_count, priority);
}

void VecBufferPool::prefetch_pages_sync(block_id_t first_page,
                                        size_t page_count, uint8_t priority) {
  const size_t end_page = first_page + page_count;

  static constexpr size_t kChunkPages = 1024;
  const size_t kChunkSize = kChunkPages * kVectorPageSize;
  char *chunk_buf =
      static_cast<char *>(ailego_aligned_malloc(kChunkSize, 4096));
  if (!chunk_buf) return;

  bool pool_full = false;
  size_t pg = first_page;
  while (pg < end_page && !pool_full) {
    if (page_table_.is_loaded(pg)) {
      page_table_.promote_evict_priority(pg, priority);
      ++pg;
      continue;
    }
    size_t run_start = pg;
    size_t run_end = pg + 1;
    while (run_end < end_page && !page_table_.is_loaded(run_end) &&
           (run_end - run_start) < kChunkPages) {
      ++run_end;
    }

    size_t run_pages = run_end - run_start;
    size_t read_bytes = run_pages * kVectorPageSize;
    size_t file_off = run_start * kVectorPageSize;
    size_t expected_bytes = std::min(read_bytes, file_size_ - file_off);
    ssize_t got = zvec_pread(fd_, chunk_buf, read_bytes, file_off);
    if (got != static_cast<ssize_t>(expected_bytes)) {
      pg = run_end;
      continue;
    }
    if (expected_bytes < read_bytes) {
      std::memset(chunk_buf + expected_bytes, 0, read_bytes - expected_bytes);
    }

    for (size_t j = 0; j < run_pages; ++j) {
      block_id_t pid = static_cast<block_id_t>(run_start + j);
      if (page_table_.is_loaded(pid)) {
        page_table_.promote_evict_priority(pid, priority);
        continue;
      }
      char *buf = nullptr;
      bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
          kVectorPageSize, buf);
      if (!found) {
        BlockEvictionQueue::get_instance().recycle();
        found = MemoryLimitPool::get_instance().try_acquire_buffer(
            kVectorPageSize, buf);
        if (!found) {
          pool_full = true;
          break;
        }
      }
      std::memcpy(buf, chunk_buf + j * kVectorPageSize, kVectorPageSize);
      page_table_.promote_evict_priority(pid, priority);
      char *installed = page_table_.set_block_acquired(
          pid, buf, file_off + j * kVectorPageSize);
      if (installed != nullptr) {
        page_table_.release_block(pid);
      }
    }
    pg = run_end;
  }
  ailego_free(chunk_buf);
}

void VecBufferPoolHandle::prefetch_range(size_t file_offset, size_t len,
                                         uint8_t priority) {
  if (len == 0 || file_offset >= pool_.file_size_) return;
  len = std::min(len, pool_.file_size_ - file_offset);
  size_t first_page = file_offset / kVectorPageSize;
  size_t last_page = (file_offset + len - 1) / kVectorPageSize;
  pool_.prefetch_pages(static_cast<block_id_t>(first_page),
                       last_page - first_page + 1, priority);
}

#if defined(__linux__)
namespace {
template <unsigned QueueDepth>
struct ThreadLocalIoUringContext {
  IoUringRing ring{};
  bool inited{false};

  bool ensure() {
    if (!inited) {
      inited = true;
      ring.setup(QueueDepth);
    }
    return ring.is_valid();
  }
};

template <unsigned QueueDepth>
struct ThreadLocalAioContext {
  io_context_t ctx{nullptr};
  bool inited{false};

  bool ensure() {
    if (inited) return ctx != nullptr;
    inited = true;
    if (!LibAioLoader::Instance().load() ||
        !LibAioLoader::Instance().is_available()) {
      return false;
    }
    if (LibAioLoader::Instance().io_setup(QueueDepth, &ctx) == 0) {
      return true;
    }
    ctx = nullptr;
    return false;
  }

  bool destroy_context(const char *context_name) {
    if (!ctx) return true;
    const int ret = LibAioLoader::Instance().io_destroy(ctx);
    if (ret != 0) {
      LOG_ERROR(
          "%s: io_destroy failed, ret=%d; in-flight buffers remain "
          "quarantined",
          context_name, ret);
      return false;
    }
    ctx = nullptr;
    return true;
  }
};

// One blocking AIO context per calling thread, shared by reads and prefetches.
struct ThreadLocalBlockingAioCtx
    : ThreadLocalAioContext<kBlockingAioBatchSize> {
  char *quarantined[kBlockingAioBatchSize]{};
  size_t quarantined_count{0};

  void quarantine(char **buffers, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      if (buffers[i]) {
        assert(quarantined_count < kBlockingAioBatchSize);
        quarantined[quarantined_count++] = buffers[i];
        buffers[i] = nullptr;
      }
    }
  }

  void release_quarantined() {
    for (size_t i = 0; i < quarantined_count; ++i) {
      MemoryLimitPool::get_instance().release_buffer(quarantined[i],
                                                     kVectorPageSize);
    }
    quarantined_count = 0;
  }

  ~ThreadLocalBlockingAioCtx() {
    if (destroy_context("ThreadLocalBlockingAioCtx")) {
      release_quarantined();
    }
  }
};
static thread_local ThreadLocalIoUringContext<kBlockingAioBatchSize>
    tl_blocking_io_uring;
static thread_local ThreadLocalBlockingAioCtx tl_blocking_aio;
}  // namespace
#endif

void VecBufferPool::prefetch_pages_aio(block_id_t first_page, size_t page_count,
                                       uint8_t priority) {
  const size_t total_pages = page_table_.entry_num();
  if (priority > kHighPriority || page_count == 0 ||
      first_page >= total_pages) {
    return;
  }
  page_count = std::min(page_count, total_pages - first_page);
  std::array<block_id_t, kBlockingAioBatchSize> pages{};
  size_t offset = 0;
  while (offset < page_count) {
    const size_t batch = std::min(kBlockingAioBatchSize, page_count - offset);
    for (size_t i = 0; i < batch; ++i) {
      pages[i] = first_page + offset + i;
    }
    if (!load_pages_aio(pages.data(), batch, priority)) {
      prefetch_pages_sync(first_page, page_count, priority);
      return;
    }
    offset += batch;
  }
}

bool VecBufferPool::load_pages_aio(const block_id_t *page_ids, size_t count,
                                   uint8_t priority) {
  if (count == 0) return true;
  if (page_ids == nullptr || priority > kHighPriority) return false;

#if defined(__linux__)
  if (!aio_enabled_) return false;
  bool use_io_uring = io_backend_type_ == IOBackendType::kIoUring &&
                      tl_blocking_io_uring.ensure();
  if (!use_io_uring && !tl_blocking_aio.ensure()) return false;

  size_t cursor = 0;
  while (cursor < count) {
    std::array<block_id_t, kBlockingAioBatchSize> candidate_pages{};
    size_t candidate_count = 0;
    while (cursor < count && candidate_count < kBlockingAioBatchSize) {
      const block_id_t pid = page_ids[cursor++];
      if (pid >= page_table_.entry_num()) return false;
      bool duplicate = false;
      for (size_t i = 0; i < candidate_count; ++i) {
        if (candidate_pages[i] == pid) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) candidate_pages[candidate_count++] = pid;
    }

    std::array<block_id_t, kBlockingAioBatchSize> load_pages{};
    size_t load_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
      const block_id_t pid = candidate_pages[i];
      const auto claim = page_table_.try_claim_block_load(pid);
      switch (claim) {
        case VectorPageTable::LoadClaimResult::kClaimed:
          load_pages[load_count++] = pid;
          break;
        case VectorPageTable::LoadClaimResult::kResident:
          page_table_.promote_evict_priority(pid, priority);
          break;
        case VectorPageTable::LoadClaimResult::kLoading:
          page_table_.promote_evict_priority(pid, priority);
          singleflight_waits_.fetch_add(1, std::memory_order_relaxed);
          break;
        case VectorPageTable::LoadClaimResult::kEvicting:
          page_table_.promote_evict_priority(pid, priority);
          break;
      }
    }
    if (load_count == 0) continue;

    std::array<char *, kBlockingAioBatchSize> buffers{};
    size_t allocated = MemoryLimitPool::get_instance().batch_acquire_buffers(
        kVectorPageSize, buffers.data(), load_count);

    // Admit into unused capacity first. Reclaim only the actual shortage so a
    // miss batch cannot evict an equal number of resident pages while the
    // shared pool still has room to grow.
    while (allocated < load_count) {
      const size_t shortage = load_count - allocated;
      if (BlockEvictionQueue::get_instance().batch_recycle(shortage) == 0) {
        break;
      }
      const size_t acquired =
          MemoryLimitPool::get_instance().batch_acquire_buffers(
              kVectorPageSize, buffers.data() + allocated, shortage);
      allocated += acquired;
      if (acquired == 0) {
        break;
      }
    }
    bool batch_ok = allocated == load_count;
    for (size_t i = allocated; i < load_count; ++i) {
      if (!page_table_.cancel_block_load(load_pages[i])) {
        LOG_ERROR(
            "VecBufferPool::load_pages_aio: failed to cancel unallocated "
            "load claim for page=%zu",
            static_cast<size_t>(load_pages[i]));
      }
    }
    if (allocated == 0) return false;

    auto abandon_claims = [&](size_t abandon_count, bool release_buffers) {
      for (size_t i = 0; i < abandon_count; ++i) {
        if (buffers[i]) {
          if (release_buffers) {
            MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                           kVectorPageSize);
          }
          buffers[i] = nullptr;
        }
        if (!page_table_.cancel_block_load(load_pages[i])) {
          LOG_ERROR(
              "VecBufferPool::load_pages_aio: failed to abandon load claim "
              "for page=%zu",
              static_cast<size_t>(load_pages[i]));
        }
      }
    };

    std::array<bool, kBlockingAioBatchSize> read_ok{};
    if (use_io_uring) {
      std::array<IoUringRead, kBlockingAioBatchSize> requests{};
      for (size_t i = 0; i < allocated; ++i) {
        const size_t offset = load_pages[i] * kVectorPageSize;
        const size_t expected = std::min(kVectorPageSize, file_size_ - offset);
        requests[i] =
            IoUringRead(offset, kVectorPageSize, buffers[i], expected);
      }
      aio_pages_submitted_.fetch_add(allocated, std::memory_order_relaxed);
      if (tl_blocking_io_uring.ring.execute(fd_, requests.data(), allocated) ==
          0) {
        std::fill_n(read_ok.begin(), allocated, true);
      } else if (tl_blocking_aio.ensure()) {
        // io_uring uses ring-owned staging, so libaio can safely reuse the
        // destination buffers after execute() reports a drained failure.
        use_io_uring = false;
      } else {
        abandon_claims(allocated, /*release_buffers=*/true);
        return false;
      }
    }

    if (!use_io_uring) {
      std::array<struct iocb, kBlockingAioBatchSize> cbs{};
      std::array<struct iocb *, kBlockingAioBatchSize> cb_ptrs{};
      for (size_t i = 0; i < allocated; ++i) {
        const size_t offset = load_pages[i] * kVectorPageSize;
        io_prep_pread(&cbs[i], fd_, buffers[i], kVectorPageSize,
                      static_cast<long long>(offset));
        cbs[i].data = reinterpret_cast<void *>(i);
        cb_ptrs[i] = &cbs[i];
      }

      const int submit_ret = LibAioLoader::Instance().io_submit(
          tl_blocking_aio.ctx, static_cast<long>(allocated), cb_ptrs.data());
      if (submit_ret <= 0 || static_cast<size_t>(submit_ret) > allocated) {
        abandon_claims(allocated, /*release_buffers=*/true);
        return false;
      }

      const size_t accepted = static_cast<size_t>(submit_ret);
      aio_pages_submitted_.fetch_add(accepted, std::memory_order_relaxed);
      batch_ok = batch_ok && accepted == allocated;
      for (size_t i = accepted; i < allocated; ++i) {
        MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                       kVectorPageSize);
        buffers[i] = nullptr;
        if (!page_table_.cancel_block_load(load_pages[i])) {
          LOG_ERROR(
              "VecBufferPool::load_pages_aio: failed to cancel unsubmitted "
              "load claim for page=%zu",
              static_cast<size_t>(load_pages[i]));
        }
      }

      std::array<struct io_event, kBlockingAioBatchSize> events{};
      size_t completed = 0;
      bool wait_failed = false;
      while (completed < accepted) {
        const int get_ret = LibAioLoader::Instance().io_getevents(
            tl_blocking_aio.ctx, static_cast<long>(accepted - completed),
            static_cast<long>(accepted - completed), events.data() + completed,
            nullptr);
        if (get_ret == -EINTR) continue;
        if (get_ret <= 0) {
          LOG_ERROR(
              "VecBufferPool::load_pages_aio: io_getevents failed, ret=%d",
              get_ret);
          wait_failed = true;
          break;
        }
        completed += static_cast<size_t>(get_ret);
      }

      if (wait_failed) {
        if (tl_blocking_aio.destroy_context("ThreadLocalBlockingAioCtx")) {
          abandon_claims(accepted, /*release_buffers=*/true);
        } else {
          tl_blocking_aio.quarantine(buffers.data(), accepted);
          abandon_claims(accepted, /*release_buffers=*/false);
        }
        return false;
      }

      std::array<bool, kBlockingAioBatchSize> seen{};
      for (size_t i = 0; i < completed; ++i) {
        const size_t idx = reinterpret_cast<size_t>(events[i].data);
        if (idx >= accepted || seen[idx] || buffers[idx] == nullptr) {
          batch_ok = false;
          continue;
        }
        seen[idx] = true;
        const size_t offset = load_pages[idx] * kVectorPageSize;
        const size_t expected = std::min(kVectorPageSize, file_size_ - offset);
        read_ok[idx] = static_cast<ssize_t>(events[i].res) ==
                           static_cast<ssize_t>(expected) &&
                       events[i].res2 == 0;
        if (!read_ok[idx]) {
          batch_ok = false;
          continue;
        }
        if (expected < kVectorPageSize) {
          std::memset(buffers[idx] + expected, 0, kVectorPageSize - expected);
        }
      }
    }

    for (size_t i = 0; i < allocated; ++i) {
      if (buffers[i] == nullptr) continue;
      if (!read_ok[i]) {
        MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                       kVectorPageSize);
        buffers[i] = nullptr;
        if (!page_table_.cancel_block_load(load_pages[i])) {
          LOG_ERROR(
              "VecBufferPool::load_pages_aio: failed to cancel failed load "
              "claim for page=%zu",
              static_cast<size_t>(load_pages[i]));
        }
        batch_ok = false;
        continue;
      }
      const block_id_t pid = load_pages[i];
      miss_count_.fetch_add(1, std::memory_order_relaxed);
      page_table_.promote_evict_priority(pid, priority);
      char *installed = page_table_.publish_claimed_block(
          pid, buffers[i], pid * kVectorPageSize);
      if (installed != nullptr) {
        page_table_.release_block(pid);
      } else {
        batch_ok = false;
      }
      buffers[i] = nullptr;
    }
    if (!batch_ok) return false;
  }
  return true;
#else
  (void)page_ids;
  (void)count;
  (void)priority;
  return false;
#endif
}

void VecBufferPool::log_stats() const {
  Stats s = stats();
  const auto resident = page_table_.resident_pages_by_priority();
  const auto queue_stats = BlockEvictionQueue::get_instance().stats();
  LOG_INFO(
      "VecBufferPool stats: file[%s] hit=%llu miss=%llu hit_rate=%.4f "
      "evict=%llu second_chance=%llu dirty_flush=%llu "
      "writeback_requests=%llu writeback_batches=%llu "
      "writeback_pages=%llu writeback_failures=%llu "
      "writeback_aio_batches=%llu writeback_aio_pages=%llu "
      "writeback_aio_fallbacks=%llu "
      "writeback_waits=%llu writeback_wait_us=%llu "
      "writeback_pending=%llu writeback_peak_pending=%llu "
      "bypass_reads=%llu "
      "bypass_bytes=%llu bypass_io_requests=%llu bypass_rechecks=%llu "
      "bypass_cache_joins=%llu singleflight_waits=%llu "
      "aio_pages_submitted=%llu "
      "admission_admitted=%llu admission_rejected=%llu "
      "ghost_hot_marks=%llu ghost_hot_hits=%llu "
      "page_table_metadata_bytes=%zu page_lock_metadata_bytes=%zu "
      "writeback_staging_bytes=%zu writeback_io_staging_bytes=%zu "
      "resident_by_priority=[%zu,%zu,%zu] "
      "promotions=[%llu,%llu,%llu] demotions=[%llu,%llu,%llu] "
      "evictions_by_priority=[%llu,%llu,%llu] "
      "global_queue_approx=[%zu,%zu,%zu] protected_aging_dequeues=%llu",
      file_name_.c_str(), static_cast<unsigned long long>(s.hit),
      static_cast<unsigned long long>(s.miss), s.hit_rate(),
      static_cast<unsigned long long>(s.evict),
      static_cast<unsigned long long>(s.second_chance),
      static_cast<unsigned long long>(s.dirty_flush),
      static_cast<unsigned long long>(s.writeback_requests),
      static_cast<unsigned long long>(s.writeback_batches),
      static_cast<unsigned long long>(s.writeback_pages),
      static_cast<unsigned long long>(s.writeback_failures),
      static_cast<unsigned long long>(s.writeback_aio_batches),
      static_cast<unsigned long long>(s.writeback_aio_pages),
      static_cast<unsigned long long>(s.writeback_aio_fallbacks),
      static_cast<unsigned long long>(s.writeback_waits),
      static_cast<unsigned long long>(s.writeback_wait_us),
      static_cast<unsigned long long>(s.writeback_pending),
      static_cast<unsigned long long>(s.writeback_peak_pending),
      static_cast<unsigned long long>(s.bypass_reads),
      static_cast<unsigned long long>(s.bypass_bytes),
      static_cast<unsigned long long>(s.bypass_io_requests),
      static_cast<unsigned long long>(s.bypass_rechecks),
      static_cast<unsigned long long>(s.bypass_cache_joins),
      static_cast<unsigned long long>(s.singleflight_waits),
      static_cast<unsigned long long>(s.aio_pages_submitted),
      static_cast<unsigned long long>(s.admission_admitted),
      static_cast<unsigned long long>(s.admission_rejected),
      static_cast<unsigned long long>(s.ghost_hot_marks),
      static_cast<unsigned long long>(s.ghost_hot_hits),
      s.page_table_metadata_bytes, s.page_lock_metadata_bytes,
      s.writeback_staging_bytes, s.writeback_io_staging_bytes, resident[0],
      resident[1], resident[2],
      static_cast<unsigned long long>(s.priority_promotions[0]),
      static_cast<unsigned long long>(s.priority_promotions[1]),
      static_cast<unsigned long long>(s.priority_promotions[2]),
      static_cast<unsigned long long>(s.priority_demotions[0]),
      static_cast<unsigned long long>(s.priority_demotions[1]),
      static_cast<unsigned long long>(s.priority_demotions[2]),
      static_cast<unsigned long long>(s.evictions_by_priority[0]),
      static_cast<unsigned long long>(s.evictions_by_priority[1]),
      static_cast<unsigned long long>(s.evictions_by_priority[2]),
      queue_stats.approximate_queue_sizes[0],
      queue_stats.approximate_queue_sizes[1],
      queue_stats.approximate_queue_sizes[2],
      static_cast<unsigned long long>(queue_stats.protected_aging_dequeues));
}

}  // namespace ailego
}  // namespace zvec
