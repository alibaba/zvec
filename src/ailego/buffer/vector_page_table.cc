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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>

#if defined(__linux) || defined(__linux__)
#include <ailego/io/libaio_loader.h>
#endif

#if defined(_MSC_VER)
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

VecBufferPool::~VecBufferPool() {
  // A caller may have used the non-blocking submit API directly. Drain the
  // current thread's batch before page buffers or file descriptors can be
  // reclaimed by teardown.
  wait_aio();
  // Emit a one-line cache summary (hit rate / evictions) before teardown so
  // operators can reason about buffer-pool efficiency per file.
  log_stats();
  // Flush any remaining dirty blocks before tearing down memory/fd so that
  // writes are not silently lost. Safe to call even in read-only mode.
  (void)this->flush_all();
  for (size_t i = 0; i < page_table_.entry_num(); ++i) {
    assert(page_table_.is_released(i));
    page_table_.force_evict_block(i);
  }
  read_epoch_domain_.drain();
  block_mutexes_.reset();
  MemoryLimitPool::get_instance().release_metadata(mutex_metadata_charge_);
  mutex_metadata_charge_ = 0;
  block_mutex_count_ = 0;
  initialized_ = false;
#if defined(_MSC_VER)
  _close(fd_);
  _close(meta_fd_);
#else
  close(fd_);
  close(meta_fd_);
#endif
}

namespace {
thread_local BufferPoolIoProfileBinding tl_io_profile_binding;
}  // namespace

BufferPoolIoProfileBinding VecBufferPool::bind_thread_io_profile(
    BufferPoolIoProfile *profile) {
  BufferPoolIoProfileBinding previous = tl_io_profile_binding;
  tl_io_profile_binding = BufferPoolIoProfileBinding{this, profile};
  return previous;
}

void VecBufferPool::restore_thread_io_profile(
    const BufferPoolIoProfileBinding &binding) {
  tl_io_profile_binding = binding;
}

BufferPoolIoProfile *VecBufferPool::current_thread_io_profile() const {
  return tl_io_profile_binding.pool == this ? tl_io_profile_binding.profile
                                            : nullptr;
}

version_t VectorPageTable::next_owner_version() {
  static std::atomic<version_t> sequence{1};
  version_t version = sequence.fetch_add(1, std::memory_order_relaxed);
  // Version zero is reserved for legacy/non-versioned owners.  Wraparound is
  // practically unreachable, but skipping zero keeps the invariant explicit.
  if (ailego_unlikely(version == 0)) {
    version = sequence.fetch_add(1, std::memory_order_relaxed);
  }
  return version;
}

bool PageReadEpochDomain::enter(Token *token) {
  if (!token) return false;
  token->slot = kSlotCount;
  for (size_t attempt = 0; attempt < kSlotCount; ++attempt) {
    const size_t idx =
        next_slot_.fetch_add(1, std::memory_order_relaxed) % kSlotCount;
    uint64_t expected = 0;
    if (!slots_[idx].epoch.compare_exchange_strong(expected, kReserved,
                                                   std::memory_order_seq_cst,
                                                   std::memory_order_relaxed)) {
      continue;
    }
    active_readers_.fetch_add(1, std::memory_order_acq_rel);

    // Publish a generation and validate it before the caller can load a page
    // pointer.  The seq_cst order closes the race with retire(): either the
    // retire scan observes this slot, or this reader observes the advanced
    // epoch and republishes before touching the page table.
    while (true) {
      const uint64_t epoch = epoch_.load(std::memory_order_seq_cst);
      slots_[idx].epoch.store(epoch + 2, std::memory_order_seq_cst);
      if (epoch_.load(std::memory_order_seq_cst) == epoch) {
        token->slot = idx;
        return true;
      }
      slots_[idx].epoch.store(kReserved, std::memory_order_seq_cst);
    }
  }
  return false;
}

void PageReadEpochDomain::exit(Token *token) {
  if (!token || !token->valid()) return;
  slots_[token->slot].epoch.store(0, std::memory_order_seq_cst);
  token->slot = kSlotCount;
  active_readers_.fetch_sub(1, std::memory_order_acq_rel);
  std::lock_guard<std::mutex> lock(retired_mutex_);
  reclaim_locked(false);
}

void PageReadEpochDomain::retire(char *buffer) {
  if (!buffer) return;
  if (active_readers_.load(std::memory_order_acquire) == 0) {
    MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
    return;
  }
  const uint64_t retired_epoch = epoch_.fetch_add(1, std::memory_order_seq_cst);
  std::lock_guard<std::mutex> lock(retired_mutex_);
  retired_.push_back(RetiredBuffer{buffer, retired_epoch});
  reclaim_locked(false);
}

void PageReadEpochDomain::drain() {
#ifndef NDEBUG
  for (const auto &slot : slots_) {
    assert(slot.epoch.load(std::memory_order_relaxed) == 0);
  }
#endif
  std::lock_guard<std::mutex> lock(retired_mutex_);
  reclaim_locked(true);
}

void PageReadEpochDomain::reclaim_locked(bool force) {
  size_t dst = 0;
  for (size_t i = 0; i < retired_.size(); ++i) {
    const RetiredBuffer retired = retired_[i];
    bool safe = force;
    if (!safe) {
      safe = true;
      for (const auto &slot : slots_) {
        const uint64_t state = slot.epoch.load(std::memory_order_seq_cst);
        if (state == kReserved || (state >= 2 && state - 2 <= retired.epoch)) {
          safe = false;
          break;
        }
      }
    }
    if (safe) {
      MemoryLimitPool::get_instance().release_buffer(retired.buffer,
                                                     kVectorPageSize);
    } else {
      retired_[dst++] = retired;
    }
  }
  retired_.resize(dst);
}

size_t VectorPageTable::metadata_bytes_for_entries(size_t entry_num) {
  if (entry_num > kMaxEntries) {
    return std::numeric_limits<size_t>::max();
  }
  const size_t segment_count =
      entry_num == 0 ? 0 : (entry_num - 1) / kSegmentSize + 1;
  return segment_count * kSegmentSize * (sizeof(Entry) + sizeof(ResidentEntry));
}

void VectorPageTable::initialize_segment(Entry *entries,
                                         ResidentEntry *resident_entries) {
  for (size_t i = 0; i < kSegmentSize; ++i) {
    entries[i].ref_count.store(std::numeric_limits<int>::min(),
                               std::memory_order_relaxed);
    entries[i].in_evict_queue.store(false, std::memory_order_relaxed);
    entries[i].is_dirty.store(false, std::memory_order_relaxed);
    entries[i].referenced.store(false, std::memory_order_relaxed);
    entries[i].evict_priority.store(0, std::memory_order_relaxed);
    entries[i].ever_loaded = false;
    entries[i].file_offset = 0;
    resident_entries[i].buffer.store(nullptr, std::memory_order_relaxed);
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
  if (!MemoryLimitPool::get_instance().try_charge_metadata(charge)) {
    LOG_ERROR(
        "VectorPageTable::init: shared memory budget cannot reserve %zu "
        "metadata bytes for %zu entries",
        charge, entry_num);
    return false;
  }

  std::vector<std::unique_ptr<Entry[]>> new_segments;
  std::vector<std::unique_ptr<ResidentEntry[]>> new_resident_segments;
  try {
    new_segments.reserve(need_segments);
    new_resident_segments.reserve(need_segments);
    for (size_t s = 0; s < need_segments; ++s) {
      auto entries = std::make_unique<Entry[]>(kSegmentSize);
      auto resident_entries = std::make_unique<ResidentEntry[]>(kSegmentSize);
      initialize_segment(entries.get(), resident_entries.get());
      new_segments.push_back(std::move(entries));
      new_resident_segments.push_back(std::move(resident_entries));
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
    segments_[s] = new_segments[s].release();
    resident_segments_[s] = new_resident_segments[s].release();
  }
  metadata_charge_ = charge;
  // Publish new segments to readers.  segment_count_ is published first
  // (release) so that a reader that acquire-loads segment_count_ before
  // entry_num_ also sees a consistent segment table; entry_num_ is the
  // primary synchronization point used by callers via entry_num().
  segment_count_.store(need_segments, std::memory_order_release);
  entry_num_.store(entry_num, std::memory_order_release);
  return true;
}

bool VectorPageTable::extend(size_t new_entry_num) {
  // Relaxed read is fine: extend() is serialized by the caller (extend_file
  // is invoked under the BufferStorage write latch).  No other writer races
  // with us on entry_num_ / segment_count_.
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
  const size_t added_charge = (new_segment_count - old_count) * kSegmentSize *
                              (sizeof(Entry) + sizeof(ResidentEntry));
  if (!MemoryLimitPool::get_instance().try_charge_metadata(added_charge)) {
    LOG_ERROR(
        "VectorPageTable::extend: shared memory budget cannot reserve %zu "
        "additional metadata bytes (old_entries=%zu, new_entries=%zu)",
        added_charge, entry_num_.load(std::memory_order_relaxed),
        new_entry_num);
    return false;
  }

  std::vector<std::unique_ptr<Entry[]>> new_segments;
  std::vector<std::unique_ptr<ResidentEntry[]>> new_resident_segments;
  try {
    new_segments.reserve(new_segment_count - old_count);
    new_resident_segments.reserve(new_segment_count - old_count);
    for (size_t s = old_count; s < new_segment_count; ++s) {
      auto entries = std::make_unique<Entry[]>(kSegmentSize);
      auto resident_entries = std::make_unique<ResidentEntry[]>(kSegmentSize);
      initialize_segment(entries.get(), resident_entries.get());
      new_segments.push_back(std::move(entries));
      new_resident_segments.push_back(std::move(resident_entries));
    }
  } catch (const std::bad_alloc &) {
    MemoryLimitPool::get_instance().release_metadata(added_charge);
    LOG_ERROR(
        "VectorPageTable::extend: allocation failed for %zu new entries "
        "(%zu additional metadata bytes)",
        new_entry_num, added_charge);
    return false;
  }
  for (size_t s = old_count; s < new_segment_count; ++s) {
    const size_t idx = s - old_count;
    segments_[s] = new_segments[idx].release();
    resident_segments_[s] = new_resident_segments[idx].release();
  }
  metadata_charge_ += added_charge;
  // Publish in the same order as init(): segment_count_ first, entry_num_
  // last.  Both are release-stores so that the prior segment allocation /
  // Entry initialization is visible to any reader that acquire-loads either
  // counter (typically via entry_num()).
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
    if (resident_entry_at(i).buffer.load(std::memory_order_relaxed) !=
            nullptr ||
        entry_at(i).ref_count.load(std::memory_order_relaxed) !=
            std::numeric_limits<int>::min()) {
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
    delete[] resident_segments_[s];
    resident_segments_[s] = nullptr;
  }
  const size_t released_charge = (current_segment_count - old_segment_count) *
                                 kSegmentSize *
                                 (sizeof(Entry) + sizeof(ResidentEntry));
  metadata_charge_ -= released_charge;
  MemoryLimitPool::get_instance().release_metadata(released_charge);
  return true;
}

char *VectorPageTable::acquire_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);
  // Increment only a resident, non-evicting page.  The old fetch_add-first
  // protocol also modified negative sentinel states, then needed a contended
  // CAS loop to undo that speculative increment.  Besides wasting work on
  // misses, those writes delayed set_block_acquired() from observing the exact
  // unloaded sentinel.  Query-level pin reuse keeps this CAS off repeated
  // accesses to the same page.
  int count = e.ref_count.load(std::memory_order_acquire);
  while (ailego_likely(count >= 0)) {
    if (e.ref_count.compare_exchange_weak(count, count + 1,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
      // Sample the observability counter and CLOCK reference bit together.
      if (sample_hit()) {
        if (!e.referenced.load(std::memory_order_relaxed)) {
          e.referenced.store(true, std::memory_order_relaxed);
        }
        inc_sampled_hit();
      }
      return resident_entry_at(block_id).buffer.load(std::memory_order_acquire);
    }
  }
  return nullptr;
}

void VectorPageTable::release_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);

  // Loaded pages are normally registered with the eviction queue once, when
  // they are installed.  Consequently the hot release path is just one RMW:
  // it does not make the last reader contend on in_evict_queue every time a
  // short-lived MemoryBlock drops the count to zero.
  //
  // The fallback below is only needed if a previous queue insertion failed.
  // It preserves recoverability under queue-capacity pressure without putting
  // a CAS on the steady-state path.
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

bool VectorPageTable::do_evict_block(block_id_t block_id, bool force) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);
  int expected = 0;
  static constexpr int kEvicting = std::numeric_limits<int>::min() / 2;
  bool evicted = false;
  if (e.ref_count.compare_exchange_strong(expected, kEvicting)) {
    // CLOCK second chance: if the page was accessed since it entered the
    // eviction queue, spare it once -- clear the bit, return it to the
    // released state and re-enqueue it for a later eviction pass.  This is
    // what turns the underlying FIFO queue into an approximate-LRU policy.
    // Skipped when `force` is set (teardown/reset must always reclaim).
    if (!force && e.referenced.load(std::memory_order_relaxed)) {
      e.referenced.store(false, std::memory_order_relaxed);
      inc_second_chance();
      // Restore the released state.  Keep in_evict_queue == true because the
      // page is (re)inserted into the queue below; recycle() only consumed
      // the previous slot.
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
    char *buffer =
        resident_entry_at(block_id).buffer.load(std::memory_order_acquire);
    if (buffer && e.is_dirty.load(std::memory_order_relaxed)) {
      int flush_rc = -1;
      if (flush_callback_) {
        try {
          flush_rc =
              flush_callback_(block_id, buffer, kVectorPageSize, e.file_offset);
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
        // A normal eviction must never turn a failed writeback into data
        // loss. Keep both the resident pointer and dirty bit intact, restore
        // the released state, and give the page another queue turn.
        e.ref_count.store(0, std::memory_order_release);
        BlockEvictionQueue::BlockType block;
        block.owner = this;
        block.owner_key = block_id;
        block.version = owner_version_;
        if (!BlockEvictionQueue::get_instance().add_single_block(
                block, static_cast<int>(
                           e.evict_priority.load(std::memory_order_relaxed)))) {
          e.in_evict_queue.store(false, std::memory_order_relaxed);
        }
        return false;
      }
      if (flush_rc == 0) {
        e.is_dirty.store(false, std::memory_order_relaxed);
        inc_dirty_flush();
      } else {
        LOG_ERROR(
            "VectorPageTable::force_evict_block: discarding dirty block %zu "
            "after flush failure during teardown",
            static_cast<size_t>(block_id));
      }
    }
    buffer = resident_entry_at(block_id).buffer.exchange(
        nullptr, std::memory_order_acq_rel);
    if (buffer) {
      if (retire_callback_) {
        retire_callback_(buffer);
      } else {
        MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
      }
    }
    inc_evict();
    // Clear the consumed queue membership before publishing the unloaded
    // sentinel. A loader may install this page as soon as it observes
    // ref_count == INT_MIN and will set in_evict_queue back to true. Clearing
    // the flag after that publication can race with the new installation and
    // strand a resident page outside the eviction queue forever.
    e.in_evict_queue.store(false, std::memory_order_relaxed);
    e.ref_count.store(std::numeric_limits<int>::min(),
                      std::memory_order_release);
    evicted = true;
    return true;
  }

  // The page is still pinned.  Keep its single logical queue membership and
  // move it to the tail instead of clearing in_evict_queue and making the
  // last reader perform another CAS + enqueue.  recycle()/batch_recycle()
  // bound their attempts, so a long-held page cannot cause an unbounded scan.
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
  return evicted;
}

char *VectorPageTable::set_block_acquired(block_id_t block_id, char *buffer,
                                          size_t file_offset) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  Entry &e = entry_at(block_id);
  using clock = std::chrono::steady_clock;
  const auto wait_start = clock::now();
  auto last_log = wait_start;
  unsigned spin_count = 0;
  bool warned = false;
  static constexpr auto kHardTimeout = std::chrono::seconds(30);
  while (true) {
    int current_count = e.ref_count.load(std::memory_order_acquire);
    if (current_count >= 0) {
      if (e.ref_count.compare_exchange_weak(current_count, current_count + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
        return resident_entry_at(block_id).buffer.load(
            std::memory_order_acquire);
      }
    } else if (current_count == std::numeric_limits<int>::min()) {
      // Epoch readers publish no ref_count edge, so the buffer pointer itself
      // must release-publish the fully initialized page contents.
      resident_entry_at(block_id).buffer.store(buffer,
                                               std::memory_order_release);
      e.file_offset = file_offset;
      e.is_dirty.store(false, std::memory_order_relaxed);
      e.referenced.store(false, std::memory_order_relaxed);
      e.ever_loaded = true;
      // Register each installed page once.  Publishing ref_count before the
      // queue item becomes visible ensures an eager evictor observes the page
      // as pinned and simply moves it to the queue tail.
      e.in_evict_queue.store(true, std::memory_order_relaxed);
      e.ref_count.store(1, std::memory_order_release);
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = block_id;
      block.version = owner_version_;
      if (!BlockEvictionQueue::get_instance().add_single_block(
              block, static_cast<int>(
                         e.evict_priority.load(std::memory_order_relaxed)))) {
        // The final release will take the rare fallback registration path.
        eviction_requeue_failed(block_id, owner_version_);
      }
      return resident_entry_at(block_id).buffer.load(std::memory_order_acquire);
    } else {
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
            "set_block_acquired: long kEvicting wait on block_id=%zu "
            "(>=100ms); evict_block may be slow",
            static_cast<size_t>(block_id));
        warned = true;
      }
      if (elapsed >= kHardTimeout) {
        LOG_ERROR(
            "set_block_acquired: hard timeout (%lld s) on block_id=%zu; "
            "giving up to prevent indefinite thread hang",
            static_cast<long long>(
                std::chrono::duration_cast<std::chrono::seconds>(elapsed)
                    .count()),
            static_cast<size_t>(block_id));
        MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
        return nullptr;
      }
      if (elapsed >= std::chrono::seconds(1) &&
          (now - last_log) >= std::chrono::seconds(1)) {
        const auto secs =
            std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        LOG_ERROR(
            "set_block_acquired: stuck in kEvicting on block_id=%zu for "
            "%lld s; evict_block owner may be hung or starved",
            static_cast<size_t>(block_id), static_cast<long long>(secs));
        last_log = now;
      }
    }
  }
}

VecBufferPool::VecBufferPool(const std::string &filename, bool writable,
                             bool enable_direct_io, bool enable_io_profile) {
  file_name_ = filename;
  writable_ = writable;
  io_profile_enabled_ = enable_io_profile;
#if defined(_MSC_VER)
  int flags = writable_ ? (O_RDWR | _O_BINARY) : (O_RDONLY | _O_BINARY);
  fd_ = _open(filename.c_str(), flags, 0644);
  meta_fd_ = _open(filename.c_str(), flags, 0644);
  (void)enable_direct_io;  // O_DIRECT not supported on this path
#else
  int base_flags = writable_ ? O_RDWR : O_RDONLY;
  // Metadata channel: always buffered IO. Serves the unaligned
  // header/footer/segment_meta reads & writes and benefits from page cache.
  meta_fd_ = ::open(filename.c_str(), base_flags, 0644);
  // Page-data channel: optionally O_DIRECT; fall back to buffered open when
  // the filesystem (tmpfs/overlayfs/...) rejects O_DIRECT.
  int data_flags = base_flags;
#ifdef O_DIRECT
  if (enable_direct_io) {
    data_flags |= O_DIRECT;
  }
#endif
  fd_ = ::open(filename.c_str(), data_flags, 0644);
#ifdef O_DIRECT
  if (fd_ < 0 && (data_flags & O_DIRECT)) {
    LOG_WARN(
        "VecBufferPool: open with O_DIRECT failed for file[%s] (errno=%d), "
        "falling back to buffered IO",
        filename.c_str(), errno);
    fd_ = ::open(filename.c_str(), base_flags, 0644);
    direct_io_enabled_ = false;
  } else {
    direct_io_enabled_ = (data_flags & O_DIRECT) != 0;
  }
#else
  (void)enable_direct_io;
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
#if defined(__linux) || defined(__linux__)
  // Capability probe only. Actual kernel AIO contexts are created lazily in
  // thread-local state, so opening N pools does not reserve 256*N aio events.
  aio_enabled_ = direct_io_enabled_ && LibAioLoader::Instance().load();
#endif
}

size_t VecBufferPool::metadata_bytes_for_page_count(size_t page_count) {
  const size_t page_table_bytes =
      VectorPageTable::metadata_bytes_for_entries(page_count);
  if (page_table_bytes == std::numeric_limits<size_t>::max()) {
    return page_table_bytes;
  }
  const size_t mutex_count =
      std::max<size_t>(1, std::min(page_count, kMutexBucketCount));
  const size_t mutex_bytes = mutex_count * sizeof(std::mutex);
  if (page_table_bytes > std::numeric_limits<size_t>::max() - mutex_bytes) {
    return std::numeric_limits<size_t>::max();
  }
  return page_table_bytes + mutex_bytes;
}

int VecBufferPool::init() {
  if (initialized_) {
    return 0;
  }
  // Configure callbacks before reserving metadata. std::function assignment
  // may allocate, so a failure here leaves both the page table and shared
  // budget untouched.
  try {
    if (!writable_) {
      page_table_.set_retire_callback(
          [this](char *buffer) { read_epoch_domain_.retire(buffer); });
    } else {
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
    }
  } catch (const std::bad_alloc &) {
    LOG_ERROR("VecBufferPool::init: failed to allocate callbacks for file[%s]",
              file_name_.c_str());
    return -1;
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
  const size_t mutex_count =
      std::max<size_t>(1, std::min(block_num, kMutexBucketCount));
  const size_t mutex_charge = mutex_count * sizeof(std::mutex);
  if (!MemoryLimitPool::get_instance().try_charge_metadata(mutex_charge)) {
    LOG_ERROR(
        "VecBufferPool::init: shared memory budget cannot reserve %zu bytes "
        "for %zu page-lock stripes (file=%s)",
        mutex_charge, mutex_count, file_name_.c_str());
    return -1;
  }
  std::unique_ptr<std::mutex[]> mutexes;
  try {
    mutexes = std::make_unique<std::mutex[]>(mutex_count);
  } catch (const std::bad_alloc &) {
    MemoryLimitPool::get_instance().release_metadata(mutex_charge);
    LOG_ERROR(
        "VecBufferPool::init: failed to allocate %zu page-lock stripes "
        "(file=%s)",
        mutex_count, file_name_.c_str());
    return -1;
  }
  if (!page_table_.init(block_num)) {
    MemoryLimitPool::get_instance().release_metadata(mutex_charge);
    LOG_ERROR(
        "VecBufferPool::init: page_table_ init failed for file[%s], "
        "file_size=%zu, block_num=%zu, required_metadata=%zu",
        file_name_.c_str(), file_size_, block_num,
        metadata_bytes_for_page_count(block_num));
    return -1;
  }
  block_mutexes_ = std::move(mutexes);
  block_mutex_count_ = mutex_count;
  mutex_metadata_charge_ = mutex_charge;
  LOG_DEBUG("entry num: %zu, file_size: %zu", page_table_.entry_num(),
            file_size_);

  initialized_ = true;
  return 0;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
  return VecBufferPoolHandle(*this);
}

char *VecBufferPool::acquire_buffer(block_id_t page_id, int retry) {
  assert(page_id < page_table_.entry_num());
  char *buffer = page_table_.acquire_block(page_id);
  if (buffer) {
    return buffer;
  }
  BufferPoolIoProfile *profile = current_thread_io_profile();
  std::unique_lock<std::mutex> lock(
      block_mutexes_[page_id % block_mutex_count_], std::defer_lock);
  if (profile) {
    const uint64_t lock_start = BufferPoolProfileNowNs();
    lock.lock();
    profile->sync_page_lock_wait_ns += BufferPoolProfileNowNs() - lock_start;
  } else {
    lock.lock();
  }
  buffer = page_table_.acquire_block(page_id);
  if (buffer) {
    return buffer;
  }
  {
    BufferPoolProfileTimer prepare_timer(profile ? &profile->sync_prepare_ns
                                                 : nullptr);
    bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
        kVectorPageSize, buffer);
    if (!found) {
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
      LOG_ERROR(
          "Buffer pool failed to get free buffer: file[%s], page_id[%zu], "
          "used[%zu], committed[%zu], free_buffers[%zu], evict[%llu], "
          "second_chance[%llu]",
          file_name_.c_str(), page_id, memory_stats.used,
          memory_stats.committed, memory_stats.free_buffers,
          static_cast<unsigned long long>(page_stats.evict),
          static_cast<unsigned long long>(page_stats.second_chance));
      return nullptr;
    }
  }

  size_t page_offset = page_id * kVectorPageSize;
  // Cold path: the page is being loaded from disk (or zero-filled), i.e. a
  // cache miss.  Count it for the pool hit-rate metric.
  miss_count_.fetch_add(1, std::memory_order_relaxed);
  // Skip pread for pages created by extend_file (beyond the original file
  // size at open time) that have never been loaded before.  Their on-disk
  // content is guaranteed to be zeros (ftruncate).  After eviction the
  // ever_loaded flag stays true so reloads correctly pread the flushed data.
  if (writable_ && page_offset >= initial_file_size_ &&
      !page_table_.is_ever_loaded(page_id)) {
    BufferPoolProfileTimer prepare_timer(profile ? &profile->sync_prepare_ns
                                                 : nullptr);
    std::memset(buffer, 0, kVectorPageSize);
  } else {
    // O_DIRECT requires the IO length to be a multiple of the device block
    // size. For files whose size is page-aligned (e.g. BufferStorage), reading
    // a full page never reads past EOF. For files that are NOT page-aligned
    // (e.g. IVF via BufferReadStorage), the last page may be a short read;
    // we accept it and zero-pad the remainder.
    size_t read_len = direct_io_enabled_
                          ? kVectorPageSize
                          : std::min(kVectorPageSize, file_size_ - page_offset);
    if (read_len < kVectorPageSize) {
      std::memset(buffer + read_len, 0, kVectorPageSize - read_len);
    }
    ssize_t read_bytes = 0;
    {
      BufferPoolProfileTimer read_timer(profile ? &profile->sync_read_ns
                                                : nullptr);
      read_bytes = zvec_pread(fd_, buffer, read_len, page_offset);
    }
    if (profile) ++profile->sync_reads;
    if (read_bytes != static_cast<ssize_t>(read_len)) {
      // Accept short read at EOF: last page may not be full kVectorPageSize
      if (read_bytes > 0 &&
          (page_offset + static_cast<size_t>(read_bytes) >= file_size_)) {
        std::memset(buffer + read_bytes, 0, kVectorPageSize - read_bytes);
      } else {
        LOG_ERROR(
            "Buffer pool failed to read file at offset: file[%s], "
            "page_id[%zu], "
            "offset[%zu], expected[%zu], got[%zd]",
            file_name_.c_str(), page_id, page_offset, read_len, read_bytes);
        MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
        return nullptr;
      }
    }
  }
  char *installed = nullptr;
  {
    BufferPoolProfileTimer install_timer(profile ? &profile->sync_install_ns
                                                 : nullptr);
    installed = page_table_.set_block_acquired(page_id, buffer, page_offset);
  }
  return installed;
}

bool VecBufferPool::acquire_pages(const block_id_t *page_ids, size_t count,
                                  char **pages) {
  if (count == 0) return true;
  if (!page_ids || !pages) return false;

  std::fill_n(pages, count, nullptr);
  static constexpr size_t kAioBatch = 128;
  std::array<block_id_t, kAioBatch> miss_batch{};
  size_t miss_count = 0;

  // Pin hits before issuing I/O.  Apart from avoiding needless submissions,
  // this guarantees eviction cannot reclaim a page between hit detection and
  // delivery to the caller.  Miss ids use a fixed stack batch so this API adds
  // no heap allocation to an all-hit query.
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
        submit_aio_async(miss_batch.data(), miss_count);
        wait_aio();
        miss_count = 0;
      }
    }
  }
  if (miss_count != 0) {
    submit_aio_async(miss_batch.data(), miss_count);
    wait_aio();
  }

  // AIO installs pages without leaving them pinned.  Resolve all original
  // output slots now, including duplicates.  On platforms without libaio this
  // is also the synchronous fallback path.
  for (size_t i = 0; i < count; ++i) {
    if (pages[i]) continue;
    pages[i] = acquire_buffer(page_ids[i], 50);
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

  char *page = static_cast<char *>(
      ailego_aligned_malloc(kVectorPageSize, kVectorPageSize));
  if (page == nullptr) {
    return false;
  }

  size_t copied = 0;
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

    const ssize_t read_bytes = zvec_pread(fd_, page, read_size, page_offset);
    if (read_bytes <= 0 ||
        within_page + copy_size > static_cast<size_t>(read_bytes)) {
      ok = false;
      break;
    }
    std::memcpy(buffer + copied, page + within_page, copy_size);
    copied += copy_size;
  }
  ailego_free(page);

  if (ok) {
    bypass_reads_.fetch_add(1, std::memory_order_relaxed);
    bypass_bytes_.fetch_add(length, std::memory_order_relaxed);
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
    // Loading the page ensures we do not clobber unrelated bytes within the
    // same page when the write is not page-aligned. acquire_buffer() pre-fills
    // from the backing file (or zero-pads beyond EOF).
    char *page = this->acquire_buffer(pg, 50);
    if (!page) {
      LOG_ERROR("write_range acquire failed: file[%s], page[%zu]",
                file_name_.c_str(), pg);
      return -1;
    }
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
  const size_t total = page_table_.entry_num();
  if (total == 0) {
    return 0;
  }

  static constexpr size_t kBatchPages = 256;
  const size_t kBatchSize = kBatchPages * kVectorPageSize;
  char *batch_buf =
      static_cast<char *>(ailego_aligned_malloc(kBatchSize, 4096));

  int rc = 0;
  size_t total_dirty = 0;
  size_t fail_count = 0;
  size_t i = 0;

  while (i < total) {
    if (!page_table_.is_block_dirty(i)) {
      ++i;
      continue;
    }

    const size_t run_start = i;
    size_t run_count = 0;
    const size_t limit = batch_buf ? kBatchPages : 1;
    while (i < total && run_count < limit && page_table_.is_block_dirty(i)) {
      char *buf = page_table_.acquire_block(i);
      if (!buf) {
        break;
      }
      // Another flusher may have completed between the optimistic dirty check
      // and the pin. Do not include a now-clean page in this contiguous run.
      if (!page_table_.is_block_dirty(i)) {
        page_table_.release_block(i);
        break;
      }
      if (batch_buf) {
        std::memcpy(batch_buf + run_count * kVectorPageSize, buf,
                    kVectorPageSize);
      }
      ++run_count;
      ++i;
    }
    if (run_count == 0) {
      ++i;
      continue;
    }
    total_dirty += run_count;

    bool ok = false;
    if (batch_buf && run_count > 0) {
      const size_t write_size = run_count * kVectorPageSize;
      ssize_t w =
          zvec_pwrite(fd_, batch_buf, write_size, run_start * kVectorPageSize);
      ok = (w == static_cast<ssize_t>(write_size));
    }
    if (ok) {
      for (size_t j = run_start; j < run_start + run_count; ++j) {
        page_table_.clear_dirty(j);
      }
    } else {
      for (size_t j = run_start; j < run_start + run_count; ++j) {
        int r = page_table_.flush_block(j);
        if (r != 0) {
          rc = r;
          ++fail_count;
        }
      }
    }
    // Keep every page pinned until the write and dirty-flag update complete.
    // The background evictor can no longer recycle a buffer while flush_all()
    // is copying or writing it.
    for (size_t j = 0; j < run_count; ++j) {
      page_table_.release_block(run_start + j);
    }
  }

  if (batch_buf) {
    ailego_free(batch_buf);
  }
  if (fail_count != 0) {
    LOG_ERROR(
        "VecBufferPool::flush_all: %zu/%zu dirty page(s) failed to flush, "
        "file[%s] last_rc=%d -- on-disk data may be stale.",
        fail_count, total_dirty, file_name_.c_str(), rc);
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
  // The backing file must stay page-aligned so that O_DIRECT full-page reads
  // never read past EOF. All current callers pass page-aligned targets.
  if (new_size % kVectorPageSize != 0) {
    LOG_ERROR(
        "extend_file target must be page-aligned: file[%s], new_size[%zu], "
        "page_size[%zu]",
        file_name_.c_str(), new_size, kVectorPageSize);
    return false;
  }
  // Pre-validate against the page table's static capacity BEFORE mutating
  // any on-disk state.  Otherwise a successful ftruncate followed by a
  // failed page_table_.extend() would leave the file size and the page
  // table out of sync (file grew, but no Entry slots cover the new range).
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
    LOG_ERROR(
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
        page = pool_.acquire_buffer(static_cast<block_id_t>(run_start + j), 50);
        if (!page) return false;
        block_id_t pid = static_cast<block_id_t>(run_start + j);
        size_t page_start = pid * kVectorPageSize;
        size_t intra_offset =
            (pid == first_page) ? (file_offset - page_start) : 0;
        size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
        std::memcpy(out + dst_cursor, page + intra_offset, chunk);
        pool_.page_table_.release_block(pid);
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
      if (!page) return false;
      size_t page_start = pg * kVectorPageSize;
      size_t intra_offset = (pg == first_page) ? (file_offset - page_start) : 0;
      size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
      std::memcpy(out + dst_cursor, page + intra_offset, chunk);
      pool_.page_table_.release_block(static_cast<block_id_t>(pg));
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
    // The bulk path bypasses acquire_buffer(), so account for the cold pages
    // fetched by this pread here. Without this, cache-miss statistics depend
    // on the host page size and whether a read crosses the bulk threshold.
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
      if (page_end_in_buf <= actually_read &&
          !pool_.page_table_.is_loaded(pid)) {
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
          pool_.page_table_.set_block_acquired(
              pid, page_buf, run_file_off + j * kVectorPageSize);
          pool_.page_table_.release_block(pid);
        }
      }
    }
    ailego_free(bulk_buf);
    pg = run_end - 1;
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
  // The caller must guarantee the block is already loaded before calling
  // acquire_one(). The return value of acquire_block() is intentionally
  // ignored here, as a null return would indicate a contract violation.
  pool_.page_table_.acquire_block(block_id);
}

void VecBufferPool::warmup() {
  const size_t total_pages = page_table_.entry_num();
  // Read in large sequential chunks to minimize syscall overhead.
  // Each chunk = 1024 pages = 4MB (maximize sequential I/O throughput).
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
      page_table_.set_block_acquired(page_id, buf,
                                     file_offset + j * kVectorPageSize);
      page_table_.release_block(page_id);
      ++loaded;
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

#if defined(__linux) || defined(__linux__)
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

  bool all_loaded = true;
  for (size_t pg = first_page; pg < end_page; ++pg) {
    if (!page_table_.is_loaded(pg)) {
      all_loaded = false;
      break;
    }
  }
  if (all_loaded) return;

  static constexpr size_t kChunkPages = 1024;
  const size_t kChunkSize = kChunkPages * kVectorPageSize;
  char *chunk_buf =
      static_cast<char *>(ailego_aligned_malloc(kChunkSize, 4096));
  if (!chunk_buf) return;

  bool pool_full = false;
  size_t pg = first_page;
  while (pg < end_page && !pool_full) {
    if (page_table_.is_loaded(pg)) {
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
    ssize_t got = zvec_pread(fd_, chunk_buf, read_bytes, file_off);
    if (got != static_cast<ssize_t>(read_bytes)) {
      pg = run_end;
      continue;
    }

    for (size_t j = 0; j < run_pages; ++j) {
      block_id_t pid = static_cast<block_id_t>(run_start + j);
      if (page_table_.is_loaded(pid)) continue;
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
      page_table_.set_evict_priority(pid, priority);
      page_table_.set_block_acquired(pid, buf, file_off + j * kVectorPageSize);
      page_table_.release_block(pid);
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

#if defined(__linux) || defined(__linux__)
namespace {
// Dedicated thread-local AIO context for the *blocking* prefetch path
// (prefetch_pages_aio). Each thread reaps only the events it submitted.
// Sharing one context across threads let thread B's io_getevents steal thread
// A's completions, which made prefetch release buffers whose kernel DMA was
// still in flight -- the DMA then overwrote the buffer's first 8 bytes (the
// MemoryLimitPool free-list next pointer), corrupting the free list.
struct ThreadLocalPrefetchAioCtx {
  io_context_t ctx{nullptr};
  bool inited{false};
  bool ok{false};
  char *quarantined[128]{};
  size_t quarantined_count{0};

  bool ensure() {
    if (inited) return ok;
    inited = true;
    if (!LibAioLoader::Instance().is_available()) return false;
    ctx = nullptr;
    if (LibAioLoader::Instance().io_setup(256, &ctx) == 0) {
      ok = true;
    }
    return ok;
  }

  bool destroy_context() {
    ok = false;
    if (!ctx) return true;
    const int ret = LibAioLoader::Instance().io_destroy(ctx);
    if (ret != 0) {
      LOG_ERROR(
          "ThreadLocalPrefetchAioCtx: io_destroy failed, ret=%d; "
          "in-flight buffers remain quarantined",
          ret);
      return false;
    }
    ctx = nullptr;
    return true;
  }

  void quarantine(char **buffers, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      if (buffers[i]) {
        assert(quarantined_count < 128);
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

  ~ThreadLocalPrefetchAioCtx() {
    if (destroy_context()) {
      release_quarantined();
    }
  }
};
static thread_local ThreadLocalPrefetchAioCtx tl_prefetch_aio;
}  // namespace
#endif

void VecBufferPool::prefetch_pages_aio([[maybe_unused]] block_id_t first_page,
                                       [[maybe_unused]] size_t page_count,
                                       [[maybe_unused]] uint8_t priority) {
  const size_t total_pages = page_table_.entry_num();
  if (priority > kHighPriority || page_count == 0 ||
      first_page >= total_pages) {
    return;
  }
  page_count = std::min(page_count, total_pages - first_page);

#if defined(__linux) || defined(__linux__)
  static constexpr size_t kMaxBatch = 128;

  // Use a thread-local AIO context: each thread waits only for its own
  // completions, so a buffer is never returned to the free-list while the
  // kernel is still DMA-ing into it.
  if (!tl_prefetch_aio.ensure()) {
    prefetch_pages_sync(first_page, page_count, priority);
    return;
  }
  io_context_t ctx = tl_prefetch_aio.ctx;

  const size_t end_page = first_page + page_count;

  size_t pg = first_page;
  while (pg < end_page) {
    std::vector<block_id_t> miss_pages;
    miss_pages.reserve(kMaxBatch);
    while (pg < end_page && miss_pages.size() < kMaxBatch) {
      if (!page_table_.is_loaded(pg)) {
        miss_pages.push_back(static_cast<block_id_t>(pg));
      }
      ++pg;
    }
    if (miss_pages.empty()) continue;

    size_t count = miss_pages.size();
    std::vector<struct iocb> cbs(count);
    std::vector<struct iocb *> cb_ptrs(count);
    std::vector<char *> buffers(count, nullptr);

    size_t submitted = 0;
    for (size_t i = 0; i < count; ++i) {
      char *buf = nullptr;
      bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
          kVectorPageSize, buf);
      if (!found) {
        BlockEvictionQueue::get_instance().recycle();
        found = MemoryLimitPool::get_instance().try_acquire_buffer(
            kVectorPageSize, buf);
      }
      if (!found) break;
      buffers[submitted] = buf;
      size_t offset = miss_pages[submitted] * kVectorPageSize;
      io_prep_pread(&cbs[submitted], fd_, buf, kVectorPageSize,
                    static_cast<long long>(offset));
      // Record the submission index so out-of-order completions can be mapped
      // back to the correct buffer/page.
      cbs[submitted].data = reinterpret_cast<void *>(submitted);
      cb_ptrs[submitted] = &cbs[submitted];
      ++submitted;
    }

    if (submitted == 0) {
      prefetch_pages_sync(first_page, page_count, priority);
      return;
    }

    int ret = LibAioLoader::Instance().io_submit(
        ctx, static_cast<long>(submitted), cb_ptrs.data());
    if (ret <= 0) {
      for (size_t i = 0; i < submitted; ++i) {
        MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                       kVectorPageSize);
      }
      prefetch_pages_sync(first_page, page_count, priority);
      return;
    }

    // io_submit may accept fewer than requested; the accepted requests are a
    // prefix of cb_ptrs.  The tail was never submitted (no in-flight DMA), so
    // its buffers are safe to release immediately.
    size_t accepted = static_cast<size_t>(ret);
    const bool needs_sync_fallback = accepted < count;
    for (size_t i = accepted; i < submitted; ++i) {
      MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                     kVectorPageSize);
      buffers[i] = nullptr;
    }

    // Block until every accepted I/O completes.  The blocking wait (nullptr
    // timeout) guarantees no DMA is still in flight when we touch the buffers
    // below, so a buffer can never be written by the kernel after being
    // returned to the free-list.
    std::vector<struct io_event> events(accepted);
    size_t done = 0;
    bool wait_failed = false;
    while (done < accepted) {
      int n = LibAioLoader::Instance().io_getevents(
          ctx, static_cast<long>(accepted - done),
          static_cast<long>(accepted - done), events.data() + done, nullptr);
      if (n == -EINTR) {
        continue;
      }
      if (n <= 0) {
        LOG_ERROR(
            "VecBufferPool::prefetch_pages_aio: io_getevents failed, ret=%d",
            n);
        wait_failed = true;
        break;
      }
      done += static_cast<size_t>(n);
    }

    if (wait_failed) {
      // io_destroy blocks until accepted requests have completed/cancelled.
      // Only after it succeeds is every buffer safe to return to the pool.
      if (tl_prefetch_aio.destroy_context()) {
        for (size_t i = 0; i < accepted; ++i) {
          if (buffers[i]) {
            MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                           kVectorPageSize);
            buffers[i] = nullptr;
          }
        }
        prefetch_pages_sync(first_page, page_count, priority);
      } else {
        // Completed events are known to be DMA-safe. Process those below;
        // buffers without an event remain intentionally quarantined because
        // the kernel context could not be stopped.
      }
      if (tl_prefetch_aio.ctx == nullptr) {
        return;
      }
    }

    for (size_t i = 0; i < done; ++i) {
      size_t idx = reinterpret_cast<size_t>(events[i].data);
      if (idx >= submitted || buffers[idx] == nullptr) continue;
      block_id_t pid = miss_pages[idx];
      if (static_cast<ssize_t>(events[i].res) ==
          static_cast<ssize_t>(kVectorPageSize)) {
        std::lock_guard<std::mutex> lock(
            block_mutexes_[pid % block_mutex_count_]);
        if (page_table_.is_loaded(pid)) {
          MemoryLimitPool::get_instance().release_buffer(buffers[idx],
                                                         kVectorPageSize);
        } else {
          page_table_.set_evict_priority(pid, priority);
          page_table_.set_block_acquired(pid, buffers[idx],
                                         pid * kVectorPageSize);
          page_table_.release_block(pid);
        }
      } else {
        MemoryLimitPool::get_instance().release_buffer(buffers[idx],
                                                       kVectorPageSize);
      }
      buffers[idx] = nullptr;
    }

    if (wait_failed) {
      tl_prefetch_aio.quarantine(buffers.data(), accepted);
      return;
    }
    if (needs_sync_fallback) {
      // io_submit can accept only a prefix, and allocation pressure can stop
      // preparation early. Do not silently skip the rest of the requested
      // prefetch range.
      prefetch_pages_sync(first_page, page_count, priority);
      return;
    }
  }
#else
  prefetch_pages_sync(first_page, page_count, priority);
#endif
}

#if defined(__linux) || defined(__linux__)
namespace {
struct ThreadLocalAioCtx {
  io_context_t ctx{nullptr};
  bool inited{false};
  bool ok{false};

  // Pending async AIO state
  char *pending_bufs[128];
  block_id_t pending_pids[128];
  size_t pending_count{0};    // total submitted (slot high-water mark)
  size_t harvested_count{0};  // how many completed & processed
  VecBufferPool *pending_pool{nullptr};

  void release_pending_buffers() {
    for (size_t i = 0; i < pending_count; ++i) {
      if (pending_bufs[i]) {
        MemoryLimitPool::get_instance().release_buffer(pending_bufs[i],
                                                       kVectorPageSize);
        pending_bufs[i] = nullptr;
      }
    }
    pending_count = 0;
    harvested_count = 0;
    pending_pool = nullptr;
  }

  bool destroy_context() {
    ok = false;
    if (!ctx) return true;
    const int ret = LibAioLoader::Instance().io_destroy(ctx);
    if (ret != 0) {
      LOG_ERROR(
          "ThreadLocalAioCtx: io_destroy failed, ret=%d; in-flight buffers "
          "remain quarantined",
          ret);
      return false;
    }
    ctx = nullptr;
    return true;
  }

  bool abort_pending() {
    // io_destroy is the ownership barrier: after it returns successfully the
    // kernel can no longer DMA into any request buffer.
    if (!destroy_context()) return false;
    release_pending_buffers();
    return true;
  }

  ~ThreadLocalAioCtx() {
    // Drain all pending AIO before destroying context (thread exit)
    size_t in_flight = pending_count - harvested_count;
    bool all_reaped = in_flight == 0;
    if (in_flight > 0 && ok) {
      struct io_event events[128];
      // Must block-wait: kernel is still DMA-ing into our buffers
      size_t done = 0;
      while (done < in_flight) {
        int ret = LibAioLoader::Instance().io_getevents(
            ctx, static_cast<long>(in_flight - done),
            static_cast<long>(in_flight - done), events + done, nullptr);
        if (ret == -EINTR) {
          continue;
        }
        if (ret <= 0) {
          LOG_ERROR(
              "ThreadLocalAioCtx: io_getevents failed during teardown, "
              "ret=%d",
              ret);
          break;
        }
        done += static_cast<size_t>(ret);
      }
      all_reaped = done == in_flight;
    }
    const bool destroyed = destroy_context();
    // Fully reaped buffers are already DMA-safe. Otherwise io_destroy must
    // succeed before any outstanding buffer is returned to the free-list.
    if (all_reaped || destroyed) {
      release_pending_buffers();
    }
  }

  bool ensure() {
    if (inited) return ok;
    inited = true;
    if (!LibAioLoader::Instance().is_available()) return false;
    ctx = nullptr;
    if (LibAioLoader::Instance().io_setup(128, &ctx) == 0) {
      ok = true;
    }
    return ok;
  }
};

static thread_local ThreadLocalAioCtx tl_aio;
}  // namespace
#endif

void VecBufferPool::submit_aio_async(const block_id_t *page_ids, size_t count,
                                     BufferPoolIoProfile *profile) {
  if (count == 0) return;
  if (!profile) profile = current_thread_io_profile();

#if defined(__linux) || defined(__linux__)
  if (!direct_io_enabled_) return;
  if (!tl_aio.ensure()) return;

  // A thread-local AIO context may be shared by searches that touch different
  // buffer pools.  Never append requests for this pool to a batch owned by a
  // different pool: completions are installed through pending_pool, so mixing
  // pools would publish pages into the wrong page table (and use the wrong
  // striped mutex set).  Drain the previous owner's batch first.
  if (tl_aio.pending_count > 0 && tl_aio.pending_pool != this) {
    wait_aio(profile);
    if (tl_aio.pending_count > 0) {
      // A fatal io_getevents error left the old DMA batch in flight.  It is
      // unsafe to reuse either the slots or their buffers.
      return;
    }
  }

  // Harvest any previously completed AIO (non-blocking)
  if (tl_aio.pending_count > 0) {
    harvest_aio();
  }

  // Time only preparation + io_submit.  A cross-pool drain above is already
  // attributed to aio_wait_ns and must not be double-counted as submit work.
  BufferPoolProfileTimer submit_timer(profile ? &profile->aio_submit_ns
                                              : nullptr);

  // Determine how many slots are available for new submissions
  size_t base = tl_aio.pending_count;  // existing in-flight count
  size_t max_new = 128 - base;
  if (max_new == 0) return;  // All slots occupied by in-flight I/O

  // Filter: skip already-loaded, in-flight, and deduplicate
  block_id_t miss_pages[128];
  size_t miss_count = 0;
  for (size_t i = 0; i < count && miss_count < max_new; ++i) {
    if (page_ids[i] >= page_table_.entry_num()) continue;
    if (page_table_.is_loaded(page_ids[i])) continue;
    // Skip if already in pending (still in-flight, buf != nullptr)
    bool in_flight = false;
    for (size_t j = 0; j < base; ++j) {
      if (tl_aio.pending_bufs[j] && tl_aio.pending_pids[j] == page_ids[i]) {
        in_flight = true;
        break;
      }
    }
    if (in_flight) continue;
    bool dup = false;
    for (size_t j = 0; j < miss_count; ++j) {
      if (miss_pages[j] == page_ids[i]) {
        dup = true;
        break;
      }
    }
    if (!dup) miss_pages[miss_count++] = page_ids[i];
  }
  if (miss_count == 0) return;

  BlockEvictionQueue::get_instance().batch_recycle(miss_count);
  char *buffers[128];
  size_t submitted = MemoryLimitPool::get_instance().batch_acquire_buffers(
      kVectorPageSize, buffers, miss_count);
  if (submitted == 0) return;

  struct iocb cbs[128];
  struct iocb *cb_ptrs[128];
  for (size_t i = 0; i < submitted; ++i) {
    size_t offset = miss_pages[i] * kVectorPageSize;
    io_prep_pread(&cbs[i], fd_, buffers[i], kVectorPageSize,
                  static_cast<long long>(offset));
    // Index = base + i so harvest can map back to pending_bufs/pids
    cbs[i].data = reinterpret_cast<void *>(base + i);
    cb_ptrs[i] = &cbs[i];
  }

  int ret = LibAioLoader::Instance().io_submit(
      tl_aio.ctx, static_cast<long>(submitted), cb_ptrs);
  if (ret <= 0) {
    for (size_t i = 0; i < submitted; ++i) {
      MemoryLimitPool::get_instance().release_buffer(buffers[i],
                                                     kVectorPageSize);
    }
    return;
  }

  // Append to pending state
  size_t actual = static_cast<size_t>(ret);
  if (profile) {
    ++profile->aio_batches;
    profile->aio_pages += actual;
    profile->aio_max_batch = std::max<uint64_t>(profile->aio_max_batch, actual);
  }
  for (size_t i = 0; i < actual; ++i) {
    tl_aio.pending_bufs[base + i] = buffers[i];
    tl_aio.pending_pids[base + i] = miss_pages[i];
  }
  // Keep cache observability consistent with the synchronous cold path.
  // The later acquire_pages() resolution is a cache hit because AIO installs
  // the page, but the physical read itself is still one miss.
  miss_count_.fetch_add(actual, std::memory_order_relaxed);
  tl_aio.pending_count = base + actual;
  tl_aio.pending_pool = this;

  // Release buffers for requests that weren't submitted
  for (size_t i = actual; i < submitted; ++i) {
    MemoryLimitPool::get_instance().release_buffer(buffers[i], kVectorPageSize);
  }
#else
  (void)page_ids;
  (void)count;
  (void)profile;
#endif
}

void VecBufferPool::harvest_aio() {
#if defined(__linux) || defined(__linux__)
  if (!tl_aio.ok || tl_aio.pending_count == 0) return;
  size_t in_flight = tl_aio.pending_count - tl_aio.harvested_count;
  if (in_flight == 0) return;

  struct io_event events[128];
  struct timespec timeout = {0, 0};  // Non-blocking
  int ret = LibAioLoader::Instance().io_getevents(
      tl_aio.ctx, 0, static_cast<long>(in_flight), events, &timeout);

  if (ret < 0 && ret != -EINTR) {
    LOG_WARN("VecBufferPool::harvest_aio: io_getevents failed, ret=%d", ret);
    (void)tl_aio.abort_pending();
    return;
  }

  size_t completed = (ret > 0) ? static_cast<size_t>(ret) : 0;
  if (completed == 0) return;  // Nothing ready yet

  VecBufferPool *pool = tl_aio.pending_pool;
  for (size_t i = 0; i < completed; ++i) {
    size_t idx = reinterpret_cast<size_t>(events[i].data);
    if (static_cast<ssize_t>(events[i].res) !=
        static_cast<ssize_t>(kVectorPageSize)) {
      MemoryLimitPool::get_instance().release_buffer(tl_aio.pending_bufs[idx],
                                                     kVectorPageSize);
    } else {
      block_id_t pid = tl_aio.pending_pids[idx];
      std::lock_guard<std::mutex> lock(
          pool->block_mutexes_[pid % pool->block_mutex_count_]);
      if (pool->page_table_.is_loaded(pid)) {
        MemoryLimitPool::get_instance().release_buffer(tl_aio.pending_bufs[idx],
                                                       kVectorPageSize);
      } else {
        pool->page_table_.set_block_acquired(pid, tl_aio.pending_bufs[idx],
                                             pid * kVectorPageSize);
        pool->page_table_.release_block(pid);
      }
    }
    tl_aio.pending_bufs[idx] = nullptr;  // Mark as processed
  }

  tl_aio.harvested_count += completed;

  // If all submitted are harvested, reset for reuse
  if (tl_aio.harvested_count == tl_aio.pending_count) {
    tl_aio.pending_count = 0;
    tl_aio.harvested_count = 0;
    tl_aio.pending_pool = nullptr;
  }
#endif
}

void VecBufferPool::wait_aio(BufferPoolIoProfile *profile) {
  if (!profile) profile = current_thread_io_profile();
#if defined(__linux) || defined(__linux__)
  if (!tl_aio.ok || tl_aio.pending_count == 0) return;

  VecBufferPool *pool = tl_aio.pending_pool;
  struct io_event events[128];
  // Block (NULL timeout) until every in-flight request has completed.  Unlike
  // harvest_aio() this never returns early, so after it the whole submitted
  // batch is guaranteed resident -- the caller can resolve every page as a hit
  // without a per-neighbour synchronous fallback.
  while (tl_aio.harvested_count < tl_aio.pending_count) {
    size_t in_flight = tl_aio.pending_count - tl_aio.harvested_count;
    int ret = 0;
    {
      BufferPoolProfileTimer wait_timer(profile ? &profile->aio_wait_ns
                                                : nullptr);
      ret = LibAioLoader::Instance().io_getevents(
          tl_aio.ctx, static_cast<long>(in_flight),
          static_cast<long>(in_flight), events, nullptr);
    }
    if (ret == -EINTR) continue;
    if (ret <= 0) {
      LOG_ERROR("VecBufferPool::wait_aio: io_getevents failed, ret=%d", ret);
      // Stop/cancel the context before returning buffers. If context teardown
      // itself fails, abort_pending deliberately keeps them charged and
      // quarantined rather than risking DMA into reused memory.
      (void)tl_aio.abort_pending();
      return;
    }
    size_t completed = static_cast<size_t>(ret);
    BufferPoolProfileTimer install_timer(profile ? &profile->aio_install_ns
                                                 : nullptr);
    for (size_t i = 0; i < completed; ++i) {
      size_t idx = reinterpret_cast<size_t>(events[i].data);
      if (static_cast<ssize_t>(events[i].res) !=
          static_cast<ssize_t>(kVectorPageSize)) {
        MemoryLimitPool::get_instance().release_buffer(tl_aio.pending_bufs[idx],
                                                       kVectorPageSize);
      } else {
        block_id_t pid = tl_aio.pending_pids[idx];
        std::unique_lock<std::mutex> lock(
            pool->block_mutexes_[pid % pool->block_mutex_count_],
            std::defer_lock);
        if (profile) {
          const uint64_t lock_start = BufferPoolProfileNowNs();
          lock.lock();
          profile->aio_page_lock_wait_ns +=
              BufferPoolProfileNowNs() - lock_start;
        } else {
          lock.lock();
        }
        if (pool->page_table_.is_loaded(pid)) {
          MemoryLimitPool::get_instance().release_buffer(
              tl_aio.pending_bufs[idx], kVectorPageSize);
        } else {
          pool->page_table_.set_block_acquired(pid, tl_aio.pending_bufs[idx],
                                               pid * kVectorPageSize);
          pool->page_table_.release_block(pid);
        }
      }
      tl_aio.pending_bufs[idx] = nullptr;
    }
    tl_aio.harvested_count += completed;
  }

  if (tl_aio.harvested_count == tl_aio.pending_count) {
    tl_aio.pending_count = 0;
    tl_aio.harvested_count = 0;
    tl_aio.pending_pool = nullptr;
  }
#else
  (void)profile;
#endif
}

void VecBufferPool::log_stats() const {
  Stats s = stats();
  LOG_INFO(
      "VecBufferPool stats: file[%s] hit=%llu miss=%llu hit_rate=%.4f "
      "evict=%llu second_chance=%llu dirty_flush=%llu bypass_reads=%llu "
      "bypass_bytes=%llu",
      file_name_.c_str(), static_cast<unsigned long long>(s.hit),
      static_cast<unsigned long long>(s.miss), s.hit_rate(),
      static_cast<unsigned long long>(s.evict),
      static_cast<unsigned long long>(s.second_chance),
      static_cast<unsigned long long>(s.dirty_flush),
      static_cast<unsigned long long>(s.bypass_reads),
      static_cast<unsigned long long>(s.bypass_bytes));
  if (io_profile_enabled_) {
    const BufferPoolIoProfile &p = s.io_profile;
    const double avg_batch =
        p.aio_batches ? static_cast<double>(p.aio_pages) / p.aio_batches : 0.0;
    const uint64_t io_reads = p.aio_pages + p.sync_reads;
    const uint64_t io_wait_ns = p.aio_wait_ns + p.sync_read_ns;
    const double software_ns_per_read =
        io_reads ? static_cast<double>(p.software_ns()) / io_reads : 0.0;
    const uint64_t classified_sync_reads =
        p.neighbor_sync_reads + p.cross_page_sync_reads + p.post_aio_sync_reads;
    const uint64_t unclassified_sync_reads =
        p.sync_reads > classified_sync_reads
            ? p.sync_reads - classified_sync_reads
            : 0;
    const uint64_t classified_sync_read_ns = p.neighbor_sync_read_ns +
                                             p.cross_page_sync_read_ns +
                                             p.post_aio_sync_read_ns;
    const uint64_t unclassified_sync_read_ns =
        p.sync_read_ns > classified_sync_read_ns
            ? p.sync_read_ns - classified_sync_read_ns
            : 0;
    const double post_aio_sync_share =
        p.sync_reads ? static_cast<double>(p.post_aio_sync_reads) /
                           static_cast<double>(p.sync_reads)
                     : 0.0;
    const double post_aio_sync_time_share =
        p.sync_read_ns ? static_cast<double>(p.post_aio_sync_read_ns) /
                             static_cast<double>(p.sync_read_ns)
                       : 0.0;
    const double post_aio_missing_rate =
        p.post_aio_requested_unique_pages
            ? static_cast<double>(p.post_aio_missing_unique_pages) /
                  static_cast<double>(p.post_aio_requested_unique_pages)
            : 0.0;
    LOG_INFO(
        "VecBufferPool io_profile: file[%s] queries=%llu query_wall_ns=%llu "
        "aio_submit_ns=%llu aio_wait_ns=%llu aio_install_ns=%llu "
        "aio_page_lock_wait_ns=%llu sync_prepare_ns=%llu sync_read_ns=%llu "
        "sync_install_ns=%llu sync_page_lock_wait_ns=%llu sync_reads=%llu "
        "epoch_transition_ns=%llu fallback_total_ns=%llu copy_ns=%llu "
        "aio_batches=%llu aio_pages=%llu "
        "aio_avg_batch=%.2f aio_max_batch=%llu fallback_batches=%llu "
        "fallback_items=%llu epoch_enter_attempts=%llu "
        "epoch_enter_failures=%llu epoch_suspends=%llu "
        "io_reads=%llu io_wait_ns=%llu software_ns=%llu "
        "software_ns_per_read=%.2f",
        file_name_.c_str(), static_cast<unsigned long long>(p.query_count),
        static_cast<unsigned long long>(p.query_wall_ns),
        static_cast<unsigned long long>(p.aio_submit_ns),
        static_cast<unsigned long long>(p.aio_wait_ns),
        static_cast<unsigned long long>(p.aio_install_ns),
        static_cast<unsigned long long>(p.aio_page_lock_wait_ns),
        static_cast<unsigned long long>(p.sync_prepare_ns),
        static_cast<unsigned long long>(p.sync_read_ns),
        static_cast<unsigned long long>(p.sync_install_ns),
        static_cast<unsigned long long>(p.sync_page_lock_wait_ns),
        static_cast<unsigned long long>(p.sync_reads),
        static_cast<unsigned long long>(p.epoch_transition_ns),
        static_cast<unsigned long long>(p.fallback_total_ns),
        static_cast<unsigned long long>(p.copy_ns),
        static_cast<unsigned long long>(p.aio_batches),
        static_cast<unsigned long long>(p.aio_pages), avg_batch,
        static_cast<unsigned long long>(p.aio_max_batch),
        static_cast<unsigned long long>(p.fallback_batches),
        static_cast<unsigned long long>(p.fallback_items),
        static_cast<unsigned long long>(p.epoch_enter_attempts),
        static_cast<unsigned long long>(p.epoch_enter_failures),
        static_cast<unsigned long long>(p.epoch_suspends),
        static_cast<unsigned long long>(io_reads),
        static_cast<unsigned long long>(io_wait_ns),
        static_cast<unsigned long long>(p.software_ns()), software_ns_per_read);
    LOG_INFO(
        "VecBufferPool io_profile_paths: file[%s] "
        "neighbor_sync_reads=%llu neighbor_sync_read_ns=%llu "
        "cross_page_sync_reads=%llu cross_page_sync_read_ns=%llu "
        "post_aio_sync_reads=%llu post_aio_sync_read_ns=%llu "
        "post_aio_sync_share=%.4f post_aio_sync_time_share=%.4f "
        "unclassified_sync_reads=%llu unclassified_sync_read_ns=%llu "
        "vector_prefetch_aio_pages=%llu vector_prefetch_aio_wait_ns=%llu "
        "vector_fallback_aio_pages=%llu vector_fallback_aio_wait_ns=%llu "
        "post_aio_publish_attempts=%llu post_aio_publish_failures=%llu "
        "post_aio_requested_unique_pages=%llu "
        "post_aio_missing_unique_pages=%llu post_aio_missing_rate=%.4f",
        file_name_.c_str(),
        static_cast<unsigned long long>(p.neighbor_sync_reads),
        static_cast<unsigned long long>(p.neighbor_sync_read_ns),
        static_cast<unsigned long long>(p.cross_page_sync_reads),
        static_cast<unsigned long long>(p.cross_page_sync_read_ns),
        static_cast<unsigned long long>(p.post_aio_sync_reads),
        static_cast<unsigned long long>(p.post_aio_sync_read_ns),
        post_aio_sync_share, post_aio_sync_time_share,
        static_cast<unsigned long long>(unclassified_sync_reads),
        static_cast<unsigned long long>(unclassified_sync_read_ns),
        static_cast<unsigned long long>(p.vector_prefetch_aio_pages),
        static_cast<unsigned long long>(p.vector_prefetch_aio_wait_ns),
        static_cast<unsigned long long>(p.vector_fallback_aio_pages),
        static_cast<unsigned long long>(p.vector_fallback_aio_wait_ns),
        static_cast<unsigned long long>(p.post_aio_publish_attempts),
        static_cast<unsigned long long>(p.post_aio_publish_failures),
        static_cast<unsigned long long>(p.post_aio_requested_unique_pages),
        static_cast<unsigned long long>(p.post_aio_missing_unique_pages),
        post_aio_missing_rate);
  }
}

}  // namespace ailego
}  // namespace zvec
