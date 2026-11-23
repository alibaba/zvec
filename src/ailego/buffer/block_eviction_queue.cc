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
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <ailego/utility/mobile_memory.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif
#endif
#if defined(__linux__) && !defined(__ANDROID__)
#include <sys/auxv.h>
#include <fcntl.h>
#include <pthread.h>
#endif
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

namespace zvec {
namespace ailego {

namespace {

constexpr size_t kMinimumPageBytes = 4096UL;
constexpr size_t kSlabBytes = MemoryLimitPool::slab_size();
constexpr size_t kSlabAlignment = MemoryLimitPool::slab_alignment();
constexpr size_t kMaxSlabDataPages = kSlabBytes / kMinimumPageBytes - 1;

constexpr size_t kQueueSmallBytes = 4096;
constexpr size_t kQueueSlabBytes = 64UL << 10;
constexpr size_t kQueueAlignment = alignof(std::max_align_t);
thread_local size_t *initial_queue_backing = nullptr;

static_assert(kQueueSmallBytes % kQueueAlignment == 0,
              "queue size classes must preserve maximum alignment");

void *map_queue_memory(size_t bytes) {
#if defined(_WIN32)
  return ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE);
#else
  void *data = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return data == MAP_FAILED ? nullptr : data;
#endif
}

bool unmap_queue_memory(void *data, size_t bytes) {
#if defined(_WIN32)
  (void)bytes;
  return ::VirtualFree(data, 0, MEM_RELEASE) != 0;
#else
  return ::munmap(data, bytes) == 0;
#endif
}

static_assert((kSlabAlignment & (kSlabAlignment - 1)) == 0,
              "slab alignment must be a power of two");
static_assert(kSlabBytes == kSlabAlignment,
              "slab owner lookup relies on size matching alignment");
static_assert(kMaxSlabDataPages <= UINT16_MAX,
              "slab page indexes must fit in uint16_t");

struct AlignedSlabMapping {
  char *base{nullptr};
  void *reservation_base{nullptr};
  size_t reservation_size{0};
};

AlignedSlabMapping reserve_aligned_slab(size_t page_size) {
  constexpr size_t kReservationBytes = kSlabBytes + kSlabAlignment;
#if defined(_WIN32)
  void *reservation =
      ::VirtualAlloc(nullptr, kReservationBytes, MEM_RESERVE, PAGE_READWRITE);
  if (reservation == nullptr) {
    return {};
  }
  const uintptr_t raw = reinterpret_cast<uintptr_t>(reservation);
  const uintptr_t aligned = (raw + kSlabAlignment - 1) & ~(kSlabAlignment - 1);
  void *header = ::VirtualAlloc(reinterpret_cast<void *>(aligned), page_size,
                                MEM_COMMIT, PAGE_READWRITE);
  if (header == nullptr) {
    ::VirtualFree(reservation, 0, MEM_RELEASE);
    return {};
  }
  return {reinterpret_cast<char *>(aligned), reservation, kReservationBytes};
#else
  (void)page_size;
  void *reservation = ::mmap(nullptr, kReservationBytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation == MAP_FAILED) {
    return {};
  }

  const uintptr_t raw = reinterpret_cast<uintptr_t>(reservation);
  const uintptr_t aligned = (raw + kSlabAlignment - 1) & ~(kSlabAlignment - 1);
  const size_t prefix = aligned - raw;
  const size_t suffix = kReservationBytes - prefix - kSlabBytes;
  if (prefix != 0 && ::munmap(reservation, prefix) != 0) {
    ::munmap(reservation, kReservationBytes);
    return {};
  }
  if (suffix != 0 &&
      ::munmap(reinterpret_cast<void *>(aligned + kSlabBytes), suffix) != 0) {
    ::munmap(reinterpret_cast<void *>(aligned), kSlabBytes + suffix);
    return {};
  }
  return {reinterpret_cast<char *>(aligned), reinterpret_cast<void *>(aligned),
          kSlabBytes};
#endif
}

bool commit_slab_page(char *page, size_t page_size) {
#if defined(_WIN32)
  return ::VirtualAlloc(page, page_size, MEM_COMMIT, PAGE_READWRITE) == page;
#else
  (void)page;
  (void)page_size;
  return true;
#endif
}

bool discard_slab_page(char *page, size_t page_size) {
#if defined(_WIN32)
  return ::VirtualFree(page, page_size, MEM_DECOMMIT) != 0;
#elif defined(__APPLE__)
  // MADV_FREE_REUSABLE is only lazy reclamation: it can leave resident pages
  // behind after their budget has been spent by another consumer. Replace
  // this exclusively owned, free page with fresh demand-zero backing before
  // refunding it. MAP_FIXED preserves the slab's address reservation without
  // an unmap/remap gap; never apply it to a live or pinned cache page.
  return ::mmap(page, page_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
#else
  return ::madvise(page, page_size, MADV_DONTNEED) == 0;
#endif
}

}  // namespace

struct alignas(std::max_align_t) MemoryLimitPool::QueueAllocationHeader {
  size_t bytes{0};
  union {
    QueueMemorySlab *owner;
    QueueAllocationHeader *next_free;
  };
};

struct MemoryLimitPool::QueueResidentMapping {
  QueueResidentMapping *previous{nullptr};
  QueueResidentMapping *next{nullptr};
  void *data{nullptr};
  size_t bytes{0};
};

struct MemoryLimitPool::QueueMappingHeader {
  QueueResidentMapping mapping;
  ProcessRssBudget::MaterializationStamp stamp{};
  QueueAllocationHeader allocation{};
};

struct MemoryLimitPool::QueueMemorySlab {
  QueueMemorySlab(size_t allocation_size, size_t region_bytes)
      : slot_size(allocation_size),
        mapping_bytes(region_bytes),
        next_unused((sizeof(QueueMemorySlab) + kQueueAlignment - 1) &
                    ~(kQueueAlignment - 1)) {}

  bool has_space() const {
    return free_head != nullptr || slot_size <= mapping_bytes - next_unused;
  }
  void link(QueueMemorySlab *&head) {
    previous = nullptr;
    next = head;
    if (head) head->previous = this;
    head = this;
  }
  void unlink(QueueMemorySlab *&head) {
    if (previous)
      previous->next = next;
    else
      head = next;
    if (next) next->previous = previous;
    previous = next = nullptr;
  }

  QueueResidentMapping mapping;
  QueueMemorySlab *previous{nullptr};
  QueueMemorySlab *next{nullptr};
  size_t slot_size;
  size_t mapping_bytes;
  size_t next_unused;
  size_t live_slots{0};
  QueueAllocationHeader *free_head{nullptr};
  ProcessRssBudget::MappingProjection projection{};
  bool has_lifetime_slot{false};
};

void *EvictionQueueTraits::malloc(size_t bytes) {
  return MemoryLimitPool::get_instance().allocate_queue_memory(bytes);
}

void EvictionQueueTraits::free(void *ptr) {
  MemoryLimitPool::get_instance().free_queue_memory(ptr);
}

void *BlockEvictionQueue::ProcessQueueTraits::malloc_lifetime(size_t bytes) {
  // Process-wide queues keep their producers and indexes until process exit.
  return MemoryLimitPool::get_instance().allocate_queue_memory(bytes, true);
}

BlockEvictionQueue::BlockEvictionQueue() {
  // Only the queues' initial block pools are allocated through the traits
  // during construction. They stay mapped for the lifetime of this instance.
  auto *previous =
      std::exchange(initial_queue_backing, &initial_backing_bytes_);
  auto restore =
      ScopeGuard::Make([previous] { initial_queue_backing = previous; });
  for (size_t i = 0; i < kQueueCount; ++i) {
    evict_queues_.emplace_back(kEvictQueueCapacity);
  }
}

void MemoryLimitPool::register_queue_mapping(QueueResidentMapping *mapping,
                                             void *data, size_t bytes) {
  mapping->data = data;
  mapping->bytes = bytes;
  auto *next = queue_mappings_;
  QueueResidentMapping *previous = nullptr;
  const auto address = reinterpret_cast<uintptr_t>(data);
  while (next && reinterpret_cast<uintptr_t>(next->data) < address) {
    previous = next;
    next = next->next;
  }
  mapping->previous = previous;
  mapping->next = next;
  if (previous)
    previous->next = mapping;
  else
    queue_mappings_ = mapping;
  if (next) next->previous = mapping;
}

bool MemoryLimitPool::unmap_queue_mapping(QueueResidentMapping *mapping) {
  auto *previous = mapping->previous;
  auto *next = mapping->next;
  // The directory node is inside this mapping. Save its links before unmap,
  // and retain both registration and charge if the OS refuses the unmap.
  if (!unmap_queue_memory(mapping->data, mapping->bytes)) return false;
  if (previous)
    previous->next = next;
  else
    queue_mappings_ = next;
  if (next) next->previous = previous;
  return true;
}

void *MemoryLimitPool::allocate_queue_memory(size_t bytes, bool lifetime) {
  if (bytes > std::numeric_limits<size_t>::max() -
                  sizeof(QueueAllocationHeader) - (kQueueAlignment - 1)) {
    return nullptr;
  }
  const size_t total =
      (sizeof(QueueAllocationHeader) + bytes + kQueueAlignment - 1) &
      ~(kQueueAlignment - 1);
  std::lock_guard<std::mutex> lock(queue_memory_mutex_);
  const bool charged = pool_size_.load(std::memory_order_relaxed) != 0;
  const auto reserve = [this, charged](size_t size) {
    if (size > std::numeric_limits<size_t>::max() -
                   queue_allocated_size_.load(std::memory_order_relaxed)) {
      return false;
    }
    if (charged && !try_reserve_allocation(size, Backing::kPending))
      return false;
    if (charged) {
      metadata_used_size_.fetch_add(size, std::memory_order_relaxed);
      queue_charged_size_.fetch_add(size, std::memory_order_relaxed);
    }
    return true;
  };
  const auto rollback = [this, charged](size_t size) {
    if (charged) {
      queue_charged_size_.fetch_sub(size, std::memory_order_relaxed);
      release_fixed(size, &metadata_used_size_, Backing::kPending);
    }
  };
  const auto denied = [this]() -> void * {
    // Do not call try_charge_metadata here: that can recurse into this same
    // queue with owner/lifecycle locks held. Existing enqueue-failure recovery
    // retries registration after capacity becomes available.
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
    return nullptr;
  };
  const size_t page = page_buffer_size();
  if (total <= kQueueSmallBytes) {
    const size_t index = total / kQueueAlignment - 1;
    auto *slab = queue_small_slabs_[index];
    if (charged && slab && !slab->free_head) {
      // All queue mappings retain their full lifetime charge, including
      // slots not yet touched. Permanent native image/stack allowances never
      // exclude a queue mapping from this ledger.
      const size_t previous_page = (slab->next_unused - 1) / page;
      const size_t last_page = (slab->next_unused + total - 1) / page;
      if (last_page != previous_page &&
          !process_rss_.try_extend_mapping((last_page - previous_page) * page,
                                           slab->mapping_bytes,
                                           slab->projection))
        return denied();
    }
    if (!slab) {
      const size_t slab_header =
          (sizeof(QueueMemorySlab) + kQueueAlignment - 1) &
          ~(kQueueAlignment - 1);
      const size_t minimum = (slab_header + total + page - 1) & ~(page - 1);
      size_t region_bytes = std::max(kQueueSlabBytes, page);
      if (!reserve(region_bytes)) {
        region_bytes = minimum;
        if (region_bytes >= std::max(kQueueSlabBytes, page) ||
            !reserve(region_bytes)) {
          return denied();
        }
      }
      void *raw = map_queue_memory(region_bytes);
      if (!raw) {
        rollback(region_bytes);
        return nullptr;
      }
      slab = new (raw) QueueMemorySlab(total, region_bytes);
      if (charged) {
        slab->projection = ProcessRssBudget::MappingProjection::Full(
            region_bytes, process_rss_.finish_reservation(region_bytes, true));
      }
      register_queue_mapping(&slab->mapping, raw, region_bytes);
      slab->link(queue_small_slabs_[index]);
      queue_allocated_size_.fetch_add(region_bytes, std::memory_order_relaxed);
    }
    QueueAllocationHeader *header = slab->free_head;
    if (header) {
      slab->free_head = header->next_free;
    } else {
      header = new (reinterpret_cast<char *>(slab) + slab->next_unused)
          QueueAllocationHeader{};
      slab->next_unused += total;
    }
    header->bytes = total;
    header->owner = slab;
    ++slab->live_slots;
    if (lifetime && !slab->has_lifetime_slot) {
      slab->has_lifetime_slot = true;
      queue_runtime_lifetime_size_.fetch_add(slab->mapping_bytes,
                                             std::memory_order_relaxed);
    }
    if (!slab->has_space()) slab->unlink(queue_small_slabs_[index]);
    return header + 1;
  }
  static_assert(offsetof(QueueMappingHeader, allocation) +
                    sizeof(QueueAllocationHeader) ==
                sizeof(QueueMappingHeader));
  constexpr size_t prefix = offsetof(QueueMappingHeader, allocation);
  if (total > std::numeric_limits<size_t>::max() - prefix - (page - 1)) {
    return nullptr;
  }
  const size_t region_bytes = (prefix + total + page - 1) & ~(page - 1);
  if (!reserve(region_bytes)) return denied();
  void *raw = map_queue_memory(region_bytes);
  if (!raw) {
    rollback(region_bytes);
    return nullptr;
  }
  auto *mapping = new (raw) QueueMappingHeader{};
  if (charged)
    mapping->stamp = process_rss_.finish_reservation(region_bytes, true);
  register_queue_mapping(&mapping->mapping, raw, region_bytes);
  auto *header = &mapping->allocation;
  header->bytes = region_bytes;
  header->owner = nullptr;
  queue_allocated_size_.fetch_add(region_bytes, std::memory_order_relaxed);
  if (initial_queue_backing) *initial_queue_backing += region_bytes;
  if (lifetime) {
    queue_runtime_lifetime_size_.fetch_add(region_bytes,
                                           std::memory_order_relaxed);
  }
  return header + 1;
}

void MemoryLimitPool::free_queue_memory(void *ptr) {
  if (!ptr) {
    return;
  }
  auto *header = static_cast<QueueAllocationHeader *>(ptr) - 1;
  std::lock_guard<std::mutex> lock(queue_memory_mutex_);
  size_t bytes = header->bytes;
  ProcessRssBudget::MappingProjection projection;
  if (auto *slab = header->owner) {
    const size_t index = slab->slot_size / kQueueAlignment - 1;
    const bool was_full = !slab->has_space();
    if (--slab->live_slots != 0) {
      header->next_free = slab->free_head;
      slab->free_head = header;
      if (was_full) slab->link(queue_small_slabs_[index]);
      return;
    }
    bytes = slab->mapping_bytes;
    projection = slab->projection;
    auto *previous = slab->previous;
    auto *next = slab->next;
    // Save links before unmapping; do not access any slab/header byte after
    // releasing the region. Other live neighbors always retain the charge.
    if (!unmap_queue_mapping(&slab->mapping)) {
      header->next_free = slab->free_head;
      slab->free_head = header;
      if (was_full) slab->link(queue_small_slabs_[index]);
      LOG_ERROR("Failed to unmap queue slab; backing remains charged");
      return;
    }
    if (!was_full) {
      if (previous)
        previous->next = next;
      else
        queue_small_slabs_[index] = next;
      if (next) next->previous = previous;
    }
  } else {
    auto *mapping = reinterpret_cast<QueueMappingHeader *>(
        reinterpret_cast<char *>(header) -
        offsetof(QueueMappingHeader, allocation));
    projection =
        ProcessRssBudget::MappingProjection::Full(bytes, mapping->stamp);
    if (!unmap_queue_mapping(&mapping->mapping)) {
      LOG_ERROR("Failed to unmap queue allocation; backing remains charged");
      return;
    }
  }
  queue_allocated_size_.fetch_sub(bytes, std::memory_order_relaxed);
  if (pool_size_.load(std::memory_order_relaxed) != 0) {
    queue_charged_size_.fetch_sub(bytes, std::memory_order_relaxed);
    release_fixed(bytes, &metadata_used_size_, Backing::kUnmapped, projection);
  }
}

struct MemoryLimitPool::ReclaimableSlab {
  static constexpr uint64_t kMagic = 0x5A564543534C4142ULL;

  ReclaimableSlab(MemoryLimitPool *pool, void *reservation,
                  size_t reservation_bytes, size_t system_page_size)
      : owner(pool),
        reservation_base(reservation),
        reservation_size(reservation_bytes),
        page_size(system_page_size),
        data_page_count(
            static_cast<uint16_t>(kSlabBytes / system_page_size - 1)) {}

  bool unmap_empty() {
    assert(committed_pages == 0);
    auto *pool = owner;
    auto *following = next;
    void *base = reservation_base;
    const size_t bytes = reservation_size;
    const size_t page = page_size;
    const auto projection = header_projection;
    this->~ReclaimableSlab();
    if (unmap_queue_memory(base, bytes)) return true;

    // The pages were discarded before teardown. Restore an empty slab so
    // the old pool can keep using its charged header and address range.
    auto *restored = new (this) ReclaimableSlab(pool, base, bytes, page);
    restored->next = following;
    restored->header_projection = projection;
    LOG_ERROR("MemoryLimitPool: failed to unmap an empty slab");
    return false;
  }

  uint64_t magic{kMagic};
  MemoryLimitPool *owner{nullptr};
  ReclaimableSlab *next{nullptr};
  void *reservation_base{nullptr};
  size_t reservation_size{0};
  size_t page_size{0};
  std::mutex mutex;
  uint16_t next_unused{1};
  uint16_t reclaimed_count{0};
  uint16_t data_page_count{0};
  size_t committed_pages{0};
  uint16_t reclaimed_pages[kMaxSlabDataPages]{};
  ProcessRssBudget::ReusablePages<kMaxSlabDataPages> projections;
  ProcessRssBudget::MaterializationStamp header_projection{};
};

struct MemoryLimitPool::RetainedBlock {
  RetainedBlock *next{nullptr};
  // Mappings are page aligned/sized; the otherwise unused low bit records
  // the logical owner without growing the existing 64-byte prefix.
  size_t size_and_metadata;
  ProcessRssBudget::MaterializationStamp projection;
  ProcessRssBudget::MaterializationStamp creation;
  RetainedBlock *mapped_prev{nullptr};
  RetainedBlock *mapped_next{nullptr};

  RetainedBlock(size_t bytes, bool metadata,
                ProcessRssBudget::MaterializationStamp stamp)
      : size_and_metadata(bytes | size_t{metadata}),
        projection(stamp),
        creation(stamp) {}
  size_t bytes() const {
    return size_and_metadata & ~size_t{1};
  }
  bool metadata() const {
    return (size_and_metadata & 1) != 0;
  }
  void set_metadata(bool metadata) {
    size_and_metadata = bytes() | size_t{metadata};
  }
};

ProcessRssBudget::ResidentSample MemoryLimitPool::read_slab_resident_sample() {
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
  const int saved_errno = errno;
  auto restore_errno = ScopeGuard::Make([&] { errno = saved_errno; });
  auto &pool = get_instance();
  // Allocators can enter RSS accounting while holding these locks. A sample
  // must never wait for the reverse order; absence of a witness stays safe.
  // POSIX try-lock also safely refuses a mutex already held by this thread;
  // std::mutex::try_lock has a stricter ownership precondition.
  if (::pthread_mutex_trylock(pool.slab_mutex_.native_handle()) != 0)
    return {MemoryHelper::SelfRSS(), 0};
  auto unlock_pool = ScopeGuard::Make(
      [&] { ::pthread_mutex_unlock(pool.slab_mutex_.native_handle()); });
  ReclaimableSlab *first_unlocked = pool.slabs_;
  auto unlock = ScopeGuard::Make([&] {
    for (auto *slab = pool.slabs_; slab != first_unlocked; slab = slab->next)
      ::pthread_mutex_unlock(slab->mutex.native_handle());
  });
  for (auto *slab = pool.slabs_; slab; slab = slab->next) {
    if (::pthread_mutex_trylock(slab->mutex.native_handle()) != 0)
      return {MemoryHelper::SelfRSS(), 0};
    first_unlocked = slab->next;
  }
  // Keep every sampled mapping alive through the ownership/RSS snapshot.
  // Cache allocation and unmapping never enter RSS accounting under this
  // lock, and a busy registry supplies no credit rather than waiting.
  if (::pthread_mutex_trylock(pool.cache_mapping_mutex_.native_handle()) != 0)
    return {MemoryHelper::SelfRSS(), 0};
  auto unlock_cache = ScopeGuard::Make([&] {
    ::pthread_mutex_unlock(pool.cache_mapping_mutex_.native_handle());
  });
  if (::pthread_mutex_trylock(pool.retained_mutex_.native_handle()) != 0)
    return {MemoryHelper::SelfRSS(), 0};
  auto unlock_retained = ScopeGuard::Make(
      [&] { ::pthread_mutex_unlock(pool.retained_mutex_.native_handle()); });
  if (::pthread_mutex_trylock(pool.queue_memory_mutex_.native_handle()) != 0)
    return {MemoryHelper::SelfRSS(), 0};
  auto unlock_queue = ScopeGuard::Make([&] {
    ::pthread_mutex_unlock(pool.queue_memory_mutex_.native_handle());
  });

  // The kernel page size also remains correct in logical-page-size tests.
#if defined(__APPLE__) && TARGET_OS_OSX
  const size_t native_page = static_cast<size_t>(::getpagesize());
#else
  const size_t native_page = ::getauxval(AT_PAGESZ);
#endif
  if (native_page < kMinimumPageBytes || native_page > kSlabBytes ||
      kSlabBytes % native_page != 0)
    return {MemoryHelper::SelfRSS(), 0};
#if defined(__linux__)
  const int fd = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {MemoryHelper::SelfRSS(), 0};
  auto close = ScopeGuard::Make([&] { ::close(fd); });
#endif
  const size_t before = MemoryHelper::SelfRSS();
  if (!before) return {};
  size_t resident = 0;
#if defined(__APPLE__) && TARGET_OS_OSX
  std::array<int, kSlabBytes / kMinimumPageBytes> entries;
  constexpr int kPrivatePresent =
      VM_PAGE_QUERY_PAGE_PRESENT | VM_PAGE_QUERY_PAGE_DIRTY;
  constexpr int kExcluded =
      VM_PAGE_QUERY_PAGE_EXTERNAL | VM_PAGE_QUERY_PAGE_PAGED_OUT |
      VM_PAGE_QUERY_PAGE_FICTITIOUS | VM_PAGE_QUERY_PAGE_REUSABLE;
#else
  std::array<uint64_t, kSlabBytes / kMinimumPageBytes> entries;
  constexpr uint64_t kPrivatePresent =
      (uint64_t{1} << 63) | (uint64_t{1} << 56);
#endif
  const auto read_entries = [&](uintptr_t address, size_t count) {
    if (count == 0 || count > entries.size()) return false;
#if defined(__APPLE__) && TARGET_OS_OSX
    mach_vm_size_t returned = count;
    return ::mach_vm_page_range_query(
               ::mach_task_self(), address, count * native_page,
               reinterpret_cast<mach_vm_address_t>(entries.data()),
               &returned) == KERN_SUCCESS &&
           returned == count;
#else
    const uintptr_t index = address / native_page;
    if (index > static_cast<uintptr_t>(std::numeric_limits<off_t>::max()) /
                    sizeof(uint64_t))
      return false;
    const off_t offset = static_cast<off_t>(index * sizeof(uint64_t));
    const size_t bytes = count * sizeof(uint64_t);
    size_t done = 0;
    while (done < bytes) {
      const ssize_t got =
          ::pread(fd, reinterpret_cast<char *>(entries.data()) + done,
                  bytes - done, offset + static_cast<off_t>(done));
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) return false;
      done += static_cast<size_t>(got);
    }
    return true;
#endif
  };
  const auto private_present = [&](size_t i) {
    bool present = (entries[i] & kPrivatePresent) == kPrivatePresent;
#if defined(__APPLE__) && TARGET_OS_OSX
    present = present && (entries[i] & kExcluded) == 0;
#endif
    return present;
  };
  for (auto *slab = pool.slabs_; slab; slab = slab->next) {
    if (slab->page_size < native_page || slab->page_size % native_page != 0)
      return {MemoryHelper::SelfRSS(), 0};
    const size_t count =
        (slab->committed_pages ? kSlabBytes : slab->page_size) / native_page;
    std::array<bool, kMaxSlabDataPages + 1> charged{};
    charged[0] = true;
    if (slab->committed_pages) {
      for (size_t i = 1; i < slab->next_unused; ++i) charged[i] = true;
      for (size_t i = 0; i < slab->reclaimed_count; ++i)
        charged[slab->reclaimed_pages[i]] = false;
    }
    if (!read_entries(reinterpret_cast<uintptr_t>(slab), count))
      return {MemoryHelper::SelfRSS(), 0};
    // Only charged private resident pages in our exact slab ranges supply
    // overlap credit. Reserved, discarded and swapped backing supplies none.
    for (size_t i = 0; i < count; ++i) {
      // THP can populate uncharged neighbors of a requested page. Their RSS
      // must not become overlap credit for a different, untouched owner.
      if (charged[i * native_page / slab->page_size] && private_present(i))
        resident += native_page;
    }
  }
  // Chunk backing is independently mapped, and remains charged while idle.
  // Adjacent registered mappings can share a bounded kernel query. Reserved,
  // untouched or compressed pages still provide no overlap credit.
  for (auto *buffer = pool.cache_mappings_; buffer;) {
    uintptr_t begin = reinterpret_cast<uintptr_t>(buffer->data);
    if (begin % native_page || buffer->bytes % native_page ||
        buffer->bytes > std::numeric_limits<uintptr_t>::max() - begin)
      return {MemoryHelper::SelfRSS(), 0};
    uintptr_t end = begin + buffer->bytes;
    buffer = buffer->mapped_next;
    while (buffer && reinterpret_cast<uintptr_t>(buffer->data) == end) {
      if (buffer->bytes % native_page ||
          buffer->bytes > std::numeric_limits<uintptr_t>::max() - end)
        return {MemoryHelper::SelfRSS(), 0};
      end += buffer->bytes;
      buffer = buffer->mapped_next;
    }
    while (begin < end) {
      const size_t count =
          std::min<size_t>((end - begin) / native_page, entries.size());
      if (!read_entries(begin, count)) return {MemoryHelper::SelfRSS(), 0};
      for (size_t i = 0; i < count; ++i)
        if (private_present(i)) resident += native_page;
      begin += count * native_page;
    }
  }
  for (auto *block = pool.retained_mappings_; block;
       block = block->mapped_next) {
    uintptr_t begin = reinterpret_cast<uintptr_t>(block);
    const size_t bytes = block->bytes();
    if (begin % native_page || bytes % native_page ||
        bytes > std::numeric_limits<uintptr_t>::max() - begin)
      return {MemoryHelper::SelfRSS(), 0};
    const uintptr_t end = begin + bytes;
    while (begin < end) {
      const size_t count =
          std::min<size_t>((end - begin) / native_page, entries.size());
      if (!read_entries(begin, count)) return {MemoryHelper::SelfRSS(), 0};
      for (size_t i = 0; i < count; ++i)
        if (private_present(i)) resident += native_page;
      begin += count * native_page;
    }
  }
  // Queue blocks and producer metadata are charged anonymous mappings too.
  // Their private resident pages can witness overlap; untouched slots cannot.
  for (auto *mapping = pool.queue_mappings_; mapping; mapping = mapping->next) {
    uintptr_t begin = reinterpret_cast<uintptr_t>(mapping->data);
    const size_t bytes = mapping->bytes;
    if (begin % native_page || bytes % native_page ||
        bytes > std::numeric_limits<uintptr_t>::max() - begin)
      return {MemoryHelper::SelfRSS(), 0};
    const uintptr_t end = begin + bytes;
    while (begin < end) {
      const size_t count =
          std::min<size_t>((end - begin) / native_page, entries.size());
      if (!read_entries(begin, count)) return {MemoryHelper::SelfRSS(), 0};
      for (size_t i = 0; i < count; ++i)
        if (private_present(i)) resident += native_page;
      begin += count * native_page;
    }
  }
  const size_t rss = MemoryHelper::SelfRSS();
  if (!rss) return {};
  // Be conservative if unrelated process memory retired during this read.
  if (rss < before) resident -= std::min(resident, before - rss);
  return {rss, std::min(resident, rss)};
#else
  return {MemoryHelper::SelfRSS(), 0};
#endif
}

ProcessRssBudget::ResidentSample
MemoryLimitPool::read_background_resident_sample() {
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
  // Copy addresses and charged-page masks under short mapping locks. Native
  // queries use values only, never descriptors that an allocator may retire.
  // The budget token separately removes all retirement/cancellation credit.
  const int saved_errno = errno;
  auto restore_errno = ScopeGuard::Make([&] { errno = saved_errno; });
  auto &pool = get_instance();
  const size_t page = static_cast<size_t>(::getpagesize());
  if (page < kMinimumPageBytes || page > kSlabBytes || kSlabBytes % page)
    return {MemoryHelper::SelfRSS(), 0};
  const size_t before = MemoryHelper::SelfRSS();
  if (!before) return {};
  // A fixed 32 KiB Linux / 16 KiB Mach buffer amortizes queries of large
  // registered workspaces without allocating on the pressure path.
  constexpr size_t kQueryPages = 4096;
#if defined(__linux__)
  const int fd = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {MemoryHelper::SelfRSS(), 0};
  auto close = ScopeGuard::Make([&] { ::close(fd); });
  std::array<uint64_t, kQueryPages> entries;
  constexpr uint64_t present = (uint64_t{1} << 63) | (uint64_t{1} << 56);
  constexpr uint64_t excluded = 0;
#else
  std::array<int, kQueryPages> entries;
  constexpr int present = VM_PAGE_QUERY_PAGE_PRESENT | VM_PAGE_QUERY_PAGE_DIRTY;
  constexpr int excluded =
      VM_PAGE_QUERY_PAGE_EXTERNAL | VM_PAGE_QUERY_PAGE_PAGED_OUT |
      VM_PAGE_QUERY_PAGE_FICTITIOUS | VM_PAGE_QUERY_PAGE_REUSABLE;
#endif
  const auto read_range =
      [&](uintptr_t address, size_t bytes,
          const std::array<bool, kMaxSlabDataPages + 1> *charged,
          size_t logical_page) {
        if (address % page || bytes % page ||
            bytes > std::numeric_limits<uintptr_t>::max() - address)
          return size_t{0};
        size_t resident = 0;
        for (size_t offset = 0; offset < bytes;) {
          const size_t count =
              std::min((bytes - offset) / page, entries.size());
#if defined(__linux__)
          const uintptr_t index = (address + offset) / page;
          if (index >
              static_cast<uintptr_t>(std::numeric_limits<off_t>::max()) /
                  sizeof(uint64_t))
            return size_t{0};
          const off_t position = static_cast<off_t>(index * sizeof(uint64_t));
          const size_t requested = count * sizeof(uint64_t);
          size_t done = 0;
          while (done < requested) {
            const ssize_t got =
                ::pread(fd, reinterpret_cast<char *>(entries.data()) + done,
                        requested - done, position + static_cast<off_t>(done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) return size_t{0};
            done += static_cast<size_t>(got);
          }
#else
          mach_vm_size_t returned = count;
          if (::mach_vm_page_range_query(
                  ::mach_task_self(), address + offset, count * page,
                  reinterpret_cast<mach_vm_address_t>(entries.data()),
                  &returned) != KERN_SUCCESS ||
              returned != count)
            return size_t{0};
#endif
          for (size_t i = 0; i < count; ++i)
            if ((!charged || (*charged)[(offset + i * page) / logical_page]) &&
                (entries[i] & present) == present && !(entries[i] & excluded))
              resident += page;
          offset += count * page;
        }
        return resident;
      };
  size_t resident = 0;
  uintptr_t cursor = 0;
  ReclaimableSlab *next_slab = nullptr;
  {
    std::unique_lock<std::mutex> lock(pool.slab_mutex_, std::try_to_lock);
    if (lock) next_slab = pool.slabs_;
  }
  // Only the maintenance worker calls this unlocked scanner. init() joins
  // that worker before tearing down slab headers; ordinary page retirement
  // discards payloads but keeps their header/list nodes alive. Follow the
  // initial list once instead of searching the whole list for each address.
  // Concurrently prepended slabs supply no credit in this conservative scan.
  while (next_slab) {
    uintptr_t address = 0;
    size_t logical_page = 0, bytes = 0;
    std::array<bool, kMaxSlabDataPages + 1> charged{};
    {
      std::unique_lock<std::mutex> lock(pool.slab_mutex_, std::try_to_lock);
      // Keep the independent ranges already witnessed. A busy allocator
      // makes the remaining range unknown; it does not invalidate earlier
      // proof. Publication still subtracts all concurrent retirements.
      if (!lock) break;
      auto *next = next_slab;
      next_slab = next->next;
      std::unique_lock<std::mutex> slab_lock(next->mutex, std::try_to_lock);
      address = reinterpret_cast<uintptr_t>(next);
      if (!slab_lock) {
        if (address > std::numeric_limits<uintptr_t>::max() - kSlabBytes)
          return {MemoryHelper::SelfRSS(), 0};
        continue;
      }
      logical_page = next->page_size;
      if (logical_page < page || logical_page % page ||
          address > std::numeric_limits<uintptr_t>::max() - kSlabBytes)
        return {MemoryHelper::SelfRSS(), 0};
      bytes = next->committed_pages ? kSlabBytes : logical_page;
      charged[0] = true;
      if (next->committed_pages) {
        for (size_t i = 1; i < next->next_unused; ++i) charged[i] = true;
        for (size_t i = 0; i < next->reclaimed_count; ++i)
          charged[next->reclaimed_pages[i]] = false;
      }
    }
    resident += read_range(address, bytes, &charged, logical_page);
  }
  struct Range {
    uintptr_t address;
    size_t bytes;
  };
  std::array<Range, 128> ranges;
  cursor = 0;
  size_t mappings_left = 0;
  {
    std::unique_lock<std::mutex> lock(pool.cache_mapping_mutex_,
                                      std::try_to_lock);
    if (lock)
      for (auto *mapping = pool.cache_mappings_; mapping;
           mapping = mapping->mapped_next)
        ++mappings_left;
  }
  while (mappings_left != 0) {
    size_t count = 0;
    {
      std::unique_lock<std::mutex> lock(pool.cache_mapping_mutex_,
                                        std::try_to_lock);
      if (!lock) break;
      auto *mapping = pool.cache_mappings_;
      while (mapping && reinterpret_cast<uintptr_t>(mapping->data) < cursor)
        mapping = mapping->mapped_next;
      for (size_t copied = 0;
           mapping && mappings_left && copied < ranges.size();
           ++copied, --mappings_left) {
        const auto address = reinterpret_cast<uintptr_t>(mapping->data);
        if (mapping->bytes > std::numeric_limits<uintptr_t>::max() - address)
          return {MemoryHelper::SelfRSS(), 0};
        if (count &&
            ranges[count - 1].address + ranges[count - 1].bytes == address)
          ranges[count - 1].bytes += mapping->bytes;
        else
          ranges[count++] = {address, mapping->bytes};
        cursor = address + mapping->bytes;
        mapping = mapping->mapped_next;
      }
    }
    if (!count) break;
    for (size_t i = 0; i < count; ++i)
      resident += read_range(ranges[i].address, ranges[i].bytes, nullptr, page);
  }
  // Retained metadata/workspace arrays use separate anonymous mappings.
  // Snapshot their charged ranges too; counting only page slabs and cache
  // chunks loses the stable metadata overlap after page-cache churn.
  cursor = 0;
  size_t retained_left = 0;
  {
    std::unique_lock<std::mutex> lock(pool.retained_mutex_, std::try_to_lock);
    if (lock)
      for (auto *block = pool.retained_mappings_; block;
           block = block->mapped_next)
        ++retained_left;
  }
  while (retained_left != 0) {
    size_t count = 0;
    {
      std::unique_lock<std::mutex> lock(pool.retained_mutex_, std::try_to_lock);
      if (!lock) break;
      auto *block = pool.retained_mappings_;
      while (block && reinterpret_cast<uintptr_t>(block) < cursor)
        block = block->mapped_next;
      for (size_t copied = 0; block && retained_left && copied < ranges.size();
           ++copied, --retained_left) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(block);
        const size_t bytes = block->bytes();
        if (bytes > std::numeric_limits<uintptr_t>::max() - address)
          return {MemoryHelper::SelfRSS(), 0};
        if (count &&
            ranges[count - 1].address + ranges[count - 1].bytes == address)
          ranges[count - 1].bytes += bytes;
        else
          ranges[count++] = {address, bytes};
        cursor = address + bytes;
        block = block->mapped_next;
      }
    }
    if (!count) break;
    for (size_t i = 0; i < count; ++i)
      resident += read_range(ranges[i].address, ranges[i].bytes, nullptr, page);
  }
  cursor = 0;
  size_t queue_left = 0;
  {
    std::unique_lock<std::mutex> lock(pool.queue_memory_mutex_,
                                      std::try_to_lock);
    if (lock)
      for (auto *mapping = pool.queue_mappings_; mapping;
           mapping = mapping->next)
        ++queue_left;
  }
  while (queue_left != 0) {
    size_t count = 0;
    {
      std::unique_lock<std::mutex> lock(pool.queue_memory_mutex_,
                                        std::try_to_lock);
      if (!lock) break;
      auto *mapping = pool.queue_mappings_;
      while (mapping && reinterpret_cast<uintptr_t>(mapping->data) < cursor)
        mapping = mapping->next;
      for (size_t copied = 0; mapping && queue_left && copied < ranges.size();
           ++copied, --queue_left) {
        const auto address = reinterpret_cast<uintptr_t>(mapping->data);
        if (mapping->bytes > std::numeric_limits<uintptr_t>::max() - address)
          return {MemoryHelper::SelfRSS(), 0};
        if (count &&
            ranges[count - 1].address + ranges[count - 1].bytes == address)
          ranges[count - 1].bytes += mapping->bytes;
        else
          ranges[count++] = {address, mapping->bytes};
        cursor = address + mapping->bytes;
        mapping = mapping->next;
      }
    }
    if (!count) break;
    for (size_t i = 0; i < count; ++i)
      resident += read_range(ranges[i].address, ranges[i].bytes, nullptr, page);
  }
  const size_t rss = MemoryHelper::SelfRSS();
  if (!rss) return {};
  if (rss < before) resident -= std::min(resident, before - rss);
  return {rss, std::min(resident, rss)};
#else
  return {MemoryHelper::SelfRSS(), 0};
#endif
}

void MemoryLimitPool::register_retained_mapping(RetainedBlock *block) {
  std::lock_guard<std::mutex> lock(retained_mutex_);
  auto *next = retained_mappings_;
  RetainedBlock *previous = nullptr;
  const uintptr_t address = reinterpret_cast<uintptr_t>(block);
  while (next && reinterpret_cast<uintptr_t>(next) < address) {
    previous = next;
    next = next->mapped_next;
  }
  block->mapped_prev = previous;
  block->mapped_next = next;
  if (previous)
    previous->mapped_next = block;
  else
    retained_mappings_ = block;
  if (next) next->mapped_prev = block;
}

bool MemoryLimitPool::unmap_retained_mapping(RetainedBlock *block) {
  auto *previous = block->mapped_prev;
  auto *next = block->mapped_next;
#if defined(_WIN32)
  const bool unmapped = ::VirtualFree(block, 0, MEM_RELEASE) != 0;
#else
  const bool unmapped = ::munmap(block, block->bytes()) == 0;
#endif
  if (!unmapped) return false;
  if (previous)
    previous->mapped_next = next;
  else
    retained_mappings_ = next;
  if (next) next->mapped_prev = previous;
  return true;
}

size_t MemoryLimitPool::retained_block_size(size_t bytes) {
  const size_t page = page_buffer_size();
  if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() -
                                kRetainedBlockHeaderBytes - (page - 1)) {
    return 0;
  }
  return ((bytes + kRetainedBlockHeaderBytes + page - 1) / page) * page;
}

void *MemoryLimitPool::acquire_retained_block(size_t bytes, bool metadata) {
  static_assert(sizeof(RetainedBlock) <= kRetainedBlockHeaderBytes,
                "retained block header must fit its reserved prefix");
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  const size_t charge = retained_block_size(bytes);
  if (charge == 0 || capacity() == 0) return nullptr;
  auto *counter = metadata ? &metadata_used_size_ : &external_used_size_;
  {
    std::lock_guard<std::mutex> lock(retained_mutex_);
    for (auto **link = &retained_free_; *link; link = &(*link)->next) {
      auto *block = *link;
      if (block->bytes() != charge) continue;
      // An idle mapping can contain pages that were never faulted in or were
      // reclaimed by the OS. Assigning a new owner can make them resident.
      // Backing is already charged, but a pending page may own the logical
      // headroom. Leave the idle block linked until admission succeeds.
      if (!try_reserve_used(charge)) break;
      if (!process_rss_.try_materialize_reusable(charge, block->projection)) {
        release_buffer(nullptr, charge);
        break;
      }
      *link = block->next;
      block->next = nullptr;
      block->set_metadata(metadata);
      retained_idle_bytes_.fetch_sub(charge, std::memory_order_relaxed);
      counter->fetch_add(charge, std::memory_order_relaxed);
      retained_block_reuses_.fetch_add(1, std::memory_order_relaxed);
      return reinterpret_cast<char *>(block) + kRetainedBlockHeaderBytes;
    }
  }
  // No allocator lock may be held across eviction or writeback callbacks.
  if (!try_charge_fixed_with_wait(charge, counter, Backing::kPending))
    return nullptr;
#if defined(_WIN32)
  void *raw =
      ::VirtualAlloc(nullptr, charge, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
  void *raw = ::mmap(nullptr, charge, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) raw = nullptr;
#endif
  if (!raw) {
    release_fixed(charge, counter, Backing::kPending);
    return nullptr;
  }
  const auto projection = process_rss_.finish_reservation(charge, true);
  auto *block = new (raw) RetainedBlock(charge, metadata, projection);
  retained_block_bytes_.fetch_add(charge, std::memory_order_relaxed);
  register_retained_mapping(block);
  return reinterpret_cast<char *>(block) + kRetainedBlockHeaderBytes;
}

bool MemoryLimitPool::try_materialize_retained_block(void *data,
                                                     bool required) {
  auto *block = reinterpret_cast<RetainedBlock *>(static_cast<char *>(data) -
                                                  kRetainedBlockHeaderBytes);
  const size_t bytes = block->bytes();
  const auto admit = [&](AdmissionDenials *denials) {
    return process_rss_.try_materialize_reusable(
        bytes, block->projection, denials ? &denials->rss : nullptr,
        denials ? &denials->rss_watch : nullptr);
  };
  if (!required) return admit(nullptr);
  // The caller pins the chunk, but must not hold its allocator mutex while
  // maintenance proves residency or invokes a reclaimer that can use it.
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_admit({0, 0}, admit, true);
}

void MemoryLimitPool::release_retained_block(void *data) {
  if (!data) return;
  auto *block = reinterpret_cast<RetainedBlock *>(static_cast<char *>(data) -
                                                  kRetainedBlockHeaderBytes);
  {
    std::lock_guard<std::mutex> lock(retained_mutex_);
    const size_t bytes = block->bytes();
    const auto projection = ProcessRssBudget::MappingProjection::ForReuse(
        bytes, block->projection, block->creation);
    auto *counter =
        block->metadata() ? &metadata_used_size_ : &external_used_size_;
    // Retained mappings share the idle reserve with free data pages. Returning
    // a burst of Arrow chunks must not leave excess backing resident until
    // paced background maintenance catches up. The owner has relinquished
    // this entire block; live/pinned allocations are never reclaimed here.
    const size_t committed = committed_size_.load(std::memory_order_relaxed);
    const size_t used = used_size_.load(std::memory_order_relaxed);
    const size_t idle = committed > used ? committed - used : 0;
    const size_t reserve =
        process_rss_.pressure() ? 0 : reserve_margin(capacity());
    bool unmapped = false;
    if (idle > reserve || bytes > reserve - idle) {
      unmapped = unmap_retained_mapping(block);
    }
    if (unmapped) {
      retained_block_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      release_committed(bytes, Backing::kUnmapped, projection);
    } else {
      // A failed physical release keeps the backing and its committed charge.
      block->next = retained_free_;
      retained_free_ = block;
      retained_idle_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    counter->fetch_sub(bytes, std::memory_order_relaxed);
    // Publish idle backing or finish unmapping before init() sees no owners.
    release_native_used(bytes);
  }
  capacity_cv_.notify_all();
}

size_t MemoryLimitPool::trim_retained_blocks(size_t bytes_needed) {
  size_t released = 0;
  std::lock_guard<std::mutex> lock(retained_mutex_);
  while (retained_free_ && released < bytes_needed) {
    auto *block = retained_free_;
    auto *next = block->next;
    const size_t bytes = block->bytes();
    const auto projection = ProcessRssBudget::MappingProjection::ForReuse(
        bytes, block->projection, block->creation);
    const bool unmapped = unmap_retained_mapping(block);
    if (!unmapped) {
      // A failed OS release must not create reusable budget.
      LOG_ERROR("MemoryLimitPool: failed to unmap retained backing block");
      break;
    }
    retained_free_ = next;
    retained_block_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    retained_idle_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    release_committed(bytes, Backing::kUnmapped, projection);
    released += bytes;
  }
  return released;
}

size_t MemoryLimitPool::page_buffer_size() {
  static const size_t page_size = []() -> size_t {
#if defined(_WIN32)
    SYSTEM_INFO info;
    ::GetSystemInfo(&info);
    return static_cast<size_t>(info.dwPageSize);
#else
    return static_cast<size_t>(::getpagesize());
#endif
  }();
  assert(page_size >= kMinimumPageBytes && page_size < kSlabBytes &&
         (page_size & (page_size - 1)) == 0);
  return page_size;
}

bool BlockEvictionQueue::dequeue_priority(size_t priority, BlockType &item) {
  // Tokenless dequeue can repeatedly choose the reclaiming thread's own
  // requeued pinned page and starve older producer streams forever. Consumer
  // tokens rotate across producers without allocating on the pressure path.
  thread_local std::array<std::optional<ProcessQueue::consumer_token_t>,
                          kQueueCount>
      consumers;
  if (!consumers[priority])
    consumers[priority].emplace(evict_queues_[priority]);
  if (!evict_queues_[priority].try_dequeue(*consumers[priority], item))
    return false;
  // Publish the dequeue after the reclaim pass has registered itself.
  approximate_queue_sizes_[priority].fetch_sub(1, std::memory_order_release);
  return true;
}

bool BlockEvictionQueue::evict_single_block(BlockType &item) {
  return evict_single_block(item, /*age_protected=*/false);
}

bool BlockEvictionQueue::evict_single_block(
    BlockType &item, bool age_protected,
    std::array<size_t, kQueueCount> *scan_remaining) {
  if (age_protected) {
    const size_t probation = approximate_queue_sizes_[kProbationPriority].load(
        std::memory_order_relaxed);
    const size_t protected_pages =
        approximate_queue_sizes_[kProtectedPriority].load(
            std::memory_order_relaxed);
    const size_t dominance_threshold =
        probation > (std::numeric_limits<size_t>::max() - 1) /
                        kProtectedDominanceRatio
            ? std::numeric_limits<size_t>::max()
            : (probation + 1) * kProtectedDominanceRatio;
    if (protected_pages > dominance_threshold &&
        (!scan_remaining || (*scan_remaining)[kProtectedPriority] != 0) &&
        dequeue_priority(kProtectedPriority, item)) {
      if (scan_remaining) --(*scan_remaining)[kProtectedPriority];
      protected_aging_dequeues_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  for (size_t i = 0; i < kQueueCount; i++) {
    if (scan_remaining && (*scan_remaining)[i] == 0) continue;
    if (dequeue_priority(i, item)) {
      if (scan_remaining) --(*scan_remaining)[i];
      return true;
    }
  }
  return false;
}

bool BlockEvictionQueue::evict_block(BlockType &item) {
  size_t attempts = 0;
  bool age_protected = true;
  return evict_block(item, attempts, std::numeric_limits<size_t>::max(),
                     age_protected);
}

bool BlockEvictionQueue::evict_block(
    BlockType &item, size_t &attempts, size_t max_attempts, bool &age_protected,
    std::array<size_t, kQueueCount> *scan_remaining) {
  while (attempts < max_attempts) {
    if (!evict_single_block(item, age_protected, scan_remaining)) {
      return false;
    }
    // Protected aging is deliberately a once-per-reclaim-batch decision.
    age_protected = false;
    ++attempts;
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner == nullptr ||
        valid_owners_.find(item.owner) == valid_owners_.end() ||
        item.owner->is_dead_block(item.owner_key, item.version)) {
      continue;
    }
    const uint8_t current_priority =
        item.owner->eviction_priority(item.owner_key);
    if (item.priority != current_priority) {
      item.priority = current_priority;
      if (!add_single_block(item, static_cast<int>(current_priority))) {
        item.owner->eviction_requeue_failed(item.owner_key, item.version);
      }
      continue;
    }
    return true;
  }
  return false;
}

void BlockEvictionQueue::recycle() {
  active_reclaim_passes_.fetch_add(1, std::memory_order_relaxed);
  auto finish = ScopeGuard::Make([this] {
    active_reclaim_passes_.fetch_sub(1, std::memory_order_release);
  });
  BlockType item;
  // A foreground page fault must not scan the whole global queue while its
  // page remains in kLoadingRefCount. Try a small CLOCK sample and let the
  // caller fall back or retry; background reclaim handles deep queue walks.
  static constexpr size_t kForegroundReclaimAttempts = 20;
  const size_t max_attempts = kForegroundReclaimAttempts;
  size_t attempts = 0;
  bool recovered = false;
  bool age_protected = true;
  // Page allocations can hit their admission limit before the process-wide
  // pool is full because part of the budget is reserved for external cache
  // consumers. Stop immediately after enough room for one page is reclaimed.
  while (MemoryLimitPool::get_instance().is_page_full() &&
         attempts < max_attempts) {
    if (!evict_block(item, attempts, max_attempts, age_protected)) {
      if (attempts >= max_attempts) {
        break;
      }
      if (recovered || recover_owner_queues() == 0) {
        break;
      }
      recovered = true;
      continue;
    }
    {
      std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
      if (item.owner != nullptr &&
          valid_owners_.find(item.owner) != valid_owners_.end() &&
          !item.owner->is_dead_block(item.owner_key, item.version)) {
        item.owner->evict_block(item.owner_key);
      }
    }
  }
}

size_t BlockEvictionQueue::recycle_inactive(size_t max_attempts) {
  // Sampling must progress even when ordinary pages are continuously
  // replenished. Collect the sample before requeueing so a hot/pinned entry
  // is not inspected twice in the same pass. Never scan a whole cache here.
  std::array<BlockType, 8> items;
  const size_t attempts = std::min(max_attempts, items.size());
  size_t found = 0;
  for (; found < attempts; ++found) {
    if (!dequeue_priority(kExplicitHotPriority, items[found])) break;
  }
  size_t evicted = 0;
  for (size_t i = 0; i < found; ++i) {
    const BlockType &item = items[i];
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner == nullptr || valid_owners_.count(item.owner) == 0 ||
        item.owner->is_dead_block(item.owner_key, item.version)) {
      continue;
    }
    const uint8_t priority = item.owner->eviction_priority(item.owner_key);
    if (priority == kExplicitHotPriority) {
      inactive_probes_.fetch_add(1, std::memory_order_relaxed);
      if (item.owner->evict_inactive_block(item.owner_key)) {
        ++evicted;
        inactive_evictions_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
    }
    if (!add_single_block(item, static_cast<int>(priority))) {
      item.owner->eviction_requeue_failed(item.owner_key, item.version);
    }
  }
  return evicted;
}

size_t BlockEvictionQueue::batch_recycle(size_t count) {
  return batch_recycle(count, count >= kProtectedAgingMinBatch);
}

size_t BlockEvictionQueue::batch_recycle(size_t count, bool age_protected) {
  return batch_recycle_impl(count, count, age_protected);
}

size_t BlockEvictionQueue::reclaim_for_allocation(size_t bytes) {
  if (!bytes) return 0;
  const size_t page = MemoryLimitPool::page_buffer_size();
  const size_t count =
      std::min<size_t>(256, bytes / page + (bytes % page != 0));
  // Keep the old mandatory-admission scan and protected aging behavior.
  return batch_recycle_impl(count, 256, true, bytes);
}

size_t BlockEvictionQueue::batch_recycle_impl(size_t count, size_t scan_count,
                                              bool age_protected,
                                              size_t bytes) {
  active_reclaim_passes_.fetch_add(1, std::memory_order_relaxed);
  auto finish = ScopeGuard::Make([this] {
    active_reclaim_passes_.fetch_sub(1, std::memory_order_release);
  });
  auto &pool = MemoryLimitPool::get_instance();
  const size_t before = pool.used();
  // Small foreground reservations preserve the existing cost preference.
  // Larger pressure batches give inactive decoded payloads bounded CLOCK
  // attention, rather than letting them monopolize the pool indefinitely.
  size_t evicted =
      age_protected ? recycle_inactive(std::min<size_t>(bytes ? 1 : 8, count))
                    : 0;
  size_t reclaimed = before - std::min(before, pool.used());
  // Bound work when no page is currently evictable.
  const size_t max_attempts =
      scan_count > (std::numeric_limits<size_t>::max() - 16) / 4
          ? std::numeric_limits<size_t>::max()
          : scan_count * 4 + 16;
  size_t attempts = 0;
  bool recovered = false;
  // Pinned entries are requeued. Do not spend the whole bounded scan on
  // those same low-priority entries while reclaimable decoded columns wait
  // in a higher-priority queue. Snapshot each tier and reserve scan work for
  // every tier; successful low-priority eviction still satisfies count first.
  std::array<size_t, kQueueCount> scan_remaining{};
  const auto reset_scan = [&] {
    size_t active_queues = 0;
    for (const auto &depth : approximate_queue_sizes_)
      active_queues += depth.load(std::memory_order_relaxed) != 0;
    const size_t per_queue =
        std::max<size_t>(1, max_attempts / std::max<size_t>(1, active_queues));
    for (size_t i = 0; i < kQueueCount; ++i) {
      const size_t depth = std::max<size_t>(
          1, approximate_queue_sizes_[i].load(std::memory_order_relaxed));
      // Allow CLOCK's second chance (and cross-tier promotion) within a
      // batch, without letting one pinned entry monopolize all scan work.
      scan_remaining[i] = depth >= per_queue / 2 ? per_queue : depth * 2;
    }
  };
  reset_scan();
  while (evicted < count && attempts < max_attempts) {
    if (bytes && reclaimed >= bytes) break;
    BlockType item;
    if (!evict_block(item, attempts, max_attempts, age_protected,
                     &scan_remaining)) {
      if (attempts >= max_attempts) {
        break;
      }
      if (recovered || recover_owner_queues() == 0) {
        break;
      }
      recovered = true;
      reset_scan();
      continue;
    }
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner != nullptr &&
        valid_owners_.find(item.owner) != valid_owners_.end() &&
        !item.owner->is_dead_block(item.owner_key, item.version)) {
      const size_t payload =
          bytes ? item.owner->eviction_bytes(item.owner_key) : 0;
      const size_t used_before = bytes ? pool.used() : 0;
      if (item.owner->evict_block(item.owner_key)) {
        ++evicted;
        // Legacy owners do not report payload sizes. Count their actual
        // release so a decoded column is not mistaken for a single page.
        const size_t released =
            bytes ? used_before - std::min(used_before, pool.used()) : 0;
        reclaimed += std::max(
            released, payload ? payload : MemoryLimitPool::page_buffer_size());
      }
    }
  }
  return evicted;
}

size_t BlockEvictionQueue::recover_owner_queues() {
  std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
  size_t recovered = 0;
  for (EvictableBlockOwner *owner : valid_owners_) {
    recovered += owner->recover_eviction_queue();
  }
  return recovered;
}

size_t BlockEvictionQueue::cleanup_dead_entries(size_t max_attempts) {
  if (max_attempts == 0) return 0;
  std::unique_lock<std::mutex> cleanup_lock(cleanup_mutex_, std::try_to_lock);
  if (!cleanup_lock.owns_lock()) return 0;
  if (!cleanup_in_progress_) {
    const auto generation = cleanup_generation_.load(std::memory_order_relaxed);
    if (generation == cleanup_generation_seen_) return 0;
    cleanup_generation_seen_ = generation;
    for (size_t i = 0; i < kQueueCount; ++i) {
      evict_queues_[i].begin_producer_scan(cleanup_scans_[i]);
    }
    cleanup_in_progress_ = true;
  }

  size_t attempts = 0;
  size_t cleaned = 0;
  std::array<size_t, kQueueCount> budgets{1, 1, 1};
  size_t first_queue = cleanup_next_queue_;
  cleanup_next_queue_ = (cleanup_next_queue_ + 1) % kQueueCount;
  if (max_attempts >= kQueueCount) {
    // Most retired pages accumulate in probation. Reserve smaller shares for
    // protected/hot entries while giving probation the first and largest scan.
    budgets[kProtectedPriority] = std::max<size_t>(1, max_attempts / 8);
    budgets[kExplicitHotPriority] = std::max<size_t>(1, max_attempts / 16);
    budgets[kProbationPriority] = max_attempts - budgets[kProtectedPriority] -
                                  budgets[kExplicitHotPriority];
    first_queue = kProbationPriority;
  }
  // Spend unused shares in priority order. Per-producer FIFO prefixes keep
  // live entries requeued on the cleaner's stream from hiding other streams.
  for (size_t i = 0; i < 2 * kQueueCount && attempts < max_attempts; ++i) {
    const size_t priority = (first_queue + i) % kQueueCount;
    const size_t budget =
        i < kQueueCount ? budgets[priority] : max_attempts - attempts;
    size_t remaining = budget;
    auto &scan = cleanup_scans_[priority];
    while (remaining != 0 && attempts < max_attempts && !scan.finished()) {
      std::array<BlockType, 64> items;
      const size_t requested =
          std::min({items.size(), remaining, max_attempts - attempts});
      const size_t found =
          evict_queues_[priority].try_dequeue_bulk_from_producer_scan(
              scan, items.begin(), requested);
      // Empty producers also consume bounded work; their metadata can outlive
      // both the original thread and every queued entry.
      const size_t work = std::max<size_t>(1, found);
      remaining -= work;
      attempts += work;
      if (found == 0) continue;
      approximate_queue_sizes_[priority].fetch_sub(found,
                                                   std::memory_order_relaxed);
      std::shared_lock<std::shared_mutex> owner_lock(valid_owners_mutex_);
      for (size_t entry = 0; entry < found; ++entry) {
        const auto &item = items[entry];
        if (item.owner == nullptr || valid_owners_.count(item.owner) == 0 ||
            item.owner->is_dead_block(item.owner_key, item.version)) {
          ++cleaned;
          continue;
        }
        // Keep the original priority/membership and CLOCK state. Only ordinary
        // pressure-driven eviction decides whether a live page should go.
        if (!add_single_block(item, static_cast<int>(priority))) {
          item.owner->eviction_requeue_failed(item.owner_key, item.version);
        }
      }
    }
  }
  cleanup_in_progress_ =
      std::any_of(cleanup_scans_.begin(), cleanup_scans_.end(),
                  [](const auto &scan) { return !scan.finished(); });
  cleanup_attempts_.fetch_add(attempts, std::memory_order_relaxed);
  dead_entries_cleaned_.fetch_add(cleaned, std::memory_order_relaxed);
  return cleaned;
}

bool BlockEvictionQueue::add_single_block(const BlockType &block,
                                          int queue_index) {
  if (queue_index < 0 || queue_index >= static_cast<int>(kQueueCount)) {
    LOG_ERROR("invalid eviction priority: %d", queue_index);
    return false;
  }
  BlockType queued = block;
  queued.priority = static_cast<uint8_t>(queue_index);
  // Publish the depth first so a concurrent consumer cannot dequeue the item
  // before its accounting is visible. Roll it back if enqueue fails.
  approximate_queue_sizes_[queue_index].fetch_add(1, std::memory_order_relaxed);
  bool ok = evict_queues_[queue_index].enqueue(queued);
  if (!ok) {
    approximate_queue_sizes_[queue_index].fetch_sub(1,
                                                    std::memory_order_relaxed);
    LOG_DEBUG("Eviction queue enqueue rejected under allocation pressure.");
    return false;
  }
  return true;
}

MemoryLimitPool::~MemoryLimitPool() {
  stop_background_evictor();
  drain_free_list();
}

bool MemoryLimitPool::drain_free_list() {
  trim_free_buffers(std::numeric_limits<size_t>::max());
  return release_all_slabs_locked();
}

size_t MemoryLimitPool::pick_shard() {
  // Keep each thread on one shard for locality.
  thread_local size_t idx = shard_seq_.fetch_add(1, std::memory_order_relaxed);
  return idx % kNumFreeShards;
}

bool MemoryLimitPool::try_reserve_native(size_t bytes, NativeGate gate,
                                         AdmissionDenials *denials) {
  const bool check_used = gate == NativeGate::kUsed;
  auto &counter = check_used ? used_size_ : committed_size_;
  const auto &generation =
      check_used ? used_release_generation_ : committed_release_generation_;
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  const auto fits = [&](size_t used) {
    return used <= capacity && bytes <= capacity - used;
  };
  const auto reserve = [&] {
    size_t used = counter.load(std::memory_order_relaxed);
    while (fits(used)) {
      if (counter.compare_exchange_weak(used, used + bytes,
                                        std::memory_order_relaxed))
        return true;
    }
    return false;
  };

  if (!denials) return reserve();
  auto &denial = check_used ? denials->used : denials->committed;
  if (!denial.blocked && reserve()) return true;
  return denials->rss_watch.check_native_gate([&] {
    std::lock_guard<std::mutex> lock(capacity_mutex_);
    if (reserve()) {
      denial.blocked = false;
      return true;
    }
    if (!denials->native_registered) {
      denials->next_native_waiter = native_waiters_;
      native_waiters_ = denials;
      denials->native_registered = true;
    }
    denial.record(generation.load(std::memory_order_relaxed));
    if (check_used) {
      // A live owner and idle backing can block both gates independently.
      // Snapshot the second gate before inline reclamation can trim that
      // backing.
      const auto committed_generation =
          committed_release_generation_.load(std::memory_order_relaxed);
      if (fits(committed_size_.load(std::memory_order_relaxed)))
        denials->committed.blocked = false;
      else
        denials->committed.record(committed_generation);
    }
    return false;
  });
}

bool MemoryLimitPool::try_reserve_used(size_t bytes,
                                       AdmissionDenials *denials) {
  return try_reserve_native(bytes, NativeGate::kUsed, denials);
}

bool MemoryLimitPool::try_reserve_page_used(size_t bytes, bool required) {
  process_rss_.refresh();
  if (!required && process_rss_.pressure()) return false;
  const size_t page_limit = page_admission_limit();
  size_t used = used_size_.load(std::memory_order_relaxed);
  while (used <= page_limit && bytes <= page_limit - used) {
    if (used_size_.compare_exchange_weak(used, used + bytes,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool MemoryLimitPool::try_reserve_committed(size_t bytes, Backing backing,
                                            size_t materialization,
                                            AdmissionDenials *denials) {
  if (!try_reserve_native(bytes, NativeGate::kCommitted, denials)) return false;
  if (backing == Backing::kPlan) return true;
  auto *rss_denied_at = denials ? &denials->rss : nullptr;
  auto *rss_watch = denials ? &denials->rss_watch : nullptr;
  const bool admitted =
      backing == Backing::kPending
          ? process_rss_.try_reserve(bytes, rss_denied_at, rss_watch)
          : process_rss_.try_acquire(bytes, materialization, rss_denied_at,
                                     rss_watch);
  if (!admitted) {
    committed_size_.fetch_sub(bytes, std::memory_order_relaxed);
    bg_cv_.notify_one();
  }
  return admitted;
}

size_t MemoryLimitPool::release_native(size_t bytes,
                                       std::atomic<size_t> &counter,
                                       std::atomic<uint64_t> &generation) {
  std::lock_guard<std::mutex> lock(capacity_mutex_);
  return release_native_locked(bytes, counter, generation);
}

size_t MemoryLimitPool::release_native_locked(
    size_t bytes, std::atomic<size_t> &counter,
    std::atomic<uint64_t> &generation) {
  const size_t previous = counter.fetch_sub(bytes, std::memory_order_relaxed);
  assert(previous >= bytes);
  if (bytes) {
    if (native_waiters_) {
      const size_t capacity = pool_size_.load(std::memory_order_relaxed);
      const size_t room = capacity - std::min(capacity, previous);
      const bool released_used = &counter == &used_size_;
      for (auto *waiter = native_waiters_; waiter;
           waiter = waiter->next_native_waiter) {
        const auto &denial = released_used ? waiter->used : waiter->committed;
        if (denial.blocked && waiter->native_bytes > room)
          waiter->native_progress.store(true, std::memory_order_release);
      }
    }
    generation.store(generation.load(std::memory_order_relaxed) + 1,
                     std::memory_order_release);
  }
  return previous;
}

void MemoryLimitPool::unregister_native_waiter(AdmissionDenials &denials) {
  std::lock_guard<std::mutex> lock(capacity_mutex_);
  auto **link = &native_waiters_;
  while (*link != &denials) link = &(*link)->next_native_waiter;
  *link = denials.next_native_waiter;
}

void MemoryLimitPool::release_native_used(size_t bytes) {
  release_native(bytes, used_size_, used_release_generation_);
}

size_t MemoryLimitPool::release_committed(
    size_t bytes, Backing backing,
    ProcessRssBudget::MappingProjection projection) {
  const size_t previous =
      release_native(bytes, committed_size_, committed_release_generation_);
  if (backing == Backing::kPending) {
    process_rss_.finish_reservation(bytes, false);
  } else if (backing == Backing::kUnmapped) {
    process_rss_.release_projected_mapping(bytes, projection);
  } else if (backing == Backing::kLive) {
    process_rss_.release(bytes);
  }
  return previous;
}

bool MemoryLimitPool::try_reserve_allocation(size_t bytes, Backing backing,
                                             size_t materialization,
                                             AdmissionDenials *denials) {
  // Pages reserve logical slots before acquiring backing. Checking only
  // committed capacity here would spend those pending slots a second time.
  // Keep both counters individually bounded without holding a lock across
  // allocation, eviction, or writeback callbacks.
  if (!try_reserve_used(bytes, denials)) return false;
  if (!try_reserve_committed(bytes, backing, materialization, denials)) {
    release_buffer(nullptr, bytes);
    return false;
  }
  return true;
}

bool MemoryLimitPool::is_cacheable_buffer_size(size_t buffer_size) {
  return buffer_size == page_buffer_size();
}

char *MemoryLimitPool::pop_free_buffer(size_t start_shard) {
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    size_t shard = (start_shard + i) % kNumFreeShards;
    std::lock_guard<std::mutex> lock(free_shards_[shard].mutex);
    char *buffer = free_shards_[shard].head;
    if (buffer) {
      free_shards_[shard].head = *reinterpret_cast<char **>(buffer);
      free_shards_[shard].count.fetch_sub(1, std::memory_order_relaxed);
      return buffer;
    }
  }
  return nullptr;
}

void MemoryLimitPool::push_free_buffer(char *buffer, size_t shard) {
  shard %= kNumFreeShards;
  std::lock_guard<std::mutex> lock(free_shards_[shard].mutex);
  *reinterpret_cast<char **>(buffer) = free_shards_[shard].head;
  free_shards_[shard].head = buffer;
  free_shards_[shard].count.fetch_add(1, std::memory_order_relaxed);
}

MemoryLimitPool::PageAllocation MemoryLimitPool::acquire_slab_buffer(
    size_t *reserved_pages) {
  static_assert(sizeof(ReclaimableSlab) <= kMinimumPageBytes,
                "slab metadata must fit in its header page");

  std::lock_guard<std::mutex> slabs_lock(slab_mutex_);
  bool attempted_backing = false;
  auto acquire_from = [this,
                       &attempted_backing](ReclaimableSlab *slab) -> char * {
    std::lock_guard<std::mutex> slab_lock(slab->mutex);
    uint16_t page_index = 0;
    bool reclaimed = false;
    if (slab->reclaimed_count != 0) {
      reclaimed = true;
      page_index = slab->reclaimed_pages[--slab->reclaimed_count];
    } else if (slab->next_unused <= slab->data_page_count) {
      page_index = slab->next_unused++;
    } else {
      return nullptr;
    }

    char *page = reinterpret_cast<char *>(slab) +
                 static_cast<size_t>(page_index) * slab->page_size;
    attempted_backing = true;
    if (!commit_slab_page(page, slab->page_size)) {
      if (reclaimed) {
        slab->reclaimed_pages[slab->reclaimed_count++] = page_index;
      } else {
        --slab->next_unused;
      }
      return nullptr;
    }
    ++slab->committed_pages;
    process_rss_.commit_reusable_page(slab->page_size, slab->projections,
                                      page_index - 1);
    return page;
  };

  if (allocation_slab_ != nullptr) {
    if (char *page = acquire_from(allocation_slab_)) {
      return {page, true};
    }
  }
  for (ReclaimableSlab *slab = slabs_; slab != nullptr; slab = slab->next) {
    if (slab == allocation_slab_) {
      continue;
    }
    if (char *page = acquire_from(slab)) {
      allocation_slab_ = slab;
      return {page, true};
    }
  }

  const size_t page_size = page_buffer_size();
  // The header is resident-capable backing too. Keep it charged even after
  // all data pages have been returned or discarded, until slab teardown.
  // Do not evict here while holding allocator locks.
  if (!try_reserve_used(page_size)) {
    // A batch may already own every logical slot. Convert one unused payload
    // slot into the header without publishing free capacity to competitors.
    // Keep at least one payload slot for the page being acquired now.
    if (!reserved_pages || *reserved_pages <= 1)
      return {nullptr, attempted_backing};
    --*reserved_pages;
  }
  if (!try_reserve_committed(page_size, Backing::kPending)) {
    release_buffer(nullptr, page_size);
    return {nullptr, attempted_backing};
  }
  metadata_used_size_.fetch_add(page_size, std::memory_order_relaxed);
  AlignedSlabMapping mapping = reserve_aligned_slab(page_size);
  if (mapping.base == nullptr) {
    release_fixed(page_size, &metadata_used_size_, Backing::kPending);
    LOG_ERROR("MemoryLimitPool: failed to reserve an aligned slab");
    return {nullptr, true};
  }
  ReclaimableSlab *slab = new (mapping.base) ReclaimableSlab(
      this, mapping.reservation_base, mapping.reservation_size, page_size);
  slab->header_projection = process_rss_.finish_reservation(page_size, true);
  slab->next = slabs_;
  slabs_ = slab;
  allocation_slab_ = slab;
  slab_count_.fetch_add(1, std::memory_order_relaxed);
  slab_mapped_bytes_.fetch_add(mapping.reservation_size,
                               std::memory_order_relaxed);
  char *page = acquire_from(slab);
  if (page != nullptr) {
    return {page, true};
  }

  ReclaimableSlab *next = slab->next;
  const auto projection = slab->header_projection;
  if (slab->unmap_empty()) {
    slabs_ = next;
    allocation_slab_ = nullptr;
    slab_count_.fetch_sub(1, std::memory_order_relaxed);
    slab_mapped_bytes_.fetch_sub(mapping.reservation_size,
                                 std::memory_order_relaxed);
    release_unmapped_metadata(page_size, projection);
  }
  return {nullptr, true};
}

bool MemoryLimitPool::reclaim_slab_buffer(char *buffer) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer);
  const uintptr_t slab_address = address & ~(kSlabAlignment - 1);
  auto *slab = reinterpret_cast<ReclaimableSlab *>(slab_address);
  const size_t offset = address - slab_address;
  if (slab->magic != ReclaimableSlab::kMagic || slab->owner != this ||
      offset < slab->page_size || offset >= kSlabBytes ||
      offset % slab->page_size != 0) {
    LOG_ERROR("MemoryLimitPool: invalid page returned to slab allocator");
    return false;
  }

  const auto page_index = static_cast<uint16_t>(offset / slab->page_size);
  std::lock_guard<std::mutex> slab_lock(slab->mutex);
  if (!discard_slab_page(buffer, slab->page_size)) {
    LOG_ERROR("MemoryLimitPool: failed to discard a free slab page");
    return false;
  }

  assert(slab->committed_pages != 0);
  --slab->committed_pages;
  release_native(slab->page_size, committed_size_,
                 committed_release_generation_);
  process_rss_.release_reusable_page(slab->page_size, slab->projections,
                                     page_index - 1);
  assert(slab->reclaimed_count < slab->data_page_count);
  slab->reclaimed_pages[slab->reclaimed_count++] = page_index;
  slab_reclaimed_pages_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool MemoryLimitPool::release_all_slabs_locked() {
  std::lock_guard<std::mutex> slabs_lock(slab_mutex_);
  allocation_slab_ = nullptr;
  while (slabs_) {
    ReclaimableSlab *slab = slabs_;
    if (slab->committed_pages != 0) return false;
    ReclaimableSlab *next = slab->next;
    const size_t page = slab->page_size;
    const size_t bytes = slab->reservation_size;
    const auto projection = slab->header_projection;
    if (!slab->unmap_empty()) return false;
    slabs_ = next;
    slab_count_.fetch_sub(1, std::memory_order_relaxed);
    slab_mapped_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    release_unmapped_metadata(page, projection);
  }
  return true;
}

size_t MemoryLimitPool::trim_free_buffers(size_t bytes_needed) {
  if (bytes_needed == 0) {
    return 0;
  }

  size_t released_bytes = trim_cache_buffers(bytes_needed);
  if (released_bytes < bytes_needed)
    released_bytes += trim_retained_blocks(bytes_needed - released_bytes);
  size_t shard = pick_shard();
  while (released_bytes < bytes_needed) {
    char *buffer = pop_free_buffer(shard);
    if (!buffer) {
      break;
    }
    // Only publish capacity after the physical page has been discarded.
    if (!reclaim_slab_buffer(buffer)) {
      push_free_buffer(buffer, shard);
      break;
    }
    released_bytes += page_buffer_size();
  }
  if (released_bytes != 0) notify_capacity_release(/*wake_all=*/true);
  return released_bytes;
}

bool MemoryLimitPool::trim_idle_buffers() {
  // These buffers have no owners and no cached contents. Keep a small reuse
  // reserve, not the historical high-water allocation of a finished build.
  // The counters are only a hint: pop/unmap still obtains exclusive ownership
  // before discarding backing and refunding committed capacity.
  size_t idle = retained_idle_bytes_.load(std::memory_order_relaxed) +
                cache_idle_bytes_.load(std::memory_order_relaxed);
  for (const auto &shard : free_shards_) {
    idle += shard.count.load(std::memory_order_relaxed) * page_buffer_size();
  }
  const size_t reserve =
      process_rss_.pressure() ? 0 : reserve_margin(capacity());
  if (idle > reserve) {
    // Bound each maintenance pass. A retained mapping is released whole;
    // live pages and live retained blocks are never touched by this path.
    return trim_free_buffers(
               std::min(idle - reserve, 64 * page_buffer_size())) != 0;
  }
  return false;
}

bool MemoryLimitPool::configure_process_rss_limit(size_t bytes) {
  return configure_process_rss_limit(bytes, 0);
}

bool MemoryLimitPool::configure_process_rss_limit(size_t bytes,
                                                  size_t permanent_backing) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  // Permanent native allowances use external lifetime reservations. Queue
  // and slab metadata remain fully charged and cannot be excluded here.
  if (permanent_backing > external_used_size_.load(std::memory_order_relaxed))
    return false;
  auto resident_reader = &MemoryLimitPool::read_slab_resident_sample;
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
  // Foreground budget refreshes remain cheap. The background worker publishes
  // independently validated ownership proofs without holding this mutex.
  resident_reader = +[]() -> ProcessRssBudget::ResidentSample {
    return {MemoryHelper::SelfRSS(), 0};
  };
#endif
  const bool configured = process_rss_.configure(
      bytes, &MemoryHelper::SelfRSS, NativeApplicationMemoryReader(),
      permanent_backing, resident_reader);
  bg_cv_.notify_one();
  return configured;
}

void MemoryLimitPool::set_memory_pressure(
    ProcessRssBudget::PressureLevel level) {
  process_rss_.set_external_pressure(level);
  bg_cv_.notify_one();
  notify_capacity_release(/*wake_all=*/true);
}

void MemoryLimitPool::set_process_rss_reclaimer(
    std::function<void()> reclaimer) {
  auto callback =
      reclaimer
          ? std::make_shared<const std::function<void()>>(std::move(reclaimer))
          : nullptr;
  std::lock_guard<std::mutex> lock(process_reclaimer_mutex_);
  process_reclaimer_ = std::move(callback);
}

template <typename Observe>
void MemoryLimitPool::reclaim_process_cache(Observe &&observe) {
  std::shared_ptr<const std::function<void()>> callback;
  {
    std::lock_guard<std::mutex> lock(process_reclaimer_mutex_);
    callback = process_reclaimer_;
    if (callback) process_reclaimer_running_ = true;
  }
  // Copying this owner allocates nothing on the pressure path. It shares the
  // same pool across DSOs, and no pool lock is held while calling another
  // cache.
  if (!callback) return;
  auto finish = ScopeGuard::Make([this, &callback] {
    callback.reset();
    std::lock_guard<std::mutex> lock(process_reclaimer_mutex_);
    process_reclaimer_running_ = false;
  });
  (*callback)();
  // Unregistration can leave the copied callback holding the final owner.
  // Destroy it before observing the memory released by this reclaim pass.
  callback.reset();
  observe();
}

int MemoryLimitPool::init(size_t pool_size) {
  // Queue storage can exist before a pool is configured. Construct it first
  // and adopt all retained allocations when publishing a non-zero budget.
  // Keep the existing preallocation policy and reject an insufficient budget.
  (void)BlockEvictionQueue::get_instance();
  std::lock_guard<std::mutex> initialization_lock(initialization_mutex_);
  // Re-publishing the active process-wide budget is a safe no-op.
  if (initialized_.load(std::memory_order_acquire) &&
      pool_size_.load(std::memory_order_relaxed) == pool_size) {
    return 0;
  }
  // A running reclaimer may allocate through the pool. Let it finish before
  // taking the reservation lock, otherwise joining maintenance can deadlock
  // with a callback waiting for that lock's shared side.
  bool restart_previous = bg_running_.load();
  stop_background_evictor();
  auto restart_evictor = ScopeGuard::Make([this, &restart_previous] {
    if (!restart_previous) return;
    try {
      start_background_evictor();
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to restart the shared-cache evictor: %s", e.what());
    } catch (...) {
      LOG_ERROR("Failed to restart the shared-cache evictor");
    }
  });
  std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  // Lifetime exclusions have already left the RSS backing ledger. Changing
  // native capacity would retire and re-adopt the queue charge, which cannot
  // be paired with that exclusion. Keep the original pool intact instead.
  if (process_rss_.permanent_backing() != 0) {
    LOG_ERROR(
        "MemoryLimitPool reinitialization rejected with permanent process "
        "backing: requested_capacity=%zu permanent_backing=%zu",
        pool_size, process_rss_.permanent_backing());
    return -1;
  }
  // The callback may have created owners. Check again only after it has
  // finished, preserving the previous pool and worker on a rejected reset.
  // Idle cache descriptors are allocator metadata, not active owners.
  trim_cache_buffers(std::numeric_limits<size_t>::max());
  const size_t used = used_size_.load(std::memory_order_relaxed);
  const size_t external = external_used_size_.load(std::memory_order_relaxed);
  const size_t metadata = metadata_used_size_.load(std::memory_order_relaxed);
  const size_t queue_charge =
      queue_charged_size_.load(std::memory_order_relaxed);
  const size_t allocator_charge =
      queue_charge +
      slab_count_.load(std::memory_order_relaxed) * page_buffer_size();
  if (used != allocator_charge || external != 0 ||
      metadata != allocator_charge) {
    LOG_ERROR(
        "MemoryLimitPool reinitialization rejected while cache memory is "
        "active: requested_capacity=%zu current_capacity=%zu used=%zu "
        "external_used=%zu metadata_used=%zu",
        pool_size, pool_size_.load(std::memory_order_relaxed), used, external,
        metadata);
    return -1;
  }

  std::unique_lock<std::mutex> queue_lock(queue_memory_mutex_);
  const size_t queue_bytes =
      queue_allocated_size_.load(std::memory_order_relaxed);
  if (pool_size != 0 && pool_size < queue_bytes) {
    LOG_ERROR(
        "Shared cache budget cannot fit eviction queues: capacity=%zu "
        "queue_bytes=%zu",
        pool_size, queue_bytes);
    return -1;
  }
  // No owner allocations remain. Keep the existing budget active until the
  // old allocator releases its backing, so failure leaves it usable.
  if (!drain_free_list() ||
      retained_block_bytes_.load(std::memory_order_relaxed) != 0) {
    return -1;
  }
  const size_t old_charge = queue_charged_size_.exchange(0);
  if (old_charge != 0) {
    release_metadata(old_charge);
  }
  pool_size_.store(0, std::memory_order_relaxed);
  process_rss_.configure(0, nullptr);
  set_process_rss_reclaimer({});
  pool_size_.store(pool_size, std::memory_order_relaxed);
  if (pool_size != 0) {
    queue_charged_size_.store(queue_bytes, std::memory_order_relaxed);
    metadata_used_size_.fetch_add(queue_bytes, std::memory_order_relaxed);
    used_size_.fetch_add(queue_bytes, std::memory_order_relaxed);
    committed_size_.fetch_add(queue_bytes, std::memory_order_relaxed);
    process_rss_.account(queue_bytes);
  }
  initialized_.store(true, std::memory_order_release);
  queue_lock.unlock();
  restart_previous = false;
  LOG_INFO("Shared cache initialized with capacity: %zu", pool_size);
  if (pool_size > 0) {
    try {
      start_background_evictor();
    } catch (const std::exception &e) {
      std::lock_guard<std::mutex> lock(queue_memory_mutex_);
      release_metadata(queue_charged_size_.exchange(0));
      pool_size_.store(0, std::memory_order_relaxed);
      initialized_.store(false, std::memory_order_release);
      LOG_ERROR("Failed to start the shared-cache evictor: %s", e.what());
      return -1;
    } catch (...) {
      std::lock_guard<std::mutex> lock(queue_memory_mutex_);
      release_metadata(queue_charged_size_.exchange(0));
      pool_size_.store(0, std::memory_order_relaxed);
      initialized_.store(false, std::memory_order_release);
      LOG_ERROR(
          "Failed to start the shared-cache evictor with an unknown error");
      return -1;
    }
  }
  return 0;
}

void MemoryLimitPool::start_background_evictor() {
  bool expected = false;
  if (!bg_running_.compare_exchange_strong(expected, true)) {
    return;  // already running
  }
  try {
    bg_thread_ = std::thread([this] { background_evict_loop(); });
  } catch (...) {
    bg_running_.store(false, std::memory_order_release);
    throw;
  }
}

void MemoryLimitPool::stop_background_evictor() {
  if (!bg_running_.exchange(false)) {
    return;  // not running
  }
  { std::lock_guard<std::mutex> lk(bg_mutex_); }
  bg_cv_.notify_all();
  if (bg_thread_.joinable()) {
    bg_thread_.join();
  }
}

void MemoryLimitPool::background_evict_loop() {
  {
    std::lock_guard<std::mutex> lock(bg_identity_mutex_);
    bg_identity_ = std::this_thread::get_id();
  }
  auto clear_background = ScopeGuard::Make([this] {
    std::lock_guard<std::mutex> lock(bg_identity_mutex_);
    bg_identity_ = {};
  });
  using std::chrono::milliseconds;
  bool idle_progress = false;
  bool reclaim_to_low = false;
  auto last_process_reclaim = std::chrono::steady_clock::time_point{};
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
#if defined(__linux__)
  // This proves overlap, rather than observing RSS or admitting allocations.
  // Foreground RSS checks keep their 20 ms cadence; retirement-driven pressure
  // still requests an immediate proof. Avoid page-table walks on a steady
  // cache.
  constexpr auto owned_sample_interval = milliseconds(1000);
#else
  constexpr auto owned_sample_interval = milliseconds(100);
#endif
  auto last_owned_sample = std::chrono::steady_clock::time_point{};
  uint64_t confirmed_generation = 0;
  const auto refresh_owned_sample = [&] {
    const auto generation = process_rss_.resident_generation();
    // A mandatory waiter can exhaust the conservative overlap projection
    // while measured RSS remains below its limit but inside hysteresis.
    // Confirm once per backing/configuration generation before its wait ends;
    // the proof still excludes untouched, retired and in-flight backing.
    const bool waiting_for_proof =
        process_rss_.has_resident_uncertainty() &&
        mandatory_waiters_.load(std::memory_order_relaxed) != 0;
    const bool confirm =
        (process_rss_.needs_resident_confirmation() || waiting_for_proof) &&
        generation != confirmed_generation;
    if (process_rss_.limit() &&
        (confirm || std::chrono::steady_clock::now() - last_owned_sample >=
                        owned_sample_interval)) {
      const auto token = process_rss_.begin_resident_sample();
      if (token.configuration) {
        const auto sample = read_background_resident_sample();
        if (process_rss_.publish_resident_sample(token, sample)) {
          last_owned_sample = std::chrono::steady_clock::now();
          confirmed_generation = generation;
        }
      }
    }
  };
#endif
  const auto page_admission_pressure = [this] {
    // Reserved headroom can block page admission below both general
    // watermarks. Keep the background queue progressing at that boundary,
    // but do not reclaim when only non-evictable fixed reservations remain.
    return used_size_.load(std::memory_order_relaxed) > fixed_used() &&
           (is_page_full() || process_rss_.pressure());
  };
  while (bg_running_.load()) {
    {
      std::unique_lock<std::mutex> lk(bg_mutex_);
      bg_cv_.wait_for(lk, milliseconds(5),
                      [this, &page_admission_pressure, &idle_progress] {
                        return !bg_running_.load() || idle_progress ||
                               should_background_reclaim() ||
                               cache_reclaim_requested_.load(
                                   std::memory_order_relaxed) != 0 ||
                               page_admission_pressure();
                      });
    }
    if (!bg_running_.load()) break;
    if (pool_size_.load(std::memory_order_relaxed) == 0) continue;
    const size_t requested =
        cache_reclaim_requested_.exchange(0, std::memory_order_relaxed);
    process_rss_.refresh();
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
    refresh_owned_sample();
#endif
    if (requested && (used() > low_watermark() || process_rss_.pressure())) {
      reclaim_to_low = true;
      const size_t n =
          BlockEvictionQueue::get_instance().reclaim_for_allocation(requested);
      bg_evicted_buffers_.fetch_add(n, std::memory_order_relaxed);
    }
    if (process_rss_.limit()) capacity_cv_.notify_all();
    const auto now = std::chrono::steady_clock::now();
    if ((process_rss_.pressure() ||
         mandatory_waiters_.load(std::memory_order_relaxed) != 0) &&
        now - last_process_reclaim >= milliseconds(20)) {
      // RocksDB has a separate cache budget. Retire its unpinned entries too,
      // including when vector pages cannot make further reclamation progress.
      reclaim_process_cache([&] {
        process_rss_.refresh(true);
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
        refresh_owned_sample();
#endif
      });
      last_process_reclaim = now;
    }
    // Flush/compaction can retire many owners without reaching a watermark.
    // Reuse queue blocks before stale memberships cause unbounded growth.
    const size_t retired =
        BlockEvictionQueue::get_instance().cleanup_dead_entries(256);
    // Page/entry retirement can leave committed backing resident even when
    // used is far below the eviction watermark. Return only idle backing.
    // After the final page owner closes, finish returning idle backing to the
    // existing reuse reserve without sleeping between every small batch. An
    // active cache retains its paced page maintenance. Keep removing retired
    // memberships while cleanup makes progress, even below the watermarks.
    // No progress (including an OS discard failure) restores the wait.
    const bool trimmed = trim_idle_buffers();
    idle_progress =
        retired != 0 ||
        (trimmed && used_size_.load(std::memory_order_relaxed) == fixed_used());
    const size_t low = low_watermark();
    // Preserve main's periodic low-watermark maintenance, including protected
    // aging for small batches. RSS/page-admission pressure is an additional
    // trigger and must survive a temporarily blocked reclamation pass.
    reclaim_to_low =
        reclaim_to_low || used_size_.load() > low || page_admission_pressure();
    if (reclaim_to_low &&
        (used_size_.load() > low || page_admission_pressure())) {
      bg_evict_rounds_.fetch_add(1, std::memory_order_relaxed);
    }
    // Reclaim to the low watermark and make room for page admission so the
    // foreground path finds ready buffers without a deep inline queue walk.
    while (bg_running_.load() && reclaim_to_low &&
           (used_size_.load() > low || page_admission_pressure())) {
      const size_t used = this->used();
      const size_t bytes =
          std::max(page_buffer_size(), used - std::min(used, low));
      const size_t pages =
          std::min<size_t>(64, (bytes - 1) / page_buffer_size() + 1);
      size_t n =
          evictable_cache_used() != 0
              ? BlockEvictionQueue::get_instance().reclaim_for_allocation(
                    pages * page_buffer_size())
              : BlockEvictionQueue::get_instance().batch_recycle(
                    pages, /*age_protected=*/true);
      if (n == 0) {
        // Back off when pressure remains but eviction makes no progress.
        bg_no_progress_sleeps_.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lk(bg_mutex_);
        bg_cv_.wait_for(lk, milliseconds(5),
                        [this] { return !bg_running_.load(); });
        break;
      }
      bg_evicted_buffers_.fetch_add(n, std::memory_order_relaxed);
      if (process_rss_.pressure()) {
        trim_idle_buffers();
        process_rss_.refresh(true);
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
        refresh_owned_sample();
#endif
      }
    }
    if (used_size_.load() <= low && !page_admission_pressure())
      reclaim_to_low = false;
  }
}

bool MemoryLimitPool::refresh_residency_on_background_thread() {
  {
    // Components can embed their own copy of the allocator implementation.
    // The worker identity must live in their shared pool, not image-local TLS.
    std::lock_guard<std::mutex> lock(bg_identity_mutex_);
    if (bg_identity_ != std::this_thread::get_id()) return false;
  }
  // A reclaimer can allocate through the pool after its own work expires an
  // earlier proof. It must sample directly instead of waiting for itself.
  const auto token = process_rss_.begin_resident_sample();
  if (token.configuration)
    process_rss_.publish_resident_sample(token,
                                         read_background_resident_sample());
  return true;
}

char *MemoryLimitPool::pop_reusable_buffer(size_t shard, size_t bytes) {
  char *buffer = pop_free_buffer(shard);
  if (!buffer) return nullptr;
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer);
  const uintptr_t slab_address = address & ~(kSlabAlignment - 1);
  auto *slab = reinterpret_cast<ReclaimableSlab *>(slab_address);
  const size_t page_index = (address - slab_address) / bytes;
  if (process_rss_.try_materialize_reusable_page(bytes, slab->projections,
                                                 page_index - 1))
    return buffer;
  push_free_buffer(buffer, shard);
  return nullptr;
}

bool MemoryLimitPool::try_acquire_buffer(const size_t buffer_size,
                                         char *&buffer) {
  return try_acquire_buffer(buffer_size, buffer, false);
}

bool MemoryLimitPool::try_acquire_buffer(size_t buffer_size, char *&buffer,
                                         bool required) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  buffer = nullptr;
  const bool cacheable = is_cacheable_buffer_size(buffer_size);
  if (buffer_size == 0 ||
      !(cacheable ? try_reserve_page_used(buffer_size, required)
                  : try_reserve_used(buffer_size))) {
    // Out of budget: wake the background evictor so the next attempt is
    // more likely to find a free buffer without inline eviction.
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
    return false;
  }

  if (cacheable) {
    buffer = pop_reusable_buffer(pick_shard(), buffer_size);
    if (buffer) {
      alloc_from_freelist_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  if (!try_reserve_committed(buffer_size, Backing::kPending)) {
    // This is primarily useful for a non-cacheable size. For the normal page
    // size, all currently visible free buffers were already checked above.
    trim_free_buffers(buffer_size);
    if (!try_reserve_committed(buffer_size, Backing::kPending)) {
      release_buffer(nullptr, buffer_size);
      high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
  }
  const auto allocation =
      cacheable ? acquire_slab_buffer()
                : PageAllocation{static_cast<char *>(ailego_aligned_malloc(
                                     buffer_size, kMinimumPageBytes)),
                                 true};
  buffer = allocation.buffer;
  if (!buffer) {
    cancel_buffer_reservation(buffer_size, allocation.attempted_backing);
    return false;
  }
  if (!cacheable) process_rss_.finish_reservation(buffer_size, true);
  alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool MemoryLimitPool::wait_for_available(const size_t buffer_size,
                                         std::chrono::milliseconds timeout) {
  return wait_for_available(buffer_size, timeout, false);
}

bool MemoryLimitPool::wait_for_available(size_t buffer_size,
                                         std::chrono::milliseconds timeout,
                                         bool required) {
  if (buffer_size == 0) return true;
  capacity_waits_.fetch_add(1, std::memory_order_relaxed);
  std::unique_lock<std::mutex> lock(capacity_mutex_);
  const bool available =
      capacity_cv_.wait_for(lock, timeout, [this, buffer_size, required] {
        const size_t capacity = pool_size_.load(std::memory_order_relaxed);
        const size_t used = used_size_.load(std::memory_order_relaxed);
        if (capacity < used || buffer_size > capacity - used) {
          return false;
        }
        if (buffer_size > process_rss_.available()) return false;
        if (!required && process_rss_.pressure()) return false;
        // A page can be denied below the total capacity: unconsumed external
        // cache headroom is not available to page admission. Use the same
        // page-limit predicate as eviction instead of turning backpressure
        // into immediate successful waits while pages still cannot fit.
        if (!is_cacheable_buffer_size(buffer_size)) return true;
        const size_t limit = page_admission_limit();
        if (used >= limit || buffer_size > limit - used) return false;
        for (const auto &shard : free_shards_)
          if (shard.count.load(std::memory_order_relaxed) != 0) return true;
        const size_t committed =
            committed_size_.load(std::memory_order_relaxed);
        const size_t fixed = fixed_used();
        const size_t backed = committed > fixed ? committed - fixed : 0;
        const size_t slots_per_slab = slab_size() / buffer_size - 1;
        const size_t slabs = slab_count_.load(std::memory_order_relaxed);
        // A full slab needs another resident header as well as the payload.
        // Otherwise an apparent one-page allowance causes immediate waits
        // to spin out before the dirty-writeback worker can run. Outstanding
        // non-page backing only makes this hint conservative; allocation is
        // retried after a timeout, without taking allocator locks under CV.
        const bool has_slot =
            slabs != 0 && backed / buffer_size / slabs < slots_per_slab;
        const size_t extra = has_slot ? 0 : buffer_size;
        return committed <= capacity && buffer_size <= capacity - committed &&
               extra <= capacity - committed - buffer_size &&
               extra <= capacity - used - buffer_size;
      });
  if (!available) {
    capacity_wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
  }
  return available;
}

bool MemoryLimitPool::try_charge_external(const size_t buffer_size) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed(buffer_size, &external_used_size_);
}

bool MemoryLimitPool::try_charge_workspace(const size_t buffer_size) {
  return try_charge_workspace(buffer_size, 0);
}

bool MemoryLimitPool::try_charge_workspace(size_t buffer_size,
                                           size_t materialization) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed_with_wait(buffer_size, &external_used_size_,
                                    Backing::kLive, materialization);
}

bool MemoryLimitPool::try_charge_process_backing(size_t bytes,
                                                 size_t materialization,
                                                 bool pending_copy) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  if (materialization > std::numeric_limits<size_t>::max() - bytes)
    return false;
  const size_t required = bytes + materialization;
  const auto admit = [&](AdmissionDenials *denials) {
    auto *denied_at = denials ? &denials->rss : nullptr;
    auto *watch = denials ? &denials->rss_watch : nullptr;
    return pending_copy ? process_rss_.try_acquire_with_pending_copy(
                              bytes, materialization, denied_at, watch)
                        : process_rss_.try_acquire(bytes, materialization,
                                                   denied_at, watch);
  };
  return try_admit({0, required}, admit, true);
}

bool MemoryLimitPool::try_charge_workspace_plan(size_t bytes) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed_with_wait(bytes, &external_used_size_,
                                    Backing::kPlan);
}

void MemoryLimitPool::release_workspace_plan(size_t bytes) {
  release_fixed(bytes, &external_used_size_, Backing::kPlan);
}

bool MemoryLimitPool::try_charge_metadata(const size_t buffer_size) {
  return try_charge_metadata(buffer_size, 0);
}

bool MemoryLimitPool::try_charge_metadata(size_t buffer_size,
                                          size_t materialization) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed_with_wait(buffer_size, &metadata_used_size_,
                                    Backing::kLive, materialization);
}

bool MemoryLimitPool::try_charge_metadata_without_reclaim(size_t bytes) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_locked(bytes, &metadata_used_size_, Backing::kLive);
}

bool MemoryLimitPool::try_charge_locked(size_t bytes,
                                        std::atomic<size_t> *counter,
                                        Backing backing) {
  if (!bytes) return true;
  if (!try_reserve_allocation(bytes, backing)) {
    // Discard only idle backing; never call an eviction owner under its lock.
    trim_free_buffers(bytes);
    if (!try_reserve_allocation(bytes, backing)) return false;
  }
  counter->fetch_add(bytes, std::memory_order_relaxed);
  bg_cv_.notify_one();
  return true;
}

MemoryLimitPool::PendingAllocation::PendingAllocation(MemoryLimitPool &pool,
                                                      Kind kind)
    : pool_(pool),
      kind_(kind),
      counter_(kind == Kind::kMetadata || kind == Kind::kLockedMetadata
                   ? &pool.metadata_used_size_
                   : &pool.external_used_size_) {}

MemoryLimitPool::PendingAllocation::~PendingAllocation() {
  if (bytes_ && kind_ == Kind::kEvictableCache)
    pool_.evictable_cache_size_.fetch_sub(bytes_, std::memory_order_relaxed);
  if (bytes_) pool_.release_fixed(bytes_, counter_, Backing::kPending);
}

bool MemoryLimitPool::PendingAllocation::reserve(size_t bytes) {
  assert(bytes_ == 0);
  if (!bytes) return true;
  std::shared_lock<std::shared_mutex> lifecycle_lock(pool_.lifecycle_mutex_);
  bool reserved = false;
  switch (kind_) {
    case Kind::kExternal:
      reserved = pool_.try_charge_fixed(bytes, counter_, Backing::kPending);
      break;
    case Kind::kWorkspace:
    case Kind::kMetadata:
      reserved =
          pool_.try_charge_fixed_with_wait(bytes, counter_, Backing::kPending);
      break;
    case Kind::kLockedMetadata:
    case Kind::kLockedExternal:
      reserved = pool_.try_charge_locked(bytes, counter_, Backing::kPending);
      break;
    case Kind::kEvictableCache: {
      const size_t used = pool_.used();
      const size_t limit = pool_.high_watermark();
      if (pool_.mandatory_waiters_.load(std::memory_order_relaxed) == 0 &&
          used <= limit && bytes <= limit - used &&
          pool_.try_reserve_allocation(bytes, Backing::kPending)) {
        counter_->fetch_add(bytes, std::memory_order_relaxed);
        pool_.evictable_cache_size_.fetch_add(bytes, std::memory_order_relaxed);
        reserved = true;
      }
      break;
    }
  }
  if (reserved) bytes_ = bytes;
  return reserved;
}

bool MemoryLimitPool::PendingAllocation::reserve_with_metadata(
    size_t bytes, PendingAllocation &metadata, size_t metadata_bytes) {
  assert(bytes_ == 0);
  assert(&pool_ == &metadata.pool_ && metadata.bytes_ == 0 &&
         metadata.kind_ == Kind::kLockedMetadata);
  assert(kind_ == Kind::kWorkspace || kind_ == Kind::kLockedExternal);
  if (bytes > std::numeric_limits<size_t>::max() - metadata_bytes) return false;
  const size_t total = bytes + metadata_bytes;
  std::shared_lock<std::shared_mutex> lifecycle_lock(pool_.lifecycle_mutex_);
  const auto admit = [&](AdmissionDenials *denials) {
    if (!pool_.try_reserve_allocation(total, Backing::kPending, 0, denials))
      return false;
    // Reserve native/RSS backing once, but publish each owner's own charge.
    // Descriptor metadata must not count as external page-admission credit.
    metadata.counter_->fetch_add(metadata_bytes, std::memory_order_relaxed);
    counter_->fetch_add(bytes, std::memory_order_relaxed);
    pool_.bg_cv_.notify_one();
    return true;
  };
  if (kind_ == Kind::kWorkspace) {
    if (!pool_.try_admit({total, total}, admit, true)) return false;
  } else {
    const auto try_optional = [&] {
      pool_.process_rss_.refresh();
      return !pool_.process_rss_.pressure() &&
             pool_.mandatory_waiters_.load(std::memory_order_relaxed) == 0 &&
             admit(nullptr);
    };
    if (!try_optional()) {
      // Match locked/optional admission: trim idle backing without owner calls.
      pool_.trim_free_buffers(total);
      if (!try_optional()) return false;
    }
  }
  bytes_ = bytes;
  metadata.bytes_ = metadata_bytes;
  return true;
}

ProcessRssBudget::MaterializationStamp
MemoryLimitPool::PendingAllocation::commit(size_t bytes) {
  return commit_mapping(bytes);
}

ProcessRssBudget::MaterializationStamp
MemoryLimitPool::PendingAllocation::commit_mapping() {
  return commit_mapping(bytes_);
}

ProcessRssBudget::MaterializationStamp
MemoryLimitPool::PendingAllocation::commit_mapping(size_t bytes) {
  assert(bytes <= bytes_);
  if (!bytes) return {};
  auto stamp = pool_.process_rss_.finish_reservation(bytes, true);
  bytes_ -= bytes;
  return stamp;
}

size_t MemoryLimitPool::native_available() const {
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  const size_t reserved =
      std::max(used_size_.load(std::memory_order_relaxed),
               committed_size_.load(std::memory_order_relaxed));
  return reserved < capacity ? capacity - reserved : 0;
}

size_t MemoryLimitPool::queue_lifetime_bytes() const {
  if (capacity() == 0) return 0;
  return BlockEvictionQueue::get_instance().initial_backing_bytes_ +
         queue_runtime_lifetime_size_.load(std::memory_order_relaxed);
}

size_t MemoryLimitPool::rss_allocation_limit() const {
  const size_t limit = process_rss_.limit();
  if (!limit) return std::numeric_limits<size_t>::max();
  // Resident overlap can credit a measured sample, but the queue's backing
  // remains in the ledger even if every other owner releases its memory.
  const size_t queue_bytes = queue_lifetime_bytes();
  return limit - std::min(limit, queue_bytes);
}

size_t MemoryLimitPool::native_allocation_limit(
    size_t retained, bool retained_may_be_permanent) const {
  const size_t capacity = this->capacity();
  if (capacity == 0) return 0;
  const size_t queue_bytes = queue_lifetime_bytes();
  const size_t permanent = process_rss_.permanent_backing();
  if (retained_may_be_permanent) retained -= std::min(retained, permanent);
  const size_t after_queues = capacity - std::min(capacity, queue_bytes);
  const size_t after_permanent =
      after_queues - std::min(after_queues, permanent);
  return after_permanent - std::min(after_permanent, retained);
}

bool MemoryLimitPool::can_fit_metadata(size_t bytes, size_t retained) const {
  const size_t limit =
      std::min(native_allocation_limit(), rss_allocation_limit());
  return retained <= limit && bytes <= limit - retained;
}

size_t MemoryLimitPool::admission_shortfall(AdmissionRequest request) const {
  const size_t native_room = native_available();
  const size_t rss_room = process_rss_.available();
  return std::max(
      request.native_bytes - std::min(request.native_bytes, native_room),
      request.rss_bytes - std::min(request.rss_bytes, rss_room));
}

template <typename Admit>
bool MemoryLimitPool::try_admit(AdmissionRequest request, Admit &&admit,
                                bool wait) {
  using Clock = std::chrono::steady_clock;
  using std::chrono::milliseconds;
  const bool reuse = request.native_bytes == 0 && request.rss_bytes == 0;
  const bool required = wait;
  AdmissionDenials denials(request.native_bytes, request.rss_bytes,
                           process_rss_);
  auto unregister_native = ScopeGuard::Make([&] {
    if (denials.native_registered) unregister_native_waiter(denials);
  });
  const auto impossible = [&] {
    return request.native_bytes > native_allocation_limit() ||
           request.rss_bytes > rss_allocation_limit();
  };
  const auto try_once = [&] {
    denials.reset_attempt();
    const bool admitted = admit(wait ? &denials : nullptr);
    if (admitted || !wait) return admitted;
    if (!denials.rss && request.rss_bytes && !impossible()) {
      const auto rss =
          process_rss_.observe_denial(request.rss_bytes, denials.rss_watch);
      if (rss.blocked()) denials.rss = rss;
    }
    return false;
  };
  // The deadline bounds waiting and starting another reclaim pass. A
  // synchronous reclaim already in progress can finish after the deadline.
  if (try_once()) return true;
  const auto denied_at = Clock::now();
  const auto first_wait_end = denied_at + milliseconds(100);
  const auto deadline = denied_at + std::chrono::seconds(5);
  const auto retry = [&] { return Clock::now() < deadline && try_once(); };
  // Maintenance can release idle backing and prove residency itself, but
  // cannot wait for its own worker or recursively invoke eviction owners.
  const bool maintenance = refresh_residency_on_background_thread();
  if (maintenance) {
    wait = false;
    if (try_once()) return true;
  }
  if (impossible()) return false;

  if (wait) {
    size_t waiting = native_waiting_bytes_.load(std::memory_order_relaxed);
    do {
      if (request.native_bytes > std::numeric_limits<size_t>::max() - waiting)
        return false;
    } while (!native_waiting_bytes_.compare_exchange_weak(
        waiting, waiting + request.native_bytes, std::memory_order_relaxed));
    mandatory_waiters_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
  }
  auto release_priority = ScopeGuard::Make([this, request, wait] {
    if (!wait) return;
    native_waiting_bytes_.fetch_sub(request.native_bytes,
                                    std::memory_order_relaxed);
    mandatory_waiters_.fetch_sub(1, std::memory_order_relaxed);
    capacity_cv_.notify_all();
  });
  // Compare only headroom this request can use. A zero-growth reuse still
  // needs RSS confirmation. A native-only plan does not consume RSS.
  const auto room_for_request = [&] {
    const size_t native = request.native_bytes ? std::min(request.native_bytes,
                                                          native_available())
                                               : 0;
    const size_t rss =
        request.rss_bytes
            ? std::min(request.rss_bytes, process_rss_.available())
            : (request.native_bytes ? 0 : process_rss_.available());
    return AdmissionRequest{native, rss};
  };
  const auto rss_recovered =
      [&](const std::optional<ProcessRssBudget::Denial> &denial) {
        return denial &&
               process_rss_.recovery_generation(reuse) != denial->recovery;
      };
  const auto has_recovery_progress = [&] {
    return denials.native_progress.load(std::memory_order_acquire) ||
           denials.rss_watch.progressed();
  };
  const auto refresh_after_reclaim = [&] {
    if (!request.rss_bytes && !reuse) return;
    if (required)
      process_rss_.refresh_after_reclaim(request.rss_bytes);
    else
      process_rss_.refresh(true);
  };

  // Close the race between the first denial and publishing priority.
  if (wait && retry()) return true;
  size_t previous_shortfall = std::numeric_limits<size_t>::max();
  while (Clock::now() < deadline) {
    const size_t shortfall = admission_shortfall(request);
    // Retry inline only after useful reclamation. A failed combined RSS gate
    // must never retry merely because either component would fit by itself.
    if (shortfall && shortfall < previous_shortfall) {
      previous_shortfall = shortfall;
      if (trim_free_buffers(shortfall) != 0) {
        refresh_after_reclaim();
        // Maintenance can confirm residency without waiting for another pass.
        if (maintenance) refresh_residency_on_background_thread();
        if (try_once()) return true;
        continue;
      }
      if (!maintenance && Clock::now() < deadline &&
          BlockEvictionQueue::get_instance().reclaim_for_allocation(
              shortfall)) {
        refresh_after_reclaim();
        // Finish the started pass by returning backing for any remaining
        // deficit. If capacity is already sufficient, skip idle trimming.
        const size_t remaining = admission_shortfall(request);
        if (remaining && trim_free_buffers(remaining) != 0)
          refresh_after_reclaim();
        if (try_once()) return true;
        continue;
      }
      if (Clock::now() >= deadline) {
        // A pass can report no reclamation while another owner refunds bytes.
        // Check the budget once before returning without starting more work.
        refresh_after_reclaim();
        if (try_once()) return true;
        break;
      }
    }
    if (!wait || Clock::now() >= deadline || impossible()) break;
    // Recovery at the gate that just refused admission can require a new
    // sample, even if the cached headroom has not increased yet.
    const auto room = room_for_request();
    capacity_waits_.fetch_add(1, std::memory_order_relaxed);
    {
      std::unique_lock<std::mutex> lock(capacity_mutex_);
      const auto now = Clock::now();
      const auto poll_until =
          now < first_wait_end ? first_wait_end : now + milliseconds(100);
      const bool progress =
          capacity_cv_.wait_until(lock, std::min(deadline, poll_until), [&] {
            if (denials.used.changed_since_attempt(
                    used_release_generation_.load(std::memory_order_acquire)) ||
                denials.committed.changed_since_attempt(
                    committed_release_generation_.load(
                        std::memory_order_acquire)) ||
                rss_recovered(denials.rss))
              return true;
            const auto current = room_for_request();
            return (denials.rss && current.rss_bytes > room.rss_bytes) ||
                   ((denials.used.latest || denials.committed.latest) &&
                    current.native_bytes > room.native_bytes);
          });
      if (!progress)
        capacity_wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
    }
    // The final timed wait can end before an unannounced RSS drop has been
    // sampled. Retry once after every wake without starting another wait.
    if (try_once()) return true;
    if (Clock::now() >= deadline) break;
    // A release can fill part of the gap before a subsequent owner returns
    // the rest. Keep the original deadline once the request has seen progress.
    if (Clock::now() >= first_wait_end && !has_recovery_progress()) {
      bool exhausted = false;
      if (denials.used.latest || denials.committed.latest) {
        const size_t fixed = fixed_used();
        const size_t capacity = pool_size_.load(std::memory_order_relaxed);
        exhausted = fixed > capacity || request.native_bytes > capacity - fixed;
      }
      const bool waiting_for_rss_proof =
          denials.rss && process_rss_.has_resident_uncertainty();
      const auto &queue = BlockEvictionQueue::get_instance();
      if (!exhausted && !waiting_for_rss_proof) {
        exhausted = std::all_of(
            queue.approximate_queue_sizes_.begin(),
            queue.approximate_queue_sizes_.end(), [](const auto &depth) {
              return depth.load(std::memory_order_acquire) == 0;
            });
      }
      if (exhausted) {
        std::lock_guard<std::mutex> lock(process_reclaimer_mutex_);
        // Queue depth falls before a dirty page is written back. The active
        // reclaim pass can still return capacity after its last dequeue.
        if (!process_reclaimer_ && !process_reclaimer_running_ &&
            queue.active_reclaim_passes_.load(std::memory_order_acquire) == 0 &&
            !has_recovery_progress())
          break;
      }
    }
    previous_shortfall = std::numeric_limits<size_t>::max();
    bg_cv_.notify_one();
  }
  high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool MemoryLimitPool::try_charge_fixed_with_wait(size_t bytes,
                                                 std::atomic<size_t> *counter,
                                                 Backing backing,
                                                 size_t materialization) {
  if (materialization > std::numeric_limits<size_t>::max() - bytes)
    return false;
  const size_t rss_bytes =
      backing == Backing::kPlan ? 0 : bytes + materialization;
  return try_admit(
      {bytes, rss_bytes},
      [&](AdmissionDenials *denials) {
        return try_charge_fixed_once(bytes, counter, backing, materialization,
                                     denials);
      },
      true);
}

bool MemoryLimitPool::try_charge_fixed(size_t bytes,
                                       std::atomic<size_t> *counter,
                                       Backing backing,
                                       size_t materialization) {
  if (materialization > std::numeric_limits<size_t>::max() - bytes)
    return false;
  const size_t rss_bytes =
      backing == Backing::kPlan ? 0 : bytes + materialization;
  return try_admit(
      {bytes, rss_bytes},
      [&](AdmissionDenials *denials) {
        return try_charge_fixed_once(bytes, counter, backing, materialization,
                                     denials);
      },
      false);
}

bool MemoryLimitPool::try_charge_fixed_once(size_t bytes,
                                            std::atomic<size_t> *counter,
                                            Backing backing,
                                            size_t materialization,
                                            AdmissionDenials *denials) {
  if (bytes == 0)
    return !materialization ||
           process_rss_.try_materialize(
               materialization, denials ? &denials->rss : nullptr,
               denials ? &denials->rss_watch : nullptr);
  if (!try_reserve_allocation(bytes, backing, materialization, denials))
    return false;
  counter->fetch_add(bytes, std::memory_order_relaxed);
  // Tiny successful reservations do not need a maintenance wakeup while
  // there is ample headroom; denial and wait paths always request it.
  if (should_background_reclaim() || is_page_full() ||
      mandatory_waiters_.load(std::memory_order_relaxed) != 0) {
    bg_charge_notifications_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
  }
  return true;
}

void MemoryLimitPool::release_buffer(char *buffer, const size_t buffer_size) {
  if (!buffer) {
    const auto previous =
        used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)previous;
    assert(previous >= buffer_size);
    notify_capacity_release();
    return;
  }
  if (!is_cacheable_buffer_size(buffer_size)) {
    ailego_free(buffer);
    size_t committed_prev = release_committed(buffer_size);
    (void)committed_prev;
    assert(committed_prev >= buffer_size);
    notify_capacity_release(/*wake_all=*/false, buffer_size);
    return;
  }
  // Owner retirement can return hundreds of MiB before the paced background
  // maintenance catches up. Keep the existing small reuse reserve, but do
  // not publish excess idle backing only to discard it several seconds later.
  // The returning page is exclusively owned here; cached/pinned pages are
  // never touched. Concurrent returns may overshoot the hint by one page per
  // returner. A discard failure keeps both the backing and its charge.
  const size_t reserve =
      process_rss_.pressure() ? 0 : reserve_margin(capacity());
  size_t committed = committed_size_.load(std::memory_order_relaxed);
  size_t used = used_size_.load(std::memory_order_relaxed);
  size_t released_retained = 0;
  if (committed > used && committed - used >= reserve &&
      retained_idle_bytes_.load(std::memory_order_relaxed) != 0) {
    // Idle mappings and free pages share one reserve. Prefer retiring an
    // exclusively idle mapping before discarding a page in an active reuse
    // cycle; otherwise old mappings can force a discard/refault on every
    // cache eviction. Unmapping still precedes any backing-budget refund.
    released_retained = trim_retained_blocks(buffer_size);
    committed = committed_size_.load(std::memory_order_relaxed);
    used = used_size_.load(std::memory_order_relaxed);
  }
  const bool discarded = committed > used && committed - used >= reserve &&
                         reclaim_slab_buffer(buffer);
  if (!discarded) {
    push_free_buffer(buffer, pick_shard());
  }
  // Publish reusable backing or physically discard it before releasing used.
  notify_capacity_release(released_retained != 0, buffer_size);
}

void MemoryLimitPool::cancel_buffer_reservation(size_t bytes,
                                                bool attempted_backing) {
  if (!attempted_backing) {
    const auto previous =
        committed_size_.fetch_sub(bytes, std::memory_order_relaxed);
    (void)previous;
    assert(previous >= bytes);
    process_rss_.cancel_unattempted_reservation(bytes);
    release_buffer(nullptr, bytes);
    return;
  }
  release_committed(bytes, Backing::kPending);
  notify_capacity_release(/*wake_all=*/false, bytes);
}

void MemoryLimitPool::notify_capacity_release(bool wake_all,
                                              size_t released_used) {
  wake_capacity_waiters(publish_capacity_release(wake_all, released_used));
}

bool MemoryLimitPool::publish_capacity_release(bool wake_all,
                                               size_t released_used) {
  std::lock_guard<std::mutex> lock(capacity_mutex_);
  if (released_used != 0)
    release_native_locked(released_used, used_size_, used_release_generation_);
  // The same CV serves different admission gates. An unrelated waiter
  // must not consume the only notification while a required call waits.
  return wake_all || mandatory_waiters_.load(std::memory_order_relaxed) != 0;
}

void MemoryLimitPool::wake_capacity_waiters(bool wake_all) {
  if (wake_all) {
    capacity_cv_.notify_all();
  } else {
    capacity_cv_.notify_one();
  }
}

void MemoryLimitPool::release_external(const size_t buffer_size) {
  release_fixed(buffer_size, &external_used_size_);
}

void MemoryLimitPool::release_heap_external(
    size_t bytes, ProcessRssBudget::HeapProjection projection) {
  if (!bytes) return;
  release_native_used(bytes);
  const size_t external =
      external_used_size_.fetch_sub(bytes, std::memory_order_relaxed);
  assert(external >= bytes);
  (void)external;
  release_native(bytes, committed_size_, committed_release_generation_);
  process_rss_.release_heap(bytes, projection);
  notify_capacity_release(/*wake_all=*/true);
}

void MemoryLimitPool::release_unmapped_external(
    size_t bytes, ProcessRssBudget::MaterializationStamp stamp) {
  release_fixed(bytes, &external_used_size_, Backing::kUnmapped,
                ProcessRssBudget::MappingProjection::Full(bytes, stamp));
}

void MemoryLimitPool::release_unmapped_cache(
    size_t bytes, ProcessRssBudget::MaterializationStamp stamp) {
  const size_t previous =
      evictable_cache_size_.fetch_sub(bytes, std::memory_order_relaxed);
  assert(previous >= bytes);
  (void)previous;
  release_unmapped_external(bytes, stamp);
}

void MemoryLimitPool::request_cache_reclaim(size_t bytes) {
  if (!bytes || evictable_cache_used() == 0) return;
  size_t previous = cache_reclaim_requested_.load(std::memory_order_relaxed);
  while (previous < bytes && !cache_reclaim_requested_.compare_exchange_weak(
                                 previous, bytes, std::memory_order_relaxed)) {
  }
  bg_cv_.notify_one();
}

void MemoryLimitPool::register_cache_mapping(CacheBuffer *buffer) {
  std::lock_guard<std::mutex> lock(cache_mapping_mutex_);
  CacheBuffer *previous = nullptr;
  auto *next = cache_mappings_;
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer->data);
  while (next && reinterpret_cast<uintptr_t>(next->data) < address) {
    previous = next;
    next = next->mapped_next;
  }
  buffer->mapped_prev = previous;
  buffer->mapped_next = next;
  if (previous)
    previous->mapped_next = buffer;
  else
    cache_mappings_ = buffer;
  if (next) next->mapped_prev = buffer;
}

bool MemoryLimitPool::unmap_cache_mapping(CacheBuffer *buffer) {
  std::lock_guard<std::mutex> lock(cache_mapping_mutex_);
  if (!File::TryMemoryUnmap(buffer->data, buffer->bytes)) return false;
  if (buffer->mapped_prev)
    buffer->mapped_prev->mapped_next = buffer->mapped_next;
  else
    cache_mappings_ = buffer->mapped_next;
  if (buffer->mapped_next)
    buffer->mapped_next->mapped_prev = buffer->mapped_prev;
  buffer->mapped_prev = buffer->mapped_next = nullptr;
  return true;
}

MemoryLimitPool::WorkspaceBuffer *MemoryLimitPool::try_acquire_workspace_buffer(
    size_t bytes, bool required) {
  const size_t page = MemoryHelper::PageSize();
  if (!bytes || bytes > std::numeric_limits<size_t>::max() - (page - 1))
    return nullptr;
  bytes = (bytes + page - 1) / page * page;
  PendingAllocation pending(
      *this, required ? PendingAllocation::Kind::kWorkspace
                      : PendingAllocation::Kind::kLockedExternal);
  PendingAllocation metadata(*this, PendingAllocation::Kind::kLockedMetadata);
  if (!pending.reserve_with_metadata(bytes, metadata, sizeof(WorkspaceBuffer)))
    return nullptr;
  auto *buffer = new (std::nothrow) WorkspaceBuffer();
  if (!buffer) return nullptr;
  buffer->data = static_cast<uint8_t *>(File::MemoryMap(bytes, 0));
  if (!buffer->data) {
    delete buffer;
    return nullptr;
  }
  buffer->bytes = bytes;
  buffer->stamp = pending.commit_mapping();
  metadata.commit();
  // A mapping can witness residency only after its pending debit commits.
  // The existing sampler counts private resident pages, not virtual capacity.
  register_cache_mapping(buffer);
  return buffer;
}

bool MemoryLimitPool::release_workspace_buffer(WorkspaceBuffer *buffer) {
  if (!buffer) return true;
  if (!unmap_cache_mapping(buffer)) return false;
  release_unmapped_external(buffer->bytes, buffer->stamp);
  delete buffer;
  release_metadata(sizeof(WorkspaceBuffer));
  return true;
}

MemoryLimitPool::CacheBuffer *MemoryLimitPool::try_acquire_cache_buffer(
    size_t bytes) {
  const size_t page = page_buffer_size();
  if (!bytes || bytes > std::numeric_limits<size_t>::max() - (page - 1))
    return nullptr;
  bytes = (bytes + page - 1) / page * page;
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  process_rss_.refresh();
  const size_t used = this->used(), limit = high_watermark();
  if (process_rss_.pressure() || mandatory_waiters_.load() != 0 ||
      used > limit || bytes > limit - used ||
      sizeof(CacheBuffer) > limit - used - bytes || !try_reserve_used(bytes))
    return nullptr;
  auto &bin = cache_buffer_bins_[(bytes / page) % cache_buffer_bins_.size()];
  {
    std::lock_guard<std::mutex> lock(bin.mutex);
    for (auto **link = &bin.head; *link; link = &(*link)->next) {
      auto *buffer = *link;
      if (buffer->bytes != bytes) continue;
      if (!try_reserve_used(sizeof(CacheBuffer))) {
        release_buffer(nullptr, bytes);
        return nullptr;
      }
      if (!process_rss_.try_materialize_reusable(bytes, buffer->stamp)) {
        release_buffer(nullptr, bytes + sizeof(CacheBuffer));
        return nullptr;
      }
      *link = buffer->next;
      buffer->next = nullptr;
      cache_idle_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      external_used_size_.fetch_add(bytes, std::memory_order_relaxed);
      evictable_cache_size_.fetch_add(bytes, std::memory_order_relaxed);
      metadata_used_size_.fetch_add(sizeof(CacheBuffer),
                                    std::memory_order_relaxed);
      return buffer;
    }
  }
  // Descriptor metadata is charged before allocation. Neither this path nor
  // idle trimming walks cache owners or waits for cache replacement.
  if (!try_charge_locked(sizeof(CacheBuffer), &metadata_used_size_,
                         Backing::kPending)) {
    release_buffer(nullptr, bytes);
    return nullptr;
  }
  auto *buffer = new (std::nothrow) CacheBuffer();
  if (!buffer) {
    release_fixed(sizeof(CacheBuffer), &metadata_used_size_, Backing::kPending);
    release_buffer(nullptr, bytes);
    return nullptr;
  }
  process_rss_.finish_reservation(sizeof(CacheBuffer), true);
  if (!try_reserve_committed(bytes, Backing::kPending)) {
    // Optional admission may discard idle backing, never a pinned owner.
    trim_free_buffers(bytes);
    if (!try_reserve_committed(bytes, Backing::kPending)) {
      delete buffer;
      release_metadata(sizeof(CacheBuffer));
      release_buffer(nullptr, bytes);
      return nullptr;
    }
  }
  buffer->data = static_cast<uint8_t *>(File::MemoryMap(bytes, 0));
  if (!buffer->data) {
    cancel_buffer_reservation(bytes, /*attempted_backing=*/true);
    delete buffer;
    release_metadata(sizeof(CacheBuffer));
    return nullptr;
  }
  buffer->bytes = bytes;
  buffer->stamp = process_rss_.finish_reservation(bytes, true);
  buffer->creation = buffer->stamp;
  // Publish only after its backing debit is committed. A pending mapping
  // must not provide overlap credit for another, possibly untouched owner.
  register_cache_mapping(buffer);
  external_used_size_.fetch_add(bytes, std::memory_order_relaxed);
  evictable_cache_size_.fetch_add(bytes, std::memory_order_relaxed);
  return buffer;
}

void MemoryLimitPool::release_cache_buffer(CacheBuffer *buffer) {
  if (!buffer) return;
  const size_t bytes = buffer->bytes;
  bool wake_all;
  {
    std::lock_guard<std::mutex> return_lock(cache_return_mutex_);
    const size_t committed = committed_size_.load(std::memory_order_relaxed);
    const size_t used = used_size_.load(std::memory_order_relaxed);
    const size_t idle = committed > used ? committed - used : 0;
    const size_t reserve =
        process_rss_.pressure() ? 0 : reserve_margin(capacity());
    // Match retained Arrow blocks and free pages: an entire released query
    // mapping must not leave excess backing resident until paced maintenance
    // catches up. The caller owns this mapping exclusively; live cache owners
    // and escaped/pinned views cannot enter this path.
    if ((idle > reserve || bytes > reserve - idle) &&
        unmap_cache_mapping(buffer)) {
      release_committed(bytes, Backing::kUnmapped,
                        ProcessRssBudget::MappingProjection::ForReuse(
                            bytes, buffer->stamp, buffer->creation));
      delete buffer;
      release_committed(sizeof(CacheBuffer));
    } else {
      auto &bin = cache_buffer_bins_[(bytes / page_buffer_size()) %
                                     cache_buffer_bins_.size()];
      // Publish reusable backing before making its logical capacity available.
      // A failed physical release keeps both the backing and its charge.
      std::lock_guard<std::mutex> lock(bin.mutex);
      buffer->next = bin.head;
      bin.head = buffer;
      cache_idle_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    evictable_cache_size_.fetch_sub(bytes, std::memory_order_relaxed);
    external_used_size_.fetch_sub(bytes, std::memory_order_relaxed);
    metadata_used_size_.fetch_sub(sizeof(CacheBuffer),
                                  std::memory_order_relaxed);
    wake_all = publish_capacity_release(/*wake_all=*/false,
                                        bytes + sizeof(CacheBuffer));
  }
  // Keep allocator-return locks out of the capacity-waiter notification path.
  wake_capacity_waiters(wake_all);
  bg_cv_.notify_one();
}

size_t MemoryLimitPool::trim_cache_buffers(size_t bytes_needed) {
  size_t released = 0;
  for (auto &bin : cache_buffer_bins_) {
    while (released < bytes_needed) {
      CacheBuffer *buffer;
      {
        std::lock_guard<std::mutex> lock(bin.mutex);
        buffer = bin.head;
        if (!buffer) break;
        bin.head = buffer->next;
      }
      if (!unmap_cache_mapping(buffer)) {
        std::lock_guard<std::mutex> lock(bin.mutex);
        buffer->next = bin.head;
        bin.head = buffer;
        break;
      }
      const size_t bytes = buffer->bytes;
      release_committed(bytes, Backing::kUnmapped,
                        ProcessRssBudget::MappingProjection::ForReuse(
                            bytes, buffer->stamp, buffer->creation));
      cache_idle_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      delete buffer;
      // Idle descriptors have backing charges but no active logical owner.
      release_committed(sizeof(CacheBuffer));
      released += bytes;
    }
    if (released >= bytes_needed) break;
  }
  return released;
}

void MemoryLimitPool::release_metadata(const size_t buffer_size) {
  release_fixed(buffer_size, &metadata_used_size_);
}

void MemoryLimitPool::release_unmapped_metadata(
    size_t bytes, ProcessRssBudget::MaterializationStamp stamp) {
  release_fixed(bytes, &metadata_used_size_, Backing::kUnmapped,
                ProcessRssBudget::MappingProjection::Full(bytes, stamp));
}

void MemoryLimitPool::release_fixed(
    size_t buffer_size, std::atomic<size_t> *counter, Backing backing,
    ProcessRssBudget::MappingProjection projection) {
  if (buffer_size == 0) {
    return;
  }
  release_native_used(buffer_size);
  size_t counter_prev =
      counter->fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)counter_prev;
  assert(counter_prev >= buffer_size);
  size_t committed_prev = release_committed(buffer_size, backing, projection);
  (void)committed_prev;
  assert(committed_prev >= buffer_size);
  notify_capacity_release(/*wake_all=*/true);
}

bool MemoryLimitPool::is_full() {
  return used_size_.load(std::memory_order_relaxed) >=
         pool_size_.load(std::memory_order_relaxed);
}

size_t MemoryLimitPool::batch_acquire_buffers(size_t buffer_size, char **out,
                                              size_t count) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  if (count == 0 || buffer_size == 0) return 0;
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  const bool cacheable = is_cacheable_buffer_size(buffer_size);
  const size_t admission_limit = cacheable ? page_admission_limit() : capacity;
  process_rss_.refresh();
  if (cacheable && process_rss_.pressure()) return 0;
  size_t total_size = 0;
  size_t actual_count = count;
  size_t expected, desired;
  do {
    expected = used_size_.load(std::memory_order_relaxed);
    if (expected >= admission_limit) return 0;
    size_t avail = (admission_limit - expected) / buffer_size;
    if (avail == 0) return 0;
    if (avail < actual_count) actual_count = avail;
    total_size = actual_count * buffer_size;
    desired = expected + total_size;
  } while (!used_size_.compare_exchange_weak(
      expected, desired, std::memory_order_relaxed, std::memory_order_relaxed));

  size_t acquired = 0;
  size_t s = pick_shard();
  if (cacheable) {
    while (acquired < actual_count) {
      out[acquired] = pop_reusable_buffer(s, buffer_size);
      if (!out[acquired]) {
        break;
      }
      ++acquired;
    }
    alloc_from_freelist_.fetch_add(acquired, std::memory_order_relaxed);
  }

  while (acquired < actual_count) {
    if (!try_reserve_committed(buffer_size, Backing::kPending)) {
      trim_free_buffers(buffer_size);
      if (!try_reserve_committed(buffer_size, Backing::kPending)) {
        break;
      }
    }
    size_t reserved_pages = actual_count - acquired;
    const auto allocation =
        cacheable ? acquire_slab_buffer(&reserved_pages)
                  : PageAllocation{static_cast<char *>(ailego_aligned_malloc(
                                       buffer_size, kMinimumPageBytes)),
                                   true};
    actual_count = acquired + reserved_pages;
    out[acquired] = allocation.buffer;
    if (!out[acquired]) {
      cancel_buffer_reservation(buffer_size, allocation.attempted_backing);
      --actual_count;
      break;
    }
    if (!cacheable) process_rss_.finish_reservation(buffer_size, true);
    alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
    ++acquired;
  }
  if (acquired < actual_count) {
    release_buffer(nullptr, (actual_count - acquired) * buffer_size);
  }
  return acquired;
}

MemoryLimitPool::PoolStats MemoryLimitPool::stats() const {
  PoolStats s;
  s.pool_size = pool_size_.load(std::memory_order_relaxed);
  s.used = used_size_.load(std::memory_order_relaxed);
  s.committed = committed_size_.load(std::memory_order_relaxed);
  s.external_used = external_used_size_.load(std::memory_order_relaxed);
  s.metadata_used = metadata_used_size_.load(std::memory_order_relaxed);
  s.queue_metadata_used = queue_metadata_used();
  s.queue_allocated = queue_allocated();
  s.retained_block_bytes =
      retained_block_bytes_.load(std::memory_order_relaxed);
  s.retained_idle_bytes = retained_idle_bytes_.load(std::memory_order_relaxed);
  s.retained_block_reuses =
      retained_block_reuses_.load(std::memory_order_relaxed);
  const size_t fixed = s.external_used + s.metadata_used;
  s.page_used = s.used >= fixed ? s.used - fixed : 0;
  size_t free_buffers = 0;
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    free_buffers += free_shards_[i].count.load(std::memory_order_relaxed);
  }
  s.free_buffers = free_buffers;
  s.slab_count = slab_count_.load(std::memory_order_relaxed);
  s.slab_mapped_bytes = slab_mapped_bytes_.load(std::memory_order_relaxed);
  s.slab_header_bytes = s.slab_count * page_buffer_size();
  s.alloc_from_freelist = alloc_from_freelist_.load(std::memory_order_relaxed);
  s.alloc_from_slab = alloc_from_slab_.load(std::memory_order_relaxed);
  s.slab_reclaimed_pages =
      slab_reclaimed_pages_.load(std::memory_order_relaxed);
  s.bg_evict_rounds = bg_evict_rounds_.load(std::memory_order_relaxed);
  s.bg_charge_notifications =
      bg_charge_notifications_.load(std::memory_order_relaxed);
  s.bg_evicted_buffers = bg_evicted_buffers_.load(std::memory_order_relaxed);
  s.bg_no_progress_sleeps =
      bg_no_progress_sleeps_.load(std::memory_order_relaxed);
  s.high_watermark_hits = high_watermark_hits_.load(std::memory_order_relaxed);
  s.capacity_waits = capacity_waits_.load(std::memory_order_relaxed);
  s.capacity_wait_timeouts =
      capacity_wait_timeouts_.load(std::memory_order_relaxed);
  return s;
}

void MemoryLimitPool::log_stats() const {
  PoolStats s = stats();
  LOG_INFO(
      "Shared cache stats: capacity=%llu used=%llu committed=%llu "
      "page_used=%llu external_used=%llu metadata_used=%llu "
      "queue_metadata_used=%llu queue_allocated=%llu "
      "free_buffers=%llu slab_count=%llu slab_mapped_bytes=%llu "
      "slab_header_bytes=%llu "
      "alloc_from_freelist=%llu alloc_from_slab=%llu "
      "slab_reclaimed_pages=%llu "
      "bg_evict_rounds=%llu bg_evicted_buffers=%llu "
      "bg_charge_notifications=%llu "
      "bg_no_progress_sleeps=%llu "
      "high_watermark_hits=%llu capacity_waits=%llu "
      "capacity_wait_timeouts=%llu",
      static_cast<unsigned long long>(s.pool_size),
      static_cast<unsigned long long>(s.used),
      static_cast<unsigned long long>(s.committed),
      static_cast<unsigned long long>(s.page_used),
      static_cast<unsigned long long>(s.external_used),
      static_cast<unsigned long long>(s.metadata_used),
      static_cast<unsigned long long>(s.queue_metadata_used),
      static_cast<unsigned long long>(s.queue_allocated),
      static_cast<unsigned long long>(s.free_buffers),
      static_cast<unsigned long long>(s.slab_count),
      static_cast<unsigned long long>(s.slab_mapped_bytes),
      static_cast<unsigned long long>(s.slab_header_bytes),
      static_cast<unsigned long long>(s.alloc_from_freelist),
      static_cast<unsigned long long>(s.alloc_from_slab),
      static_cast<unsigned long long>(s.slab_reclaimed_pages),
      static_cast<unsigned long long>(s.bg_evict_rounds),
      static_cast<unsigned long long>(s.bg_evicted_buffers),
      static_cast<unsigned long long>(s.bg_charge_notifications),
      static_cast<unsigned long long>(s.bg_no_progress_sleeps),
      static_cast<unsigned long long>(s.high_watermark_hits),
      static_cast<unsigned long long>(s.capacity_waits),
      static_cast<unsigned long long>(s.capacity_wait_timeouts));
}

}  // namespace ailego
}  // namespace zvec
