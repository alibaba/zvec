// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <limits>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {

// Unit-test view for tests that specify a page/workspace budget, not a total
// process cache budget. Reserve real capacity for the persistent queue first,
// then keep the payload-lifetime assertions independent of retained allocator
// metadata. Page-count tests must also reserve room for slab headers.
// Production and allocator-accounting tests use MemoryLimitPool directly.
class PoolPayloadBudget {
 public:
  static PoolPayloadBudget &get_instance() {
    static PoolPayloadBudget instance;
    return instance;
  }
  static size_t page_buffer_size() {
    return MemoryLimitPool::page_buffer_size();
  }
  static constexpr size_t slab_size() {
    return MemoryLimitPool::slab_size();
  }
  static constexpr size_t slab_alignment() {
    return MemoryLimitPool::slab_alignment();
  }
  using PoolStats = MemoryLimitPool::PoolStats;

  int init(size_t bytes) {
    auto &pool = actual();
    if (bytes != 0 && pool.used() == pool.queue_metadata_used() +
                                         pool.stats().slab_header_bytes) {
      // Create the main test thread's producer metadata before measuring a
      // one-page payload budget. Workers still pay for their own producers.
      if (pool.init(0) != 0) return -1;
      auto &queue = BlockEvictionQueue::get_instance();
      for (size_t i = 0; i < BlockEvictionQueue::kQueueCount; ++i) {
        if (!queue.add_single_block({}, static_cast<int>(i))) return -1;
      }
      BlockEvictionQueue::BlockType stale;
      while (queue.evict_single_block(stale)) {
      }
    }
    const size_t overhead = bytes == 0 ? 0 : pool.queue_allocated();
    if (bytes > std::numeric_limits<size_t>::max() - overhead) return -1;
    return pool.init(bytes + overhead);
  }
  size_t capacity() const {
    return without_allocator_metadata(actual().capacity());
  }
  size_t used() const {
    return without_allocator_metadata(actual().used());
  }
  size_t committed() const {
    return without_allocator_metadata(actual().committed());
  }
  size_t metadata_used() const {
    return without_allocator_metadata(actual().metadata_used());
  }
  size_t external_used() const {
    return actual().external_used();
  }
  size_t available() const {
    return actual().available();
  }
  bool initialized() const {
    return actual().initialized();
  }
  bool is_full() {
    return actual().is_full();
  }
  bool is_page_full() const {
    return actual().is_page_full();
  }
  bool under_cache_pressure() const {
    return actual().under_cache_pressure();
  }
  size_t page_admission_reserve() const {
    return actual().page_admission_reserve();
  }
  bool try_acquire_buffer(size_t bytes, char *&buffer) {
    return actual().try_acquire_buffer(bytes, buffer);
  }
  size_t batch_acquire_buffers(size_t bytes, char **out, size_t count) {
    return actual().batch_acquire_buffers(bytes, out, count);
  }
  bool try_charge_external(size_t bytes) {
    return actual().try_charge_external(bytes);
  }
  bool try_charge_workspace(size_t bytes) {
    return actual().try_charge_workspace(bytes);
  }
  bool try_charge_metadata(size_t bytes) {
    return actual().try_charge_metadata(bytes);
  }
  void release_buffer(char *buffer, size_t bytes) {
    actual().release_buffer(buffer, bytes);
  }
  void release_external(size_t bytes) {
    actual().release_external(bytes);
  }
  void release_metadata(size_t bytes) {
    actual().release_metadata(bytes);
  }
  bool wait_for_available(size_t bytes, std::chrono::milliseconds timeout) {
    return actual().wait_for_available(bytes, timeout);
  }
  PoolStats stats() const {
    auto result = actual().stats();
    const size_t allocator =
        result.queue_metadata_used + result.slab_header_bytes;
    result.pool_size -= std::min(result.pool_size, allocator);
    result.used -= std::min(result.used, allocator);
    result.committed -= std::min(result.committed, allocator);
    result.metadata_used -= std::min(result.metadata_used, allocator);
    return result;
  }

 private:
  static size_t without_allocator_metadata(size_t bytes) {
    // Pool statistics are relaxed snapshots; concurrent producer creation
    // must not turn a temporarily newer queue counter into an underflow.
    const auto stats = actual().stats();
    return bytes -
           std::min(bytes, stats.queue_metadata_used + stats.slab_header_bytes);
  }
  static MemoryLimitPool &actual() {
    return MemoryLimitPool::get_instance();
  }
};
}  // namespace zvec::ailego
