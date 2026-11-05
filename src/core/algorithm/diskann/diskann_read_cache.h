// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>

namespace zvec {
namespace core {

// A per-file cache for immutable, fixed-size DiskANN records. Clock replacement
// gives repeatedly hit nodes up to three chances to survive one-time scans.
// sets bound both payload and bookkeeping; no per-hit heap allocation or
// whole-index table. I/O happens outside the set lock. Concurrent misses may
// duplicate a read, but never publish partial data or retain caller buffers.
class DiskAnnReadCache {
 public:
  static constexpr size_t kSectorBytes = 4096;

  explicit DiskAnnReadCache(size_t budget, size_t record_bytes = kSectorBytes)
      : record_bytes_(record_bytes) {
    if (record_bytes && budget > sizeof(*this) + sizeof(Set)) {
      const size_t available = budget - sizeof(*this);
      // More ways reduce conflict eviction near the working-set capacity.
      // Small caches shrink the ways before falling back to uncached reads.
      while (ways_ > 1 && record_bytes > (available - sizeof(Set)) / ways_)
        ways_ /= 2;
      if (record_bytes > (available - sizeof(Set)) / ways_) return;
      set_count_ = available / (sizeof(Set) + ways_ * record_bytes);
      if (set_count_) {
        sets_.reset(new Set[set_count_]);
        data_.reset(new uint8_t[set_count_ * ways_ * record_bytes]);
      }
    }
  }

  size_t allocated_bytes() const {
    return set_count_ ? sizeof(*this) +
                            set_count_ * (sizeof(Set) + ways_ * record_bytes_)
                      : sizeof(*this);
  }

  size_t capacity() const {
    return set_count_ * ways_;
  }

  // Copy hits before scheduling I/O. A miss may have copied a prefix, which
  // read() safely overwrites from the same immutable file.
  bool try_read(uint64_t offset, size_t length, void *destination) {
    if (!set_count_ || length % record_bytes_ != 0) return false;
    auto *out = static_cast<uint8_t *>(destination);
    for (size_t consumed = 0; consumed < length; consumed += record_bytes_) {
      const uint64_t key = offset + consumed;
      const size_t slot = hash(key) % set_count_;
      Set &set = sets_[slot];
      std::lock_guard<std::mutex> lock(set.mutex);
      bool found = false;
      for (size_t i = 0; i < ways_; ++i) {
        auto &entry = set.entries[i];
        if (entry.valid && entry.offset == key) {
          std::memcpy(out + consumed, payload(slot, i), record_bytes_);
          if (entry.chances < 3) ++entry.chances;
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
    return true;
  }

  // Non-record requests bypass the cache without reading beyond their range.
  // read_one(offset, length, destination) must fully fill the destination on
  // success and return zero; errors are returned without caching that sector.
  template <typename ReadOne>
  int read(uint64_t offset, size_t length, void *destination,
           const ReadOne &read_one) {
    if (!set_count_ || length % record_bytes_ != 0) {
      return read_one(offset, length, destination);
    }
    auto *out = static_cast<uint8_t *>(destination);
    for (size_t consumed = 0; consumed < length; consumed += record_bytes_) {
      const uint64_t key = offset + consumed;
      if (try_read(key, record_bytes_, out + consumed)) continue;
      const size_t slot = hash(key) % set_count_;
      Set &set = sets_[slot];
      int ret = read_one(key, record_bytes_, out + consumed);
      if (ret != 0) return ret;
      {
        std::lock_guard<std::mutex> lock(set.mutex);
        // Another reader may have published this immutable sector meanwhile.
        bool published = false;
        for (size_t i = 0; i < ways_; ++i) {
          auto &entry = set.entries[i];
          if (entry.valid && entry.offset == key) {
            if (entry.chances < 3) ++entry.chances;
            published = true;
            break;
          }
        }
        if (published) continue;
        while (set.entries[set.hand].valid &&
               set.entries[set.hand].chances) {
          --set.entries[set.hand].chances;
          set.hand = (set.hand + 1) % ways_;
        }
        auto &entry = set.entries[set.hand];
        std::memcpy(payload(slot, set.hand), out + consumed, record_bytes_);
        entry.offset = key;
        entry.valid = true;
        entry.chances = 1;
        set.hand = (set.hand + 1) % ways_;
      }
    }
    return 0;
  }

  // Startup-only process override for the macOS FileReadStorage path. Keep
  // values bounded, and reject signs, trailing text and overflow explicitly.
  static bool parse_budget_mib(const char *value, size_t *bytes) {
    size_t mib = 32;
    if (value) {
      if (!*value) return false;
      mib = 0;
      for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        mib = mib * 10 + static_cast<size_t>(*p - '0');
        if (mib > 1024) return false;
      }
    }
    *bytes = mib * 1024 * 1024;
    return true;
  }

 private:
  struct Entry {
    uint64_t offset{0};
    bool valid{false};
    uint8_t chances{0};
  };
  struct Set {
    std::mutex mutex;
    std::array<Entry, 16> entries;
    size_t hand{0};
  };
  static uint64_t hash(uint64_t key) {
    // Mix the full offset: vector segments need not start at file offset 0.
    key ^= key >> 30;
    key *= UINT64_C(0xbf58476d1ce4e5b9);
    key ^= key >> 27;
    key *= UINT64_C(0x94d049bb133111eb);
    return key ^ (key >> 31);
  }
  uint8_t *payload(size_t set, size_t entry) {
    return data_.get() + (set * ways_ + entry) * record_bytes_;
  }
  size_t set_count_{0};
  size_t ways_{16};
  size_t record_bytes_{kSectorBytes};
  std::unique_ptr<Set[]> sets_;
  std::unique_ptr<uint8_t[]> data_;
};

}  // namespace core
}  // namespace zvec
