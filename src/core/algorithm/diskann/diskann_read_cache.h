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

// A per-file cache for immutable DiskANN node sectors. Fixed four-way clock
// sets bound both payload and bookkeeping; no per-hit heap allocation or
// whole-index table. I/O happens outside the set lock. Concurrent misses may
// duplicate a read, but never publish partial data or retain caller buffers.
class DiskAnnReadCache {
 public:
  static constexpr size_t kSectorBytes = 4096;

  explicit DiskAnnReadCache(size_t budget) {
    if (budget > sizeof(*this)) {
      set_count_ = (budget - sizeof(*this)) / sizeof(Set);
      if (set_count_) sets_.reset(new Set[set_count_]);
    }
  }

  size_t allocated_bytes() const {
    return sizeof(*this) + set_count_ * sizeof(Set);
  }

  // Non-sector requests bypass the cache without reading beyond their range.
  // read_one(offset, length, destination) must fully fill the destination on
  // success and return zero; errors are returned without caching that sector.
  template <typename ReadOne>
  int read(uint64_t offset, size_t length, void *destination,
           const ReadOne &read_one) {
    if (!set_count_ || length % kSectorBytes != 0) {
      return read_one(offset, length, destination);
    }
    auto *out = static_cast<uint8_t *>(destination);
    for (size_t consumed = 0; consumed < length; consumed += kSectorBytes) {
      const uint64_t key = offset + consumed;
      Set &set = sets_[hash(key) % set_count_];
      bool hit = false;
      {
        std::lock_guard<std::mutex> lock(set.mutex);
        for (auto &entry : set.entries) {
          if (entry.valid && entry.offset == key) {
            std::memcpy(out + consumed, entry.data.data(), kSectorBytes);
            entry.referenced = true;
            hit = true;
            break;
          }
        }
      }
      if (hit) continue;
      int ret = read_one(key, kSectorBytes, out + consumed);
      if (ret != 0) return ret;
      {
        std::lock_guard<std::mutex> lock(set.mutex);
        // Another reader may have published this immutable sector meanwhile.
        bool published = false;
        for (auto &entry : set.entries) {
          if (entry.valid && entry.offset == key) {
            entry.referenced = true;
            published = true;
            break;
          }
        }
        if (published) continue;
        while (set.entries[set.hand].valid &&
               set.entries[set.hand].referenced) {
          set.entries[set.hand].referenced = false;
          set.hand = (set.hand + 1) % set.entries.size();
        }
        auto &entry = set.entries[set.hand];
        std::memcpy(entry.data.data(), out + consumed, kSectorBytes);
        entry.offset = key;
        entry.valid = entry.referenced = true;
        set.hand = (set.hand + 1) % set.entries.size();
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
    bool referenced{false};
    std::array<uint8_t, kSectorBytes> data{};
  };
  struct Set {
    std::mutex mutex;
    std::array<Entry, 4> entries;
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
  size_t set_count_{0};
  std::unique_ptr<Set[]> sets_;
};

}  // namespace core
}  // namespace zvec
