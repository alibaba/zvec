// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

// Value sets for IN and list membership: dense indices of distinct members.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace zvec::stile::internal {

template <typename T>
uint64_t Bits(T value) {
  uint64_t key = 0;
  std::memcpy(&key, &value, sizeof(value));
  return key;
}

// Open addressing on 64-bit keys, sized once for at most `max_keys`, plus
// a bitmap over the key range when it is small (typical integer IN lists),
// which answers contains() with one load and no branch.
class KeyIndex {
 public:
  explicit KeyIndex(size_t max_keys) {
    // Sparse tables keep most misses to one probe.
    const unsigned factor = max_keys <= 1024 ? 4 : 2;
    if (max_keys > std::numeric_limits<int32_t>::max() ||
        max_keys > slots_.max_size() / factor)
      throw std::length_error("set exceeds hash table capacity");
    const size_t slots = factor * max_keys;
    int bits = 3;
    size_t capacity = 8;
    while (capacity < slots) {
      if (capacity > slots_.max_size() / 2)
        throw std::length_error("set exceeds hash table capacity");
      capacity *= 2;
      ++bits;
    }
    shift_ = 64 - bits;
    slots_.assign(capacity, Slot{0, -1});
  }
  int32_t insert(uint64_t key) {
    dense_.clear();
    for (size_t i = home(key);; i = (i + 1) & (slots_.size() - 1)) {
      Slot &slot = slots_[i];
      if (slot.index < 0) {
        slot = Slot{key, size_++};
        low_ = size_ == 1 ? key : std::min(low_, key);
        high_ = size_ == 1 ? key : std::max(high_, key);
      }
      if (slot.key == key) return slot.index;
    }
  }
  // Builds the bitmap when the key range is small; call after inserting.
  void finish() {
    dense_.clear();
    if (size_ == 0) return;
    const uint64_t range = high_ - low_;
    const uint64_t limit = std::max<uint64_t>(65535, 16 * uint64_t(size_));
    if (range > std::min<uint64_t>(limit, uint64_t{1} << 24)) return;
    range_ = range;
    // One more bit, always clear, absorbs keys outside the range.
    dense_.assign(static_cast<size_t>((range + 1) / 64 + 1), 0);
    for (const Slot &slot : slots_) {
      if (slot.index < 0) continue;
      const uint64_t bit = slot.key - low_;
      dense_[bit >> 6] |= uint64_t{1} << (bit & 63);
    }
  }
  bool contains(uint64_t key) const {
    if (!dense_.empty()) {
      const uint64_t bit = std::min(key - low_, range_ + 1);
      return (dense_[bit >> 6] >> (bit & 63)) & 1;
    }
    return find(key) >= 0;
  }
  int32_t find(uint64_t key) const {
    for (size_t i = home(key);; i = (i + 1) & (slots_.size() - 1)) {
      const Slot &slot = slots_[i];
      if (slot.index < 0 || slot.key == key) return slot.index;
    }
  }
  int32_t size() const {
    return size_;
  }

 private:
  struct Slot {
    uint64_t key;
    int32_t index;
  };
  size_t home(uint64_t key) const {
    return static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> shift_);
  }
  std::vector<Slot> slots_;
  int shift_;
  int32_t size_ = 0;
  uint64_t low_ = 0, high_ = 0, range_ = 0;
  std::vector<uint64_t> dense_;
};

class StringIndex {
 public:
  explicit StringIndex(size_t max_keys) {
    if (max_keys > std::numeric_limits<int32_t>::max())
      throw std::length_error("set exceeds member index capacity");
    storage_.reserve(max_keys);  // The map's views must not move.
  }
  int32_t insert(std::string_view key) {
    if (const int32_t found = find(key); found >= 0) return found;
    storage_.emplace_back(key);
    const auto index = static_cast<int32_t>(map_.size());
    map_.emplace(storage_.back(), index);
    return index;
  }
  int32_t find(std::string_view key) const {
    const auto found = map_.find(key);
    return found == map_.end() ? -1 : found->second;
  }
  bool contains(std::string_view key) const {
    return find(key) >= 0;
  }
  void finish() {}
  int32_t size() const {
    return static_cast<int32_t>(map_.size());
  }

 private:
  std::vector<std::string> storage_;
  std::unordered_map<std::string_view, int32_t> map_;
};

template <typename Value>
using IndexFor = std::conditional_t<std::is_same_v<Value, std::string_view>,
                                    StringIndex, KeyIndex>;

}  // namespace zvec::stile::internal
