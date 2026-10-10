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

#include "buffered_input.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "utility/temporary_buffer_storage.h"

namespace zvec::core_interface {
namespace {

// Charge before allocating, including container reallocation peaks and the
// implementation's map node size. These are live bytes, not an RSS promise.
template <class T>
class InputAllocator {
 public:
  using value_type = T;
  InputAllocator() = default;
  template <class U>
  InputAllocator(const InputAllocator<U> &) {}
  T *allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    const size_t bytes = n * sizeof(T);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!pool.try_charge_external(bytes) &&
        !(pool.wait_for_available(bytes, std::chrono::milliseconds(100)) &&
          pool.try_charge_external(bytes))) {
      throw std::bad_alloc();
    }
    try {
      return std::allocator<T>{}.allocate(n);
    } catch (...) {
      pool.release_external(bytes);
      throw;
    }
  }
  void deallocate(T *p, size_t n) {
    std::allocator<T>{}.deallocate(p, n);
    ailego::MemoryLimitPool::get_instance().release_external(n * sizeof(T));
  }
  template <class U>
  bool operator==(const InputAllocator<U> &) const {
    return true;
  }
  template <class U>
  bool operator!=(const InputAllocator<U> &) const {
    return false;
  }
};

template <class T>
using InputVector = std::vector<T, InputAllocator<T>>;

}  // namespace

struct BufferedInput::State {
  using Entry = std::pair<uint32_t, size_t>;
  using Map = std::map<uint32_t, size_t, std::less<uint32_t>,
                       InputAllocator<std::pair<const uint32_t, size_t>>>;
  core::IndexQueryMeta meta;
  std::string prefix;
  core::TemporaryBufferStorage::Pointer file;
  InputVector<core::IndexStorage::Segment::Pointer> segments;
  Map slots;
  InputVector<Entry> ordinals;
  size_t rows_per_segment{0};
  size_t allocated_slots{0};
  size_t spare{std::numeric_limits<size_t>::max()};
  bool prepared{false};

  int read(size_t slot, void *out) const {
    const size_t bytes = meta.element_size();
    const size_t segment = slot / rows_per_segment;
    if (segment >= segments.size()) return core::IndexError_OutOfRange;
    return segments[segment]->fetch((slot % rows_per_segment) * bytes, out,
                                    bytes) == bytes
               ? 0
               : core::IndexError_ReadData;
  }
};

BufferedInput::BufferedInput(core::IndexQueryMeta meta, std::string prefix)
    : state_(std::make_unique<State>()) {
  state_->meta = std::move(meta);
  state_->prefix = std::move(prefix);
  if (element_size() != 0)
    state_->rows_per_segment = std::max<size_t>(1, (4U << 20) / element_size());
}
BufferedInput::~BufferedInput() = default;

int BufferedInput::add(uint32_t key, const void *data) {
  auto &s = *state_;
  if (!data || element_size() == 0) return core::IndexError_InvalidArgument;
  try {
    if (!s.file) {
      int ret = core::TemporaryBufferStorage::CreateEmpty(s.prefix, &s.file);
      if (ret != 0) return ret;
    }
    const size_t slot = s.spare != std::numeric_limits<size_t>::max()
                            ? s.spare
                            : s.allocated_slots;
    const size_t segment = slot / s.rows_per_segment;
    if (segment == s.segments.size()) {
      // Reserve the segment handle before growing the file. If append/get
      // fails, a retry reuses the same segment name rather than losing rows.
      s.segments.reserve(segment + 1);
      const auto name = "input_" + std::to_string(segment);
      auto next = s.file->storage()->get(name);
      if (!next) {
        int ret = s.file->storage()->append(
            name, s.rows_per_segment * element_size());
        if (ret != 0) return ret;
        next = s.file->storage()->get(name);
      }
      if (!next) return core::IndexError_Runtime;
      s.segments.push_back(std::move(next));
    }
    // Insert metadata before I/O, but only publish the new slot after a full
    // write. Overwrites are copy-on-write with one reusable spare slot, so a
    // failed/short write cannot corrupt a previously accepted document.
    auto inserted = s.slots.emplace(key, slot);
    const size_t old = inserted.first->second;
    size_t written = 0;
    try {
      written = s.segments[segment]->write(
          (slot % s.rows_per_segment) * element_size(), data, element_size());
    } catch (...) {
      if (inserted.second) s.slots.erase(inserted.first);
      throw;
    }
    if (written != element_size()) {
      if (inserted.second) s.slots.erase(inserted.first);
      return core::IndexError_WriteData;
    }
    inserted.first->second = slot;
    if (slot == s.allocated_slots) ++s.allocated_slots;
    s.spare = inserted.second ? std::numeric_limits<size_t>::max() : old;
    s.prepared = false;
    InputVector<State::Entry>().swap(s.ordinals);
    return 0;
  } catch (const std::bad_alloc &) {
    return core::IndexError_NoMemory;
  } catch (const std::exception &) {
    return core::IndexError_WriteData;
  }
}

int BufferedInput::fetch(uint32_t key, std::string *out) const {
  if (!out) return core::IndexError_InvalidArgument;
  const auto &s = *state_;
  const auto it = s.slots.find(key);
  if (it == s.slots.end()) {
    return s.slots.empty() || key > s.slots.rbegin()->first
               ? core::IndexError_OutOfRange
               : core::IndexError_NoExist;
  }
  try {
    // The public fetch result is caller-owned. No full-input copy or pin
    // survives this call, just this one row.
    std::string result(element_size(), '\0');
    int ret = s.read(it->second, result.data());
    if (ret == 0) *out = std::move(result);
    return ret;
  } catch (const std::bad_alloc &) {
    return core::IndexError_NoMemory;
  } catch (const std::exception &) {
    return core::IndexError_ReadData;
  }
}

int BufferedInput::prepare() {
  auto &s = *state_;
  if (s.prepared) return 0;
  // Make ingestion pages reclaimable before competing with training scratch.
  // A failed flush leaves the accepted documents available for retry.
  const int ret = flush();
  if (ret != 0) return ret;
  try {
    InputVector<State::Entry> next;
    next.reserve(s.slots.size());
    for (const auto &entry : s.slots)
      next.emplace_back(entry.first, entry.second);
    s.ordinals.swap(next);
    s.prepared = true;
    return 0;
  } catch (const std::bad_alloc &) {
    return core::IndexError_NoMemory;
  }
}

int BufferedInput::flush() {
  return state_->file ? state_->file->flush() : 0;
}
void BufferedInput::clear() {
  auto &s = *state_;
  InputVector<State::Entry>().swap(s.ordinals);
  State::Map().swap(s.slots);
  InputVector<core::IndexStorage::Segment::Pointer>().swap(s.segments);
  s.file.reset();
  s.allocated_slots = 0;
  s.spare = std::numeric_limits<size_t>::max();
  s.prepared = false;
}
size_t BufferedInput::count() const {
  return state_->slots.size();
}
size_t BufferedInput::dimension() const {
  return state_->meta.dimension();
}
core::IndexMeta::DataType BufferedInput::data_type() const {
  return state_->meta.data_type();
}
size_t BufferedInput::element_size() const {
  return state_->meta.element_size();
}

class BufferedInput::Reader : public core::OrdinalAccessHolder::Reader {
 public:
  explicit Reader(std::shared_ptr<BufferedInput> owner)
      : owner_(std::move(owner)) {}
  int read(size_t ordinal, uint64_t *key, const void **data) override {
    if (!key || !data) return core::IndexError_InvalidArgument;
    *data = nullptr;
    const auto &s = *owner_->state_;
    if (!s.prepared) return core::IndexError_NoReady;
    if (ordinal >= s.ordinals.size()) return core::IndexError_OutOfRange;
    try {
      row_.resize(owner_->element_size());
      const auto &entry = s.ordinals[ordinal];
      int ret = s.read(entry.second, row_.data());
      if (ret != 0) return ret;
      *key = entry.first;
      *data = row_.data();
      return 0;
    } catch (const std::bad_alloc &) {
      return core::IndexError_NoMemory;
    } catch (const std::exception &) {
      return core::IndexError_ReadData;
    }
  }
  void reset() override {
    InputVector<char>().swap(row_);
  }

 private:
  std::shared_ptr<BufferedInput> owner_;
  InputVector<char> row_;
};

class BufferedInput::Iterator : public core::IndexHolder::Iterator {
 public:
  explicit Iterator(std::shared_ptr<BufferedInput> owner)
      : owner_(std::move(owner)), reader_(owner_) {}
  bool is_valid() const override {
    return status_ == 0 && ordinal_ < owner_->count();
  }
  int status() const override {
    return status_;
  }
  const void *data() const override {
    if (!is_valid()) return nullptr;
    if (!data_) status_ = reader_.read(ordinal_, &key_, &data_);
    return data_;
  }
  uint64_t key() const override {
    data();
    return key_;
  }
  void next() override {
    ++ordinal_;
    data_ = nullptr;
  }

 private:
  std::shared_ptr<BufferedInput> owner_;
  mutable Reader reader_;
  size_t ordinal_{0};
  mutable uint64_t key_{0};
  mutable const void *data_{nullptr};
  mutable int status_{0};
};

core::IndexHolder::Iterator::Pointer BufferedInput::create_iterator() {
  return std::make_unique<Iterator>(shared_from_this());
}
int BufferedInput::create_ordinal_reader(
    core::OrdinalAccessHolder::Reader::Pointer *out) {
  if (!out) return core::IndexError_InvalidArgument;
  if (!state_->prepared) return core::IndexError_NoReady;
  try {
    *out = std::make_unique<Reader>(shared_from_this());
    return 0;
  } catch (const std::bad_alloc &) {
    return core::IndexError_NoMemory;
  }
}

}  // namespace zvec::core_interface
