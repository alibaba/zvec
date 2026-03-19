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
#pragma once

#include "hnsw_rabitq_chunk.h"

namespace zvec {
namespace core {

//! Persistent hashmap implement through open addressing algorithm
template <class Key, class Val, Val EmptyVal = 0U,
          typename =
              typename std::enable_if<std::is_integral<Key>::value>::type>
class HnswIndexHashMap {
  using key_type = Key;
  using val_type = Val;

  struct Iterator {
    key_type first;
    val_type second;
  };
  typedef Iterator *iterator;
  typedef Iterator Item;
  typedef const Iterator *const_iterator;

  class Slot {
   public:
    Slot(Chunk::Pointer &&chunk, const void *data)
        : chunk_(std::move(chunk)),
          items_(reinterpret_cast<const Item *>(data)) {}
    //! Return a empty loc or the key item loc

    Slot(Chunk::Pointer &&chunk, IndexStorage::MemoryBlock &&mem_block)
        : chunk_(std::move(chunk)), items_block_(std::move(mem_block)) {
      items_ = reinterpret_cast<const Item *>(items_block_.data());
    }
    const_iterator find(key_type key, uint32_t max_items, uint32_t mask) const {
      auto it = &items_[key & mask];
      for (auto i = 0U; i < max_items; ++i) {
        if (it->first == key || it->second == EmptyVal) {
          // LOG_DEBUG("i=%u", i);
          return it;
        }
        ++it;
        if (it == &items_[max_items]) {
          it = &items_[0];
        }
      }
      return nullptr;
    }

    bool update(const_iterator it) {
      uint32_t offset = reinterpret_cast<const uint8_t *>(it) -
                        reinterpret_cast<const uint8_t *>(&items_[0]);
      if (ailego_unlikely(chunk_->write(offset, it, sizeof(Item)) !=
                          sizeof(Item))) {
        LOG_ERROR("Chunk write failed");
        return false;
      }
      return true;
    }

   private:
    Chunk::Pointer chunk_{};
    const Item *items_{nullptr};  // point to chunk data
    IndexStorage::MemoryBlock items_block_{};
  };

 public:
  //! Init the hash
  //! broker      the index allocator
  //! chunk_size  the size of per chunk allocated, actual size may greater
  //! factor      factor = 1/ratio, ratio is the probability of a squence
  //! number inserted to this container
  //! max         the max number key can be inserted
  //! expansion_ratio   memory expansion ratio
  int init(HnswRabitqChunkBroker::Pointer &broker, uint32_t chunk_size,
           uint32_t factor, size_t max, float expansion_ratio) {
    ailego_assert_with(expansion_ratio > 1.0f, "ratio must > 1.0f");
    broker_ = broker;

    size_t items = std::ceil(chunk_size * 1.0f / sizeof(Item));
    slot_items_ = 1UL << static_cast<size_t>((std::ceil(std::log2(items))));
    size_t range = slot_items_ * factor / expansion_ratio;
    mask_bits_ = std::floor(std::log2(range));
    range = 1UL << mask_bits_;
    size_t max_slots = std::ceil(max * 1.0f / range);
    slots_.reserve(max_slots);
    slot_loc_mask_ = slot_items_ - 1U;
    int ret = load();
    if (ret != 0) {
      return ret;
    }

    LOG_DEBUG(
        "HnswRabitqIndexHash init, chunkSize=%u factor=%u max=%zu "
        "ratio=%f slotItems=%u maxSlots=%zu maskBits=%u "
        "range=%zu",
        chunk_size, factor, max, expansion_ratio, slot_items_, max_slots,
        mask_bits_, range);

    return 0;
  }

  int cleanup(void) {
    broker_.reset();
    slots_.clear();
    slots_.shrink_to_fit();
    mask_bits_ = 0U;
    slot_items_ = 0U;
    slot_loc_mask_ = 0U;

    return 0;
  }

  const_iterator end(void) const {
    return nullptr;
  }

  const_iterator find(const key_type key) const {
    auto idx = key >> mask_bits_;
    if (idx >= slots_.size()) {
      return end();
    }
    auto it = slots_[idx].find(key, slot_items_, slot_loc_mask_);
    return it && it->second != EmptyVal ? it : nullptr;
  }

  bool insert(key_type key, val_type val) {
    auto idx = key >> mask_bits_;
    if (idx >= slots_.size()) {
      if (ailego_unlikely(idx >= slots_.capacity())) {
        LOG_ERROR("no space to insert");
        return false;
      }
      for (auto i = slots_.size(); i <= idx; ++i) {
        if (ailego_unlikely(!alloc_slot(i))) {
          return false;
        }
      }
    }
    auto it = slots_[idx].find(key, slot_items_, slot_loc_mask_);
    if (ailego_unlikely(it == nullptr)) {
      LOG_ERROR("no space to insert");
      return false;
    }

    //! TODO: write memory is ok?
    const_cast<iterator>(it)->first = key;
    const_cast<iterator>(it)->second = val;

    return slots_[idx].update(it);
  }

 private:
  bool alloc_slot(size_t idx) {
    ailego_assert_with(idx == slots_.size(), "invalid idx");

    size_t size = slot_items_ * sizeof(Item);
    auto p = broker_->alloc_chunk(
        HnswRabitqChunkBroker::CHUNK_TYPE_NEIGHBOR_INDEX, idx, size);
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc data chunk failed");
      return false;
    }
    Chunk::Pointer chunk = p.second;
    if (ailego_unlikely(chunk->resize(size) != size)) {
      LOG_ERROR("Chunk resize failed, size=%zu", size);
      return false;
    }
    //! Read the whole data to memory
    IndexStorage::MemoryBlock data_block;
    if (ailego_unlikely(chunk->read(0U, data_block, size) != size)) {
      LOG_ERROR("Chunk read failed, size=%zu", size);
      return false;
    }

    slots_.emplace_back(std::move(chunk), std::move(data_block));
    return true;
  }

  int load(void) {
    size_t slots_cnt = broker_->get_chunk_cnt(
        HnswRabitqChunkBroker::CHUNK_TYPE_NEIGHBOR_INDEX);
    for (size_t i = 0UL; i < slots_cnt; ++i) {
      auto chunk = broker_->get_chunk(
          HnswRabitqChunkBroker::CHUNK_TYPE_NEIGHBOR_INDEX, i);
      if (!chunk) {
        LOG_ERROR("Get chunk failed, seq=%zu", i);
        return IndexError_InvalidFormat;
      }
      size_t size = sizeof(Item) * slot_items_;
      if (chunk->data_size() < size) {
        LOG_ERROR(
            "Hash params may be mismatch, seq=%zu, data_size=%zu "
            "expect=%zu",
            i, chunk->data_size(), size);
        return IndexError_InvalidFormat;
      }
      //! Read the whole data to memory
      IndexStorage::MemoryBlock data_block;
      if (ailego_unlikely(chunk->read(0U, data_block, size) != size)) {
        LOG_ERROR("Chunk read failed, size=%zu", size);
        return false;
      }
      slots_.emplace_back(std::move(chunk), std::move(data_block));
    }
    return 0;
  }

 private:
  HnswRabitqChunkBroker::Pointer broker_{};  // chunk broker
  std::vector<Slot> slots_{};
  uint32_t mask_bits_{0U};
  uint32_t slot_items_{};  // must be a power of 2
  uint32_t slot_loc_mask_{};
};

}  // namespace core
}  // namespace zvec
