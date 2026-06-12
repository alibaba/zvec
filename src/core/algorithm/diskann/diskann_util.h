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

#include <iostream>
#include <zvec/core/framework/index_framework.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

class DiskAnnUtil {
 public:
  static constexpr uint64_t kSectorSize = 4096;
  static constexpr uint64_t kMaxSectorReadNum = 128;

 public:
  static inline size_t div_round_up(size_t x, size_t y) {
    return (x / y + (x % y != 0));
  }

  static inline size_t round_up(size_t x, size_t y) {
    return div_round_up(x, y) * y;
  }

  static inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = ::aligned_alloc(align, size);
  }

  static inline void free_aligned(void *ptr) {
    if (ptr == nullptr) {
      return;
    }

    free(ptr);
  }

  template <typename T>
  static inline void convert_vector_to_residual(T *data, uint32_t blocksize_,
                                                size_t dim, void *centroid) {
    const T *centroid_ptr = reinterpret_cast<const T *>(centroid);
    for (size_t i = 0; i < blocksize_; i++) {
      for (uint64_t d = 0; d < dim; d++) {
        float data_float = data[i * dim + d];
        data_float -= centroid_ptr[d];
        data[i * dim + d] = data_float;
      }
    }
  }

  static inline void convert_types_uint32_to_uint8(const uint32_t *src,
                                                   uint8_t *dest, size_t npts,
                                                   size_t dim) {
    for (size_t i = 0; i < npts; i++) {
      for (size_t j = 0; j < dim; j++) {
        dest[i * dim + j] = src[i * dim + j];
      }
    }
  }

  static inline uint64_t get_node_sector(uint32_t node_per_sector,
                                         uint32_t max_nodesize_,
                                         uint32_t sectorsize_,
                                         diskann_id_t node_id) {
    return (node_per_sector > 0
                ? node_id / node_per_sector
                : node_id * div_round_up(max_nodesize_, sectorsize_));
  }

  static inline uint32_t *offset_to_node_neighbor(uint8_t *node_buf,
                                                  uint32_t elementsize_) {
    return (uint32_t *)(node_buf + elementsize_);
  }

  static inline uint8_t *offset_to_node(uint32_t node_per_sector,
                                        uint32_t max_nodesize_,
                                        uint8_t *sector_buf,
                                        diskann_id_t node_id) {
    return sector_buf + (node_per_sector == 0
                             ? 0
                             : (node_id % node_per_sector) * max_nodesize_);
  }

  static inline const uint8_t *offset_to_node_const(uint32_t node_per_sector,
                                                    uint32_t max_nodesize_,
                                                    const uint8_t *sector_buf,
                                                    diskann_id_t node_id) {
    return sector_buf + (node_per_sector == 0
                             ? 0
                             : (node_id % node_per_sector) * max_nodesize_);
  }
};

//! Neighbor
struct Neighbor {
 public:
  Neighbor() = default;

  Neighbor(diskann_id_t id, float distance)
      : id{id}, distance{distance}, expanded(false) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance ||
           (distance == other.distance && id < other.id);
  }

  inline bool operator==(const Neighbor &other) const {
    return (id == other.id);
  }

 public:
  diskann_id_t id;
  float distance;
  bool expanded;
};

//! NeighborPriorityQueue
class NeighborPriorityQueue {
 public:
  NeighborPriorityQueue() : size_(0), capacity_(0), cur_(0) {}

  explicit NeighborPriorityQueue(size_t capacity)
      : size_(0), capacity_(capacity), cur_(0), data_(capacity + 1) {}

  void insert(const Neighbor &nbr) {
    if (size_ == capacity_ && data_[size_ - 1] < nbr) {
      return;
    }

    size_t low = 0, high = size_;
    while (low < high) {
      size_t mid = (low + high) >> 1;
      if (nbr < data_[mid]) {
        high = mid;
      } else if (data_[mid].id == nbr.id) {
        return;
      } else {
        low = mid + 1;
      }
    }

    if (low < capacity_) {
      std::memmove(&data_[low + 1], &data_[low],
                   (size_ - low) * sizeof(Neighbor));
    }

    data_[low] = {nbr.id, nbr.distance};
    if (size_ < capacity_) {
      size_++;
    }

    if (low < cur_) {
      cur_ = low;
    }
  }

  Neighbor closest_unexpanded() {
    data_[cur_].expanded = true;
    size_t pre = cur_;
    while (cur_ < size_ && data_[cur_].expanded) {
      cur_++;
    }
    return data_[pre];
  }

  bool has_unexpanded_node() const {
    return cur_ < size_;
  }

  size_t size() const {
    return size_;
  }

  size_t capacity() const {
    return capacity_;
  }

  void reserve(size_t capacity) {
    if (capacity + 1 > data_.size()) {
      data_.resize(capacity + 1);
    }
    capacity_ = capacity;
  }

  Neighbor &operator[](size_t i) {
    return data_[i];
  }

  Neighbor operator[](size_t i) const {
    return data_[i];
  }

  void sort() {
    std::sort(data_.begin(), data_.begin() + size_);
  }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

 private:
  size_t size_;
  size_t capacity_;
  size_t cur_;
  std::vector<Neighbor> data_;
};

}  // namespace core
}  // namespace zvec