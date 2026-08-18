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

#include <cstring>
#include <iostream>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_factory.h>
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
    if (size == 0) {
      *ptr = nullptr;
      return;
    }
    // Unlike aligned_alloc(), posix_memalign() does not require size to be an
    // integral multiple of alignment and is available on Linux and macOS.
    if (::posix_memalign(ptr, align, size) != 0) {
      *ptr = nullptr;
    }
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

  //! Resolve the quantizer implementation name from a serialized quantizer
  //! meta buffer (see turbo::QuantizerSerHeader).  Extend the mapping here when
  //! DiskAnn supports a new quantize type.
  static const char *quantizer_name_from_meta_buffer(
      const std::string &meta_buffer) {
    if (meta_buffer.size() < sizeof(turbo::QuantizerSerHeader)) {
      return nullptr;
    }
    turbo::QuantizerSerHeader hdr;
    std::memcpy(&hdr, meta_buffer.data(), sizeof(hdr));
    if (hdr.magic != turbo::kQuantizerMagic) {
      return nullptr;
    }
    switch (static_cast<turbo::QuantizeType>(hdr.quant_type)) {
      case turbo::QuantizeType::kPQ:
        return "PqInt8Quantizer";
      case turbo::QuantizeType::kFp16:
        return "Fp16Quantizer";
      case turbo::QuantizeType::kFp32:
        return "Fp32Quantizer";
      default:
        return nullptr;
    }
  }

  //! Derive the meta the quantizer is initialized with from the index meta,
  //! mirroring the conversion done at build time: InnerProduct and Cosine
  //! are trained/searched in SquaredEuclidean space.  Two Cosine layouts are
  //! supported: the legacy one folds the stored norm into an inflated
  //! dimension, while the new one keeps the raw dimension and accounts the
  //! norm with extra_meta_size.
  static int quantizer_init_meta(const IndexMeta &meta, IndexMeta *out) {
    *out = meta;
    if (meta.metric_name() == "InnerProduct") {
      out->set_metric("SquaredEuclidean", 0, ailego::Params());
    } else if (meta.metric_name() == "Cosine") {
      out->set_metric("SquaredEuclidean", 0, ailego::Params());

      if (meta.data_type() != IndexMeta::DataType::DT_FP32 &&
          meta.data_type() != IndexMeta::DataType::DT_FP16) {
        LOG_ERROR("Unsupported cosine data type: %u", meta.data_type());
        return IndexError_Unsupported;
      }

      if (meta.extra_meta_size() > 0) {
        // New layout: the dimension is already the raw one; drop the tail
        // so element_size() covers the raw vector only, as in the legacy
        // conversion below.
        out->set_extra_meta_size(0);
      } else if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
        if (meta.dimension() <= 1) {
          LOG_ERROR("Invalid FP32 cosine dimension: %u", meta.dimension());
          return IndexError_InvalidArgument;
        }
        out->set_dimension(meta.dimension() - 1);
      } else {
        if (meta.dimension() <= 2) {
          LOG_ERROR("Invalid FP16 cosine dimension: %u", meta.dimension());
          return IndexError_InvalidArgument;
        }
        out->set_dimension(meta.dimension() - 2);
      }
    }
    return 0;
  }

  //! Construct the quantizer persisted in `meta_buffer`, picking the
  //! implementation from the meta buffer header instead of a hardcoded type.
  //! Contract: the quantizer is initialized with the meta derived from
  //! `index_meta` before deserialize(), so the metric policy comes from the
  //! meta instead of the default-constructed one.
  static turbo::Quantizer::Pointer create_quantizer_from_meta_buffer(
      std::string &meta_buffer, const IndexMeta &index_meta,
      uint32_t chunk_num) {
    const char *name = quantizer_name_from_meta_buffer(meta_buffer);
    if (name == nullptr) {
      LOG_ERROR("Unsupported or corrupted quantizer meta buffer");
      return turbo::Quantizer::Pointer();
    }
    auto quantizer = IndexFactory::CreateQuantizer(name);
    if (!quantizer) {
      LOG_ERROR("Create quantizer %s failed", name);
      return turbo::Quantizer::Pointer();
    }
    IndexMeta init_meta;
    if (quantizer_init_meta(index_meta, &init_meta) != 0) {
      return turbo::Quantizer::Pointer();
    }
    ailego::Params params;
    params.set("num_chunk", chunk_num);
    if (quantizer->init(init_meta, params) != 0) {
      LOG_ERROR("Quantizer %s init failed", name);
      return turbo::Quantizer::Pointer();
    }
    if (quantizer->deserialize(meta_buffer) != 0) {
      LOG_ERROR("Quantizer %s deserialize failed", name);
      return turbo::Quantizer::Pointer();
    }
    return quantizer;
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
