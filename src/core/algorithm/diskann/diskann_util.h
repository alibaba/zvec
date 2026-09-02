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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_entity.h"

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#endif

namespace zvec {
namespace core {

class DiskAnnUtil {
 public:
  static constexpr uint64_t kSectorSize = 4096;
  static constexpr uint64_t kMaxSectorReadNum = 128;

  static constexpr uint64_t cache_load_batch_size(
      uint64_t sector_num_per_node) {
    return sector_num_per_node == 0 || sector_num_per_node > kMaxSectorReadNum
               ? 1
               : kMaxSectorReadNum / sector_num_per_node;
  }

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
#if defined(_WIN32) || defined(_WIN64)
    *ptr = ::_aligned_malloc(size, align);
#else
    // Unlike aligned_alloc(), posix_memalign() does not require size to be an
    // integral multiple of alignment and is available on Linux and macOS.
    if (::posix_memalign(ptr, align, size) != 0) {
      *ptr = nullptr;
    }
#endif
  }

  static inline void free_aligned(void *ptr) {
    if (ptr == nullptr) {
      return;
    }
#if defined(_WIN32) || defined(_WIN64)
    ::_aligned_free(ptr);
#else
    free(ptr);
#endif
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

  //! True when the buffer has no quantizer header: indexes dumped before the
  //! codebook moved into the quantizer stored the raw PQ codebook here.
  static bool is_legacy_pq_meta_buffer(const std::string &meta_buffer) {
    if (meta_buffer.size() < sizeof(uint32_t)) {
      return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, meta_buffer.data(), sizeof(magic));
    return magic != turbo::kQuantizerMagic;
  }

  //! Repack a legacy codebook into the layout the PQ quantizer keeps in memory:
  //! on disk 256 pivot rows of `dim` components, a per-dimension mean and
  //! chunk_num + 1 chunk offsets; in memory [chunk][256][chunk_dim].  Nothing
  //! else needs translating, both sides work in squared euclidean space and the
  //! writers of this layout always persisted a zero mean.
  static int legacy_pq_codebook_to_centroids(const std::string &meta_buffer,
                                             const IndexMeta &quantizer_meta,
                                             uint32_t chunk_num,
                                             std::string *centroids) {
    constexpr size_t kClusterNum = 256;

    const size_t dim = quantizer_meta.dimension();
    const size_t unit = quantizer_meta.unit_size();
    if (dim == 0 || unit == 0 || chunk_num == 0 || dim % chunk_num != 0) {
      // Legacy chunking gave the leading chunks one extra dimension when the
      // count did not divide the dimension; the quantizer only knows uniform
      // chunks.
      LOG_ERROR(
          "Legacy DiskAnn PQ codebook with %u chunks over %zu dimensions is "
          "not loadable, rebuild the index",
          chunk_num, dim);
      return IndexError_Unsupported;
    }
    const size_t sub_dim = dim / chunk_num;

    const size_t pivot_bytes = kClusterNum * dim * unit;
    const size_t mean_bytes = dim * unit;
    const size_t offset_bytes = (chunk_num + 1) * sizeof(uint32_t);
    if (meta_buffer.size() != pivot_bytes + mean_bytes + offset_bytes) {
      LOG_ERROR(
          "Legacy DiskAnn PQ codebook size mismatch, expect: %zu, actual: %zu",
          pivot_bytes + mean_bytes + offset_bytes, meta_buffer.size());
      return IndexError_InvalidFormat;
    }

    const char *pivots = meta_buffer.data();
    const char *mean = pivots + pivot_bytes;
    // Copied out instead of cast in place: the offsets are not guaranteed to be
    // aligned inside the buffer.
    std::vector<uint32_t> offsets(chunk_num + 1, 0);
    std::memcpy(offsets.data(), mean + mean_bytes, offset_bytes);
    for (size_t m = 0; m <= chunk_num; ++m) {
      if (offsets[m] != m * sub_dim) {
        LOG_ERROR(
            "Legacy DiskAnn PQ codebook has non-uniform chunk offsets, rebuild "
            "the index");
        return IndexError_Unsupported;
      }
    }

    // The quantizer is initialized without a mean, so a residual codebook would
    // be decoded against the wrong origin.
    if (std::any_of(mean, mean + mean_bytes,
                    [](char byte) { return byte != 0; })) {
      LOG_ERROR(
          "Legacy DiskAnn PQ codebook carries a non-zero mean, rebuild the "
          "index");
      return IndexError_Unsupported;
    }

    centroids->assign(pivot_bytes, '\0');
    char *dest = &(*centroids)[0];
    for (size_t m = 0; m < chunk_num; ++m) {
      for (size_t c = 0; c < kClusterNum; ++c) {
        std::memcpy(dest + ((m * kClusterNum + c) * sub_dim) * unit,
                    pivots + (c * dim + m * sub_dim) * unit, sub_dim * unit);
      }
    }

    return 0;
  }

  //! Construct the quantizer persisted in `meta_buffer`, picking the
  //! implementation from the meta buffer header instead of a hardcoded type.
  //! Contract: the quantizer is initialized with the meta derived from
  //! `index_meta` before deserialize(), so the metric policy comes from the
  //! meta instead of the default-constructed one.  A headerless buffer holds
  //! the legacy PQ codebook and is adopted after repacking.
  static turbo::Quantizer::Pointer create_quantizer_from_meta_buffer(
      std::string &meta_buffer, const IndexMeta &index_meta,
      uint32_t chunk_num) {
    const bool legacy = is_legacy_pq_meta_buffer(meta_buffer);
    const char *name = legacy ? "PqInt8Quantizer"
                              : quantizer_name_from_meta_buffer(meta_buffer);
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
    if (legacy) {
      std::string centroids;
      if (legacy_pq_codebook_to_centroids(meta_buffer, init_meta, chunk_num,
                                          &centroids) != 0) {
        return turbo::Quantizer::Pointer();
      }
      if (quantizer->import_codebook(centroids.data(), centroids.size()) != 0) {
        LOG_ERROR("Quantizer %s legacy codebook import failed", name);
        return turbo::Quantizer::Pointer();
      }
      return quantizer;
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
