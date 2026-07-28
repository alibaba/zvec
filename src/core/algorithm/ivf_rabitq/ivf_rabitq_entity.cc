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

#include "ivf_rabitq_entity.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <rabitqlib/fastscan/fastscan.hpp>
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include <rabitqlib/utils/space.hpp>
#include <zvec/ailego/logger/logger.h>
#include "zvec/core/framework/index_error.h"
#include "ivf_rabitq_params.h"

namespace zvec {
namespace core {

namespace {

void ExtractCompactCode(const uint8_t *batch_code, size_t padded_dim,
                        uint32_t lane, uint8_t *compact_code) {
  uint32_t perm_pos = 0;
  uint32_t perm_lane = lane % 16U;
  for (uint32_t i = 0; i < rabitqlib::fastscan::kPerm0.size(); ++i) {
    if (static_cast<uint32_t>(rabitqlib::fastscan::kPerm0[i]) == perm_lane) {
      perm_pos = i;
      break;
    }
  }

  bool high_half = lane >= 16U;
  size_t code_bytes = padded_dim / 8;
  for (size_t col = 0; col < code_bytes; ++col) {
    const uint8_t *block = batch_code + (col * rabitqlib::fastscan::kBatchSize);
    uint8_t high_bits =
        high_half ? ((block[perm_pos] >> 4) & 0x0F) : (block[perm_pos] & 0x0F);
    uint8_t low_bits = high_half ? ((block[perm_pos + 16] >> 4) & 0x0F)
                                 : (block[perm_pos + 16] & 0x0F);
    compact_code[col] = static_cast<uint8_t>((high_bits << 4) | low_bits);
  }
}

}  // namespace

int IvfRabitqEntity::load(IndexStorage::Pointer storage) {
  if (!storage) {
    LOG_ERROR("Invalid storage for IvfRabitqEntity::load");
    return IndexError_InvalidArgument;
  }

  // Read header
  auto header_seg = storage->get(IVF_RABITQ_HEADER_SEG_ID);
  if (!header_seg) {
    LOG_ERROR("Failed to get segment %s", IVF_RABITQ_HEADER_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock hdr_block;
  size_t read_size = header_seg->read(0, hdr_block, sizeof(IvfRabitqHeader));
  if (read_size != sizeof(IvfRabitqHeader)) {
    LOG_ERROR("Failed to read IvfRabitqHeader");
    return IndexError_InvalidFormat;
  }
  memcpy(&header_, hdr_block.data(), sizeof(IvfRabitqHeader));

  // Read cluster metas
  auto cluster_meta_seg = storage->get(IVF_RABITQ_CLUSTER_META_SEG_ID);
  if (!cluster_meta_seg) {
    LOG_ERROR("Failed to get segment %s",
              IVF_RABITQ_CLUSTER_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  size_t meta_size = header_.cluster_count * sizeof(IvfRabitqClusterMeta);
  IndexStorage::MemoryBlock meta_block;
  read_size = cluster_meta_seg->read(0, meta_block, meta_size);
  if (read_size != meta_size) {
    LOG_ERROR("Failed to read cluster metas");
    return IndexError_InvalidFormat;
  }
  cluster_metas_.resize(header_.cluster_count);
  memcpy(cluster_metas_.data(), meta_block.data(), meta_size);

  // Read batch data
  auto batch_seg = storage->get(IVF_RABITQ_BATCH_DATA_SEG_ID);
  if (!batch_seg) {
    LOG_ERROR("Failed to get segment %s", IVF_RABITQ_BATCH_DATA_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  read_size = batch_seg->read(0, batch_data_block_, header_.batch_data_size);
  if (read_size != header_.batch_data_size) {
    LOG_ERROR("Failed to read batch data: got %zu, expected %zu", read_size,
              (size_t)header_.batch_data_size);
    return IndexError_InvalidFormat;
  }
  batch_data_ = reinterpret_cast<const char *>(batch_data_block_.data());

  // Read ex data
  ex_data_ = nullptr;
  if (header_.ex_data_size > 0) {
    auto ex_seg = storage->get(IVF_RABITQ_EX_DATA_SEG_ID);
    if (!ex_seg) {
      LOG_ERROR("Failed to get segment %s", IVF_RABITQ_EX_DATA_SEG_ID.c_str());
      return IndexError_InvalidFormat;
    }
    read_size = ex_seg->read(0, ex_data_block_, header_.ex_data_size);
    if (read_size != header_.ex_data_size) {
      LOG_ERROR("Failed to read ex data");
      return IndexError_InvalidFormat;
    }
    ex_data_ = reinterpret_cast<const char *>(ex_data_block_.data());
  }

  // Read keys
  auto keys_seg = storage->get(IVF_RABITQ_KEYS_SEG_ID);
  if (!keys_seg) {
    LOG_ERROR("Failed to get segment %s", IVF_RABITQ_KEYS_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  size_t keys_size = header_.total_vector_count * sizeof(uint64_t);
  read_size = keys_seg->read(0, keys_block_, keys_size);
  if (read_size != keys_size) {
    LOG_ERROR("Failed to read keys");
    return IndexError_InvalidFormat;
  }
  keys_ = reinterpret_cast<const uint64_t *>(keys_block_.data());

  int ret = load_key_order_mapping(storage);
  if (ret != 0) {
    return ret;
  }
  ret = init_quantized_vector_layout();
  if (ret != 0) {
    return ret;
  }

  LOG_INFO(
      "IvfRabitqEntity loaded: total_vectors=%zu, clusters=%zu, "
      "padded_dim=%zu, ex_bits=%zu, batch_data_size=%zu, ex_data_size=%zu",
      (size_t)header_.total_vector_count, (size_t)header_.cluster_count,
      (size_t)header_.padded_dim, (size_t)header_.ex_bits,
      (size_t)header_.batch_data_size, (size_t)header_.ex_data_size);

  // Resolve function pointer once at load time
  ip_func_ = rabitqlib::select_excode_ipfunc(header_.ex_bits);

  return 0;
}

int IvfRabitqEntity::search_cluster(uint32_t cluster_id,
                                    const IvfRabitqQueryState &query_state,
                                    size_t padded_dim, size_t ex_bits,
                                    IndexDocumentHeap *heap) const {
  return search_cluster_impl<false>(cluster_id, query_state, padded_dim,
                                    ex_bits, nullptr, heap);
}

int IvfRabitqEntity::search_cluster(uint32_t cluster_id,
                                    const IvfRabitqQueryState &query_state,
                                    size_t padded_dim, size_t ex_bits,
                                    const IndexFilter &filter,
                                    IndexDocumentHeap *heap) const {
  return search_cluster_impl<true>(cluster_id, query_state, padded_dim, ex_bits,
                                   &filter, heap);
}

uint64_t IvfRabitqEntity::get_key(size_t id) const {
  if (id >= header_.total_vector_count || !keys_) {
    return std::numeric_limits<uint64_t>::max();
  }
  return keys_[id];
}

const void *IvfRabitqEntity::get_vector(size_t id) const {
  int ret = materialize_quantized_vector(id, &quantized_vector_buffer_);
  if (ret != 0) {
    return nullptr;
  }
  return quantized_vector_buffer_.data();
}

int IvfRabitqEntity::get_vector(size_t id,
                                IndexStorage::MemoryBlock &block) const {
  return materialize_quantized_vector(id, block);
}

const void *IvfRabitqEntity::get_vector_by_key(uint64_t key) const {
  uint32_t id = key_to_id(key);
  if (id == std::numeric_limits<uint32_t>::max()) {
    return nullptr;
  }
  return get_vector(id);
}

int IvfRabitqEntity::get_vector_by_key(uint64_t key,
                                       IndexStorage::MemoryBlock &block) const {
  uint32_t id = key_to_id(key);
  if (id == std::numeric_limits<uint32_t>::max()) {
    return IndexError_NoExist;
  }
  return get_vector(id, block);
}

uint32_t IvfRabitqEntity::key_to_id(uint64_t key) const {
  if (!keys_ || !mapping_) {
    return std::numeric_limits<uint32_t>::max();
  }

  uint32_t start = 0U;
  uint32_t end = header_.total_vector_count;
  const void *data = nullptr;
  while (start < end) {
    uint32_t idx = start + (end - start) / 2U;
    if (mapping_->read(idx * sizeof(uint32_t), &data, sizeof(uint32_t)) !=
        sizeof(uint32_t)) {
      LOG_ERROR("Failed to read mapping segment, idx=%zu", (size_t)idx);
      return std::numeric_limits<uint32_t>::max();
    }
    uint32_t local_id = *reinterpret_cast<const uint32_t *>(data);
    if (local_id >= header_.total_vector_count) {
      LOG_ERROR("Invalid mapping local_id=%zu, vector_count=%zu",
                (size_t)local_id, (size_t)header_.total_vector_count);
      return std::numeric_limits<uint32_t>::max();
    }
    uint64_t local_key = keys_[local_id];
    if (local_key < key) {
      start = idx + 1U;
    } else if (local_key > key) {
      end = idx;
    } else {
      return local_id;
    }
  }
  return std::numeric_limits<uint32_t>::max();
}

int IvfRabitqEntity::load_key_order_mapping(IndexStorage::Pointer storage) {
  mapping_.reset();

  if (header_.total_vector_count == 0) {
    return 0;
  }
  if (!keys_) {
    LOG_ERROR("Missing keys for mapping");
    return IndexError_InvalidFormat;
  }

  size_t mapping_size =
      static_cast<size_t>(header_.total_vector_count) * sizeof(uint32_t);
  mapping_ = storage->get(IVF_RABITQ_MAPPING_SEG_ID);
  if (!mapping_) {
    LOG_ERROR("Failed to get segment %s", IVF_RABITQ_MAPPING_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  if (mapping_->data_size() != mapping_size) {
    LOG_ERROR("Invalid mapping segment size=%zu, expected=%zu",
              mapping_->data_size(), mapping_size);
    return IndexError_InvalidFormat;
  }
  return 0;
}

int IvfRabitqEntity::init_quantized_vector_layout() {
  quantized_vector_buffer_.clear();
  quantized_vector_element_size_ = 0;
  compact_code_size_ = header_.padded_dim / 8;
  bin_data_size_ = rabitqlib::BinDataMap<float>::data_bytes(header_.padded_dim);
  batch_data_size_ =
      rabitqlib::BatchDataMap<float>::data_bytes(header_.padded_dim);
  ex_data_size_ = rabitqlib::ExDataMap<float>::data_bytes(header_.padded_dim,
                                                          header_.ex_bits);
  quantized_vector_element_size_ = bin_data_size_ + ex_data_size_;

  if (header_.total_vector_count == 0) {
    return 0;
  }
  if (!batch_data_) {
    LOG_ERROR("Missing batch data for quantized vectors");
    return IndexError_InvalidFormat;
  }
  if (header_.ex_bits > 0 && !ex_data_) {
    LOG_ERROR("Missing extra-bit data for quantized vectors");
    return IndexError_InvalidFormat;
  }

  return 0;
}

const IvfRabitqClusterMeta *IvfRabitqEntity::find_cluster_meta(
    size_t id) const {
  auto it =
      std::upper_bound(cluster_metas_.begin(), cluster_metas_.end(), id,
                       [](size_t value, const IvfRabitqClusterMeta &meta) {
                         return value < meta.key_offset;
                       });
  if (it == cluster_metas_.begin()) {
    return nullptr;
  }
  --it;
  if (id >= static_cast<size_t>(it->key_offset) + it->vector_count) {
    return nullptr;
  }
  return &(*it);
}

int IvfRabitqEntity::materialize_quantized_vector(size_t id,
                                                  std::string *buffer) const {
  if (!buffer) {
    return IndexError_InvalidArgument;
  }
  if (id >= header_.total_vector_count || quantized_vector_element_size_ == 0) {
    return IndexError_NoExist;
  }

  buffer->resize(quantized_vector_element_size_);
  char *dst = &(*buffer)[0];
  return materialize_quantized_vector(id, dst);
}

int IvfRabitqEntity::materialize_quantized_vector(
    size_t id, IndexStorage::MemoryBlock &block) const {
  if (id >= header_.total_vector_count || quantized_vector_element_size_ == 0) {
    return IndexError_NoExist;
  }

  void *data = ailego_malloc(quantized_vector_element_size_);
  if (!data) {
    return IndexError_NoMemory;
  }

  int ret = materialize_quantized_vector(id, static_cast<char *>(data));
  if (ret != 0) {
    ailego_free(data);
    return ret;
  }
  block = IndexStorage::MemoryBlock::MakeOwned(data,
                                               quantized_vector_element_size_);
  return 0;
}

int IvfRabitqEntity::materialize_quantized_vector(size_t id, char *dst) const {
  if (!dst) {
    return IndexError_InvalidArgument;
  }
  if (id >= header_.total_vector_count || quantized_vector_element_size_ == 0) {
    return IndexError_NoExist;
  }
  if (!batch_data_) {
    LOG_ERROR("Missing batch data for quantized vectors");
    return IndexError_InvalidFormat;
  }
  if (header_.ex_bits > 0 && !ex_data_) {
    LOG_ERROR("Missing extra-bit data for quantized vectors");
    return IndexError_InvalidFormat;
  }

  const auto *meta = find_cluster_meta(id);
  if (!meta) {
    LOG_ERROR("Failed to locate cluster for vector id=%zu", id);
    return IndexError_InvalidFormat;
  }

  size_t local_id = id - meta->key_offset;
  uint32_t batch_id = static_cast<uint32_t>(local_id) /
                      static_cast<uint32_t>(rabitqlib::fastscan::kBatchSize);
  uint32_t lane = static_cast<uint32_t>(local_id) %
                  static_cast<uint32_t>(rabitqlib::fastscan::kBatchSize);

  const char *batch_ptr = batch_data_ + meta->batch_data_offset +
                          (static_cast<size_t>(batch_id) * batch_data_size_);
  rabitqlib::ConstBatchDataMap<float> batch_map(batch_ptr, header_.padded_dim);

  ExtractCompactCode(batch_map.bin_code(), header_.padded_dim, lane,
                     reinterpret_cast<uint8_t *>(dst));
  std::memcpy(dst + compact_code_size_, batch_map.f_add() + lane,
              sizeof(float));
  std::memcpy(dst + compact_code_size_ + sizeof(float),
              batch_map.f_rescale() + lane, sizeof(float));
  std::memcpy(dst + compact_code_size_ + (sizeof(float) * 2),
              batch_map.f_error() + lane, sizeof(float));

  if (ex_data_size_ > 0) {
    const char *cluster_ex = ex_data_ + meta->ex_data_offset;
    std::memcpy(dst + bin_data_size_, cluster_ex + (local_id * ex_data_size_),
                ex_data_size_);
  }

  return 0;
}

// Cluster scan with lower-bound pruning — mirrors rabitqlib IVF::scan_one_batch
template <bool HasFilter>
int IvfRabitqEntity::search_cluster_impl(uint32_t cluster_id,
                                         const IvfRabitqQueryState &query_state,
                                         size_t padded_dim, size_t ex_bits,
                                         const IndexFilter *filter,
                                         IndexDocumentHeap *heap) const {
  if (cluster_id >= cluster_metas_.size()) {
    LOG_ERROR("Invalid cluster_id=%zu", (size_t)cluster_id);
    return IndexError_OutOfRange;
  }

  const auto &meta = cluster_metas_[cluster_id];
  if (meta.vector_count == 0) {
    return 0;
  }

  const auto *bq = static_cast<const rabitqlib::SplitBatchQuery<float> *>(
      query_state.batch_query.get());
  if (!bq) {
    LOG_ERROR("Null batch_query in query_state");
    return IndexError_InvalidArgument;
  }

  const size_t batch_stride =
      rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  const size_t ex_stride =
      rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);

  const char *cur_batch = batch_data_ + meta.batch_data_offset;
  const char *cur_ex = nullptr;
  if (ex_bits > 0) {
    if (!ex_data_) {
      LOG_ERROR("Missing extra-bit data for ex_bits=%zu", ex_bits);
      return IndexError_InvalidFormat;
    }
    cur_ex = ex_data_ + meta.ex_data_offset;
  }
  const uint64_t *cur_keys = keys_ + meta.key_offset;

  uint32_t remaining = meta.vector_count;
  uint32_t offset = 0;

  while (remaining > 0) {
    const uint32_t batch_size =
        std::min(remaining, (uint32_t)rabitqlib::fastscan::kBatchSize);

    // Step 1: 1-bit estimation for all 32 vectors at once (cheap, SIMD)
    std::array<float, rabitqlib::fastscan::kBatchSize> est_dist;
    std::array<float, rabitqlib::fastscan::kBatchSize> low_dist;
    std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr;

    rabitqlib::split_batch_estdist(cur_batch, *bq, padded_dim, est_dist.data(),
                                   low_dist.data(), ip_x0_qr.data(),
                                   true /* use_hacc */);

    if (ex_bits == 0) {
      // 1-bit only: insert all directly
      for (uint32_t i = 0; i < batch_size; ++i) {
        uint64_t key = cur_keys[offset + i];
        if constexpr (HasFilter) {
          if ((*filter)(key)) {
            continue;
          }
        }
        heap->emplace(key, est_dist[i]);
      }
    } else {
      // Step 2: lower-bound pruning + extra-bit boosting (same as rabitqlib)
      // distk: current worst distance in heap, updated after each insertion
      float distk = heap->full() ? heap->begin()->score()
                                 : std::numeric_limits<float>::max();

      for (uint32_t i = 0; i < batch_size; ++i) {
        if (low_dist[i] < distk) {
          uint64_t key = cur_keys[offset + i];
          if constexpr (HasFilter) {
            if ((*filter)(key)) {
              continue;
            }
          }
          // Extra-bit boosting (expensive, but only when needed)
          const float ex_dist = rabitqlib::split_distance_boosting(
              cur_ex + i * ex_stride, ip_func_, *bq, padded_dim, ex_bits,
              ip_x0_qr[i]);
          heap->emplace(key, ex_dist);
          if (heap->full()) {
            distk = heap->begin()->score();
          }
        }
      }
    }

    cur_batch += batch_stride;
    if (ex_bits > 0) {
      cur_ex += batch_size * ex_stride;
    }
    offset += batch_size;
    remaining -= batch_size;
  }

  return 0;
}

}  // namespace core
}  // namespace zvec
