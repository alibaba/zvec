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
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "zvec/core/framework/index_error.h"
#include "ivf_rabitq_params.h"

namespace zvec {
namespace core {

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
  if (header_seg->data_size() != sizeof(IvfRabitqHeader)) {
    LOG_ERROR("Invalid IVF RaBitQ header segment size=%zu, expected=%zu",
              header_seg->data_size(), sizeof(IvfRabitqHeader));
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock hdr_block;
  size_t read_size = header_seg->read(0, hdr_block, sizeof(IvfRabitqHeader));
  if (read_size != sizeof(IvfRabitqHeader)) {
    LOG_ERROR("Failed to read IvfRabitqHeader");
    return IndexError_InvalidFormat;
  }
  memcpy(&header_, hdr_block.data(), sizeof(IvfRabitqHeader));
  int ret = validate_header();
  if (ret != 0) {
    return ret;
  }

  // Read cluster metas
  auto cluster_meta_seg = storage->get(IVF_RABITQ_CLUSTER_META_SEG_ID);
  if (!cluster_meta_seg) {
    LOG_ERROR("Failed to get segment %s",
              IVF_RABITQ_CLUSTER_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  size_t meta_size =
      static_cast<size_t>(header_.cluster_count) * sizeof(IvfRabitqClusterMeta);
  if (cluster_meta_seg->data_size() != meta_size) {
    LOG_ERROR("Invalid cluster meta segment size=%zu, expected=%zu",
              cluster_meta_seg->data_size(), meta_size);
    return IndexError_InvalidFormat;
  }
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
  size_t batch_data_size = static_cast<size_t>(header_.batch_data_size);
  if (static_cast<uint64_t>(batch_data_size) != header_.batch_data_size ||
      batch_seg->data_size() != batch_data_size) {
    LOG_ERROR("Invalid batch data segment size=%zu, expected=%zu",
              batch_seg->data_size(), batch_data_size);
    return IndexError_InvalidFormat;
  }
  read_size = batch_seg->read(0, batch_data_block_, batch_data_size);
  if (read_size != batch_data_size) {
    LOG_ERROR("Failed to read batch data: got %zu, expected %zu", read_size,
              batch_data_size);
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
    size_t ex_data_size = static_cast<size_t>(header_.ex_data_size);
    if (static_cast<uint64_t>(ex_data_size) != header_.ex_data_size ||
        ex_seg->data_size() != ex_data_size) {
      LOG_ERROR("Invalid extra-bit data segment size=%zu, expected=%zu",
                ex_seg->data_size(), ex_data_size);
      return IndexError_InvalidFormat;
    }
    read_size = ex_seg->read(0, ex_data_block_, ex_data_size);
    if (read_size != ex_data_size) {
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
  size_t keys_size =
      static_cast<size_t>(header_.total_vector_count) * sizeof(uint64_t);
  if (keys_seg->data_size() != keys_size) {
    LOG_ERROR("Invalid keys segment size=%zu, expected=%zu",
              keys_seg->data_size(), keys_size);
    return IndexError_InvalidFormat;
  }
  read_size = keys_seg->read(0, keys_block_, keys_size);
  if (read_size != keys_size) {
    LOG_ERROR("Failed to read keys");
    return IndexError_InvalidFormat;
  }
  keys_ = reinterpret_cast<const uint64_t *>(keys_block_.data());

  ret = load_key_order_mapping(storage);
  if (ret != 0) {
    return ret;
  }
  ret = init_data_layout();
  if (ret != 0) {
    return ret;
  }
  ret = validate_cluster_metas();
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
  if (header_.ex_bits > 0 && !ip_func_) {
    LOG_ERROR("Unsupported IVF RaBitQ ex_bits=%zu", (size_t)header_.ex_bits);
    return IndexError_InvalidFormat;
  }

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

int IvfRabitqEntity::search_cluster_group_by(
    uint32_t cluster_id, const IvfRabitqQueryState &query_state,
    size_t padded_dim, size_t ex_bits, const IndexGroupBy &group_by,
    uint32_t group_topk, float threshold,
    IvfRabitqContext::GroupTopkHeaps *heaps) const {
  return search_cluster_group_by_impl<false>(
      cluster_id, query_state, padded_dim, ex_bits, nullptr, group_by,
      group_topk, threshold, heaps);
}

int IvfRabitqEntity::search_cluster_group_by(
    uint32_t cluster_id, const IvfRabitqQueryState &query_state,
    size_t padded_dim, size_t ex_bits, const IndexFilter &filter,
    const IndexGroupBy &group_by, uint32_t group_topk, float threshold,
    IvfRabitqContext::GroupTopkHeaps *heaps) const {
  return search_cluster_group_by_impl<true>(cluster_id, query_state, padded_dim,
                                            ex_bits, &filter, group_by,
                                            group_topk, threshold, heaps);
}

uint64_t IvfRabitqEntity::get_key(size_t id) const {
  if (id >= header_.total_vector_count || !keys_) {
    return std::numeric_limits<uint64_t>::max();
  }
  return keys_[id];
}

const void *IvfRabitqEntity::get_vector_by_key(uint64_t /*key*/) const {
  return nullptr;
}

int IvfRabitqEntity::get_vector_by_key(
    uint64_t /*key*/, IndexStorage::MemoryBlock & /*block*/) const {
  return IndexError_Unsupported;
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

uint32_t IvfRabitqEntity::get_cluster_id(uint32_t id) const {
  const auto *meta = find_cluster_meta(id);
  if (!meta) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(meta - cluster_metas_.data());
}

int IvfRabitqEntity::validate_header() const {
  if (header_.magic != kIvfRabitqIndexMagic) {
    LOG_ERROR("Invalid IVF RaBitQ index magic: got=%zu, expected=%zu",
              (size_t)header_.magic, (size_t)kIvfRabitqIndexMagic);
    return IndexError_InvalidFormat;
  }
  if (header_.version != kIvfRabitqIndexVersion) {
    LOG_ERROR("Unsupported IVF RaBitQ index version: got=%zu, expected=%zu",
              (size_t)header_.version, (size_t)kIvfRabitqIndexVersion);
    return IndexError_InvalidFormat;
  }
  if (header_.dimension < kMinRabitqDimSize ||
      header_.dimension > kMaxRabitqDimSize) {
    LOG_ERROR("Invalid IVF RaBitQ dimension=%zu", (size_t)header_.dimension);
    return IndexError_InvalidFormat;
  }
  uint32_t expected_padded_dim = ((header_.dimension + 63U) / 64U) * 64U;
  if (header_.padded_dim != expected_padded_dim) {
    LOG_ERROR("Invalid IVF RaBitQ padded_dim=%zu, expected=%zu",
              (size_t)header_.padded_dim, (size_t)expected_padded_dim);
    return IndexError_InvalidFormat;
  }
  if (header_.total_vector_count == 0 || header_.cluster_count == 0 ||
      header_.cluster_count > header_.total_vector_count) {
    LOG_ERROR("Invalid IVF RaBitQ vector_count=%zu, cluster_count=%zu",
              (size_t)header_.total_vector_count,
              (size_t)header_.cluster_count);
    return IndexError_InvalidFormat;
  }
  if (header_.ex_bits > 8) {
    LOG_ERROR("Invalid IVF RaBitQ ex_bits=%zu", (size_t)header_.ex_bits);
    return IndexError_InvalidFormat;
  }
  return 0;
}

int IvfRabitqEntity::validate_cluster_metas() const {
  size_t expected_key_offset = 0;
  for (size_t cluster_id = 0; cluster_id < cluster_metas_.size();
       ++cluster_id) {
    const auto &meta = cluster_metas_[cluster_id];
    size_t expected_batch_count = (static_cast<size_t>(meta.vector_count) +
                                   rabitqlib::fastscan::kBatchSize - 1) /
                                  rabitqlib::fastscan::kBatchSize;
    size_t cluster_batch_size =
        static_cast<size_t>(meta.batch_count) * batch_data_size_;
    size_t cluster_ex_size =
        static_cast<size_t>(meta.vector_count) * ex_data_size_;
    if (meta.batch_count != expected_batch_count ||
        meta.key_offset != expected_key_offset ||
        meta.key_offset > header_.total_vector_count ||
        meta.vector_count > header_.total_vector_count - meta.key_offset ||
        meta.batch_data_offset > header_.batch_data_size ||
        cluster_batch_size > header_.batch_data_size - meta.batch_data_offset ||
        meta.ex_data_offset > header_.ex_data_size ||
        cluster_ex_size > header_.ex_data_size - meta.ex_data_offset) {
      LOG_ERROR("Invalid IVF RaBitQ cluster range, cluster_id=%zu", cluster_id);
      return IndexError_InvalidFormat;
    }
    expected_key_offset += meta.vector_count;
  }
  if (expected_key_offset != header_.total_vector_count) {
    LOG_ERROR("Invalid IVF RaBitQ cluster key coverage=%zu, expected=%zu",
              expected_key_offset, (size_t)header_.total_vector_count);
    return IndexError_InvalidFormat;
  }
  return 0;
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

int IvfRabitqEntity::init_data_layout() {
  batch_data_size_ =
      rabitqlib::BatchDataMap<float>::data_bytes(header_.padded_dim);
  ex_data_size_ = rabitqlib::ExDataMap<float>::data_bytes(header_.padded_dim,
                                                          header_.ex_bits);

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

int IvfRabitqEntity::compute_distances(uint32_t cluster_id,
                                       const std::vector<uint32_t> &ids,
                                       const IvfRabitqQueryState &query_state,
                                       size_t padded_dim, size_t ex_bits,
                                       IndexDocumentList *documents) const {
  if (!documents) {
    return IndexError_InvalidArgument;
  }
  if (cluster_id >= cluster_metas_.size()) {
    return IndexError_OutOfRange;
  }
  const auto *batch_query =
      static_cast<const rabitqlib::SplitBatchQuery<float> *>(
          query_state.batch_query.get());
  if (!batch_query) {
    return IndexError_InvalidArgument;
  }

  const auto &meta = cluster_metas_[cluster_id];
  std::vector<uint32_t> sorted_ids(ids);
  std::sort(sorted_ids.begin(), sorted_ids.end());

  const size_t batch_stride =
      rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  const size_t ex_stride =
      rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
  std::array<float, rabitqlib::fastscan::kBatchSize> est_dist;
  std::array<float, rabitqlib::fastscan::kBatchSize> low_dist;
  std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr;
  uint32_t loaded_batch = std::numeric_limits<uint32_t>::max();

  documents->clear();
  documents->reserve(sorted_ids.size());
  for (uint32_t id : sorted_ids) {
    if (static_cast<size_t>(id) < meta.key_offset ||
        static_cast<size_t>(id) >= meta.key_offset + meta.vector_count) {
      return IndexError_InvalidArgument;
    }
    const uint32_t local_id = id - static_cast<uint32_t>(meta.key_offset);
    const uint32_t batch_id =
        local_id / static_cast<uint32_t>(rabitqlib::fastscan::kBatchSize);
    const uint32_t lane =
        local_id % static_cast<uint32_t>(rabitqlib::fastscan::kBatchSize);
    if (batch_id != loaded_batch) {
      const char *batch_data = batch_data_ + meta.batch_data_offset +
                               (static_cast<size_t>(batch_id) * batch_stride);
      rabitqlib::split_batch_estdist(batch_data, *batch_query, padded_dim,
                                     est_dist.data(), low_dist.data(),
                                     ip_x0_qr.data(), true /* use_hacc */);
      loaded_batch = batch_id;
    }

    float distance = est_dist[lane];
    if (ex_bits > 0) {
      const char *ex_data = ex_data_ + meta.ex_data_offset +
                            (static_cast<size_t>(local_id) * ex_stride);
      distance = rabitqlib::split_distance_boosting(
          ex_data, ip_func_, *batch_query, padded_dim, ex_bits, ip_x0_qr[lane]);
    }
    documents->emplace_back(keys_[id], distance);
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
      // 1-bit only: scalar fast-reject before the heavy heap emplace so that
      // rejected candidates cost a single float compare instead of building a
      // full IndexDocument (matches rabitqlib SearchBuffer semantics).
      float distk = heap->full() ? heap->begin()->score()
                                 : std::numeric_limits<float>::max();
      for (uint32_t i = 0; i < batch_size; ++i) {
        if (est_dist[i] >= distk) {
          continue;
        }
        uint64_t key = cur_keys[offset + i];
        if constexpr (HasFilter) {
          if ((*filter)(key)) {
            continue;
          }
        }
        heap->emplace(key, est_dist[i]);
        if (heap->full()) {
          distk = heap->begin()->score();
        }
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

template <bool HasFilter>
int IvfRabitqEntity::search_cluster_group_by_impl(
    uint32_t cluster_id, const IvfRabitqQueryState &query_state,
    size_t padded_dim, size_t ex_bits, const IndexFilter *filter,
    const IndexGroupBy &group_by, uint32_t group_topk, float threshold,
    IvfRabitqContext::GroupTopkHeaps *heaps) const {
  if (cluster_id >= cluster_metas_.size()) {
    return IndexError_OutOfRange;
  }
  if (!heaps || !group_by.is_valid() || group_topk == 0) {
    return IndexError_InvalidArgument;
  }

  const auto &meta = cluster_metas_[cluster_id];
  if (meta.vector_count == 0) {
    return 0;
  }

  const auto *batch_query =
      static_cast<const rabitqlib::SplitBatchQuery<float> *>(
          query_state.batch_query.get());
  if (!batch_query) {
    return IndexError_InvalidArgument;
  }

  const size_t batch_stride =
      rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  const size_t ex_stride =
      rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
  const char *batch_data = batch_data_ + meta.batch_data_offset;
  const char *ex_data = ex_bits > 0 ? ex_data_ + meta.ex_data_offset : nullptr;
  const uint64_t *keys = keys_ + meta.key_offset;

  uint32_t remaining = meta.vector_count;
  uint32_t offset = 0;
  while (remaining > 0) {
    const uint32_t batch_size =
        std::min(remaining, (uint32_t)rabitqlib::fastscan::kBatchSize);
    std::array<float, rabitqlib::fastscan::kBatchSize> est_dist;
    std::array<float, rabitqlib::fastscan::kBatchSize> low_dist;
    std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr;
    rabitqlib::split_batch_estdist(batch_data, *batch_query, padded_dim,
                                   est_dist.data(), low_dist.data(),
                                   ip_x0_qr.data(), true /* use_hacc */);

    for (uint32_t i = 0; i < batch_size; ++i) {
      const uint64_t key = keys[offset + i];
      if constexpr (HasFilter) {
        if ((*filter)(key)) {
          continue;
        }
      }

      const std::string group_id = group_by(key);
      auto &heap = (*heaps)[group_id];
      if (heap.empty()) {
        heap.limit(group_topk);
        heap.set_threshold(threshold);
      }

      if (ex_bits == 0) {
        if (heap.full() && est_dist[i] >= heap.begin()->score()) {
          continue;
        }
        heap.emplace(key, est_dist[i]);
        continue;
      }

      const float group_threshold =
          heap.full() ? heap.begin()->score() : threshold;
      if (low_dist[i] >= group_threshold) {
        continue;
      }
      const float distance = rabitqlib::split_distance_boosting(
          ex_data + (static_cast<size_t>(i) * ex_stride), ip_func_,
          *batch_query, padded_dim, ex_bits, ip_x0_qr[i]);
      heap.emplace(key, distance);
    }

    batch_data += batch_stride;
    if (ex_bits > 0) {
      ex_data += static_cast<size_t>(batch_size) * ex_stride;
    }
    offset += batch_size;
    remaining -= batch_size;
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
