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

#include "flat_sparse_searcher_entity.h"
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_logger.h>
#include "flat_sparse_utility.h"

namespace zvec {
namespace core {

FlatSparseSearcherEntity::FlatSparseSearcherEntity() {}

int FlatSparseSearcherEntity::load(const IndexStorage::Pointer &container,
                                   const IndexMeta &index_meta) {
  if (container_) {
    LOG_ERROR("An storage instance is already opened");
    return IndexError_Duplicate;
  }

  int ret = this->load_container(container);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to load storage index");
    return ret;
  }

  if (init_measure(index_meta) != 0) {
    LOG_ERROR("Failed to init measure");
    return IndexError_InvalidFormat;
  }

  container_ = container;
  return 0;
}

int FlatSparseSearcherEntity::init_measure(const IndexMeta &meta) {
  measure_ = IndexFactory::CreateMetric(meta.metric_name());
  if (!measure_) {
    LOG_ERROR("Failed to create measure %s", meta.metric_name().c_str());
    return IndexError_NoExist;
  }
  int ret = measure_->init(meta, meta.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failled to init measure, ret=%d", ret);
    return ret;
  }

  if (!measure_->sparse_distance()) {
    LOG_ERROR("Invalid measure distance");
    return IndexError_InvalidArgument;
  }

  search_sparse_distance_ = measure_->sparse_distance();

  if (measure_->query_metric() && measure_->query_metric()->distance()) {
    search_sparse_distance_ = measure_->query_metric()->sparse_distance();
  }
  sparse_unit_size_ = meta.unit_size();

  return 0;
}

int FlatSparseSearcherEntity::load_container(
    const IndexStorage::Pointer &container) {
  // meta
  auto segment = container->get(PARAM_FLAT_SPARSE_META_SEG_ID);
  if (!segment || segment->data_size() < sizeof(meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  const void *data;
  if (ailego_unlikely(segment->read(0, &data, sizeof(meta_)) !=
                      sizeof(meta_))) {
    LOG_ERROR("Failed to read meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  meta_ = *(reinterpret_cast<const decltype(meta_) *>(data));

  // keys segment
  keys_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID);
  if (!keys_chunk_) {
    LOG_ERROR("Missing segment %s", PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  // mapping segment
  mapping_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID);
  if (!mapping_chunk_) {
    LOG_ERROR("Missing segment %s",
              PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  // offset segment
  sparse_offset_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID);
  if (!sparse_offset_chunk_) {
    LOG_ERROR("Missing segment %s",
              PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  // data segment
  sparse_data_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID);
  if (!sparse_data_chunk_) {
    LOG_ERROR("Missing segment %s", PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  return 0;
}

int FlatSparseSearcherEntity::unload() {
  container_.reset();
  sparse_data_chunk_.reset();
  sparse_offset_chunk_.reset();
  keys_chunk_.reset();
  mapping_chunk_.reset();

  return 0;
}

FlatSparseSearcherEntity::Pointer FlatSparseSearcherEntity::clone() const {
  auto entity = new (std::nothrow)
      FlatSparseSearcherEntity(meta_, sparse_data_chunk_, sparse_offset_chunk_,
                               keys_chunk_, mapping_chunk_);
  return FlatSparseSearcherEntity::Pointer(entity);
}

int FlatSparseSearcherEntity::get_sparse_vector_ptr_by_id(
    node_id_t id, const void **sparse_vector_ptr,
    uint32_t *sparse_vector_len_ptr) const {
  uint32_t offset_chunk_offset = id * offset_size_per_node();

  const void *offset_info = nullptr;
  if (ailego_unlikely(sparse_offset_chunk_->read(
                          offset_chunk_offset, &offset_info,
                          offset_size_per_node()) != offset_size_per_node())) {
    LOG_ERROR("Read offset info failed, offset=%u", offset_chunk_offset);
    return IndexError_ReadData;
  };

  // sparse offset
  uint64_t sparse_offset = *(uint64_t *)offset_info;
  uint32_t sparse_vector_len =
      *(uint32_t *)((uint8_t *)offset_info + sizeof(uint64_t));

  if (sparse_vector_len > 0) {
    const void *sparse_data =
        get_sparse_vector_data(sparse_offset, sparse_vector_len);
    if (ailego_unlikely(sparse_data == nullptr)) {
      LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)sparse_offset,
                sparse_vector_len);

      return IndexError_ReadData;
    }
    *sparse_vector_ptr = sparse_data;
    *sparse_vector_len_ptr = sparse_vector_len;
  }

  return 0;
}

const void *FlatSparseSearcherEntity::get_sparse_vector_data(
    uint64_t offset, uint32_t length) const {
  const void *data;
  auto size = sparse_data_chunk_->read(offset, &data, length);
  if (size != length) {
    LOG_ERROR(
        "read sparse vector data failed: offset=%zu, "
        "length=%u, size=%zu",
        (size_t)offset, length, size);
    return nullptr;
  }
  return data;
}


node_id_t FlatSparseSearcherEntity::get_id(uint64_t key) const {
  if (ailego_unlikely(!mapping_chunk_)) {
    LOG_ERROR("Index missing mapping segment");
    return kInvalidNodeId;
  }

  //! Do binary search
  node_id_t start = 0UL;
  node_id_t end = doc_cnt();
  const void *data;
  node_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    if (ailego_unlikely(mapping_chunk_->read(idx * sizeof(node_id_t), &data,
                                             sizeof(node_id_t)) !=
                        sizeof(node_id_t))) {
      LOG_ERROR("Read key from segment failed");
      return kInvalidNodeId;
    }
    const uint64_t *mkey;
    node_id_t local_id = *reinterpret_cast<const node_id_t *>(data);
    if (ailego_unlikely(keys_chunk_->read(
                            local_id * sizeof(uint64_t), (const void **)(&mkey),
                            sizeof(uint64_t)) != sizeof(uint64_t))) {
      LOG_ERROR("Read key from segment failed");
      return kInvalidNodeId;
    }
    if (*mkey < key) {
      start = idx + 1;
    } else if (*mkey > key) {
      end = idx;
    } else {
      return local_id;
    }
  }
  return kInvalidNodeId;
}

uint64_t FlatSparseSearcherEntity::get_key(node_id_t id) const {
  const void *key;
  if (ailego_unlikely(
          keys_chunk_->read(id * sizeof(uint64_t), &key, sizeof(uint64_t)) !=
          sizeof(uint64_t))) {
    LOG_ERROR("Read key from segment failed");
    return kInvalidKey;
  }
  return *(reinterpret_cast<const uint64_t *>(key));
}

}  // namespace core
}  // namespace zvec