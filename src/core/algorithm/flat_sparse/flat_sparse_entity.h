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

#include <unordered_map>
#include <zvec/core/framework/index_framework.h>
#include "flat_sparse_index_format.h"

namespace zvec {
namespace core {

using node_id_t = uint32_t;
constexpr node_id_t kInvalidNodeId = static_cast<node_id_t>(-1);

/*! Flat Sparse Entity
 */
class FlatSparseEntity {
 public:
  typedef std::shared_ptr<FlatSparseEntity> Pointer;

  //! Constructor
  explicit FlatSparseEntity() {}

  //! Destructor
  ~FlatSparseEntity() = default;

  //! Disable them
  FlatSparseEntity(const FlatSparseEntity &) = delete;
  FlatSparseEntity &operator=(const FlatSparseEntity &) = delete;

  //! Search in linear list with filter
  int search(const std::string &sparse_vector, const IndexFilter &filter,
             IndexDocumentHeap *heap) const {
    for (node_id_t i = 0; i < doc_cnt(); i++) {
      uint64_t key = get_key(i);
      if (ailego_unlikely(key == kInvalidKey)) {
        // LOG_ERROR("The key of node_id[%u] not found in keys map", i);
        // return IndexError_Runtime;
        continue;
      }
      if (!filter.is_valid() || !filter(key)) {
        float dist = get_search_distance(sparse_vector, i);
        heap->emplace(key, dist);
      }
    }

    return 0;
  }

  //! Search in linear list with filter and target pkeys
  int search_p_keys(const std::string &sparse_vector,
                    const std::vector<uint64_t> &p_keys,
                    const IndexFilter &filter, IndexDocumentHeap *heap) const {
    for (auto p_key : p_keys) {
      if (!filter.is_valid() || !filter(p_key)) {
        auto node_id = get_id(p_key);
        if (node_id != kInvalidNodeId) {
          float dist = get_search_distance(sparse_vector, node_id);
          heap->emplace(p_key, dist);
        }
      }
    }

    return 0;
  }

  //! Group search in linear list with filter
  int search_group(
      const std::string &sparse_vector, const IndexFilter &filter,
      const std::function<std::string(uint64_t)> &group_by_func, uint32_t topk,
      std::unordered_map<std::string, IndexDocumentHeap> *heap) const {
    for (node_id_t i = 0; i < doc_cnt(); i++) {
      uint64_t key = get_key(i);
      if (ailego_unlikely(key == kInvalidKey)) {
        LOG_ERROR("The key of node_id[%u] not found in keys map", i);
        return IndexError_Runtime;
      }
      if (!filter.is_valid() || !filter(key)) {
        float dist = get_search_distance(sparse_vector, i);

        std::string group_id = group_by_func(key);

        auto &group_heap = (*heap)[group_id];
        if (group_heap.empty()) {
          group_heap.limit(topk);
        }
        group_heap.emplace(key, dist);
      }
    }

    return 0;
  }

  //! Group search in linear list with filter and target pkeys
  int search_group_p_keys(
      const std::string &sparse_vector, const std::vector<uint64_t> &p_keys,
      const IndexFilter &filter,
      const std::function<std::string(uint64_t)> &group_by_func, uint32_t topk,
      std::unordered_map<std::string, IndexDocumentHeap> *heap) const {
    for (auto p_key : p_keys) {
      if (!filter.is_valid() || !filter(p_key)) {
        auto node_id = get_id(p_key);
        if (node_id != kInvalidNodeId) {
          float dist = get_search_distance(sparse_vector, node_id);

          std::string group_id = group_by_func(p_key);

          auto &group_heap = (*heap)[group_id];
          if (group_heap.empty()) {
            group_heap.limit(topk);
          }
          group_heap.emplace(p_key, dist);
        }
      }
    }

    return 0;
  }

  //! Get sparse vector by key
  int get_sparse_vector(uint64_t key, std::string *sparse_vector) const {
    const void *sparse_vector_ptr;
    uint32_t sparse_vector_len;
    int ret = get_sparse_vector_ptr_by_key(key, &sparse_vector_ptr,
                                           &sparse_vector_len);
    if (ret != 0) {
      return ret;
    }
    *sparse_vector = std::string(static_cast<const char *>(sparse_vector_ptr),
                                 sparse_vector_len);
    return 0;
  }

  //! Get sparse vector by node id
  const void *get_sparse_vector(node_id_t id) const {
    const void *sparse_vector_ptr;
    uint32_t sparse_vector_len;
    int ret =
        get_sparse_vector_ptr_by_id(id, &sparse_vector_ptr, &sparse_vector_len);
    if (ret != 0) {
      return nullptr;
    }
    return sparse_vector_ptr;
  }

  int get_sparse_vector_by_key(const uint64_t key,
                               std::string *sparse_vector) const {
    uint32_t sparse_vector_len;
    IndexStorage::MemoryBlock sparse_vector_block;
    int ret = get_sparse_vector_ptr_by_key(key, sparse_vector_block,
                                           &sparse_vector_len);
    if (ret != 0) {
      return ret;
    }
    *sparse_vector =
        std::string(static_cast<const char *>(sparse_vector_block.data()),
                    sparse_vector_len);
    return 0;
  }

  int get_sparse_vector(node_id_t id,
                        IndexStorage::MemoryBlock &sparse_vector_block) const {
    uint32_t sparse_vector_len;
    return get_sparse_vector_ptr_by_id(id, sparse_vector_block,
                                       &sparse_vector_len);
  }

  int get_sparse_vector_ptr_by_key(uint64_t key, const void **sparse_vector_ptr,
                                   uint32_t *sparse_vector_len_ptr) const {
    auto node_id = get_id(key);
    if (node_id == kInvalidNodeId) {
      return IndexError_NoExist;
    }

    return get_sparse_vector_ptr_by_id(node_id, sparse_vector_ptr,
                                       sparse_vector_len_ptr);
  }

  int get_sparse_vector_ptr_by_key(
      const uint64_t key, IndexStorage::MemoryBlock &sparse_vector_block,
      uint32_t *sparse_vector_len_ptr) const {
    auto node_id = get_id(key);
    if (node_id == kInvalidNodeId) {
      return IndexError_NoExist;
    }

    return get_sparse_vector_ptr_by_id(node_id, sparse_vector_block,
                                       sparse_vector_len_ptr);
  }

  std::vector<uint64_t> get_keys() const {
    std::vector<uint64_t> keys;
    node_id_t doc_total_cnt = doc_cnt();
    for (node_id_t node_id = 0; node_id < doc_total_cnt; ++node_id) {
      uint64_t key = get_key(node_id);
      if (key == kInvalidKey) {
        return {kInvalidKey};
      } else {
        keys.push_back(key);
      }
    }

    return keys;
  }


 public:
  virtual uint32_t doc_cnt() const = 0;

  virtual uint32_t total_sparse_count() const = 0;

  virtual node_id_t get_id(uint64_t key) const = 0;

  virtual uint64_t get_key(node_id_t id) const = 0;

  virtual int get_sparse_vector_ptr_by_id(
      node_id_t id, const void **sparse_vector,
      uint32_t *sparse_vector_len) const = 0;

  virtual int get_sparse_vector_ptr_by_id(
      const node_id_t /*id*/,
      IndexStorage::MemoryBlock & /*sparse_vector_block*/,
      uint32_t * /*sparse_vector_len*/) const {
    return IndexError_NotImplemented;
  }


  virtual float get_search_distance(const std::string &vector,
                                    node_id_t target_node_id) const = 0;
  virtual size_t sparse_unit_size() const = 0;
};

}  // namespace core
}  // namespace zvec
