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

#include <ailego/math/inner_product_matrix.h>
#include "flat_sparse_context.h"

namespace zvec {
namespace core {

static inline IndexGroupDocumentList ConvertGroupMapToResult(
    std::unordered_map<std::string, IndexDocumentHeap> group_map,
    uint32_t group_num) {
  IndexGroupDocumentList result;

  std::vector<std::pair<std::string, float>> best_score_in_groups;
  for (auto itr = group_map.begin(); itr != group_map.end(); itr++) {
    const std::string &group_id = (*itr).first;
    auto &heap = (*itr).second;

    if (heap.size() > 0) {
      float best_score = heap[0].score();
      best_score_in_groups.push_back(std::make_pair(group_id, best_score));
    }
  }

  std::sort(best_score_in_groups.begin(), best_score_in_groups.end(),
            [](const std::pair<std::string, float> &a,
               const std::pair<std::string, float> &b) -> int {
              return a.second < b.second;
            });

  // truncate to group num
  for (uint32_t i = 0; i < group_num && i < best_score_in_groups.size(); ++i) {
    const std::string &group_id = best_score_in_groups[i].first;

    result.emplace_back(
        GroupIndexDocument(group_id, std::move(group_map[group_id])));
  }

  return result;
}

static inline int FlatSearch(const uint32_t *sparse_count,
                             const uint32_t *sparse_indices,
                             const void *sparse_query, bool with_p_keys,
                             const std::vector<std::vector<uint64_t>> &p_keys,
                             const IndexQueryMeta &qmeta, uint32_t count,
                             const IndexMeta, IndexContext::Pointer &context,
                             FlatSparseEntity *entity) {
  int ret;

  FlatSparseContext *ctx = dynamic_cast<FlatSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to FlatSparseContext failed");
    return IndexError_Cast;
  }

  // reset context results
  ctx->reset_results(count);

  const uint32_t *sparse_indices_tmp = sparse_indices;
  const void *sparse_query_tmp = sparse_query;

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
      return ctx->group_by()(key);
    };

    for (size_t q = 0; q < count; ++q) {
      std::string sparse_query_buffer;
      ailego::MinusInnerProductSparseMatrix<float>::transform_sparse_format(
          sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
          sparse_query_buffer);

      std::unordered_map<std::string, IndexDocumentHeap> group_heap{};

      if (with_p_keys) {
        ret = entity->search_group_p_keys(sparse_query_buffer, p_keys[q],
                                          ctx->filter(), group_by,
                                          ctx->group_topk(), &group_heap);
      } else {
        ret = entity->search_group(sparse_query_buffer, ctx->filter(), group_by,
                                   ctx->group_topk(), &group_heap);
      }

      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Failed to search group, ret=%s", IndexError::What(ret));
        return ret;
      }

      // sort group heap
      for (auto &group : group_heap) {
        group.second.sort();
      }

      auto group_result =
          ConvertGroupMapToResult(std::move(group_heap), ctx->group_num());

      // Populate sparse vector data when fetch_vector is enabled
      if (ctx->fetch_vector()) {
        for (auto &group_doc : group_result) {
          for (auto &doc : *group_doc.mutable_docs()) {
            node_id_t id = entity->get_id(doc.key());
            if (id != kInvalidNodeId) {
              IndexSparseDocument sparse_doc;
              IndexStorage::MemoryBlock vec_block;
              entity->get_sparse_vector(id, vec_block);
              const void *sparse_data = vec_block.data();
              if (sparse_data != nullptr) {
                SparseUtility::ReverseSparseFormat(sparse_data, sparse_doc,
                                                   entity->sparse_unit_size());
              }
              // Reconstruct doc with sparse vector data
              doc = IndexDocument(doc.key(), doc.score(), id, nullptr,
                                  sparse_doc);
            }
          }
        }
      }

      ctx->mutable_group_result(q)->swap(group_result);
    }
  } else {
    for (size_t q = 0; q < count; ++q) {
      std::string sparse_query_buffer;
      ailego::MinusInnerProductSparseMatrix<float>::transform_sparse_format(
          sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
          sparse_query_buffer);

      auto heap = ctx->result_heap();

      if (with_p_keys) {
        ret = entity->search_p_keys(sparse_query_buffer, p_keys[q],
                                    ctx->filter(), heap);
      } else {
        ret = entity->search(sparse_query_buffer, ctx->filter(), heap);
      }

      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Failed to search, ret=%s", IndexError::What(ret));
        return ret;
      }

      ctx->topk_to_result(q);

      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }
  }

  return 0;
}

}  // namespace core
}  // namespace zvec
