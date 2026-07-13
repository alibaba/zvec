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
#include "combined_vector_column_indexer.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>

namespace zvec {

namespace {

bool IsBetterScore(MetricType metric_type, float lhs, float rhs) {
  switch (metric_type) {
    case MetricType::IP:
      return lhs > rhs;
    case MetricType::L2:
    case MetricType::COSINE:
    default:
      return lhs < rhs;
  }
}

bool HasRevertedValues(const std::vector<std::string> &values) {
  return std::any_of(values.begin(), values.end(),
                     [](const auto &value) { return !value.empty(); });
}

bool HasRevertedValues(const std::vector<std::vector<std::string>> &values) {
  return std::any_of(values.begin(), values.end(), [](const auto &group) {
    return HasRevertedValues(group);
  });
}

struct ResultDoc {
  core::IndexDocument doc;
  // Keep fetched/reverted payloads attached to the doc while sorting and
  // truncating, so parallel result vectors cannot drift out of sync.
  std::string reverted_vector;
  std::string reverted_sparse_values;
};

class VectorResultAccumulator {
 public:
  // Collect plain topk results from each block after translating block-local
  // doc IDs back to segment-level IDs.
  void AddBlock(uint32_t block_offset, VectorIndexResults *results) {
    auto &docs = results->docs();
    auto &reverted_vectors = results->reverted_vector_list();
    auto &reverted_sparse_values = results->reverted_sparse_values_list();
    docs_.reserve(docs_.size() + docs.size());

    for (size_t i = 0; i < docs.size(); ++i) {
      auto doc = std::move(docs[i]);
      doc.set_key(block_offset + doc.key());

      ResultDoc result_doc{std::move(doc), {}, {}};
      if (i < reverted_vectors.size()) {
        result_doc.reverted_vector = std::move(reverted_vectors[i]);
      }
      if (i < reverted_sparse_values.size()) {
        result_doc.reverted_sparse_values =
            std::move(reverted_sparse_values[i]);
      }
      docs_.emplace_back(std::move(result_doc));
    }
  }

  IndexResults::Ptr Finish(bool is_sparse, MetricType metric_type,
                           uint32_t topk) {
    // Finish turns accumulated block docs into the public result format:
    // rank all docs globally, keep topk, then split ResultDoc back into the
    // doc list and optional reverted payload lists expected by
    // VectorIndexResults.
    std::sort(docs_.begin(), docs_.end(),
              [metric_type](const ResultDoc &lhs, const ResultDoc &rhs) {
                return IsBetterScore(metric_type, lhs.doc.score(),
                                     rhs.doc.score());
              });
    if (docs_.size() > topk) {
      docs_.resize(topk);
    }

    core::IndexDocumentList doc_list;
    std::vector<std::string> reverted_vector_list;
    std::vector<std::string> reverted_sparse_values_list;
    doc_list.reserve(docs_.size());
    reverted_vector_list.reserve(docs_.size());
    reverted_sparse_values_list.reserve(docs_.size());

    for (auto &doc : docs_) {
      doc_list.emplace_back(std::move(doc.doc));
      reverted_vector_list.emplace_back(std::move(doc.reverted_vector));
      reverted_sparse_values_list.emplace_back(
          std::move(doc.reverted_sparse_values));
    }
    if (!HasRevertedValues(reverted_vector_list)) {
      reverted_vector_list.clear();
    }
    if (!HasRevertedValues(reverted_sparse_values_list)) {
      reverted_sparse_values_list.clear();
    }

    return std::make_unique<VectorIndexResults>(
        is_sparse, std::move(doc_list), std::move(reverted_vector_list),
        std::move(reverted_sparse_values_list));
  }

 private:
  std::vector<ResultDoc> docs_;
};

class GroupResultAccumulator {
 private:
  struct GroupResult {
    std::string group_id;
    std::vector<ResultDoc> docs;
  };

 public:
  // Merge same-named groups across blocks. The per-doc payload stays inside
  // ResultDoc until the final GroupVectorIndexResults is materialized.
  void AddBlock(uint32_t block_offset, GroupVectorIndexResults *results) {
    auto &groups = results->groups();
    auto &reverted_vectors = results->reverted_vector_list();
    auto &reverted_sparse_values = results->reverted_sparse_values_list();

    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      auto &group = groups[group_idx];
      auto *docs = group.mutable_docs();
      auto &merged_docs = docs_by_group_[group.group_id()];
      merged_docs.reserve(merged_docs.size() + docs->size());

      for (size_t doc_idx = 0; doc_idx < docs->size(); ++doc_idx) {
        auto doc = std::move((*docs)[doc_idx]);
        doc.set_key(block_offset + doc.key());

        ResultDoc result_doc{std::move(doc), {}, {}};
        if (group_idx < reverted_vectors.size() &&
            doc_idx < reverted_vectors[group_idx].size()) {
          result_doc.reverted_vector =
              std::move(reverted_vectors[group_idx][doc_idx]);
        }
        if (group_idx < reverted_sparse_values.size() &&
            doc_idx < reverted_sparse_values[group_idx].size()) {
          result_doc.reverted_sparse_values =
              std::move(reverted_sparse_values[group_idx][doc_idx]);
        }
        merged_docs.emplace_back(std::move(result_doc));
      }
    }
  }

  bool empty() const {
    return docs_by_group_.empty();
  }

  IndexResults::Ptr Finish(MetricType metric_type, uint32_t group_topk,
                           uint32_t group_count) {
    // Finish first ranks docs inside each merged group and trims group_topk.
    // It then ranks groups by their best remaining doc, trims group_count, and
    // finally expands ResultDoc back into GroupVectorIndexResults payloads.
    std::vector<GroupResult> groups;
    groups.reserve(docs_by_group_.size());

    for (auto &[group_id, docs] : docs_by_group_) {
      if (docs.empty()) {
        continue;
      }
      std::sort(docs.begin(), docs.end(),
                [metric_type](const ResultDoc &lhs, const ResultDoc &rhs) {
                  return IsBetterScore(metric_type, lhs.doc.score(),
                                       rhs.doc.score());
                });
      if (group_topk > 0 && docs.size() > group_topk) {
        docs.resize(group_topk);
      }
      groups.emplace_back(GroupResult{group_id, std::move(docs)});
    }

    std::sort(groups.begin(), groups.end(),
              [metric_type](const GroupResult &lhs, const GroupResult &rhs) {
                if (lhs.docs.empty() || rhs.docs.empty()) {
                  return !lhs.docs.empty() && rhs.docs.empty();
                }
                const float lhs_score = lhs.docs[0].doc.score();
                const float rhs_score = rhs.docs[0].doc.score();
                if (lhs_score == rhs_score) {
                  return lhs.group_id < rhs.group_id;
                }
                return IsBetterScore(metric_type, lhs_score, rhs_score);
              });
    if (group_count > 0 && groups.size() > group_count) {
      groups.resize(group_count);
    }

    core::IndexGroupDocumentList group_list;
    std::vector<std::vector<std::string>> reverted_vector_list;
    std::vector<std::vector<std::string>> reverted_sparse_values_list;
    group_list.reserve(groups.size());
    reverted_vector_list.reserve(groups.size());
    reverted_sparse_values_list.reserve(groups.size());

    for (auto &group : groups) {
      core::GroupIndexDocument group_doc;
      group_doc.set_group_id(group.group_id);
      auto *docs = group_doc.mutable_docs();
      docs->reserve(group.docs.size());

      std::vector<std::string> group_reverted_vectors;
      std::vector<std::string> group_reverted_sparse_values;
      group_reverted_vectors.reserve(group.docs.size());
      group_reverted_sparse_values.reserve(group.docs.size());
      for (auto &doc : group.docs) {
        docs->emplace_back(std::move(doc.doc));
        group_reverted_vectors.emplace_back(std::move(doc.reverted_vector));
        group_reverted_sparse_values.emplace_back(
            std::move(doc.reverted_sparse_values));
      }

      group_list.emplace_back(std::move(group_doc));
      reverted_vector_list.emplace_back(std::move(group_reverted_vectors));
      reverted_sparse_values_list.emplace_back(
          std::move(group_reverted_sparse_values));
    }

    if (!HasRevertedValues(reverted_vector_list)) {
      reverted_vector_list.clear();
    }
    if (!HasRevertedValues(reverted_sparse_values_list)) {
      reverted_sparse_values_list.clear();
    }

    return std::make_unique<GroupVectorIndexResults>(
        std::move(group_list), std::move(reverted_vector_list),
        std::move(reverted_sparse_values_list));
  }

 private:
  std::unordered_map<std::string, std::vector<ResultDoc>> docs_by_group_;
};

}  // namespace

CombinedVectorColumnIndexer::CombinedVectorColumnIndexer(
    const std::vector<VectorColumnIndexer::Ptr> &indexers,
    const std::vector<VectorColumnIndexer::Ptr> &normal_indexers,
    const FieldSchema &field_schema, const SegmentMeta &segment_meta,
    std::vector<BlockMeta> blocks, MetricType metric_type, bool is_quantized)
    : field_schema_(field_schema),
      indexers_(std::move(indexers)),
      normal_indexers_(std::move(normal_indexers)),
      blocks_(std::move(blocks)),
      metric_type_(metric_type),
      is_quantized_(is_quantized) {
  if (segment_meta.has_writing_forward_block()) {
    if (is_quantized_) {
      BlockMeta quant_block = segment_meta.writing_forward_block().value();
      quant_block.set_type(BlockType::VECTOR_INDEX_QUANTIZE);
      blocks_.push_back(std::move(quant_block));
    } else {
      BlockMeta block = segment_meta.writing_forward_block().value();
      block.set_type(BlockType::VECTOR_INDEX);
      blocks_.push_back(std::move(block));
    }
  }

  int block_offset = 0;
  for (size_t i = 0; i < indexers_.size(); ++i) {
    auto &block_meta = blocks_[i];
    block_offsets_.push_back(block_offset);
    block_offset += block_meta.doc_count_;
  }

  min_doc_id_ = segment_meta.min_doc_id();
}

Result<IndexResults::Ptr> CombinedVectorColumnIndexer::Search(
    const vector_column_params::VectorData &vector_data,
    const vector_column_params::QueryParams &query_params) {
  // Search runs each block with block-local query params, then folds those
  // partial results into one segment-level result. The accumulators keep doc
  // IDs and fetched/reverted payloads aligned while final sorting and
  // truncation are deferred until every block has been searched.
  VectorResultAccumulator vector_results;
  GroupResultAccumulator group_results;

  // query_params.bf_pks is segment level, here we need to convert it to block
  // level
  std::vector<std::vector<uint64_t>> block_bf_pks(indexers_.size());

  if (!query_params.bf_pks.empty()) {
    // dispatcher pks to corresponding block_bf_pks
    for (auto &pk : query_params.bf_pks[0]) {
      for (size_t i = 0; i < block_offsets_.size(); ++i) {
        if (pk >= block_offsets_[i] &&
            pk < block_offsets_[i] + blocks_[i].doc_count_) {
          block_bf_pks[i].push_back(
              static_cast<uint64_t>(pk - block_offsets_[i]));
          break;
        }
      }
    }
  }

  auto q_params = query_params.query_params;
  for (size_t i = 0; i < indexers_.size(); ++i) {
    if (!query_params.bf_pks.empty() && block_bf_pks[i].empty()) {
      LOG_DEBUG(
          "query_params has bf_pks, but block_bf_pks[%zu] is empty, just skip "
          "this indexer",
          i);
      continue;
    }
    zvec::Result<zvec::IndexResults::Ptr> result{nullptr};
    float scale_factor{};
    bool need_refine{false};
    if (q_params && q_params->is_using_refiner()) {
      if (normal_indexers_.size() != indexers_.size()) {
        return tl::make_unexpected(Status::InvalidArgument(
            "normal indexers size[", normal_indexers_.size(),
            "] not match indexers size[", indexers_.size(), "]"));
      }
      // query_params of HNSW doesn't have scale_factor
      if (q_params->type() == IndexType::FLAT) {
        scale_factor = std::dynamic_pointer_cast<FlatQueryParams>(q_params)
                           ->scale_factor();
      } else if (q_params->type() == IndexType::IVF) {
        scale_factor =
            std::dynamic_pointer_cast<IVFQueryParams>(q_params)->scale_factor();
      }
      need_refine = true;
    }

    // Rewrite segment-level query state to the current block: filters and
    // group_by callbacks see segment IDs, while the underlying block indexer
    // searches with block-local doc IDs.
    const IndexFilter *filter{nullptr};
    auto per_block_filter =
        BlockOffsetFilter{query_params.filter, block_offsets_[i]};
    if (query_params.filter) {
      if (block_offsets_[i] > 0) {
        filter = &per_block_filter;
      } else {
        filter = query_params.filter;
      }
    }

    std::unique_ptr<vector_column_params::GroupByParams> group_by;
    if (query_params.group_by) {
      auto group_by_func = query_params.group_by->group_by;
      auto block_offset = block_offsets_[i];
      group_by = std::make_unique<vector_column_params::GroupByParams>(
          query_params.group_by->group_topk, query_params.group_by->group_count,
          [group_by_func = std::move(group_by_func),
           block_offset](uint64_t block_doc_id) {
            return group_by_func(block_doc_id + block_offset);
          });
    }

    vector_column_params::QueryParams modified_query_params{
        query_params.data_type,
        query_params.dimension,
        query_params.topk,
        filter,
        query_params.fetch_vector,
        query_params.query_params,
        std::move(group_by),
        {},
        need_refine ? std::shared_ptr<vector_column_params::RefinerParam>(
                          new vector_column_params::RefinerParam{
                              scale_factor, normal_indexers_[i]})
                    : nullptr,
        query_params.extra_params};

    if (!query_params.bf_pks.empty()) {
      modified_query_params.bf_pks.emplace_back(block_bf_pks[i]);
    }

    result = indexers_[i]->Search(vector_data, modified_query_params);
    if (!result) {
      return tl::make_unexpected(result.error());
    }

    auto index_results = result.value();

    GroupVectorIndexResults *group_index_results =
        dynamic_cast<GroupVectorIndexResults *>(index_results.get());
    if (group_index_results != nullptr) {
      group_results.AddBlock(block_offsets_[i], group_index_results);
      continue;
    }

    VectorIndexResults *vector_index_results =
        dynamic_cast<VectorIndexResults *>(index_results.get());
    if (vector_index_results != nullptr) {
      vector_results.AddBlock(block_offsets_[i], vector_index_results);
    }
  }

  if (!group_results.empty()) {
    const uint32_t group_topk =
        query_params.group_by ? query_params.group_by->group_topk : 0;
    const uint32_t group_count =
        query_params.group_by ? query_params.group_by->group_count : 0;
    return group_results.Finish(metric_type_, group_topk, group_count);
  }
  return vector_results.Finish(field_schema_.is_sparse_vector(), metric_type_,
                               query_params.topk);
}

Result<vector_column_params::VectorDataBuffer>
CombinedVectorColumnIndexer::Fetch(uint32_t segment_doc_id) const {
  int32_t target_block_doc_id = -1;
  size_t target_block_idx = 0;

  uint32_t block_offset = 0;
  for (size_t i = 0; i < blocks_.size(); ++i) {
    auto &block_meta = blocks_[i];
    if (block_offset <= segment_doc_id &&
        segment_doc_id < block_offset + block_meta.doc_count_) {
      target_block_doc_id = segment_doc_id - block_offset;
      target_block_idx = i;
      break;
    }
    block_offset += block_meta.doc_count_;
  }

  if (target_block_doc_id == -1) {
    LOG_ERROR("Can't find block for doc_id[%u]", segment_doc_id);
    return tl::make_unexpected(
        Status::NotFound("Can't find block for doc_id:", segment_doc_id));
  }

  auto indexer = indexers_[target_block_idx];
  return indexer->Fetch(target_block_doc_id);
}

}  // namespace zvec
