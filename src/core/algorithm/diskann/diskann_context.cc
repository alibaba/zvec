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

#include "diskann_context.h"
#include <chrono>
#include <new>
#include "diskann_params.h"
#include "diskann_result_payloads.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnContext::DiskAnnContext(const IndexMeta &meta,
                               const IndexMetric::Pointer &measure,
                               const DiskAnnEntity::Pointer &entity,
                               const turbo::Quantizer::Pointer &data_quantizer)
    : IndexContext(measure),
      dc_(entity.get(), meta, measure, data_quantizer),
      entity_{entity} {}

DiskAnnContext::SearchResults::SearchResults(DiskAnnContext &context)
    : context_(context) {
  context_.clear_scratch();
  context_.staged_results_ = this;
}

DiskAnnContext::SearchResults::~SearchResults() {
  context_.staged_results_ = nullptr;
  if (committed_) {
    context_.release_result_scratch();
  } else {
    context_.clear_scratch(/*discard_slots=*/true);
  }
}

void DiskAnnContext::SearchResults::commit() {
  const auto publish = [](auto &current, auto &staged) {
    // Grow before publishing any document. Swap individual lists so a caller
    // that holds result(i) across searches can still use that reference.
    current.resize(staged.size());
    for (size_t i = 0; i < staged.size(); ++i) current[i].swap(staged[i]);
  };
  if (context_.group_by_search()) {
    publish(context_.group_results_, group_results_);
    for (auto &result : context_.results_) result.clear();
  } else {
    publish(context_.results_, results_);
    for (auto &result : context_.group_results_) result.clear();
  }
  committed_ = true;
}

void DiskAnnContext::clear_scratch(bool discard_slots) {
  const size_t capacity = candidate_vectors_.capacity();
  release_result_scratch(
      discard_slots ||
      (capacity &&
       (capacity / 2 > topk_ ||
        ailego::MemoryLimitPool::get_instance().under_process_rss_pressure())));
  best_list_nodes_.clear();
  expanded_nodes_.clear();
  visit_filter_.clear();
  error_code_ = 0;
}

void DiskAnnContext::release_result_scratch(bool discard_slots) {
  release_candidates();
  topk_heap_.clear();
  // Keep small allocations for reuse when the context returns to the pool.
  constexpr size_t kRetainedEntries = 256;
  auto &entries = topk_heap_.mutable_container();
  if (entries.capacity() > kRetainedEntries) {
    std::decay_t<decltype(entries)>{}.swap(entries);
  }
  if (discard_slots || candidate_vectors_.capacity() > kRetainedEntries) {
    decltype(candidate_vectors_){}.swap(candidate_vectors_);
  }
  group_topk_heaps_.clear();
}

void DiskAnnContext::prepare_vector_candidates() {
  // Prepay at most one sector of small result vectors. Under RSS pressure,
  // or if the batch does not fit, allocate each slot only when needed.
  const size_t bytes_per_slot = size_t{element_size_} + 1;
  const size_t count =
      std::min<size_t>(topk_, DiskAnnUtil::kSectorSize / bytes_per_slot);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  if (count < 2 || element_size_ <= std::string{}.capacity() ||
      pool.under_process_rss_pressure())
    return;

  const size_t maximum = 2 * count * bytes_per_slot;
  if (!pool.try_reserve_process_backing(maximum)) return;
  size_t bytes = 0;
  try {
    candidate_vectors_.resize(count);
    for (auto &candidate : candidate_vectors_) {
      candidate.bytes.resize(element_size_);
      const size_t allocated = candidate.bytes.capacity() + 1;
      if (allocated > maximum - bytes) throw std::bad_alloc();
      bytes += allocated;
    }
  } catch (const std::bad_alloc &) {
    candidate_vectors_.clear();
    pool.finish_process_backing(maximum, false);
    return;
  }
  const auto stamp = pool.finish_reserved_process_backing(maximum, bytes);
  for (auto &candidate : candidate_vectors_) {
    candidate.memory.adopt(candidate.bytes.capacity() + 1, stamp);
  }
}

void DiskAnnContext::emplace_vector_candidate(diskann_id_t id, float score,
                                              const void *src) {
  if (topk_heap_.full() && !(score < topk_heap_.front().second.dist_)) return;
  diskann_id_t slot =
      topk_heap_.full() ? topk_heap_.front().second.vector_slot_ : kInvalidId;
  if (slot == kInvalidId) {
    if (candidate_vectors_.empty()) prepare_vector_candidates();
    slot = candidate_slots_used_;
    if (slot == candidate_vectors_.size()) {
      auto &candidate = candidate_vectors_.emplace_back();
      candidate.memory.resize(candidate.bytes, element_size_);
    }
    ++candidate_slots_used_;
  }
  std::memcpy(candidate_vectors_[slot].bytes.data(), src, element_size_);
  topk_heap_.emplace(id, VectorInfo(score, slot));
}

void DiskAnnContext::topk_to_result(uint32_t idx) {
  using Builder = TopkResultBuilder<DiskAnnContext, TopkHeap>;
  if (!use_vector_candidates()) {
    Builder::topk_to_result(idx);
    return;
  }
  auto release = ailego::ScopeGuard::Make([&] { release_candidates(); });
  candidate_vectors_.resize(candidate_slots_used_);
  Builder::topk_to_single_result(
      idx, [&](IndexDocumentList &docs, TopkHeap &heap, uint32_t size) {
        uint32_t accepted = 0;
        while (accepted < size && !(heap[accepted].second.dist_ > threshold()))
          ++accepted;
        DiskAnnResultPayloads batch(accepted);
        IndexDocumentList result;
        result.reserve(accepted);
        for (uint32_t i = 0; i < accepted; ++i) {
          const auto &entry = heap[i];
          diskann_key_t key;
          const int ret = entity_->read_key(entry.first, &key);
          if (ret != 0) {
            set_error_code(ret);
            return;
          }
          auto &candidate = candidate_vectors_[entry.second.vector_slot_];
          auto payload = std::allocate_shared<std::string>(
              DiskAnnResultPayloads::Allocator<std::string>{},
              std::move(candidate.bytes));
          batch.attach_vector(std::move(candidate.memory));
          result.emplace_back(key, entry.second.dist_, entry.first,
                              std::move(payload));
        }
        batch.commit();
        docs.swap(result);
      });
}

DiskAnnContext::Pointer DiskAnnContext::create_fetch_context(
    const IndexMeta &meta, const IndexMetric::Pointer &measure,
    const DiskAnnEntity::Pointer &entity, int *status) try {
  if (!measure || !entity) {
    if (status) *status = IndexError_InvalidArgument;
    return nullptr;
  }
  auto candidate = std::make_unique<DiskAnnContext>(meta, measure, entity);
  const int ret = candidate->init(kFetchContext, entity->max_degree(),
                                  entity->pq_chunk_num(), meta.element_size());
  if (status) *status = ret;
  if (ret != 0) return nullptr;
  return candidate;
} catch (const std::bad_alloc &) {
  if (status) *status = IndexError_NoMemory;
  return nullptr;
}

int DiskAnnContext::ensure_sector_buffer(size_t sectors) try {
  if (sectors == 0 || sectors > DiskAnnUtil::kMaxSectorReadNum)
    return IndexError_InvalidArgument;
  const size_t bytes = sectors * DiskAnnUtil::kSectorSize;
  if (bytes > sector_buffer_.size()) {
    sector_buffer_.resize_process_rss(bytes, DiskAnnUtil::kSectorSize);
  }
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnContext::resize_fetch_sector_buffer(
    const DiskAnnEntity::Pointer &entity) {
  if (!entity) {
    LOG_ERROR("Cannot size a DiskAnn fetch buffer without an entity");
    return IndexError_InvalidArgument;
  }

  const uint64_t sector_num_per_node =
      entity->node_per_sector() > 0
          ? 1
          : DiskAnnUtil::div_round_up(entity->max_node_size(),
                                      DiskAnnUtil::kSectorSize);
  if (sector_num_per_node == 0 ||
      sector_num_per_node > DiskAnnUtil::kMaxSectorReadNum) {
    LOG_ERROR("Invalid DiskAnn fetch sector count: %lu",
              static_cast<unsigned long>(sector_num_per_node));
    return IndexError_InvalidArgument;
  }

  return ensure_sector_buffer(static_cast<size_t>(sector_num_per_node));
}

int DiskAnnContext::init(ContextType type, uint32_t /*graph_degree*/,
                         uint32_t pq_chunk_num, uint32_t element_size,
                         bool setup_io_context) try {
  if (!entity_ || element_size == 0) {
    LOG_ERROR("Invalid DiskAnn context parameters");
    return IndexError_InvalidArgument;
  }
  type_ = type;
  element_size_ = element_size;
  pq_chunk_num_ = pq_chunk_num;

  if (type == kSearcherContext) {
    const size_t stride =
        DiskAnnUtil::round_up(static_cast<size_t>(element_size_), 256);
    if (stride > (std::numeric_limits<size_t>::max)() / 3)
      return IndexError_InvalidLength;
    query_workspace_.resize_process_rss(3 * stride, 256);
    auto *buffer = static_cast<char *>(query_workspace_.data());
    query_ = buffer;
    query_rotated_ = buffer + stride;
    coord_buffer_ = buffer + 2 * stride;
  } else if (type != kFetchContext) {
    DiskAnnUtil::alloc_aligned((void **)&query_, element_size_, 32);
    DiskAnnUtil::alloc_aligned((void **)&query_rotated_, element_size_, 32);
    if (!query_ || !query_rotated_) {
      LOG_ERROR("Failed to allocate DiskAnn query buffers");
      return IndexError_NoMemory;
    }
  }

  int ret;
  switch (type) {
    case kBuilderContext:
      ret = visit_filter_.init(VisitFilter::ByteMap, entity_->doc_cnt(),
                               entity_->doc_cnt(), negative_probility_);
      if (ret != 0) {
        LOG_ERROR("Create build visit filter failed");
        return ret;
      }
      break;

    case kSearcherContext:
      if (pq_chunk_num_ == 0) {
        LOG_ERROR("Invalid DiskAnn search context dimensions");
        return IndexError_InvalidArgument;
      }

      search_visit_filter_.init(entity_->doc_cnt());

      if (setup_io_context) {
        ret = setup_io_ctx(io_ctx_);
        if (ret != 0) {
          LOG_ERROR("setup io ctx error, ret=%d", ret);
          return ret;
        }
      }
      break;

    case kFetchContext:
      ret = resize_fetch_sector_buffer(entity_);
      if (ret != 0) {
        return ret;
      }

      if (setup_io_context) {
        ret = setup_io_ctx(io_ctx_);
        if (ret != 0) {
          LOG_ERROR("setup fetch io ctx error, ret=%d", ret);
          return ret;
        }
      }
      break;

    default:
      LOG_ERROR("Init context failed");
      return IndexError_Runtime;
  }

  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

DiskAnnContext::~DiskAnnContext() {
  // The sector buffer may still be the destination of an overlapped read if a
  // query exits early. Cancel and wait for every request before releasing any
  // memory that the I/O context can reference.
  if (type_ == kSearcherContext || type_ == kFetchContext) {
    destroy_io_ctx(io_ctx_);
  }

  visit_filter_.destroy();
  if (type_ != kSearcherContext) {
    DiskAnnUtil::free_aligned(query_);
    DiskAnnUtil::free_aligned(query_rotated_);
  }
}

int DiskAnnContext::update(const ailego::Params &params) {
  uint32_t list_size = list_size_;
  params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size);
  list_size_ = list_size;
  return 0;
}

int DiskAnnContext::update_context(
    ContextType type, const IndexMeta &meta,
    const IndexMetric::Pointer &measure, const DiskAnnEntity::Pointer &entity,
    uint32_t magic_num, const turbo::Quantizer::Pointer &data_quantizer) {
  if (ailego_unlikely(type != static_cast<ContextType>(type_))) {
    LOG_ERROR(
        "DiskAnnContext does not support shared by different type, "
        "src=%u dst=%u",
        type_, type);
    return IndexError_Unsupported;
  }

  magic_ = kInvalidMgic;

  switch (type) {
    case kBuilderContext:
      LOG_ERROR("BuildContext does not support update");
      return IndexError_NotImplemented;

    case kSearcherContext:
      break;

    case kFetchContext: {
      const int ret = resize_fetch_sector_buffer(entity);
      if (ret != 0) {
        return ret;
      }
      break;
    }

    case kReducerContext:
      break;

    default:
      LOG_ERROR("update context failed");
      return IndexError_Runtime;
  }

  entity_ = entity;
  update_index_metric(measure);
  dc_.update(entity_.get(), meta, measure, data_quantizer);
  magic_ = magic_num;

  return 0;
}

}  // namespace core
}  // namespace zvec
