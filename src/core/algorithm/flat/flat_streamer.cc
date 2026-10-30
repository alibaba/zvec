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

#include "flat_streamer.h"
#include <limits>
#include <zvec/core/framework/index_factory.h>
#include "flat_streamer_context.h"
#include "flat_streamer_dumper.h"
#include "flat_streamer_provider.h"

namespace zvec {
namespace core {

#define WRITE_LOCK_GUARD(MUTEX, LOCK_NAME) \
  ailego::WriteLock write_lock(MUTEX);     \
  std::unique_lock<ailego::WriteLock> LOCK_NAME(write_lock);

#define READ_LOCK_GUARD_DEFER(MUTEX, LOCK_NAME) \
  ailego::ReadLock read_lock(MUTEX);            \
  std::unique_lock<ailego::ReadLock> LOCK_NAME(read_lock, std::defer_lock);

template <size_t BATCH_SIZE>
FlatStreamer<BATCH_SIZE>::FlatStreamer() = default;

template <size_t BATCH_SIZE>
FlatStreamer<BATCH_SIZE>::~FlatStreamer() {
  if (state_ == STATE_INITED || state_ == STATE_OPENED) {
    this->cleanup();
  }
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::init(const IndexMeta &imeta,
                                   const ailego::Params &params) {
  meta_ = imeta;
  meta_.set_streamer("FlatStreamer", 0U, params);

  // With a turbo quantizer the distances are computed by the quantizer's
  // SIMD kernels; the metric cannot describe the quantized layout
  // (e.g. Cosine rejects DT_INT8), so skip metric setup entirely.
  if (!quantizer_) {
    int error_code = InitializeMetric(meta_, &metric_);
    if (error_code != 0) {
      LOG_ERROR("Failed to initialize index metric %s, error=%d, %s",
                meta_.metric_name().c_str(), error_code,
                IndexError::What(error_code));
      return error_code;
    }
    if (metric_->query_metric()) {
      metric_ = metric_->query_metric();
    }
  }

  // 参数设置
  if (params.get(PARAM_FLAT_COLUMN_MAJOR_ORDER, &column_major_order_)) {
    meta_.set_major_order(column_major_order_ ? IndexMeta::MO_COLUMN
                                              : IndexMeta::MO_ROW);
  }
  // Verify column major order
  if (meta_.major_order() != IndexMeta::MO_ROW) {
    IndexMeta::DataType ft = meta_.data_type();

    bool support_column_major = true;
    if ((ft != IndexMeta::DT_FP32 && ft != IndexMeta::DT_FP16 &&
         ft != IndexMeta::DT_INT8 && ft != IndexMeta::DT_INT4 &&
         ft != IndexMeta::DT_BINARY32 && ft != IndexMeta::DT_BINARY64) ||
        (meta_.unit_size() != IndexMeta::UnitSizeof(ft))) {
      if (meta_.major_order() == IndexMeta::MO_COLUMN) {
        LOG_ERROR("Unsupported type %d with unit size %u.", ft,
                  meta_.unit_size());
        return IndexError_Unsupported;
      } else {
        support_column_major = false;
      }
    }
    if (meta_.element_size() % IndexMeta::AlignSizeof(ft) != 0) {
      if (meta_.major_order() == IndexMeta::MO_COLUMN) {
        LOG_ERROR("Unsupported type %d with dimension %u.", ft,
                  meta_.dimension());
        return IndexError_Unsupported;
      } else {
        support_column_major = false;
      }
    }

    if (meta_.major_order() == IndexMeta::MO_UNDEFINED &&
        support_column_major) {
      meta_.set_major_order(IndexMeta::MO_ROW);
    }
  }

  if (!quantizer_ && !VerifyMetric(meta_)) {
    LOG_ERROR("Invalid index metric %s.", meta_.metric_name().c_str());
    return IndexError_InvalidArgument;
  }

  read_block_size_ = FLAT_DEFAULT_READ_BLOCK_SIZE;
  params.get(PARAM_FLAT_READ_BLOCK_SIZE, &read_block_size_);
  params.get(PARAM_FLAT_USE_ID_MAP, &use_key_info_map_);
  params.get(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, &use_contiguous_memory_);

  use_external_vector_ = false;
  params.get(PARAM_FLAT_USE_EXTERNAL_VECTOR, &use_external_vector_);
  if (use_external_vector_) {
    if (use_contiguous_memory_ || meta_.major_order() != IndexMeta::MO_ROW ||
        meta_.data_type() != IndexMeta::DT_FP32 ||
        meta_.meta_type() != IndexMeta::MT_DENSE || meta_.dimension() == 0 ||
        meta_.element_size() != meta_.dimension() * sizeof(float) ||
        (meta_.metric_name() != "SquaredEuclidean" &&
         meta_.metric_name() != "InnerProduct") ||
        (quantizer_ && meta_.quantizer_name() != "Fp32Quantizer")) {
      return IndexError_Unsupported;
    }
    use_key_info_map_ = true;
  }

  state_ = STATE_INITED;

  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::init(
    const IndexMeta &imeta, const ailego::Params &params,
    const std::shared_ptr<zvec::turbo::Quantizer> &quantizer) {
  if (!quantizer) {
    return this->init(imeta, params);
  }

  bool column_major_order = false;
  params.get(PARAM_FLAT_COLUMN_MAJOR_ORDER, &column_major_order);
  if (column_major_order || imeta.major_order() == IndexMeta::MO_COLUMN) {
    LOG_ERROR("Quantizer distance does not support column index.");
    return IndexError_Unsupported;
  }

  quantizer_ = quantizer;

  IndexMeta row_meta = imeta;
  row_meta.set_major_order(IndexMeta::MO_ROW);
  int error_code = this->init(row_meta, params);
  if (error_code != 0) {
    quantizer_.reset();
    return error_code;
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::cleanup() {
  if (state_ == STATE_OPENED) {
    this->close();
  }

  entity_.reset();
  quantizer_.reset();

  LOG_DEBUG("FlatStreamer cleanup");
  state_ = STATE_INIT;
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::open(IndexStorage::Pointer stg) {
  if (!stg) {
    LOG_ERROR("Failed to open for invalid storage");
    return IndexError_InvalidArgument;
  }
  if (ailego_unlikely(state_ != STATE_INITED)) {
    LOG_ERROR("Open storage failed, init streamer first!");
    return IndexError_NoReady;
  }

  LOG_DEBUG("FlatStreamer open with %s", stg->name().c_str());

  const bool can_use_contiguous =
      stg->memory_block_type() != IndexStorage::MemoryBlock::MBT_BUFFERPOOL;
  const bool use_contiguous = use_contiguous_memory_ && can_use_contiguous;
  if (use_contiguous) {
    entity_ = std::make_unique<FlatContiguousStreamerEntity>(stats_);
  } else {
    entity_ = std::make_unique<FlatStreamerEntity>(stats_);
  }

  entity_->set_block_vector_count(kDefaultBlockVecCount);
  entity_->set_segment_size(kDefaultSegmentSize);
  entity_->enable_filter_same_key(true);
  entity_->set_linear_list_count(1);
  entity_->set_use_key_info_map(use_key_info_map_);
  entity_->set_use_external_vector(use_external_vector_);
  entity_->set_quantizer(quantizer_);
  *entity_->mutable_meta() = meta_;

  int ret = entity_->open(std::move(stg), meta_);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to open storage");
    return ret;
  }
  if (use_contiguous) {
    auto &contiguous_entity =
        static_cast<FlatContiguousStreamerEntity &>(*entity_);
    ret = contiguous_entity.build_contiguous_memory();
    if (ret != 0) {
      LOG_ERROR("Failed to build Flat contiguous memory");
      entity_->close();
      entity_.reset();
      return ret;
    }
  }
  magic_ = IndexContext::GenerateMagic();

  state_ = STATE_OPENED;

  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::close() {
  LOG_DEBUG("FlatStreamer close");

  int ret = entity_->flush_linear_meta();
  if (ret != 0) {
    return ret;
  }

  stats_.clear();

  ret = entity_->close();
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  state_ = STATE_INITED;
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::flush(uint64_t checkpoint) {
  LOG_INFO("FlatStreamer flush with checkpoint %zu", (size_t)checkpoint);
  return entity_->flush(checkpoint);
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::dump(const IndexDumper::Pointer &dumper) {
  if (use_external_vector_) return IndexError_Unsupported;
  std::string searcher_name = "FlatSearcher";
  if constexpr (BATCH_SIZE == 16) {
    searcher_name = "FlatSearcher16";
  }
  meta_.set_searcher(searcher_name, 0U, ailego::Params());
  WRITE_LOCK_GUARD(dump_mutex_, dump_lock);
  std::shared_ptr<FlatStreamerDumper<BATCH_SIZE>> bf_dumper =
      std::make_shared<FlatStreamerDumper<BATCH_SIZE>>(this);
  int ret = bf_dumper->dump(dumper);
  *(stats_.mutable_dumped_size()) += bf_dumper->dump_size();
  return ret;
}

template <size_t BATCH_SIZE>
IndexStreamer::Context::UPointer FlatStreamer<BATCH_SIZE>::create_context()
    const {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to create Context, open storage first!");
    return Context::UPointer();
  }
  return IndexStreamer::Context::Pointer(
      new FlatStreamerContext<BATCH_SIZE>(this));
}

template <size_t BATCH_SIZE>
IndexProvider::Pointer FlatStreamer<BATCH_SIZE>::create_provider() const {
  return IndexProvider::Pointer(new (std::nothrow)
                                    FlatStreamerProvider<BATCH_SIZE>(this));
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::add_impl(uint64_t pkey, const void *query,
                                       const IndexQueryMeta &qmeta,
                                       Context::UPointer &context) {
  if (!query || qmeta.dimension() != meta_.dimension() ||
      qmeta.data_type() != meta_.data_type() ||
      qmeta.element_size() != meta_.element_size()) {
    LOG_ERROR(
        "Failed to add for invalid arguments, query=%p, qmeta(type=%u "
        "dim=%u size=%u) vs meta(type=%u dim=%u size=%u)",
        query, qmeta.data_type(), qmeta.dimension(), qmeta.element_size(),
        meta_.data_type(), meta_.dimension(), meta_.element_size());
    (*stats_.mutable_discarded_count())++;
    return IndexError_InvalidArgument;
  }

  auto *ctx = dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!ctx) {
    LOG_ERROR("Failed to cast FlatStreamerContext");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Cast;
  }
  if (use_external_vector_) {
    if (pkey >= kInvalidNodeId) return IndexError_InvalidArgument;
    return add_with_id_impl(static_cast<uint32_t>(pkey), query, qmeta, context);
  }
  READ_LOCK_GUARD_DEFER(dump_mutex_, dump_lock);

  if (!dump_lock.try_lock()) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }

  // IndexQueryMeta iv_qmeta;
  // int ret = entity_.convert(query, qmeta, &query, &iv_qmeta);
  // if (ret != 0) {
  //   LOG_ERROR("Failed to convert record for %s",
  //             IndexError::What(ret));
  //   (*stats_.mutable_discarded_count())++;
  //   return ret;
  // }

  int ret = entity_->add(pkey, query, qmeta.element_size());
  if (ret != 0) {
    LOG_ERROR("Failed to add record for %s", IndexError::What(ret));
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::add_with_id_impl(uint32_t id, const void *query,
                                               const IndexQueryMeta &qmeta,
                                               Context::Pointer &context) {
  if (!query || qmeta.dimension() != meta_.dimension() ||
      qmeta.data_type() != meta_.data_type() ||
      qmeta.element_size() != meta_.element_size()) {
    LOG_ERROR(
        "Failed to add for invalid arguments, query=%p, qmeta(type=%u "
        "dim=%u size=%u) vs meta(type=%u dim=%u size=%u)",
        query, qmeta.data_type(), qmeta.dimension(), qmeta.element_size(),
        meta_.data_type(), meta_.dimension(), meta_.element_size());
    (*stats_.mutable_discarded_count())++;
    return IndexError_InvalidArgument;
  }

  auto *ctx = dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!ctx) {
    LOG_ERROR("Failed to cast FlatStreamerContext");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Cast;
  }
  if (use_external_vector_) {
    if (state_ != STATE_OPENED || !ctx->vector_source())
      return IndexError_NoReady;
    if (id == kInvalidNodeId) return IndexError_InvalidArgument;
    if (!ctx->vector_source()->get_vector(id)) return IndexError_NoExist;
  }
  READ_LOCK_GUARD_DEFER(dump_mutex_, dump_lock);

  if (!dump_lock.try_lock()) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }

  int ret = entity_->add_vector_with_id(id, query, qmeta.element_size());
  if (ret != 0) {
    LOG_ERROR("Failed to add record for %s", IndexError::What(ret));
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::search_bf_impl(const void *query,
                                             const IndexQueryMeta &qmeta,
                                             uint32_t count,
                                             Context::Pointer &context) const {
  if (use_external_vector_) {
    if (state_ != STATE_OPENED) return IndexError_NoReady;
    return search_external(query, qmeta, count, nullptr, context);
  }
  if (state_ != STATE_OPENED) return IndexError_NoReady;
  ailego_assert(query && count && !!context);
  ailego_assert(quantizer_ || metric_->is_matched(meta_, qmeta));

  FlatStreamerContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force streamer context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }

  if (!bf_context->threshold_is_valid()) return IndexError_NoReady;

  if (bf_context->group_by_search()) {
    return group_by_search_impl(query, qmeta, count, context);
  }

  bf_context->reset_results(count);

  for (size_t q = 0; q < count; ++q) {
    auto *heap = bf_context->result_heap();
    // Each query has its own result set; keep the configured threshold but
    // discard candidates (and heap ordering) from the previous query.
    heap->clear();
    uint32_t scan_count = 0;
    int ret = entity_->search(query, bf_context->filter(), &scan_count, heap,
                              bf_context->mutable_stats(q),
                              bf_context->search_scratch(), BATCH_SIZE);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to search for %s", IndexError::What(ret));
      return ret;
    }
    heap->sort();
    bf_context->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (use_external_vector_) {
    if (state_ != STATE_OPENED) return IndexError_NoReady;
    return search_external(query, qmeta, count, &p_keys, context);
  }
  if (state_ != STATE_OPENED) return IndexError_NoReady;
  if (count == 0 || count > p_keys.size()) return IndexError_InvalidArgument;
  ailego_assert(query && count && !!context);
  ailego_assert(quantizer_ || metric_->is_matched(meta_, qmeta));

  FlatStreamerContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force streamer context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }

  if (!bf_context->threshold_is_valid()) return IndexError_NoReady;

  if (bf_context->group_by_search()) {
    return group_by_search_p_keys_impl(query, p_keys, qmeta, count, context);
  }

  bf_context->reset_results(count);

  for (size_t q = 0; q < count; ++q) {
    auto *heap = bf_context->result_heap();
    heap->clear();
    // Candidate searches are already bounded by p_keys. Let the metric choose
    // its row batches without introducing storage-sized splits here.
    const size_t batch_size = std::max(size_t{1}, p_keys[q].size());
    int ret =
        entity_->search_by_p_keys(query, p_keys[q], bf_context->filter(), heap,
                                  bf_context->search_scratch(), batch_size);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to refine Flat candidates for %s",
                IndexError::What(ret));
      return ret;
    }
    bf_context->take_topk_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::search_external(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    const std::vector<std::vector<uint64_t>> *keys,
    Context::Pointer &context) const {
  auto *ctx = dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!ctx || !query || !count || (keys && keys->size() < count) ||
      qmeta.meta_type() != meta_.meta_type() ||
      qmeta.data_type() != meta_.data_type() ||
      qmeta.dimension() != meta_.dimension() ||
      qmeta.element_size() != meta_.element_size()) {
    return IndexError_InvalidArgument;
  }
  if (ctx->magic() != magic_) ctx->reset(this);
  const auto *source = ctx->vector_source();
  if (!source || !ctx->threshold_is_valid()) return IndexError_NoReady;
  if (ctx->group_by_search()) return IndexError_Unsupported;
  ctx->reset_results(count);
  // Bound both pointer scratch space and per-request pinned vector bytes.
  const uint32_t batch_limit = static_cast<uint32_t>(std::min<size_t>(
      1024,
      std::max<size_t>(BATCH_SIZE, read_block_size_ / meta_.element_size())));
  auto *scratch = ctx->search_scratch();

  for (uint32_t q = 0; q < count; ++q) {
    const void *current_query =
        static_cast<const char *>(query) + q * qmeta.element_size();
    auto *heap = ctx->result_heap();
    heap->clear();
    // Destroy the outstanding lease before its cursor, including on errors.
    VectorIterator::Pointer cursor;
    VectorBatch batch;
    if (!keys) cursor = source->create_iterator();
    auto clear_batch = [&]() {
      if (auto *pointers = std::get_if<PointerBatch>(&batch)) {
        pointers->ids.clear();
        pointers->vectors.clear();
        pointers->lease.reset();
      } else {
        std::get<ContiguousBatch>(batch) = {};
      }
    };
    auto consume = [&]() -> int {
      const auto *contiguous = std::get_if<ContiguousBatch>(&batch);
      const auto *pointers = std::get_if<PointerBatch>(&batch);
      if (!contiguous && !pointers) return IndexError_ReadData;
      const size_t batch_size =
          contiguous ? contiguous->count : pointers->ids.size();
      if (batch_size > batch_limit) return IndexError_ReadData;
      if (contiguous) {
        if (batch_size &&
            (!contiguous->data || contiguous->stride < meta_.element_size() ||
             batch_size - 1 >
                 (std::numeric_limits<size_t>::max() - meta_.element_size()) /
                     contiguous->stride ||
             contiguous->first_id >= kInvalidNodeId ||
             batch_size > kInvalidNodeId - contiguous->first_id))
          return IndexError_ReadData;
      } else if (pointers->ids.size() != pointers->vectors.size()) {
        return IndexError_ReadData;
      }
      const auto *ids = pointers ? pointers->ids.data() : nullptr;
      const uint32_t first_id = contiguous ? contiguous->first_id : 0;
      scratch->vector_ptrs.clear();
      scratch->vector_keys.clear();
      scratch->vector_ptrs.reserve(batch_size);
      scratch->vector_keys.reserve(batch_size);
      if (cursor)
        entity_->key_presence(ids, batch_size, first_id,
                              &scratch->key_presence);
      for (size_t i = 0; i < batch_size; ++i) {
        const auto id = ids ? ids[i] : first_id + static_cast<uint32_t>(i);
        if (id == kInvalidNodeId) return IndexError_ReadData;
        if (cursor && (!scratch->key_presence[i] ||
                       (ctx->filter().is_valid() && ctx->filter()(id)))) {
          ++*ctx->mutable_stats(q)->mutable_filtered_count();
          continue;
        }
        // Invisible or unregistered rows need not have a readable vector.
        const void *vector = contiguous
                                 ? static_cast<const char *>(contiguous->data) +
                                       i * contiguous->stride
                                 : pointers->vectors[i];
        if (!vector) return IndexError_ReadData;
        scratch->vector_ptrs.push_back(vector);
        scratch->vector_keys.push_back(id);
      }
      const size_t size = scratch->vector_ptrs.size();
      if (!size) return 0;
      scratch->distances.resize(size);
      if (quantizer_) {
        quantizer_->calc_distance_dp_query_batch(
            scratch->vector_ptrs.data(), static_cast<int>(size), current_query,
            scratch->distances.data());
      } else if (auto distance = metric_->batch_distance()) {
        distance(scratch->vector_ptrs.data(), current_query, size,
                 meta_.dimension(), scratch->distances.data(), nullptr);
      } else {
        auto scalar_distance = metric_->distance();
        for (size_t i = 0; i < size; ++i) {
          scalar_distance(current_query, scratch->vector_ptrs[i],
                          meta_.dimension(), &scratch->distances[i]);
        }
      }
      *ctx->mutable_stats(q)->mutable_dist_calced_count() += size;
      for (size_t i = 0; i < size; ++i) {
        const auto id = scratch->vector_keys[i];
        const auto distance = scratch->distances[i];
        heap->emplace(id, distance);
      }
      return 0;
    };

    if (cursor) {
      while (true) {
        clear_batch();
        int ret = cursor->next_batch(batch_limit, &batch);
        if (ret != 0) return ret;
        ret = consume();
        if (ret != 0) return ret;
        const auto *contiguous = std::get_if<ContiguousBatch>(&batch);
        if (contiguous ? contiguous->count == 0
                       : std::get<PointerBatch>(batch).ids.empty())
          break;
      }
    } else {
      std::vector<uint32_t> ids;
      ids.reserve(batch_limit);
      auto read_batch = [&]() -> int {
        if (ids.empty()) return 0;
        clear_batch();
        auto &pointers = std::get<PointerBatch>(batch);
        pointers.ids = ids;
        pointers.vectors.resize(ids.size());
        source->get_vectors(ids.data(), ids.size(), pointers.vectors.data());
        int ret = consume();
        ids.clear();
        return ret;
      };
      auto read_candidates = [&](const std::vector<uint64_t> &candidates) {
        for (auto key : candidates) {
          if (key >= kInvalidNodeId || (keys && !entity_->contains_key(key)))
            continue;
          if (ctx->filter().is_valid() && ctx->filter()(key)) {
            if (!keys) ++*ctx->mutable_stats(q)->mutable_filtered_count();
            continue;
          }
          ids.push_back(static_cast<uint32_t>(key));
          if (ids.size() == batch_limit) {
            int ret = read_batch();
            if (ret != 0) return ret;
          }
        }
        return read_batch();
      };
      if (keys) {
        int ret = read_candidates((*keys)[q]);
        if (ret != 0) return ret;
      } else {
        // Sources without a scan cursor can use the existing random reads.
        // Fix the scan length and copy only one batch of registered IDs.
        const size_t registered_count = entity_->registered_key_count();
        std::vector<uint64_t> registered_keys;
        for (size_t offset = 0; offset < registered_count;
             offset += batch_limit) {
          entity_->get_registered_keys(
              offset, std::min<size_t>(batch_limit, registered_count - offset),
              &registered_keys);
          int ret = read_candidates(registered_keys);
          if (ret != 0) return ret;
        }
      }
    }
    clear_batch();
    ctx->take_topk_result(q);
    if (ctx->fetch_vector()) {
      // Match HNSW's borrowed random-access result contract. Scan pages have
      // already been released, so result pointers must come from the source.
      for (auto &doc : *ctx->mutable_result(q)) {
        const void *vector = source->get_vector(doc.key());
        if (!vector) return IndexError_NoExist;
        IndexStorage::MemoryBlock block;
        block.reset(const_cast<void *>(vector));
        doc = IndexDocument(doc.key(), doc.score(), doc.index(), block);
      }
    }
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::search_by_p_keys_fast(
    const void *query, const std::vector<uint64_t> &keys, int64_t *output_ids,
    float *output_scores, size_t topk, const IndexQueryMeta &qmeta,
    Context::UPointer &context) const {
  if (state_ != STATE_OPENED) return IndexError_NoReady;
  if (use_external_vector_) return IndexError_NotImplemented;
  if (!query || !output_ids || topk == 0 || !context) {
    return IndexError_InvalidArgument;
  }
  auto *flat_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!flat_context) return IndexError_InvalidArgument;
  if (flat_context->filter().is_valid() || flat_context->group_by_search() ||
      flat_context->threshold_set() || flat_context->fetch_vector()) {
    return IndexError_NotImplemented;
  }
  if (quantizer_) {
    // The caller has already prepared the query using this quantizer.
    if (qmeta.dimension() != static_cast<uint32_t>(quantizer_->dim()) ||
        qmeta.quantize_type() != static_cast<uint32_t>(quantizer_->type()) ||
        qmeta.element_size() != quantizer_->quantized_query_vector_length()) {
      return IndexError_InvalidArgument;
    }
  } else if (!metric_ || !metric_->is_matched(meta_, qmeta)) {
    return IndexError_InvalidArgument;
  }
  if (flat_context->magic() != magic_) flat_context->reset(this);
  return entity_->search_by_p_keys_fast(query, keys, output_ids, output_scores,
                                        topk, flat_context->search_scratch());
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::group_by_search_impl(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (use_external_vector_) return IndexError_Unsupported;
  FlatStreamerContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force streamer context");
    return IndexError_InvalidArgument;
  }

  bf_context->resize_group_results(count);
  if (!bf_context->group_by().is_valid()) {
    LOG_ERROR("Invalid group-by function");
    return IndexError_InvalidArgument;
  }

  std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
    return bf_context->group_by()(key);
  };

  auto iterator = entity_->creater_iterator();

  for (size_t q = 0; q < count; ++q) {
    bf_context->group_topk_heaps().clear();
    for (node_id_t id = 0; id < entity_->vector_count(); ++id) {
      uint64_t key = entity_->key(id);
      if (!bf_context->filter().is_valid() || !bf_context->filter()(key)) {
        dist_t dist = 0;
        IndexStorage::MemoryBlock block;
        if (entity_->get_vector_by_key(key, block) != 0) continue;
        entity_->row_major_distance(query, block.data(), 1, &dist);

        std::string group_id = group_by(key);
        auto &topk_heap = bf_context->group_topk_heaps()[group_id];
        if (topk_heap.empty()) {
          topk_heap.limit(bf_context->group_topk());
        }
        topk_heap.emplace(key, dist);
      }
    }
    bf_context->topk_to_group_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::group_by_search_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (use_external_vector_) return IndexError_Unsupported;
  if (count == 0 || count > p_keys.size()) return IndexError_InvalidArgument;
  FlatStreamerContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force streamer context");
    return IndexError_InvalidArgument;
  }

  bf_context->resize_group_results(count);
  if (!bf_context->group_by().is_valid()) {
    LOG_ERROR("Invalid group-by function");
    return IndexError_InvalidArgument;
  }

  std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
    return bf_context->group_by()(key);
  };

  auto iterator = entity_->creater_iterator();

  for (size_t q = 0; q < count; ++q) {
    bf_context->group_topk_heaps().clear();
    for (node_id_t idx = 0; idx < p_keys[q].size(); ++idx) {
      uint64_t key = p_keys[q][idx];
      if (!bf_context->filter().is_valid() || !bf_context->filter()(key)) {
        dist_t dist = 0;
        IndexStorage::MemoryBlock block;
        if (entity_->get_vector_by_key(key, block) != 0) continue;
        entity_->row_major_distance(query, block.data(), 1, &dist);

        std::string group_id = group_by(key);
        auto &topk_heap = bf_context->group_topk_heaps()[group_id];
        if (topk_heap.empty()) {
          topk_heap.limit(bf_context->group_topk());
        }
        topk_heap.emplace(key, dist);
      }
    }
    bf_context->topk_to_group_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

// The direct operation is not virtual on the streamer.
template int FlatStreamer<16>::search_by_p_keys_fast(
    const void *, const std::vector<uint64_t> &, int64_t *, float *, size_t,
    const IndexQueryMeta &, Context::UPointer &) const;
template int FlatStreamer<32>::search_by_p_keys_fast(
    const void *, const std::vector<uint64_t> &, int64_t *, float *, size_t,
    const IndexQueryMeta &, Context::UPointer &) const;

INDEX_FACTORY_REGISTER_STREAMER_ALIAS(LinearStreamer, FlatStreamer<32>);
INDEX_FACTORY_REGISTER_STREAMER_ALIAS(FlatStreamer, FlatStreamer<32>);
INDEX_FACTORY_REGISTER_STREAMER_ALIAS(FlatStreamer16, FlatStreamer<16>);
INDEX_FACTORY_REGISTER_STREAMER_ALIAS(FlatStreamer32, FlatStreamer<32>);
}  // namespace core
}  // namespace zvec
