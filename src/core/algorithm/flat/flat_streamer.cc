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
FlatStreamer<BATCH_SIZE>::FlatStreamer() : entity_(stats_) {}

template <size_t BATCH_SIZE>
FlatStreamer<BATCH_SIZE>::~FlatStreamer() {
  if (state_ == STATE_INITED) {
    this->cleanup();
  }
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::init(const IndexMeta &imeta,
                                   const ailego::Params &params) {
  meta_ = imeta;
  meta_.set_streamer("FlatStreamer", 0U, params);

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

  if (!VerifyMetric(meta_)) {
    LOG_ERROR("Invalid index metric %s.", meta_.metric_name().c_str());
    return IndexError_InvalidArgument;
  }

  read_block_size_ = FLAT_DEFAULT_READ_BLOCK_SIZE;
  params.get(PARAM_FLAT_READ_BLOCK_SIZE, &read_block_size_);
  params.get(PARAM_FLAT_USE_ID_MAP, &use_key_info_map_);

  // entity init
  uint32_t block_vector_count = kDefaultBlockVecCount;
  uint32_t segment_size = kDefaultSegmentSize;
  bool filter_same_key = true;
  entity_.set_block_vector_count(block_vector_count);
  entity_.set_segment_size(segment_size);
  entity_.enable_filter_same_key(filter_same_key);
  entity_.set_linear_list_count(1);
  entity_.set_use_key_info_map(use_key_info_map_);
  *entity_.mutable_meta() = meta_;

  state_ = STATE_INITED;

  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::cleanup() {
  if (state_ == STATE_OPENED) {
    this->close();
  }

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

  int ret = entity_.open(std::move(stg), meta_);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to open storage");
    return ret;
  }
  magic_ = IndexContext::GenerateMagic();

  state_ = STATE_OPENED;

  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::close(void) {
  LOG_DEBUG("FlatStreamer close");

  entity_.flush_linear_meta();

  stats_.clear();

  int ret = entity_.close();
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  state_ = STATE_INITED;
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::flush(uint64_t checkpoint) {
  LOG_INFO("FlatStreamer flush with checkpoint %zu", (size_t)checkpoint);
  return entity_.flush(checkpoint);
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::dump(const IndexDumper::Pointer &dumper) {
  std::string searcher_name = "FlatSearcher";
  if (BATCH_SIZE == 16) {
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
IndexStreamer::Context::UPointer FlatStreamer<BATCH_SIZE>::create_context(
    void) const {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to create Context, open storage first!");
    return Context::UPointer();
  }
  return IndexStreamer::Context::Pointer(
      new FlatStreamerContext<BATCH_SIZE>(this));
}

template <size_t BATCH_SIZE>
IndexProvider::Pointer FlatStreamer<BATCH_SIZE>::create_provider(void) const {
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

  int ret = entity_.add(pkey, query, qmeta.element_size());
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

  READ_LOCK_GUARD_DEFER(dump_mutex_, dump_lock);

  if (!dump_lock.try_lock()) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }

  int ret = entity_.add_vector_with_id(id, query, qmeta.element_size());
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
  ailego_assert(query && count && !!context);
  ailego_assert(metric_->is_matched(meta_, qmeta));

  FlatStreamerContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force streamer context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }

  if (bf_context->group_by_search()) {
    return group_by_search_impl(query, qmeta, count, context);
  }

  bf_context->reset_results(count);
  auto &filter = bf_context->filter();

  for (size_t q = 0; q < count; ++q) {
    auto *heap = bf_context->result_heap();
    auto *context_stats = bf_context->mutable_stats(q);
    uint32_t scan_count = 0;
    int ret = entity_.search(query, filter, &scan_count, heap, context_stats);
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
  ailego_assert(query && count && !!context);
  ailego_assert(metric_->is_matched(meta_, qmeta));

  FlatStreamerContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatStreamerContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force streamer context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }

  if (bf_context->group_by_search()) {
    return group_by_search_p_keys_impl(query, p_keys, qmeta, count, context);
  }

  bf_context->reset_results(count);
  auto &filter = bf_context->filter();

  for (size_t q = 0; q < count; ++q) {
    auto *heap = bf_context->result_heap();
    for (node_id_t idx = 0; idx < p_keys[q].size(); ++idx) {
      uint64_t key = p_keys[q][idx];
      if (!filter.is_valid() || !filter(key)) {
        dist_t dist = 0;
        IndexStorage::MemoryBlock block;
        if (entity_.get_vector_by_key(key, block) != 0) continue;
        entity_.row_major_distance(query, block.data(), 1, &dist);
        heap->emplace(key, dist);
      }
    }
    heap->sort();
    bf_context->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatStreamer<BATCH_SIZE>::group_by_search_impl(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
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

  auto iterator = entity_.creater_iterator();

  for (size_t q = 0; q < count; ++q) {
    bf_context->group_topk_heaps().clear();
    for (node_id_t id = 0; id < entity_.vector_count(); ++id) {
      uint64_t key = entity_.key(id);
      if (!bf_context->filter().is_valid() || !bf_context->filter()(key)) {
        dist_t dist = 0;
        IndexStorage::MemoryBlock block;
        if (entity_.get_vector_by_key(key, block) != 0) continue;
        entity_.row_major_distance(query, block.data(), 1, &dist);

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

  auto iterator = entity_.creater_iterator();

  for (size_t q = 0; q < count; ++q) {
    bf_context->group_topk_heaps().clear();
    for (node_id_t idx = 0; idx < p_keys[q].size(); ++idx) {
      uint64_t key = p_keys[q][idx];
      if (!bf_context->filter().is_valid() || !bf_context->filter()(key)) {
        dist_t dist = 0;
        IndexStorage::MemoryBlock block;
        if (entity_.get_vector_by_key(key, block) != 0) continue;
        entity_.row_major_distance(query, block.data(), 1, &dist);

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

INDEX_FACTORY_REGISTER_STREAMER_ALIAS(LinearStreamer, FlatStreamer<32>);
INDEX_FACTORY_REGISTER_STREAMER_ALIAS(FlatStreamer, FlatStreamer<32>);
INDEX_FACTORY_REGISTER_STREAMER_ALIAS(FlatStreamer16, FlatStreamer<16>);
INDEX_FACTORY_REGISTER_STREAMER_ALIAS(FlatStreamer32, FlatStreamer<32>);
}  // namespace core
}  // namespace zvec
