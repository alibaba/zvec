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
#include "flat_searcher.h"
#include <framework/index_helper.h>
#include <framework/index_searcher.h>
#include "flat_distance_matrix.h"
#include "flat_searcher_context.h"
#include "flat_searcher_provider.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

template <size_t BATCH_SIZE>
IndexProvider::Pointer FlatSearcher<BATCH_SIZE>::create_provider(void) const {
  std::lock_guard<std::mutex> lock(mapping_mutex_);

  if (mapping_.empty()) {
    auto mapping_segment = container_->get(FLAT_SEGMENT_MAPPING_SEG_ID);
    if (!mapping_segment) {
      LOG_ERROR("Failed to fetch segment %s",
                FLAT_SEGMENT_MAPPING_SEG_ID.c_str());
      return nullptr;
    }

    if (mapping_segment->data_size() % sizeof(uint32_t) != 0) {
      LOG_ERROR("Invalid data size %zu of mapping segment",
                mapping_segment->data_size());
      return nullptr;
    }

    size_t mapping_count = mapping_segment->data_size() / sizeof(uint32_t);
    if (mapping_count * meta_.element_size() !=
        features_segment_->data_size()) {
      LOG_ERROR("Invalid data size %zd of mapping segment",
                features_segment_->data_size());
      return nullptr;
    }

    const uint32_t *mapping = nullptr;
    if (mapping_segment->read(0, reinterpret_cast<const void **>(&mapping),
                              mapping_segment->data_size()) !=
        mapping_segment->data_size()) {
      LOG_ERROR("Failed to read data (%zu bytes) from mapping segment",
                mapping_segment->data_size());
      return nullptr;
    }
    mapping_.clear();
    mapping_.reserve(mapping_count);
    std::copy(mapping, mapping + mapping_count, std::back_inserter(mapping_));
  }
  return IndexProvider::Pointer(new (std::nothrow)
                                    FlatSearcherProvider<BATCH_SIZE>(this));
}

template <size_t BATCH_SIZE>
int FlatSearcher<BATCH_SIZE>::load(IndexStorage::Pointer cntr,
                                   IndexMetric::Pointer measure) {
  ailego::ElapsedTime stamp;
  if (!cntr) {
    return IndexError_InvalidArgument;
  }

  int error_code = IndexHelper::DeserializeFromStorage(cntr.get(), &meta_);
  if (error_code != 0) {
    LOG_ERROR(
        "Failed to deserialize index meta from container %s, error=%d, %s",
        cntr->name().c_str(), error_code, IndexError::What(error_code));
    return error_code;
  }

  if (!measure) {
    error_code = InitializeMetric(meta_, &measure_);
    if (error_code != 0) {
      LOG_ERROR("Failed to initialize index measure %s, error=%d, %s",
                meta_.metric_name().c_str(), error_code,
                IndexError::What(error_code));
      return error_code;
    }
    if (measure_->query_metric()) {
      measure_ = measure_->query_metric();
    }
  } else {
    if (!measure->is_matched(meta_)) {
      LOG_ERROR(
          "The index measure is unmatched with index meta from container.");
      return IndexError_Mismatch;
    }
    measure_ = std::move(measure);
  }

  column_major_order_ = (meta_.major_order() == IndexMeta::MO_COLUMN);
  distance_matrix_.initialize(*measure_);

  if (column_major_order_) {
    if (!distance_matrix_.is_valid()) {
      LOG_ERROR("Lack of distance functions to support column index.");
      return IndexError_Unsupported;
    }
  } else {
    if (!distance_matrix_.is_valid(1, 1)) {
      LOG_ERROR("Lack of distance functions to support row index.");
      return IndexError_Unsupported;
    }
  }

  auto keys_segment = cntr->get(FLAT_SEGMENT_KEYS_SEG_ID);
  if (!keys_segment) {
    LOG_ERROR("Failed to fetch segment %s", FLAT_SEGMENT_KEYS_SEG_ID.c_str());
    return IndexError_NoExist;
  }
  features_segment_ = cntr->get(FLAT_SEGMENT_FEATURES_SEG_ID);
  if (!features_segment_) {
    LOG_ERROR("Failed to fetch segment %s", FLAT_SEGMENT_KEYS_SEG_ID.c_str());
    return IndexError_NoExist;
  }

  if (keys_segment->data_size() % sizeof(uint64_t) != 0) {
    LOG_ERROR("Invalid data size %zu of keys segment",
              keys_segment->data_size());
    return IndexError_InvalidLength;
  }

  size_t keys_count = keys_segment->data_size() / sizeof(uint64_t);
  if (keys_count * meta_.element_size() != features_segment_->data_size()) {
    LOG_ERROR("Invalid data size %zd of features segment",
              features_segment_->data_size());
    return IndexError_Mismatch;
  }

  if (keys_segment->read(0, reinterpret_cast<const void **>(&keys_),
                         keys_segment->data_size()) !=
      keys_segment->data_size()) {
    LOG_ERROR("Failed to read data (%zu bytes) from keys segment",
              keys_segment->data_size());
    return IndexError_ReadData;
  }

  for (size_t i = 0; i < keys_count; i++) {
    key_id_mapping_[keys_[i]] = i;
  }

  container_ = cntr;
  magic_ = IndexContext::GenerateMagic();
  stats_.set_loaded_count(keys_count);
  stats_.set_loaded_costtime(stamp.milli_seconds());
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcher<BATCH_SIZE>::search_impl(const void *query,
                                          const IndexQueryMeta &qmeta,
                                          Context::Pointer &context) const {
  ailego_assert(query && !!context);
  ailego_assert(measure_->is_matched(meta_, qmeta));

  FlatSearcherContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatSearcherContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force searcher context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }
  if (bf_context->group_by_search()) {
    return bf_context->group_by_search_impl(query, qmeta, 1);
  } else {
    return (column_major_order_ ? bf_context->search_column(query, qmeta)
                                : bf_context->search_row(query, qmeta));
  }
}

template <size_t BATCH_SIZE>
int FlatSearcher<BATCH_SIZE>::search_impl(const void *query,
                                          const IndexQueryMeta &qmeta,
                                          uint32_t count,
                                          Context::Pointer &context) const {
  ailego_assert(query && count && !!context);
  ailego_assert(measure_->is_matched(meta_, qmeta));

  FlatSearcherContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatSearcherContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force searcher context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }

  if (bf_context->group_by_search()) {
    return bf_context->group_by_search_impl(query, qmeta, count);
  } else {
    return (column_major_order_ ? bf_context->search_column(query, qmeta, count)
                                : bf_context->search_row(query, qmeta, count));
  }
}

template <size_t BATCH_SIZE>
int FlatSearcher<BATCH_SIZE>::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  ailego_assert(query && count && !!context);
  ailego_assert(measure_->is_matched(meta_, qmeta));

  if (ailego_unlikely(p_keys.size() != count)) {
    LOG_ERROR("The size of p_keys is not equal to count");
    return IndexError_InvalidArgument;
  }

  FlatSearcherContext<BATCH_SIZE> *bf_context =
      dynamic_cast<FlatSearcherContext<BATCH_SIZE> *>(context.get());
  if (!bf_context) {
    LOG_ERROR("Invalid brute-force searcher context");
    return IndexError_InvalidArgument;
  }

  if (bf_context->magic() != magic_) {
    bf_context->reset(this);
  }

  return bf_context->search_bf_by_p_keys_impl(query, p_keys, qmeta, count);
}

template <size_t BATCH_SIZE>
IndexSearcher::Context::Pointer FlatSearcher<BATCH_SIZE>::create_context(
    void) const {
  return IndexSearcher::Context::Pointer(
      new FlatSearcherContext<BATCH_SIZE>(this));
}

INDEX_FACTORY_REGISTER_SEARCHER_ALIAS(LinearSearcher, FlatSearcher<32>);
INDEX_FACTORY_REGISTER_SEARCHER_ALIAS(FlatSearcher, FlatSearcher<32>);
INDEX_FACTORY_REGISTER_SEARCHER_ALIAS(FlatSearcher16, FlatSearcher<16>);
INDEX_FACTORY_REGISTER_SEARCHER_ALIAS(FlatSearcher32, FlatSearcher<32>);
}  // namespace core
}  // namespace zvec
