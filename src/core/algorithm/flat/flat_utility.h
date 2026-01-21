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

#include <mutex>
#include <ailego/utility/matrix_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_metric.h>

namespace zvec {
namespace core {

//! The default size of reading a block
static constexpr uint32_t FLAT_READ_BLOCK_SIZE = 4 * 1024 * 1024;
static const std::string FLAT_LINEAR_META_SEG_ID = "linear.linear_meta";
static const std::string FLAT_LINEAR_LIST_HEAD_SEG_ID =
    "linear.linear_list_head";

// reducer params
static const std::string PARAM_FLAT_REDUCER_WORKING_PATH(
    "proxima.linear.reducer.working_path");
static const std::string PARAM_FLAT_REDUCER_NUM_OF_ADD_THREADS(
    "proxima.linear.reducer.num_of_add_threads");
static const std::string PARAM_FLAT_REDUCER_INDEX_NAME(
    "proxima.linear.reducer.index_name");

// index params
static const std::string PARAM_FLAT_SEGMENT_KEYS("linear.keys");
static const std::string PARAM_FLAT_SEGMENT_FEATURES("linear.features");
static const std::string PARAM_FLAT_SEGMENT_MAPPING("linear.mapping");

static const std::string FLAT_BUILDER_COLUMN_MAJOR_ORDER(
    "proxima.linear.builder.column_major_order");
static const std::string FLAT_SEARCHER_BATCH_SIZE(
    "proxima.linear.searcher.batch_size");
static const std::string FLAT_SEARCHER_READ_BLOCK_SIZE(
    "proxima.linear.searcher.read_block_size");
static const std::string FLAT_SEARCHER_USE_ID_MAP(
    "proxima.linear.searcher.use_id_map");

//! Determines if a number is equal to two to the power of n.
template <size_t K>
struct IsEqualPowerofTwo
    : std::integral_constant<bool, K != 0 && (K ^ (K - 1)) == (K | (K - 1))> {};

//! Transpose a block
template <size_t M>
static inline void ReverseTranspose(size_t align_size, const void *src,
                                    size_t dim, void *dst) {
  switch (align_size) {
    case 2:
      ailego::MatrixHelper::ReverseTranspose<uint16_t, M>(src, dim, dst);
      break;
    case 4:
      ailego::MatrixHelper::ReverseTranspose<uint32_t, M>(src, dim, dst);
      break;
    case 8:
      ailego::MatrixHelper::ReverseTranspose<uint64_t, M>(src, dim, dst);
      break;
  }
}

static inline void ReverseTranspose(size_t align_size, const void *src,
                                    size_t m, size_t dim, void *dst) {
  switch (align_size) {
    case 2:
      ailego::MatrixHelper::ReverseTranspose<uint16_t>(src, m, dim, dst);
      break;
    case 4:
      ailego::MatrixHelper::ReverseTranspose<uint32_t>(src, m, dim, dst);
      break;
    case 8:
      ailego::MatrixHelper::ReverseTranspose<uint64_t>(src, m, dim, dst);
      break;
  }
}

template <typename T>
static inline void TransposeOne(const void *src, size_t M, size_t N,
                                void *dst) {
  for (size_t i = 0; i < N; ++i) {
    reinterpret_cast<T *>(dst)[i] = reinterpret_cast<const T *>(src)[i * M];
  }
}

static inline void Transpose(size_t align_size, const void *src, size_t m,
                             size_t dim, void *dst) {
  switch (align_size) {
    case 2:
      ailego::MatrixHelper::Transpose<uint16_t>(src, m, dim, dst);
      break;
    case 4:
      ailego::MatrixHelper::Transpose<uint32_t>(src, m, dim, dst);
      break;
    case 8:
      ailego::MatrixHelper::Transpose<uint64_t>(src, m, dim, dst);
      break;
  }
}

//! Transpose queries
template <size_t K>
void TransposeQueries(const void *query, const IndexQueryMeta &qmeta,
                      size_t query_count, std::string *out) {
  if (K <= 1) {
    ailego_assert(query_count == 1);
    (void)query_count;
    out->append(reinterpret_cast<const char *>(query) + out->size(),
                qmeta.element_size());
  } else {
    ailego_assert_with(IsEqualPowerofTwo<K>::value,
                       "K must be equal to two to the power of n.");

    size_t query_batch_count = query_count / K;
    size_t query_offset = out->size();
    out->resize(query_offset + query_batch_count * K * qmeta.element_size());

    switch (IndexMeta::AlignSizeof(qmeta.data_type())) {
      case 2:
        for (size_t i = 0; i != query_batch_count; ++i) {
          ailego::MatrixHelper::Transpose<uint16_t, K>(
              (const char *)query + query_offset,
              qmeta.element_size() / sizeof(uint16_t), &((*out)[query_offset]));
          query_offset += qmeta.element_size() * K;
        }
        break;

      case 4:
        for (size_t i = 0; i != query_batch_count; ++i) {
          ailego::MatrixHelper::Transpose<uint32_t, K>(
              (const char *)query + query_offset,
              qmeta.element_size() / sizeof(uint32_t), &((*out)[query_offset]));

          query_offset += qmeta.element_size() * K;
        }
        break;

      case 8:
        for (size_t i = 0; i != query_batch_count; ++i) {
          ailego::MatrixHelper::Transpose<uint64_t, K>(
              (const char *)query + query_offset,
              qmeta.element_size() / sizeof(uint64_t), &((*out)[query_offset]));
          query_offset += qmeta.element_size() * K;
        }
        break;

      default:
        ailego_check_with(0, "BAD CASE");
    }
    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      TransposeQueries<(K >> 1)>(query, qmeta, query_left_count, out);
    }
  }
}

//! Create and initialize measure
static inline int InitializeMetric(const IndexMeta &mt,
                                   IndexMetric::Pointer *out) {
  IndexMetric::Pointer measure = IndexFactory::CreateMetric(mt.metric_name());
  if (!measure) {
    return IndexError_NoExist;
  }

  int error_code = measure->init(mt, mt.metric_params());
  if (error_code != 0) {
    return error_code;
  }
  *out = measure;
  return 0;
}

//! Verify measure
static inline bool VerifyMetric(const IndexMeta &meta) {
  IndexMetric::Pointer measure = IndexFactory::CreateMetric(meta.metric_name());
  if (!measure) {
    return false;
  }
  int error_code = measure->init(meta, meta.metric_params());
  if (error_code != 0) {
    return false;
  }
  return true;
}

}  // namespace core
}  // namespace zvec
