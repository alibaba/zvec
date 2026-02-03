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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include "db/common/constants.h"
#include "db/common/typedef.h"
#include "db/index/common/meta.h"
#include "db/index/segment/segment.h"
#include "db/index/storage/store_helper.h"
#include "zvec/db/collection.h"
#include "zvec/db/doc.h"
#include "zvec/db/schema.h"
#include "zvec/db/type.h"

namespace zvec::test {

template <typename T>
bool vectors_equal_when_sorted(std::vector<T> a, std::vector<T> b) {
  if (a.size() != b.size()) {
    return false;
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

template <typename T>
double inner_produce_double(const std::vector<T> &vec1,
                            const std::vector<T> &vec2) {
  double result = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    result += vec1[i] * vec2[i];
  }
  return result;
}


template <typename T>
inline float cosine_distance_dense(const std::vector<T> &vec1,
                                   const std::vector<T> &vec2) {
  const auto dot = inner_produce_double(vec1, vec2);
  const auto norm1 = std::sqrt((inner_produce_double(vec1, vec1)));
  const auto norm2 = std::sqrt((inner_produce_double(vec2, vec2)));

  if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
  return 1.0f - dot / (norm1 * norm2);
}

template <typename T>
inline float dp_distance_dense(const std::vector<T> &vec1,
                               const std::vector<T> &vec2) {
  double result = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    result += vec1[i] * vec2[i];
  }
  return result;
}

template <typename T>
inline float euclidean_distance_dense(const std::vector<T> &vec1,
                                      const std::vector<T> &vec2) {
  double sum = 0.0f;
  for (size_t i = 0; i < vec1.size(); ++i) {
    const float diff =
        static_cast<float>(vec1[i]) - static_cast<float>(vec2[i]);
    sum += diff * diff;
  }
  return sum;
}

template <typename T>
inline float distance_dense(const std::vector<T> &vec1,
                            const std::vector<T> &vec2, MetricType metric) {
  switch (metric) {
    case MetricType::COSINE:
      return cosine_distance_dense(vec1, vec2);
    case MetricType::L2:
      return euclidean_distance_dense(vec1, vec2);
    case MetricType::IP:
      return dp_distance_dense(vec1, vec2);
    default:
      throw std::invalid_argument("Unsupported metric for FP32");
  }
}

using SparseVecFP32 = std::pair<std::vector<uint32_t>, std::vector<float>>;
using SparseVecFP16 = std::pair<std::vector<uint32_t>, std::vector<float16_t>>;
using SparseVec = SparseVecFP32;

template <typename T>
inline float sparse_dot_product(const std::vector<uint32_t> &idx1,
                                const std::vector<T> &val1,
                                const std::vector<uint32_t> &idx2,
                                const std::vector<T> &val2) {
  double dot = 0.0f;
  size_t i = 0, j = 0;

  while (i < idx1.size() && j < idx2.size()) {
    if (idx1[i] == idx2[j]) {
      dot += static_cast<float>(val1[i]) * static_cast<float>(val2[j]);
      ++i;
      ++j;
    } else if (idx1[i] < idx2[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return dot;
}

inline float distance_sparse(const SparseVecFP32 &vec1,
                             const SparseVecFP32 &vec2) {
  return sparse_dot_product(vec1.first, vec1.second, vec2.first, vec2.second);
}

inline float distance_sparse(const SparseVecFP16 &vec1,
                             const SparseVecFP16 &vec2) {
  return sparse_dot_product(vec1.first, vec1.second, vec2.first, vec2.second);
}


class TestHelper {
 public:
  static CollectionSchema::Ptr CreateTempSchema();

  static CollectionSchema::Ptr CreateScalarSchema();

  static CollectionSchema::Ptr CreateNormalSchema(
      bool nullable = false, std::string name = "demo",
      IndexParams::Ptr scalar_index_params = nullptr,
      IndexParams::Ptr vector_index_params = nullptr,
      uint64_t max_doc_count = MAX_DOC_COUNT_PER_SEGMENT);

  static CollectionSchema::Ptr CreateSchemaWithScalarIndex(
      bool nullable = false, bool enable_optimize = false,
      std::string name = "demo");

  static CollectionSchema::Ptr CreateSchemaWithVectorIndex(
      bool nullable = false, std::string name = "demo",
      IndexParams::Ptr vector_index_params = nullptr);

  static CollectionSchema::Ptr CreateSchemaWithMaxDocCount(uint64_t doc_count);

  static std::string MakePK(const uint64_t doc_id);

  static uint64_t ExtractDocId(const std::string &pk);

  static Doc CreateDoc(const uint64_t doc_id, const CollectionSchema &schema,
                       std::string pk = "");

  static Doc CreateDocNull(const uint64_t doc_id,
                           const CollectionSchema &schema, std::string pk = "");

  static Status SegmentInsertDoc(const Segment::Ptr &segment,
                                 const CollectionSchema &schema,
                                 const uint64_t start_doc_id,
                                 const uint64_t end_doc_id,
                                 bool nullable = false, bool upsert = false,
                                 bool batch = false);

  static Status CollectionInsertDoc(const Collection::Ptr &collection,
                                    const uint64_t start_doc_id,
                                    const uint64_t end_doc_id,
                                    bool nullable = false, bool upsert = false,
                                    bool batch = false);

  static Status CollectionUpsertDoc(const Collection::Ptr &collection,
                                    const uint64_t start_doc_id,
                                    const uint64_t end_doc_id,
                                    bool nullable = false, bool batch = false);

  static Segment::Ptr CreateSegmentWithDoc(
      const std::string &col_path, const CollectionSchema &schema,
      SegmentID segment_id, uint64_t min_doc_id, const IDMap::Ptr &id_map,
      const DeleteStore::Ptr &delete_store,
      const VersionManager::Ptr &version_manager, const SegmentOptions &options,
      uint64_t start_doc_id, uint32_t doc_count, bool nullable = false,
      bool upsert = false);

  static Collection::Ptr CreateCollectionWithDoc(
      const std::string &path, const CollectionSchema &schema,
      const CollectionOptions &options, uint64_t start_doc_id,
      uint32_t doc_count, bool nullable = false, bool upsert = false);


  static arrow::Status WriteTestFile(const std::string &filepath,
                                     FileFormat format,
                                     uint32_t start_doc_id = 0,
                                     uint32_t end_doc_id = 10,
                                     uint32_t batch_size = 3);
};


}  // namespace zvec::test