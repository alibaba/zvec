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

// #include "db/doc.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include <cassert>
#include <cstdint>
#include <gtest/gtest.h>
#include "db/index/column/vector_column/vector_column_params.h"
#include "zvec/ailego/utility/float_helper.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec;


std::string print_dense_vector(const void *vector, size_t dim,
                               DataType data_type) {
  std::stringstream ss;
  switch (data_type) {
    case DataType::VECTOR_FP32: {
      const float *data = reinterpret_cast<const float *>(vector);

      for (size_t i = 0; i < dim; ++i) {
        ss << data[i] << " ";
      }
    } break;
    case DataType::VECTOR_FP16: {
      const zvec::float16_t *data =
          reinterpret_cast<const zvec::float16_t *>(vector);
      for (size_t i = 0; i < dim; ++i) {
        ss << data[i] << " ";
      }
    } break;
    default:
      LOG_ERROR("Unsupported data type: %d", static_cast<int>(data_type));
      break;
  }
  return ss.str();
}

TEST(VectorColumnIndexerTest, General) {
  auto func = [&](const IndexParams::Ptr index_params,
                  const QueryParams::Ptr query_params) {
    const std::string index_file_path = "test_indexer.index";
    constexpr idx_t kDocId = 2345;

    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
    system(cmd_buf);

    // 1. create indexer
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 4, false, index_params));
    ASSERT_TRUE(indexer);

    // 2. open
    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    {
      // can't use `DenseVector{std::vector<float>{1.0f, 2.0f, 3.0f}.data()}}`,
      // which will be destroyed immediately
      auto vector = std::vector<float>{1.0f, 2.0f, 3.0f, 0};

      // 3. add data
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{vector.data()}};
      ASSERT_TRUE(indexer->Insert(data, kDocId).ok());
    }

    {
      auto vector = std::vector<float>{1.0f, 2000.0f, 3.0f, 0};
      // 1 * 1 + 2 * 2000 + 3 * 3 = 12006
      ASSERT_TRUE(indexer
                      ->Insert(
                          vector_column_params::VectorData{
                              vector_column_params::DenseVector{vector.data()}},
                          kDocId + 10)
                      .ok());
    }

    {  // add_with_id() won't check duplication, overwrite last one
      auto vector = std::vector<float>{1.0f, 0, 3.0f, 0};
      // 1 * 1 + 2 * 0 + 3 * 3 = 10
      ASSERT_TRUE(indexer
                      ->Insert(
                          vector_column_params::VectorData{
                              vector_column_params::DenseVector{vector.data()}},
                          kDocId + 10)
                      .ok());
    }

    // 5. fetch
    auto fetched_data = indexer->Fetch(kDocId);
    ASSERT_TRUE(fetched_data);
    const float *dense_vector = reinterpret_cast<const float *>(
        std::get<vector_column_params::DenseVectorBuffer>(
            fetched_data->vector_buffer)
            .data.data());
    ASSERT_NEAR(dense_vector[0], 1.0, 0.1);
    ASSERT_NEAR(dense_vector[1], 2.0, 0.1);
    ASSERT_NEAR(dense_vector[2], 3.0, 0.1);
    ASSERT_NEAR(dense_vector[3], 0, 0.1);

    // 4. search
    // https://stackoverflow.com/questions/69009389/how-to-get-away-with-using-designated-initializers-in-c17-or-why-is-it-seemi
    auto query_vector = std::vector<float>{1.0f, 2.0f, 3.0f, 0};
    auto query = vector_column_params::VectorData{
        vector_column_params::DenseVector{.data = query_vector.data()}};
    auto indexer_query_params =
        vector_column_params::QueryParams{.topk = 10,
                                          .filter = nullptr,
                                          .fetch_vector = true,
                                          .query_params = query_params};
    auto results = indexer->Search(query, indexer_query_params);
    ASSERT_TRUE(results.has_value());

    auto vector_results =
        dynamic_cast<VectorIndexResults *>(results.value().get());
    ASSERT_TRUE(vector_results);
    ASSERT_EQ(vector_results->count(), 2);

    {
      int count = 0;
      auto iter = vector_results->create_iterator();
      while (iter->valid()) {
        count++;
        iter->next();
      }
      ASSERT_EQ(count, 2);
    }

    {  // top1 doc
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId);
      if (iter->score() > 14) {
        ASSERT_NEAR(iter->score(), 14.0, 0.1);
      }

      // top2
      iter->next();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId + 10);
      ASSERT_NEAR(iter->score(), 10.0, 0.1);
    }

    auto vector_index_params =
        reinterpret_cast<VectorIndexParams *>(index_params.get());
    if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
      ASSERT_TRUE(vector_results->docs().size() == 2);
      ASSERT_TRUE(vector_results->reverted_vector_list().size() == 2);
      ASSERT_TRUE(vector_results->reverted_sparse_values_list().empty());
    }

    indexer->Close();

    system(cmd_buf);
  };

  func(std::make_shared<FlatIndexParams>(MetricType::IP),
       std::make_shared<QueryParams>(IndexType::FLAT));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       std::make_shared<HnswQueryParams>(300));
  func(std::make_shared<IVFIndexParams>(MetricType::IP),
       std::make_shared<IVFQueryParams>(10));

  func(std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16),
       std::make_shared<QueryParams>(IndexType::FLAT));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                         QuantizeType::FP16),
       std::make_shared<HnswQueryParams>(300));
  func(std::make_shared<IVFIndexParams>(MetricType::IP, 1024, 10, false,
                                        QuantizeType::FP16),
       std::make_shared<IVFQueryParams>(10));

  func(std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::INT8),
       std::make_shared<QueryParams>(IndexType::FLAT));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                         QuantizeType::INT8),
       std::make_shared<HnswQueryParams>(300));
  func(std::make_shared<IVFIndexParams>(MetricType::IP, 1024, 10, false,
                                        QuantizeType::INT8),
       std::make_shared<IVFQueryParams>(10));

  func(std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::INT4),
       std::make_shared<QueryParams>(IndexType::FLAT));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                         QuantizeType::INT4),
       std::make_shared<HnswQueryParams>(300));
}

TEST(VectorColumnIndexerTest, DenseDataTypeFP16) {
  auto func = [&](const IndexParams::Ptr index_params,
                  const QueryParams::Ptr query_params) {
    const std::string index_file_path = "test_indexer.index";
    constexpr idx_t kDocId = 2345;
    constexpr int dimension = 4;

    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
    system(cmd_buf);

    // 1. create indexer
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path, FieldSchema("test", DataType::VECTOR_FP16, dimension,
                                     false, index_params));
    ASSERT_TRUE(indexer);

    // 2. open
    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    {
      // can't use `DenseVector{std::vector<float>{1.0f, 2.0f, 3.0f}.data()}}`,
      // which will be destroyed immediately
      auto origin_vector = std::vector<float>{1.0f, 2.0f, 3.0f, 0};
      std::vector<uint16_t> buffer(dimension);
      ailego::FloatHelper::ToFP16((float *)origin_vector.data(), dimension,
                                  buffer.data());
      auto vector = buffer;

      // 3. add data
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{vector.data()}};
      ASSERT_TRUE(indexer->Insert(data, kDocId).ok());
    }

    {
      auto origin_vector = std::vector<float>{1.0f, 2000.0f, 3.0f, 0};
      std::vector<uint16_t> buffer(dimension);
      ailego::FloatHelper::ToFP16((float *)origin_vector.data(), dimension,
                                  buffer.data());
      auto vector = buffer;
      // 1 * 1 + 2 * 2000 + 3 * 3 = 12006
      ASSERT_TRUE(indexer
                      ->Insert(
                          vector_column_params::VectorData{
                              vector_column_params::DenseVector{vector.data()}},
                          kDocId + 10)
                      .ok());
    }

    {  // add_with_id() won't check duplication, overwrite last one
      auto origin_vector = std::vector<float>{1.0f, 0, 3.0f, 0};
      std::vector<uint16_t> buffer(dimension);
      ailego::FloatHelper::ToFP16((float *)origin_vector.data(), dimension,
                                  buffer.data());
      auto vector = buffer;
      // 1 * 1 + 2 * 0 + 3 * 3 = 10
      ASSERT_TRUE(indexer
                      ->Insert(
                          vector_column_params::VectorData{
                              vector_column_params::DenseVector{vector.data()}},
                          kDocId + 10)
                      .ok());
    }
    // 5. fetch
    {
      auto fetched_data = indexer->Fetch(kDocId);
      ASSERT_TRUE(fetched_data);
      const uint16_t *dense_vector = reinterpret_cast<const uint16_t *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[0]), 1.0, 0.1);
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[1]), 2.0, 0.1);
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[2]), 3.0, 0.1);
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[3]), 0, 0.1);
    }
    {
      auto fetched_data = indexer->Fetch(kDocId + 10);
      ASSERT_TRUE(fetched_data);
      const uint16_t *dense_vector = reinterpret_cast<const uint16_t *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[0]), 1.0, 0.1);
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[1]), 0, 0.1);
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[2]), 3.0, 0.1);
      ASSERT_NEAR(ailego::FloatHelper::ToFP32(dense_vector[3]), 0, 0.1);
    }

    // 4. search
    // https://stackoverflow.com/questions/69009389/how-to-get-away-with-using-designated-initializers-in-c17-or-why-is-it-seemi
    auto origin_query_vector = std::vector<float>{1.0f, 2.0f, 3.0f, 0};
    std::vector<uint16_t> buffer(dimension);
    ailego::FloatHelper::ToFP16((float *)origin_query_vector.data(), dimension,
                                buffer.data());
    auto query_vector = buffer;
    auto query = vector_column_params::VectorData{
        vector_column_params::DenseVector{.data = query_vector.data()}};
    auto indexer_query_params =
        vector_column_params::QueryParams{.topk = 10,
                                          .filter = nullptr,
                                          .fetch_vector = true,
                                          .query_params = query_params};
    auto results = indexer->Search(query, indexer_query_params);
    ASSERT_TRUE(results.has_value());

    auto vector_results =
        dynamic_cast<VectorIndexResults *>(results.value().get());
    ASSERT_TRUE(vector_results);
    ASSERT_EQ(vector_results->count(), 2);

    {
      int count = 0;
      auto iter = vector_results->create_iterator();
      while (iter->valid()) {
        count++;
        iter->next();
      }
      ASSERT_EQ(count, 2);
    }

    {  // top1 doc
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId);
      if (iter->score() > 14) {
        ASSERT_NEAR(iter->score(), 14.0, 0.1);
      }

      // top2
      iter->next();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId + 10);
      ASSERT_NEAR(iter->score(), 10.0, 0.1);
    }

    auto vector_index_params =
        reinterpret_cast<VectorIndexParams *>(index_params.get());
    if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
      ASSERT_TRUE(vector_results->docs().size() == 2);
      ASSERT_TRUE(vector_results->reverted_vector_list().size() == 2);
      ASSERT_TRUE(vector_results->reverted_sparse_values_list().empty());
    }

    indexer->Close();

    system(cmd_buf);
  };

  func(std::make_shared<FlatIndexParams>(MetricType::IP),
       std::make_shared<QueryParams>(IndexType::FLAT));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       std::make_shared<HnswQueryParams>(300));
}

TEST(VectorColumnIndexerTest, DenseDataTypeINT8) {
  auto func = [&](const IndexParams::Ptr index_params,
                  const QueryParams::Ptr query_params) {
    const std::string index_file_path = "test_indexer.index";
    constexpr idx_t kDocId = 2345;
    constexpr int dimension = 4;

    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
    system(cmd_buf);

    // 1. create indexer
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path, FieldSchema("test", DataType::VECTOR_INT8, dimension,
                                     false, index_params));
    ASSERT_TRUE(indexer);

    // 2. open
    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    {
      // can't use `DenseVector{std::vector<float>{1.0f, 2.0f, 3.0f}.data()}}`,
      // which will be destroyed immediately
      auto vector = std::vector<uint8_t>{1, 2, 3, 0};

      // 3. add data
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{vector.data()}};
      ASSERT_TRUE(indexer->Insert(data, kDocId).ok());
    }

    {
      auto vector = std::vector<uint8_t>{1, 200, 3, 0};
      // 1 * 1 + 2 * 2000 + 3 * 3 = 12006
      ASSERT_TRUE(indexer
                      ->Insert(
                          vector_column_params::VectorData{
                              vector_column_params::DenseVector{vector.data()}},
                          kDocId + 10)
                      .ok());
    }

    {  // add_with_id() won't check duplication, overwrite last one
      auto vector = std::vector<uint8_t>{1, 0, 3, 0};
      // 1 * 1 + 2 * 0 + 3 * 3 = 10
      ASSERT_TRUE(indexer
                      ->Insert(
                          vector_column_params::VectorData{
                              vector_column_params::DenseVector{vector.data()}},
                          kDocId + 10)
                      .ok());
    }
    // 5. fetch
    {
      auto fetched_data = indexer->Fetch(kDocId);
      ASSERT_TRUE(fetched_data);
      const uint8_t *dense_vector = reinterpret_cast<const uint8_t *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(dense_vector[0], 1.0, 0.1);
      ASSERT_NEAR(dense_vector[1], 2.0, 0.1);
      ASSERT_NEAR(dense_vector[2], 3.0, 0.1);
      ASSERT_NEAR(dense_vector[3], 0, 0.1);
    }
    {
      auto fetched_data = indexer->Fetch(kDocId + 10);
      ASSERT_TRUE(fetched_data);
      const uint8_t *dense_vector = reinterpret_cast<const uint8_t *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(dense_vector[0], 1.0, 0.1);
      ASSERT_NEAR(dense_vector[1], 0, 0.1);
      ASSERT_NEAR(dense_vector[2], 3.0, 0.1);
      ASSERT_NEAR(dense_vector[3], 0, 0.1);
    }

    // 4. search
    // https://stackoverflow.com/questions/69009389/how-to-get-away-with-using-designated-initializers-in-c17-or-why-is-it-seemi
    auto query_vector = std::vector<uint8_t>{1, 2, 3, 0};
    auto query = vector_column_params::VectorData{
        vector_column_params::DenseVector{.data = query_vector.data()}};
    auto indexer_query_params =
        vector_column_params::QueryParams{.topk = 10,
                                          .filter = nullptr,
                                          .fetch_vector = true,
                                          .query_params = query_params};
    auto results = indexer->Search(query, indexer_query_params);
    ASSERT_TRUE(results.has_value());

    auto vector_results =
        dynamic_cast<VectorIndexResults *>(results.value().get());
    ASSERT_TRUE(vector_results);
    ASSERT_EQ(vector_results->count(), 2);

    {
      int count = 0;
      auto iter = vector_results->create_iterator();
      while (iter->valid()) {
        count++;
        iter->next();
      }
      ASSERT_EQ(count, 2);
    }

    {  // top1 doc
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId);
      if (iter->score() > 14) {
        ASSERT_NEAR(iter->score(), 14.0, 0.1);
      }

      // top2
      iter->next();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId + 10);
      ASSERT_NEAR(iter->score(), 10.0, 0.1);
    }

    auto vector_index_params =
        reinterpret_cast<VectorIndexParams *>(index_params.get());
    if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
      ASSERT_TRUE(vector_results->docs().size() == 2);
      ASSERT_TRUE(vector_results->reverted_vector_list().size() == 2);
      ASSERT_TRUE(vector_results->reverted_sparse_values_list().empty());
    }

    indexer->Close();

    system(cmd_buf);
  };

  func(std::make_shared<FlatIndexParams>(MetricType::IP),
       std::make_shared<QueryParams>(IndexType::FLAT));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       std::make_shared<HnswQueryParams>(300));
}


TEST(VectorColumnIndexerTest, SparseGeneral) {
  constexpr uint32_t kSparseCount = 3;
  auto func = [&](const IndexParams::Ptr index_params) {
    const std::string index_file_path = "test_indexer.index";
    constexpr idx_t kDocId = 2345;

    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
    system(cmd_buf);

    // create indexer
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::SPARSE_VECTOR_FP32, false, index_params));
    ASSERT_TRUE(indexer);

    // open
    if (auto ret = indexer->Open(vector_column_params::ReadOptions{true, true});
        !ret.ok()) {
      std::cout << ret.message() << std::endl;
      ASSERT_TRUE(false);
    }

    std::vector<uint32_t> indices(kSparseCount);
    std::vector<float> values(kSparseCount);
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      indices[i] = i;
      values[i] = i;
    }
    vector_column_params::SparseVector vector{kSparseCount, indices.data(),
                                              values.data()};
    ASSERT_TRUE(
        indexer->Insert(vector_column_params::VectorData{vector}, kDocId).ok());

    // fetch
    auto fetched_data = indexer->Fetch(kDocId);
    ASSERT_TRUE(fetched_data.has_value());
    auto fetched_sparse_vector =
        std::get<vector_column_params::SparseVectorBuffer>(
            fetched_data.value().vector_buffer);
    auto fetched_indices = reinterpret_cast<const uint32_t *>(
        fetched_sparse_vector.indices.data());
    auto fetched_values =
        reinterpret_cast<const float *>(fetched_sparse_vector.values.data());
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      ASSERT_EQ(i, fetched_indices[i]);
      ASSERT_FLOAT_EQ(i, fetched_values[i]);
    }

    // search
    auto query =
        vector_column_params::VectorData{vector_column_params::SparseVector{
            kSparseCount, indices.data(), values.data()}};
    auto query_params = vector_column_params::QueryParams{
        .topk = 10, .filter = nullptr, .fetch_vector = true};
    auto results = indexer->Search(query, query_params);
    ASSERT_TRUE(results.has_value());

    auto vector_results =
        dynamic_cast<VectorIndexResults *>(results.value().get());
    ASSERT_TRUE(vector_results);
    ASSERT_EQ(vector_results->count(), 1);

    {
      int count = 0;
      auto iter = vector_results->create_iterator();
      while (iter->valid()) {
        count++;
        iter->next();
      }
      ASSERT_EQ(count, 1);
    }

    {
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId);
      ASSERT_FLOAT_EQ(iter->score(), 5.0);

      auto vector = iter->vector();
      auto sparse_vector =
          std::get<vector_column_params::SparseVector>(vector.vector);
      auto indices = reinterpret_cast<const uint32_t *>(sparse_vector.indices);
      auto values = reinterpret_cast<const float *>(sparse_vector.values);
      ASSERT_EQ(sparse_vector.count, kSparseCount);
      for (uint32_t i = 0; i < kSparseCount; ++i) {
        ASSERT_EQ(i, indices[i]);
        ASSERT_FLOAT_EQ(i, values[i]);
      }
      auto vector_index_params =
          reinterpret_cast<VectorIndexParams *>(index_params.get());
      if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
        ASSERT_TRUE(vector_results->docs().size() == 1);
        ASSERT_TRUE(vector_results->reverted_sparse_values_list().size() == 1);
        ASSERT_TRUE(vector_results->reverted_vector_list().empty());
      }
    }

    indexer->Close();

    system(cmd_buf);
  };

  func(std::make_shared<FlatIndexParams>(MetricType::IP));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100));
  func(std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                         QuantizeType::FP16));
}

TEST(VectorColumnIndexerTest, SparseDataTypeFP16) {
  constexpr uint32_t kSparseCount = 3;
  auto func = [&](const IndexParams::Ptr index_params) {
    const std::string index_file_path = "test_indexer.index";
    constexpr idx_t kDocId = 2345;

    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
    system(cmd_buf);

    // create indexer
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::SPARSE_VECTOR_FP16, false, index_params));
    ASSERT_TRUE(indexer);

    // open
    if (auto ret = indexer->Open(vector_column_params::ReadOptions{true, true});
        !ret.ok()) {
      std::cout << ret.message() << std::endl;
      ASSERT_TRUE(false);
    }

    std::vector<uint32_t> indices(kSparseCount);
    std::vector<float> origin_values(kSparseCount);
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      indices[i] = i;
      origin_values[i] = i;
    }
    std::vector<uint16_t> buffer1(kSparseCount);
    ailego::FloatHelper::ToFP16((float *)origin_values.data(), kSparseCount,
                                buffer1.data());
    auto values = buffer1;
    vector_column_params::SparseVector vector{kSparseCount, indices.data(),
                                              values.data()};
    ASSERT_TRUE(
        indexer->Insert(vector_column_params::VectorData{vector}, kDocId).ok());

    // fetch
    auto fetched_data = indexer->Fetch(kDocId);
    ASSERT_TRUE(fetched_data.has_value());
    auto fetched_sparse_vector =
        std::get<vector_column_params::SparseVectorBuffer>(
            fetched_data.value().vector_buffer);
    auto fetched_indices = reinterpret_cast<const uint32_t *>(
        fetched_sparse_vector.indices.data());
    auto fetched_values =
        reinterpret_cast<const uint16_t *>(fetched_sparse_vector.values.data());
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      ASSERT_EQ(i, fetched_indices[i]);
      ASSERT_FLOAT_EQ(i, ailego::FloatHelper::ToFP32(fetched_values[i]));
    }

    // search
    auto query =
        vector_column_params::VectorData{vector_column_params::SparseVector{
            kSparseCount, indices.data(), values.data()}};
    auto query_params = vector_column_params::QueryParams{
        .topk = 10, .filter = nullptr, .fetch_vector = true};
    auto results = indexer->Search(query, query_params);
    ASSERT_TRUE(results.has_value());

    auto vector_results =
        dynamic_cast<VectorIndexResults *>(results.value().get());
    ASSERT_TRUE(vector_results);
    ASSERT_EQ(vector_results->count(), 1);

    {
      int count = 0;
      auto iter = vector_results->create_iterator();
      while (iter->valid()) {
        count++;
        iter->next();
      }
      ASSERT_EQ(count, 1);
    }

    {
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), kDocId);
      ASSERT_FLOAT_EQ(iter->score(), 5.0);

      auto vector = iter->vector();
      auto sparse_vector =
          std::get<vector_column_params::SparseVector>(vector.vector);
      auto indices = reinterpret_cast<const uint32_t *>(sparse_vector.indices);
      auto values = reinterpret_cast<const uint16_t *>(sparse_vector.values);
      ASSERT_EQ(sparse_vector.count, kSparseCount);
      for (uint32_t i = 0; i < kSparseCount; ++i) {
        ASSERT_EQ(i, indices[i]);
        ASSERT_FLOAT_EQ(i, ailego::FloatHelper::ToFP32(values[i]));
      }
      auto vector_index_params =
          reinterpret_cast<VectorIndexParams *>(index_params.get());
      if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
        ASSERT_TRUE(vector_results->docs().size() == 1);
        ASSERT_TRUE(vector_results->reverted_sparse_values_list().size() == 1);
        ASSERT_TRUE(vector_results->reverted_vector_list().empty());
      }
    }

    indexer->Close();

    system(cmd_buf);
  };

  func(std::make_shared<FlatIndexParams>(MetricType::IP));
  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100));
}

TEST(VectorColumnIndexerTest, Merge) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test_indexer.index"};

  auto del_index_file_func = [&](const std::string file_name) {
    auto cmd_buf = "rm -f " + file_name;
    system(cmd_buf.c_str());
  };

  auto create_indexer_func =
      [&](const IndexParams::Ptr &index_params,
          const std::string &index_name) -> VectorColumnIndexer::Ptr {
    del_index_file_func(index_name);
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_name, FieldSchema("test", DataType::VECTOR_FP32, kDimension,
                                false, index_params));
    if (indexer == nullptr ||
        !indexer->Open(vector_column_params::ReadOptions{true, true}).ok()) {
      return nullptr;
    }
    return indexer;
  };

  auto func = [&](const IndexParams::Ptr &param1,
                  const IndexParams::Ptr &param2,
                  const IndexParams::Ptr &param3) {
    auto indexer1 = create_indexer_func(param1, index_name + "1");
    ASSERT_NE(nullptr, indexer1);
    auto indexer2 = create_indexer_func(param2, index_name + "2");
    ASSERT_NE(nullptr, indexer2);

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 123.0f;
    auto vector_data = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector.data()}};
    ASSERT_TRUE(indexer1->Insert(vector_data, 0).ok());

    vector[1] = 2.0f;
    ASSERT_TRUE(indexer2->Insert(vector_data, 0).ok());
    vector[1] = 3.0f;
    ASSERT_TRUE(indexer2->Insert(vector_data, 1).ok());

    {
      auto fetched_data = indexer1->Fetch(0);
      ASSERT_TRUE(fetched_data.has_value());
      const float *fetched_vector = reinterpret_cast<const float *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(1.0f, fetched_vector[1], 0.1);
      ASSERT_NEAR(123.0f, fetched_vector[2], 0.1);
    }
    {
      auto fetched_data = indexer2->Fetch(0);
      ASSERT_TRUE(fetched_data.has_value());
      const float *fetched_vector = reinterpret_cast<const float *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(2.0f, fetched_vector[1], 0.1);
      ASSERT_NEAR(123.0f, fetched_vector[2], 0.1);
    }
    {
      auto fetched_data = indexer2->Fetch(1);
      ASSERT_TRUE(fetched_data.has_value());
      const float *fetched_vector = reinterpret_cast<const float *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      ASSERT_NEAR(3.0f, fetched_vector[1], 0.1);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }

    {  // test reduce
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, nullptr).ok());
      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        const float *fetched_vector = reinterpret_cast<const float *>(
            std::get<vector_column_params::DenseVectorBuffer>(
                fetched_data->vector_buffer)
                .data.data());
        ASSERT_NEAR(1.0f, fetched_vector[1], 0.1);
        ASSERT_NEAR(123.0f, fetched_vector[2], 0.1);
      }
      {
        auto fetched_data = indexer3->Fetch(1);
        ASSERT_TRUE(fetched_data.has_value());
        const float *fetched_vector = reinterpret_cast<const float *>(
            std::get<vector_column_params::DenseVectorBuffer>(
                fetched_data->vector_buffer)
                .data.data());
        ASSERT_NEAR(2.0f, fetched_vector[1], 0.1);
        ASSERT_NEAR(123.0f, fetched_vector[2], 0.1);
      }
      indexer3->Close();
      del_index_file_func(index_name + "3");
    }

    {  // test reduce with filter
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      auto filter = std::make_shared<EasyIndexFilter>(
          [](uint64_t key) { return key == 0; });
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, filter).ok());
      // 0.0 -> x ; 1.0 -> 0 ; 1.1 -> 1
      ASSERT_TRUE(indexer3->doc_count() == 2);
      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        const float *fetched_vector = reinterpret_cast<const float *>(
            std::get<vector_column_params::DenseVectorBuffer>(
                fetched_data->vector_buffer)
                .data.data());
        ASSERT_NEAR(2.0f, fetched_vector[1], 0.1);
        ASSERT_NEAR(123.0f, fetched_vector[2], 0.1);
      }

      {
        // search with fetch vector
        auto query = vector_column_params::VectorData{
            vector_column_params::DenseVector{vector.data()}};
        auto query_params = vector_column_params::QueryParams{
            .topk = 10, .filter = nullptr, .fetch_vector = true};
        auto results = indexer2->Search(query, query_params);
        ASSERT_TRUE(results.has_value());
        auto vector_results =
            dynamic_cast<VectorIndexResults *>(results.value().get());
        ASSERT_TRUE(vector_results);
        ASSERT_EQ(vector_results->count(), 2);
        auto iter = vector_results->create_iterator();
        ASSERT_TRUE(iter->valid());

        {
          ASSERT_TRUE(iter->valid());
          auto doc_id = iter->doc_id();
          LOG_DEBUG("topk1 pk: %zu", (size_t)doc_id);
          LOG_DEBUG("topk1 score: %.10f", iter->score());

          LOG_DEBUG(
              "topk1 fetched_vector:%s",
              print_dense_vector(std::get<vector_column_params::DenseVector>(
                                     iter->vector().vector)
                                     .data,
                                 3, DataType::VECTOR_FP32)
                  .c_str());
          {
            auto fetched_vector = vector_results->docs()[0].vector();

            LOG_DEBUG(
                "topk1 fetched_vector - original:%s",
                print_dense_vector(fetched_vector, 3, DataType::VECTOR_FP16)
                    .c_str());
          }
          if (!vector_results->reverted_vector_list().empty()) {
            auto fetched_vector =
                vector_results->reverted_vector_list()[0].data();

            LOG_DEBUG(
                "topk1 fetched_vector - reverted:%s",
                print_dense_vector(fetched_vector, 3, DataType::VECTOR_FP32)
                    .c_str());
          }
          // ASSERT_TRUE(iter->score() < 2.01);
          // ASSERT_TRUE(iter->score() > -0.01);
        }
      }

      indexer3->Close();
      del_index_file_func(index_name + "3");
    }

    {  // test reduce with filter in parallel
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      auto filter = std::make_shared<EasyIndexFilter>(
          [](uint64_t key) { return key == 0; });
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, filter, {3}).ok());

      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        const float *fetched_vector = reinterpret_cast<const float *>(
            std::get<vector_column_params::DenseVectorBuffer>(
                fetched_data->vector_buffer)
                .data.data());
        ASSERT_NEAR(2.0f, fetched_vector[1], 0.1);
        ASSERT_NEAR(123.0f, fetched_vector[2], 0.1);
      }
      indexer3->Close();
      del_index_file_func(index_name + "3");
    }


    indexer1->Close();
    indexer2->Close();
    del_index_file_func(index_name + "1");
    del_index_file_func(index_name + "2");
  };

  // same index with different quantize type
  auto test_different_quantize_type = [&](MetricType metric_type,
                                          QuantizeType quantize_type) {
    LOG_INFO(
        "Merge test_different_quantize_type(): with metric type %s and "
        "quantize type %s",
        MetricTypeCodeBook::AsString(metric_type).c_str(),
        QuantizeTypeCodeBook::AsString(quantize_type).c_str());

    auto param_flat = std::make_shared<FlatIndexParams>(metric_type);
    auto param_flat_fp16 =
        std::make_shared<FlatIndexParams>(metric_type, quantize_type);
    auto param_hnsw = std::make_shared<HnswIndexParams>(metric_type, 10, 100);
    auto param_hnsw_fp16 =
        std::make_shared<HnswIndexParams>(metric_type, 10, 100, quantize_type);

    func(param_flat, param_flat, param_hnsw_fp16);

    std::vector<IndexParams::Ptr> fp32_params = {param_flat, param_hnsw};
    std::vector<IndexParams::Ptr> fp16_params = {param_flat_fp16,
                                                 param_hnsw_fp16};
    // can't mix
    for (auto param_target : fp32_params) {
      func(param_flat_fp16, param_hnsw_fp16, param_target);
      // for (auto param1 : fp16_params) {
      //   for (auto param2 : fp16_params) {
      //     func(param1, param2, param_target);
      //   }
      // }
      func(param_hnsw, param_flat, param_target);
      // for (auto param1 : fp32_params) {
      //   for (auto param2 : fp32_params) {
      //     func(param1, param2, param_target);
      //   }
      // }
    }

    for (auto param_target : fp16_params) {
      func(param_flat_fp16, param_hnsw_fp16, param_target);
      // for (auto param1 : fp16_params) {
      //   for (auto param2 : fp16_params) {
      //     func(param1, param2, param_target);
      //   }
      // }
      func(param_hnsw, param_flat, param_target);
      // for (auto param1 : fp32_params) {
      //   for (auto param2 : fp32_params) {
      //     func(param1, param2, param_target);
      //   }
      // }
    }
  };
  test_different_quantize_type(MetricType::L2, QuantizeType::UNDEFINED);
  test_different_quantize_type(MetricType::L2, QuantizeType::FP16);
  test_different_quantize_type(MetricType::IP, QuantizeType::FP16);
  test_different_quantize_type(MetricType::L2, QuantizeType::INT8);
  // test_different_quantize_type(MetricType::IP, QuantizeType::INT8);
  // The quantization error is toooooo large for INT4 =_=
  // test_different_quantize_type(MetricType::L2, QuantizeType::INT4);
  // test_different_quantize_type(MetricType::IP, QuantizeType::INT4);
  // test_different_quantize_type(MetricType::COSINE);
}

TEST(VectorColumnIndexerTest, SparseMerge) {
  constexpr uint32_t kSparseCount = 3;
  constexpr uint32_t kUnitSize = sizeof(float);  // VECTOR_FP32
  const std::string index_name{"test_indexer.index"};

  auto del_index_file_func = [&](const std::string file_name) {
    auto cmd_buf = "rm -f " + file_name;
    system(cmd_buf.c_str());
  };

  auto create_indexer_func =
      [&](const IndexParams::Ptr &index_params,
          const std::string &index_name) -> VectorColumnIndexer::Ptr {
    del_index_file_func(index_name);
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_name,
        FieldSchema("test", DataType::SPARSE_VECTOR_FP32, false, index_params));
    if (indexer == nullptr ||
        !indexer->Open(vector_column_params::ReadOptions{true, true}).ok()) {
      return nullptr;
    }
    return indexer;
  };

  auto func = [&](const IndexParams::Ptr &param1,
                  const IndexParams::Ptr &param2,
                  const IndexParams::Ptr &param3) {
    auto indexer1 = create_indexer_func(param1, index_name + "1");
    ASSERT_NE(nullptr, indexer1);
    auto indexer2 = create_indexer_func(param2, index_name + "2");
    ASSERT_NE(nullptr, indexer2);

    std::vector<uint32_t> indices(kSparseCount);
    std::vector<float> values(kSparseCount);
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      indices[i] = i;
      values[i] = (float)i;
    }
    vector_column_params::SparseVector vector{kSparseCount, indices.data(),
                                              values.data()};
    auto vector_data = vector_column_params::VectorData{vector};
    ASSERT_TRUE(indexer1->Insert(vector_data, 0).ok());

    values[1] = 2.0f;
    ASSERT_TRUE(indexer2->Insert(vector_data, 0).ok());
    values[1] = 3.0f;
    ASSERT_TRUE(indexer2->Insert(vector_data, 1).ok());

    {
      auto fetched_data = indexer1->Fetch(0);
      ASSERT_TRUE(fetched_data.has_value());
      auto fetched_sparse_vector =
          std::get<vector_column_params::SparseVectorBuffer>(
              fetched_data->vector_buffer);
      ASSERT_EQ(kSparseCount,
                fetched_sparse_vector.indices.size() / sizeof(uint32_t));
      ASSERT_EQ(kSparseCount, fetched_sparse_vector.values.size() / kUnitSize);

      auto fetched_indices = reinterpret_cast<const uint32_t *>(
          fetched_sparse_vector.indices.data());
      auto fetched_values =
          reinterpret_cast<const float *>(fetched_sparse_vector.values.data());
      for (uint32_t i = 0; i < kSparseCount; ++i) {
        ASSERT_EQ(i, fetched_indices[i]);
      }
      ASSERT_EQ(0.0f, fetched_values[0]);
      ASSERT_EQ(1.0f, fetched_values[1]);
      ASSERT_EQ(2.0f, fetched_values[2]);
    }
    {
      auto fetched_data = indexer2->Fetch(0);
      ASSERT_TRUE(fetched_data.has_value());
      auto fetched_sparse_vector =
          std::get<vector_column_params::SparseVectorBuffer>(
              fetched_data->vector_buffer);
      ASSERT_EQ(kSparseCount,
                fetched_sparse_vector.indices.size() / sizeof(uint32_t));
      ASSERT_EQ(kSparseCount, fetched_sparse_vector.values.size() / kUnitSize);

      auto fetched_indices = reinterpret_cast<const uint32_t *>(
          fetched_sparse_vector.indices.data());
      auto fetched_values =
          reinterpret_cast<const float *>(fetched_sparse_vector.values.data());
      for (uint32_t i = 0; i < kSparseCount; ++i) {
        ASSERT_EQ(i, fetched_indices[i]);
      }
      ASSERT_EQ(0.0f, fetched_values[0]);
      ASSERT_EQ(2.0f, fetched_values[1]);
      ASSERT_EQ(2.0f, fetched_values[2]);
    }
    {
      auto fetched_data = indexer2->Fetch(1);
      ASSERT_TRUE(fetched_data.has_value());
      auto fetched_sparse_vector =
          std::get<vector_column_params::SparseVectorBuffer>(
              fetched_data->vector_buffer);
      ASSERT_EQ(kSparseCount,
                fetched_sparse_vector.indices.size() / sizeof(uint32_t));
      ASSERT_EQ(kSparseCount, fetched_sparse_vector.values.size() / kUnitSize);

      auto fetched_indices = reinterpret_cast<const uint32_t *>(
          fetched_sparse_vector.indices.data());
      auto fetched_values =
          reinterpret_cast<const float *>(fetched_sparse_vector.values.data());
      for (uint32_t i = 0; i < kSparseCount; ++i) {
        ASSERT_EQ(i, fetched_indices[i]);
      }
      ASSERT_EQ(0.0f, fetched_values[0]);
      ASSERT_EQ(3.0f, fetched_values[1]);
      ASSERT_EQ(2.0f, fetched_values[2]);
    }

    {  // test reduce
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, nullptr).ok());
      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        auto fetched_sparse_vector =
            std::get<vector_column_params::SparseVectorBuffer>(
                fetched_data->vector_buffer);
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.indices.size() / sizeof(uint32_t));
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.values.size() / kUnitSize);
        auto fetched_indices = reinterpret_cast<const uint32_t *>(
            fetched_sparse_vector.indices.data());
        auto fetched_values = reinterpret_cast<const float *>(
            fetched_sparse_vector.values.data());
        for (uint32_t i = 0; i < kSparseCount; ++i) {
          ASSERT_EQ(i, fetched_indices[i]);
        }
        ASSERT_EQ(0.0f, fetched_values[0]);
        ASSERT_EQ(1.0f, fetched_values[1]);
        ASSERT_EQ(2.0f, fetched_values[2]);
      }
      {
        auto fetched_data = indexer3->Fetch(1);
        ASSERT_TRUE(fetched_data.has_value());
        auto fetched_sparse_vector =
            std::get<vector_column_params::SparseVectorBuffer>(
                fetched_data->vector_buffer);
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.indices.size() / sizeof(uint32_t));
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.values.size() / kUnitSize);
        auto fetched_indices = reinterpret_cast<const uint32_t *>(
            fetched_sparse_vector.indices.data());
        auto fetched_values = reinterpret_cast<const float *>(
            fetched_sparse_vector.values.data());
        for (uint32_t i = 0; i < kSparseCount; ++i) {
          ASSERT_EQ(i, fetched_indices[i]);
        }
        ASSERT_EQ(0.0f, fetched_values[0]);
        ASSERT_EQ(2.0f, fetched_values[1]);
        ASSERT_EQ(2.0f, fetched_values[2]);
      }
      indexer3->Close();
      del_index_file_func(index_name + "3");
    }

    {  // test reduce with filter
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      auto filter = std::make_shared<EasyIndexFilter>(
          [](uint64_t key) { return key == 0; });
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, filter).ok());
      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        auto fetched_sparse_vector =
            std::get<vector_column_params::SparseVectorBuffer>(
                fetched_data->vector_buffer);
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.indices.size() / sizeof(uint32_t));
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.values.size() / kUnitSize);
        auto fetched_indices = reinterpret_cast<const uint32_t *>(
            fetched_sparse_vector.indices.data());
        auto fetched_values = reinterpret_cast<const float *>(
            fetched_sparse_vector.values.data());
        for (uint32_t i = 0; i < kSparseCount; ++i) {
          ASSERT_EQ(i, fetched_indices[i]);
        }
        ASSERT_EQ(0.0f, fetched_values[0]);
        ASSERT_EQ(2.0f, fetched_values[1]);
        ASSERT_EQ(2.0f, fetched_values[2]);
      }
      indexer3->Close();
      del_index_file_func(index_name + "3");
    }

    {  // test reduce with filter in parallel
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      auto filter = std::make_shared<EasyIndexFilter>(
          [](uint64_t key) { return key == 0; });
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, filter, {3}).ok());
      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        auto fetched_sparse_vector =
            std::get<vector_column_params::SparseVectorBuffer>(
                fetched_data->vector_buffer);
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.indices.size() / sizeof(uint32_t));
        ASSERT_EQ(kSparseCount,
                  fetched_sparse_vector.values.size() / kUnitSize);
        auto fetched_indices = reinterpret_cast<const uint32_t *>(
            fetched_sparse_vector.indices.data());
        auto fetched_values = reinterpret_cast<const float *>(
            fetched_sparse_vector.values.data());
        for (uint32_t i = 0; i < kSparseCount; ++i) {
          ASSERT_EQ(i, fetched_indices[i]);
        }
        ASSERT_EQ(0.0f, fetched_values[0]);
        ASSERT_EQ(2.0f, fetched_values[1]);
        ASSERT_EQ(2.0f, fetched_values[2]);
      }
      indexer3->Close();
      del_index_file_func(index_name + "3");
    }


    indexer1->Close();
    indexer2->Close();
    del_index_file_func(index_name + "1");
    del_index_file_func(index_name + "2");
  };


  //===============================================
  // Fp32
  //===============================================
  {
    auto param_flat = std::make_shared<FlatIndexParams>(MetricType::IP);
    auto param_hnsw =
        std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100);
    LOG_INFO("SparseMerge: param_flat, param_flat, param_flat");
    func(param_flat, param_flat, param_flat);

    LOG_INFO("SparseMerge: param_hnsw, param_hnsw, param_hnsw");
    func(param_hnsw, param_hnsw, param_hnsw);

    LOG_INFO("SparseMerge: param_flat, param_hnsw, param_hnsw");
    func(param_flat, param_hnsw, param_hnsw);

    LOG_INFO("SparseMerge: param_hnsw, param_flat, param_flat");
    func(param_hnsw, param_flat, param_flat);
    LOG_INFO("SparseMerge: param_flat, param_hnsw, param_flat");
    func(param_flat, param_hnsw, param_flat);

    LOG_INFO("SparseMerge: param_hnsw, param_flat, param_hnsw");
    func(param_hnsw, param_flat, param_hnsw);
  }

  //===============================================
  // Fp16 fp32
  //===============================================
  {
    auto param_flat = std::make_shared<FlatIndexParams>(MetricType::IP);
    auto param_hnsw = std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                                        QuantizeType::FP16);
    LOG_INFO("SparseMerge - fp16: param_flat, param_flat -> param_flat");
    func(param_flat, param_flat, param_flat);

    LOG_INFO("SparseMerge - fp16: param_hnsw, param_hnsw -> param_hnsw");
    func(param_hnsw, param_hnsw, param_hnsw);

    LOG_INFO("SparseMerge - fp16: param_hnsw, param_hnsw -> param_flat");
    func(param_hnsw, param_hnsw, param_flat);

    LOG_INFO("SparseMerge - fp16: param_flat, param_flat -> param_hnsw");
    func(param_flat, param_flat, param_hnsw);
  }
}


TEST(VectorColumnIndexerTest, BfPks) {
  auto func = [&](const IndexParams::Ptr index_params) {
    const std::string index_file_path = "test_indexer.index";

    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
    system(cmd_buf);

    // 1. create indexer
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false, index_params));
    ASSERT_TRUE(indexer);

    // 2. open
    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    auto vector1 = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto vector2 = std::vector<float>{4.0f, 5.0f, 6.0f};

    // 3. add data
    auto data1 = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector1.data()}};
    ASSERT_TRUE(indexer->Insert(data1, 1).ok());

    auto data2 = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector2.data()}};
    ASSERT_TRUE(indexer->Insert(data2, 2).ok());

    {
      auto bf_pks = std::vector<uint64_t>{1};
      auto query =
          vector_column_params::VectorData{vector_column_params::DenseVector{
              .data = std::vector<float>{1.0f, 2.0f, 3.0f}.data()}};
      auto query_params =
          vector_column_params::QueryParams{.topk = 10,
                                            .filter = nullptr,
                                            .fetch_vector = true,
                                            .bf_pks = {bf_pks}};
      auto results = indexer->Search(query, query_params);
      ASSERT_TRUE(results.has_value());

      auto vector_results =
          dynamic_cast<VectorIndexResults *>(results.value().get());
      ASSERT_TRUE(vector_results);
      ASSERT_EQ(vector_results->count(), 1);
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), 1);
      auto fetched_vector =
          std::get<vector_column_params::DenseVector>(iter->vector().vector);
      const float *fetched_vector_data =
          reinterpret_cast<const float *>(fetched_vector.data);
      for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(fetched_vector_data[i], vector1[i]);
      }
    }

    {
      auto bf_pks = std::vector<uint64_t>{1, 2};
      auto query =
          vector_column_params::VectorData{vector_column_params::DenseVector{
              .data = std::vector<float>{1.0f, 2.0f, 3.0f}.data()}};
      auto query_params =
          vector_column_params::QueryParams{.topk = 10,
                                            .filter = nullptr,
                                            .fetch_vector = true,
                                            .bf_pks = {bf_pks}};
      auto results = indexer->Search(query, query_params);
      ASSERT_TRUE(results.has_value());

      auto vector_results =
          dynamic_cast<VectorIndexResults *>(results.value().get());
      ASSERT_TRUE(vector_results);
      ASSERT_EQ(vector_results->count(), 2);
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), 1);
      auto fetched_vector =
          std::get<vector_column_params::DenseVector>(iter->vector().vector);
      const float *fetched_vector_data =
          reinterpret_cast<const float *>(fetched_vector.data);
      for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(fetched_vector_data[i], vector1[i]);
      }
    }

    {
      auto bf_pks = std::vector<uint64_t>{2};
      auto query =
          vector_column_params::VectorData{vector_column_params::DenseVector{
              .data = std::vector<float>{1.0f, 2.0f, 3.0f}.data()}};
      auto query_params =
          vector_column_params::QueryParams{.topk = 10,
                                            .filter = nullptr,
                                            .fetch_vector = true,
                                            .bf_pks = {bf_pks}};
      auto results = indexer->Search(query, query_params);
      ASSERT_TRUE(results.has_value());

      auto vector_results =
          dynamic_cast<VectorIndexResults *>(results.value().get());
      ASSERT_TRUE(vector_results);
      ASSERT_EQ(vector_results->count(), 1);
      auto iter = vector_results->create_iterator();
      ASSERT_TRUE(iter->valid());
      ASSERT_EQ(iter->doc_id(), 2);
      auto fetched_vector =
          std::get<vector_column_params::DenseVector>(iter->vector().vector);
      const float *fetched_vector_data =
          reinterpret_cast<const float *>(fetched_vector.data);
      for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(fetched_vector_data[i], vector2[i]);
      }
    }

    indexer->Close();

    system(cmd_buf);
  };

  func(std::make_shared<FlatIndexParams>(MetricType::COSINE));
  func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100));
}


using DenseVectorDataBuffer = vector_column_params::DenseVectorBuffer;
using SparseVectorBuffer = vector_column_params::SparseVectorBuffer;

DenseVectorDataBuffer create_dense_vector(int dim, DataType data_type, int pk,
                                          size_t count,
                                          float float_offset = 0.1f) {
  count += 1;
  switch (data_type) {
    case DataType::VECTOR_FP32: {
      std::string ret;
      ret.resize(dim * sizeof(float));
      float *data = reinterpret_cast<float *>(ret.data());
      for (int i = 0; i < dim; ++i) {
        data[i] = pk + i + float_offset;
      }
      return DenseVectorDataBuffer{std::move(ret)};
    }
    case DataType::VECTOR_FP16: {
      std::string ret;
      ret.resize(dim * sizeof(zvec::float16_t));
      zvec::float16_t *data = reinterpret_cast<zvec::float16_t *>(ret.data());
      for (int i = 0; i < dim; ++i) {
        data[i] = pk + i + float_offset;
      }
      return DenseVectorDataBuffer{std::move(ret)};
    }
    case DataType::VECTOR_INT8: {
      std::string ret;
      ret.resize(dim * sizeof(int8_t));
      int8_t *data = reinterpret_cast<int8_t *>(ret.data());
      for (int i = 0; i < dim; ++i) {
        data[i] = pk + i;
      }
      return DenseVectorDataBuffer{std::move(ret)};
    }
    case DataType::VECTOR_INT16: {
      std::string ret;
      ret.resize(dim * sizeof(int16_t));
      int16_t *data = reinterpret_cast<int16_t *>(ret.data());
      for (int i = 0; i < dim; ++i) {
        data[i] = pk + i;
      }
      return DenseVectorDataBuffer{std::move(ret)};
    }
    case DataType::VECTOR_BINARY32:
    case DataType::VECTOR_BINARY64: {
      std::string ret;
      ret.resize(dim / 8);
      uint8_t *data = reinterpret_cast<uint8_t *>(ret.data());
      for (int i = 0; i < dim; ++i) {
        data[i / 8] |= ((pk + i) % 2) << (i % 8);
      }
      return DenseVectorDataBuffer{std::move(ret)};
    }
    default:
      LOG_ERROR("Unsupported data type: %d", static_cast<int>(data_type));
      return DenseVectorDataBuffer{};
  }
}


SparseVectorBuffer create_sparse_vector(int dim, DataType data_type, int pk,
                                        float float_offset = 0.1f) {
  SparseVectorBuffer ret;
  switch (data_type) {
    case DataType::SPARSE_VECTOR_FP32: {
      std::vector<float> values(dim);
      for (int i = 0; i < dim; ++i) {
        values[i] = pk * 100 + i + float_offset;
      }
      ret.values = std::string(reinterpret_cast<char *>(values.data()),
                               values.size() * sizeof(float));
    } break;
    case DataType::SPARSE_VECTOR_FP16: {
      std::vector<zvec::float16_t> values(dim);
      for (int i = 0; i < dim; ++i) {
        values[i] = pk * 100 + i + float_offset;
      }
      ret.values = std::string(reinterpret_cast<char *>(values.data()),
                               values.size() * sizeof(zvec::float16_t));
    } break;
    default:
      LOG_ERROR("Unsupported data type: %d", static_cast<int>(data_type));
      return SparseVectorBuffer{};
  }
  std::vector<uint32_t> indices(dim);
  for (int i = 0; i < dim; ++i) {
    indices[i] = i;
  }
  ret.indices = std::string(reinterpret_cast<char *>(indices.data()),
                            indices.size() * sizeof(uint32_t));
  return ret;
}

bool compare_dense_vector(const DenseVectorDataBuffer &lhs, const void *rhs,
                          DataType data_type) {
  switch (data_type) {
    case DataType::VECTOR_FP32: {
      size_t dim = lhs.data.size() / sizeof(float);
      auto rhs_data = reinterpret_cast<const float *>(rhs);
      auto lhs_data = reinterpret_cast<const float *>(lhs.data.data());
      for (size_t i = 0; i < dim; ++i) {
        if (std::abs(lhs_data[i] - rhs_data[i]) > 1) {  // reformer
          LOG_ERROR("lhs_data[%zu] = %f, rhs_data[%zu] = %f", i,
                    (float)lhs_data[i], i, (float)rhs_data[i]);
          return false;
        }
      }
      return true;
    };
    case DataType::VECTOR_FP16: {
      size_t dim = lhs.data.size() / sizeof(zvec::float16_t);
      auto rhs_data = reinterpret_cast<const zvec::float16_t *>(rhs);
      auto lhs_data =
          reinterpret_cast<const zvec::float16_t *>(lhs.data.data());
      for (size_t i = 0; i < dim; ++i) {
        if (std::abs(lhs_data[i] - rhs_data[i]) > 1e-2) {  // reformer
          LOG_ERROR("lhs_data[%zu] = %f, rhs_data[%zu] = %f", i,
                    (float)lhs_data[i], i, (float)rhs_data[i]);
          return false;
        }
      }
      return true;
    }
    default:
      return memcmp(lhs.data.data(), rhs, lhs.data.size()) == 0;
  }
}


bool compare_sparse_vector(const SparseVectorBuffer &lhs,
                           const void *rhs_indices, const void *rhs_values,
                           DataType data_type) {
  if (memcmp(lhs.indices.data(), rhs_indices, lhs.indices.size()) != 0) {
    return false;
  }
  size_t dim = lhs.indices.size() / sizeof(uint32_t);
  switch (data_type) {
    case DataType::SPARSE_VECTOR_FP32: {
      auto rhs_values_data = reinterpret_cast<const float *>(rhs_values);
      auto lhs_values_data = reinterpret_cast<const float *>(lhs.values.data());
      for (size_t i = 0; i < dim; ++i) {
        if (std::abs(lhs_values_data[i] - rhs_values_data[i]) >
            1e-2) {  // reformer
          LOG_ERROR("lhs_values_data[%zu] = %f, rhs_values_data[%zu] = %f", i,
                    (float)lhs_values_data[i], i, (float)rhs_values_data[i]);
          return false;
        }
      }
      return true;
    }
    case DataType::SPARSE_VECTOR_FP16: {
      auto rhs_values_data =
          reinterpret_cast<const zvec::float16_t *>(rhs_values);
      auto lhs_values_data =
          reinterpret_cast<const zvec::float16_t *>(lhs.values.data());
      for (size_t i = 0; i < dim; ++i) {
        if (std::abs(lhs_values_data[i] - rhs_values_data[i]) >
            1e-2) {  // reformer
          LOG_ERROR("lhs_values_data[%zu] = %f, rhs_values_data[%zu] = %f", i,
                    (float)lhs_values_data[i], i, (float)rhs_values_data[i]);
          return false;
        }
      }
      return true;
    }
    default:
      return memcmp(lhs.values.data(), rhs_values, lhs.values.size()) == 0;
  }
}


TEST(VectorColumnIndexerTest, CosineGeneral) {
  const std::string index_file_path = "test_indexer.index";
  const int kDim = 20;
  const int kCount = 20;  // can't set too large, or the qunatization error
                          // will be too large due to float's precision
  const int kTopk = 10;

  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
  system(cmd_buf);

  auto func = [&](const IndexParams::Ptr index_params, DataType data_type) {
    system(cmd_buf);
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", data_type, kDim, false, index_params));
    ASSERT_TRUE(indexer);

    if (auto ret = indexer->Open(vector_column_params::ReadOptions{true, true});
        !ret.ok()) {
      LOG_ERROR("Failed to open indexer: %s", ret.message().c_str());
      return;
    }

    // insert
    for (int i = 0; i < kCount; ++i) {
      auto buffer = create_dense_vector(kDim, data_type, i, kCount, 0.1f);
      // print_dense_vector(buffer.data.data(), kDim, data_type);
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{buffer.data.data()}};
      ASSERT_TRUE(indexer->Insert(data, i).ok());
    }

    // fetch
    for (int i = 0; i < kCount; ++i) {
      auto fetched_data = indexer->Fetch(i);
      ASSERT_TRUE(fetched_data);
      ASSERT_TRUE(compare_dense_vector(
          create_dense_vector(kDim, data_type, i, kCount, 0.1f),
          std::get<DenseVectorDataBuffer>(fetched_data->vector_buffer)
              .data.data(),
          data_type));
    }

    // query
    for (int i = 0; i < kCount; ++i) {
      auto buffer = create_dense_vector(kDim, data_type, i, kCount, 0.3f);
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{buffer.data.data()}};
      auto _t = std::make_shared<zvec::HnswQueryParams>(100);
      _t->set_is_linear(true);
      auto query_params =
          vector_column_params::QueryParams{.topk = kTopk,
                                            .filter = nullptr,
                                            .fetch_vector = true,
                                            .query_params = _t};
      auto results = indexer->Search(data, query_params);
      ASSERT_TRUE(results.has_value());
      auto vector_results =
          dynamic_cast<VectorIndexResults *>(results.value().get());
      ASSERT_TRUE(vector_results);
      ASSERT_EQ(vector_results->count(), kTopk);
      auto iter = vector_results->create_iterator();
      LOG_INFO("===query pk: %d", i);
      LOG_INFO("query_vector:%s",
               print_dense_vector(buffer.data.data(), kDim, data_type).c_str());
      {  // topk1
        ASSERT_TRUE(iter->valid());
        LOG_INFO("topk1 pk:%zu", (size_t)iter->doc_id());
        LOG_INFO("topk1 score:%.10f", iter->score());

        if (!(iter->score() > -0.01 && iter->score() < 2.01)) {
          ASSERT_TRUE(iter->score() < 2.01);
        }

        ASSERT_TRUE(iter->score() < 2.01);
        ASSERT_TRUE(iter->score() > -0.01);

        auto fetched_vector =
            std::get<vector_column_params::DenseVector>(iter->vector().vector);
        LOG_INFO(
            "topk1 fetched_vector:%s",
            print_dense_vector(fetched_vector.data, kDim, data_type).c_str());

        // ASSERT_EQ(iter->doc_id(), i);
        ASSERT_TRUE(compare_dense_vector(
            create_dense_vector(kDim, data_type, iter->doc_id(), kCount, 0.1f),
            fetched_vector.data, data_type));
      }
    }
    indexer->Destroy();
  };

  LOG_INFO("Test FlatIndexParams(MetricType::COSINE), VECTOR_FP32");
  func(std::make_shared<FlatIndexParams>(MetricType::COSINE),
       DataType::VECTOR_FP32);
  LOG_INFO("Test HnswIndexParams(MetricType::COSINE), VECTOR_FP32");
  func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100),
       DataType::VECTOR_FP32);
  LOG_INFO(
      "Test FlatIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::FP16");
  func(
      std::make_shared<FlatIndexParams>(MetricType::COSINE, QuantizeType::FP16),
      DataType::VECTOR_FP32);
  LOG_INFO(
      "Test HnswIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::FP16");
  func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100,
                                         QuantizeType::FP16),
       DataType::VECTOR_FP32);

  LOG_INFO(
      "Test FlatIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::INT8");
  func(
      std::make_shared<FlatIndexParams>(MetricType::COSINE, QuantizeType::INT8),
      DataType::VECTOR_FP32);
  LOG_INFO(
      "Test HnswIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::INT8");
  func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100,
                                         QuantizeType::INT8),
       DataType::VECTOR_FP32);

  LOG_INFO(
      "Test FlatIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::INT4");
  func(
      std::make_shared<FlatIndexParams>(MetricType::COSINE, QuantizeType::INT4),
      DataType::VECTOR_FP32);
  LOG_INFO(
      "Test HnswIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::INT4");
  func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100,
                                         QuantizeType::INT4),
       DataType::VECTOR_FP32);

  // cosine doesn't support int8/int4 datatype, but support int8/int4 quantizer

  // LOG_INFO("Test FlatIndexParams(MetricType::COSINE), VECTOR_FP16");
  // func(
  //     std::make_shared<FlatIndexParams>(MetricType::COSINE,
  //     QuantizeType::FP16), DataType::VECTOR_FP16);
  // LOG_INFO("Test HnswIndexParams(MetricType::COSINE), VECTOR_FP16");
  // func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100,
  //                                        QuantizeType::FP16),
  //      DataType::VECTOR_FP16);
}


TEST(VectorColumnIndexerTest, Score) {
  const std::string index_file_path = "test_indexer.index";
  const int kTopk = 10;
  constexpr idx_t kDocId1 = 2345;
  constexpr idx_t kDocId2 = 5432;
  auto vector1 = std::vector<float>{3.0f, 4.0f, 5.0f};
  auto vector2 = std::vector<float>{1.0f, 20.0f, 3.0f};
  auto vector_id_map = std::unordered_map<idx_t, std::vector<float>>{
      {kDocId1, vector1},
      {kDocId2, vector2},
  };
  auto sparse_indices = std::vector<uint32_t>{0, 1, 2};
  auto query_vector = std::vector<float>{1.0f, 2.0f, 3.0f};

  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
  system(cmd_buf);


  auto check_score = [&](VectorIndexResults *vector_results,
                         MetricType metric_type) {
    ASSERT_TRUE(vector_results);
    ASSERT_EQ(vector_results->count(), 2);

    auto inner_produce_score_func = [&](const std::vector<float> &v1,
                                        const std::vector<float> &v2) {
      return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    };

    auto cosine_score_func = [&](const std::vector<float> &v1,
                                 const std::vector<float> &v2) {
      return 1 - inner_produce_score_func(v1, v2) /
                     (std::sqrt(inner_produce_score_func(v1, v1)) *
                      std::sqrt(inner_produce_score_func(v2, v2)));
    };

    // SquaredEuclidean
    auto l2_score_func = [&](const std::vector<float> &v1,
                             const std::vector<float> &v2) {
      assert(v1.size() == 3);
      assert(v2.size() == 3);
      float ret = 0.0f;
      for (size_t i = 0; i < v1.size(); ++i) {
        ret += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      }
      return ret;
    };

    std::function<float(const std::vector<float> &, const std::vector<float> &)>
        score_func;

    switch (metric_type) {
      case MetricType::IP:
        score_func = inner_produce_score_func;
        break;
      case MetricType::COSINE:
        score_func = cosine_score_func;
        break;
      case MetricType::L2:
        score_func = l2_score_func;
        break;
      default:
        ASSERT_TRUE(false);
    }
    auto iter = vector_results->create_iterator();
    ASSERT_TRUE(iter->valid());
    printf("iter->score() top1: %f\n", iter->score());
    printf("score_func(vector_id_map[iter->doc_id()], query_vector): %f\n",
           score_func(vector_id_map[iter->doc_id()], query_vector));
    ASSERT_TRUE(
        std::abs(iter->score() - score_func(vector_id_map[iter->doc_id()],
                                            query_vector)) < 1e-2);
    iter->next();
    ASSERT_TRUE(iter->valid());
    printf("iter->score() top2: %f\n", iter->score());
    printf("score_func(vector_id_map[iter->doc_id()], query_vector): %f\n",
           score_func(vector_id_map[iter->doc_id()], query_vector));
    ASSERT_TRUE(
        std::abs(iter->score() - score_func(vector_id_map[iter->doc_id()],
                                            query_vector)) < 1e-2);
  };

  auto dense_func = [&](const std::shared_ptr<VectorIndexParams>
                            &index_params) {
    auto metric_type = index_params->metric_type();
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false, index_params));
    ASSERT_TRUE(indexer);

    if (auto ret = indexer->Open(vector_column_params::ReadOptions{true, true});
        !ret.ok()) {
      LOG_ERROR("Failed to open indexer: %s", ret.message().c_str());
      ASSERT_TRUE(false);
    }

    ASSERT_TRUE(indexer
                    ->Insert(
                        vector_column_params::VectorData{
                            vector_column_params::DenseVector{vector1.data()}},
                        kDocId1)
                    .ok());
    ASSERT_TRUE(indexer
                    ->Insert(
                        vector_column_params::VectorData{
                            vector_column_params::DenseVector{vector2.data()}},
                        kDocId2)
                    .ok());

    auto query = vector_column_params::VectorData{
        vector_column_params::DenseVector{.data = query_vector.data()}};
    auto query_params = vector_column_params::QueryParams{
        .topk = kTopk, .filter = nullptr, .fetch_vector = true};
    auto results = indexer->Search(query, query_params);
    ASSERT_TRUE(results.has_value());

    check_score(dynamic_cast<VectorIndexResults *>(results.value().get()),
                metric_type);

    indexer->Destroy();
  };

  auto sparse_func = [&](const std::shared_ptr<VectorIndexParams>
                             &index_params) {
    auto metric_type = index_params->metric_type();
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::SPARSE_VECTOR_FP32, false, index_params));
    ASSERT_TRUE(indexer);

    if (auto ret = indexer->Open(vector_column_params::ReadOptions{true, true});
        !ret.ok()) {
      LOG_ERROR("Failed to open indexer: %s", ret.message().c_str());
      ASSERT_TRUE(false);
    }

    ASSERT_TRUE(
        indexer
            ->Insert(
                vector_column_params::VectorData{
                    vector_column_params::SparseVector{
                        3,
                        reinterpret_cast<const void *>(sparse_indices.data()),
                        vector1.data()}},
                kDocId1)
            .ok());
    ASSERT_TRUE(
        indexer
            ->Insert(
                vector_column_params::VectorData{
                    vector_column_params::SparseVector{
                        3,
                        reinterpret_cast<const void *>(sparse_indices.data()),
                        vector2.data()}},
                kDocId2)
            .ok());

    auto query =
        vector_column_params::VectorData{vector_column_params::SparseVector{
            3, reinterpret_cast<const void *>(sparse_indices.data()),
            query_vector.data()}};
    auto query_params = vector_column_params::QueryParams{
        .topk = 10, .filter = nullptr, .fetch_vector = true};
    auto results = indexer->Search(query, query_params);
    ASSERT_TRUE(results.has_value());

    check_score(dynamic_cast<VectorIndexResults *>(results.value().get()),
                metric_type);
    indexer->Destroy();
  };

  LOG_INFO("Test DenseVector, MetricType::IP");
  dense_func(std::make_shared<FlatIndexParams>(MetricType::IP));
  dense_func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100));
  LOG_INFO("Test DenseVector, MetricType::IP, QuantizeType::FP16");
  dense_func(
      std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16));
  dense_func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                               QuantizeType::FP16));

  LOG_INFO("Test DenseVector, MetricType::COSINE");
  dense_func(std::make_shared<FlatIndexParams>(MetricType::COSINE));
  dense_func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100));

  LOG_INFO("Test DenseVector, MetricType::COSINE, QuantizeType::FP16");
  dense_func(std::make_shared<FlatIndexParams>(MetricType::COSINE,
                                               QuantizeType::FP16));
  dense_func(std::make_shared<HnswIndexParams>(MetricType::COSINE, 10, 100,
                                               QuantizeType::FP16));

  LOG_INFO("Test DenseVector, MetricType::L2");
  dense_func(std::make_shared<FlatIndexParams>(MetricType::L2));
  dense_func(std::make_shared<HnswIndexParams>(MetricType::L2, 10, 100));
  LOG_INFO("Test DenseVector, MetricType::L2, QuantizeType::FP16");
  dense_func(
      std::make_shared<FlatIndexParams>(MetricType::L2, QuantizeType::FP16));
  dense_func(std::make_shared<HnswIndexParams>(MetricType::L2, 10, 100,
                                               QuantizeType::FP16));

  LOG_INFO("Test SparseVector, MetricType::IP");
  sparse_func(std::make_shared<FlatIndexParams>(MetricType::IP));
  sparse_func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100));
  LOG_INFO("Test SparseVector, MetricType::IP, QuantizeType::FP16");
  sparse_func(
      std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16));
  sparse_func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                                QuantizeType::FP16));
}

TEST(VectorColumnIndexerTest, Failure) {
  const std::string index_file_path = "test_indexer_failure.index";
  constexpr idx_t kDocId = 1234;
  auto vector = std::vector<float>{1.0f, 2.0f, 3.0f};

  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
  system(cmd_buf);

  // Test case 1: Operations on unopened indexer
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);

    // Test Flush on unopened indexer
    auto flush_result = indexer->Flush();
    ASSERT_FALSE(flush_result.ok());
    ASSERT_EQ(flush_result.message(), "Index not opened");

    // Test Close on unopened indexer
    auto close_result = indexer->Close();
    ASSERT_FALSE(close_result.ok());
    ASSERT_EQ(close_result.message(), "Index not opened");

    // Test Destroy on unopened indexer
    auto destroy_result = indexer->Destroy();
    ASSERT_FALSE(destroy_result.ok());
    ASSERT_EQ(destroy_result.message(), "Index not opened");

    // Test Insert on unopened indexer
    auto data = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector.data()}};
    auto insert_result = indexer->Insert(data, kDocId);
    ASSERT_FALSE(insert_result.ok());
    ASSERT_EQ(insert_result.message(), "Index not opened");

    // Test Fetch on unopened indexer
    auto fetch_result = indexer->Fetch(kDocId);
    ASSERT_FALSE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.error().message(), "Index not opened");

    // Test Search on unopened indexer
    auto query = vector_column_params::VectorData{
        vector_column_params::DenseVector{.data = vector.data()}};
    auto query_params = vector_column_params::QueryParams{
        .topk = 10, .filter = nullptr, .fetch_vector = false};
    auto search_result = indexer->Search(query, query_params);
    ASSERT_FALSE(search_result.has_value());
    ASSERT_EQ(search_result.error().message(), "Index not opened");

    // Test Merge on unopened indexer
    auto merge_result = indexer->Merge({}, nullptr);
    ASSERT_FALSE(merge_result.ok());
    ASSERT_EQ(merge_result.message(), "Index not opened");
  }

  // Test case 2: Unsupported engine name
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)),
        "unsupported_engine");
    ASSERT_TRUE(indexer);

    auto open_result =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_FALSE(open_result.ok());
    ASSERT_EQ(open_result.message(), "Engine name not supported");
  }

  // Test case 3: Invalid field schema (nullptr index_params)
  {
    FieldSchema invalid_schema("test", DataType::VECTOR_FP32, 3, false,
                               nullptr);
    auto indexer =
        std::make_shared<VectorColumnIndexer>(index_file_path, invalid_schema);
    ASSERT_TRUE(indexer);

    auto open_result =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_FALSE(open_result.ok());
    ASSERT_EQ(open_result.message(), "field_schema.index_params nullptr");
  }

  // Test case 4: Unsupported data type in engine helper
  {
    // Create a mock index params with unsupported data type
    // We'll use a data type that's not supported by convert_to_engine_data_type
    FieldSchema unsupported_schema(
        "test", DataType::UNDEFINED, 3, false,
        std::make_shared<FlatIndexParams>(MetricType::IP));
    auto indexer = std::make_shared<VectorColumnIndexer>(index_file_path,
                                                         unsupported_schema);
    ASSERT_TRUE(indexer);

    auto open_result =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_FALSE(open_result.ok());
    ASSERT_EQ(open_result.message(),
              "failed to build index param: unsupported data type");
  }

  // Test case 5: Unsupported metric type in engine helper
  {
    FieldSchema unsupported_schema(
        "test", DataType::VECTOR_FP32, 3, false,
        std::make_shared<FlatIndexParams>(MetricType::UNDEFINED));
    auto indexer = std::make_shared<VectorColumnIndexer>(index_file_path,
                                                         unsupported_schema);
    ASSERT_TRUE(indexer);

    auto open_result =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_FALSE(open_result.ok());
    ASSERT_EQ(open_result.message(),
              "failed to build index param: unsupported metric type");
  }

  // Test case 6: Unsupported quantize type in engine helper
  {
    auto index_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    index_params->set_quantize_type(static_cast<QuantizeType>(999));


    FieldSchema unsupported_schema("test", DataType::VECTOR_FP32, 3, false,
                                   index_params);
    auto indexer = std::make_shared<VectorColumnIndexer>(index_file_path,
                                                         unsupported_schema);
    ASSERT_TRUE(indexer);

    auto open_result =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_FALSE(open_result.ok());
    ASSERT_EQ(open_result.message(),
              "failed to build index param: unsupported quantize type");
  }

  // // Test case 7: Unsupported index type in engine helper
  // {
  //   // Create a custom index params with unsupported index type
  //   class UnsupportedIndexTypeParams : public FlatIndexParams {
  //    public:
  //     UnsupportedIndexTypeParams() : FlatIndexParams(MetricType::IP) {}
  //     void mock() {
  //       type_ = static_cast<IndexType>(999);
  //     }
  //   };
  //   auto index_params = std::make_shared<UnsupportedIndexTypeParams>();
  //   index_params->mock();
  //   FieldSchema unsupported_schema("test", DataType::VECTOR_FP32, 3, false,
  //                                  index_params);
  //   auto indexer = std::make_shared<VectorColumnIndexer>(index_file_path,
  //                                                        unsupported_schema);
  //   ASSERT_TRUE(indexer);
  //
  //   auto open_result =
  //       indexer->Open(vector_column_params::ReadOptions{true, true});
  //   ASSERT_FALSE(open_result.ok());
  //   ASSERT_EQ(open_result.message(), "not supported");
  // }

  // Test case 8: bf_pks size > 1 error
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);

    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    // Insert some data first
    auto data = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector.data()}};
    ASSERT_TRUE(indexer->Insert(data, kDocId).ok());

    // Test search with bf_pks size > 1
    auto query = vector_column_params::VectorData{
        vector_column_params::DenseVector{.data = vector.data()}};
    auto bf_pks1 = std::vector<uint64_t>{1, 2};
    auto bf_pks2 = std::vector<uint64_t>{3, 4};
    auto query_params =
        vector_column_params::QueryParams{.topk = 10,
                                          .filter = nullptr,
                                          .fetch_vector = false,
                                          .bf_pks = {bf_pks1, bf_pks2}};

    auto search_result = indexer->Search(query, query_params);
    ASSERT_FALSE(search_result.has_value());
    ASSERT_EQ(search_result.error().message(),
              "bf_pks size > 1 is not supported");

    indexer->Destroy();
  }

  // Test case 9: Invalid field schema for query param conversion
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false, nullptr));
    ASSERT_TRUE(indexer);

    ASSERT_FALSE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());
  }

  // Test case 10: use_mmap = false
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);
    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true, false})
            .ok());
    // Insert some data first
    auto data = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector.data()}};
    ASSERT_TRUE(indexer->Insert(data, kDocId).ok());
    ASSERT_TRUE(indexer->Flush().ok());
    ASSERT_TRUE(indexer->Close().ok());
    {
      auto indexer = std::make_shared<VectorColumnIndexer>(
          index_file_path,
          FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                      std::make_shared<FlatIndexParams>(MetricType::IP)));
      ASSERT_TRUE(indexer);
      auto open_result =
          indexer->Open(vector_column_params::ReadOptions{false, false, true});
      ASSERT_TRUE(open_result.ok());
      indexer->Destroy();
    }
  }

  // Test case 11: Index already opened error
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);

    // First open should succeed
    auto open_result1 =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_TRUE(open_result1.ok());

    // Second open should fail
    auto open_result2 =
        indexer->Open(vector_column_params::ReadOptions{true, true});
    ASSERT_FALSE(open_result2.ok());
    ASSERT_EQ(open_result2.message(), "Index already opened");

    indexer->Destroy();
  }

  // Test case 12: Test doc_count() on unopened indexer
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);

    // doc_count() should return -1 for unopened indexer
    ASSERT_EQ(indexer->doc_count(), static_cast<size_t>(-1));
  }

  // Test case 13: Test Merge with empty indexers list
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);

    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    // Merge with empty indexers list should succeed
    auto merge_result = indexer->Merge({}, nullptr);
    ASSERT_TRUE(merge_result.ok());

    indexer->Destroy();
  }

  // Test case 14: Test Merge with same index file path (should be skipped)
  {
    auto indexer1 = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer1);

    ASSERT_TRUE(
        indexer1->Open(vector_column_params::ReadOptions{true, true}).ok());

    // Insert some data
    auto data = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector.data()}};
    ASSERT_TRUE(indexer1->Insert(data, kDocId).ok());

    // Merge with itself (same index file path) should succeed (skipped)
    auto merge_result = indexer1->Merge({indexer1}, nullptr);
    ASSERT_TRUE(merge_result.ok());

    indexer1->Destroy();
  }

  // Test case 15: Test Fetch with non-existent doc_id
  {
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", DataType::VECTOR_FP32, 3, false,
                    std::make_shared<FlatIndexParams>(MetricType::IP)));
    ASSERT_TRUE(indexer);

    ASSERT_TRUE(
        indexer->Open(vector_column_params::ReadOptions{true, true}).ok());

    // Fetch non-existent doc_id should fail
    auto fetch_result = indexer->Fetch(99999);
    ASSERT_FALSE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.error().message(),
              "Failed to fetch vector from index");

    indexer->Destroy();
  }

  // // Test case 16: Test Search with invalid query params (unsupported index
  // // type)
  // {
  //   // Create a custom index params with unsupported index type for query
  //   class UnsupportedQueryIndexParams : public IndexParams {
  //    public:
  //     IndexType type() const override {
  //       return static_cast<IndexType>(999);
  //     }
  //     MetricType metric_type() const override {
  //       return MetricType::IP;
  //     }
  //     QuantizeType quantize_type() const override {
  //       return QuantizeType::UNDEFINED;
  //     }
  //     IndexParams::Ptr clone() const override {
  //       return std::make_shared<UnsupportedQueryIndexParams>();
  //     }
  //   };
  //
  //   FieldSchema unsupported_schema(
  //       "test", DataType::VECTOR_FP32, 3, false,
  //       std::make_shared<UnsupportedQueryIndexParams>());
  //   auto indexer = std::make_shared<VectorColumnIndexer>(index_file_path,
  //                                                        unsupported_schema);
  //   ASSERT_TRUE(indexer);
  //
  //   ASSERT_TRUE(
  //       indexer->Open(vector_column_params::ReadOptions{true, true}).ok());
  //
  //   // Insert some data first
  //   auto data = vector_column_params::VectorData{
  //       vector_column_params::DenseVector{vector.data()}};
  //   ASSERT_TRUE(indexer->Insert(data, kDocId).ok());
  //
  //   // Test search with unsupported index type
  //   auto query = vector_column_params::VectorData{
  //       vector_column_params::DenseVector{.data = vector.data()}};
  //   auto query_params = vector_column_params::QueryParams{
  //       .topk = 10, .filter = nullptr, .fetch_vector = false};
  //
  //   auto search_result = indexer->Search(query, query_params);
  //   ASSERT_FALSE(search_result.has_value());
  //   ASSERT_EQ(search_result.error().message(), "not supported");
  //
  //   indexer->Close();
  // }

  system(cmd_buf);
}

TEST(VectorColumnIndexerTest, CosineMerge) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test_indexer.index"};

  auto del_index_file_func = [&](const std::string file_name) {
    auto cmd_buf = "rm -f " + file_name;
    system(cmd_buf.c_str());
  };

  auto create_indexer_func =
      [&](const IndexParams::Ptr &index_params,
          const std::string &index_name) -> VectorColumnIndexer::Ptr {
    del_index_file_func(index_name);
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_name, FieldSchema("test", DataType::VECTOR_FP32, kDimension,
                                false, index_params));
    if (indexer == nullptr ||
        !indexer->Open(vector_column_params::ReadOptions{true, true}).ok()) {
      return nullptr;
    }
    return indexer;
  };

  auto func = [&](const IndexParams::Ptr &param1,
                  const IndexParams::Ptr &param2,
                  const IndexParams::Ptr &param3) {
    auto indexer1 = create_indexer_func(param1, index_name + "1");
    ASSERT_NE(nullptr, indexer1);
    auto indexer2 = create_indexer_func(param2, index_name + "2");
    ASSERT_NE(nullptr, indexer2);

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 123.0f;
    auto vector_data = vector_column_params::VectorData{
        vector_column_params::DenseVector{vector.data()}};
    ASSERT_TRUE(indexer1->Insert(vector_data, 0).ok());

    vector[1] = 2.0f;
    ASSERT_TRUE(indexer2->Insert(vector_data, 0).ok());
    vector[1] = 3.0f;
    ASSERT_TRUE(indexer2->Insert(vector_data, 1).ok());

    {
      auto fetched_data = indexer1->Fetch(0);
      ASSERT_TRUE(fetched_data.has_value());
      const float *fetched_vector = reinterpret_cast<const float *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      LOG_INFO(
          "indexer1 fetched_vector doc_id:0:%s",
          print_dense_vector(fetched_vector, 3, DataType::VECTOR_FP32).c_str());
      ASSERT_TRUE(fetched_vector[1] - 1.0f < 1e-2);
      ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
    }
    {
      auto fetched_data = indexer2->Fetch(0);
      ASSERT_TRUE(fetched_data.has_value());
      const float *fetched_vector = reinterpret_cast<const float *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      LOG_INFO(
          "indexer2 fetched_vector doc_id:0:%s",
          print_dense_vector(fetched_vector, 3, DataType::VECTOR_FP32).c_str());
      ASSERT_TRUE(fetched_vector[1] - 2.0f < 1e-2);
      ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
    }
    {
      auto fetched_data = indexer2->Fetch(1);
      ASSERT_TRUE(fetched_data.has_value());
      const float *fetched_vector = reinterpret_cast<const float *>(
          std::get<vector_column_params::DenseVectorBuffer>(
              fetched_data->vector_buffer)
              .data.data());
      LOG_INFO(
          "indexer2 fetched_vector doc_id:1:%s",
          print_dense_vector(fetched_vector, 3, DataType::VECTOR_FP32).c_str());
      ASSERT_TRUE(fetched_vector[1] - 3.0f < 1e-2);
      ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
    }

    // {  // test reduce
    //   auto indexer3 = create_indexer_func(param3, index_name + "3");
    //   ASSERT_NE(nullptr, indexer3);
    //   ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, nullptr).ok());
    //   {
    //     auto fetched_data = indexer3->Fetch(0);
    //     ASSERT_TRUE(fetched_data.has_value());
    //     const float *fetched_vector = reinterpret_cast<const float *>(
    //         std::get<vector_column_params::DenseVectorBuffer>(
    //             fetched_data->vector_buffer)
    //             .data.data());
    //     LOG_INFO("indexer3 fetched_vector doc_id:0:%s",
    //              print_dense_vector(fetched_vector, 3,
    //              DataType::VECTOR_FP32)
    //                  .c_str());
    //     ASSERT_TRUE(fetched_vector[1] - 1.0f < 1e-2);
    //     ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
    //   }
    //   {
    //     auto fetched_data = indexer3->Fetch(1);
    //     ASSERT_TRUE(fetched_data.has_value());
    //     const float *fetched_vector = reinterpret_cast<const float *>(
    //         std::get<vector_column_params::DenseVectorBuffer>(
    //             fetched_data->vector_buffer)
    //             .data.data());
    //     LOG_INFO("indexer3 fetched_vector doc_id:1:%s",
    //              print_dense_vector(fetched_vector, 3,
    //              DataType::VECTOR_FP32)
    //                  .c_str());
    //     ASSERT_TRUE(fetched_vector[1] - 2.0f < 1e-2);
    //     ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
    //   }
    //   indexer3->Close();
    //   del_index_file_func(index_name + "3");
    // }
    //
    {  // test reduce with filter
      auto indexer3 = create_indexer_func(param3, index_name + "3");
      ASSERT_NE(nullptr, indexer3);
      auto filter = std::make_shared<EasyIndexFilter>(
          [](uint64_t key) { return key == 0; });
      ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, filter).ok());
      // 0.0 -> x ; 1.0 -> 0 ; 1.1 -> 1
      ASSERT_TRUE(indexer3->doc_count() == 2);
      {
        auto fetched_data = indexer3->Fetch(0);
        ASSERT_TRUE(fetched_data.has_value());
        const float *fetched_vector = reinterpret_cast<const float *>(
            std::get<vector_column_params::DenseVectorBuffer>(
                fetched_data->vector_buffer)
                .data.data());
        LOG_INFO("indexer3 fetched_vector doc_id:0:%s",
                 print_dense_vector(fetched_vector, 3, DataType::VECTOR_FP32)
                     .c_str());
        ASSERT_TRUE(fetched_vector[1] - 2.0f < 1e-2);
        ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
      }

      {
        vector[1] = 3.0f;
        // search with fetch vector
        auto query = vector_column_params::VectorData{
            vector_column_params::DenseVector{vector.data()}};
        auto query_params = vector_column_params::QueryParams{
            .topk = 10, .filter = nullptr, .fetch_vector = true};
        auto results = indexer2->Search(query, query_params);
        ASSERT_TRUE(results.has_value());
        auto vector_results =
            dynamic_cast<VectorIndexResults *>(results.value().get());
        ASSERT_TRUE(vector_results);
        ASSERT_EQ(vector_results->count(), 2);
        auto iter = vector_results->create_iterator();
        ASSERT_TRUE(iter->valid());

        {
          int doc_idx = 0;
          auto query_results_doc = vector_results->docs()[doc_idx];
          LOG_INFO("topk%d pk: %zu", doc_idx, (size_t)query_results_doc.key());
          LOG_INFO("topk%d score: %.10f", doc_idx, query_results_doc.score());
          LOG_INFO("topk%d fetched_vector - reverted:%s", doc_idx,
                   print_dense_vector(
                       vector_results->reverted_vector_list()[doc_idx].data(),
                       kDimension, DataType::VECTOR_FP32)
                       .c_str());
          LOG_INFO("topk%d fetched_vector - original:%s", doc_idx,
                   print_dense_vector(query_results_doc.vector(), kDimension,
                                      DataType::VECTOR_FP16)
                       .c_str());
          ASSERT_TRUE(query_results_doc.score() < 2.01);
          ASSERT_TRUE(query_results_doc.score() > -0.01);
        }
        {
          int doc_idx = 1;
          auto query_results_doc = vector_results->docs()[doc_idx];
          LOG_INFO("topk%d pk: %zu", doc_idx, (size_t)query_results_doc.key());
          LOG_INFO("topk%d score: %.10f", doc_idx, query_results_doc.score());
          LOG_INFO("topk%d fetched_vector - reverted:%s", doc_idx,
                   print_dense_vector(
                       vector_results->reverted_vector_list()[doc_idx].data(),
                       kDimension, DataType::VECTOR_FP32)
                       .c_str());
          LOG_INFO("topk%d fetched_vector - original:%s", doc_idx,
                   print_dense_vector(query_results_doc.vector(), kDimension,
                                      DataType::VECTOR_FP16)
                       .c_str());
          ASSERT_TRUE(query_results_doc.score() < 2.01);
          ASSERT_TRUE(query_results_doc.score() > -0.01);
        }
        // ASSERT_TRUE(vector_results->docs()[0].key() == 1);
      }

      indexer3->Close();
      del_index_file_func(index_name + "3");
    }
    //
    // {  // test reduce with filter in parallel
    //   auto indexer3 = create_indexer_func(param3, index_name + "3");
    //   ASSERT_NE(nullptr, indexer3);
    //   auto filter = std::make_shared<EasyIndexFilter>(
    //       [](uint64_t key) { return key == 0; });
    //   ASSERT_TRUE(indexer3->Merge({indexer1, indexer2}, filter, {3}).ok());
    //
    //   {
    //     auto fetched_data = indexer3->Fetch(0);
    //     ASSERT_TRUE(fetched_data.has_value());
    //     const float *fetched_vector = reinterpret_cast<const float *>(
    //         std::get<vector_column_params::DenseVectorBuffer>(
    //             fetched_data->vector_buffer)
    //             .data.data());
    //     LOG_INFO("indexer3 fetched_vector doc_id:0:%s",
    //              print_dense_vector(fetched_vector, 3,
    //              DataType::VECTOR_FP32)
    //                  .c_str());
    //     ASSERT_TRUE(fetched_vector[1] - 2.0f < 1e-2);
    //     ASSERT_TRUE(fetched_vector[2] - 123.0f < 1);
    //   }
    //   indexer3->Close();
    //   del_index_file_func(index_name + "3");
    // }


    indexer1->Close();
    indexer2->Close();
    del_index_file_func(index_name + "1");
    del_index_file_func(index_name + "2");
  };

  // same index with different quantize type
  {
    LOG_INFO("Merge: same index - FlatIndex with different quantize type");
    auto metric_type = MetricType::COSINE;
    auto param_flat = std::make_shared<FlatIndexParams>(metric_type);
    auto param_flat_fp16 =
        std::make_shared<FlatIndexParams>(metric_type, QuantizeType::FP16);
    auto param_hnsw = std::make_shared<HnswIndexParams>(metric_type, 10, 100);
    auto param_hnsw_fp16 = std::make_shared<HnswIndexParams>(
        metric_type, 10, 100, QuantizeType::FP16);
    // func(param, param_fp16, param_fp16);
    // func(param, param_fp16, param);
    // func(param_fp16, param, param_fp16);
    // func(param_fp16, param, param);
    // func(param_fp16, param_fp16, param_fp16);
    func(param_hnsw_fp16, param_flat_fp16, param_flat_fp16);
  }
}

TEST(VectorColumnIndexerTest, Refiner) {
  const std::string kIndexFilePath = "test_indexer.index";
  const int kDim = 20;
  const int kCount = 20;  // can't set too large, or the qunatization error
                          // will be too large due to float's precision
  const int kTopk = 10;

  auto del_index_file_func = [&](const std::string &file_name) {
    auto cmd_buf = "rm -f " + file_name;
    system(cmd_buf.c_str());
  };

  auto create_indexer_func =
      [&](const IndexParams::Ptr &index_params,
          const std::string &index_file_path,
          DataType data_type) -> VectorColumnIndexer::Ptr {
    del_index_file_func(index_file_path);
    auto indexer = std::make_shared<VectorColumnIndexer>(
        index_file_path,
        FieldSchema("test", data_type, kDim, false, index_params));
    if (indexer == nullptr ||
        !indexer->Open(vector_column_params::ReadOptions{true, true}).ok()) {
      return nullptr;
    }
    return indexer;
  };

  auto func = [&](const IndexParams::Ptr &index_params,
                  const IndexParams::Ptr &reference_index_params,
                  DataType data_type) {
    auto indexer = create_indexer_func(index_params, kIndexFilePath, data_type);
    if (indexer == nullptr) {
      return;
    }
    auto reference_indexer = create_indexer_func(
        reference_index_params, kIndexFilePath + "_reference", data_type);
    if (reference_indexer == nullptr) {
      return;
    }

    // insert
    for (int i = 0; i < kCount; ++i) {
      auto buffer = create_dense_vector(kDim, data_type, i, kCount, 0.1f);
      // print_dense_vector(buffer.data.data(), kDim, data_type);
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{buffer.data.data()}};
      ASSERT_TRUE(indexer->Insert(data, i).ok());
      ASSERT_TRUE(reference_indexer->Insert(data, i).ok());
    }

    // query
    for (int i = 0; i < kCount; ++i) {
      auto buffer = create_dense_vector(kDim, data_type, i, kCount, 0.3f);
      auto data = vector_column_params::VectorData{
          vector_column_params::DenseVector{buffer.data.data()}};
      ;
      auto query_params = vector_column_params::QueryParams{
          .topk = kTopk,
          .filter = nullptr,
          .fetch_vector = true,
          .query_params = std::make_shared<zvec::HnswQueryParams>(100),
          .refiner_param = std::make_shared<vector_column_params::RefinerParam>(
              vector_column_params::RefinerParam{
                  .scale_factor_ = 10,
                  .reference_indexer = reference_indexer})};
      auto results = indexer->Search(data, query_params);
      ASSERT_TRUE(results.has_value());
      auto vector_results =
          dynamic_cast<VectorIndexResults *>(results.value().get());
      ASSERT_TRUE(vector_results);
      ASSERT_EQ(vector_results->count(), kTopk);
      auto iter = vector_results->create_iterator();
      LOG_INFO("===query pk: %d", i);
      LOG_INFO("query_vector:%s",
               print_dense_vector(buffer.data.data(), kDim, data_type).c_str());
    }
    indexer->Destroy();
  };

  LOG_INFO(
      "Test FlatIndexParams(MetricType::IP), VECTOR_FP32, "
      "QuantizeType::FP16");

  func(std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                         QuantizeType::FP16),
       std::make_shared<FlatIndexParams>(MetricType::IP),
       DataType::VECTOR_FP32);

  func(std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16),
       std::make_shared<FlatIndexParams>(MetricType::IP),
       DataType::VECTOR_FP32);

  LOG_INFO(
      "Test FlatIndexParams(MetricType::MIPSL2), VECTOR_FP32, "
      "QuantizeType::FP16");

  func(std::make_shared<HnswIndexParams>(MetricType::MIPSL2, 10, 100,
                                         QuantizeType::FP16),
       std::make_shared<FlatIndexParams>(MetricType::IP),
       DataType::VECTOR_FP32);

  func(
      std::make_shared<FlatIndexParams>(MetricType::MIPSL2, QuantizeType::FP16),
      std::make_shared<FlatIndexParams>(MetricType::IP), DataType::VECTOR_FP32);

  LOG_INFO(
      "Test FlatIndexParams(MetricType::COSINE), VECTOR_FP32, "
      "QuantizeType::FP16");
  func(
      std::make_shared<FlatIndexParams>(MetricType::COSINE, QuantizeType::FP16),
      std::make_shared<FlatIndexParams>(MetricType::COSINE),
      DataType::VECTOR_FP32);

  LOG_INFO(
      "Test FlatIndexParams(MetricType::L2), VECTOR_FP32, "
      "QuantizeType::Int8");
  func(std::make_shared<FlatIndexParams>(MetricType::L2, QuantizeType::INT8),
       std::make_shared<FlatIndexParams>(MetricType::L2),
       DataType::VECTOR_FP32);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif