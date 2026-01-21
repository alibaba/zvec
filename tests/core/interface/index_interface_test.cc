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
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <gtest/gtest.h>
#include "zvec/ailego/buffer/buffer_manager.h"
#include "zvec/core/interface/index.h"
#include "zvec/core/interface/index_factory.h"
#include "zvec/core/interface/index_param.h"
#include "zvec/core/interface/index_param_builders.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core_interface;

TEST(IndexInterface, General) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test.index"};
  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_name.c_str());

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    system(cmd_buf);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);


    index->Open(index_name, {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 2.0f;
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ASSERT_TRUE(0 == index->Add(vector_data, 233));
    ASSERT_TRUE(0 == index->Train());

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{vector.data()};
    index->Search(query, query_param, &result);
    ASSERT_EQ(1, result.doc_list_.size());
    ASSERT_EQ(233, result.doc_list_[0].key());
    ASSERT_FLOAT_EQ(5.0f, result.doc_list_[0].score());
    if (query_param->fetch_vector) {
      auto &doc = result.doc_list_[0];
      if (result.reverted_vector_list_.size() != 0) {
        // cosine metric or bf16 quantizer
        ASSERT_EQ(1, result.reverted_vector_list_.size());
        auto reverted_vector = reinterpret_cast<const float *>(
            result.reverted_vector_list_[0].data());
        ASSERT_FLOAT_EQ(1.0f, reverted_vector[1]);
        ASSERT_FLOAT_EQ(2.0f, reverted_vector[2]);
      } else {
        auto vector = reinterpret_cast<const float *>(doc.vector());
        ASSERT_FLOAT_EQ(1.0f, vector[1]);
        ASSERT_FLOAT_EQ(2.0f, vector[2]);
      }
    }

    vector[1] = 0;
    vector[2] = 0;
    VectorDataBuffer fetched_vector_data;
    ASSERT_TRUE(0 == index->Fetch(233, &fetched_vector_data));
    float *fetched_vector = reinterpret_cast<float *>(
        std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
            .data.data());
    ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
    ASSERT_FLOAT_EQ(2.0f, fetched_vector[2]);
    index->Close();
    system(cmd_buf);
  };


  auto param = FlatIndexParamBuilder()
                   .WithMetricType(MetricType::kInnerProduct)
                   .WithDataType(DataType::DT_FP32)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .Build();
  func(param,
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
  func(FlatIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(HNSWIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithEFConstruction(100)
           .Build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(HNSWIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithEFConstruction(100)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(IVFIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithNList(10)
           .Build(),
       IVFQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
  func(IVFIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithNList(10)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       IVFQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
}

TEST(IndexInterface, BufferGeneral) {
  zvec::ailego::BufferManager::Instance().init(100 * 1024 * 1024, 1);
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test.index"};
  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s*", index_name.c_str());

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    std::string real_index_name = index_name;
    system(cmd_buf);
    auto write_index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, write_index);

    write_index->Open(real_index_name,
                      {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 2.0f;
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ASSERT_TRUE(0 == write_index->Add(vector_data, 233));
    write_index->Close();

    auto read_index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, read_index);
    read_index->Open(real_index_name,
                     {StorageOptions::StorageType::kBufferPool, false});

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{vector.data()};
    read_index->Search(query, query_param, &result);
    ASSERT_EQ(1, result.doc_list_.size());
    ASSERT_EQ(233, result.doc_list_[0].key());
    ASSERT_FLOAT_EQ(5.0f, result.doc_list_[0].score());
    if (query_param->fetch_vector) {
      auto &doc = result.doc_list_[0];
      if (result.reverted_vector_list_.size() != 0) {
        // cosine metric or bf16 quantizer
        ASSERT_EQ(1, result.reverted_vector_list_.size());
        auto reverted_vector = reinterpret_cast<const float *>(
            result.reverted_vector_list_[0].data());
        ASSERT_FLOAT_EQ(1.0f, reverted_vector[1]);
        ASSERT_FLOAT_EQ(2.0f, reverted_vector[2]);
      } else {
        auto vector = reinterpret_cast<const float *>(doc.vector());
        ASSERT_FLOAT_EQ(1.0f, vector[1]);
        ASSERT_FLOAT_EQ(2.0f, vector[2]);
      }
    }

    vector[1] = 0;
    vector[2] = 0;
    VectorDataBuffer fetched_vector_data;
    ASSERT_TRUE(0 == read_index->Fetch(233, &fetched_vector_data));
    float *fetched_vector = reinterpret_cast<float *>(
        std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
            .data.data());
    ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
    ASSERT_FLOAT_EQ(2.0f, fetched_vector[2]);
    read_index->Close();
    system(cmd_buf);
  };


  auto param = FlatIndexParamBuilder()
                   .WithMetricType(MetricType::kInnerProduct)
                   .WithDataType(DataType::DT_FP32)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .Build();
  func(param,
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
  func(FlatIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(HNSWIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithEFConstruction(100)
           .Build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(HNSWIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithDimension(kDimension)
           .WithIsSparse(false)
           .WithEFConstruction(100)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
}


TEST(IndexInterface, SparseGeneral) {
  constexpr uint32_t kSparseCount = 3;
  const std::string index_name{"test.index"};
  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_name.c_str());

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    system(cmd_buf);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);


    index->Open(index_name, {StorageOptions::StorageType::kMMAP, true});

    std::vector<uint32_t> indices(kSparseCount);
    std::vector<float> values(kSparseCount);
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      indices[i] = i;
      values[i] = i;
    }

    VectorData vector_data{
        SparseVector{kSparseCount, indices.data(), values.data()}};
    ASSERT_TRUE(0 == index->Add(vector_data, 233));


    SearchResult result;
    VectorData query = {
        SparseVector{kSparseCount, indices.data(), values.data()}};
    index->Search(query, query_param, &result);
    ASSERT_EQ(1, result.doc_list_.size());
    ASSERT_EQ(233, result.doc_list_[0].key());
    ASSERT_FLOAT_EQ(5.0f, result.doc_list_[0].score());

    if (query_param->fetch_vector) {
      auto &sparse_doc = result.doc_list_[0].sparse_doc();
      auto sparse_indices = reinterpret_cast<const uint32_t *>(
          sparse_doc.sparse_indices().data());
      for (uint32_t i = 0; i < kSparseCount; ++i) {
        ASSERT_EQ(i, sparse_indices[i]);
      }
      if (!result.reverted_sparse_values_list_.empty()) {
        ASSERT_EQ(1, result.reverted_sparse_values_list_.size());
        auto reverted_sparse_values = reinterpret_cast<const float *>(
            result.reverted_sparse_values_list_[0].data());
        for (uint32_t i = 0; i < kSparseCount; ++i) {
          ASSERT_EQ(i, reverted_sparse_values[i]);
        }
      } else {
        auto sparse_values =
            reinterpret_cast<const float *>(sparse_doc.sparse_values().data());
        for (uint32_t i = 0; i < kSparseCount; ++i) {
          ASSERT_EQ(i, sparse_values[i]);
        }
      }
    }

    values[1] = 0;
    values[2] = 0;
    VectorDataBuffer fetched_vector_data;
    ASSERT_TRUE(0 == index->Fetch(233, &fetched_vector_data));
    const SparseVectorBuffer &sparse_vector_buffer =
        std::get<SparseVectorBuffer>(fetched_vector_data.vector_buffer);
    const uint32_t *fetched_indices =
        reinterpret_cast<const uint32_t *>(sparse_vector_buffer.indices.data());
    const float *fetched_values =
        reinterpret_cast<const float *>(sparse_vector_buffer.values.data());
    ASSERT_EQ(kSparseCount, sparse_vector_buffer.count);
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      ASSERT_EQ(i, fetched_indices[i]);
      ASSERT_EQ(i, fetched_values[i]);
    }
    index->Close();
    system(cmd_buf);
  };


  auto param = FlatIndexParamBuilder()
                   .WithMetricType(MetricType::kInnerProduct)
                   .WithDataType(DataType::DT_FP32)
                   .WithIsSparse(true)
                   .Build();
  // func(param, FlatQueryParam{{.topk = 10, .fetch_vector = true}});
  func(FlatIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithIsSparse(true)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(HNSWIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithIsSparse(true)
           .WithEFConstruction(100)
           .Build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(HNSWIndexParamBuilder()
           .WithMetricType(MetricType::kInnerProduct)
           .WithDataType(DataType::DT_FP32)
           .WithIsSparse(true)
           .WithEFConstruction(100)
           .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
           .Build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
}


TEST(IndexInterface, Merge) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test.index"};

  auto del_index_file_func = [&](const std::string file_name) {
    auto cmd_buf = "rm -f " + file_name;
    system(cmd_buf.c_str());
  };

  auto create_index_func =
      [&](const BaseIndexParam::Pointer &param,
          const std::string &index_name) -> Index::Pointer {
    del_index_file_func(index_name);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    if (index == nullptr ||
        0 != index->Open(index_name,
                         {StorageOptions::StorageType::kMMAP, true})) {
      return nullptr;
    }
    return index;
  };

  auto func = [&](const BaseIndexParam::Pointer &param_target,
                  const BaseIndexParam::Pointer &param_source) {
    auto index1 = create_index_func(param_source, index_name + "1");
    ASSERT_NE(nullptr, index1);
    auto index2 = create_index_func(param_source, index_name + "2");
    ASSERT_NE(nullptr, index2);


    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 123.0f;
    VectorData vector_data{DenseVector{vector.data()}};
    ASSERT_TRUE(0 == index1->Add(vector_data, 0));

    vector[1] = 2.0f;
    ASSERT_TRUE(0 == index2->Add(vector_data, 0));
    vector[1] = 3.0f;
    ASSERT_TRUE(0 == index2->Add(vector_data, 1));

    {
      VectorDataBuffer fetched_vector_data;
      ASSERT_TRUE(0 == index1->Fetch(0, &fetched_vector_data));
      float *fetched_vector = reinterpret_cast<float *>(
          std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
              .data.data());
      ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }
    {
      VectorDataBuffer fetched_vector_data;
      ASSERT_TRUE(0 == index2->Fetch(0, &fetched_vector_data));
      float *fetched_vector = reinterpret_cast<float *>(
          std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
              .data.data());
      ASSERT_FLOAT_EQ(2.0f, fetched_vector[1]);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }
    {
      VectorDataBuffer fetched_vector_data;
      ASSERT_TRUE(0 == index2->Fetch(1, &fetched_vector_data));
      float *fetched_vector = reinterpret_cast<float *>(
          std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
              .data.data());
      ASSERT_FLOAT_EQ(3.0f, fetched_vector[1]);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }

    {  // test reduce
      auto index3 = create_index_func(param_target, index_name + "3");
      ASSERT_NE(nullptr, index3);
      ASSERT_TRUE(0 == index3->Merge({index1, index2}, IndexFilter()));
      ASSERT_TRUE(3 == index3->GetDocCount());
      {
        VectorDataBuffer fetched_vector_data;
        ASSERT_TRUE(0 == index3->Fetch(0, &fetched_vector_data));
        float *fetched_vector = reinterpret_cast<float *>(
            std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
                .data.data());
        ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
        ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
      }
      {
        VectorDataBuffer fetched_vector_data;
        ASSERT_TRUE(0 == index3->Fetch(1, &fetched_vector_data));
        float *fetched_vector = reinterpret_cast<float *>(
            std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
                .data.data());
        ASSERT_FLOAT_EQ(2.0f, fetched_vector[1]);
        ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
      }
      index3->Close();
      del_index_file_func(index_name + "3");
    }

    {  // test reduce with filter
      auto index3 = create_index_func(param_target, index_name + "3");
      ASSERT_NE(nullptr, index3);
      auto filter = IndexFilter();
      filter.set([](uint64_t key) { return key == 0; });  // TODO: uint32?
      ASSERT_TRUE(0 == index3->Merge({index1, index2}, filter));
      ASSERT_TRUE(2 == index3->GetDocCount());
      {
        VectorDataBuffer fetched_vector_data;
        ASSERT_TRUE(0 == index3->Fetch(0, &fetched_vector_data));
        float *fetched_vector = reinterpret_cast<float *>(
            std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
                .data.data());
        ASSERT_FLOAT_EQ(2.0f, fetched_vector[1]);
        ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
      }
      index3->Close();
      del_index_file_func(index_name + "3");
    }

    index1->Close();
    index2->Close();
    del_index_file_func(index_name + "1");
    del_index_file_func(index_name + "2");
  };

  // same index
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(kDimension)
                     .WithIsSparse(false)
                     .Build();
    func(param, param);
  }
  {
    auto param = HNSWIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(kDimension)
                     .WithIsSparse(false)
                     .Build();
    func(param, param);
  }

  // different index
  {
    auto param_flat = FlatIndexParamBuilder()
                          .WithMetricType(MetricType::kInnerProduct)
                          .WithDataType(DataType::DT_FP32)
                          .WithDimension(kDimension)
                          .WithIsSparse(false)
                          .Build();
    auto param_hnsw = HNSWIndexParamBuilder()
                          .WithMetricType(MetricType::kInnerProduct)
                          .WithDataType(DataType::DT_FP32)
                          .WithDimension(kDimension)
                          .WithIsSparse(false)
                          .Build();
    func(param_flat, param_hnsw);
    func(param_hnsw, param_flat);
  }
}


TEST(IndexInterface, Serialize) {
  {
    std::cout << "\n\n----flat index----" << std::endl;
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .WithQuantizerParam(QuantizerParam{QuantizerType::kFP16})
                     .Build();

    std::cout << "flat index -- omit=true: " << param->SerializeToJson(true)
              << std::endl;
    std::cout << "omit=false: " << param->SerializeToJson() << std::endl;

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(param->SerializeToJson());
    ASSERT_NE(nullptr, deserialized_param.get());


    std::cout << "serialize then de then se:"
              << deserialized_param->SerializeToJson() << std::endl;

    ASSERT_TRUE(deserialized_param->SerializeToJson() ==
                param->SerializeToJson());
    ASSERT_TRUE(deserialized_param->SerializeToJson(true) ==
                param->SerializeToJson(true));
  }

  {
    std::cout << "\n\n----hnsw index----" << std::endl;
    auto param = HNSWIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .WithQuantizerParam(QuantizerParam{QuantizerType::kFP16})
                     .Build();

    std::cout << "hnsw index -- omit=true: " << param->SerializeToJson(true)
              << std::endl;
    std::cout << "hnsw index -- omit=false: " << param->SerializeToJson()
              << std::endl;

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(param->SerializeToJson());
    ASSERT_NE(nullptr, deserialized_param.get());

    std::cout << "serialize then de then se:"
              << deserialized_param->SerializeToJson() << std::endl;


    ASSERT_TRUE(deserialized_param->SerializeToJson() ==
                param->SerializeToJson());
    ASSERT_TRUE(deserialized_param->SerializeToJson(true) ==
                param->SerializeToJson(true));
  }

  {
    std::cout << "\n\n----flat query----" << std::endl;
    auto param =
        FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build();
    std::cout << "flat query -- omit=true: "
              << IndexFactory::QueryParamSerializeToJson(*param, true)
              << std::endl;
    std::cout << "flat query -- omit=false: "
              << IndexFactory::QueryParamSerializeToJson(*param) << std::endl;

    auto deserialized_param =
        IndexFactory::QueryParamDeserializeFromJson<FlatQueryParam>(
            IndexFactory::QueryParamSerializeToJson(*param));
    ASSERT_NE(nullptr, deserialized_param.get());

    std::cout << "serialize then de then se:"
              << IndexFactory::QueryParamSerializeToJson(*deserialized_param)
              << std::endl;

    ASSERT_TRUE(IndexFactory::QueryParamSerializeToJson(*deserialized_param) ==
                IndexFactory::QueryParamSerializeToJson(*param));
  }

  {
    std::cout << "\n\n----hnsw query----" << std::endl;
    auto param = HNSWQueryParamBuilder()
                     .with_topk(10)
                     .with_fetch_vector(true)
                     .with_ef_search(20)
                     .build();
    std::cout << "hnsw query -- omit=true: "
              << IndexFactory::QueryParamSerializeToJson(*param, true)
              << std::endl;
    std::cout << "hnsw query -- omit=false: "
              << IndexFactory::QueryParamSerializeToJson(*param, false)
              << std::endl;

    auto deserialized_param =
        IndexFactory::QueryParamDeserializeFromJson<HNSWQueryParam>(
            IndexFactory::QueryParamSerializeToJson(*param));
    ASSERT_NE(nullptr, deserialized_param.get());

    std::cout << "serialize then de then se:"
              << IndexFactory::QueryParamSerializeToJson(*deserialized_param)
              << std::endl;

    ASSERT_TRUE(IndexFactory::QueryParamSerializeToJson(*deserialized_param) ==
                IndexFactory::QueryParamSerializeToJson(*param));
  }
}

TEST(IndexInterface, Failure) {
  // Test unsupported index type
  {
    auto param = std::make_shared<BaseIndexParam>(IndexType::kIVF);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test unsupported metric type
  {
    auto param =
        FlatIndexParamBuilder()
            .WithMetricType(MetricType::kNone)  // L2 not supported for sparse
            .WithDataType(DataType::DT_FP32)
            .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test unsupported metric type for sparse index
  {
    auto param =
        FlatIndexParamBuilder()
            .WithMetricType(MetricType::kL2sq)  // L2 not supported for sparse
            .WithDataType(DataType::DT_FP32)
            .WithIsSparse(true)
            .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // // Test unsupported quantizer type
  // {
  //   auto param = FlatIndexParamBuilder()
  //                    .WithMetricType(MetricType::kInnerProduct)
  //                    .WithDataType(DataType::DT_INT4)
  //                    .WithDimension(64)
  //                    .WithIsSparse(false)
  //                    .WithQuantizerParam(
  //                        QuantizerParam(QuantizerType::kInt8))  //
  //                        Unsupported
  //                    .Build();
  //   auto index = IndexFactory::CreateAndInitIndex(*param);
  //   ASSERT_EQ(nullptr, index);
  // }
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(true)
                     .WithQuantizerParam(
                         QuantizerParam(QuantizerType::kInt8))  // Unsupported
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test unsupported data type for cosine metric
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kCosine)
                     .WithDataType(DataType::DT_INT8)  // Unsupported for cosine
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test invalid storage type
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    StorageOptions invalid_storage;
    invalid_storage.type = StorageOptions::StorageType::kNone;  // Unsupported
    int ret = index->Open("test.index", invalid_storage);
    ASSERT_NE(0, ret);
  }

  // Test invalid vector data type for dense operations
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->Open("test.index", {StorageOptions::StorageType::kMMAP, true});

    // Try to add sparse vector to dense index
    std::vector<uint32_t> indices = {0, 1, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    VectorData sparse_vector_data{
        SparseVector{3, indices.data(), values.data()}};

    int ret = index->Add(sparse_vector_data, 1);
    ASSERT_NE(0, ret);

    index->Close();
    system("rm -f test.index");
  }

  // Test invalid vector data type for sparse operations
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithIsSparse(true)
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->Open("test.index", {StorageOptions::StorageType::kMMAP, true});

    // Try to add dense vector to sparse index
    std::vector<float> vector(64, 1.0f);
    VectorData dense_vector_data{DenseVector{vector.data()}};

    int ret = index->Add(dense_vector_data, 1);
    ASSERT_NE(0, ret);

    index->Close();
    system("rm -f test.index");
  }

  // Test fetch non-existent document
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->Open("test.index", {StorageOptions::StorageType::kMMAP, true});

    VectorDataBuffer fetched_vector_data;
    int ret = index->Fetch(999, &fetched_vector_data);  // Non-existent doc_id
    ASSERT_NE(0, ret);

    index->Close();
    system("rm -f test.index");
  }

  // Test search with invalid vector data
  {
    auto param = FlatIndexParamBuilder()
                     .WithMetricType(MetricType::kInnerProduct)
                     .WithDataType(DataType::DT_FP32)
                     .WithDimension(64)
                     .WithIsSparse(false)
                     .Build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->Open("test.index", {StorageOptions::StorageType::kMMAP, true});

    // Add a vector first
    std::vector<float> vector(64, 1.0f);
    VectorData vector_data{DenseVector{vector.data()}};
    ASSERT_EQ(0, index->Add(vector_data, 1));

    // Try to search with sparse vector in dense index
    std::vector<uint32_t> indices = {0, 1, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    VectorData sparse_query{SparseVector{3, indices.data(), values.data()}};

    SearchResult result;
    FlatQueryParam::Pointer query_param =
        FlatQueryParamBuilder().with_topk(10).with_fetch_vector(false).build();
    int ret = index->Search(sparse_query, query_param, &result);
    ASSERT_NE(0, ret);

    index->Close();
    system("rm -f test.index");
  }

  // Test merge with invalid write concurrency
  {
    auto param1 = FlatIndexParamBuilder()
                      .WithMetricType(MetricType::kInnerProduct)
                      .WithDataType(DataType::DT_FP32)
                      .WithDimension(64)
                      .WithIsSparse(false)
                      .Build();
    auto index1 = IndexFactory::CreateAndInitIndex(*param1);
    ASSERT_NE(nullptr, index1);
    index1->Open("test1.index", {StorageOptions::StorageType::kMMAP, true});

    auto param2 = FlatIndexParamBuilder()
                      .WithMetricType(MetricType::kInnerProduct)
                      .WithDataType(DataType::DT_FP32)
                      .WithDimension(64)
                      .WithIsSparse(false)
                      .Build();
    auto index2 = IndexFactory::CreateAndInitIndex(*param2);
    ASSERT_NE(nullptr, index2);
    index2->Open("test2.index", {StorageOptions::StorageType::kMMAP, true});

    auto param3 = FlatIndexParamBuilder()
                      .WithMetricType(MetricType::kInnerProduct)
                      .WithDataType(DataType::DT_FP32)
                      .WithDimension(64)
                      .WithIsSparse(false)
                      .Build();
    auto index3 = IndexFactory::CreateAndInitIndex(*param3);
    ASSERT_NE(nullptr, index3);
    index3->Open("test3.index", {StorageOptions::StorageType::kMMAP, true});

    MergeOptions invalid_options;
    invalid_options.write_concurrency = 0;  // Invalid: must be > 0

    int ret = index3->Merge({index1, index2}, IndexFilter(), invalid_options);
    ASSERT_NE(0, ret);

    index1->Close();
    index2->Close();
    index3->Close();
    system("rm -f test1.index test2.index test3.index");
  }
}

TEST(IndexInterface, SerializeFailure) {
  // Test invalid JSON deserialization
  {
    std::string invalid_json = "invalid json string";
    auto param = IndexFactory::DeserializeIndexParamFromJson(invalid_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test JSON with invalid enum value
  {
    std::string invalid_enum_json = R"({
      "index_type": "kInvalidType",
      "metric_type": "kL2",
      "dimension": 64,
      "is_sparse": false,
      "data_type": "DT_FP32"
    })";
    auto param = IndexFactory::DeserializeIndexParamFromJson(invalid_enum_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test JSON with invalid field type
  {
    std::string invalid_type_json = R"({
      "index_type": "kFlat",
      "metric_type": "kL2",
      "dimension": "not_a_number",
      "is_sparse": false,
      "data_type": "DT_FP32"
    })";
    auto param = IndexFactory::DeserializeIndexParamFromJson(invalid_type_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test JSON with invalid field type
  {
    std::string invalid_type_json = R"({
      "index_type": "kHNSW",
      "metric_type": "kL2",
      "dimension": 1,
      "is_sparse": "false",
      "data_type": "DT_FP32"
    })";
    auto param = IndexFactory::DeserializeIndexParamFromJson(invalid_type_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test unsupported index_type
  {
    std::string wrong_type_json = R"({
      "index_type": "kNone",
      "metric_type": "kL2",
      "dimension": 64,
      "is_sparse": false,
      "data_type": "DT_FP32"
    })";
    auto param = IndexFactory::DeserializeIndexParamFromJson(wrong_type_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test QueryParam deserialization with invalid JSON
  {
    std::string invalid_json = "invalid json";
    auto param = IndexFactory::QueryParamDeserializeFromJson<FlatQueryParam>(
        invalid_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test QueryParam deserialization with invalid enum
  {
    std::string invalid_enum_json = R"({
      "index_type": "kInvalidType",
      "topk": 10,
      "fetch_vector": false,
      "radius": 0.0,
      "is_linear": false
    })";
    auto param = IndexFactory::QueryParamDeserializeFromJson<FlatQueryParam>(
        invalid_enum_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test QueryParam deserialization with invalid field type
  {
    std::string invalid_type_json = R"({
      "index_type": "kFlat",
      "topk": "not_a_number",
      "fetch_vector": false,
      "radius": 0.0,
      "is_linear": false
    })";
    auto param = IndexFactory::QueryParamDeserializeFromJson<FlatQueryParam>(
        invalid_type_json);
    ASSERT_EQ(nullptr, param);
  }

  // Test HNSWQueryParam deserialization with invalid field type
  {
    std::string invalid_type_json = R"({
      "index_type": "kHNSW",
      "topk": 10,
      "fetch_vector": false,
      "radius": 0.0,
      "is_linear": false,
      "ef_search": "not_a_number"
    })";
    auto param = IndexFactory::QueryParamDeserializeFromJson<HNSWQueryParam>(
        invalid_type_json);
    ASSERT_EQ(nullptr, param);
  }
}

TEST(IndexInterface, Score) {
  const std::string index_file_path = "test_indexer.index";
  const int kTopk = 10;
  constexpr uint32_t kDocId1 = 2345;
  constexpr uint32_t kDocId2 = 5432;
  auto vector1 = std::vector<float>{3.0f, 4.0f, 5.0f};
  auto vector2 = std::vector<float>{1.0f, 20.0f, 3.0f};
  auto vector_id_map = std::unordered_map<uint32_t, std::vector<float>>{
      {kDocId1, vector1},
      {kDocId2, vector2},
  };
  auto sparse_indices = std::vector<uint32_t>{0, 1, 2};
  auto query_vector = std::vector<float>{1.0f, 2.0f, 3.0f};

  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_file_path.c_str());
  system(cmd_buf);

  auto check_score = [&](const SearchResult &result, MetricType metric_type) {
    ASSERT_EQ(result.doc_list_.size(), 2);

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
      for (int i = 0; i < v1.size(); ++i) {
        ret += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      }
      return ret;
    };

    std::function<float(const std::vector<float> &, const std::vector<float> &)>
        score_func;

    switch (metric_type) {
      case MetricType::kInnerProduct:
        score_func = inner_produce_score_func;
        break;
      case MetricType::kCosine:
        score_func = cosine_score_func;
        break;
      case MetricType::kL2sq:
        score_func = l2_score_func;
        break;
      default:
        ASSERT_TRUE(false);
    }

    // Iterate over doc_list_ and check scores
    ASSERT_GE(result.doc_list_.size(), 2);
    printf("result.doc_list_[0].score() top1: %f\n",
           result.doc_list_[0].score());
    printf(
        "score_func(vector_id_map[result.doc_list_[0].key()], query_vector): "
        "%f\n",
        score_func(vector_id_map[result.doc_list_[0].key()], query_vector));
    ASSERT_TRUE(std::abs(result.doc_list_[0].score() -
                         score_func(vector_id_map[result.doc_list_[0].key()],
                                    query_vector)) < 1e-2);
    printf("result.doc_list_[1].score() top2: %f\n",
           result.doc_list_[1].score());
    printf(
        "score_func(vector_id_map[result.doc_list_[1].key()], query_vector): "
        "%f\n",
        score_func(vector_id_map[result.doc_list_[1].key()], query_vector));
    ASSERT_TRUE(std::abs(result.doc_list_[1].score() -
                         score_func(vector_id_map[result.doc_list_[1].key()],
                                    query_vector)) < 1e-2);
  };

  auto dense_func = [&](const BaseIndexParam::Pointer &param,
                        const BaseIndexQueryParam::Pointer query_param,
                        MetricType metric_type) {
    system(cmd_buf);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->Open(index_file_path, {StorageOptions::StorageType::kMMAP, true});

    VectorData vector_data1;
    vector_data1.vector = DenseVector{vector1.data()};
    ASSERT_EQ(0, index->Add(vector_data1, kDocId1));

    VectorData vector_data2;
    vector_data2.vector = DenseVector{vector2.data()};
    ASSERT_EQ(0, index->Add(vector_data2, kDocId2));

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{query_vector.data()};
    index->Search(query, query_param, &result);

    check_score(result, metric_type);

    index->Close();
    system(cmd_buf);
  };

  auto sparse_func = [&](const BaseIndexParam::Pointer &param,
                         const BaseIndexQueryParam::Pointer query_param,
                         MetricType metric_type) {
    system(cmd_buf);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->Open(index_file_path, {StorageOptions::StorageType::kMMAP, true});

    VectorData vector_data1;
    vector_data1.vector =
        SparseVector{3, reinterpret_cast<const void *>(sparse_indices.data()),
                     vector1.data()};
    ASSERT_EQ(0, index->Add(vector_data1, kDocId1));

    VectorData vector_data2;
    vector_data2.vector =
        SparseVector{3, reinterpret_cast<const void *>(sparse_indices.data()),
                     vector2.data()};
    ASSERT_EQ(0, index->Add(vector_data2, kDocId2));

    SearchResult result;
    VectorData query;
    query.vector =
        SparseVector{3, reinterpret_cast<const void *>(sparse_indices.data()),
                     query_vector.data()};
    index->Search(query, query_param, &result);

    check_score(result, metric_type);

    index->Close();
    system(cmd_buf);
  };

  constexpr uint32_t kDimension = 3;

  LOG_INFO("Test DenseVector, MetricType::kInnerProduct");
  dense_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kInnerProduct)
          .WithDataType(DataType::DT_FP32)
          .WithDimension(kDimension)
          .WithIsSparse(false)
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  dense_func(HNSWIndexParamBuilder()
                 .WithMetricType(MetricType::kInnerProduct)
                 .WithDataType(DataType::DT_FP32)
                 .WithDimension(kDimension)
                 .WithIsSparse(false)
                 .WithEFConstruction(100)
                 .Build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kInnerProduct);

  LOG_INFO("Test DenseVector, MetricType::kInnerProduct, QuantizerType::kFP16");
  dense_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kInnerProduct)
          .WithDataType(DataType::DT_FP32)
          .WithDimension(kDimension)
          .WithIsSparse(false)
          .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  dense_func(HNSWIndexParamBuilder()
                 .WithMetricType(MetricType::kInnerProduct)
                 .WithDataType(DataType::DT_FP32)
                 .WithDimension(kDimension)
                 .WithIsSparse(false)
                 .WithEFConstruction(100)
                 .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
                 .Build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kInnerProduct);

  LOG_INFO("Test DenseVector, MetricType::kCosine");
  dense_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kCosine)
          .WithDataType(DataType::DT_FP32)
          .WithDimension(kDimension)
          .WithIsSparse(false)
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kCosine);
  dense_func(HNSWIndexParamBuilder()
                 .WithMetricType(MetricType::kCosine)
                 .WithDataType(DataType::DT_FP32)
                 .WithDimension(kDimension)
                 .WithIsSparse(false)
                 .WithEFConstruction(100)
                 .Build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kCosine);

  LOG_INFO("Test DenseVector, MetricType::kCosine, QuantizerType::kFP16");
  dense_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kCosine)
          .WithDataType(DataType::DT_FP32)
          .WithDimension(kDimension)
          .WithIsSparse(false)
          .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kCosine);
  dense_func(HNSWIndexParamBuilder()
                 .WithMetricType(MetricType::kCosine)
                 .WithDataType(DataType::DT_FP32)
                 .WithDimension(kDimension)
                 .WithIsSparse(false)
                 .WithEFConstruction(100)
                 .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
                 .Build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kCosine);

  LOG_INFO("Test DenseVector, MetricType::kL2sq");
  dense_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kL2sq)
          .WithDataType(DataType::DT_FP32)
          .WithDimension(kDimension)
          .WithIsSparse(false)
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kL2sq);
  dense_func(HNSWIndexParamBuilder()
                 .WithMetricType(MetricType::kL2sq)
                 .WithDataType(DataType::DT_FP32)
                 .WithDimension(kDimension)
                 .WithIsSparse(false)
                 .WithEFConstruction(100)
                 .Build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kL2sq);

  LOG_INFO("Test DenseVector, MetricType::kL2sq, QuantizerType::kFP16");
  dense_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kL2sq)
          .WithDataType(DataType::DT_FP32)
          .WithDimension(kDimension)
          .WithIsSparse(false)
          .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kL2sq);
  dense_func(HNSWIndexParamBuilder()
                 .WithMetricType(MetricType::kL2sq)
                 .WithDataType(DataType::DT_FP32)
                 .WithDimension(kDimension)
                 .WithIsSparse(false)
                 .WithEFConstruction(100)
                 .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
                 .Build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kL2sq);

  LOG_INFO("Test SparseVector, MetricType::kInnerProduct");
  sparse_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kInnerProduct)
          .WithDataType(DataType::DT_FP32)
          .WithIsSparse(true)
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  sparse_func(HNSWIndexParamBuilder()
                  .WithMetricType(MetricType::kInnerProduct)
                  .WithDataType(DataType::DT_FP32)
                  .WithIsSparse(true)
                  .WithEFConstruction(100)
                  .Build(),
              HNSWQueryParamBuilder()
                  .with_topk(kTopk)
                  .with_fetch_vector(true)
                  .with_ef_search(20)
                  .build(),
              MetricType::kInnerProduct);

  LOG_INFO(
      "Test SparseVector, MetricType::kInnerProduct, QuantizerType::kFP16");
  sparse_func(
      FlatIndexParamBuilder()
          .WithMetricType(MetricType::kInnerProduct)
          .WithDataType(DataType::DT_FP32)
          .WithIsSparse(true)
          .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
          .Build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  sparse_func(HNSWIndexParamBuilder()
                  .WithMetricType(MetricType::kInnerProduct)
                  .WithDataType(DataType::DT_FP32)
                  .WithIsSparse(true)
                  .WithEFConstruction(100)
                  .WithQuantizerParam(QuantizerParam(QuantizerType::kFP16))
                  .Build(),
              HNSWQueryParamBuilder()
                  .with_topk(kTopk)
                  .with_fetch_vector(true)
                  .with_ef_search(20)
                  .build(),
              MetricType::kInnerProduct);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif