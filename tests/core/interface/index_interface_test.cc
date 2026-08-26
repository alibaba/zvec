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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <gtest/gtest.h>
#include "tests/test_util.h"
#if RABITQ_SUPPORTED
#include "core/algorithm/hnsw_rabitq/rabitq_converter.h"
#include "zvec/core/framework/index_provider.h"
#endif
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/vamana/vamana_streamer.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/interface/index.h"
#include "zvec/core/interface/index_factory.h"
#include "zvec/core/interface/index_param.h"
#include "zvec/core/interface/index_param_builders.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core_interface;

TEST(IndexInterface, IndexTypeKeepsExistingValues) {
  EXPECT_EQ(5, static_cast<int>(IndexType::kDiskAnn));
  EXPECT_EQ(6, static_cast<int>(IndexType::kVamana));
  EXPECT_EQ(7, static_cast<int>(IndexType::kIVFRabitq));
}

TEST(IndexInterface, DiskAnnParamJsonRoundTrip) {
  auto param = DiskAnnIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(768)
                   .with_max_degree(48)
                   .with_list_size(80)
                   .with_pq_chunk_num(16)
                   .build();

  auto restored =
      IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json());
  auto diskann = std::dynamic_pointer_cast<DiskAnnIndexParam>(restored);
  ASSERT_NE(nullptr, diskann);
  EXPECT_EQ(48, diskann->max_degree);
  EXPECT_EQ(80, diskann->list_size);
  EXPECT_EQ(16, diskann->pq_chunk_num);
}

#if RABITQ_SUPPORTED
TEST(IndexInterface, IvfRabitqValidatesBuildParams) {
  auto make_param = [](int nlist, int sample_count) {
    return IVFRabitqIndexParamBuilder()
        .with_metric_type(MetricType::kInnerProduct)
        .with_data_type(DataType::DT_FP32)
        .with_dimension(64)
        .with_n_list(nlist)
        .with_total_bits(7)
        .with_sample_count(sample_count)
        .build();
  };

  EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*make_param(0, 0)));
  EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*make_param(-1, 0)));
  EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*make_param(32, -1)));
  EXPECT_NE(nullptr, IndexFactory::CreateAndInitIndex(*make_param(1, 0)));
  EXPECT_NE(nullptr, IndexFactory::CreateAndInitIndex(*make_param(1024, 1)));
  EXPECT_NE(nullptr, IndexFactory::CreateAndInitIndex(*make_param(1025, 0)));
}

TEST(IndexInterface, IvfRabitqFetchUnsupported) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"ivf_rabitq_fetch.index"};
  zvec::test_util::RemoveTestFiles(index_name);

  auto param = IVFRabitqIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_n_list(1)
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<float> vector(kDimension, 1.0f);
  VectorData vector_data{DenseVector{vector.data()}};
  ASSERT_EQ(0, index->add(vector_data, 0));

  VectorDataBuffer fetched;
  EXPECT_EQ(zvec::core::IndexError_Unsupported, index->fetch(0, &fetched));
  ASSERT_EQ(0, index->train());
  EXPECT_EQ(zvec::core::IndexError_Unsupported, index->fetch(0, &fetched));

  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(index_name);
}

TEST(IndexInterface, IvfRabitqSearchIgnoresFetchVector) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"ivf_rabitq_search_fetch_vector.index"};
  zvec::test_util::RemoveTestFiles(index_name);

  auto param = IVFRabitqIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_n_list(1)
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<float> vector(kDimension, 1.0f);
  VectorData vector_data{DenseVector{vector.data()}};
  ASSERT_EQ(0, index->add(vector_data, 0));
  ASSERT_EQ(0, index->train());

  auto query_param = std::make_shared<IVFRabitqQueryParam>();
  query_param->topk = 1;
  query_param->fetch_vector = true;
  query_param->nprobe = 1;
  SearchResult result;
  ASSERT_EQ(0, index->search(vector_data, query_param, &result));
  EXPECT_TRUE(result.reverted_vector_list_.empty());
  for (const auto &doc : result.doc_list_) {
    EXPECT_EQ(nullptr, doc.vector());
  }

  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(index_name);
}
#endif

class ReformerInspectableHNSWIndex : public HNSWIndex {
 public:
  int InitForTest(const BaseIndexParam &param) {
    return Init(param);
  }

  int TransformForTest(const std::vector<float> &query) const {
    if (!reformer_) {
      return zvec::core::IndexError_Uninitialized;
    }
    zvec::core::IndexQueryMeta input_meta(
        zvec::core::IndexMeta::DataType::DT_FP32, query.size());
    zvec::core::IndexQueryMeta output_meta;
    std::string output;
    return reformer_->transform(query.data(), input_meta, &output,
                                &output_meta);
  }
};

TEST(IndexInterface, General) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test.index"};

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    zvec::test_util::RemoveTestFiles(index_name);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);


    index->open(index_name, {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 2.0f;
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ASSERT_TRUE(0 == index->add(vector_data, 233));
    ASSERT_TRUE(0 == index->train());

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{vector.data()};
    index->search(query, query_param, &result);
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
    ASSERT_TRUE(0 == index->fetch(233, &fetched_vector_data));
    float *fetched_vector = reinterpret_cast<float *>(
        std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
            .data.data());
    ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
    ASSERT_FLOAT_EQ(2.0f, fetched_vector[2]);
    index->close();
    zvec::test_util::RemoveTestFiles(index_name);
  };


  auto param = FlatIndexParamBuilder()
                   .with_metric_type(MetricType::kInnerProduct)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_is_sparse(false)
                   .build();
  func(param,
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
  func(FlatIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(IVFIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_n_list(10)
           .build(),
       IVFQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
  func(IVFIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_n_list(10)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
       IVFQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(VamanaIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_max_degree(32)
           .with_search_list_size(100)
           .with_alpha(1.2f)
           .build(),
       VamanaQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(50)
           .build());

  // Vamana with topk > ef_search to exercise _get_coarse_search_topk branch
  // that picks max(topk, ef_search).
  func(VamanaIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_max_degree(32)
           .with_search_list_size(100)
           .with_alpha(1.2f)
           .build(),
       VamanaQueryParamBuilder()
           .with_topk(100)
           .with_fetch_vector(true)
           .with_ef_search(10)
           .build());
}

TEST(IndexInterface, ReopenRestoresUniformReformer) {
  constexpr size_t kDimension = 16;
  struct TestCase {
    QuantizerType quantizer_type;
    const char *converter_name;
    const char *index_name;
  };
  const TestCase test_cases[] = {
      {QuantizerType::kUniformUint7, "UniformUint7Converter",
       "test_uniform_uint7_reopen.index"},
      {QuantizerType::kUniformUint8, "UniformUint8Converter",
       "test_uniform_uint8_reopen.index"},
  };

  for (const auto &test_case : test_cases) {
    SCOPED_TRACE(test_case.converter_name);
    zvec::test_util::RemoveTestFiles(test_case.index_name);

    zvec::core::IndexMeta input_meta(zvec::core::IndexMeta::DataType::DT_FP32,
                                     kDimension);
    input_meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params());

    auto converter =
        zvec::core::IndexFactory::CreateConverter(test_case.converter_name);
    ASSERT_NE(nullptr, converter);
    ASSERT_EQ(0, converter->init(input_meta, zvec::ailego::Params()));

    auto holder = std::make_shared<zvec::core::MultiPassIndexHolder<
        zvec::core::IndexMeta::DataType::DT_FP32>>(kDimension);
    for (uint64_t key = 0; key < 2; ++key) {
      zvec::ailego::NumericalVector<float> vector(kDimension);
      for (size_t i = 0; i < kDimension; ++i) {
        vector[i] = static_cast<float>(key * kDimension + i);
      }
      ASSERT_TRUE(holder->emplace(key, std::move(vector)));
    }
    ASSERT_EQ(0, converter->train(holder));

    zvec::ailego::Params streamer_params;
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT,
                        16U);
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE,
                        true);
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_EF,
                        kDefaultHnswEfSearch);
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_USE_ID_MAP, true);
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_USE_CONTIGUOUS_MEMORY,
                        false);
    streamer_params.set(zvec::core::PARAM_HNSW_STREAMER_USE_EXTERNAL_VECTOR,
                        false);

    auto streamer = zvec::core::IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_NE(nullptr, streamer);
    ASSERT_EQ(0, streamer->init(converter->meta(), streamer_params));

    auto storage = zvec::core::IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(zvec::ailego::Params()));
    ASSERT_EQ(0, storage->open(test_case.index_name, true));
    ASSERT_EQ(0, streamer->open(storage));
    ASSERT_EQ(0, streamer->flush(0));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, streamer->cleanup());
    ASSERT_EQ(0, storage->close());

    auto param =
        HNSWIndexParamBuilder()
            .with_metric_type(MetricType::kL2sq)
            .with_data_type(DataType::DT_FP32)
            .with_dimension(kDimension)
            .with_is_sparse(false)
            .with_m(16)
            .with_ef_construction(100)
            .with_quantizer_param(QuantizerParam(test_case.quantizer_type))
            .build();
    ReformerInspectableHNSWIndex index;
    ASSERT_EQ(0, index.InitForTest(*param));

    const std::vector<float> query(kDimension, 1.0f);
    ASSERT_NE(0, index.TransformForTest(query));
    ASSERT_EQ(0, index.open(test_case.index_name,
                            {StorageOptions::StorageType::kMMAP,
                             /*create_new=*/false, /*read_only=*/true}));
    EXPECT_EQ(0, index.TransformForTest(query));
    ASSERT_EQ(0, index.close());

    zvec::test_util::RemoveTestFiles(test_case.index_name);
  }
}

TEST(IndexInterface, CopyOnWrite) {
  constexpr uint32_t kDimension = 64;
  constexpr uint32_t kNumVectors = 50;
  const std::string index_name{"test_cow.index"};

  auto make_vec = [&](uint32_t seed) {
    std::vector<float> v(kDimension, 0.0f);
    v[seed % kDimension] = 1.0f;
    return v;
  };

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    zvec::test_util::RemoveTestFiles(index_name);

    // Phase 1: build the index with shared mmap (writeable shared mapping)
    // since the COW mode isn't used as the initial ingest path here.
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(
          0, index->open(index_name, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/true, /*read_only=*/false,
                                      /*copy_on_write=*/false}));

      std::vector<std::vector<float>> vecs;
      vecs.reserve(kNumVectors);
      for (uint32_t i = 0; i < kNumVectors; ++i) {
        vecs.emplace_back(make_vec(i));
        VectorData vd;
        vd.vector = DenseVector{vecs.back().data()};
        ASSERT_EQ(0, index->add(vd, /*key=*/100 + i));
      }
      ASSERT_EQ(0, index->train());
      ASSERT_EQ(0, index->close());
    }

    // Phase 2: reopen with COW mmap. Search and Fetch must succeed against
    // the persisted file.
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(
          0, index->open(index_name, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/false, /*read_only=*/true,
                                      /*copy_on_write=*/true}));

      for (uint32_t i = 0; i < kNumVectors; ++i) {
        auto target = make_vec(i);
        VectorData query;
        query.vector = DenseVector{target.data()};
        SearchResult result;
        ASSERT_EQ(0, index->search(query, query_param, &result));
        ASSERT_FALSE(result.doc_list_.empty());
        ASSERT_EQ(100u + i, result.doc_list_[0].key());

        VectorDataBuffer fetched;
        ASSERT_EQ(0, index->fetch(100 + i, &fetched));
        auto *fetched_ptr = reinterpret_cast<const float *>(
            std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
        ASSERT_FLOAT_EQ(1.0f, fetched_ptr[i % kDimension]);
      }
      ASSERT_EQ(0, index->close());
    }

    // Phase 3: reopen with shared mmap to confirm the file is intact after
    // the COW session.
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(
          0, index->open(index_name, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/false, /*read_only=*/true,
                                      /*copy_on_write=*/false}));

      auto target = make_vec(13);
      VectorData query;
      query.vector = DenseVector{target.data()};
      SearchResult result;
      ASSERT_EQ(0, index->search(query, query_param, &result));
      ASSERT_FALSE(result.doc_list_.empty());
      ASSERT_EQ(113u, result.doc_list_[0].key());
      ASSERT_EQ(0, index->close());
    }

    // Phase 4: repeated open/close under COW mmap must not lose entries.
    for (int cycle = 0; cycle < 3; ++cycle) {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(
          0, index->open(index_name, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/false, /*read_only=*/true,
                                      /*copy_on_write=*/true}));
      uint32_t i = static_cast<uint32_t>(cycle * 5 + 2);
      auto target = make_vec(i);
      VectorData query;
      query.vector = DenseVector{target.data()};
      SearchResult result;
      ASSERT_EQ(0, index->search(query, query_param, &result));
      ASSERT_FALSE(result.doc_list_.empty());
      ASSERT_EQ(100u + i, result.doc_list_[0].key());
      ASSERT_EQ(0, index->close());
    }

    // Phase 5: open in COW mmap (writable MAP_PRIVATE with forced flush).
    // Without performing writes the close path still exercises the pwrite
    // branch with no dirty pages, which must not corrupt the file.
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(
          0, index->open(index_name, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/false, /*read_only=*/true,
                                      /*copy_on_write=*/true}));

      auto target = make_vec(21);
      VectorData query;
      query.vector = DenseVector{target.data()};
      SearchResult result;
      ASSERT_EQ(0, index->search(query, query_param, &result));
      ASSERT_FALSE(result.doc_list_.empty());
      ASSERT_EQ(121u, result.doc_list_[0].key());
      ASSERT_EQ(0, index->close());
    }

    // Phase 6: reopen with shared mmap to confirm Phase 5's open/close left
    // the file intact.
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(
          0, index->open(index_name, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/false, /*read_only=*/true,
                                      /*copy_on_write=*/false}));
      for (uint32_t i = 0; i < kNumVectors; ++i) {
        auto target = make_vec(i);
        VectorData query;
        query.vector = DenseVector{target.data()};
        SearchResult result;
        ASSERT_EQ(0, index->search(query, query_param, &result));
        ASSERT_FALSE(result.doc_list_.empty());
        ASSERT_EQ(100u + i, result.doc_list_[0].key());
      }
      ASSERT_EQ(0, index->close());
    }

    zvec::test_util::RemoveTestFiles(index_name);
  };

  func(FlatIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .build(),
       FlatQueryParamBuilder().with_topk(5).with_fetch_vector(false).build());

  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .build(),
       HNSWQueryParamBuilder()
           .with_topk(5)
           .with_fetch_vector(false)
           .with_ef_search(20)
           .build());

  func(VamanaIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_max_degree(32)
           .with_search_list_size(64)
           .with_alpha(1.2f)
           .build(),
       VamanaQueryParamBuilder()
           .with_topk(5)
           .with_fetch_vector(false)
           .with_ef_search(32)
           .build());

  // Flat-only durability check for COW mmap: writes performed under
  // MAP_PRIVATE must be pwrite-flushed back and visible after a shared-mmap
  // reopen. Flat is used because Add/Flush against a previously-built file is
  // straightforward to reason about for this storage layer.
  {
    const std::string persist_index{"test_cow_persist.index"};
    zvec::test_util::RemoveTestFiles(persist_index);
    auto persist_param = FlatIndexParamBuilder()
                             .with_metric_type(MetricType::kInnerProduct)
                             .with_data_type(DataType::DT_FP32)
                             .with_dimension(kDimension)
                             .with_is_sparse(false)
                             .build();
    auto persist_query =
        FlatQueryParamBuilder().with_topk(5).with_fetch_vector(false).build();

    {
      auto index = IndexFactory::CreateAndInitIndex(*persist_param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(0, index->open(persist_index,
                               {StorageOptions::StorageType::kMMAP,
                                /*create_new=*/true, /*read_only=*/false,
                                /*copy_on_write=*/false}));
      auto v0 = make_vec(0);
      VectorData vd;
      vd.vector = DenseVector{v0.data()};
      ASSERT_EQ(0, index->add(vd, /*key=*/500));
      ASSERT_EQ(0, index->train());
      ASSERT_EQ(0, index->close());
    }

    // Add a new vector through COW mmap and explicitly Flush so
    // dirty private pages are written back to the file.
    {
      auto index = IndexFactory::CreateAndInitIndex(*persist_param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(0, index->open(persist_index,
                               {StorageOptions::StorageType::kMMAP,
                                /*create_new=*/false, /*read_only=*/false,
                                /*copy_on_write=*/true}));
      auto v1 = make_vec(1);
      VectorData vd;
      vd.vector = DenseVector{v1.data()};
      ASSERT_EQ(0, index->add(vd, /*key=*/501));
      ASSERT_EQ(0, index->flush());
      ASSERT_EQ(0, index->close());
    }

    // Reopen with shared mmap: the entry written in COW mode must be durable
    // on disk.
    {
      auto index = IndexFactory::CreateAndInitIndex(*persist_param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(0, index->open(persist_index,
                               {StorageOptions::StorageType::kMMAP,
                                /*create_new=*/false, /*read_only=*/true,
                                /*copy_on_write=*/false}));
      auto target = make_vec(1);
      VectorData query;
      query.vector = DenseVector{target.data()};
      SearchResult result;
      ASSERT_EQ(0, index->search(query, persist_query, &result));
      ASSERT_FALSE(result.doc_list_.empty());
      ASSERT_EQ(501u, result.doc_list_[0].key());

      VectorDataBuffer fetched;
      ASSERT_EQ(0, index->fetch(501, &fetched));
      auto *fetched_ptr = reinterpret_cast<const float *>(
          std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
      ASSERT_FLOAT_EQ(1.0f, fetched_ptr[1 % kDimension]);
      ASSERT_EQ(0, index->close());
    }
    zvec::test_util::RemoveTestFiles(persist_index);
  }
}

TEST(IndexInterface, BufferGeneral) {
  zvec::ailego::MemoryLimitPool::get_instance().init(100 * 1024 * 1024);
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test.index"};

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    std::string real_index_name = index_name;
    zvec::test_util::RemoveTestFiles(index_name + "*");
    auto write_index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, write_index);

    write_index->open(real_index_name,
                      {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 2.0f;
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ASSERT_TRUE(0 == write_index->add(vector_data, 233));
    write_index->close();

    auto read_index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, read_index);
    read_index->open(real_index_name,
                     {StorageOptions::StorageType::kBufferPool, false});

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{vector.data()};
    read_index->search(query, query_param, &result);
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
    ASSERT_TRUE(0 == read_index->fetch(233, &fetched_vector_data));
    float *fetched_vector = reinterpret_cast<float *>(
        std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
            .data.data());
    ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
    ASSERT_FLOAT_EQ(2.0f, fetched_vector[2]);
    result.doc_list_.clear();
    read_index->close();
    zvec::test_util::RemoveTestFiles(index_name + "*");
  };


  auto param = FlatIndexParamBuilder()
                   .with_metric_type(MetricType::kInnerProduct)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_is_sparse(false)
                   .build();
  func(param,
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());
  func(FlatIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
}


TEST(IndexInterface, SparseGeneral) {
  constexpr uint32_t kSparseCount = 3;
  const std::string index_name{"test.index"};

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    zvec::test_util::RemoveTestFiles(index_name);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);


    index->open(index_name, {StorageOptions::StorageType::kMMAP, true});

    std::vector<uint32_t> indices(kSparseCount);
    std::vector<float> values(kSparseCount);
    for (uint32_t i = 0; i < kSparseCount; ++i) {
      indices[i] = i;
      values[i] = i;
    }

    VectorData vector_data{
        SparseVector{kSparseCount, indices.data(), values.data()}};
    ASSERT_TRUE(0 == index->add(vector_data, 233));


    SearchResult result;
    VectorData query = {
        SparseVector{kSparseCount, indices.data(), values.data()}};
    index->search(query, query_param, &result);
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
    ASSERT_TRUE(0 == index->fetch(233, &fetched_vector_data));
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
    index->close();
    zvec::test_util::RemoveTestFiles(index_name);
  };


  auto param = FlatIndexParamBuilder()
                   .with_metric_type(MetricType::kInnerProduct)
                   .with_data_type(DataType::DT_FP32)
                   .with_is_sparse(true)
                   .build();
  // func(param, FlatQueryParam{{.topk = 10, .fetch_vector = true}});
  func(FlatIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_is_sparse(true)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
       FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build());

  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_is_sparse(true)
           .with_ef_construction(100)
           .build(),
       HNSWQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(true)
           .with_ef_search(20)
           .build());
  func(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_is_sparse(true)
           .with_ef_construction(100)
           .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
           .build(),
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
    zvec::test_util::RemoveTestFiles(file_name);
  };

  auto create_index_func =
      [&](const BaseIndexParam::Pointer &param,
          const std::string &index_name) -> Index::Pointer {
    del_index_file_func(index_name);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    if (index == nullptr ||
        0 != index->open(index_name,
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
    ASSERT_TRUE(0 == index1->add(vector_data, 0));

    vector[1] = 2.0f;
    ASSERT_TRUE(0 == index2->add(vector_data, 0));
    vector[1] = 3.0f;
    ASSERT_TRUE(0 == index2->add(vector_data, 1));

    {
      VectorDataBuffer fetched_vector_data;
      ASSERT_TRUE(0 == index1->fetch(0, &fetched_vector_data));
      float *fetched_vector = reinterpret_cast<float *>(
          std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
              .data.data());
      ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }
    {
      VectorDataBuffer fetched_vector_data;
      ASSERT_TRUE(0 == index2->fetch(0, &fetched_vector_data));
      float *fetched_vector = reinterpret_cast<float *>(
          std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
              .data.data());
      ASSERT_FLOAT_EQ(2.0f, fetched_vector[1]);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }
    {
      VectorDataBuffer fetched_vector_data;
      ASSERT_TRUE(0 == index2->fetch(1, &fetched_vector_data));
      float *fetched_vector = reinterpret_cast<float *>(
          std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
              .data.data());
      ASSERT_FLOAT_EQ(3.0f, fetched_vector[1]);
      ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
    }

    {  // test reduce
      auto index3 = create_index_func(param_target, index_name + "3");
      ASSERT_NE(nullptr, index3);
      MergeOptions merge_options;
      merge_options.write_concurrency = (std::numeric_limits<uint32_t>::max)();
      ASSERT_TRUE(
          0 == index3->merge({index1, index2}, IndexFilter(), merge_options));
      ASSERT_TRUE(3 == index3->get_doc_count());
      {
        VectorDataBuffer fetched_vector_data;
        ASSERT_TRUE(0 == index3->fetch(0, &fetched_vector_data));
        float *fetched_vector = reinterpret_cast<float *>(
            std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
                .data.data());
        ASSERT_FLOAT_EQ(1.0f, fetched_vector[1]);
        ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
      }
      {
        VectorDataBuffer fetched_vector_data;
        ASSERT_TRUE(0 == index3->fetch(1, &fetched_vector_data));
        float *fetched_vector = reinterpret_cast<float *>(
            std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
                .data.data());
        ASSERT_FLOAT_EQ(2.0f, fetched_vector[1]);
        ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
      }
      index3->close();
      del_index_file_func(index_name + "3");
    }

    {  // test reduce with filter
      auto index3 = create_index_func(param_target, index_name + "3");
      ASSERT_NE(nullptr, index3);
      auto filter = IndexFilter();
      filter.set([](uint64_t key) { return key == 0; });  // TODO: uint32?
      zvec::ailego::ThreadPool pool(1, false);
      MergeOptions merge_options;
      merge_options.write_concurrency = (std::numeric_limits<uint32_t>::max)();
      merge_options.pool = &pool;
      ASSERT_TRUE(0 == index3->merge({index1, index2}, filter, merge_options));
      ASSERT_TRUE(2 == index3->get_doc_count());
      {
        VectorDataBuffer fetched_vector_data;
        ASSERT_TRUE(0 == index3->fetch(0, &fetched_vector_data));
        float *fetched_vector = reinterpret_cast<float *>(
            std::get<DenseVectorBuffer>(fetched_vector_data.vector_buffer)
                .data.data());
        ASSERT_FLOAT_EQ(2.0f, fetched_vector[1]);
        ASSERT_FLOAT_EQ(123.0f, fetched_vector[2]);
      }
      index3->close();
      del_index_file_func(index_name + "3");
    }

    index1->close();
    index2->close();
    del_index_file_func(index_name + "1");
    del_index_file_func(index_name + "2");
  };

  // same index
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(kDimension)
                     .with_is_sparse(false)
                     .build();
    func(param, param);
  }
  {
    auto param = HNSWIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(kDimension)
                     .with_is_sparse(false)
                     .build();
    func(param, param);
  }

  // different index
  {
    auto param_flat = FlatIndexParamBuilder()
                          .with_metric_type(MetricType::kInnerProduct)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_is_sparse(false)
                          .build();
    auto param_hnsw = HNSWIndexParamBuilder()
                          .with_metric_type(MetricType::kInnerProduct)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_is_sparse(false)
                          .build();
    func(param_flat, param_hnsw);
    func(param_hnsw, param_flat);
  }
}

TEST(IndexInterface, FlatStorageDataTypeConvertsFp32InputAndQuery) {
  constexpr uint32_t kDimension = 17;
  constexpr uint32_t kVectorCount = 8;
  constexpr uint32_t kTopk = 4;
  const std::string source_name{"native_flat_refine_source.index"};
  zvec::test_util::RemoveTestFiles(source_name);

  auto source_param = FlatIndexParamBuilder()
                          .with_metric_type(MetricType::kL2sq)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_is_sparse(false)
                          .build();
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(
      0, source->open(source_name, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<std::vector<float>> vectors(kVectorCount,
                                          std::vector<float>(kDimension));
  for (uint32_t i = 0; i < kVectorCount; ++i) {
    for (uint32_t d = 0; d < kDimension; ++d) {
      vectors[i][d] = static_cast<float>((i * 47 + d * 29) % 311) - 23.25F;
    }
    ASSERT_EQ(0, source->add(VectorData{DenseVector{vectors[i].data()}}, i));
  }

  auto to_native = [](const std::vector<float> &input, DataType target_type) {
    std::vector<float> output(input.size());
    if (target_type == DataType::DT_FP16) {
      std::vector<uint16_t> fp16(input.size());
      zvec::ailego::FloatHelper::ToFP16(input.data(), input.size(),
                                        fp16.data());
      zvec::ailego::FloatHelper::ToFP32(fp16.data(), fp16.size(),
                                        output.data());
    } else {
      for (size_t i = 0; i < input.size(); ++i) {
        const float value = input[i];
        output[i] = !(value > 0.0F) ? 0.0F
                    : value >= 255.0F
                        ? 255.0F
                        : static_cast<float>(static_cast<uint8_t>(value));
      }
    }
    return output;
  };

  for (const auto target_type : {DataType::DT_FP16, DataType::DT_UINT8}) {
    for (const bool use_contiguous : {false, true}) {
      const std::string target_name =
          std::string("native_flat_refine_") +
          (target_type == DataType::DT_FP16 ? "fp16_" : "uint8_") +
          (use_contiguous ? "contiguous.index" : "generic.index");
      zvec::test_util::RemoveTestFiles(target_name);

      auto target_param = FlatIndexParamBuilder()
                              .with_metric_type(MetricType::kL2sq)
                              .with_data_type(DataType::DT_FP32)
                              .with_storage_data_type(target_type)
                              .with_dimension(kDimension)
                              .with_is_sparse(false)
                              .with_use_contiguous_memory(use_contiguous)
                              .build();
      auto target = IndexFactory::CreateAndInitIndex(*target_param);
      ASSERT_NE(nullptr, target);
      ASSERT_EQ(0, target->open(target_name,
                                {StorageOptions::StorageType::kMMAP, true}));
      for (uint32_t i = 0; i < kVectorCount; ++i) {
        ASSERT_EQ(0,
                  target->add(VectorData{DenseVector{vectors[i].data()}}, i));
      }

      auto refiner_param = std::make_shared<RefinerParam>();
      refiner_param->scale_factor_ = kVectorCount;
      refiner_param->reference_index = target;
      auto query_param = FlatQueryParamBuilder()
                             .with_topk(kTopk)
                             .with_fetch_vector(true)
                             .with_refiner_param(refiner_param)
                             .build();

      constexpr uint32_t kQueryId = 3;
      SearchResult result;
      ASSERT_EQ(
          0, source->search(VectorData{DenseVector{vectors[kQueryId].data()}},
                            query_param, &result));

      auto direct_query_param = FlatQueryParamBuilder()
                                    .with_topk(kTopk)
                                    .with_fetch_vector(true)
                                    .build();
      SearchResult direct_result;
      ASSERT_EQ(
          0, target->search(VectorData{DenseVector{vectors[kQueryId].data()}},
                            direct_query_param, &direct_result));
      ASSERT_EQ(
          0, source->search(VectorData{DenseVector{vectors[kQueryId].data()}},
                            query_param, &result));

      const auto native_query = to_native(vectors[kQueryId], target_type);
      std::vector<std::pair<float, uint64_t>> expected;
      expected.reserve(kVectorCount);
      for (uint32_t i = 0; i < kVectorCount; ++i) {
        const auto native_vector = to_native(vectors[i], target_type);
        float distance = 0.0F;
        for (uint32_t d = 0; d < kDimension; ++d) {
          const float delta = native_vector[d] - native_query[d];
          distance += delta * delta;
        }
        expected.emplace_back(distance, i);
      }
      std::sort(expected.begin(), expected.end());

      ASSERT_EQ(kTopk, result.doc_list_.size());
      ASSERT_EQ(kTopk, result.reverted_vector_list_.size());
      ASSERT_EQ(kTopk, direct_result.doc_list_.size());
      ASSERT_EQ(kTopk, direct_result.reverted_vector_list_.size());
      for (size_t i = 0; i < kTopk; ++i) {
        EXPECT_EQ(expected[i].second, result.doc_list_[i].key());
        EXPECT_NEAR(expected[i].first, result.doc_list_[i].score(), 1e-3F);
        EXPECT_EQ(result.doc_list_[i].key(), direct_result.doc_list_[i].key());
        EXPECT_FLOAT_EQ(result.doc_list_[i].score(),
                        direct_result.doc_list_[i].score());

        const auto expected_vector =
            to_native(vectors[expected[i].second], target_type);
        const auto *restored = reinterpret_cast<const float *>(
            result.reverted_vector_list_[i].data());
        const auto *direct_restored = reinterpret_cast<const float *>(
            direct_result.reverted_vector_list_[i].data());
        for (uint32_t d = 0; d < kDimension; ++d) {
          EXPECT_FLOAT_EQ(expected_vector[d], restored[d]);
          EXPECT_FLOAT_EQ(restored[d], direct_restored[d]);
        }
      }

      ASSERT_EQ(0, target->close());
      zvec::test_util::RemoveTestFiles(target_name);
    }
  }

  ASSERT_EQ(0, source->close());
  zvec::test_util::RemoveTestFiles(source_name);
}

TEST(IndexInterface, Fp16CosineRefineMatchesNativeFlatPipeline) {
  constexpr uint32_t kDimension = 33;
  constexpr uint32_t kVectorCount = 40;
  constexpr uint32_t kTopk = 16;
  const std::string source_name{"fp16_cosine_refine_source.index"};
  zvec::test_util::RemoveTestFiles(source_name);

  auto source_param = FlatIndexParamBuilder()
                          .with_metric_type(MetricType::kCosine)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_is_sparse(false)
                          .build();
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(
      0, source->open(source_name, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<std::vector<float>> vectors(kVectorCount,
                                          std::vector<float>(kDimension));
  for (uint32_t i = 0; i < kVectorCount; ++i) {
    for (uint32_t d = 0; d < kDimension; ++d) {
      const float base = 0.25F + static_cast<float>(d) * 0.017F;
      const int32_t offset = static_cast<int32_t>((i * 37 + d * 19) % 97) - 48;
      vectors[i][d] = base + static_cast<float>(offset) * 0.00011F;
    }
    ASSERT_EQ(0, source->add(VectorData{DenseVector{vectors[i].data()}}, i));
  }

  std::vector<float> query = vectors[7];
  for (uint32_t d = 0; d < kDimension; ++d) {
    query[d] += static_cast<float>(static_cast<int32_t>(d % 5) - 2) * 0.00017F;
  }
  std::vector<uint16_t> native_query(kDimension);
  zvec::ailego::FloatHelper::ToFP16(query.data(), query.size(),
                                    native_query.data());

  for (const bool use_contiguous : {false, true}) {
    const std::string target_name = use_contiguous
                                        ? "fp16_cosine_refine_contiguous.index"
                                        : "fp16_cosine_refine_generic.index";
    const std::string native_name = use_contiguous
                                        ? "fp16_cosine_native_contiguous.index"
                                        : "fp16_cosine_native_generic.index";
    zvec::test_util::RemoveTestFiles(target_name);
    zvec::test_util::RemoveTestFiles(native_name);
    auto target_param = FlatIndexParamBuilder()
                            .with_metric_type(MetricType::kCosine)
                            .with_data_type(DataType::DT_FP32)
                            .with_storage_data_type(DataType::DT_FP16)
                            .with_dimension(kDimension)
                            .with_is_sparse(false)
                            .with_use_contiguous_memory(use_contiguous)
                            .build();
    auto native_target_param = FlatIndexParamBuilder()
                                   .with_metric_type(MetricType::kCosine)
                                   .with_data_type(DataType::DT_FP16)
                                   .with_dimension(kDimension)
                                   .with_is_sparse(false)
                                   .with_use_contiguous_memory(use_contiguous)
                                   .build();

    auto native_target = IndexFactory::CreateAndInitIndex(*native_target_param);
    ASSERT_NE(nullptr, native_target);
    ASSERT_EQ(0, native_target->open(
                     native_name, {StorageOptions::StorageType::kMMAP, true}));
    for (uint32_t i = 0; i < kVectorCount; ++i) {
      std::vector<uint16_t> native(kDimension);
      zvec::ailego::FloatHelper::ToFP16(vectors[i].data(), vectors[i].size(),
                                        native.data());
      ASSERT_EQ(0,
                native_target->add(VectorData{DenseVector{native.data()}}, i));
    }

    {
      auto writer = IndexFactory::CreateAndInitIndex(*target_param);
      ASSERT_NE(nullptr, writer);
      ASSERT_EQ(0, writer->open(target_name,
                                {StorageOptions::StorageType::kMMAP, true}));
      for (uint32_t i = 0; i < kVectorCount; ++i) {
        ASSERT_EQ(0,
                  writer->add(VectorData{DenseVector{vectors[i].data()}}, i));
      }
      ASSERT_EQ(0, writer->close());
    }

    auto target = IndexFactory::CreateAndInitIndex(*target_param);
    ASSERT_NE(nullptr, target);
    ASSERT_EQ(0, target->open(target_name,
                              {StorageOptions::StorageType::kMMAP, false}));

    auto native_param = FlatQueryParamBuilder()
                            .with_topk(kTopk)
                            .with_fetch_vector(true)
                            .build();
    SearchResult native_result;
    ASSERT_EQ(
        0, native_target->search(VectorData{DenseVector{native_query.data()}},
                                 native_param, &native_result));
    SearchResult direct_result;
    ASSERT_EQ(0, target->search(VectorData{DenseVector{query.data()}},
                                native_param, &direct_result));

    auto refiner_param = std::make_shared<RefinerParam>();
    refiner_param->scale_factor_ = kVectorCount;
    refiner_param->reference_index = target;
    auto refine_param = FlatQueryParamBuilder()
                            .with_topk(kTopk)
                            .with_fetch_vector(true)
                            .with_refiner_param(refiner_param)
                            .build();
    SearchResult refined_result;
    ASSERT_EQ(0, source->search(VectorData{DenseVector{query.data()}},
                                refine_param, &refined_result));

    ASSERT_EQ(native_result.doc_list_.size(), refined_result.doc_list_.size());
    ASSERT_EQ(native_result.doc_list_.size(), direct_result.doc_list_.size());
    ASSERT_EQ(native_result.doc_list_.size(),
              native_result.reverted_vector_list_.size());
    ASSERT_EQ(direct_result.doc_list_.size(),
              direct_result.reverted_vector_list_.size());
    ASSERT_EQ(direct_result.reverted_vector_list_.size(),
              refined_result.reverted_vector_list_.size());
    for (size_t i = 0; i < native_result.doc_list_.size(); ++i) {
      EXPECT_EQ(native_result.doc_list_[i].key(),
                direct_result.doc_list_[i].key());
      EXPECT_EQ(direct_result.doc_list_[i].key(),
                refined_result.doc_list_[i].key());
      EXPECT_NEAR(native_result.doc_list_[i].score(),
                  direct_result.doc_list_[i].score(), 1e-7F);
      EXPECT_NEAR(direct_result.doc_list_[i].score(),
                  refined_result.doc_list_[i].score(), 1e-7F);

      ASSERT_EQ(kDimension * sizeof(uint16_t),
                native_result.reverted_vector_list_[i].size());
      ASSERT_EQ(kDimension * sizeof(float),
                refined_result.reverted_vector_list_[i].size());
      ASSERT_EQ(kDimension * sizeof(float),
                direct_result.reverted_vector_list_[i].size());
      std::vector<float> expected(kDimension);
      zvec::ailego::FloatHelper::ToFP32(
          reinterpret_cast<const uint16_t *>(
              native_result.reverted_vector_list_[i].data()),
          kDimension, expected.data());
      const auto *restored = reinterpret_cast<const float *>(
          refined_result.reverted_vector_list_[i].data());
      const auto *direct_restored = reinterpret_cast<const float *>(
          direct_result.reverted_vector_list_[i].data());
      for (uint32_t d = 0; d < kDimension; ++d) {
        EXPECT_FLOAT_EQ(expected[d], restored[d]);
        EXPECT_FLOAT_EQ(expected[d], direct_restored[d]);
      }
    }

    ASSERT_EQ(0, native_target->close());
    ASSERT_EQ(0, target->close());
    zvec::test_util::RemoveTestFiles(native_name);
    zvec::test_util::RemoveTestFiles(target_name);
  }

  ASSERT_EQ(0, source->close());
  zvec::test_util::RemoveTestFiles(source_name);
}

TEST(IndexInterface, VamanaTwoPassFinalizeOnMerge) {
  constexpr uint32_t kDimension = 16;
  constexpr uint32_t kVectorCount = 64;
  const std::string source_name{"vamana_two_pass_source.index"};
  const std::string target_name{"vamana_two_pass_target.index"};

  auto remove_files = [](const std::string &path) {
    zvec::test_util::RemoveTestFiles(path);
  };
  remove_files(source_name);
  remove_files(target_name);

  auto source_param = FlatIndexParamBuilder()
                          .with_metric_type(MetricType::kL2sq)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_is_sparse(false)
                          .build();
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(
      0, source->open(source_name, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<std::vector<float>> vectors(kVectorCount,
                                          std::vector<float>(kDimension));
  for (uint32_t i = 0; i < kVectorCount; ++i) {
    for (uint32_t d = 0; d < kDimension; ++d) {
      vectors[i][d] = static_cast<float>((i * 17 + d * 13) % 101);
    }
    VectorData data{DenseVector{vectors[i].data()}};
    ASSERT_EQ(0, source->add(data, i));
  }

  auto run_merge = [&](bool two_pass_build) {
    remove_files(target_name);
    auto target_param = VamanaIndexParamBuilder()
                            .with_metric_type(MetricType::kL2sq)
                            .with_data_type(DataType::DT_FP32)
                            .with_dimension(kDimension)
                            .with_is_sparse(false)
                            .with_max_degree(16)
                            .with_search_list_size(32)
                            .with_alpha(1.5f)
                            .with_two_pass_build(two_pass_build)
                            .build();
    auto target = IndexFactory::CreateAndInitIndex(*target_param);
    ASSERT_NE(nullptr, target);
    ASSERT_EQ(0, target->open(target_name,
                              {StorageOptions::StorageType::kMMAP, true}));
    ASSERT_EQ(0, target->merge({source}, IndexFilter()));
    ASSERT_EQ(kVectorCount, target->get_doc_count());

    const uint32_t expected_refine_passes = two_pass_build ? 1U : 0U;
    const uint32_t expected_build_passes = two_pass_build ? 2U : 1U;
    const float expected_initial_alpha = two_pass_build ? 1.0f : 1.5f;

    uint32_t refine_pass_count = (std::numeric_limits<uint32_t>::max)();
    ASSERT_TRUE(target->index_searcher()->stats().get_attribute(
        "vamana_refine_pass_count", &refine_pass_count));
    EXPECT_EQ(expected_refine_passes, refine_pass_count);

    uint32_t build_pass_count = (std::numeric_limits<uint32_t>::max)();
    ASSERT_TRUE(target->index_searcher()->stats().get_attribute(
        "vamana_build_pass_count", &build_pass_count));
    EXPECT_EQ(expected_build_passes, build_pass_count);

    float initial_build_alpha = 0.0f;
    ASSERT_TRUE(target->index_searcher()->stats().get_attribute(
        "vamana_initial_build_alpha", &initial_build_alpha));
    EXPECT_FLOAT_EQ(expected_initial_alpha, initial_build_alpha);

    // Finalization is idempotent: persistence must not rerun the second pass.
    auto *vamana_streamer = dynamic_cast<zvec::core::VamanaStreamer *>(
        target->index_searcher().get());
    ASSERT_NE(nullptr, vamana_streamer);
    ASSERT_EQ(0, vamana_streamer->finalize_build());
    ASSERT_TRUE(target->index_searcher()->stats().get_attribute(
        "vamana_refine_pass_count", &refine_pass_count));
    EXPECT_EQ(expected_refine_passes, refine_pass_count);
    ASSERT_TRUE(target->index_searcher()->stats().get_attribute(
        "vamana_build_pass_count", &build_pass_count));
    EXPECT_EQ(expected_build_passes, build_pass_count);

    ASSERT_EQ(0, target->flush());
    ASSERT_EQ(0, target->close());

    auto reopened = IndexFactory::CreateAndInitIndex(*target_param);
    ASSERT_NE(nullptr, reopened);
    ASSERT_EQ(0, reopened->open(target_name,
                                {StorageOptions::StorageType::kMMAP, false}));
    auto query_param = VamanaQueryParamBuilder()
                           .with_topk(10)
                           .with_fetch_vector(false)
                           .with_ef_search(64)
                           .build();
    VectorData query{DenseVector{vectors[7].data()}};
    SearchResult result;
    ASSERT_EQ(0, reopened->search(query, query_param, &result));
    ASSERT_FALSE(result.doc_list_.empty());
    EXPECT_TRUE(std::any_of(result.doc_list_.begin(), result.doc_list_.end(),
                            [](const auto &doc) { return doc.key() == 7U; }));
    ASSERT_EQ(0, reopened->close());
  };

  run_merge(false);
  run_merge(true);

  ASSERT_EQ(0, source->close());
  remove_files(source_name);
  remove_files(target_name);
}

TEST(IndexInterface, Serialize) {
  {
    std::cout << "\n\n----flat index----" << std::endl;
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .with_storage_data_type(DataType::DT_FP16)
                     .build();

    std::cout << "flat index -- omit=true: " << param->serialize_to_json(true)
              << std::endl;
    std::cout << "omit=false: " << param->serialize_to_json() << std::endl;

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json());
    ASSERT_NE(nullptr, deserialized_param.get());
    auto flat_param =
        std::dynamic_pointer_cast<FlatIndexParam>(deserialized_param);
    ASSERT_NE(nullptr, flat_param);
    EXPECT_EQ(DataType::DT_FP16, flat_param->storage_data_type);

    std::cout << "serialize then de then se:"
              << deserialized_param->serialize_to_json() << std::endl;

    ASSERT_TRUE(deserialized_param->serialize_to_json() ==
                param->serialize_to_json());
    ASSERT_TRUE(deserialized_param->serialize_to_json(true) ==
                param->serialize_to_json(true));
  }

  {
    std::cout << "\n\n----hnsw index----" << std::endl;
    auto param = HNSWIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .with_quantizer_param(QuantizerParam{QuantizerType::kFP16})
                     .build();

    std::cout << "hnsw index -- omit=true: " << param->serialize_to_json(true)
              << std::endl;
    std::cout << "hnsw index -- omit=false: " << param->serialize_to_json()
              << std::endl;

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json());
    ASSERT_NE(nullptr, deserialized_param.get());

    std::cout << "serialize then de then se:"
              << deserialized_param->serialize_to_json() << std::endl;


    ASSERT_TRUE(deserialized_param->serialize_to_json() ==
                param->serialize_to_json());
    ASSERT_TRUE(deserialized_param->serialize_to_json(true) ==
                param->serialize_to_json(true));
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

  {
    std::cout << "\n\n----vamana index----" << std::endl;
    auto param = VamanaIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .with_max_degree(32)
                     .with_search_list_size(100)
                     .with_alpha(1.2f)
                     .build();

    std::cout << "vamana index -- omit=true: " << param->serialize_to_json(true)
              << std::endl;
    std::cout << "vamana index -- omit=false: " << param->serialize_to_json()
              << std::endl;

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json());
    ASSERT_NE(nullptr, deserialized_param.get());

    std::cout << "serialize then de then se:"
              << deserialized_param->serialize_to_json() << std::endl;

    ASSERT_TRUE(deserialized_param->serialize_to_json() ==
                param->serialize_to_json());
    ASSERT_TRUE(deserialized_param->serialize_to_json(true) ==
                param->serialize_to_json(true));
  }

  {
    std::cout << "\n\n----hnsw index with use_contiguous_memory----"
              << std::endl;
    auto param = std::make_shared<HNSWIndexParam>();
    param->metric_type = MetricType::kL2sq;
    param->data_type = DataType::DT_FP32;
    param->dimension = 64;
    param->use_contiguous_memory = true;

    auto json_str = param->serialize_to_json();
    std::cout << "hnsw contiguous -- json: " << json_str << std::endl;
    ASSERT_TRUE(json_str.find("use_contiguous_memory") != std::string::npos);

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(json_str);
    ASSERT_NE(nullptr, deserialized_param.get());
    auto hnsw_param =
        std::dynamic_pointer_cast<HNSWIndexParam>(deserialized_param);
    ASSERT_NE(nullptr, hnsw_param.get());
    ASSERT_TRUE(hnsw_param->use_contiguous_memory);

    ASSERT_TRUE(deserialized_param->serialize_to_json() == json_str);
  }

  {
    std::cout << "\n\n----vamana index with use_contiguous_memory----"
              << std::endl;
    auto param = std::make_shared<VamanaIndexParam>();
    param->metric_type = MetricType::kL2sq;
    param->data_type = DataType::DT_FP32;
    param->dimension = 64;
    param->max_degree = 48;
    param->search_list_size = 200;
    param->alpha = 1.5f;
    param->use_contiguous_memory = true;
    param->two_pass_build = true;

    auto json_str = param->serialize_to_json();
    std::cout << "vamana contiguous -- json: " << json_str << std::endl;
    ASSERT_TRUE(json_str.find("use_contiguous_memory") != std::string::npos);
    ASSERT_TRUE(json_str.find("two_pass_build") != std::string::npos);

    auto deserialized_param =
        IndexFactory::DeserializeIndexParamFromJson(json_str);
    ASSERT_NE(nullptr, deserialized_param.get());
    auto vamana_param =
        std::dynamic_pointer_cast<VamanaIndexParam>(deserialized_param);
    ASSERT_NE(nullptr, vamana_param.get());
    ASSERT_TRUE(vamana_param->use_contiguous_memory);
    ASSERT_TRUE(vamana_param->two_pass_build);
    ASSERT_EQ(48, vamana_param->max_degree);
    ASSERT_EQ(200, vamana_param->search_list_size);
    ASSERT_FLOAT_EQ(1.5f, vamana_param->alpha);

    ASSERT_TRUE(deserialized_param->serialize_to_json() == json_str);
  }

  {
    std::cout << "\n\n----vamana query----" << std::endl;
    auto param = VamanaQueryParamBuilder()
                     .with_topk(10)
                     .with_fetch_vector(true)
                     .with_ef_search(50)
                     .build();
    std::cout << "vamana query -- omit=true: "
              << IndexFactory::QueryParamSerializeToJson(*param, true)
              << std::endl;
    std::cout << "vamana query -- omit=false: "
              << IndexFactory::QueryParamSerializeToJson(*param) << std::endl;

    auto deserialized_param =
        IndexFactory::QueryParamDeserializeFromJson<VamanaQueryParam>(
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
            .with_metric_type(MetricType::kNone)  // L2 not supported for sparse
            .with_data_type(DataType::DT_FP32)
            .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test unsupported metric type for sparse index
  {
    auto param =
        FlatIndexParamBuilder()
            .with_metric_type(MetricType::kL2sq)  // L2 not supported for sparse
            .with_data_type(DataType::DT_FP32)
            .with_is_sparse(true)
            .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // // Test unsupported quantizer type
  // {
  //   auto param = FlatIndexParamBuilder()
  //                    .with_metric_type(MetricType::kInnerProduct)
  //                    .with_data_type(DataType::DT_INT4)
  //                    .with_dimension(64)
  //                    .with_is_sparse(false)
  //                    .with_quantizer_param(
  //                        QuantizerParam(QuantizerType::kInt8))  //
  //                        Unsupported
  //                    .build();
  //   auto index = IndexFactory::CreateAndInitIndex(*param);
  //   ASSERT_EQ(nullptr, index);
  // }
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(true)
                     .with_quantizer_param(
                         QuantizerParam(QuantizerType::kInt8))  // Unsupported
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test unsupported data type for cosine metric
  {
    auto param =
        FlatIndexParamBuilder()
            .with_metric_type(MetricType::kCosine)
            .with_data_type(DataType::DT_INT8)  // Unsupported for cosine
            .with_dimension(64)
            .with_is_sparse(false)
            .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_EQ(nullptr, index);
  }

  // Test invalid storage type
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    StorageOptions invalid_storage;
    invalid_storage.type = StorageOptions::StorageType::kNone;  // Unsupported
    int ret = index->open("test.index", invalid_storage);
    ASSERT_NE(0, ret);
  }

  // Test invalid vector data type for dense operations
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    // Try to add sparse vector to dense index
    std::vector<uint32_t> indices = {0, 1, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    VectorData sparse_vector_data{
        SparseVector{3, indices.data(), values.data()}};

    int ret = index->add(sparse_vector_data, 1);
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
  }

  // Test invalid vector data type for sparse operations
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_is_sparse(true)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    // Try to add dense vector to sparse index
    std::vector<float> vector(64, 1.0f);
    VectorData dense_vector_data{DenseVector{vector.data()}};

    int ret = index->add(dense_vector_data, 1);
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
  }

  // Test fetch non-existent document
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    VectorDataBuffer fetched_vector_data;
    int ret = index->fetch(999, &fetched_vector_data);  // Non-existent doc_id
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
  }

  // Test search with invalid vector data
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    // Add a vector first
    std::vector<float> vector(64, 1.0f);
    VectorData vector_data{DenseVector{vector.data()}};
    ASSERT_EQ(0, index->add(vector_data, 1));

    // Try to search with sparse vector in dense index
    std::vector<uint32_t> indices = {0, 1, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    VectorData sparse_query{SparseVector{3, indices.data(), values.data()}};

    SearchResult result;
    FlatQueryParam::Pointer query_param =
        FlatQueryParamBuilder().with_topk(10).with_fetch_vector(false).build();
    int ret = index->search(sparse_query, query_param, &result);
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
  }

  // Test merge with invalid write concurrency
  {
    auto param1 = FlatIndexParamBuilder()
                      .with_metric_type(MetricType::kInnerProduct)
                      .with_data_type(DataType::DT_FP32)
                      .with_dimension(64)
                      .with_is_sparse(false)
                      .build();
    auto index1 = IndexFactory::CreateAndInitIndex(*param1);
    ASSERT_NE(nullptr, index1);
    index1->open("test1.index", {StorageOptions::StorageType::kMMAP, true});

    auto param2 = FlatIndexParamBuilder()
                      .with_metric_type(MetricType::kInnerProduct)
                      .with_data_type(DataType::DT_FP32)
                      .with_dimension(64)
                      .with_is_sparse(false)
                      .build();
    auto index2 = IndexFactory::CreateAndInitIndex(*param2);
    ASSERT_NE(nullptr, index2);
    index2->open("test2.index", {StorageOptions::StorageType::kMMAP, true});

    auto param3 = FlatIndexParamBuilder()
                      .with_metric_type(MetricType::kInnerProduct)
                      .with_data_type(DataType::DT_FP32)
                      .with_dimension(64)
                      .with_is_sparse(false)
                      .build();
    auto index3 = IndexFactory::CreateAndInitIndex(*param3);
    ASSERT_NE(nullptr, index3);
    index3->open("test3.index", {StorageOptions::StorageType::kMMAP, true});

    MergeOptions invalid_options;
    invalid_options.write_concurrency = 0;  // Invalid: must be > 0

    int ret = index3->merge({index1, index2}, IndexFilter(), invalid_options);
    ASSERT_NE(0, ret);

    index1->close();
    index2->close();
    index3->close();
    zvec::test_util::RemoveTestFiles("test1.index");
    zvec::test_util::RemoveTestFiles("test2.index");
    zvec::test_util::RemoveTestFiles("test3.index");
  }

  // Test Vamana search with ef_search == 0 (invalid, ef_search must be > 0)
  {
    auto param = VamanaIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .with_max_degree(32)
                     .with_search_list_size(100)
                     .with_alpha(1.2f)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(64, 1.0f);
    VectorData vector_data{DenseVector{vector.data()}};
    ASSERT_EQ(0, index->add(vector_data, 1));

    VectorData query{DenseVector{vector.data()}};
    auto query_param = VamanaQueryParamBuilder()
                           .with_topk(10)
                           .with_fetch_vector(false)
                           .with_ef_search(0)
                           .build();
    SearchResult result;
    int ret = index->search(query, query_param, &result);
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
  }

  // Test Vamana search with ef_search > 2048 (invalid upper bound)
  {
    auto param = VamanaIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .with_max_degree(32)
                     .with_search_list_size(100)
                     .with_alpha(1.2f)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(64, 1.0f);
    VectorData vector_data{DenseVector{vector.data()}};
    ASSERT_EQ(0, index->add(vector_data, 1));

    VectorData query{DenseVector{vector.data()}};
    auto query_param = VamanaQueryParamBuilder()
                           .with_topk(10)
                           .with_fetch_vector(false)
                           .with_ef_search(4096)
                           .build();
    SearchResult result;
    int ret = index->search(query, query_param, &result);
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
  }

  // Test Vamana search with wrong query param type (HNSWQueryParam instead of
  // VamanaQueryParam)
  {
    auto param = VamanaIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(64)
                     .with_is_sparse(false)
                     .with_max_degree(32)
                     .with_search_list_size(100)
                     .with_alpha(1.2f)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open("test.index", {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(64, 1.0f);
    VectorData vector_data{DenseVector{vector.data()}};
    ASSERT_EQ(0, index->add(vector_data, 1));

    VectorData query{DenseVector{vector.data()}};
    // Intentionally pass an HNSWQueryParam to a Vamana index
    auto wrong_query_param = HNSWQueryParamBuilder()
                                 .with_topk(10)
                                 .with_fetch_vector(false)
                                 .with_ef_search(50)
                                 .build();
    SearchResult result;
    int ret = index->search(query, wrong_query_param, &result);
    ASSERT_NE(0, ret);

    index->close();
    zvec::test_util::RemoveTestFiles("test.index");
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

  zvec::test_util::RemoveTestFiles(index_file_path);

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
    zvec::test_util::RemoveTestFiles(index_file_path);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open(index_file_path, {StorageOptions::StorageType::kMMAP, true});

    VectorData vector_data1;
    vector_data1.vector = DenseVector{vector1.data()};
    ASSERT_EQ(0, index->add(vector_data1, kDocId1));

    VectorData vector_data2;
    vector_data2.vector = DenseVector{vector2.data()};
    ASSERT_EQ(0, index->add(vector_data2, kDocId2));

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{query_vector.data()};
    index->search(query, query_param, &result);

    check_score(result, metric_type);

    index->close();
    zvec::test_util::RemoveTestFiles(index_file_path);
  };

  auto sparse_func = [&](const BaseIndexParam::Pointer &param,
                         const BaseIndexQueryParam::Pointer query_param,
                         MetricType metric_type) {
    zvec::test_util::RemoveTestFiles(index_file_path);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open(index_file_path, {StorageOptions::StorageType::kMMAP, true});

    VectorData vector_data1;
    vector_data1.vector =
        SparseVector{3, reinterpret_cast<const void *>(sparse_indices.data()),
                     vector1.data()};
    ASSERT_EQ(0, index->add(vector_data1, kDocId1));

    VectorData vector_data2;
    vector_data2.vector =
        SparseVector{3, reinterpret_cast<const void *>(sparse_indices.data()),
                     vector2.data()};
    ASSERT_EQ(0, index->add(vector_data2, kDocId2));

    SearchResult result;
    VectorData query;
    query.vector =
        SparseVector{3, reinterpret_cast<const void *>(sparse_indices.data()),
                     query_vector.data()};
    index->search(query, query_param, &result);

    check_score(result, metric_type);

    index->close();
    zvec::test_util::RemoveTestFiles(index_file_path);
  };

  constexpr uint32_t kDimension = 3;

  LOG_INFO("Test DenseVector, MetricType::kInnerProduct");
  dense_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kInnerProduct)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_is_sparse(false)
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  dense_func(HNSWIndexParamBuilder()
                 .with_metric_type(MetricType::kInnerProduct)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_ef_construction(100)
                 .build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kInnerProduct);

  dense_func(VamanaIndexParamBuilder()
                 .with_metric_type(MetricType::kInnerProduct)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_max_degree(32)
                 .with_search_list_size(100)
                 .with_alpha(1.2f)
                 .build(),
             VamanaQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(50)
                 .build(),
             MetricType::kInnerProduct);

  LOG_INFO("Test DenseVector, MetricType::kInnerProduct, QuantizerType::kFP16");
  dense_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kInnerProduct)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_is_sparse(false)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  dense_func(HNSWIndexParamBuilder()
                 .with_metric_type(MetricType::kInnerProduct)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_ef_construction(100)
                 .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
                 .build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kInnerProduct);

  LOG_INFO("Test DenseVector, MetricType::kCosine");
  dense_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kCosine)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_is_sparse(false)
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kCosine);
  dense_func(HNSWIndexParamBuilder()
                 .with_metric_type(MetricType::kCosine)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_ef_construction(100)
                 .build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kCosine);

  LOG_INFO("Test DenseVector, MetricType::kCosine, QuantizerType::kFP16");
  dense_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kCosine)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_is_sparse(false)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kCosine);
  dense_func(HNSWIndexParamBuilder()
                 .with_metric_type(MetricType::kCosine)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_ef_construction(100)
                 .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
                 .build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kCosine);

  LOG_INFO("Test DenseVector, MetricType::kL2sq");
  dense_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_is_sparse(false)
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kL2sq);
  dense_func(HNSWIndexParamBuilder()
                 .with_metric_type(MetricType::kL2sq)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_ef_construction(100)
                 .build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kL2sq);

  LOG_INFO("Test DenseVector, MetricType::kL2sq, QuantizerType::kFP16");
  dense_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_is_sparse(false)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kL2sq);
  dense_func(HNSWIndexParamBuilder()
                 .with_metric_type(MetricType::kL2sq)
                 .with_data_type(DataType::DT_FP32)
                 .with_dimension(kDimension)
                 .with_is_sparse(false)
                 .with_ef_construction(100)
                 .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
                 .build(),
             HNSWQueryParamBuilder()
                 .with_topk(kTopk)
                 .with_fetch_vector(true)
                 .with_ef_search(20)
                 .build(),
             MetricType::kL2sq);

  LOG_INFO("Test SparseVector, MetricType::kInnerProduct");
  sparse_func(
      FlatIndexParamBuilder()
          .with_metric_type(MetricType::kInnerProduct)
          .with_data_type(DataType::DT_FP32)
          .with_is_sparse(true)
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  sparse_func(HNSWIndexParamBuilder()
                  .with_metric_type(MetricType::kInnerProduct)
                  .with_data_type(DataType::DT_FP32)
                  .with_is_sparse(true)
                  .with_ef_construction(100)
                  .build(),
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
          .with_metric_type(MetricType::kInnerProduct)
          .with_data_type(DataType::DT_FP32)
          .with_is_sparse(true)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build(),
      FlatQueryParamBuilder().with_topk(kTopk).with_fetch_vector(true).build(),
      MetricType::kInnerProduct);
  sparse_func(HNSWIndexParamBuilder()
                  .with_metric_type(MetricType::kInnerProduct)
                  .with_data_type(DataType::DT_FP32)
                  .with_is_sparse(true)
                  .with_ef_construction(100)
                  .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
                  .build(),
              HNSWQueryParamBuilder()
                  .with_topk(kTopk)
                  .with_fetch_vector(true)
                  .with_ef_search(20)
                  .build(),
              MetricType::kInnerProduct);
}

#if RABITQ_SUPPORTED
TEST(IndexInterface, HNSWRabitqGeneral) {
  constexpr uint32_t kDimension = 64;
  const std::string index_name{"test_rabitq.index"};
  const std::string cleanup_pattern = index_name + "*";

  auto func = [&](const BaseIndexParam::Pointer &param,
                  const BaseIndexQueryParam::Pointer &query_param) {
    zvec::test_util::RemoveTestFiles(cleanup_pattern);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);

    index->open(index_name, {StorageOptions::StorageType::kMMAP, true});

    std::vector<float> vector(kDimension);
    vector[1] = 1.0f;
    vector[2] = 2.0f;
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ASSERT_TRUE(0 == index->add(vector_data, 233));
    ASSERT_TRUE(0 == index->train());

    SearchResult result;
    VectorData query;
    query.vector = DenseVector{vector.data()};
    index->search(query, query_param, &result);
    ASSERT_EQ(1, result.doc_list_.size());
    ASSERT_EQ(233, result.doc_list_[0].key());

    // Fetch is meaningless for HNSWRabitq
    index->close();
    zvec::test_util::RemoveTestFiles(cleanup_pattern);
  };

  using namespace zvec::core;
  using namespace zvec::ailego;
  auto holder = std::make_shared<
      zvec::core::MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(
      kDimension);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(kDimension);
    for (size_t j = 0; j < kDimension; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  std::shared_ptr<IndexMeta> index_meta_ptr_;
  index_meta_ptr_.reset(
      new (std::nothrow) IndexMeta(IndexMeta::DataType::DT_FP32, kDimension));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);

  // HNSWRabitq with default total_bits
  func(HNSWRabitqIndexParamBuilder()
           .with_metric_type(MetricType::kL2sq)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .with_provider(holder)
           .with_reformer(index_reformer)
           .build(),
       HNSWRabitqQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(false)
           .with_ef_search(50)
           .build());

  // HNSWRabitq with InnerProduct metric
  func(HNSWRabitqIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .with_provider(holder)
           .with_reformer(index_reformer)
           .build(),
       HNSWRabitqQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(false)
           .with_ef_search(50)
           .build());

  // HNSWRabitq with custom total_bits
  // Reformer must be re-created with matching total_bits to keep ex_bits
  // consistent between reformer and entity.
  RabitqConverter converter2;
  Params converter2_params;
  converter2_params.set(PARAM_RABITQ_TOTAL_BITS, 2u);
  converter2.init(*index_meta_ptr_, converter2_params);
  ASSERT_EQ(converter2.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer2;
  ASSERT_EQ(converter2.to_reformer(&index_reformer2), 0);

  func(HNSWRabitqIndexParamBuilder()
           .with_metric_type(MetricType::kL2sq)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .with_total_bits(2)
           .with_provider(holder)
           .with_reformer(index_reformer2)
           .build(),
       HNSWRabitqQueryParamBuilder()
           .with_topk(10)
           .with_fetch_vector(false)
           .with_ef_search(50)
           .build());
}
#endif

// Verify that enabling use_contiguous_memory on HNSW / Vamana index params at
// the interface layer is correctly propagated to the underlying streamer and
// yields a working build -> close -> reopen-for-search pipeline. This guards
// the interface -> streamer param binding introduced for contiguous memory
// mode.
TEST(IndexInterface, ContiguousMemoryEndToEnd) {
  constexpr uint32_t kDimension = 32;
  constexpr uint32_t kNumDocs = 500;
  constexpr int kTopk = 10;
  const std::string index_name{"test_contiguous.index"};

  // build_then_search builds an index from scratch (with use_contiguous_memory
  // possibly enabled), closes it, then reopens with the same params and runs a
  // search for each inserted vector, asserting top-1 is itself.
  auto build_then_search =
      [&](const BaseIndexParam::Pointer &param,
          const BaseIndexQueryParam::Pointer &query_param) {
        zvec::test_util::RemoveTestFiles(index_name);

        // Phase 1: build & persist.
        {
          auto index = IndexFactory::CreateAndInitIndex(*param);
          ASSERT_NE(nullptr, index);
          ASSERT_EQ(0, index->open(index_name,
                                   {StorageOptions::StorageType::kMMAP, true}));

          std::vector<float> vec(kDimension);
          for (uint32_t i = 0; i < kNumDocs; ++i) {
            for (uint32_t d = 0; d < kDimension; ++d) {
              vec[d] = static_cast<float>(i);
            }
            VectorData data{DenseVector{vec.data()}};
            ASSERT_EQ(0, index->add(data, i));
          }
          ASSERT_EQ(0, index->train());
          ASSERT_EQ(0, index->close());
        }

        // Phase 2: reopen with same params (contiguous memory takes effect
        // here) and search.
        {
          auto index = IndexFactory::CreateAndInitIndex(*param);
          ASSERT_NE(nullptr, index);
          ASSERT_EQ(0,
                    index->open(index_name,
                                {StorageOptions::StorageType::kMMAP, false}));

          std::vector<float> q(kDimension);
          for (uint32_t i = 0; i < kNumDocs; i += 50) {
            for (uint32_t d = 0; d < kDimension; ++d) {
              q[d] = static_cast<float>(i);
            }
            VectorData query{DenseVector{q.data()}};
            SearchResult result;
            ASSERT_EQ(0, index->search(query, query_param, &result));
            ASSERT_GT(result.doc_list_.size(), 0UL);
            ASSERT_EQ(i, result.doc_list_[0].key());
          }
          ASSERT_EQ(0, index->close());
        }

        zvec::test_util::RemoveTestFiles(index_name);
      };

  // HNSW + use_contiguous_memory=true
  build_then_search(HNSWIndexParamBuilder()
                        .with_metric_type(MetricType::kL2sq)
                        .with_data_type(DataType::DT_FP32)
                        .with_dimension(kDimension)
                        .with_is_sparse(false)
                        .with_m(16)
                        .with_ef_construction(64)
                        .with_use_contiguous_memory(true)
                        .build(),
                    HNSWQueryParamBuilder()
                        .with_topk(kTopk)
                        .with_fetch_vector(false)
                        .with_ef_search(64)
                        .build());

  // HNSW + use_contiguous_memory=false (baseline, same harness)
  build_then_search(HNSWIndexParamBuilder()
                        .with_metric_type(MetricType::kL2sq)
                        .with_data_type(DataType::DT_FP32)
                        .with_dimension(kDimension)
                        .with_is_sparse(false)
                        .with_m(16)
                        .with_ef_construction(64)
                        .with_use_contiguous_memory(false)
                        .build(),
                    HNSWQueryParamBuilder()
                        .with_topk(kTopk)
                        .with_fetch_vector(false)
                        .with_ef_search(64)
                        .build());

  // Vamana + use_contiguous_memory=true
  build_then_search(VamanaIndexParamBuilder()
                        .with_metric_type(MetricType::kL2sq)
                        .with_data_type(DataType::DT_FP32)
                        .with_dimension(kDimension)
                        .with_is_sparse(false)
                        .with_max_degree(32)
                        .with_search_list_size(100)
                        .with_alpha(1.2f)
                        .with_use_contiguous_memory(true)
                        .build(),
                    VamanaQueryParamBuilder()
                        .with_topk(kTopk)
                        .with_fetch_vector(false)
                        .with_ef_search(64)
                        .build());

  // Vamana + use_contiguous_memory=false (baseline, same harness)
  build_then_search(VamanaIndexParamBuilder()
                        .with_metric_type(MetricType::kL2sq)
                        .with_data_type(DataType::DT_FP32)
                        .with_dimension(kDimension)
                        .with_is_sparse(false)
                        .with_max_degree(32)
                        .with_search_list_size(100)
                        .with_alpha(1.2f)
                        .with_use_contiguous_memory(false)
                        .build(),
                    VamanaQueryParamBuilder()
                        .with_topk(kTopk)
                        .with_fetch_vector(false)
                        .with_ef_search(64)
                        .build());
}

class TestVectorSource : public zvec::core::VectorSource {
 public:
  TestVectorSource(const float *base, uint32_t dim) : base_(base), dim_(dim) {}

  const void *get_vector(uint32_t node_id) const override {
    return base_ + static_cast<size_t>(node_id) * dim_;
  }

 private:
  const float *base_;
  uint32_t dim_;
};

TEST(IndexInterface, ExternalVectorEndToEnd) {
  constexpr uint32_t kDimension = 64;
  constexpr uint32_t kNumVectors = 100;
  const std::string index_name{"test_external.index"};

  std::vector<float> all_vectors(kDimension * kNumVectors);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    for (uint32_t d = 0; d < kDimension; ++d) {
      all_vectors[i * kDimension + d] =
          static_cast<float>(i * kDimension + d) * 0.01f;
    }
  }

  TestVectorSource source(all_vectors.data(), kDimension);

  zvec::test_util::RemoveTestFiles(index_name + "*");

  auto param = HNSWIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_is_sparse(false)
                   .with_ef_construction(100)
                   .with_use_external_vector(true)
                   .build();

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);

  index->open(index_name, {StorageOptions::StorageType::kMMAP, true});

  for (uint32_t i = 0; i < kNumVectors; ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{all_vectors.data() + i * kDimension};
    int ret = index->add_with_source(vector_data, i, source);
    ASSERT_EQ(0, ret) << "AddWithSource failed for doc_id=" << i;
  }

  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(5)
                         .with_fetch_vector(false)
                         .with_ef_search(50)
                         .build();

  VectorData query;
  query.vector = DenseVector{all_vectors.data()};
  SearchResult result;
  int ret = index->search_with_source(query, query_param, source, &result);
  ASSERT_EQ(0, ret);
  ASSERT_GE(result.doc_list_.size(), 1u);
  ASSERT_EQ(0u, result.doc_list_[0].key());
  ASSERT_FLOAT_EQ(0.0f, result.doc_list_[0].score());

  VectorData query2;
  query2.vector = DenseVector{all_vectors.data() + 50 * kDimension};
  SearchResult result2;
  ret = index->search_with_source(query2, query_param, source, &result2);
  ASSERT_EQ(0, ret);
  ASSERT_GE(result2.doc_list_.size(), 1u);
  ASSERT_EQ(50u, result2.doc_list_[0].key());
  ASSERT_FLOAT_EQ(0.0f, result2.doc_list_[0].score());

  index->close();

  auto index2 = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index2);
  index2->open(index_name, {StorageOptions::StorageType::kMMAP, false});

  SearchResult result3;
  ret = index2->search_with_source(query, query_param, source, &result3);
  ASSERT_EQ(0, ret);
  ASSERT_GE(result3.doc_list_.size(), 1u);
  ASSERT_EQ(0u, result3.doc_list_[0].key());
  ASSERT_FLOAT_EQ(0.0f, result3.doc_list_[0].score());

  index2->close();
  zvec::test_util::RemoveTestFiles(index_name + "*");
}

TEST(IndexInterface, ExternalVectorInnerProduct) {
  constexpr uint32_t kDimension = 16;
  constexpr uint32_t kNumVectors = 10;
  const std::string index_name{"test_external_ip.index"};

  std::vector<float> all_vectors(kDimension * kNumVectors, 0.0f);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    all_vectors[i * kDimension + i % kDimension] = static_cast<float>(i + 1);
  }

  TestVectorSource source(all_vectors.data(), kDimension);

  zvec::test_util::RemoveTestFiles(index_name + "*");

  auto param = HNSWIndexParamBuilder()
                   .with_metric_type(MetricType::kInnerProduct)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_is_sparse(false)
                   .with_ef_construction(100)
                   .with_use_external_vector(true)
                   .build();

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  index->open(index_name, {StorageOptions::StorageType::kMMAP, true});

  for (uint32_t i = 0; i < kNumVectors; ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{all_vectors.data() + i * kDimension};
    ASSERT_EQ(0, index->add_with_source(vector_data, i, source));
  }

  std::vector<float> query_vec(kDimension, 0.0f);
  query_vec[0] = 1.0f;
  VectorData query;
  query.vector = DenseVector{query_vec.data()};

  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(1)
                         .with_fetch_vector(false)
                         .with_ef_search(50)
                         .build();

  SearchResult result;
  ASSERT_EQ(0, index->search_with_source(query, query_param, source, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  ASSERT_EQ(0u, result.doc_list_[0].key());
  ASSERT_FLOAT_EQ(1.0f, result.doc_list_[0].score());

  index->close();
  zvec::test_util::RemoveTestFiles(index_name + "*");
}

TEST(IndexInterface, ExternalVectorFastSearchRecallRegression) {
  constexpr uint32_t kDimension = 64;
  constexpr uint32_t kNumVectors = 2000;
  constexpr uint32_t kTopk = 20;
  const std::string index_name{"test_external_fast_search.index"};

  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> all_vectors(kDimension * kNumVectors);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    float *vector = all_vectors.data() + i * kDimension;
    float squared_norm = 0.0f;
    for (uint32_t d = 0; d < kDimension; ++d) {
      vector[d] = distribution(generator);
      squared_norm += vector[d] * vector[d];
    }
    const float norm = std::sqrt(squared_norm);
    for (uint32_t d = 0; d < kDimension; ++d) {
      vector[d] /= norm;
    }
  }

  std::vector<float> query_vector(kDimension);
  float query_squared_norm = 0.0f;
  for (float &value : query_vector) {
    value = distribution(generator);
    query_squared_norm += value * value;
  }
  const float query_norm = std::sqrt(query_squared_norm);
  for (float &value : query_vector) {
    value /= query_norm;
  }

  std::vector<std::pair<float, uint32_t>> exact_results;
  exact_results.reserve(kNumVectors);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    const float *vector = all_vectors.data() + i * kDimension;
    const float score = std::inner_product(query_vector.begin(),
                                           query_vector.end(), vector, 0.0f);
    exact_results.emplace_back(score, i);
  }
  std::sort(exact_results.begin(), exact_results.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first > rhs.first;
              }
              return lhs.second < rhs.second;
            });

  TestVectorSource source(all_vectors.data(), kDimension);
  zvec::test_util::RemoveTestFiles(index_name + "*");

  auto param = HNSWIndexParamBuilder()
                   .with_metric_type(MetricType::kInnerProduct)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_is_sparse(false)
                   .with_m(16)
                   .with_ef_construction(200)
                   .with_use_external_vector(true)
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));

  for (uint32_t i = 0; i < kNumVectors; ++i) {
    VectorData vector_data{DenseVector{all_vectors.data() + i * kDimension}};
    ASSERT_EQ(0, index->add_with_source(vector_data, i, source));
  }

  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(kTopk)
                         .with_fetch_vector(true)
                         .with_ef_search(500)
                         .build();
  VectorData query{DenseVector{query_vector.data()}};
  SearchResult result;
  ASSERT_EQ(0, index->search_with_source(query, query_param, source, &result));
  ASSERT_EQ(kTopk, result.doc_list_.size());

  uint32_t recall_count = 0;
  for (const auto &doc : result.doc_list_) {
    const uint32_t key = doc.key();
    ASSERT_LT(key, kNumVectors);
    const float *vector = all_vectors.data() + key * kDimension;
    const float exact_score = std::inner_product(
        query_vector.begin(), query_vector.end(), vector, 0.0f);
    EXPECT_NEAR(exact_score, doc.score(), 1e-5f) << "key=" << key;

    for (uint32_t i = 0; i < kTopk; ++i) {
      if (exact_results[i].second == key) {
        ++recall_count;
        break;
      }
    }
  }
  EXPECT_GE(recall_count, 18u);

  index->close();
  zvec::test_util::RemoveTestFiles(index_name + "*");
}

TEST(IndexInterface, IsDirty) {
  constexpr uint32_t kDimension = 16;
  const std::string index_name{"test_is_dirty.index"};

  auto test = [&](const BaseIndexParam::Pointer &param) {
    zvec::test_util::RemoveTestFiles(index_name);

    // Before open: not dirty (no storage)
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_FALSE(index->is_dirty());
    }

    // Create the index file: dirty from initial metadata writes
    {
      auto index = IndexFactory::CreateAndInitIndex(*param);
      index->open(index_name, {StorageOptions::StorageType::kMMAP, true});
      ASSERT_TRUE(index->is_dirty());
      ASSERT_EQ(0, index->flush());
      ASSERT_FALSE(index->is_dirty());
      index->close();
    }

    // Reopen existing file: should be clean
    auto index = IndexFactory::CreateAndInitIndex(*param);
    index->open(index_name, {StorageOptions::StorageType::kMMAP, false});
    ASSERT_FALSE(index->is_dirty());

    // Add a vector: should become dirty
    std::vector<float> vec(kDimension, 1.0f);
    VectorData vd;
    vd.vector = DenseVector{vec.data()};
    ASSERT_EQ(0, index->add(vd, 1));
    ASSERT_TRUE(index->is_dirty());

    // Flush: should become clean
    ASSERT_EQ(0, index->flush());
    ASSERT_FALSE(index->is_dirty());

    // Add another vector: dirty again
    ASSERT_EQ(0, index->add(vd, 2));
    ASSERT_TRUE(index->is_dirty());

    // Close flushes implicitly, verify no crash
    index->close();
    zvec::test_util::RemoveTestFiles(index_name);
  };

  test(FlatIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .build());

  test(HNSWIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_ef_construction(100)
           .build());
}

TEST(IndexInterface, IsDirtyBufferPool) {
  constexpr uint32_t kDimension = 16;
  const std::string index_name{"test_is_dirty_bp.index"};

  zvec::test_util::RemoveTestFiles(index_name);

  // First create and populate the index with MMAP storage
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(kDimension)
                     .with_is_sparse(false)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    index->open(index_name, {StorageOptions::StorageType::kMMAP, true});
    std::vector<float> vec(kDimension, 1.0f);
    VectorData vd;
    vd.vector = DenseVector{vec.data()};
    ASSERT_EQ(0, index->add(vd, 1));
    index->close();
  }

  // Reopen with BufferPool storage in writable mode
  {
    auto param = FlatIndexParamBuilder()
                     .with_metric_type(MetricType::kInnerProduct)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(kDimension)
                     .with_is_sparse(false)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    index->open(index_name, {StorageOptions::StorageType::kBufferPool, true});

    ASSERT_FALSE(index->is_dirty());

    std::vector<float> vec(kDimension, 2.0f);
    VectorData vd;
    vd.vector = DenseVector{vec.data()};
    ASSERT_EQ(0, index->add(vd, 2));
    ASSERT_TRUE(index->is_dirty());

    ASSERT_EQ(0, index->flush());
    ASSERT_FALSE(index->is_dirty());

    index->close();
  }

  zvec::test_util::RemoveTestFiles(index_name);
}

TEST(IndexInterface, BuilderSetsAllBaseFields) {
  auto param =
      FlatIndexParamBuilder()
          .with_version(42)
          .with_index_type(IndexType::kFlat)
          .with_metric_type(MetricType::kInnerProduct)
          .with_dimension(128)
          .with_data_type(DataType::DT_FP32)
          .with_is_sparse(true)
          .with_use_id_map(false)
          .with_use_external_vector(true)
          .with_preprocess_param(PreprocessorParam(PreprocessorType::kPCA))
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build();

  ASSERT_NE(nullptr, param);
  EXPECT_EQ(42, param->version);
  EXPECT_EQ(IndexType::kFlat, param->index_type);
  EXPECT_EQ(MetricType::kInnerProduct, param->metric_type);
  EXPECT_EQ(128, param->dimension);
  EXPECT_EQ(DataType::DT_FP32, param->data_type);
  EXPECT_TRUE(param->is_sparse);
  EXPECT_FALSE(param->use_id_map);
  EXPECT_TRUE(param->use_external_vector);
  EXPECT_EQ(PreprocessorType::kPCA, param->preprocess_param.type);
  EXPECT_EQ(QuantizerType::kFP16, param->quantizer_param->type);
}

TEST(IndexInterface, QuantizerParamDefaultIsNull) {
  auto param = FlatIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_dimension(64)
                   .build();

  ASSERT_NE(nullptr, param);
  EXPECT_EQ(nullptr, param->quantizer_param);
}

TEST(IndexInterface, QuantizerParamPqFields) {
  auto param = FlatIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_dimension(128)
                   .with_quantizer_param(PqQuantizerParam(16, 8))
                   .build();

  ASSERT_NE(nullptr, param);
  ASSERT_NE(nullptr, param->quantizer_param);
  EXPECT_EQ(QuantizerType::kPQ, param->quantizer_param->type);

  // the builder must keep the concrete type instead of slicing it
  auto pq_param =
      std::dynamic_pointer_cast<PqQuantizerParam>(param->quantizer_param);
  ASSERT_NE(nullptr, pq_param);
  EXPECT_EQ(16, pq_param->num_chunk);
  EXPECT_EQ(8, pq_param->num_bits);

  // enable_rotate is a common field, setting it keeps the concrete type
  auto rotated_param = FlatIndexParamBuilder()
                           .with_quantizer_param(PqQuantizerParam(16, 8))
                           .with_enable_rotate(true)
                           .build();
  ASSERT_NE(nullptr, rotated_param->quantizer_param);
  EXPECT_TRUE(rotated_param->quantizer_param->enable_rotate);
  EXPECT_NE(nullptr, std::dynamic_pointer_cast<PqQuantizerParam>(
                         rotated_param->quantizer_param));
}

TEST(IndexInterface, QuantizerParamPqJsonRoundTrip) {
  auto param = FlatIndexParamBuilder()
                   .with_index_type(IndexType::kFlat)
                   .with_metric_type(MetricType::kL2sq)
                   .with_dimension(128)
                   .with_data_type(DataType::DT_FP32)
                   .with_quantizer_param(PqQuantizerParam(32, 4))
                   .build();

  auto deserialized_param =
      IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json());
  ASSERT_NE(nullptr, deserialized_param);
  EXPECT_EQ(param->serialize_to_json(),
            deserialized_param->serialize_to_json());
  EXPECT_EQ(param->serialize_to_json(true),
            deserialized_param->serialize_to_json(true));

  auto pq_param = std::dynamic_pointer_cast<PqQuantizerParam>(
      deserialized_param->quantizer_param);
  ASSERT_NE(nullptr, pq_param);
  EXPECT_EQ(QuantizerType::kPQ, pq_param->type);
  EXPECT_EQ(32, pq_param->num_chunk);
  EXPECT_EQ(4, pq_param->num_bits);
}

TEST(IndexInterface, QuantizerParamLegacyJsonCompat) {
  // legacy json only carries the common fields
  const std::string json_str =
      R"({"index_type":"kFlat","metric_type":"kL2sq","dimension":128,)"
      R"("data_type":"DT_FP32",)"
      R"("quantizer_param":{"type":"kInt8","enable_rotate":true}})";

  auto param = IndexFactory::DeserializeIndexParamFromJson(json_str);
  ASSERT_NE(nullptr, param);
  ASSERT_NE(nullptr, param->quantizer_param);
  EXPECT_EQ(QuantizerType::kInt8, param->quantizer_param->type);
  EXPECT_TRUE(param->quantizer_param->enable_rotate);
  EXPECT_EQ(nullptr, std::dynamic_pointer_cast<PqQuantizerParam>(
                         param->quantizer_param));
}

TEST(IndexInterface, BuilderChainingReturnsCorrectType) {
  HNSWIndexParamBuilder builder;
  auto &ref = builder.with_version(1)
                  .with_metric_type(MetricType::kL2sq)
                  .with_dimension(64)
                  .with_m(16)
                  .with_ef_construction(200);
  auto param = ref.build();

  ASSERT_NE(nullptr, param);
  EXPECT_EQ(1, param->version);
  EXPECT_EQ(MetricType::kL2sq, param->metric_type);
  EXPECT_EQ(64, param->dimension);
  EXPECT_EQ(16, param->m);
  EXPECT_EQ(200, param->ef_construction);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
