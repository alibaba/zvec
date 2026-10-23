// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/diskann/diskann_dist_calculator.h"
#include "algorithm/diskann/diskann_params.h"
#include "algorithm/diskann/diskann_quantizer.h"
#include "algorithm/diskann/diskann_util.h"
#include "tests/test_util.h"
#include "turbo/quantizer/fp32_quantizer/fp32_quantizer.h"

#if DISKANN_SUPPORTED
namespace zvec::core_interface {
namespace {

TEST(DiskAnnRabitq, ByteStrideIsDistinctFromPqChunkCount) {
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 128);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  core::DiskAnnPqMeta codes;
  codes.chunk_num = 192;
  codes.quantizer_meta_buffer_size = sizeof(turbo::QuantizerSerHeader) + 100;
  bool legacy = false;
  EXPECT_NE(0,
            core::DiskAnnQuantizer::normalize_pq_meta(meta, &codes, &legacy));
  meta.set_quantizer("RabitqQuantizer", 0, ailego::Params{});
  EXPECT_EQ(0,
            core::DiskAnnQuantizer::normalize_pq_meta(meta, &codes, &legacy));
  EXPECT_FALSE(legacy);
  codes.chunk_num = uint64_t{1} << 32;
  EXPECT_NE(0,
            core::DiskAnnQuantizer::normalize_pq_meta(meta, &codes, &legacy));

  core::IndexMeta coarse;
  meta.set_metric("InnerProduct", 0, ailego::Params{});
  ASSERT_EQ(0, core::DiskAnnQuantizer::candidate_quantizer_meta(meta, &coarse));
  EXPECT_EQ("InnerProduct", coarse.metric_name());
  meta.set_quantizer("", 0, ailego::Params{});
  ASSERT_EQ(0, core::DiskAnnQuantizer::candidate_quantizer_meta(meta, &coarse));
  EXPECT_EQ("SquaredEuclidean", coarse.metric_name());
}

// PQ used to ignore this argument. RaBitQ validates its asymmetric layout.
class CheckedQueryQuantizer : public turbo::Fp32Quantizer {
 public:
  turbo::DistanceImpl distance(
      const void *query, const core::IndexQueryMeta &meta) const override {
    if (meta.element_size() != quantized_query_vector_length()) return {};
    return Fp32Quantizer::distance(query, meta);
  }
};

TEST(DiskAnnRabitq, CandidateDistanceReceivesQueryLayout) {
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 2);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  auto metric = core::IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(meta, {}));
  CheckedQueryQuantizer quantizer;
  ASSERT_EQ(0, quantizer.init(meta, {}));
  core::DistCalculator calculator(nullptr, meta, metric);
  const float codes[] = {1, 2, 3, 4};
  const float query[] = {1, 2};
  calculator.bind_quantizer(&quantizer,
                            reinterpret_cast<const uint8_t *>(codes), 8);
  ASSERT_EQ(0, calculator.quantize_query(query));
  const core::diskann_id_t ids[] = {0, 1};
  float distances[2];
  calculator.quantized_dist(ids, 2, distances);
  EXPECT_FLOAT_EQ(0, distances[0]);
  EXPECT_FLOAT_EQ(8, distances[1]);
}

class SmallDiskAnn : public DiskAnnIndex {
 public:
  int initialize(const BaseIndexParam &param) {
    proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_THREAD_COUNT, 2U);
    return init(param);
  }
};

auto Parameters(MetricType metric, int bits, int dim = 129) {
  RabitqQuantizerParam quantizer(bits);
  quantizer.num_clusters = 4;
  quantizer.sample_count = 64;
  quantizer.niters = 4;
  return DiskAnnIndexParamBuilder()
      .with_dimension(dim)
      .with_data_type(DataType::DT_FP32)
      .with_metric_type(metric)
      .with_max_degree(16)
      .with_list_size(100)
      .with_pq_chunk_num(0)
      .with_quantizer_param(quantizer)
      .build();
}

#if RABITQ_SUPPORTED
TEST(DiskAnnRabitq, BuildAdapterPreservesFixedPrecisionAndSettings) {
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 129);
  meta.set_metric("InnerProduct", 0, ailego::Params{});
  RabitqQuantizerParam settings(9);
  settings.num_clusters = 4;
  settings.sample_count = 64;
  settings.niters = 4;
  ailego::Params params;
  params.set(core::PARAM_DISKANN_BUILDER_QUANTIZER_CLASS, "RabitqQuantizer");
  params.set(core::PARAM_DISKANN_BUILDER_QUANTIZER_PARAMS,
             settings.to_params());
  params.set(core::PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 10000U);
  core::DiskAnnQuantizer adapter;
  ASSERT_EQ(0, adapter.init(&meta, params));
  ASSERT_EQ("RabitqQuantizer", meta.quantizer_name());
  ASSERT_EQ(0, adapter.prepare(100, false, 0, 1));
  const size_t stride = adapter.code_size();
  EXPECT_GT(stride, 129u);
  EXPECT_EQ(core::IndexError_NoMemory,
            adapter.prepare(100, true, stride * 100 - 1, 1));
  ASSERT_EQ(0, adapter.prepare(100, true, stride * 100, 1));
  EXPECT_EQ(stride, adapter.code_size());
  core::DiskAnnQuantizer inherited;
  ASSERT_EQ(0, inherited.init(&meta, {}));
  ASSERT_EQ(0, inherited.prepare(100, false, 0, 1));
  EXPECT_EQ(stride, inherited.code_size());
}

TEST(DiskAnnRabitq, BuildFilterFetchAndReopenWithDefaultParameters) {
  constexpr uint32_t count = 96;
  constexpr size_t dim = 129;
  std::mt19937 random(8121);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
  for (auto &vector : vectors) {
    double norm = 0;
    for (auto &value : vector) {
      value = normal(random);
      norm += value * value;
    }
    for (auto &value : vector) value /= std::sqrt(norm);
  }
  for (auto metric :
       {MetricType::kL2sq, MetricType::kInnerProduct, MetricType::kCosine}) {
    for (int bits : {3, 9}) {
      for (auto storage_type : {StorageOptions::StorageType::kMMAP,
                                StorageOptions::StorageType::kBufferPool}) {
        SCOPED_TRACE(::testing::Message()
                     << int(metric) << ':' << bits << ':' << int(storage_type));
        const std::string path = "diskann_rabitq.index";
        test_util::RemoveTestFiles(path);
        auto params = Parameters(metric, bits);
        auto initial = std::make_shared<SmallDiskAnn>();
        ASSERT_EQ(0, initial->initialize(*params));
        Index::Pointer index = initial;
        ASSERT_EQ(0, index->open(path, {storage_type, true}));
        for (uint32_t i = 0; i < count; ++i)
          ASSERT_EQ(0,
                    index->add(VectorData{DenseVector{vectors[i].data()}}, i));
        ASSERT_EQ(0, index->train());
        auto stored_meta = index->index_searcher()->meta();
        ASSERT_EQ("RabitqQuantizer", stored_meta.quantizer_name());
        std::vector<uint64_t> previous;
        for (int reopen = 0; reopen < 2; ++reopen) {
          auto query = std::make_shared<DiskAnnQueryParam>();
          query->topk = 10;
          query->list_size = count;
          query->fetch_vector = true;
          SearchResult result;
          ASSERT_EQ(0,
                    index->search(VectorData{DenseVector{vectors[17].data()}},
                                  query, &result));
          ASSERT_EQ(10u, result.doc_list_.size());
          EXPECT_EQ(17u, result.doc_list_[0].key());
          std::vector<uint64_t> ids;
          for (const auto &doc : result.doc_list_) ids.push_back(doc.key());
          if (reopen)
            EXPECT_EQ(previous, ids);
          else
            previous = ids;
          VectorDataBuffer fetched;
          ASSERT_EQ(0, index->fetch(17, &fetched));
          const auto &data =
              std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
          ASSERT_EQ(dim * sizeof(float), data.size());
          const auto *actual = reinterpret_cast<const float *>(data.data());
          for (size_t d = 0; d < dim; ++d)
            EXPECT_NEAR(vectors[17][d], actual[d], 1e-6);
          query->filter = std::make_shared<IndexFilter>();
          query->filter->set([](uint64_t key) { return key < 48; });
          ASSERT_EQ(0,
                    index->search(VectorData{DenseVector{vectors[17].data()}},
                                  query, &result));
          ASSERT_FALSE(result.doc_list_.empty());
          for (const auto &doc : result.doc_list_) EXPECT_GE(doc.key(), 48u);
          ASSERT_EQ(0, index->close());
          if (!reopen) {
            // Persisted metadata must select RaBitQ even with PQ defaults.
            params->quantizer_param = std::make_shared<QuantizerParam>();
            index = IndexFactory::CreateAndInitIndex(*params);
            ASSERT_TRUE(index);
            ASSERT_EQ(0, index->open(path, {storage_type, false}));
          }
        }
        test_util::RemoveTestFiles(path);
      }
    }
  }
}
#else
TEST(DiskAnnRabitq, UnsupportedPlatformRejectsInsteadOfSilentlyBuildingPq) {
  auto params = Parameters(MetricType::kL2sq, 3);
  SmallDiskAnn index;
  EXPECT_NE(0, index.initialize(*params));
}
#endif
}  // namespace
}  // namespace zvec::core_interface
#endif
