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
  ASSERT_EQ(0, calculator.quantized_dist(ids, 2, distances));
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

TEST(DiskAnnReopen, RequiresMatchingVectorLayout) {
  const std::string path = "diskann_reopen_layout.index";
  constexpr int dim = 32;
  for (auto quantizer :
       {QuantizerType::kNone, QuantizerType::kFP16, QuantizerType::kRabitq}) {
    if (quantizer == QuantizerType::kRabitq && !RABITQ_SUPPORTED) continue;
    SCOPED_TRACE(static_cast<int>(quantizer));
    test_util::RemoveTestFiles(path);
    auto params = Parameters(MetricType::kL2sq, 3, dim);
    if (quantizer != QuantizerType::kRabitq)
      params->quantizer_param = std::make_shared<QuantizerParam>(quantizer);
    auto writer = std::make_shared<SmallDiskAnn>();
    ASSERT_EQ(0, writer->initialize(*params));
    Index::Pointer index = writer;
    ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < 300; ++i) {
      std::fill(row.begin(), row.end(), static_cast<float>(i));
      ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, i));
    }
    ASSERT_EQ(0, index->train());
    ASSERT_EQ(0, index->close());

    const StorageOptions read{StorageOptions::StorageType::kMMAP, false, true};
    if (quantizer == QuantizerType::kRabitq)
      params->quantizer_param = std::make_shared<QuantizerParam>();
    auto wrong_dim = *params, wrong_type = *params, wrong_metric = *params;
    wrong_dim.dimension = dim / 2;
    // Raw FP16 and FP32 converted to FP16 have identical physical widths.
    wrong_type.data_type = DataType::DT_FP16;
    wrong_type.quantizer_param = std::make_shared<QuantizerParam>();
    wrong_metric.metric_type = MetricType::kInnerProduct;
    for (const auto *incompatible : {&wrong_dim, &wrong_type, &wrong_metric}) {
      auto reader = IndexFactory::CreateAndInitIndex(*incompatible);
      ASSERT_TRUE(reader);
      ASSERT_EQ(core::IndexError_Mismatch, reader->open(path, read));
      EXPECT_EQ(core::IndexError_Mismatch, reader->open(path, read));
    }
    auto reader = IndexFactory::CreateAndInitIndex(*params);
    ASSERT_TRUE(reader);
    ASSERT_EQ(0, reader->open(path, read));
    VectorDataBuffer fetched;
    ASSERT_EQ(0, reader->fetch(17, &fetched));
    std::fill(row.begin(), row.end(), 17.f);
    EXPECT_EQ(std::string(reinterpret_cast<const char *>(row.data()),
                          dim * sizeof(float)),
              std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
    auto query = std::make_shared<DiskAnnQueryParam>();
    query->topk = 1;
    SearchResult result;
    ASSERT_EQ(
        0, reader->search(VectorData{DenseVector{row.data()}}, query, &result));
    ASSERT_EQ(1u, result.doc_list_.size());
    EXPECT_EQ(17u, result.doc_list_.front().key());
    ASSERT_EQ(0, reader->close());
    test_util::RemoveTestFiles(path);
  }
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

TEST(DiskAnnRabitq, MergeWithSingleWorkerPool) {
  constexpr size_t dim = 64;
  const std::string source_path = "diskann_single_worker.flat";
  const std::string target_path = "diskann_single_worker.index";
  test_util::RemoveTestFiles(source_path);
  test_util::RemoveTestFiles(target_path);
  auto source =
      IndexFactory::CreateAndInitIndex(*FlatIndexParamBuilder()
                                            .with_dimension(dim)
                                            .with_metric_type(MetricType::kL2sq)
                                            .with_data_type(DataType::DT_FP32)
                                            .build());
  SmallDiskAnn diskann;
  Index &target = diskann;
  ASSERT_TRUE(source);
  ASSERT_EQ(0, diskann.initialize(*Parameters(MetricType::kL2sq, 7, dim)));
  const StorageOptions create{StorageOptions::StorageType::kMMAP, true};
  ASSERT_EQ(0, source->open(source_path, create));
  ASSERT_EQ(0, target.open(target_path, create));
  for (uint32_t id = 0; id < 64; ++id) {
    std::vector<float> v(dim, static_cast<float>(id));
    ASSERT_EQ(0, source->add(VectorData{DenseVector{v.data()}}, id));
  }
  ailego::ThreadPool pool(1, false);
  ASSERT_EQ(0, target.merge({source}, {}, {1, &pool}));
  EXPECT_EQ(64u, target.get_doc_count());
  ASSERT_EQ(0, source->close());
  ASSERT_EQ(0, target.close());
  test_util::RemoveTestFiles(source_path);
  test_util::RemoveTestFiles(target_path);
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
TEST(DiskAnnRabitq, UnsupportedBuildReopensPqAndRejectsNewRabitq) {
  const std::string path = "diskann_unsupported_rabitq.index";
  test_util::RemoveTestFiles(path);
  const StorageOptions create{StorageOptions::StorageType::kMMAP, true};
  auto params = Parameters(MetricType::kL2sq, 3, 32);
  auto reader = IndexFactory::CreateAndInitIndex(*params);
  ASSERT_TRUE(reader);
  params->quantizer_param = std::make_shared<QuantizerParam>();
  params->pq_chunk_num = 4;
  auto writer = std::make_shared<SmallDiskAnn>();
  ASSERT_EQ(0, writer->initialize(*params));
  Index::Pointer index = writer;
  ASSERT_EQ(0, index->open(path, create));
  std::vector<float> row(32);
  for (uint32_t id = 0; id < 300; ++id) {
    std::fill(row.begin(), row.end(), static_cast<float>(id));
    ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, id));
  }
  ASSERT_EQ(0, index->train());
  ASSERT_EQ(0, index->close());
  // Reject create before touching the existing PQ file.
  EXPECT_EQ(core::IndexError_Unsupported, reader->open(path, create));
  ASSERT_EQ(0, reader->open(path, {create.type, false, true}));
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 64;
  std::fill(row.begin(), row.end(), 17.0f);
  SearchResult result;
  ASSERT_EQ(
      0, reader->search(VectorData{DenseVector{row.data()}}, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(17u, result.doc_list_.front().key());
  ASSERT_EQ(0, reader->close());
  test_util::RemoveTestFiles(path);
}
#endif
}  // namespace
}  // namespace zvec::core_interface
#endif
