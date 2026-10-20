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
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/common/pq_quantizer/packed_code_quantizer.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/ivf/ivf_builder.h"
#include "algorithm/ivf/ivf_searcher.h"
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

constexpr uint32_t kDimension = 17;
constexpr uint32_t kCount = 37;
constexpr uint32_t kTopK = 7;
using TestCase = std::tuple<QuantizerType, MetricType>;

const char *MetricName(MetricType metric) {
  switch (metric) {
    case MetricType::kInnerProduct:
      return "InnerProduct";
    case MetricType::kCosine:
      return "Cosine";
    default:
      return "SquaredEuclidean";
  }
}

class IVFTurboIndexTest : public testing::TestWithParam<TestCase> {
 protected:
  void SetUp() override {
    const auto *info = testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::string("ivf_turbo_interface_") + info->name() + ".index";
    std::replace(path_.begin(), path_.end(), '/', '_');
    test_util::RemoveTestFiles(path_);
    test_util::RemoveTestFiles(path_ + ".merged");
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, kDimension);
    meta.set_metric(MetricName(std::get<1>(GetParam())), 0, ailego::Params());
    quantizer_ = core::IndexFactory::CreateQuantizer(
        std::get<0>(GetParam()) == QuantizerType::kNone ? "Fp32Quantizer"
                                                        : "Int8Quantizer");
    ASSERT_NE(nullptr, quantizer_);
    ASSERT_EQ(0, quantizer_->init(meta, ailego::Params()));
    std::mt19937 random(8121);
    std::uniform_real_distribution<float> value(-3.0f, 2.0f);
    for (uint32_t i = 0; i < kCount; ++i) {
      vectors_.emplace_back(kDimension);
      for (auto &v : vectors_.back()) {
        v = value(random);
      }
    }
  }

  void TearDown() override {
    test_util::RemoveTestFiles(path_);
    test_util::RemoveTestFiles(path_ + ".merged");
  }

  IVFIndexParam::Pointer param(QuantizerType quantizer) const {
    return IVFIndexParamBuilder()
        .with_metric_type(std::get<1>(GetParam()))
        .with_data_type(DataType::DT_FP32)
        .with_dimension(kDimension)
        .with_n_list(4)
        .with_n_iters(4)
        .with_quantizer_param(QuantizerParam(quantizer))
        .build();
  }

  void populate(Index *index) const {
    for (uint32_t i = 0; i < kCount; ++i) {
      ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors_[i].data()}}, i));
    }
    ASSERT_EQ(0, index->train());
    EXPECT_EQ(kCount, index->get_doc_count());
  }

  std::string decoded_vector(uint32_t id) const {
    core::IndexQueryMeta encoded_meta;
    std::string encoded;
    std::string decoded;
    EXPECT_EQ(0, quantizer_->quantize(
                     vectors_[id].data(),
                     core::IndexQueryMeta(core::IndexMeta::DT_FP32, kDimension),
                     &encoded, &encoded_meta));
    EXPECT_EQ(0,
              quantizer_->dequantize(encoded.data(), encoded_meta, &decoded));
    return decoded;
  }

  std::vector<float> expected_scores(
      const std::vector<std::vector<float>> &vectors,
      turbo::Quantizer::Pointer quantizer, const float *query) const {
    std::vector<float> distances;
    std::string code(quantizer->quantized_datapoint_vector_length(), '\0');
    for (const auto &row : vectors) {
      quantizer->quantize_data(row.data(), code.data());
      distances.push_back(
          quantizer->calc_distance_dp_query_unquantized(code.data(), query));
    }
    std::sort(distances.begin(), distances.end());
    if (std::get<1>(GetParam()) == MetricType::kInnerProduct) {
      for (auto &distance : distances) {
        distance = -distance;
      }
    }
    return distances;
  }

  std::vector<std::pair<uint64_t, float>> check_search_and_fetch(
      Index *index) const {
    const auto *query = vectors_[7].data();
    const auto expected = expected_scores(vectors_, quantizer_, query);
    auto params = IVFQueryParamBuilder()
                      .with_topk(kTopK)
                      .with_nprobe(4)
                      .with_fetch_vector(true)
                      .build();
    SearchResult result;
    EXPECT_EQ(0,
              index->search(VectorData{DenseVector{query}}, params, &result));
    EXPECT_EQ(kTopK, result.doc_list_.size());
    std::vector<std::pair<uint64_t, float>> rows;
    for (size_t i = 0; i < result.doc_list_.size(); ++i) {
      const auto &doc = result.doc_list_[i];
      EXPECT_LT(doc.key(), kCount);
      if (doc.key() >= kCount) {
        continue;
      }
      rows.emplace_back(doc.key(), doc.score());
      EXPECT_NEAR(expected[i], doc.score(),
                  1e-4f * std::max(1.0f, std::abs(expected[i])));
      const auto decoded = decoded_vector(static_cast<uint32_t>(doc.key()));
      const void *vector = result.reverted_vector_list_.empty()
                               ? doc.vector()
                               : result.reverted_vector_list_[i].data();
      EXPECT_NE(nullptr, vector);
      if (vector) {
        EXPECT_EQ(0, std::memcmp(decoded.data(), vector, decoded.size()));
      }
      VectorDataBuffer fetched;
      EXPECT_EQ(0, index->fetch(static_cast<uint32_t>(doc.key()), &fetched));
      EXPECT_EQ(decoded,
                std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
    }
    return rows;
  }

  std::string path_;
  turbo::Quantizer::Pointer quantizer_;
  std::vector<std::vector<float>> vectors_;
};

TEST_P(IVFTurboIndexTest, AddTrainSearchFetchAndReopenWithDefaultQuantizer) {
  auto index =
      IndexFactory::CreateAndInitIndex(*param(std::get<0>(GetParam())));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path_, {StorageOptions::StorageType::kMMAP, true}));
  populate(index.get());
  auto before = check_search_and_fetch(index.get());
  ASSERT_EQ(0, index->close());
  index.reset();

  // The on-disk quantizer must override the default FP32 configuration.
  auto reopened =
      IndexFactory::CreateAndInitIndex(*param(QuantizerType::kNone));
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0,
            reopened->open(path_, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_EQ(before, check_search_and_fetch(reopened.get()));
  ASSERT_EQ(0, reopened->close());
  ASSERT_EQ(0,
            reopened->open(path_, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_EQ(before, check_search_and_fetch(reopened.get()));
  ASSERT_EQ(0, reopened->close());
}

TEST_P(IVFTurboIndexTest, MergeDecodesSourceVectorsBeforeRebuilding) {
  auto source =
      IndexFactory::CreateAndInitIndex(*param(std::get<0>(GetParam())));
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(0, source->open(path_, {StorageOptions::StorageType::kMMAP, true}));
  populate(source.get());
  auto target = IndexFactory::CreateAndInitIndex(*param(QuantizerType::kNone));
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(0, target->open(path_ + ".merged",
                            {StorageOptions::StorageType::kMMAP, true}));
  IndexFilter filter;
  filter.set([](uint64_t key) { return key == 7; });
  ASSERT_EQ(0, target->merge({source}, filter));
  EXPECT_EQ(kCount - 1, target->get_doc_count());

  std::vector<std::vector<float>> merged_vectors;
  for (uint32_t i = 0; i < kCount; ++i) {
    if (i != 7) {
      auto decoded = decoded_vector(i);
      merged_vectors.emplace_back(kDimension);
      std::memcpy(merged_vectors.back().data(), decoded.data(), decoded.size());
    }
  }
  auto fp32 = core::IndexFactory::CreateQuantizer("Fp32Quantizer");
  ASSERT_NE(nullptr, fp32);
  core::IndexMeta meta;
  meta.set_meta(core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric(MetricName(std::get<1>(GetParam())), 0, ailego::Params());
  ASSERT_EQ(0, fp32->init(meta, ailego::Params()));
  auto expected = expected_scores(merged_vectors, fp32, vectors_[7].data());
  SearchResult result;
  ASSERT_EQ(0,
            target->search(
                VectorData{DenseVector{vectors_[7].data()}},
                IVFQueryParamBuilder().with_nprobe(4).with_topk(kTopK).build(),
                &result));
  ASSERT_EQ(kTopK, result.doc_list_.size());
  for (size_t i = 0; i < kTopK; ++i) {
    EXPECT_NEAR(expected[i], result.doc_list_[i].score(),
                1e-4f * std::max(1.0f, std::abs(expected[i])));
  }
  ASSERT_EQ(0, source->close());
  ASSERT_EQ(0, target->close());
}

INSTANTIATE_TEST_SUITE_P(
    Fp32AndInt8, IVFTurboIndexTest,
    testing::Combine(
        testing::Values(QuantizerType::kNone, QuantizerType::kInt8),
        testing::Values(MetricType::kL2sq, MetricType::kInnerProduct,
                        MetricType::kCosine)));

TEST(IVFTurboCompatibility, ReopensLegacyFp32File) {
  const std::string path = "ivf_turbo_legacy_fp32.index";
  test_util::RemoveTestFiles(path);
  core::IndexMeta meta;
  meta.set_meta(core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(core::PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  auto holder =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(
          kDimension);
  ailego::NumericalVector<float> vector(kDimension);
  for (uint32_t j = 0; j < kDimension; ++j) {
    vector[j] = 0.125f * j;
  }
  holder->emplace(29, vector);
  core::IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params));
  ASSERT_EQ(0, builder.train(core::IndexThreads::Pointer(), holder));
  ASSERT_EQ(0, builder.build(core::IndexThreads::Pointer(), holder));
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder.dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto param = IVFIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_n_list(1)
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
  SearchResult result;
  ASSERT_EQ(0, index->search(
                   VectorData{DenseVector{vector.data()}},
                   IVFQueryParamBuilder().with_topk(1).with_nprobe(1).build(),
                   &result));
  ASSERT_EQ(1U, result.doc_list_.size());
  EXPECT_EQ(29U, result.doc_list_[0].key());
  EXPECT_FLOAT_EQ(0.0f, result.doc_list_[0].score());
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(29, &fetched));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(vector.data()),
                        kDimension * sizeof(float)),
            std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  ASSERT_EQ(0, index->close());
  test_util::RemoveTestFiles(path);
}

// Force the old public pipeline to produce a genuine converter-based artifact.
class LegacyIVFIndexForTest : public IVFIndex {
 public:
  int init_legacy(const BaseIndexParam &param) {
    return init(param);
  }

 protected:
  int create_and_init_converter_reformer(const QuantizerParam &quantizer,
                                         const BaseIndexParam &param) override {
    return Index::create_and_init_converter_reformer(quantizer, param);
  }
};

TEST(IVFTurboCompatibility, ReopensLegacyInt8AndMergesWithTurbo) {
  constexpr uint32_t dim = 8;
  constexpr uint32_t count = 20;
  const std::string legacy_path = "ivf_turbo_legacy_int8.index";
  const std::string turbo_path = "ivf_turbo_mixed_source.index";
  const std::string target_path = "ivf_turbo_mixed_target.index";
  struct Cleanup {
    std::vector<std::string> paths;
    ~Cleanup() {
      for (const auto &path : paths) test_util::RemoveTestFiles(path);
    }
  } cleanup{{legacy_path, turbo_path, target_path}};
  for (const auto &path : cleanup.paths) test_util::RemoveTestFiles(path);
  auto param = IVFIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(dim)
                   .with_n_list(2)
                   .with_n_iters(4)
                   .with_quantizer_param(QuantizerParam(QuantizerType::kInt8))
                   .build();
  std::vector<std::vector<float>> vectors(count * 2, std::vector<float>(dim));
  for (uint32_t i = 0; i < count * 2; ++i) {
    for (uint32_t j = 0; j < dim; ++j) {
      vectors[i][j] = 0.13f * static_cast<float>((i * 17 + j * 7) % 31) - 1.0f;
    }
  }

  auto legacy_writer = std::make_shared<LegacyIVFIndexForTest>();
  ASSERT_EQ(0, legacy_writer->init_legacy(*param));
  Index::Pointer old_index = legacy_writer;
  ASSERT_EQ(0, old_index->open(legacy_path,
                               {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t i = 0; i < count; ++i) {
    ASSERT_EQ(0, old_index->add(VectorData{DenseVector{vectors[i].data()}}, i));
  }
  ASSERT_EQ(0, old_index->train());
  VectorDataBuffer original_fetch;
  ASSERT_EQ(0, old_index->fetch(3, &original_fetch));
  ASSERT_EQ(0, old_index->close());
  old_index.reset();
  legacy_writer.reset();

  auto legacy = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, legacy);
  ASSERT_EQ(0, legacy->open(legacy_path,
                            {StorageOptions::StorageType::kMMAP, false}));
  VectorDataBuffer reopened_fetch;
  ASSERT_EQ(0, legacy->fetch(3, &reopened_fetch));
  EXPECT_EQ(std::get<DenseVectorBuffer>(original_fetch.vector_buffer).data,
            std::get<DenseVectorBuffer>(reopened_fetch.vector_buffer).data);
  SearchResult legacy_result;
  ASSERT_EQ(0, legacy->search(
                   VectorData{DenseVector{vectors[3].data()}},
                   IVFQueryParamBuilder().with_nprobe(2).with_topk(1).build(),
                   &legacy_result));
  ASSERT_EQ(1U, legacy_result.doc_list_.size());
  EXPECT_EQ(3U, legacy_result.doc_list_[0].key());

  auto turbo = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, turbo);
  ASSERT_EQ(
      0, turbo->open(turbo_path, {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t i = count; i < count * 2; ++i) {
    ASSERT_EQ(0, turbo->add(VectorData{DenseVector{vectors[i].data()}}, i));
  }
  ASSERT_EQ(0, turbo->train());

  const uint32_t removed_key = 5;
  std::vector<float> expected_distances;
  for (uint32_t i = 0; i < count * 2; ++i) {
    if (i == removed_key) continue;
    VectorDataBuffer fetched;
    ASSERT_EQ(0, (i < count ? legacy : turbo)->fetch(i, &fetched));
    const auto &bytes = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    ASSERT_EQ(dim * sizeof(float), bytes.size());
    std::vector<float> decoded(dim);
    std::memcpy(decoded.data(), bytes.data(), bytes.size());
    float distance = 0.0f;
    for (uint32_t j = 0; j < dim; ++j) {
      const float delta = decoded[j] - vectors[3][j];
      distance += delta * delta;
    }
    expected_distances.push_back(distance);
  }
  std::sort(expected_distances.begin(), expected_distances.end());

  auto target_param = IVFIndexParamBuilder()
                          .with_metric_type(MetricType::kL2sq)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(dim)
                          .with_n_list(2)
                          .with_n_iters(4)
                          .build();
  auto target = IndexFactory::CreateAndInitIndex(*target_param);
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  IndexFilter filter;
  filter.set(
      [removed_key = removed_key](uint64_t key) { return key == removed_key; });
  ASSERT_EQ(0, target->merge({legacy, turbo}, filter));
  EXPECT_EQ(count * 2 - 1, target->get_doc_count());
  SearchResult merged_result;
  ASSERT_EQ(
      0, target->search(
             VectorData{DenseVector{vectors[3].data()}},
             IVFQueryParamBuilder().with_nprobe(2).with_topk(count * 2).build(),
             &merged_result));
  ASSERT_EQ(expected_distances.size(), merged_result.doc_list_.size());
  for (size_t i = 0; i < expected_distances.size(); ++i) {
    EXPECT_NEAR(expected_distances[i], merged_result.doc_list_[i].score(),
                1e-4f * std::max(1.0f, expected_distances[i]));
  }
  ASSERT_EQ(0, target->close());
  ASSERT_EQ(0, turbo->close());
  ASSERT_EQ(0, legacy->close());
}

constexpr uint32_t kPqDimension = 12;
constexpr uint32_t kPqCount = 321;
constexpr uint32_t kPqTopK = 7;
using PqCase = std::tuple<int, bool, MetricType, bool>;

class IVFPqTest : public testing::TestWithParam<PqCase> {
 protected:
  void SetUp() override {
    path_ = std::string("ivf_pq_") +
            testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace(path_.begin(), path_.end(), '/', '_');
    std::mt19937 random(963);
    std::normal_distribution<float> value(0.0f, 2.0f);
    for (uint32_t i = 0; i < kPqCount; ++i) {
      vectors_.emplace_back(kPqDimension);
      for (auto &v : vectors_.back()) v = value(random);
    }
  }
  void TearDown() override {
    for (const auto &suffix : {"", ".fine", ".merge"})
      test_util::RemoveTestFiles(path_ + suffix);
  }
  IVFIndexParam::Pointer param(bool pq = true) const {
    auto builder = IVFIndexParamBuilder()
                       .with_dimension(kPqDimension)
                       .with_data_type(DataType::DT_FP32)
                       .with_metric_type(std::get<2>(GetParam()))
                       .with_n_list(3)
                       .with_n_iters(3);
    if (pq) {
      int mode = std::get<0>(GetParam());
      PqQuantizerParam quantizer(3, mode == 8 ? 8 : 4, std::get<1>(GetParam()));
      quantizer.fast_scan = mode == 0;
      quantizer.opq_iter = 2;
      quantizer.opq_pq_iter = 2;
      builder.with_quantizer_param(quantizer);
    }
    return builder.build();
  }
  StorageOptions::StorageType storage() const {
    return std::get<3>(GetParam()) ? StorageOptions::StorageType::kBufferPool
                                   : StorageOptions::StorageType::kMMAP;
  }
  void build(const Index::Pointer &index) const {
    for (uint32_t i = 0; i < kPqCount; ++i) {
      ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors_[i].data()}}, i));
    }
    ASSERT_EQ(0, index->train());
  }
  void same(const SearchResult &expected, const SearchResult &actual) const {
    ASSERT_EQ(expected.doc_list_.size(), actual.doc_list_.size());
    std::map<uint64_t, float> expected_by_key, actual_by_key;
    for (size_t i = 0; i < expected.doc_list_.size(); ++i) {
      // Quantized ties may be emitted in either order, but ranks and the
      // complete set of (ID, score) pairs must agree.
      EXPECT_FLOAT_EQ(expected.doc_list_[i].score(),
                      actual.doc_list_[i].score());
      expected_by_key.emplace(expected.doc_list_[i].key(),
                              expected.doc_list_[i].score());
      actual_by_key.emplace(actual.doc_list_[i].key(),
                            actual.doc_list_[i].score());
    }
    ASSERT_EQ(expected_by_key.size(), actual_by_key.size());
    for (const auto &entry : expected_by_key) {
      auto found = actual_by_key.find(entry.first);
      ASSERT_NE(actual_by_key.end(), found);
      EXPECT_FLOAT_EQ(entry.second, found->second);
    }
  }
  std::string path_;
  std::vector<std::vector<float>> vectors_;
};

TEST_P(IVFPqTest, ScoresPersistenceCandidatesAndRefinement) {
  auto config = param();
  IVFIndexParam roundtrip;
  ASSERT_TRUE(roundtrip.deserialize_from_json(config->serialize_to_json()));
  const auto *pq =
      dynamic_cast<PqQuantizerParam *>(roundtrip.quantizer_param.get());
  ASSERT_NE(nullptr, pq);
  EXPECT_EQ(std::get<0>(GetParam()) == 0, pq->fast_scan);
  EXPECT_EQ(std::get<1>(GetParam()), pq->enable_rotate);
  EXPECT_EQ(2u, pq->opq_iter);
  EXPECT_EQ(2u, pq->opq_pq_iter);
  auto index = IndexFactory::CreateAndInitIndex(roundtrip);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path_, {storage(), true}));
  build(index);
  const VectorData query{DenseVector{vectors_[7].data()}};
  auto query_param =
      IVFQueryParamBuilder().with_nprobe(3).with_topk(kPqCount).build();
  SearchResult all;
  ASSERT_EQ(0, index->search(query, query_param, &all));
  ASSERT_EQ(kPqCount, all.doc_list_.size());

  // Independently encode and score each row with the persisted Turbo state.
  auto disk = core::IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, disk->init(ailego::Params{}));
  ASSERT_EQ(0, disk->open(path_, false));
  core::IVFSearcher searcher;
  ASSERT_EQ(0, searcher.init(ailego::Params{}));
  ASSERT_EQ(0, searcher.load(disk, nullptr));
  auto quantizer = searcher.quantizer();
  ASSERT_NE(nullptr, quantizer);
  std::string encoded_query(quantizer->quantized_query_vector_length(), '\0');
  quantizer->quantize_query(vectors_[7].data(), encoded_query.data());
  const size_t code_size = quantizer->quantized_datapoint_vector_length();
  std::string code(code_size, '\0');
  std::string packed(code_size * 32, '\0');
  auto *packer = dynamic_cast<turbo::PackedCodeQuantizer *>(quantizer.get());
  for (const auto &doc : all.doc_list_) {
    ASSERT_LT(doc.key(), kPqCount);
    quantizer->quantize_data(vectors_[doc.key()].data(), code.data());
    float expected;
    if (packer) {
      ASSERT_EQ(0,
                packer->pack_codes(code.data(), 1, code_size, packed.data()));
      packer->calc_distance_packed_block(packed.data(), 1, encoded_query.data(),
                                         &expected);
    } else {
      expected =
          quantizer->calc_distance_dp_query(code.data(), encoded_query.data());
    }
    if (std::get<2>(GetParam()) == MetricType::kInnerProduct)
      expected = -expected;
    EXPECT_NEAR(expected, doc.score(),
                1e-4f * std::max(1.0f, std::abs(expected)));
  }
  ASSERT_EQ(0, searcher.unload());
  ASSERT_EQ(0, disk->close());

  auto filtered_param = IVFQueryParamBuilder()
                            .with_nprobe(3)
                            .with_topk(kPqTopK)
                            .with_fetch_vector(true)
                            .build();
  auto set_filter = [](const IVFQueryParam::Pointer &p) {
    p->filter = std::make_shared<IndexFilter>();
    p->filter->set([](uint64_t id) { return id % 3 == 0; });
  };
  set_filter(filtered_param);
  SearchResult filtered;
  ASSERT_EQ(0, index->search(query, filtered_param, &filtered));
  ASSERT_EQ(kPqTopK, filtered.doc_list_.size());
  for (size_t i = 0; i < filtered.doc_list_.size(); ++i) {
    auto id = filtered.doc_list_[i].key();
    EXPECT_NE(0u, id % 3);
    VectorDataBuffer fetched;
    ASSERT_EQ(0, index->fetch(id, &fetched));
    auto decoded = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    EXPECT_EQ(decoded, filtered.reverted_vector_list_[i]);
    if (packer)
      EXPECT_EQ(0, std::memcmp(decoded.data(), vectors_[id].data(),
                               kPqDimension * sizeof(float)));
  }
  auto excluded =
      IVFQueryParamBuilder().with_nprobe(3).with_topk(kPqTopK).build();
  excluded->filter = std::make_shared<IndexFilter>();
  excluded->filter->set([](uint64_t) { return true; });
  SearchResult empty;
  ASSERT_EQ(0, index->search(query, excluded, &empty));
  EXPECT_TRUE(empty.doc_list_.empty());
  excluded->filter.reset();
  excluded->bf_pks = std::make_shared<std::vector<uint64_t>>();
  ASSERT_EQ(0, index->search(query, excluded, &empty));
  EXPECT_TRUE(empty.doc_list_.empty());

  // Candidate-ID search must retain this index's own PQ scores.
  auto selected = IVFQueryParamBuilder().with_topk(kPqTopK).build();
  selected->bf_pks = std::make_shared<std::vector<uint64_t>>();
  for (const auto &doc : filtered.doc_list_)
    selected->bf_pks->push_back(doc.key());
  selected->bf_pks->push_back(kPqCount + 100);
  SearchResult selected_result;
  ASSERT_EQ(0, index->search(query, selected, &selected_result));
  same(filtered, selected_result);

  auto fine = IndexFactory::CreateAndInitIndex(*param(false));
  ASSERT_NE(nullptr, fine);
  ASSERT_EQ(0, fine->open(path_ + ".fine", {storage(), true}));
  build(fine);
  auto refiner = std::make_shared<RefinerParam>();
  refiner->reference_index = fine;
  for (float scale : {0.0f, 0.5f, 1.0f, 2.5f, 4.0f}) {
    SCOPED_TRACE(scale);
    auto candidates_param =
        IVFQueryParamBuilder()
            .with_topk(uint32_t(kPqTopK * std::max(1.0f, scale)))
            .with_nprobe(3)
            .build();
    set_filter(candidates_param);
    SearchResult candidates;
    ASSERT_EQ(0, index->search(query, candidates_param, &candidates));
    auto exact_param = IVFQueryParamBuilder()
                           .with_topk(kPqTopK)
                           .with_fetch_vector(true)
                           .build();
    exact_param->bf_pks = std::make_shared<std::vector<uint64_t>>();
    for (const auto &doc : candidates.doc_list_)
      exact_param->bf_pks->push_back(doc.key());
    SearchResult expected, actual;
    ASSERT_EQ(0, fine->search(query, exact_param, &expected));
    auto refined_param = IVFQueryParamBuilder()
                             .with_topk(kPqTopK)
                             .with_nprobe(3)
                             .with_fetch_vector(true)
                             .with_refiner_param(refiner)
                             .build();
    set_filter(refined_param);
    refiner->scale_factor_ = scale;
    ASSERT_EQ(0, index->search(query, refined_param, &actual));
    same(expected, actual);
    EXPECT_EQ(expected.reverted_vector_list_, actual.reverted_vector_list_);
  }
  for (float invalid : {-1.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::max()}) {
    refiner->scale_factor_ = invalid;
    auto p = IVFQueryParamBuilder()
                 .with_topk(kPqTopK)
                 .with_refiner_param(refiner)
                 .build();
    SearchResult result;
    EXPECT_NE(0, index->search(query, p, &result));
  }
  ASSERT_EQ(0, fine->close());
  ASSERT_EQ(0, index->close());
  auto reopened = IndexFactory::CreateAndInitIndex(*param(false));
  ASSERT_EQ(0, reopened->open(path_, {storage(), false}));
  SearchResult after;
  ASSERT_EQ(0, reopened->search(query, query_param, &after));
  same(all, after);
  // Merge must consume decoded/original input, never packed posting bytes.
  std::vector<std::string> decoded(kPqCount);
  for (uint32_t id = 0; id < kPqCount; ++id) {
    VectorDataBuffer fetched;
    ASSERT_EQ(0, reopened->fetch(id, &fetched));
    decoded[id] = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
  }
  auto merged = IndexFactory::CreateAndInitIndex(*param(false));
  ASSERT_EQ(0, merged->open(path_ + ".merge", {storage(), true}));
  IndexFilter merge_filter;
  merge_filter.set([](uint64_t id) { return id == 7; });
  ASSERT_EQ(0, merged->merge({reopened}, merge_filter));
  EXPECT_EQ(kPqCount - 1, merged->get_doc_count());
  for (uint32_t id = 0; id < kPqCount; ++id) {
    if (id == 7) continue;
    VectorDataBuffer fetched;
    // Merge's existing contract compacts surviving IDs in source order.
    ASSERT_EQ(0, merged->fetch(id < 7 ? id : id - 1, &fetched));
    const auto &actual =
        std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    ASSERT_EQ(decoded[id].size(), actual.size());
    for (uint32_t d = 0; d < kPqDimension; ++d) {
      float expected_value, actual_value;
      std::memcpy(&expected_value, decoded[id].data() + d * sizeof(float),
                  sizeof(float));
      std::memcpy(&actual_value, actual.data() + d * sizeof(float),
                  sizeof(float));
      // Cosine FP32 storage normalizes and reconstructs its components.
      EXPECT_NEAR(expected_value, actual_value,
                  1e-5f * std::max(1.0f, std::abs(expected_value)));
    }
  }
  ASSERT_EQ(0, merged->close());
  ASSERT_EQ(0, reopened->close());
}

INSTANTIATE_TEST_SUITE_P(
    Turbo, IVFPqTest,
    testing::Combine(testing::Values(8, 4, 0), testing::Bool(),
                     testing::Values(MetricType::kL2sq,
                                     MetricType::kInnerProduct,
                                     MetricType::kCosine),
                     testing::Bool()));

TEST(IVFPqStorage, PackedBlocksAcrossPages) {
  const std::string path = "ivf_pq_cross_page.index";
  struct Cleanup {
    std::string path;
    ~Cleanup() {
      test_util::RemoveTestFiles(path);
      test_util::RemoveTestFiles(path + ".merged");
    }
  } cleanup{path};
  // Five chunks produce 96-byte blocks. The posting body exceeds a 16 KiB
  // page and the block stride does not divide either 4 KiB or 16 KiB pages.
  PqQuantizerParam pq(5, 4);
  pq.fast_scan = true;
  auto config = IVFIndexParamBuilder()
                    .with_dimension(30)
                    .with_data_type(DataType::DT_FP32)
                    .with_metric_type(MetricType::kL2sq)
                    .with_n_list(7)
                    .with_n_iters(3)
                    .with_quantizer_param(pq)
                    .build();
  auto mmap = IndexFactory::CreateAndInitIndex(*config);
  ASSERT_NE(nullptr, mmap);
  ASSERT_EQ(0, mmap->open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::mt19937 random(905);
  std::normal_distribution<float> value(0.0f, 1.0f);
  std::vector<float> vector(30);
  for (uint32_t id = 0; id < 8193; ++id) {
    for (auto &v : vector) v = value(random);
    ASSERT_EQ(0, mmap->add(VectorData{DenseVector{vector.data()}}, id));
  }
  ASSERT_EQ(0, mmap->train());
  auto pooled = IndexFactory::CreateAndInitIndex(*config);
  ASSERT_EQ(
      0, pooled->open(path, {StorageOptions::StorageType::kBufferPool, false}));
  auto run = [&](const Index::Pointer &index, uint32_t nprobe, bool filtered,
                 float radius = 0.0f) {
    auto param = IVFQueryParamBuilder()
                     .with_nprobe(nprobe)
                     .with_topk(8193)
                     .with_radius(radius)
                     .build();
    if (filtered) {
      param->filter = std::make_shared<IndexFilter>();
      param->filter->set([](uint64_t id) { return id % 7 == 0; });
    }
    SearchResult result;
    EXPECT_EQ(0, index->search(VectorData{DenseVector{vector.data()}}, param,
                               &result));
    std::map<uint64_t, float> scores;
    for (const auto &doc : result.doc_list_)
      scores.emplace(doc.key(), doc.score());
    return scores;
  };
  for (uint32_t nprobe : {1u, 7u, 0u}) {
    SCOPED_TRACE(nprobe);
    for (bool filtered : {false, true}) {
      SCOPED_TRACE(filtered);
      auto expected = run(mmap, nprobe, filtered);
      ASSERT_FALSE(expected.empty());
      if (nprobe == 7 && !filtered) EXPECT_EQ(8193u, expected.size());
      if (nprobe == 1 || nprobe == 0) EXPECT_LT(expected.size(), 8193u);
      for (int pass = 0; pass < 3; ++pass) {
        // Cold contiguous reads and warm resident page spans must agree.
        EXPECT_EQ(expected, run(pooled, nprobe, filtered));
      }
    }
  }
  auto all_scores = run(mmap, 7, false);
  run(pooled, 7, false, 1.0f);
  EXPECT_EQ(all_scores, run(pooled, 7, false));
  auto merged = IndexFactory::CreateAndInitIndex(*config);
  ASSERT_EQ(0, merged->open(path + ".merged",
                            {StorageOptions::StorageType::kBufferPool, true}));
  ASSERT_EQ(0, merged->merge({pooled}, IndexFilter{}));
  EXPECT_EQ(8193u, merged->get_doc_count());
  VectorDataBuffer fetched;
  ASSERT_EQ(0, merged->fetch(8192, &fetched));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(vector.data()),
                        vector.size() * sizeof(float)),
            std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  EXPECT_EQ(8193u, run(merged, 7, false).size());
  ASSERT_EQ(0, merged->close());
  ASSERT_EQ(0, pooled->close());
  ASSERT_EQ(0, mmap->close());
}

TEST(IVFPqValidation, InvalidParameters) {
  for (const auto &pq : {PqQuantizerParam(0, 4), PqQuantizerParam(3, 6),
                         PqQuantizerParam(5, 4), PqQuantizerParam(13, 8)}) {
    auto param = IVFIndexParamBuilder()
                     .with_dimension(kPqDimension)
                     .with_data_type(DataType::DT_FP32)
                     .with_metric_type(MetricType::kL2sq)
                     .with_quantizer_param(pq)
                     .build();
    EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*param));
  }
  PqQuantizerParam pq(3, 8);
  pq.fast_scan = true;
  auto param = IVFIndexParamBuilder()
                   .with_dimension(kPqDimension)
                   .with_data_type(DataType::DT_FP32)
                   .with_metric_type(MetricType::kL2sq)
                   .with_quantizer_param(pq)
                   .build();
  EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*param));
}

}  // namespace
}  // namespace zvec::core_interface
