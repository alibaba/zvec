// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <cmath>
#include <functional>
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/ivf/ivf_index_format.h"
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

IVFIndexParam::Pointer RabitqParam(MetricType metric, int bits = 7) {
  return IVFIndexParamBuilder()
      .with_dimension(128)
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_n_list(4)
      .with_n_iters(3)
      .with_total_bits(bits)
      .with_sample_count(80)
      .with_quantizer_param(QuantizerParam(QuantizerType::kRabitq))
      .build();
}

TEST(IVFRabitqIntegration, ParamsRoundTrip) {
  auto param = RabitqParam(MetricType::kCosine, 9);
  auto decoded = std::dynamic_pointer_cast<IVFIndexParam>(
      IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json()));
  ASSERT_NE(nullptr, decoded);
  EXPECT_EQ(IndexType::kIVF, decoded->index_type);
  EXPECT_EQ(QuantizerType::kRabitq, decoded->quantizer_param->type);
  EXPECT_EQ(4, decoded->nlist);
  EXPECT_EQ(3, decoded->niters);
  EXPECT_EQ(9, decoded->total_bits);
  EXPECT_EQ(80, decoded->sample_count);
  auto typed =
      std::dynamic_pointer_cast<RabitqQuantizerParam>(decoded->quantizer_param);
  ASSERT_NE(nullptr, typed);
  EXPECT_EQ(9, typed->total_bits);
  EXPECT_EQ(4, typed->num_clusters);
  EXPECT_EQ(80, typed->sample_count);
  EXPECT_EQ(3, typed->niters);
}

TEST(IVFRabitqIntegration, TypedParametersOverrideLegacyFields) {
  RabitqQuantizerParam quantizer(5);
  quantizer.num_clusters = 3;
  quantizer.sample_count = 24;
  quantizer.niters = 6;
  auto param = RabitqParam(MetricType::kL2sq, 9);
  param->quantizer_param = quantizer.clone();
  auto decoded = std::dynamic_pointer_cast<IVFIndexParam>(
      IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json()));
  ASSERT_NE(nullptr, decoded);
  auto typed =
      std::dynamic_pointer_cast<RabitqQuantizerParam>(decoded->quantizer_param);
  ASSERT_NE(nullptr, typed);
  EXPECT_EQ(quantizer.serialize_to_json(), typed->serialize_to_json());
}

TEST(IVFRabitqIntegration, RejectsInvalidNestedAndLegacyJson) {
  auto param = RabitqParam(MetricType::kL2sq);
  param->total_bits = 10;
  EXPECT_EQ(nullptr, IndexFactory::DeserializeIndexParamFromJson(
                         param->serialize_to_json()));
  RabitqQuantizerParam invalid(7);
  invalid.niters = 0;
  param->quantizer_param = invalid.clone();
  EXPECT_EQ(nullptr, IndexFactory::DeserializeIndexParamFromJson(
                         param->serialize_to_json()));
}

TEST(IVFRabitqIntegration, ScanQueryOptionsRoundTrip) {
  IVFQueryParam param;
  param.rabitq_pruning = false;
  param.rabitq_reuse_coarse_dot = true;
  const auto json = IndexFactory::QueryParamSerializeToJson(param, true);
  auto typed = IndexFactory::QueryParamDeserializeFromJson<IVFQueryParam>(json);
  auto base = std::dynamic_pointer_cast<IVFQueryParam>(
      IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(json));
  auto copy = std::dynamic_pointer_cast<IVFQueryParam>(param.clone());
  for (const auto &p : {typed, base, copy}) {
    ASSERT_NE(nullptr, p);
    EXPECT_FALSE(p->rabitq_pruning);
    EXPECT_TRUE(p->rabitq_reuse_coarse_dot);
  }
  auto defaults = IndexFactory::QueryParamDeserializeFromJson<IVFQueryParam>(
      R"({"index_type":"kIVF","nprobe":16})");
  ASSERT_NE(nullptr, defaults);
  EXPECT_TRUE(defaults->rabitq_pruning);
  EXPECT_FALSE(defaults->rabitq_reuse_coarse_dot);
}

TEST(IVFRabitqIntegration, DedicatedQueryOptionsRoundTrip) {
  IVFRabitqQueryParam param;
  param.nprobe = 16;
  const auto json = IndexFactory::QueryParamSerializeToJson(param, true);
  EXPECT_EQ(std::string::npos, json.find("rabitq_pruning"));
  auto typed =
      IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(json);
  auto base = std::dynamic_pointer_cast<IVFRabitqQueryParam>(
      IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(json));
  auto copy = std::dynamic_pointer_cast<IVFRabitqQueryParam>(param.clone());
  for (const auto &p : {typed, base, copy}) {
    ASSERT_NE(nullptr, p);
    EXPECT_EQ(16u, p->nprobe);
  }
  auto defaults =
      IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
          R"({"index_type":"kIVFRabitq","nprobe":16})");
  ASSERT_NE(nullptr, defaults);
  EXPECT_EQ(16u, defaults->nprobe);
  const auto invalid = R"({"index_type":"kIVFRabitq","nprobe":"yes"})";
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
                invalid));
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(
                invalid));
}

#if RABITQ_SUPPORTED
class IVFRabitqIntegrationTest
    : public testing::TestWithParam<std::tuple<MetricType, int, bool>> {};

TEST_P(IVFRabitqIntegrationTest, TrainSearchAndReopen) {
  const auto metric = std::get<0>(GetParam());
  const auto bits = std::get<1>(GetParam());
  auto param = RabitqParam(metric, bits);
  if (std::get<2>(GetParam())) {
    RabitqQuantizerParam quantizer(bits);
    quantizer.num_clusters = 3;  // independent of the four IVF lists
    quantizer.sample_count = 80;
    quantizer.niters = 3;
    param->quantizer_param = quantizer.clone();
  }
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, std::dynamic_pointer_cast<IVFIndex>(index));
  const std::string path = "ivf_integrated_rabitq.index";
  test_util::RemoveTestFiles(path);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::mt19937 rng(2026);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> vectors(80, std::vector<float>(128));
  for (size_t i = 0; i < vectors.size(); ++i) {
    float norm = 0;
    for (auto &v : vectors[i]) {
      v = normal(rng);
      norm += v * v;
    }
    for (auto &v : vectors[i]) v /= std::sqrt(norm);
    // Sparse public IDs exercise persisted key mapping.
    ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  }
  ASSERT_EQ(0, index->train());
  ASSERT_EQ(0, index->train());
  EXPECT_EQ(80, index->get_doc_count());
  auto storage = core::IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  ASSERT_EQ(0, storage->open(path, false));
  core::IndexMeta persisted;
  ASSERT_EQ(
      0, core::IndexHelper::DeserializeFromStorage(storage.get(), &persisted));
  EXPECT_EQ("RabitqQuantizer", persisted.quantizer_name());
  EXPECT_NE(nullptr, storage->get(core::IVF_TURBO_QUANTIZER_SEG_ID));
  EXPECT_NE(nullptr, storage->get(core::IVF_TURBO_SCAN_SEG_ID));
  ASSERT_EQ(0, storage->close());
  auto query = IVFQueryParamBuilder().with_topk(5).with_nprobe(4).build();
  SearchResult before;
  auto input = VectorData{DenseVector{vectors[7].data()}};
  ASSERT_EQ(0, index->search(input, query, &before));
  ASSERT_EQ(5, before.doc_list_.size());
  EXPECT_EQ(21, before.doc_list_[0].key());
  if (bits >= 7) {
    EXPECT_NEAR(metric == MetricType::kInnerProduct ? 1.0f : 0.0f,
                before.doc_list_[0].score(), 0.1f);
  }
  // Public controls reach the query-local scanner and do not leak through
  // thread-local contexts when the following query returns to defaults.
  SearchResult toggled;
  for (bool reuse : {false, true}) {
    query->rabitq_reuse_coarse_dot = reuse;
    query->rabitq_pruning = false;
    ASSERT_EQ(0, index->search(input, query, &toggled));
    EXPECT_EQ(before.doc_list_.size(), toggled.doc_list_.size());
  }
  query->rabitq_pruning = true;
  query->rabitq_reuse_coarse_dot = false;
  ASSERT_EQ(0, index->search(input, query, &toggled));
  ASSERT_EQ(before.doc_list_.size(), toggled.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(before.doc_list_[i].key(), toggled.doc_list_[i].key());
    EXPECT_FLOAT_EQ(before.doc_list_[i].score(), toggled.doc_list_[i].score());
  }
  // Invalid probes must not silently reuse the previous thread-local setting.
  for (int nprobe : {-1, 0}) {
    query->nprobe = nprobe;
    SearchResult invalid;
    EXPECT_EQ(core::IndexError_InvalidArgument,
              index->search(input, query, &invalid));
  }
  query->nprobe = 4;
  auto group_query = IVFQueryParamBuilder().with_topk(8).with_nprobe(4).build();
  group_query->group_by_param = std::make_shared<GroupByParam>();
  group_query->group_by_param->group_count = 4;
  group_query->group_by_param->group_topk = 2;
  group_query->group_by_param->group_by = [](uint64_t key) {
    return std::to_string(key % 4);
  };
  SearchResult grouped;
  ASSERT_EQ(0, index->search(input, group_query, &grouped));
  ASSERT_EQ(4, grouped.group_doc_list_.size());
  for (const auto &group : grouped.group_doc_list_) {
    ASSERT_EQ(2, group.docs().size());
    for (const auto &doc : group.docs()) {
      EXPECT_EQ(std::to_string(doc.key() % 4), group.group_id());
    }
  }
  auto plain_param = RabitqParam(metric);
  plain_param->quantizer_param = std::make_shared<QuantizerParam>();
  auto plain = IndexFactory::CreateAndInitIndex(*plain_param);
  ASSERT_NE(nullptr, plain);
  const std::string plain_path = path + ".plain";
  test_util::RemoveTestFiles(plain_path);
  ASSERT_EQ(
      0, plain->open(plain_path, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i) {
    ASSERT_EQ(0, plain->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  }
  ASSERT_EQ(0, plain->train());
  SearchResult rejected_group;
  EXPECT_EQ(core::IndexError_Unsupported,
            plain->search(input, group_query, &rejected_group));
  // Alternate backends in one thread: both use the public IVF context slot.
  for (int attempt = 0; attempt < 2; ++attempt) {
    SearchResult result;
    ASSERT_EQ(0, plain->search(input, query, &result));
    ASSERT_EQ(5, result.doc_list_.size());
    EXPECT_EQ(21, result.doc_list_[0].key());
    ASSERT_EQ(0, index->search(input, group_query, &result));
    ASSERT_EQ(4, result.group_doc_list_.size());
    ASSERT_EQ(0, index->search(input, query, &result));
    ASSERT_EQ(5, result.doc_list_.size());
    EXPECT_EQ(21, result.doc_list_[0].key());
  }
  // Collection optimize builds IVF from raw/decoded Flat or IVF providers.
  const std::string merge_path = path + ".merge";
  test_util::RemoveTestFiles(merge_path);
  auto merged = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, merged);
  ASSERT_EQ(
      0, merged->open(merge_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, merged->merge({plain}, IndexFilter{}));
  SearchResult merged_result;
  ASSERT_EQ(0, merged->search(input, query, &merged_result));
  ASSERT_EQ(5, merged_result.doc_list_.size());
  EXPECT_EQ(7, merged_result.doc_list_[0].key());
  ASSERT_EQ(0, merged->close());
  test_util::RemoveTestFiles(merge_path);
  ASSERT_EQ(0, plain->close());
  test_util::RemoveTestFiles(plain_path);
  query->fetch_vector = true;
  SearchResult unsupported;
  EXPECT_NE(0, index->search(input, query, &unsupported));
  query->fetch_vector = false;
  ASSERT_EQ(0, index->close());
  index.reset();

  // Detect the persisted RaBitQ postings even with default quantization.
  param->quantizer_param = std::make_shared<QuantizerParam>();
  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  for (int attempt = 0; attempt < 2; ++attempt) {
    ASSERT_EQ(
        0, reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
    SearchResult after;
    ASSERT_EQ(0, reopened->search(input, query, &after));
    ASSERT_EQ(before.doc_list_.size(), after.doc_list_.size());
    for (size_t i = 0; i < before.doc_list_.size(); ++i) {
      EXPECT_EQ(before.doc_list_[i].key(), after.doc_list_[i].key());
      EXPECT_FLOAT_EQ(before.doc_list_[i].score(), after.doc_list_[i].score());
    }
    ASSERT_EQ(0, reopened->search(input, group_query, &after));
    ASSERT_EQ(4, after.group_doc_list_.size());
    ASSERT_EQ(0, reopened->close());
  }
  // The legacy format remains readable through the standard IVF API.
  auto legacy_param =
      IVFRabitqIndexParamBuilder()
          .with_dimension(128)
          .with_data_type(DataType::DT_FP32)
          .with_metric_type(metric)
          .with_n_list(4)
          .with_total_bits(bits)
          .with_quantizer_param(QuantizerParam(QuantizerType::kRabitq))
          .build();
  auto legacy = IndexFactory::CreateAndInitIndex(*legacy_param);
  ASSERT_NE(nullptr, legacy);
  const std::string legacy_path = path + ".legacy";
  test_util::RemoveTestFiles(legacy_path);
  ASSERT_EQ(
      0, legacy->open(legacy_path, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i)
    ASSERT_EQ(0,
              legacy->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  ASSERT_EQ(0, legacy->train());
  ASSERT_EQ(0, reopened->open(legacy_path,
                              {StorageOptions::StorageType::kMMAP, false}));
  ASSERT_EQ(0, reopened->search(input, query, &before));
  ASSERT_EQ(0, reopened->search(input, group_query, &grouped));
  auto legacy_query = std::make_shared<IVFRabitqQueryParam>();
  legacy_query->nprobe = 4;
  legacy_query->topk = 5;
  SearchResult legacy_result;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  ASSERT_EQ(before.doc_list_.size(), legacy_result.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(before.doc_list_[i].key(), legacy_result.doc_list_[i].key());
    EXPECT_FLOAT_EQ(before.doc_list_[i].score(),
                    legacy_result.doc_list_[i].score());
  }
  legacy_query->topk = vectors.size();
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  EXPECT_EQ(vectors.size(), legacy_result.doc_list_.size());
  legacy_query->topk = group_query->topk;
  legacy_query->group_by_param = group_query->group_by_param;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  ASSERT_EQ(grouped.group_doc_list_.size(),
            legacy_result.group_doc_list_.size());
  for (size_t g = 0; g < grouped.group_doc_list_.size(); ++g) {
    const auto &expected = grouped.group_doc_list_[g];
    const auto &actual = legacy_result.group_doc_list_[g];
    EXPECT_EQ(expected.group_id(), actual.group_id());
    ASSERT_EQ(expected.docs().size(), actual.docs().size());
    for (size_t i = 0; i < expected.docs().size(); ++i) {
      EXPECT_EQ(expected.docs()[i].key(), actual.docs()[i].key());
      EXPECT_FLOAT_EQ(expected.docs()[i].score(), actual.docs()[i].score());
    }
  }
  ASSERT_EQ(0, legacy->close());
  ASSERT_EQ(0, reopened->close());
  test_util::RemoveTestFiles(legacy_path);
  test_util::RemoveTestFiles(path);
}

// Dot reuse is the one scan option that composes with the kFlat refiner:
// it only changes the arithmetic inside the base search's quantized
// refine, and just candidate keys cross the refine boundary. Disabling
// pruning stays rejected under a refiner, and no scan option may
// leak across the shared thread-local context when the mode flips.
TEST_P(IVFRabitqIntegrationTest, ReuseWithKFlatRefinerAndContextSwitching) {
  const auto metric = std::get<0>(GetParam());
  const auto bits = std::get<1>(GetParam());
  auto index = IndexFactory::CreateAndInitIndex(*RabitqParam(metric, bits));
  ASSERT_NE(nullptr, std::dynamic_pointer_cast<IVFIndex>(index));
  const std::string path = "ivf_integrated_rabitq_refiner.index";
  test_util::RemoveTestFiles(path);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::mt19937 rng(2026);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> vectors(80, std::vector<float>(128));
  for (size_t i = 0; i < vectors.size(); ++i) {
    float norm = 0;
    for (auto &v : vectors[i]) {
      v = normal(rng);
      norm += v * v;
    }
    for (auto &v : vectors[i]) v /= std::sqrt(norm);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  }
  ASSERT_EQ(0, index->train());

  // Reference kFlat index holding the same vectors under the same keys.
  auto flat_param = FlatIndexParamBuilder()
                        .with_dimension(128)
                        .with_metric_type(metric)
                        .with_data_type(DataType::DT_FP32)
                        .build();
  auto flat = IndexFactory::CreateAndInitIndex(*flat_param);
  ASSERT_NE(nullptr, flat);
  const std::string flat_path = path + ".flat";
  test_util::RemoveTestFiles(flat_path);
  ASSERT_EQ(0,
            flat->open(flat_path, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i) {
    ASSERT_EQ(0, flat->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  }
  ASSERT_EQ(0, flat->train());
  const auto input = VectorData{DenseVector{vectors[7].data()}};

  auto refiner = std::make_shared<RefinerParam>();
  refiner->reference_index = flat;
  auto query = IVFQueryParamBuilder()
                   .with_topk(5)
                   .with_nprobe(4)
                   .with_refiner_param(refiner)
                   .build();
  auto exact_query = FlatQueryParamBuilder().with_topk(5).build();

  // Plain mode baseline, captured before any reuse/refiner query touches
  // the shared thread-local context.
  SearchResult plain_before;
  query->refiner_param = nullptr;
  ASSERT_EQ(0, index->search(input, query, &plain_before));
  ASSERT_EQ(5u, plain_before.doc_list_.size());

  // With the coarse set covering every vector, the refined result is the
  // exact FP32 top-5 whether reuse is off or on.
  refiner->scale_factor_ = static_cast<float>(vectors.size());
  query->refiner_param = refiner;
  SearchResult exact, off, on;
  ASSERT_EQ(0, flat->search(input, exact_query, &exact));
  ASSERT_EQ(0, index->search(input, query, &off));
  query->rabitq_reuse_coarse_dot = true;
  ASSERT_EQ(0, index->search(input, query, &on));
  ASSERT_EQ(exact.doc_list_.size(), off.doc_list_.size());
  ASSERT_EQ(exact.doc_list_.size(), on.doc_list_.size());
  for (size_t i = 0; i < exact.doc_list_.size(); ++i) {
    EXPECT_EQ(exact.doc_list_[i].key(), off.doc_list_[i].key());
    EXPECT_EQ(exact.doc_list_[i].key(), on.doc_list_[i].key());
    EXPECT_NEAR(exact.doc_list_[i].score(), off.doc_list_[i].score(), 1e-6f);
    EXPECT_NEAR(exact.doc_list_[i].score(), on.doc_list_[i].score(), 1e-6f);
  }

  // Production-like scale factor: reuse with the refiner is accepted and
  // still returns full-topk refined results.
  refiner->scale_factor_ = 1.1f;
  query->rabitq_reuse_coarse_dot = false;
  ASSERT_EQ(0, index->search(input, query, &off));
  query->rabitq_reuse_coarse_dot = true;
  ASSERT_EQ(0, index->search(input, query, &on));
  EXPECT_EQ(5u, off.doc_list_.size());
  EXPECT_EQ(5u, on.doc_list_.size());
  EXPECT_EQ(off.doc_list_[0].key(), on.doc_list_[0].key());

  // Disabling pruning stays rejected under the refiner, with or without reuse.
  const std::pair<const char *, std::function<void(IVFQueryParam &)>> bad[] = {
      {"no_pruning", [](IVFQueryParam &q) { q.rabitq_pruning = false; }},
      {"no_pruning_with_reuse", [](IVFQueryParam &q) {
         q.rabitq_pruning = false;
         q.rabitq_reuse_coarse_dot = true;
       }}};
  for (const auto &[name, flip] : bad) {
    query->rabitq_reuse_coarse_dot = false;
    query->rabitq_pruning = true;
    flip(*query);
    SearchResult rejected;
    EXPECT_EQ(core::IndexError_Unsupported,
              index->search(input, query, &rejected))
        << name;
  }
  query->rabitq_pruning = true;
  query->rabitq_reuse_coarse_dot = false;

  // Context-switch regression: reuse+refiner, reuse-only and refiner-only
  // queries interleave on one thread; returning to plain mode must
  // reproduce the untouched baseline exactly.
  for (int round = 0; round < 2; ++round) {
    query->refiner_param = refiner;
    query->rabitq_reuse_coarse_dot = true;
    ASSERT_EQ(0, index->search(input, query, &on));
    EXPECT_EQ(5u, on.doc_list_.size());
    query->refiner_param = nullptr;
    ASSERT_EQ(0, index->search(input, query, &on));
    query->rabitq_reuse_coarse_dot = false;
    query->refiner_param = refiner;
    ASSERT_EQ(0, index->search(input, query, &off));
    EXPECT_EQ(5u, off.doc_list_.size());
    query->refiner_param = nullptr;
    SearchResult plain_after;
    ASSERT_EQ(0, index->search(input, query, &plain_after));
    ASSERT_EQ(plain_before.doc_list_.size(), plain_after.doc_list_.size());
    for (size_t i = 0; i < plain_before.doc_list_.size(); ++i) {
      EXPECT_EQ(plain_before.doc_list_[i].key(),
                plain_after.doc_list_[i].key());
      EXPECT_FLOAT_EQ(plain_before.doc_list_[i].score(),
                      plain_after.doc_list_[i].score());
    }
  }

  ASSERT_EQ(0, flat->close());
  test_util::RemoveTestFiles(flat_path);
  ASSERT_EQ(0, index->close());
  test_util::RemoveTestFiles(path);
}

INSTANTIATE_TEST_SUITE_P(
    MetricsAndBits, IVFRabitqIntegrationTest,
    testing::Combine(testing::Values(MetricType::kL2sq,
                                     MetricType::kInnerProduct,
                                     MetricType::kCosine),
                     testing::Values(1, 7, 9), testing::Bool()));
#else
TEST(IVFRabitqIntegration, UnsupportedPlatform) {
  EXPECT_EQ(nullptr,
            IndexFactory::CreateAndInitIndex(*RabitqParam(MetricType::kL2sq)));
}
#endif
}  // namespace
}  // namespace zvec::core_interface
