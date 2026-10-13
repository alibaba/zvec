// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <cmath>
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
  param.rabitq_collect_stats = true;
  param.rabitq_segment_timing = true;
  param.rabitq_export_scan_lists = true;
  const auto json = IndexFactory::QueryParamSerializeToJson(param, true);
  auto typed = IndexFactory::QueryParamDeserializeFromJson<IVFQueryParam>(json);
  auto base = std::dynamic_pointer_cast<IVFQueryParam>(
      IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(json));
  auto copy = std::dynamic_pointer_cast<IVFQueryParam>(param.clone());
  for (const auto &p : {typed, base, copy}) {
    ASSERT_NE(nullptr, p);
    EXPECT_FALSE(p->rabitq_pruning);
    EXPECT_TRUE(p->rabitq_reuse_coarse_dot);
    EXPECT_TRUE(p->rabitq_collect_stats);
    EXPECT_TRUE(p->rabitq_segment_timing);
    EXPECT_TRUE(p->rabitq_export_scan_lists);
  }
  auto defaults = IndexFactory::QueryParamDeserializeFromJson<IVFQueryParam>(
      R"({"index_type":"kIVF","nprobe":16})");
  ASSERT_NE(nullptr, defaults);
  EXPECT_TRUE(defaults->rabitq_pruning);
  EXPECT_FALSE(defaults->rabitq_reuse_coarse_dot);
  EXPECT_FALSE(defaults->rabitq_collect_stats);
  EXPECT_FALSE(defaults->rabitq_segment_timing);
  EXPECT_FALSE(defaults->rabitq_export_scan_lists);
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<IVFQueryParam>(
                R"({"index_type":"kIVF","rabitq_segment_timing":"yes"})"));
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<IVFQueryParam>(
                R"({"index_type":"kIVF","rabitq_export_scan_lists":"yes"})"));
}

TEST(IVFRabitqIntegration, DedicatedScanOptionsRoundTrip) {
  IVFRabitqQueryParam param;
  param.rabitq_collect_stats = true;
  param.rabitq_pruning = false;
  param.rabitq_export_scan_lists = true;
  const auto json = IndexFactory::QueryParamSerializeToJson(param, true);
  auto typed =
      IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(json);
  auto base = std::dynamic_pointer_cast<IVFRabitqQueryParam>(
      IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(json));
  auto copy = std::dynamic_pointer_cast<IVFRabitqQueryParam>(param.clone());
  for (const auto &p : {typed, base, copy}) {
    ASSERT_NE(nullptr, p);
    EXPECT_TRUE(p->rabitq_collect_stats);
    EXPECT_FALSE(p->rabitq_pruning);
    EXPECT_TRUE(p->rabitq_export_scan_lists);
  }
  auto defaults =
      IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
          R"({"index_type":"kIVFRabitq","nprobe":16})");
  ASSERT_NE(nullptr, defaults);
  EXPECT_FALSE(defaults->rabitq_collect_stats);
  EXPECT_TRUE(defaults->rabitq_pruning);
  EXPECT_FALSE(defaults->rabitq_export_scan_lists);
  const auto invalid =
      R"({"index_type":"kIVFRabitq","rabitq_collect_stats":"yes"})";
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
                invalid));
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(
                invalid));
  EXPECT_EQ(nullptr,
            IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
                R"({"index_type":"kIVFRabitq","rabitq_pruning":"yes"})"));
  EXPECT_EQ(
      nullptr,
      IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
          R"({"index_type":"kIVFRabitq","rabitq_export_scan_lists":"yes"})"));
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
  SearchResult diagnostics;
  query->rabitq_collect_stats = true;
  for (bool reuse : {false, true}) {
    query->rabitq_reuse_coarse_dot = reuse;
    query->rabitq_pruning = false;
    ASSERT_EQ(0, index->search(input, query, &diagnostics));
    EXPECT_TRUE(diagnostics.ivf_scan_stats_.collected);
    EXPECT_TRUE(diagnostics.ivf_scan_stats_.active);
    EXPECT_TRUE(diagnostics.ivf_scan_stats_.mapped);
    EXPECT_EQ(80u, diagnostics.ivf_scan_stats_.coarse_count);
    EXPECT_EQ(80u, diagnostics.ivf_scan_stats_.refined_count);
    EXPECT_EQ(0u, diagnostics.ivf_scan_stats_.coarse_reads);
    EXPECT_EQ(0u, diagnostics.ivf_scan_stats_.row_reads);
  }
  query->rabitq_pruning = true;
  query->rabitq_reuse_coarse_dot = false;
  query->rabitq_collect_stats = false;
  query->rabitq_segment_timing = true;
  ASSERT_EQ(0, index->search(input, query, &diagnostics));
  const auto &timing = diagnostics.ivf_scan_stats_;
  EXPECT_TRUE(timing.collected);
  EXPECT_GT(timing.query_ns, 0.0);
  EXPECT_GT(timing.prep_ns, 0.0);
  EXPECT_GT(timing.coarse_ns, 0.0);
  EXPECT_GT(timing.post_scan_ns, 0.0);
  EXPECT_LE(timing.prep_ns + timing.cand_ns + timing.coarse_ns +
                timing.post_scan_ns + timing.finalize_ns,
            timing.query_ns);
  ASSERT_EQ(before.doc_list_.size(), diagnostics.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(before.doc_list_[i].key(), diagnostics.doc_list_[i].key());
    EXPECT_FLOAT_EQ(before.doc_list_[i].score(),
                    diagnostics.doc_list_[i].score());
  }
  query->rabitq_segment_timing = false;
  ASSERT_EQ(0, index->search(input, query, &diagnostics));
  EXPECT_FALSE(diagnostics.ivf_scan_stats_.collected);
  EXPECT_EQ(0.0, diagnostics.ivf_scan_stats_.query_ns);
  ASSERT_EQ(before.doc_list_.size(), diagnostics.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(before.doc_list_[i].key(), diagnostics.doc_list_[i].key());
    EXPECT_FLOAT_EQ(before.doc_list_[i].score(),
                    diagnostics.doc_list_[i].score());
  }
  // Real scanned-list export reproduces the default results, accounts for
  // every scanned vector, and must not leak into later default queries.
  query->rabitq_export_scan_lists = true;
  SearchResult exported;
  ASSERT_EQ(0, index->search(input, query, &exported));
  const auto &lists = exported.ivf_scan_stats_;
  EXPECT_TRUE(lists.collected);
  EXPECT_TRUE(lists.scan_lists_collected);
  EXPECT_FALSE(lists.scan_lists_limited);
  EXPECT_LE(lists.scan_list_ids.size(), 4u);
  size_t scanned_vectors = 0;
  for (size_t count : lists.scan_list_counts) scanned_vectors += count;
  EXPECT_EQ(lists.coarse_count, scanned_vectors);
  ASSERT_EQ(before.doc_list_.size(), exported.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(before.doc_list_[i].key(), exported.doc_list_[i].key());
    EXPECT_FLOAT_EQ(before.doc_list_[i].score(), exported.doc_list_[i].score());
  }
  query->rabitq_export_scan_lists = false;
  SearchResult after_export;
  ASSERT_EQ(0, index->search(input, query, &after_export));
  EXPECT_FALSE(after_export.ivf_scan_stats_.collected);
  ASSERT_EQ(before.doc_list_.size(), after_export.doc_list_.size());
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
  // Dedicated statistics must be query-local and must not change results.
  const auto native_result = legacy_result;
  legacy_query->rabitq_collect_stats = true;
  size_t refined_count = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
    const auto &scan = legacy_result.ivf_scan_stats_;
    EXPECT_TRUE(scan.collected);
    EXPECT_TRUE(scan.active);
    EXPECT_TRUE(scan.mapped);
    EXPECT_EQ(vectors.size(), scan.coarse_count);  // excludes padded lanes
    EXPECT_GT(scan.refined_count, 0u);
    EXPECT_LE(scan.refined_count, scan.coarse_count);
    if (attempt) EXPECT_EQ(refined_count, scan.refined_count);
    refined_count = scan.refined_count;
    EXPECT_EQ(0u, scan.key_reads + scan.coarse_reads + scan.row_reads +
                      scan.coarse_copy_bytes);
    ASSERT_EQ(native_result.doc_list_.size(), legacy_result.doc_list_.size());
    for (size_t i = 0; i < native_result.doc_list_.size(); ++i) {
      EXPECT_EQ(native_result.doc_list_[i].key(),
                legacy_result.doc_list_[i].key());
      EXPECT_FLOAT_EQ(native_result.doc_list_[i].score(),
                      legacy_result.doc_list_[i].score());
    }
  }
  // Export alone also supplies truthful counts, without changing results.
  legacy_query->rabitq_collect_stats = false;
  legacy_query->rabitq_export_scan_lists = true;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  const auto &dedicated_export = legacy_result.ivf_scan_stats_;
  EXPECT_TRUE(dedicated_export.collected);
  EXPECT_TRUE(dedicated_export.active);
  EXPECT_TRUE(dedicated_export.mapped);
  EXPECT_TRUE(dedicated_export.scan_lists_collected);
  EXPECT_FALSE(dedicated_export.scan_lists_limited);
  EXPECT_EQ(vectors.size(), dedicated_export.coarse_count);
  EXPECT_EQ(refined_count, dedicated_export.refined_count);
  EXPECT_FALSE(dedicated_export.scan_list_ids.empty());
  ASSERT_EQ(dedicated_export.scan_list_ids.size(),
            dedicated_export.scan_list_counts.size());
  size_t dedicated_export_scanned = 0;
  for (size_t count : dedicated_export.scan_list_counts)
    dedicated_export_scanned += count;
  EXPECT_EQ(dedicated_export.coarse_count, dedicated_export_scanned);
  ASSERT_EQ(native_result.doc_list_.size(), legacy_result.doc_list_.size());
  for (size_t i = 0; i < native_result.doc_list_.size(); ++i) {
    EXPECT_EQ(native_result.doc_list_[i].key(),
              legacy_result.doc_list_[i].key());
    EXPECT_FLOAT_EQ(native_result.doc_list_[i].score(),
                    legacy_result.doc_list_[i].score());
  }
  // Disable-pruning diagnostic: codes, lists and the distance formula stay
  // fixed and only the prune check is skipped, so every lane is refined and
  // the exported lists account for the whole scanned volume. Resetting the
  // switch must restore the native default-path results.
  legacy_query->topk = 5;
  legacy_query->rabitq_pruning = false;
  legacy_query->rabitq_export_scan_lists = true;
  SearchResult unpruned_result;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &unpruned_result));
  const auto &unpruned = unpruned_result.ivf_scan_stats_;
  EXPECT_TRUE(unpruned.collected);
  EXPECT_TRUE(unpruned.scan_lists_collected);
  EXPECT_FALSE(unpruned.scan_lists_limited);
  EXPECT_EQ(unpruned.coarse_count, vectors.size());
  EXPECT_EQ(unpruned.refined_count, unpruned.coarse_count);
  EXPECT_LE(unpruned.scan_list_ids.size(), 4u);
  size_t unpruned_scanned = 0;
  for (size_t count : unpruned.scan_list_counts) unpruned_scanned += count;
  EXPECT_EQ(unpruned.coarse_count, unpruned_scanned);
  legacy_query->rabitq_pruning = true;
  legacy_query->rabitq_export_scan_lists = false;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  EXPECT_FALSE(legacy_result.ivf_scan_stats_.collected);
  EXPECT_FALSE(legacy_result.ivf_scan_stats_.scan_lists_collected);
  EXPECT_EQ(0u, legacy_result.ivf_scan_stats_.coarse_count);
  EXPECT_EQ(0u, legacy_result.ivf_scan_stats_.refined_count);
  EXPECT_TRUE(legacy_result.ivf_scan_stats_.scan_list_ids.empty());
  EXPECT_TRUE(legacy_result.ivf_scan_stats_.scan_list_counts.empty());
  ASSERT_EQ(native_result.doc_list_.size(), legacy_result.doc_list_.size());
  for (size_t i = 0; i < native_result.doc_list_.size(); ++i) {
    EXPECT_EQ(native_result.doc_list_[i].key(),
              legacy_result.doc_list_[i].key());
    EXPECT_FLOAT_EQ(native_result.doc_list_[i].score(),
                    legacy_result.doc_list_[i].score());
  }
  // A heap holding the entire corpus cannot prune a real lane before it fills.
  legacy_query->rabitq_collect_stats = true;
  legacy_query->topk = vectors.size();
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  EXPECT_EQ(vectors.size(), legacy_result.ivf_scan_stats_.refined_count);
  legacy_query->rabitq_collect_stats = false;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  EXPECT_FALSE(legacy_result.ivf_scan_stats_.collected);
  EXPECT_EQ(0u, legacy_result.ivf_scan_stats_.coarse_count);
  legacy_query->rabitq_collect_stats = true;
  legacy_query->bf_pks = std::make_shared<std::vector<uint64_t>>();
  EXPECT_EQ(core::IndexError_InvalidArgument,
            legacy->search(input, legacy_query, &legacy_result));
  EXPECT_FALSE(legacy_result.ivf_scan_stats_.collected);
  legacy_query->bf_pks.reset();
  legacy_query->group_by_param = group_query->group_by_param;
  EXPECT_EQ(core::IndexError_InvalidArgument,
            legacy->search(input, legacy_query, &legacy_result));
  legacy_query->rabitq_collect_stats = false;
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
