// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "ailego/pattern/defer.h"
#include "algorithm/ivf/ivf_builder.h"
#include "algorithm/ivf/ivf_index_format.h"
#include "algorithm/ivf/ivf_streamer.h"
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

IVFIndexParam::Pointer RabitqParam(MetricType metric, int bits = 7) {
  RabitqQuantizerParam quantizer(bits);
  quantizer.num_clusters = 4;
  quantizer.sample_count = 80;
  quantizer.niters = 3;
  return IVFIndexParamBuilder()
      .with_dimension(128)
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_n_list(4)
      .with_n_iters(3)
      .with_quantizer_param(quantizer)
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
  auto json = param->serialize_to_json();
  json.insert(1, "\"total_bits\":9,\"sample_count\":80,");
  auto decoded = std::dynamic_pointer_cast<IVFIndexParam>(
      IndexFactory::DeserializeIndexParamFromJson(json));
  ASSERT_NE(nullptr, decoded);
  auto typed =
      std::dynamic_pointer_cast<RabitqQuantizerParam>(decoded->quantizer_param);
  ASSERT_NE(nullptr, typed);
  EXPECT_EQ(quantizer.serialize_to_json(), typed->serialize_to_json());
}

TEST(IVFRabitqIntegration, MigratesLegacyJsonWithoutDuplicatingNewParameters) {
  auto param = RabitqParam(MetricType::kL2sq);
  param->quantizer_param =
      std::make_shared<QuantizerParam>(QuantizerType::kRabitq);
  const auto current = param->serialize_to_json();
  // A generic default quantizer no longer borrows IVF partition parameters.
  auto defaults = std::dynamic_pointer_cast<IVFIndexParam>(
      IndexFactory::DeserializeIndexParamFromJson(current));
  ASSERT_NE(nullptr, defaults);
  auto q = std::dynamic_pointer_cast<RabitqQuantizerParam>(
      defaults->quantizer_param);
  ASSERT_NE(nullptr, q);
  EXPECT_EQ(16, q->num_clusters);
  EXPECT_EQ(20, q->niters);
  auto legacy = current;
  legacy.insert(1, "\"total_bits\":5,\"sample_count\":24,");
  auto migrated = std::dynamic_pointer_cast<IVFIndexParam>(
      IndexFactory::DeserializeIndexParamFromJson(legacy));
  ASSERT_NE(nullptr, migrated);
  q = std::dynamic_pointer_cast<RabitqQuantizerParam>(
      migrated->quantizer_param);
  ASSERT_NE(nullptr, q);
  EXPECT_EQ(5, q->total_bits);
  EXPECT_EQ(24, q->sample_count);
  EXPECT_EQ(4, q->num_clusters);
  EXPECT_EQ(3, q->niters);
  const auto canonical = migrated->serialize_to_json();
  auto pos = canonical.find("\"total_bits\"");
  ASSERT_NE(std::string::npos, pos);
  EXPECT_EQ(std::string::npos, canonical.find("\"total_bits\"", pos + 1));
  auto reopened = IndexFactory::DeserializeIndexParamFromJson(canonical);
  ASSERT_NE(nullptr, reopened);
  EXPECT_EQ(canonical, reopened->serialize_to_json());
}

TEST(IVFRabitqIntegration, RejectsInvalidNestedAndLegacyJson) {
  auto param = RabitqParam(MetricType::kL2sq);
  param->quantizer_param =
      std::make_shared<QuantizerParam>(QuantizerType::kRabitq);
  auto legacy = param->serialize_to_json();
  legacy.insert(1, "\"total_bits\":10,\"sample_count\":80,");
  EXPECT_EQ(nullptr, IndexFactory::DeserializeIndexParamFromJson(legacy));
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
// Observe the groups used during build (after IVF partition training) while
// forwarding all work to the actual pool supplied by merge().
struct MergeWorkerProbe {
  ailego::ThreadPool pool{1, false};
  std::atomic<size_t> groups{0};
};

class ObservedBuildThreads : public core::IndexThreads {
 public:
  ObservedBuildThreads(Pointer threads, MergeWorkerProbe *probe)
      : threads_(std::move(threads)), probe_(probe) {}
  size_t count() const override {
    return threads_->count();
  }
  int indexof_this() const override {
    return threads_->indexof_this();
  }
  void stop() override {
    threads_->stop();
  }
  void submit(ailego::ClosureHandler &&task) override {
    threads_->submit(std::move(task));
  }
  TaskGroup::Pointer make_group() override {
    auto group = threads_->make_group();
    group->submit(ailego::Closure::New([probe = probe_]() {
      EXPECT_EQ(0, probe->pool.indexof_this());
      ++probe->groups;
    }));
    return group;
  }

 private:
  Pointer threads_;
  MergeWorkerProbe *probe_;
};

class ObservedIVFBuilder : public core::IVFBuilder {
 public:
  explicit ObservedIVFBuilder(MergeWorkerProbe *probe) : probe_(probe) {}
  int build(core::IndexThreads::Pointer threads,
            core::IndexHolder::Pointer holder) override {
    if (!threads) return core::IndexError_InvalidArgument;
    return IVFBuilder::build(
        std::make_shared<ObservedBuildThreads>(std::move(threads), probe_),
        std::move(holder));
  }

 private:
  MergeWorkerProbe *probe_;
};

TEST(IVFRabitqIntegration, MergeUsesCallerWorkerForQuantizerClustering) {
  const std::string path = "ivf_rabitq_merge_worker.index";
  const auto source_path = path + ".flat";
  test_util::RemoveTestFiles(source_path);
  const StorageOptions create{StorageOptions::StorageType::kMMAP, true};
  auto source =
      IndexFactory::CreateAndInitIndex(*FlatIndexParamBuilder()
                                            .with_dimension(128)
                                            .with_data_type(DataType::DT_FP32)
                                            .with_metric_type(MetricType::kL2sq)
                                            .build());
  ASSERT_TRUE(source);
  ASSERT_EQ(0, source->open(source_path, create));
  for (uint32_t i = 0; i < 64; ++i) {
    std::vector<float> row(128, static_cast<float>(i));
    ASSERT_EQ(0, source->add(VectorData{DenseVector{row.data()}}, i));
  }

  MergeWorkerProbe probe;
  using Factory = ailego::Factory<core::IndexBuilder>;
  Factory::Register<ObservedIVFBuilder> observe("IVFBuilder", &probe);
  AILEGO_DEFER(
      []() { Factory::Register<core::IVFBuilder> restore("IVFBuilder"); });
  size_t groups_without_clustering = 0;
  for (int samples : {4, 64}) {
    auto param = RabitqParam(MetricType::kL2sq);
    auto q =
        std::dynamic_pointer_cast<RabitqQuantizerParam>(param->quantizer_param);
    ASSERT_TRUE(q);
    q->sample_count = samples;
    auto target = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_TRUE(target);
    test_util::RemoveTestFiles(path);
    ASSERT_EQ(0, target->open(path, create));
    probe.groups = 0;
    ASSERT_EQ(0, target->merge({source}, {}, {1, &probe.pool}));
    EXPECT_EQ(64u, target->get_doc_count());
    // Four samples are used directly as centers. With 64, only RaBitQ
    // clustering introduces extra groups into the same builder workload.
    if (samples == 4) {
      groups_without_clustering = probe.groups.load();
      EXPECT_GT(groups_without_clustering, 0u);
    } else {
      EXPECT_GT(probe.groups.load(), groups_without_clustering);
    }
    ASSERT_EQ(0, target->close());
    test_util::RemoveTestFiles(path);
  }
  ASSERT_EQ(0, source->close());
  test_util::RemoveTestFiles(source_path);
}

TEST(IVFRabitqIntegration, MergeRetryTrainsOnReplacementInputs) {
  const std::string path = "ivf_rabitq_merge_retry";
  const auto source_path = path + ".flat";
  const StorageOptions create{StorageOptions::StorageType::kMMAP, true};
  for (const auto &file : {path, source_path}) test_util::RemoveTestFiles(file);
  auto source =
      IndexFactory::CreateAndInitIndex(*FlatIndexParamBuilder()
                                            .with_dimension(128)
                                            .with_metric_type(MetricType::kL2sq)
                                            .with_data_type(DataType::DT_FP32)
                                            .build());
  ASSERT_TRUE(source);
  ASSERT_EQ(0, source->open(source_path, create));
  std::vector<std::vector<float>> rows(8, std::vector<float>(128));
  std::mt19937 random(73);
  for (uint32_t id = 0; id < rows.size(); ++id) {
    auto &row = rows[id];
    for (auto &value : row) value = static_cast<float>(random() % 127) / 127;
    ASSERT_EQ(0, source->add(VectorData{DenseVector{row.data()}}, id));
  }
  auto params = RabitqParam(MetricType::kL2sq);
  auto target = IndexFactory::CreateAndInitIndex(*params);
  ASSERT_TRUE(target);
  std::weak_ptr<turbo::Quantizer> build_model =
      std::dynamic_pointer_cast<core::IVFStreamer>(target->index_searcher())
          ->quantizer();
  {
    std::ofstream blocker(path);
    ASSERT_TRUE(blocker.good());
  }
  ASSERT_EQ(0, target->open(path + "/index", create));
  IndexFilter first, second;
  first.set([](uint64_t id) { return id >= 4; });
  second.set([](uint64_t id) { return id < 4; });
  EXPECT_NE(0, target->merge({source}, first));
  test_util::RemoveTestFiles(path);
  ASSERT_EQ(0, target->merge({source}, second));
  EXPECT_TRUE(build_model.expired());
  auto direct = IndexFactory::CreateAndInitIndex(*params);
  ASSERT_TRUE(direct);
  ASSERT_EQ(0, direct->open(path + "/direct", create));
  ASSERT_EQ(0, direct->merge({source}, second));
  // Each of the four training vectors is its own centroid. Both indexes
  // should return the exact self distance with the replacement codebook.
  auto query = IVFQueryParamBuilder().with_nprobe(4).with_topk(1).build();
  for (uint32_t id = 4; id < 8; ++id) {
    const auto &row = rows[id];
    for (const auto &index : {direct, target}) {
      SearchResult result;
      ASSERT_EQ(0, index->search(VectorData{DenseVector{row.data()}}, query,
                                 &result));
      ASSERT_EQ(1u, result.doc_list_.size());
      EXPECT_EQ(id - 4, result.doc_list_.front().key());
      EXPECT_FLOAT_EQ(0, result.doc_list_.front().score());
    }
  }
  ASSERT_EQ(0, direct->close());
  ASSERT_EQ(0, target->close());
  ASSERT_EQ(0, source->close());
  test_util::RemoveTestFiles(path);
  test_util::RemoveTestFiles(source_path);
}

TEST(IVFRabitqIntegration, OneBitKeyedScoresMatchListScan) {
  constexpr size_t count = 96;
  auto params = RabitqParam(MetricType::kL2sq, 1);
  params->nlist = 2;
  auto index = IndexFactory::CreateAndInitIndex(*params);
  ASSERT_TRUE(index);
  const std::string path = "ivf_one_bit_scores.index";
  test_util::RemoveTestFiles(path);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::mt19937 random(912);
  std::normal_distribution<float> normal;
  std::vector<float> vector(128);
  auto keys = std::make_shared<std::vector<uint64_t>>();
  for (size_t i = 0; i < count; ++i) {
    for (auto &v : vector) v = normal(random);
    keys->push_back(3 * i);
    ASSERT_EQ(0,
              index->add(VectorData{DenseVector{vector.data()}}, keys->back()));
  }
  ASSERT_EQ(0, index->train());
  std::shuffle(keys->begin(), keys->end(), random);
  for (auto &v : vector) v = normal(random);
  auto query = IVFQueryParamBuilder().with_topk(count).with_nprobe(2).build();
  auto input = VectorData{DenseVector{vector.data()}};
  SearchResult all;
  ASSERT_EQ(0, index->search(input, query, &all));
  ASSERT_EQ(count, all.doc_list_.size());
  const auto compare = [&](size_t expected_count) {
    SearchResult scan, keyed;
    query->bf_pks.reset();
    ASSERT_EQ(0, index->search(input, query, &scan));
    query->bf_pks = keys;
    ASSERT_EQ(0, index->search(input, query, &keyed));
    ASSERT_EQ(expected_count, scan.doc_list_.size());
    ASSERT_EQ(expected_count, keyed.doc_list_.size());
    for (size_t i = 0; i < expected_count; ++i) {
      EXPECT_EQ(scan.doc_list_[i].key(), keyed.doc_list_[i].key());
      EXPECT_EQ(scan.doc_list_[i].score(), keyed.doc_list_[i].score());
    }
  };
  compare(count);
  query->topk = 1;
  compare(1);
  query->topk = count;
  query->radius = (all.doc_list_[47].score() + all.doc_list_[48].score()) / 2;
  compare(48);
  ASSERT_EQ(0, index->close());
  test_util::RemoveTestFiles(path);
}

TEST(IVFRabitqIntegration, CosineSearchPreservesLargeFiniteDirections) {
  const std::string path = "ivf_cosine_scale.index";
  for (float scale : {1.f, 1e38f}) {
    auto params = RabitqParam(MetricType::kCosine, 7);
    params->nlist = 1;
    auto index = IndexFactory::CreateAndInitIndex(*params);
    ASSERT_TRUE(index);
    test_util::RemoveTestFiles(path);
    ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
    std::vector<float> values(params->dimension, scale);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{values.data()}}, 1));
    std::fill(values.begin(), values.end(), -scale);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{values.data()}}, 2));
    ASSERT_EQ(0, index->train());
    auto params_query =
        IVFQueryParamBuilder().with_topk(2).with_nprobe(1).build();
    params_query->bf_pks = std::make_shared<std::vector<uint64_t>>(
        std::initializer_list<uint64_t>{1, 2});
    for (float query_scale : {1.f, 1e38f}) {
      std::fill(values.begin(), values.end(), query_scale);
      SearchResult results;
      ASSERT_EQ(0, index->search(VectorData{DenseVector{values.data()}},
                                 params_query, &results));
      ASSERT_EQ(2u, results.doc_list_.size());
      EXPECT_EQ(1u, results.doc_list_[0].key());
      EXPECT_NEAR(0, results.doc_list_[0].score(), 1e-4);
      EXPECT_EQ(2u, results.doc_list_[1].key());
      EXPECT_NEAR(2, results.doc_list_[1].score(), 1e-4);
    }
    ASSERT_EQ(0, index->close());
    test_util::RemoveTestFiles(path);
  }
}

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
  std::weak_ptr<turbo::Quantizer> build_model =
      std::dynamic_pointer_cast<core::IVFStreamer>(index->index_searcher())
          ->quantizer();
  ASSERT_FALSE(build_model.expired());
  ASSERT_EQ(0, index->train());
  EXPECT_TRUE(build_model.expired());
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
  if (metric == MetricType::kL2sq && bits == 7 && !std::get<2>(GetParam())) {
    auto wrong = RabitqParam(metric);
    wrong->dimension = 64;
    auto retry = IndexFactory::CreateAndInitIndex(*wrong);
    ASSERT_TRUE(retry);
    EXPECT_EQ(core::IndexError_Mismatch,
              retry->open(legacy_path,
                          {StorageOptions::StorageType::kMMAP, false, true}));
    const auto fresh = legacy_path + ".new";
    test_util::RemoveTestFiles(fresh);
    ASSERT_EQ(0,
              retry->open(fresh, {StorageOptions::StorageType::kMMAP, true}));
    for (uint32_t i = 0; i < 8; ++i)
      ASSERT_EQ(0, retry->add(VectorData{DenseVector{vectors[i].data()}}, i));
    ASSERT_EQ(0, retry->train());
    EXPECT_EQ("RabitqQuantizer",
              retry->index_searcher()->meta().quantizer_name());
    ASSERT_EQ(0, retry->close());
    test_util::RemoveTestFiles(fresh);
  }
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
  auto index =
      IndexFactory::CreateAndInitIndex(*RabitqParam(MetricType::kL2sq));
  ASSERT_TRUE(index);
  EXPECT_EQ(core::IndexError_Unsupported,
            index->open("ivf_unsupported.index",
                        {StorageOptions::StorageType::kMMAP, true}));
}
#endif
}  // namespace
}  // namespace zvec::core_interface
