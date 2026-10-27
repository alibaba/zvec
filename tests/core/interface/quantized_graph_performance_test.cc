// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <unistd.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_quantized_graph.h"
#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"
#include "algorithm/quantized_graph/quantized_graph_sq8_center.h"
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
#endif

namespace ci = zvec::core_interface;
using namespace zvec::core;
struct TestHnswIndex : ci::HNSWIndex {
  using ci::HNSWIndex::init;
  const float *stored_vector(uint32_t id) const {
    return static_cast<const float *>(streamer_->get_vector_by_id(id));
  }
};

TEST(QuantizedGraphPerformance, QueryJsonPreservesExplicitZeroAndDefaults) {
  for (bool omit : {false, true}) {
    for (bool defaults : {false, true}) {
      ci::HNSWQueryParam input;
      if (!defaults) {
        input.quantized_graph_margin = 0;
        input.quantized_graph_bucket_width = .004f;
        input.quantized_graph_route_beam = 0;
        input.quantized_graph_max_expansions = 256;
      }
      const auto json =
          ci::IndexFactory::QueryParamSerializeToJson(input, omit);
      auto typed =
          ci::IndexFactory::QueryParamDeserializeFromJson<ci::HNSWQueryParam>(
              json);
      auto base = ci::IndexFactory::QueryParamDeserializeFromJson<
          ci::BaseIndexQueryParam>(json);
      auto derived = std::dynamic_pointer_cast<ci::HNSWQueryParam>(base);
      for (const auto &output : {typed, derived}) {
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(input.quantized_graph_margin, output->quantized_graph_margin);
        EXPECT_EQ(input.quantized_graph_bucket_width,
                  output->quantized_graph_bucket_width);
        EXPECT_EQ(input.quantized_graph_route_beam,
                  output->quantized_graph_route_beam);
        EXPECT_EQ(input.quantized_graph_max_expansions,
                  output->quantized_graph_max_expansions);
      }
    }
  }
  for (const std::string json :
       {R"({"index_type":"kHNSW","quantized_graph_route_beam":5})",
        R"({"index_type":"kHNSW","quantized_graph_route_beam":-1})",
        R"({"index_type":"kHNSW","quantized_graph_route_beam":4294967296})",
        R"({"index_type":"kHNSW","quantized_graph_max_expansions":-1})",
        R"({"index_type":"kHNSW","quantized_graph_max_expansions":4294967296})",
        R"({"index_type":"kHNSW","quantized_graph_margin":-2})",
        R"({"index_type":"kHNSW","quantized_graph_bucket_width":1.1})",
        R"({"index_type":"kHNSW","quantized_graph_margin":"bad"})"}) {
    EXPECT_EQ(
        nullptr,
        ci::IndexFactory::QueryParamDeserializeFromJson<ci::HNSWQueryParam>(
            json));
    EXPECT_EQ(nullptr, ci::IndexFactory::QueryParamDeserializeFromJson<
                           ci::BaseIndexQueryParam>(json));
  }
}

TEST(QuantizedGraphPerformance, ParameterRoundTripAndUnsupportedLayouts) {
  auto p = ci::HNSWIndexParamBuilder()
               .with_metric_type(ci::MetricType::kCosine)
               .with_dimension(768)
               .with_use_quantized_graph(true)
               .with_quantized_graph_model("model with spaces/\"router\".bin")
               .build();
  for (bool omit : {false, true}) {
    ci::HNSWIndexParam restored;
    ASSERT_TRUE(restored.deserialize_from_json(p->serialize_to_json(omit)))
        << p->serialize_to_json(omit);
    EXPECT_EQ(restored.quantized_graph_model, p->quantized_graph_model);
    EXPECT_TRUE(restored.use_quantized_graph);
  }
  ci::HNSWIndexParam legacy;
  EXPECT_TRUE(legacy.quantized_graph_model.empty());
  EXPECT_FALSE(legacy.use_quantized_graph);
  TestHnswIndex index;
  p->data_type = ci::DataType::DT_FP32;
  p->dimension = 4097;
  EXPECT_NE(0, index.init(*p));
  p->dimension = 768;
  p->metric_type = ci::MetricType::kInnerProduct;
  EXPECT_NE(0, index.init(*p));
  p->metric_type = ci::MetricType::kCosine;
  p->use_external_vector = true;
  EXPECT_NE(0, index.init(*p));
  p->use_external_vector = false;
  p->quantizer_param =
      std::make_shared<ci::QuantizerParam>(ci::QuantizerType::kInt8);
  EXPECT_NE(0, index.init(*p));
}

#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
namespace {
struct TemporaryFiles {
  std::filesystem::path dir = std::filesystem::temp_directory_path() /
                              ("zvec-qg-" + std::to_string(getpid()));
  TemporaryFiles() {
    std::filesystem::create_directory(dir);
  }
  ~TemporaryFiles() {
    std::filesystem::remove_all(dir);
  }
};
struct AllValid {
  bool valid(uint32_t) const {
    return true;
  }
};
void WriteModel(const std::filesystem::path &path, uint32_t docs,
                int32_t target = -1, uint32_t tag = 0x53514701) {
  std::ofstream out(path, std::ios::binary);
  uint32_t header[]{0x51475259, 2, 384, 1, docs};
  std::array<float, 768> mean{};
  uint32_t count = 1;
  std::array<int32_t, 16> targets{};
  targets[0] = target;
  std::array<float, 16> factors{};
  std::array<uint8_t, 16 * 384> codes{};
  auto put = [&](const void *p, size_t n) {
    out.write(static_cast<const char *>(p), n);
  };
  put(header, sizeof(header));
  put(mean.data(), sizeof(mean));
  put(&tag, sizeof(tag));
  put(mean.data(), sizeof(mean));
  put(&count, sizeof(count));
  put(targets.data(), sizeof(targets));
  for (int i = 0; i < 3; ++i) put(factors.data(), sizeof(factors));
  put(codes.data(), sizeof(codes));
  ASSERT_TRUE(out.good());
}
}  // namespace
TEST(QuantizedGraphPerformance, ModelValidation) {
  if (!HnswQuantizedGraph::supported()) {
    GTEST_SKIP();
  }
  TemporaryFiles tmp;
  const auto path = (tmp.dir / "router.bin").string();
  QuantizedGraphIntegerRouter router;
  std::vector<float> mean(768, 0), rotated(768, 0);
  auto load = [&](size_t docs = 100) {
    return router.load(path.c_str(), mean, AllValid{}, docs, &rotated);
  };
  WriteModel(path, 100);
  ASSERT_TRUE(QuantizedGraphIntegerRouter::read_mean(path.c_str(), 100, mean));
  ASSERT_TRUE(load());
  IntegerCenter::Query query;
  ASSERT_TRUE(router.prepare_projected(rotated.data(), query));
  EXPECT_EQ(0u, router.choose(query, 1));
  EXPECT_FALSE(load(101));
  EXPECT_TRUE(router.empty());
  WriteModel(path, 100, -101);
  EXPECT_FALSE(load());
  WriteModel(path, 100, 0);  // Back edge / cycle.
  EXPECT_FALSE(load());
  WriteModel(path, 100, -1, 0);  // Incompatible rotation.
  EXPECT_FALSE(load());
  WriteModel(path, 100);
  {
    std::ofstream out(path, std::ios::app | std::ios::binary);
    out.put('x');
  }
  EXPECT_FALSE(load());
  std::filesystem::resize_file(path, 100);
  EXPECT_FALSE(load());
  EXPECT_FALSE(QuantizedGraphIntegerRouter::read_mean(path.c_str(), 100, mean));
}
TEST(QuantizedGraphPerformance, IntegerBoundAndRotation) {
  if (!HnswQuantizedGraph::supported()) {
    GTEST_SKIP();
  }
  std::mt19937 rng(20261027);
  std::normal_distribution<float> normal;
  zvec::turbo::SymphonyQGRotation rotation(768);
  std::vector<float> q(768), c(768), mean(768), rotated;
  std::array<uint8_t, 768> codes{};
  for (int trial = 0; trial < 128; ++trial) {
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = normal(rng) / 28;
      c[i] = normal(rng) / 28;
      mean[i] = normal(rng) / 100;
    }
    double norm = 0, rotated_norm = 0;
    for (float x : q) norm += double(x) * x;
    rotation.rotate(q.data(), rotated);
    ASSERT_EQ(768u, rotated.size());
    for (float x : rotated) rotated_norm += double(x) * x;
    EXPECT_NEAR(norm, rotated_norm, 1e-6);
    float error, norm2, offset, scale, projection, projection_error, rn, cn;
    double md, mean_norm2 = 0, exact = 1;
    ASSERT_TRUE(SQ8Center::encode_shifted(c.data(), mean.data(), 768,
                                          codes.data(), error, norm2, offset,
                                          scale, projection, projection_error));
    IntegerCenter::annotate(c.data(), mean.data(), codes.data(), 768, offset,
                            scale, error, rn, cn, md);
    IntegerCenter::Query query;
    ASSERT_TRUE(IntegerCenter::prepare(q.data(), mean.data(), 768, query));
    for (size_t i = 0; i < q.size(); ++i) {
      mean_norm2 += double(mean[i]) * mean[i];
      exact -= double(q[i]) * c[i];
    }
    float estimate = IntegerCenter::distance(query, codes.data(), 768, offset,
                                             scale, md, mean_norm2);
    EXPECT_LE(IntegerCenter::lower_bound(estimate, query, error, rn, cn, norm2,
                                         md, mean_norm2, 768),
              exact);
  }
}
TEST(QuantizedGraphPerformance, SnapshotSearchAndMutationGuard) {
  if (!HnswQuantizedGraph::supported()) {
    GTEST_SKIP();
  }
  TemporaryFiles tmp;
  const auto index_path = (tmp.dir / "index").string();
  const auto model_path = (tmp.dir / "model").string();
  constexpr size_t count = 256;
  std::mt19937 rng(123);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> vectors(count, std::vector<float>(768));
  for (auto &v : vectors)
    for (float &x : v) x = normal(rng);
  ci::HNSWIndexParam p(ci::MetricType::kCosine, 768, 16, 100);
  p.data_type = ci::DataType::DT_FP32;
  p.use_id_map = true;
  ci::StorageOptions options;
  options.type = ci::StorageOptions::StorageType::kMMAP;
  options.create_new = true;
  TestHnswIndex ordinary;
  ASSERT_EQ(0, ordinary.init(p));
  ASSERT_EQ(0, ordinary.open(index_path, options));
  for (size_t i = 0; i < count; ++i)
    ASSERT_EQ(0, ordinary.add({ci::DenseVector{vectors[i].data()}}, i));
  QuantizedGraphRouterBuilder::Data training;
  training.docs = count;
  std::array<double, 768> mean{};
  for (size_t i = 0; i < count; ++i) {
    // Deliberately reverse training order: row offsets are not graph IDs.
    const uint32_t id = count - i - 1;
    const float *v = ordinary.stored_vector(id);
    ASSERT_NE(v, nullptr);
    training.ids.push_back(id);
    training.vectors.emplace_back(768);
    std::copy_n(v, 768, training.vectors.back().begin());
    double norm = 0;
    for (size_t d = 0; d < 768; ++d) norm += double(v[d]) * v[d];
    ASSERT_NEAR(std::sqrt(norm), 1., 1e-4) << "internal ID " << id;
    for (size_t d = 0; d < 768; ++d) mean[d] += v[d];
  }
  for (size_t d = 0; d < 768; ++d) training.mean[d] = float(mean[d] / count);
  ASSERT_EQ(0, ordinary.flush());
  ASSERT_EQ(0, ordinary.close());
  {
    const auto model = QuantizedGraphRouterBuilder::train(training, {});
    std::ofstream out(model_path, std::ios::binary);
    model.write(out);
    out.close();
    ASSERT_TRUE(out.good());
  }
  options.create_new = false;
  p.quantized_graph_model = model_path;
  TestHnswIndex optimized;
  ASSERT_EQ(0, optimized.init(p));
  ASSERT_EQ(0, optimized.open(index_path, options));
  auto qp = std::make_shared<ci::HNSWQueryParam>();
  qp->topk = 10;
  qp->ef_search = 800;
  for (size_t i : {0, 37, 101, 255}) {
    ci::SearchResult result;
    ASSERT_EQ(
        0, optimized.search({ci::DenseVector{vectors[i].data()}}, qp, &result));
    ASSERT_EQ(10u, result.doc_list_.size());
    EXPECT_EQ(i, result.doc_list_[0].key());
    for (const auto &doc : result.doc_list_) {
      double dot = 0, qnorm = 0, vnorm = 0;
      const auto &v = vectors.at(doc.key());
      for (size_t d = 0; d < v.size(); ++d) {
        dot += double(v[d]) * vectors[i][d];
        qnorm += double(vectors[i][d]) * vectors[i][d];
        vnorm += double(v[d]) * v[d];
      }
      EXPECT_NEAR(1 - dot / std::sqrt(qnorm * vnorm), doc.score(), 1e-5);
    }
  }
  EXPECT_NE(0, optimized.add({ci::DenseVector{vectors[0].data()}}, count));
  EXPECT_EQ(count, optimized.get_doc_count());
  ASSERT_EQ(0, optimized.close());
  WriteModel(model_path, count + 1);
  ASSERT_EQ(0, optimized.init(p));
  EXPECT_NE(0, optimized.open(index_path, options));
}

TEST(QuantizedGraphPerformance, GenericRoutingAndNoRouterSearch) {
  if (!HnswQuantizedGraph::supported()) {
    GTEST_SKIP();
  }
  for (bool cosine : {false, true}) {
    for (size_t dim : {size_t{33}, size_t{960}}) {
      TemporaryFiles tmp;
      const auto path = (tmp.dir / "l2-index").string();
      const auto model_path = (tmp.dir / "l2-router").string();
      constexpr size_t count = 128;
      std::mt19937 rng(72);
      std::uniform_real_distribution<float> dist(0.f, .2f);
      QuantizedGraphRouterBuilder::Data training;
      training.docs = count;
      training.cosine = cosine;
      training.mean.assign(dim, 0);
      for (size_t i = 0; i < count; ++i) {
        training.ids.push_back(i);
        training.vectors.emplace_back(dim);
        for (size_t d = 0; d < dim; ++d) {
          training.vectors.back()[d] = dist(rng);
        }
        double norm2 = 0;
        for (float v : training.vectors.back()) norm2 += double(v) * v;
        for (size_t d = 0; d < dim; ++d) {
          if (cosine) training.vectors.back()[d] /= std::sqrt(norm2);
          training.mean[d] += training.vectors.back()[d] / count;
        }
      }
      ci::HNSWIndexParam p(
          cosine ? ci::MetricType::kCosine : ci::MetricType::kL2sq, dim, 16,
          100);
      p.data_type = ci::DataType::DT_FP32;
      p.use_id_map = true;
      ci::StorageOptions storage;
      storage.type = ci::StorageOptions::StorageType::kMMAP;
      storage.create_new = true;
      TestHnswIndex source;
      ASSERT_EQ(source.init(p), 0);
      ASSERT_EQ(source.open(path, storage), 0);
      for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(source.add({ci::DenseVector{training.vectors[i].data()}}, i),
                  0);
      }
      ASSERT_EQ(source.flush(), 0);
      ASSERT_EQ(source.close(), 0);
      {
        const auto model = QuantizedGraphRouterBuilder::train(training, {});
        std::ofstream out(model_path, std::ios::binary);
        model.write(out);
      }
      storage.create_new = false;
      storage.read_only = true;
      p.use_quantized_graph = true;
      for (bool model : {false, true}) {
        p.quantized_graph_model = model ? model_path : "";
        TestHnswIndex index;
        ASSERT_EQ(index.init(p), 0);
        ASSERT_EQ(index.open(path, storage), 0);
        auto qp = std::make_shared<ci::HNSWQueryParam>();
        qp->topk = 10;
        qp->ef_search = 800;
        for (uint32_t beam : {0u, 1u, 2u}) {
          qp->quantized_graph_route_beam = beam;
          for (size_t i : {size_t{0}, size_t{37}, size_t{127}}) {
            ci::SearchResult result;
            ASSERT_EQ(
                index.search({ci::DenseVector{training.vectors[i].data()}}, qp,
                             &result),
                0);
            ASSERT_EQ(result.doc_list_.size(), 10u);
            EXPECT_EQ(result.doc_list_[0].key(), i);
            for (const auto &doc : result.doc_list_) {
              double exact = 0, dot = 0, qnorm = 0, xnorm = 0;
              for (size_t d = 0; d < dim; ++d) {
                const double diff = double(training.vectors[i][d]) -
                                    training.vectors.at(doc.key())[d];
                exact += diff * diff;
                const double q = training.vectors[i][d],
                             x = training.vectors.at(doc.key())[d];
                dot += q * x;
                qnorm += q * q;
                xnorm += x * x;
              }
              if (cosine) exact = 1 - dot / std::sqrt(qnorm * xnorm);
              EXPECT_NEAR(exact, doc.score(), 1e-5);
            }
          }
        }
        qp->quantized_graph_margin = std::numeric_limits<float>::quiet_NaN();
        ci::SearchResult invalid;
        EXPECT_NE(index.search({ci::DenseVector{training.vectors[0].data()}},
                               qp, &invalid),
                  0);
        qp->quantized_graph_margin = -1;
        qp->quantized_graph_route_beam = 5;
        EXPECT_NE(index.search({ci::DenseVector{training.vectors[0].data()}},
                               qp, &invalid),
                  0);
        ASSERT_EQ(index.close(), 0);
      }
    }
  }
}

TEST(QuantizedGraphPerformance, L2ReconstructionLowerBound) {
  if (!HnswQuantizedGraph::supported()) {
    GTEST_SKIP();
  }
  std::mt19937 rng(93);
  std::uniform_real_distribution<float> random(-3.f, 4.f);
  for (size_t dim : {size_t{33}, size_t{960}, size_t{1024}}) {
    for (size_t trial = 0; trial < 64; ++trial) {
      std::vector<float> center(dim), query(dim);
      std::vector<uint8_t> codes(dim);
      double exact = 0;
      for (size_t d = 0; d < dim; ++d) {
        center[d] = random(rng);
        query[d] = trial % 2 ? center[d] + .001f * random(rng) : random(rng);
        const double diff = double(query[d]) - center[d];
        exact += diff * diff;
      }
      float error, norm, offset, scale;
      ASSERT_TRUE(SQ8Center::encode(center.data(), dim, codes.data(), error,
                                    norm, offset, scale));
      const float estimate =
          SQ8Center::distance(query.data(), codes.data(), dim, offset, scale);
      EXPECT_LE(SQ8Center::l2_lower_bound(estimate, error, dim), exact);
    }
  }
}
#endif
