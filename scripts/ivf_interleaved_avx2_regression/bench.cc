// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_framework.h>
#include "metric/metric_params.h"
#include "ivf_centroid_index.h"
#include "ivf_dumper.h"
#include "ivf_entity.h"
#include "ivf_searcher.h"

namespace zvec::core {
namespace {

// Use the legacy writer explicitly: no Turbo quantizer descriptor or segment.
// The counts exercise full interleaved blocks and row-major tails in the same
// persisted inverted list; reference scores are computed from stored values.
class IVFLegacyMatrixTest
    : public testing::TestWithParam<std::tuple<IndexMeta::DataType, bool>> {
 protected:
  using Bytes = std::vector<uint8_t>;
  bool record_quantized_{false};
  void validate();
  IndexMeta meta_;
  std::vector<Bytes> rows_, queries_;
  std::vector<std::vector<double>> values_, query_values_;
  const std::string path_{"ivf_legacy_matrix_validation.index"};

  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_ERROR);
  }
  void TearDown() override {
    ailego::File::RemovePath(path_);
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);
  }

  void make_vectors(size_t count, size_t dim, std::mt19937 &rng,
                    std::vector<Bytes> *encoded,
                    std::vector<std::vector<double>> *decoded) {
    encoded->assign(count, Bytes(meta_.element_size(), 0));
    decoded->assign(count, std::vector<double>(dim));
    for (size_t i = 0; i < count; ++i) {
      for (size_t d = 0; d < dim; ++d) {
        const int raw = int(rng() % 256) - 128;
        auto &out = (*encoded)[i];
        double value = 0;
        switch (meta_.data_type()) {
          case IndexMeta::DT_FP32: {
            const float v = float(raw) / 37.0f;
            std::memcpy(out.data() + d * sizeof(v), &v, sizeof(v));
            value = v;
            break;
          }
          case IndexMeta::DT_FP16: {
            const ailego::Float16 v(float(raw) / 37.0f);
            std::memcpy(out.data() + d * sizeof(v), &v, sizeof(v));
            value = float(v);
            break;
          }
          case IndexMeta::DT_INT8:
            out[d] = static_cast<uint8_t>(raw);
            value = raw;
            break;
          case IndexMeta::DT_INT4: {
            const int v = int(rng() % 16) - 8;
            out[d / 2] |= (v & 15) << (4 * (d % 2));
            value = v;
            break;
          }
          default:
            FAIL() << "Unsupported test type";
        }
        (*decoded)[i][d] = value;
      }
      if (record_quantized_) {
        float sum = 0, sum2 = 0;
        const float scale = 0.25f * (1 + i % 3);
        const float bias = 0.5f * (int(i % 5) - 2);
        for (double &v : (*decoded)[i]) {
          sum += v;
          sum2 += v * v;
          v = scale * v + bias;
        }
        const size_t offset =
            meta_.data_type() == IndexMeta::DT_INT4 ? dim / 2 : dim;
        const float tail[] = {scale, bias, sum, sum2};
        std::memcpy((*encoded)[i].data() + offset, tail, sizeof(tail));
        if (meta_.data_type() == IndexMeta::DT_INT8) {
          const int32_t code_sum = int32_t(sum);
          std::memcpy((*encoded)[i].data() + offset + 16, &code_sum, 4);
        } else if (std::get<1>(GetParam())) {
          const int32_t code_sum = int32_t(sum);
          std::memcpy((*encoded)[i].data() + offset + 12, &code_sum, 4);
        }
      }
    }
  }

  void write_index(size_t dim, const std::vector<size_t> &list_counts) {
    const auto type = std::get<0>(GetParam());
    const bool ip = std::get<1>(GetParam());
    meta_.set_meta(
        type,
        dim + (record_quantized_ ? (type == IndexMeta::DT_INT4 ? 32 : 20) : 0));
    ailego::Params metric_params;
    const char *origin = ip ? "InnerProduct" : "SquaredEuclidean";
    if (record_quantized_)
      metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME, origin);
    meta_.set_metric(record_quantized_ ? "QuantizedInteger" : origin, 0,
                     metric_params);
    meta_.set_major_order(IndexMeta::MO_COLUMN);
    size_t count = 0;
    for (size_t n : list_counts) count += n;
    std::mt19937 rng(20261105);
    make_vectors(count, dim, rng, &rows_, &values_);
    make_vectors(3, dim, rng, &queries_, &query_values_);
    queries_[0] = rows_.front();
    query_values_[0] = values_.front();
    queries_[1] = rows_.back();
    query_values_[1] = values_.back();

    IndexCluster::CentroidList centroids;
    for (size_t i = 0; i < list_counts.size(); ++i) {
      IndexCluster::Centroid centroid;
      centroid.set_feature(rows_[i % rows_.size()].data(),
                           meta_.element_size());
      centroids.emplace_back(std::move(centroid));
    }
    IVFCentroidIndex centroid_index;
    ASSERT_EQ(0, centroid_index.init(meta_, ailego::Params()));
    ASSERT_EQ(0, centroid_index.build(centroids));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    IVFDumper postings(meta_, dumper, list_counts.size());
    size_t id = 0;
    for (size_t list = 0; list < list_counts.size(); ++list) {
      for (size_t i = 0; i < list_counts[list]; ++i, ++id) {
        ASSERT_EQ(0, postings.dump_inverted_vector(list, id, rows_[id].data()));
      }
    }
    ASSERT_EQ(0, postings.dump_inverted_vector_finished());
    ASSERT_EQ(0, postings.dump_centroid_index(centroid_index.data(),
                                              centroid_index.size()));
    ASSERT_EQ(0, IndexHelper::SerializeToDumper(meta_, dumper.get()));
    ASSERT_EQ(0, dumper->close());
  }

  double reference(size_t q, size_t id) const {
    double sum = 0;
    for (size_t d = 0; d < values_[id].size(); ++d) {
      const double a = values_[id][d], b = query_values_[q][d];
      sum += std::get<1>(GetParam()) ? -a * b : (a - b) * (a - b);
    }
    return sum;
  }

  static bool excluded(uint64_t id, int filter) {
    return filter == 1 ? id % 3 == 0 : filter == 2 && id < 32;
  }

  void check_results(const IndexDocumentList &actual, size_t q, size_t topk,
                     int filter) {
    std::vector<double> expected;
    for (size_t i = 0; i < rows_.size(); ++i)
      if (!excluded(i, filter)) expected.push_back(reference(q, i));
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(std::min(topk, expected.size()), actual.size());
    std::set<uint64_t> seen;
    // Bound cancellation error by the sum of absolute products as well as
    // the result magnitude, rather than assuming a relative IP error bound.
    for (size_t rank = 0; rank < actual.size(); ++rank) {
      const auto id = actual[rank].key();
      ASSERT_LT(id, rows_.size());
      EXPECT_FALSE(excluded(id, filter));
      EXPECT_TRUE(seen.insert(id).second);
      double scale = 1;
      for (size_t d = 0; d < values_[id].size(); ++d)
        scale += std::abs(values_[id][d] * query_values_[q][d]);
      const double tolerance =
          2e-5 * std::max(scale, std::abs(reference(q, id)));
      EXPECT_NEAR(reference(q, id), actual[rank].score(), tolerance);
      EXPECT_NEAR(expected[rank], actual[rank].score(), tolerance);
    }
  }

  IndexStorage::Pointer open_storage(const char *name) {
    auto storage = IndexFactory::CreateStorage(name);
    if (!storage || storage->init(ailego::Params()) ||
        storage->open(path_, false))
      return nullptr;
    return storage;
  }

  void init_searcher(const IndexStorage::Pointer &storage,
                     IVFSearcher *searcher) {
    ASSERT_FALSE(storage->has(IVF_TURBO_QUANTIZER_SEG_ID));
    IVFEntity entity;
    ASSERT_EQ(0, entity.load(storage));
    ASSERT_EQ(IndexMeta::MO_COLUMN, entity.meta().major_order());
    ASSERT_EQ(32U, entity.block_vector_count());
    ASSERT_EQ(nullptr, entity.quantizer());
    ailego::Params params;
    params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
    params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
    ASSERT_EQ(0, searcher->init(params));
    ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
    ASSERT_EQ(nullptr, searcher->quantizer());
  }
};

TEST_P(IVFLegacyMatrixTest, ManyBenchmark) {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const bool prepare = std::getenv("PREPARE") != nullptr;
  const auto type = std::get<0>(GetParam());
  const bool integer = type == IndexMeta::DT_INT8 || type == IndexMeta::DT_INT4;
  const char *label = type == IndexMeta::DT_FP32   ? "FP32"
                      : type == IndexMeta::DT_FP16 ? "FP16"
                      : type == IndexMeta::DT_INT8 ? "INT8"
                                                   : "INT4";
  const char *metric = std::get<1>(GetParam()) ? "IP" : "L2";
  std::ofstream csv("timings.csv", std::ios::app),
      output("query_results.txt", std::ios::app);
  csv << std::setprecision(12);
  output << std::setprecision(9);
  for (bool record : {false, true}) {
    if (record && !integer) continue;
    record_quantized_ = record;
    for (uint32_t dim : {128U, 768U}) {
      const std::string name = std::string(record ? "Record-" : "Raw-") + label;
      const std::string path = "../indexes/" + name + "_" + metric + "_" +
                               std::to_string(dim) + ".index";
      if (prepare) {
        std::vector<size_t> counts(32, 256);
        counts[0] = 255;
        counts[1] = 257;
        ASSERT_NO_FATAL_FAILURE(write_index(dim, counts));
        ASSERT_EQ(0, std::rename(path_.c_str(), path.c_str()));
        continue;
      }
      meta_.set_meta(
          type, dim + (record ? (type == IndexMeta::DT_INT4 ? 32 : 20) : 0));
      ailego::Params mp;
      const char *origin =
          std::get<1>(GetParam()) ? "InnerProduct" : "SquaredEuclidean";
      if (record) mp.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME, origin);
      meta_.set_metric(record ? "QuantizedInteger" : origin, 0, mp);
      meta_.set_major_order(IndexMeta::MO_COLUMN);
      std::mt19937 random(20261112);
      make_vectors(32, dim, random, &queries_, &query_values_);
      const IndexQueryMeta qm(type, meta_.dimension());
      auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
      ASSERT_EQ(0, storage->init(ailego::Params()));
      ASSERT_EQ(0, storage->open(path, false));
      IVFSearcher searcher;
      ailego::Params params;
      params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0U);
      ASSERT_EQ(0, searcher.init(params));
      ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
      auto ctx = searcher.create_context();
      ASSERT_NE(nullptr, ctx);
      ctx->set_topk(10);
      for (bool shared : {false, true}) {
        Bytes batch;
        for (size_t q = 0; q < 32; ++q) {
          const auto &v = queries_[shared ? 0 : q];
          batch.insert(batch.end(), v.begin(), v.end());
        }
        for (uint32_t nprobe : {1U, 8U})
          for (uint32_t count : {1U, 8U, 32U}) {
            params.set(PARAM_IVF_SEARCHER_NPROBE, nprobe);
            ASSERT_EQ(0, ctx->update(params));
            const std::string key =
                name + "," + metric + "," + std::to_string(dim) + "," +
                (shared ? "shared" : "spread") + "," + std::to_string(nprobe) +
                "," + std::to_string(count);
            if (std::getenv("BENCH_CASE") && key != std::getenv("BENCH_CASE"))
              continue;
            for (int warm = 0; warm < 5; ++warm)
              ASSERT_EQ(0, searcher.search_impl(batch.data(), qm, count, ctx));
            for (uint32_t q = 0; q < count; ++q)
              for (const auto &doc : ctx->result(q))
                output << key << "," << q << "," << doc.key() << ","
                       << doc.score() << "\n";
            std::vector<double> samples;
            for (int sample = 0; sample < 5; ++sample) {
              const auto start = std::chrono::steady_clock::now();
              size_t iterations = 0;
              double elapsed;
              do {
                ASSERT_EQ(0,
                          searcher.search_impl(batch.data(), qm, count, ctx));
                ++iterations;
                elapsed = std::chrono::duration<double, std::nano>(
                              std::chrono::steady_clock::now() - start)
                              .count();
              } while (elapsed < (std::getenv("BENCH_MIN_MS")
                                      ? std::atof(std::getenv("BENCH_MIN_MS"))
                                      : 20.0) *
                                     1000000.0);
              samples.push_back(elapsed / iterations / count);
            }
            std::sort(samples.begin(), samples.end());
            csv << key << "," << samples[2] << ","
                << static_cast<IVFSearcherContext *>(ctx.get())
                       ->stats()
                       .dist_calced_count()
                << "\n";
            csv.flush();
          }
      }
    }
  }
}
INSTANTIATE_TEST_SUITE_P(
    RawTypes, IVFLegacyMatrixTest,
    testing::Combine(testing::Values(IndexMeta::DT_FP32, IndexMeta::DT_FP16,
                                     IndexMeta::DT_INT8, IndexMeta::DT_INT4),
                     testing::Bool()));
}  // namespace
}  // namespace zvec::core
