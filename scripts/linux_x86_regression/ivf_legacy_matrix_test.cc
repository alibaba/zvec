// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  void Validate();
  void Benchmark();
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

  void write_index(size_t dim, const std::vector<size_t> &list_counts,
                   size_t query_count = 3) {
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
    make_vectors(query_count, dim, rng, &queries_, &query_values_);
    if (query_count == 3) {
      queries_[0] = rows_.front();
      query_values_[0] = values_.front();
      queries_[1] = rows_.back();
      query_values_[1] = values_.back();
    }

    std::string fixture_name = std::string(record_quantized_ ? "record_" : "") +
                               std::to_string(int(type)) + "_" +
                               (ip ? "ip_" : "l2_") + std::to_string(dim);
    for (size_t n : list_counts) fixture_name += "_" + std::to_string(n);
    fixture_name += ".index";
    // Cross-version verification: the parent executable writes these files,
    // then the current executable reads them against the same seeded oracle.
    if (const char *directory = std::getenv("ZVEC_LEGACY_READ_FIXTURES")) {
      std::error_code error;
      std::filesystem::copy_file(
          std::filesystem::path(directory) / fixture_name, path_,
          std::filesystem::copy_options::overwrite_existing, error);
      ASSERT_FALSE(error) << error.message();
      return;
    }

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
    if (const char *directory = std::getenv("ZVEC_LEGACY_WRITE_FIXTURES")) {
      std::error_code error;
      std::filesystem::create_directories(directory, error);
      ASSERT_FALSE(error) << error.message();
      std::filesystem::copy_file(
          path_, std::filesystem::path(directory) / fixture_name,
          std::filesystem::copy_options::overwrite_existing, error);
      ASSERT_FALSE(error) << error.message();
    }
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

TEST_P(IVFLegacyMatrixTest, PersistedColumnBlocksMatchIndependentReference) {
  Validate();
}

TEST_P(IVFLegacyMatrixTest,
       PersistedRecordQuantizedBlocksMatchIndependentReference) {
  if (std::get<0>(GetParam()) != IndexMeta::DT_INT8 &&
      std::get<0>(GetParam()) != IndexMeta::DT_INT4)
    GTEST_SKIP();
  record_quantized_ = true;
  Validate();
}

void IVFLegacyMatrixTest::Validate() {
  const bool integer = std::get<0>(GetParam()) == IndexMeta::DT_INT8 ||
                       std::get<0>(GetParam()) == IndexMeta::DT_INT4;
  for (size_t dim : {8U, 17U, 128U, 512U}) {
    if (integer && dim == 17) continue;  // Legacy word-interleaved alignment.
    for (const auto &counts : std::vector<std::vector<size_t>>{
             {31}, {32}, {33}, {65}, {33, 0, 32, 1}}) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " lists="
                                      << testing::PrintToString(counts));
      ASSERT_NO_FATAL_FAILURE(write_index(dim, counts));
      Bytes batch;
      for (const auto &q : queries_)
        batch.insert(batch.end(), q.begin(), q.end());
      const IndexQueryMeta qmeta(meta_.data_type(), meta_.dimension());
      for (const char *storage_name :
           {"MMapFileReadStorage", "FileReadStorage", "BufferReadStorage"}) {
        SCOPED_TRACE(storage_name);
        auto storage = open_storage(storage_name);
        ASSERT_NE(nullptr, storage);
        IVFSearcher searcher;
        ASSERT_NO_FATAL_FAILURE(init_searcher(storage, &searcher));
        auto context = searcher.create_context();
        ASSERT_NE(nullptr, context);
        for (size_t topk : {size_t(7), rows_.size()}) {
          context->set_topk(topk);
          for (int filter : {0, 1, 2}) {
            SCOPED_TRACE(testing::Message()
                         << "topk=" << topk << " filter=" << filter);
            if (filter == 0) {
              context->reset_filter();
            } else {
              context->set_filter(
                  [filter](uint64_t id) { return excluded(id, filter); });
            }
            for (bool bf : {false, true}) {
              SCOPED_TRACE(bf);
              ASSERT_EQ(
                  0,
                  bf ? searcher.search_bf_impl(batch.data(), qmeta, 3, context)
                     : searcher.search_impl(batch.data(), qmeta, 3, context));
              for (size_t q = 0; q < queries_.size(); ++q)
                ASSERT_NO_FATAL_FAILURE(
                    check_results(context->result(q), q, topk, filter));
              if (filter == 0) {
                const auto *ivf_context =
                    dynamic_cast<const IVFSearcherContext *>(context.get());
                ASSERT_NE(nullptr, ivf_context);
                for (size_t q = 0; q < queries_.size(); ++q)
                  EXPECT_EQ(rows_.size(),
                            ivf_context->stats(q).dist_calced_count());
              }
              ASSERT_EQ(0, bf ? searcher.search_bf_impl(queries_[2].data(),
                                                        qmeta, context)
                              : searcher.search_impl(queries_[2].data(), qmeta,
                                                     context));
              ASSERT_NO_FATAL_FAILURE(
                  check_results(context->result(), 2, topk, filter));
            }
          }
        }
        ASSERT_EQ(0, searcher.unload());
      }
    }
  }
}

// Opt in for measurement. Use the identical source and persisted format with
// the parent and current implementations; time only warm search_impl calls.
TEST_P(IVFLegacyMatrixTest, DISABLED_WarmColumnScanBenchmark) {
  Benchmark();
}

TEST_P(IVFLegacyMatrixTest, DISABLED_WarmRecordColumnScanBenchmark) {
  if (std::get<0>(GetParam()) != IndexMeta::DT_INT8 &&
      std::get<0>(GetParam()) != IndexMeta::DT_INT4)
    GTEST_SKIP();
  record_quantized_ = true;
  Benchmark();
}

void IVFLegacyMatrixTest::Benchmark() {
  for (size_t dim : {128U, 512U}) {
    ASSERT_NO_FATAL_FAILURE(write_index(dim, {8193}, 16));
    auto storage = open_storage("MMapFileReadStorage");
    ASSERT_NE(nullptr, storage);
    IVFSearcher searcher;
    ASSERT_NO_FATAL_FAILURE(init_searcher(storage, &searcher));
    auto context = searcher.create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(10);
    const IndexQueryMeta qmeta(meta_.data_type(), meta_.dimension());
    for (const auto &query : queries_)
      ASSERT_EQ(0, searcher.search_impl(query.data(), qmeta, context));
    std::vector<double> timings;
    for (size_t round = 0; round < 7; ++round) {
      const auto start = std::chrono::steady_clock::now();
      for (const auto &query : queries_)
        ASSERT_EQ(0, searcher.search_impl(query.data(), qmeta, context));
      const auto end = std::chrono::steady_clock::now();
      timings.push_back(
          std::chrono::duration<double, std::micro>(end - start).count() /
          queries_.size());
    }
    std::sort(timings.begin(), timings.end());
    std::printf(
        "LEGACY_BENCH,type=%d,metric=%s,dim=%zu,count=%zu,record=%d,median_us=%"
        ".3f,min_"
        "us=%.3f,max_us=%.3f\n",
        int(meta_.data_type()),
        std::get<1>(GetParam()) ? "InnerProduct" : "SquaredEuclidean", dim,
        rows_.size(), int(record_quantized_), timings[3], timings.front(),
        timings.back());
    ASSERT_EQ(0, searcher.unload());
  }
}

INSTANTIATE_TEST_SUITE_P(
    RawTypes, IVFLegacyMatrixTest,
    testing::Combine(testing::Values(IndexMeta::DT_FP32, IndexMeta::DT_FP16,
                                     IndexMeta::DT_INT8, IndexMeta::DT_INT4),
                     testing::Bool()));

}  // namespace
}  // namespace zvec::core
