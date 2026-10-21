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
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "flat/flat_distance_matrix.h"
#include "quantizer/distance_quantizer.h"
#include "quantizer/fp32_quantizer/fp32_quantizer.h"

using namespace zvec;
using namespace zvec::core;

namespace {
template <typename T>
void CheckMatrices(IndexMeta::DataType type, const std::string &metric_name) {
  // Unequal batch widths catch output transposition; dimensions also exercise
  // SIMD tails and Cosine's stored norm, which must not contribute to distance.
  constexpr size_t logical_dim = 19;
  // The FP32 norm occupies two FP16 elements or one FP32 element.
  const size_t norm_elements = type == IndexMeta::DT_FP16 ? 2 : 1;
  const size_t dim =
      logical_dim + (metric_name == "Cosine" ? norm_elements : 0);
  IndexMeta meta;
  meta.set_meta(type, dim);
  meta.set_metric(metric_name, 0, ailego::Params());
  auto quantizer = CreateDistanceQuantizer(meta);
  ASSERT_NE(nullptr, quantizer);
  EXPECT_EQ(logical_dim, static_cast<size_t>(quantizer->dim()));
  EXPECT_EQ(meta.element_size(),
            quantizer->quantized_datapoint_vector_length());
  for (size_t m : {1U, 4U, 32U}) {
    for (size_t n : {1U, 2U, 8U}) {
      SCOPED_TRACE(metric_name + " " + std::to_string(m) + "x" +
                   std::to_string(n));
      std::vector<T> rows(m * dim), queries(n * dim), a(m * dim), b(n * dim);
      std::vector<const void *> pointers(m);
      for (size_t i = 0; i < m; ++i) {
        pointers[i] = rows.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) {
          rows[i * dim + d] = static_cast<T>(
              d < logical_dim
                  ? (static_cast<int>((i * 7 + d) % 17) - 8) * 0.125f
                  : 100.0f + i);
          a[d * m + i] = rows[i * dim + d];
        }
      }
      for (size_t j = 0; j < n; ++j) {
        for (size_t d = 0; d < dim; ++d) {
          queries[j * dim + d] = static_cast<T>(
              d < logical_dim
                  ? (static_cast<int>((j * 3 + d * 2) % 13) - 6) * 0.125f
                  : 200.0f + j);
          b[d * n + j] = queries[j * dim + d];
        }
      }
      std::vector<float> result(m * n), batch(m);
      QuantizedFlatMatrixDistance<T>(*quantizer, a.data(), b.data(), m, n,
                                     result.data());
      if (m == 32 && n == 8) {
        FlatDistanceMatrix<32> matrix;
        matrix.initialize(quantizer);
        ASSERT_TRUE(matrix.is_valid());
        matrix.template distance<32, 8>(a.data(), b.data(), dim, result.data());
      }
      for (size_t j = 0; j < n; ++j) {
        quantizer->calc_distance_dp_query_batch(
            pointers.data(), m, queries.data() + j * dim, batch.data());
        for (size_t i = 0; i < m; ++i) {
          float expected = 0;
          for (size_t d = 0; d < logical_dim; ++d) {
            float x = rows[i * dim + d], y = queries[j * dim + d];
            expected +=
                metric_name == "SquaredEuclidean" ? (x - y) * (x - y) : -x * y;
          }
          if (metric_name == "Cosine") ++expected;
          EXPECT_NEAR(expected, result[j * m + i], 1e-4f);
          EXPECT_NEAR(expected, batch[i], 1e-4f);
        }
      }
    }
  }
}
}  // namespace

TEST(FlatQuantizerDistance, Fp32Matrices) {
  for (const auto *name : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    CheckMatrices<float>(IndexMeta::DT_FP32, name);
  }
}
TEST(FlatQuantizerDistance, Fp16Matrices) {
  for (const auto *name : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    CheckMatrices<ailego::Float16>(IndexMeta::DT_FP16, name);
  }
}
TEST(FlatQuantizerDistance, UnsupportedRepresentationRetainsLegacyPath) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_INT8, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  EXPECT_EQ(nullptr, CreateDistanceQuantizer(meta));
  meta.set_meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("MipsSquaredEuclidean", 0, ailego::Params());
  EXPECT_EQ(nullptr, CreateDistanceQuantizer(meta));
  meta.set_metric("Cosine", 0, ailego::Params());
  meta.set_meta(IndexMeta::DT_FP32, 1);
  EXPECT_EQ(nullptr, CreateDistanceQuantizer(meta));
}

TEST(FlatQuantizerDistance, CosineExtraMetadataUsesLogicalDimension) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, 3);
  meta.set_extra_meta_size(sizeof(float));
  meta.set_metric("Cosine", 0, ailego::Params());
  auto quantizer = CreateDistanceQuantizer(meta);
  ASSERT_NE(nullptr, quantizer);
  EXPECT_EQ(3, quantizer->dim());
  // Centers are averages of unit vectors: preserve their length, and ignore
  // the stored norm instead of normalizing the average a second time.
  const float center[] = {0.25f, 0.0f, 0.5f, 100.0f};
  const float query[] = {0.0f, 0.0f, 1.0f, 200.0f};
  EXPECT_FLOAT_EQ(0.5f, quantizer->calc_distance_dp_dp(center, query));
  // Column blocks include the metadata tail even though meta.dimension()
  // counts only the three logical components.
  const float columns[] = {0.25f, 0.5f,  0.0f,   0.0f,
                           0.5f,  0.25f, 100.0f, 300.0f};
  float scores[2];
  QuantizedFlatMatrixDistance<float>(*quantizer, columns, query, 2, 1, scores);
  EXPECT_FLOAT_EQ(0.5f, scores[0]);
  EXPECT_FLOAT_EQ(0.75f, scores[1]);
}

TEST(FlatQuantizerDistance, RawFp16L2AccumulatesInFp32) {
  constexpr size_t dim = 129;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP16, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto quantizer = CreateDistanceQuantizer(meta);
  ASSERT_NE(nullptr, quantizer);
  std::vector<ailego::Float16> a(dim, static_cast<ailego::Float16>(1000.0f));
  std::vector<ailego::Float16> b(dim, static_cast<ailego::Float16>(-1000.0f));
  const float expected = dim * 4000000.0f;
  EXPECT_FLOAT_EQ(expected, quantizer->calc_distance_dp_dp(a.data(), b.data()));
  const void *candidates[] = {a.data(), b.data()};
  float scores[2];
  quantizer->calc_distance_dp_query_batch(candidates, 2, b.data(), scores);
  EXPECT_FLOAT_EQ(expected, scores[0]);
  EXPECT_FLOAT_EQ(0.0f, scores[1]);
}

namespace {
template <IndexMeta::DataType Type, typename T>
void CheckEncodedSearcher(const char *metric_name,
                          IndexMeta::MajorOrder order) {
  SCOPED_TRACE(std::string(metric_name) + " type=" + std::to_string(Type) +
               " order=" + std::to_string(order));
  const bool cosine = std::string(metric_name) == "Cosine";
  // The FP32 norm occupies two FP16 elements or one FP32 element.
  constexpr size_t norm_elements = Type == IndexMeta::DT_FP16 ? 2 : 1;
  const size_t dim = 17 + (cosine ? norm_elements : 0);
  const std::string path = "flat_encoded_centers.index";
  struct Cleanup {
    std::string path;
    ~Cleanup() {
      ailego::File::RemovePath(path);
      ailego::File::RemovePath(path + ".reference");
    }
  } cleanup{path};
  ailego::File::RemovePath(path);
  IndexMeta meta;
  meta.set_meta(Type, dim);
  meta.set_metric(metric_name, 0, ailego::Params());
  meta.set_major_order(order);
  auto holder = std::make_shared<MultiPassIndexHolder<Type>>(dim);
  std::vector<T> queries;
  for (size_t i = 0; i < 41; ++i) {
    ailego::NumericalVector<T> row(dim);
    for (size_t d = 0; d < dim; ++d) {
      row[d] = static_cast<T>((i + 1) * 0.03125f + d * 0.015625f);
    }
    if (i == 2 || i == 17 || i == 34) {
      // Offset from the grid so L2 neighbors have distinct distances. Heap
      // and buffered top-k need not select/order equal-distance IDs alike.
      for (size_t d = 0; d < dim; ++d) {
        queries.push_back(static_cast<T>(row[d] + 0.0078125f));
      }
    }
    ASSERT_TRUE(holder->emplace(i, row));
  }
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, builder);
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->open(path, false));
  // Use the legacy row path as the reference: its Cosine implementation does
  // not provide every matrix width needed to load a column-major index.
  auto row_meta = meta;
  row_meta.set_major_order(IndexMeta::MO_ROW);
  auto row_builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_EQ(0, row_builder->init(row_meta, ailego::Params()));
  ASSERT_EQ(0, dumper->create(path + ".reference"));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(row_builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
  auto row_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, row_storage->open(path + ".reference", false));
  auto legacy = IndexFactory::CreateSearcher("FlatSearcher");
  auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
  ASSERT_NE(nullptr, legacy);
  ASSERT_NE(nullptr, searcher);
  ASSERT_EQ(0, legacy->init(ailego::Params()));
  ASSERT_FALSE(searcher->owns_query_quantization());
  ASSERT_EQ(0, legacy->load(row_storage, nullptr));
  for (size_t reopen = 0; reopen < 2; ++reopen) {
    auto quantizer = CreateDistanceQuantizer(meta);
    ASSERT_NE(nullptr, quantizer);
    std::weak_ptr<turbo::Quantizer> weak_quantizer = quantizer;
    ASSERT_EQ(0, searcher->init(ailego::Params(), quantizer));
    quantizer.reset();  // The searcher owns the distance provider's lifetime.
    EXPECT_FALSE(weak_quantizer.expired());
    ASSERT_EQ(0, searcher->load(storage, nullptr));
    auto reference = legacy->create_context();
    auto actual = searcher->create_context();
    reference->set_topk(6);
    actual->set_topk(6);
    IndexQueryMeta query_meta(Type, dim);
    for (uint32_t count : {1U, 3U}) {
      ASSERT_EQ(
          0, legacy->search_impl(queries.data(), query_meta, count, reference));
      ASSERT_EQ(
          0, searcher->search_impl(queries.data(), query_meta, count, actual));
      for (size_t j = 0; j < count; ++j) {
        const auto &expected = reference->result(j);
        const auto &result = actual->result(j);
        ASSERT_EQ(6U, result.size());
        ASSERT_EQ(expected.size(), result.size());
        for (size_t k = 0; k < result.size(); ++k) {
          EXPECT_EQ(expected[k].key(), result[k].key());
          // FP16 kernels can use a different accumulation order/precision.
          const float tolerance =
              Type == IndexMeta::DT_FP16
                  ? 1e-3f * std::max(1.0f, std::abs(expected[k].score()))
                  : 1e-4f;
          EXPECT_NEAR(expected[k].score(), result[k].score(), tolerance);
        }
      }
    }
    actual.reset();
    ASSERT_EQ(0, reopen == 0 ? searcher->unload() : searcher->cleanup());
    EXPECT_TRUE(weak_quantizer.expired());
  }
  IndexMeta wrong = meta;
  wrong.set_meta(Type, dim + 1);
  ASSERT_EQ(0,
            searcher->init(ailego::Params(), CreateDistanceQuantizer(wrong)));
  EXPECT_EQ(IndexError_Mismatch, searcher->load(storage, nullptr));
}
}  // namespace

TEST(FlatQuantizerDistance, EncodedFp32SearcherMatchesLegacyAndReopens) {
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    for (const auto *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
      CheckEncodedSearcher<IndexMeta::DT_FP32, float>(metric, order);
    }
  }
}

TEST(FlatQuantizerDistance, EncodedFp16SearcherMatchesLegacyAndReopens) {
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    for (const auto *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
      CheckEncodedSearcher<IndexMeta::DT_FP16, ailego::Float16>(metric, order);
    }
  }
}

namespace {
class CountingFp32Quantizer : public turbo::Fp32Quantizer {
 public:
  float calc_distance_dp_query(const void *dp,
                               const void *query) const override {
    if (!inside_strided_) ++single_calls;
    return Fp32Quantizer::calc_distance_dp_query(dp, query);
  }

  void calc_distance_dp_query_batch(const void *const *dp_list, int dp_num,
                                    const void *query,
                                    float *dist_list) const override {
    if (!inside_strided_) ++gather_calls;
    Fp32Quantizer::calc_distance_dp_query_batch(dp_list, dp_num, query,
                                                dist_list);
  }

  void calc_distance_dp_query_batch_strided(const void *base,
                                            size_t stride_bytes, int dp_num,
                                            const void *query,
                                            float *dist_list) const override {
    EXPECT_EQ(quantized_datapoint_vector_length(), stride_bytes);
    EXPECT_GT(dp_num, 0);
    EXPECT_LE(dp_num, 512);
    batches.push_back(dp_num);
    // A platform without a native strided kernel may gather inside this
    // method. Only count calls made directly by the Flat scanner.
    inside_strided_ = true;
    Fp32Quantizer::calc_distance_dp_query_batch_strided(
        base, stride_bytes, dp_num, query, dist_list);
    inside_strided_ = false;
  }

  void reset_counts() const {
    single_calls = 0;
    gather_calls = 0;
    batches.clear();
  }

  mutable size_t single_calls{0};
  mutable size_t gather_calls{0};
  mutable std::vector<int> batches;

 private:
  mutable bool inside_strided_{false};
};

struct FlatIndexFile {
  explicit FlatIndexFile(std::string value) : path(std::move(value)) {
    ailego::File::RemovePath(path);
  }
  ~FlatIndexFile() {
    ailego::File::RemovePath(path);
  }
  std::string path;
};

void BuildFp32Rows(const std::string &path, const IndexMeta &meta,
                   const std::vector<float> &rows,
                   const std::vector<uint64_t> &keys) {
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(
      meta.dimension());
  for (size_t i = 0; i < keys.size(); ++i) {
    ailego::NumericalVector<float> row(meta.dimension());
    std::copy_n(rows.data() + i * meta.dimension(), meta.dimension(),
                row.data());
    ASSERT_TRUE(holder->emplace(keys[i], row));
  }
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, builder);
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
}

void CheckStridedRows(const char *metric_name) {
  SCOPED_TRACE(metric_name);
  constexpr size_t logical_dim = 17;
  constexpr size_t row_count = 2203;
  constexpr size_t query_count = 3;
  const bool cosine = std::string(metric_name) == "Cosine";
  const bool l2 = std::string(metric_name) == "SquaredEuclidean";
  const size_t dim = logical_dim + (cosine ? 1 : 0);
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  meta.set_metric(metric_name, 0, ailego::Params());
  meta.set_major_order(IndexMeta::MO_ROW);
  FlatIndexFile file(std::string("flat_strided_rows_") + metric_name);
  std::vector<float> rows(row_count * dim), queries(query_count * dim);
  std::vector<uint64_t> keys(row_count);
  for (size_t i = 0; i < row_count; ++i) {
    keys[i] = 1000000 - i * 37;
    for (size_t d = 0; d < logical_dim; ++d) {
      rows[i * dim + d] = d == 0 ? (i + 1) / 4096.0f : (d + 1) / 128.0f;
    }
    if (cosine) rows[i * dim + logical_dim] = 1000.0f + i;
  }
  for (size_t j = 0; j < query_count; ++j) {
    for (size_t d = 0; d < logical_dim; ++d) {
      queries[j * dim + d] =
          d == 0 ? (l2 ? -1.0f : 1.0f) * (j + 1) / 8.0f : (d + j + 1) / 256.0f;
    }
    if (cosine) queries[j * dim + logical_dim] = 100000.0f + j;
  }
  BuildFp32Rows(file.path, meta, rows, keys);
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->open(file.path, false));
  IndexMeta input(IndexMeta::DT_FP32, logical_dim);
  input.set_metric(metric_name, 0, ailego::Params());
  auto quantizer = std::make_shared<CountingFp32Quantizer>();
  ASSERT_EQ(0, quantizer->init(input, ailego::Params()));
  ailego::Params params;
  // Two 1,056-row reads followed by 91 rows. Each complete read contains
  // two full 512-row batches and a 32-row tail.
  params.set("proxima.flat.read_block_size",
             static_cast<uint32_t>(1056 * dim * sizeof(float)));
  for (bool use_quantizer : {true, false}) {
    SCOPED_TRACE(use_quantizer);
    auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
    ASSERT_NE(nullptr, searcher);
    ASSERT_EQ(0, use_quantizer ? searcher->init(params, quantizer)
                               : searcher->init(params));
    ASSERT_EQ(0, searcher->load(storage, nullptr));
    auto context = searcher->create_context();
    ASSERT_NE(nullptr, context);
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
    for (uint32_t topk : {0U, 17U, 129U, 2210U}) {
      SCOPED_TRACE(topk);
      context->set_topk(topk);
      for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(mode);
        quantizer->reset_counts();
        const size_t count = mode == 2 ? query_count : 1;
        ASSERT_EQ(0, mode == 0
                         ? searcher->search_impl(queries.data(), qmeta, context)
                         : searcher->search_impl(queries.data(), qmeta, count,
                                                 context));
        if (use_quantizer) {
          EXPECT_EQ(0U, quantizer->single_calls);
          EXPECT_EQ(0U, quantizer->gather_calls);
          size_t scanned_rows = 0;
          for (int batch : quantizer->batches) scanned_rows += batch;
          EXPECT_EQ(row_count * count, scanned_rows);
          EXPECT_NE(quantizer->batches.end(),
                    std::find(quantizer->batches.begin(),
                              quantizer->batches.end(), 512));
          EXPECT_NE(quantizer->batches.end(),
                    std::find(quantizer->batches.begin(),
                              quantizer->batches.end(), 91));
        }
        for (size_t j = 0; j < count; ++j) {
          std::vector<std::pair<float, uint64_t>> expected;
          for (size_t i = 0; i < row_count; ++i) {
            double score = cosine ? 1.0 : 0.0;
            for (size_t d = 0; d < logical_dim; ++d) {
              const double x = rows[i * dim + d];
              const double y = queries[j * dim + d];
              score += l2 ? (x - y) * (x - y) : -x * y;
            }
            expected.emplace_back(static_cast<float>(score), keys[i]);
          }
          std::sort(expected.begin(), expected.end());
          expected.resize(std::min(row_count, size_t(std::max(topk, 1U))));
          const auto &result = context->result(j);
          ASSERT_EQ(expected.size(), result.size());
          for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(expected[i].second, result[i].key());
            EXPECT_NEAR(expected[i].first, result[i].score(), 1e-5f);
          }
        }
      }
    }
  }
}
}  // namespace

TEST(FlatQuantizerDistance, StridedRowsAcrossReadsMatchScalarReference) {
  for (const auto *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    CheckStridedRows(metric);
  }
}

TEST(FlatQuantizerDistance, RowTopKThresholdIncludesBoundaryAndRejectsNaN) {
  IndexMeta meta(IndexMeta::DT_FP32, 1);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  meta.set_major_order(IndexMeta::MO_ROW);
  const std::vector<float> rows{1, 2, 3,
                                4, 5, std::numeric_limits<float>::quiet_NaN()};
  const std::vector<uint64_t> keys{107, 53, 199, 13, 71, 997};
  const float queries[] = {0, -1};
  const IndexQueryMeta qmeta(IndexMeta::DT_FP32, 1);
  FlatIndexFile file("flat_strided_threshold.index");
  BuildFp32Rows(file.path, meta, rows, keys);
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->open(file.path, false));
  for (bool use_quantizer : {true, false}) {
    SCOPED_TRACE(use_quantizer);
    auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
    ASSERT_NE(nullptr, searcher);
    ASSERT_EQ(0, use_quantizer ? searcher->init(ailego::Params(),
                                                CreateDistanceQuantizer(meta))
                               : searcher->init(ailego::Params()));
    ASSERT_EQ(0, searcher->load(storage, nullptr));
    auto context = searcher->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(20);
    for (float threshold : {9.0f, 0.25f}) {
      context->set_threshold(threshold);
      ASSERT_EQ(0, searcher->search_impl(queries, qmeta, 2, context));
      for (size_t j = 0; j < 2; ++j) {
        const auto &result = context->result(j);
        const size_t count = threshold == 9.0f ? 3 - j : 0;
        ASSERT_EQ(count, result.size());
        for (size_t i = 0; i < count; ++i) {
          EXPECT_EQ(keys[i], result[i].key());
          EXPECT_FLOAT_EQ((rows[i] - queries[j]) * (rows[i] - queries[j]),
                          result[i].score());
        }
      }
    }
    context->reset_threshold();
    ASSERT_EQ(0, searcher->search_impl(queries, qmeta, context));
    ASSERT_EQ(5U, context->result().size());
    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(keys[i], context->result()[i].key());
    }
  }
}
