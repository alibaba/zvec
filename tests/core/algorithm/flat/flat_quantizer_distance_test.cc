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
#include <cmath>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "flat/flat_distance_matrix.h"
#include "quantizer/distance_quantizer.h"

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
      queries.insert(queries.end(), row.data(), row.data() + dim);
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
