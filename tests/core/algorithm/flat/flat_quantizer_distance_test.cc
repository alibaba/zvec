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
#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_factory.h>
#include "flat/flat_distance_matrix.h"
#include "flat/flat_streamer.h"
#include "quantizer/distance_quantizer.h"
#include "quantizer/fp32_quantizer/fp32_quantizer.h"
#include "tests/test_util.h"
#include "utility/batch_distance_budget.h"

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

namespace {
class BudgetFp32Quantizer : public turbo::Fp32Quantizer {
 public:
  float calc_distance_dp_query(const void *row,
                               const void *query) const override {
    ++pairwise_calls;
    return Fp32Quantizer::calc_distance_dp_query(row, query);
  }
  void calc_distance_dp_query_batch(const void *const *rows, int count,
                                    const void *query,
                                    float *output) const override {
    EXPECT_GT(count, 0);
    EXPECT_LE(static_cast<size_t>(count),
              std::max(size_t{32}, BatchDistanceRows(meta())));
    batches.push_back(count);
    Fp32Quantizer::calc_distance_dp_query_batch(rows, count, query, output);
  }
  mutable std::vector<int> batches;
  mutable size_t pairwise_calls{0};
};

struct BudgetTestCleanup {
  explicit BudgetTestCleanup(std::string value)
      : path(std::move(value)),
        capacity(ailego::MemoryLimitPool::get_instance().capacity()) {
    test_util::RemoveTestPath(path);
  }
  ~BudgetTestCleanup() {
    test_util::RemoveTestPath(path);
    ailego::MemoryLimitPool::get_instance().init(capacity);
  }
  std::string path;
  size_t capacity;
};

bool ReadBudgetSegment(const IndexStorage::Segment::Pointer &segment,
                       size_t offset, void *output, size_t bytes) {
  IndexStorage::MemoryBlock data;
  if (!segment || segment->read(offset, data, bytes) != bytes || !data.data())
    return false;
  std::memcpy(output, data.data(), bytes);
  return true;
}

void CheckBudgetStreamer(size_t dim, size_t count, const char *storage_name,
                         bool contiguous) {
  SCOPED_TRACE(dim);
  SCOPED_TRACE(storage_name);
  SCOPED_TRACE(contiguous);
  BudgetTestCleanup cleanup("flat_byte_budget_test");
  // Writable storage also reserves writeback buffers. Reopen below with a
  // smaller read-only pool to test the query's bounded page lifetime.
  ailego::MemoryLimitPool::get_instance().init(64 * 1024 * 1024UL);
  auto storage = IndexFactory::CreateStorage(storage_name);
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(cleanup.path + "/index", true));
  auto quantizer = std::make_shared<BudgetFp32Quantizer>();
  IndexMeta raw_meta(IndexMeta::DT_FP32, dim);
  raw_meta.set_metric("Cosine", 0, ailego::Params());
  ASSERT_EQ(0, quantizer->init(raw_meta, ailego::Params()));
  IndexMeta meta = quantizer->meta();
  meta.set_quantizer("Fp32Quantizer", 0, ailego::Params());
  EXPECT_EQ(dim * sizeof(float) + sizeof(float), meta.element_size());
  const size_t rows = BatchDistanceRows(meta);
  EXPECT_EQ(std::max(size_t{1}, kBatchDistanceBytes / (dim * sizeof(float))),
            rows);
  ailego::Params params;
  params.set(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, contiguous);
  auto streamer = IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(meta, params, quantizer));
  ASSERT_EQ(0, streamer->open(storage));
  auto context = streamer->create_context();
  ASSERT_NE(nullptr, context);
  context->set_topk(count);
  constexpr uint64_t first_key = uint64_t{1} << 44;
  std::mt19937 random(20261028);
  std::uniform_real_distribution<float> uniform(-1, 1);
  std::vector<float> raw(dim);
  for (auto &value : raw) value = uniform(random);
  std::string query;
  IndexQueryMeta qmeta;
  ASSERT_EQ(0, quantizer->quantize(raw.data(),
                                   IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                   &query, &qmeta));
  std::vector<std::string> codes(count, std::string(meta.element_size(), '\0'));
  for (size_t i = 0; i < count; ++i) {
    for (auto &value : raw) value = uniform(random);
    quantizer->quantize_data(raw.data(), codes[i].data());
    ASSERT_EQ(0, streamer->add_impl(first_key + i * 17, codes[i].data(), qmeta,
                                    context));
  }
  ASSERT_EQ(0, streamer->flush(0));
  auto reopen = [&](bool writable, size_t memory_limit) {
    context.reset();
    ASSERT_EQ(0, streamer->close());
    ASSERT_EQ(0, storage->close());
    ailego::MemoryLimitPool::get_instance().init(memory_limit);
    ASSERT_EQ(0, storage->open(cleanup.path + "/index", writable));
    ASSERT_EQ(0, streamer->open(storage));
    context = streamer->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(count);
  };
  // Exercise temporary page-backed reads with a small read-only pool.
  reopen(false, 512 * 1024UL);
  ASSERT_NE(nullptr, context);
  auto *flat = dynamic_cast<FlatStreamer<32> *>(streamer.get());
  ASSERT_NE(nullptr, flat);
  EXPECT_EQ(32U, flat->entity().block_vector_count());
  const bool snapshot =
      dynamic_cast<const FlatContiguousStreamerEntity *>(&flat->entity());
  std::vector<uint64_t> excluded;
  auto check_results = [&](int filter_mode) {
    size_t expected_count = 0;
    for (size_t i = 0; i < count; ++i) {
      if (filter_mode == 2 || (filter_mode == 1 && i % 3 == 0) ||
          std::find(excluded.begin(), excluded.end(), first_key + i * 17) !=
              excluded.end())
        continue;
      ++expected_count;
    }
    ASSERT_EQ(expected_count, context->result().size());
    std::vector<bool> seen(count, false);
    for (const auto &hit : context->result()) {
      ASSERT_GE(hit.key(), first_key);
      const size_t id = (hit.key() - first_key) / 17;
      ASSERT_LT(id, count);
      EXPECT_EQ(first_key + id * 17, hit.key());
      EXPECT_EQ(excluded.end(),
                std::find(excluded.begin(), excluded.end(), hit.key()));
      EXPECT_FALSE(seen[id]);
      seen[id] = true;
      if (filter_mode == 1) EXPECT_NE(0U, id % 3);
      const auto *row = reinterpret_cast<const float *>(codes[id].data());
      const auto *q = reinterpret_cast<const float *>(query.data());
      double expected = 1;
      for (size_t d = 0; d < dim; ++d) expected -= double(row[d]) * q[d];
      EXPECT_NEAR(expected, hit.score(), 2e-6);
    }
  };
  for (int filtered : {0, 1, 2, 0}) {
    SCOPED_TRACE(filtered);
    if (filtered == 1) {
      context->set_filter(
          [](uint64_t key) { return ((key - first_key) / 17) % 3 == 0; });
    } else if (filtered == 2) {
      context->set_filter([](uint64_t) { return true; });
    } else {
      context->reset_filter();
    }
    quantizer->batches.clear();
    quantizer->pairwise_calls = 0;
    ASSERT_EQ(0, streamer->search_bf_impl(query.data(), qmeta, 1, context));
    check_results(filtered);
    size_t expected_pairwise =
        snapshot || !filtered ? 0 : context->result().size();
    if (!filtered) {
      std::vector<int> expected;
      if (snapshot) {
        for (size_t begin = 0; begin < count; begin += rows)
          expected.push_back(std::min(rows, count - begin));
      } else {
        std::vector<size_t> physical;
        if (count % 32) physical.push_back(count % 32);
        physical.insert(physical.end(), count / 32, 32);
        const bool mapped_group =
            std::string(storage_name) == "MMapFileStorage" && rows / 32 > 1;
        const size_t group_blocks = mapped_group ? rows / 32 : 1;
        const size_t call_rows =
            mapped_group ? rows : std::min(size_t{32}, rows);
        for (size_t begin = 0; begin < physical.size(); begin += group_blocks) {
          size_t n = 0;
          for (size_t i = begin;
               i < std::min(physical.size(), begin + group_blocks); ++i)
            n += physical[i];
          for (size_t offset = 0; offset < n; offset += call_rows) {
            const size_t batch = std::min(call_rows, n - offset);
            if (batch == 1)
              ++expected_pairwise;
            else
              expected.push_back(batch);
          }
        }
      }
      EXPECT_EQ(expected, quantizer->batches);
    } else if (!snapshot || filtered == 2) {
      EXPECT_TRUE(quantizer->batches.empty());
    }
    EXPECT_EQ(expected_pairwise, quantizer->pairwise_calls);
  }
  // Reuse one query scratch across changing limits, thresholds and prefilled
  // result heaps. The heap must retain its key and inclusive threshold
  // contract.
  if (!snapshot) {
    const auto all = context->result();
    FlatSearchScratch scratch;
    for (size_t limit : {size_t{7}, size_t{129}, size_t{7}}) {
      for (bool bounded : {false, true}) {
        const float threshold = bounded ? all[all.size() / 2].score()
                                        : std::numeric_limits<float>::max();
        IndexDocumentHeap expected(limit, threshold);
        IndexDocumentHeap actual(limit, threshold);
        for (size_t i = 0; i < limit; ++i) {
          const uint64_t key = first_key - 1 - i;
          const float score = i == 0 ? -5.0f : 10.0f + float(i) * 0.01f;
          expected.emplace(key, score);
          actual.emplace(key, score);
        }
        for (const auto &hit : all) expected.emplace(hit.key(), hit.score());
        uint32_t scanned = 0;
        IndexContext::Stats stats;
        ASSERT_EQ(0,
                  flat->entity().search(query.data(), IndexFilter(), &scanned,
                                        &actual, &stats, &scratch));
        EXPECT_EQ(count, scanned);
        expected.sort();
        actual.sort();
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(expected[i].key(), actual[i].key());
          EXPECT_NEAR(expected[i].score(), actual[i].score(), 2e-6);
        }
      }
    }
  }
  if (!snapshot) {
    reopen(true, 64 * 1024 * 1024UL);
    ASSERT_NE(nullptr, context);
    StreamerLinearMeta linear;
    BlockLocation block;
    ASSERT_TRUE(ReadBudgetSegment(storage->get(FLAT_LINEAR_META_SEG_ID), 0,
                                  &linear, sizeof(linear)));
    ASSERT_TRUE(ReadBudgetSegment(storage->get(FLAT_LINEAR_LIST_HEAD_SEG_ID), 0,
                                  &block, sizeof(block)));
    size_t block_number = 0;
    while (block.segment_id != 0) {
      auto segment = storage->get(FLAT_SEGMENT_FEATURES_SEG_ID +
                                  std::to_string(block.segment_id));
      const size_t start = block.block_index * linear.header.block_size;
      const size_t header_offset =
          start + linear.header.block_size - sizeof(BlockHeader);
      BlockHeader header;
      ASSERT_TRUE(
          ReadBudgetSegment(segment, header_offset, &header, sizeof(header)));
      ++block_number;
      if (block_number == 2 || block_number == 3) {
        uint64_t key;
        const size_t key_offset = start + 32 * meta.element_size();
        ASSERT_TRUE(ReadBudgetSegment(segment, key_offset, &key, sizeof(key)));
        excluded.push_back(key);
        if (block_number == 2) {
          // Invalid keys in clean blocks are still calculated, then skipped.
          const uint64_t invalid = kInvalidKey;
          ASSERT_EQ(sizeof(invalid),
                    segment->write(key_offset, &invalid, sizeof(invalid)));
        } else {
          // Flush the preceding clean blocks before the original dirty path.
          DeletionMap deleted;
          deleted.set(0);
          ASSERT_EQ(sizeof(deleted),
                    segment->write(header_offset - sizeof(deleted),
                                   deleted.bitset.data(), sizeof(deleted)));
          break;
        }
      }
      block = header.next;
    }
    ASSERT_EQ(2u, excluded.size());
    ASSERT_EQ(0, storage->flush());
    reopen(false, 512 * 1024UL);
    ASSERT_NE(nullptr, context);
    quantizer->batches.clear();
    quantizer->pairwise_calls = 0;
    ASSERT_EQ(0, streamer->search_bf_impl(query.data(), qmeta, 1, context));
    check_results(0);
    // The dirty block keeps its original 31 pairwise calls; clean blocks
    // may also use pairwise calls when the byte budget admits only one row.
    EXPECT_GE(quantizer->pairwise_calls, 31u);
  }
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
}
}  // namespace

TEST(FlatQuantizerDistance,
     ByteBudgetGroupsStableBlocksAndPreservesFilteredPaths) {
  for (const char *storage : {"MMapFileStorage", "BufferStorage"}) {
    for (bool contiguous : {false, true}) {
      CheckBudgetStreamer(256, 1073, storage, contiguous);
      // Sub-block budgets split within the original read, without retaining
      // temporary addresses or changing one-row calls to batch calls.
      CheckBudgetStreamer(8192, 65, storage, contiguous);
    }
  }
}

namespace {
void CheckBudgetSearcher(size_t dim, size_t count) {
  SCOPED_TRACE(dim);
  BudgetTestCleanup cleanup("flat_static_byte_budget_test");
  const std::string path = cleanup.path + "/index";
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  constexpr uint64_t first_key = uint64_t{1} << 45;
  for (size_t i = 0; i < count; ++i) {
    ailego::NumericalVector<float> row(dim);
    std::fill(row.data(), row.data() + dim, 0.0f);
    row[0] = static_cast<float>(i) * 0.25f;
    ASSERT_TRUE(holder->emplace(first_key + i * 19, row));
  }
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, builder);
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
  const size_t rows = BatchDistanceRows(meta);
  // Add one physical block after a full budget, or use one physical block
  // when high-dimensional rows require smaller batches within that read.
  const size_t read_rows = (rows / 32 + 1) * 32;
  std::vector<int> expected_batches;
  for (size_t begin = 0; begin < count; begin += read_rows) {
    const size_t available = std::min(read_rows, count - begin);
    for (size_t offset = 0; offset < available; offset += rows)
      expected_batches.push_back(std::min(rows, available - offset));
  }
  for (const char *storage_name :
       {"MMapFileReadStorage", "FileReadStorage", "BufferReadStorage"}) {
    SCOPED_TRACE(storage_name);
    ailego::MemoryLimitPool::get_instance().init(4 * 1024 * 1024UL);
    auto storage = IndexFactory::CreateStorage(storage_name);
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    auto quantizer = std::make_shared<BudgetFp32Quantizer>();
    ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));
    auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
    ASSERT_NE(nullptr, searcher);
    ailego::Params params;
    params.set(PARAM_FLAT_READ_BLOCK_SIZE,
               static_cast<uint32_t>(read_rows * meta.element_size()));
    ASSERT_EQ(0, searcher->init(params, quantizer));
    ASSERT_EQ(0, searcher->load(storage, nullptr));
    auto context = searcher->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(count);
    std::vector<float> query(dim, 0.0f);
    for (bool filtered : {false, true, false}) {
      if (filtered) {
        context->set_filter(
            [](uint64_t key) { return ((key - first_key) / 19) % 2 == 0; });
      } else {
        context->reset_filter();
      }
      quantizer->batches.clear();
      ASSERT_EQ(0, searcher->search_impl(
                       query.data(), IndexQueryMeta(IndexMeta::DT_FP32, dim),
                       context));
      EXPECT_EQ(filtered ? std::vector<int>{} : expected_batches,
                quantizer->batches);
      ASSERT_EQ(filtered ? count / 2 : count, context->result().size());
      for (size_t rank = 0; rank < context->result().size(); ++rank) {
        const size_t id = filtered ? rank * 2 + 1 : rank;
        const auto &hit = context->result()[rank];
        EXPECT_EQ(first_key + id * 19, hit.key());
        EXPECT_EQ(id, hit.index());
        EXPECT_FLOAT_EQ(static_cast<float>(id * id) * 0.0625f, hit.score());
      }
    }
    // Exercise selection and query-state reuse separately from the all-row
    // checks above, including the small input and changing-k fallback.
    for (size_t limit : {size_t{1}, size_t{7}, size_t{129}, size_t{1}}) {
      context->set_topk(limit);
      for (bool bounded : {false, true}) {
        if (bounded)
          context->set_threshold(0.5625f);
        else
          context->reset_threshold();
        quantizer->batches.clear();
        ASSERT_EQ(0, searcher->search_impl(
                         query.data(), IndexQueryMeta(IndexMeta::DT_FP32, dim),
                         context));
        EXPECT_EQ(expected_batches, quantizer->batches);
        const size_t eligible = bounded ? std::min(count, size_t{4}) : count;
        ASSERT_EQ(std::min(limit, eligible), context->result().size());
        for (size_t rank = 0; rank < context->result().size(); ++rank) {
          const auto &hit = context->result()[rank];
          EXPECT_EQ(first_key + rank * 19, hit.key());
          EXPECT_EQ(rank, hit.index());
          EXPECT_FLOAT_EQ(static_cast<float>(rank * rank) * 0.0625f,
                          hit.score());
        }
      }
    }
    context.reset();
    ASSERT_EQ(0, searcher->cleanup());
    ASSERT_EQ(0, storage->close());
  }
}
}  // namespace

TEST(FlatQuantizerDistance, ByteBudgetSearcherKeepsReadLocalTails) {
  CheckBudgetSearcher(256, 3);
  CheckBudgetSearcher(256, 1073);
  CheckBudgetSearcher(256, 65);
  CheckBudgetSearcher(256, 17);
  CheckBudgetSearcher(8193, 65);
}
