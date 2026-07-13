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

#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_params.h"
#include "tests/test_util.h"
#include "zvec/db/index_params.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec;

namespace {

constexpr uint32_t kGbDimension = 4;
constexpr uint32_t kGbNumDocs = 12;
constexpr uint32_t kGbNumGroups = 3;
constexpr uint32_t kGbGroupTopk = 2;
constexpr uint32_t kGbSearchTopk = 100;
constexpr uint32_t kGbSparseCount = 5;

struct GroupByCase {
  std::string name;
  IndexParams::Ptr index_params;
  QueryParams::Ptr query_params;  // core-layer query params; nullptr for sparse
  bool is_sparse = false;
  uint32_t dimension = kGbDimension;
  bool optional = false;  // skip when plugin unavailable (e.g. DiskAnn)
  bool with_bf_pks = false;
  bool fetch_vector = false;
};

std::unique_ptr<vector_column_params::GroupByParams> MakeGroupByParams(
    uint32_t group_count = kGbNumGroups) {
  return std::make_unique<vector_column_params::GroupByParams>(
      kGbGroupTopk, group_count, [](uint64_t key) -> std::string {
        return std::to_string(key % kGbNumGroups);
      });
}

std::unique_ptr<vector_column_params::GroupByParams> MakeSegmentGroupByParams(
    uint32_t group_topk, uint32_t group_count) {
  return std::make_unique<vector_column_params::GroupByParams>(
      group_topk, group_count,
      [](uint64_t key) -> std::string { return key < 2 ? "low" : "high"; });
}

std::vector<uint64_t> AllPks() {
  std::vector<uint64_t> pks(kGbNumDocs);
  std::iota(pks.begin(), pks.end(), 0ull);
  return pks;
}

class GroupByIndexerTest : public ::testing::Test {
 protected:
  void RunOk(const GroupByCase &tc) {
    Run(tc, /*expect_error=*/false);
  }
  void RunRejected(const GroupByCase &tc) {
    Run(tc, /*expect_error=*/true);
  }

 private:
  struct QueryHolder {
    std::vector<float> dense;
    std::vector<uint32_t> sparse_indices;
    std::vector<float> sparse_values;
    vector_column_params::VectorData data;
  };

  void Run(const GroupByCase &tc, bool expect_error) {
    const std::string path = "test_groupby_" + tc.name + ".index";
    zvec::test_util::RemoveTestFiles(path);

    auto indexer = OpenIndexer(tc, path);
    if (indexer == nullptr) {
      zvec::test_util::RemoveTestFiles(path);
      return;  // optional plugin unavailable
    }

    InsertDocs(indexer, tc);

    QueryHolder holder = MakeQuery(tc);
    vector_column_params::QueryParams qp = MakeQueryParams(tc);

    auto results = indexer->Search(holder.data, qp);

    if (expect_error) {
      ASSERT_FALSE(results.has_value())
          << "group_by should be rejected for " << tc.name;
    } else {
      ASSERT_TRUE(results.has_value()) << tc.name;
      AssertGroupedResult(results.value().get(), tc);
    }

    indexer->Close();
    zvec::test_util::RemoveTestFiles(path);
  }

  static FieldSchema MakeSchema(const GroupByCase &tc) {
    if (tc.is_sparse) {
      return FieldSchema("test", DataType::SPARSE_VECTOR_FP32, false,
                         tc.index_params);
    }
    return FieldSchema("test", DataType::VECTOR_FP32, tc.dimension, false,
                       tc.index_params);
  }

  static VectorColumnIndexer::Ptr OpenIndexer(const GroupByCase &tc,
                                              const std::string &path) {
    auto indexer = std::make_shared<VectorColumnIndexer>(path, MakeSchema(tc));
    if (!indexer->Open(vector_column_params::ReadOptions{true, true}).ok()) {
      return nullptr;
    }
    return indexer;
  }

  static void InsertDocs(const VectorColumnIndexer::Ptr &indexer,
                         const GroupByCase &tc) {
    for (uint32_t i = 0; i < kGbNumDocs; ++i) {
      if (tc.is_sparse) {
        std::vector<uint32_t> indices(kGbSparseCount);
        std::vector<float> values(kGbSparseCount);
        for (uint32_t j = 0; j < kGbSparseCount; ++j) {
          indices[j] = i * kGbSparseCount + j;
          values[j] = static_cast<float>(i + 1);
        }
        vector_column_params::SparseVector sv{kGbSparseCount, indices.data(),
                                              values.data()};
        ASSERT_TRUE(
            indexer->Insert(vector_column_params::VectorData{sv}, i).ok());
      } else {
        std::vector<float> vec(tc.dimension, static_cast<float>(i));
        vector_column_params::DenseVector dv{vec.data()};
        ASSERT_TRUE(
            indexer->Insert(vector_column_params::VectorData{dv}, i).ok());
      }
    }
  }

  static QueryHolder MakeQuery(const GroupByCase &tc) {
    QueryHolder h;
    if (tc.is_sparse) {
      h.sparse_indices.resize(kGbSparseCount);
      h.sparse_values.assign(kGbSparseCount, 1.0f);
      std::iota(h.sparse_indices.begin(), h.sparse_indices.end(), 0u);
      h.data =
          vector_column_params::VectorData{vector_column_params::SparseVector{
              kGbSparseCount, h.sparse_indices.data(), h.sparse_values.data()}};
    } else {
      h.dense.assign(tc.dimension, 1.0f);
      h.data = vector_column_params::VectorData{
          vector_column_params::DenseVector{h.dense.data()}};
    }
    return h;
  }

  static vector_column_params::QueryParams MakeQueryParams(
      const GroupByCase &tc) {
    vector_column_params::QueryParams qp;
    qp.topk = kGbSearchTopk;
    qp.filter = nullptr;
    qp.fetch_vector = tc.fetch_vector;
    qp.query_params = tc.query_params;
    if (tc.with_bf_pks) {
      qp.bf_pks = {AllPks()};
    }
    qp.group_by = MakeGroupByParams();
    return qp;
  }

  static void AssertGroupedResult(IndexResults *results,
                                  const GroupByCase &tc) {
    auto *group_results = dynamic_cast<GroupVectorIndexResults *>(results);
    ASSERT_TRUE(group_results)
        << "Expected GroupVectorIndexResults for " << tc.name;
    ASSERT_EQ(kGbNumGroups, group_results->groups().size()) << tc.name;

    std::set<std::string> group_ids;
    for (const auto &group : group_results->groups()) {
      group_ids.insert(group.group_id());
      ASSERT_LE(group.docs().size(), kGbGroupTopk) << tc.name;
      ASSERT_GE(group.docs().size(), 1u) << tc.name;

      const uint32_t expected_mod = std::stoul(group.group_id());
      for (const auto &doc : group.docs()) {
        ASSERT_EQ(expected_mod, doc.key() % kGbNumGroups)
            << tc.name << " doc " << doc.key();
      }
      for (size_t j = 1; j < group.docs().size(); ++j) {
        ASSERT_GE(group.docs()[j - 1].score(), group.docs()[j].score())
            << tc.name << " group " << group.group_id();
      }
    }
    for (uint32_t g = 0; g < kGbNumGroups; ++g) {
      ASSERT_TRUE(group_ids.count(std::to_string(g)) > 0)
          << tc.name << " missing group " << g;
    }

    auto iter = group_results->create_iterator();
    size_t total = 0;
    while (iter->valid()) {
      if (tc.fetch_vector && !tc.is_sparse) {
        const auto vector_data = iter->vector();
        const auto &dense_vector =
            std::get<vector_column_params::DenseVector>(vector_data.vector);
        const float *vector =
            reinterpret_cast<const float *>(dense_vector.data);
        const float expected = static_cast<float>(iter->doc_id());
        for (uint32_t i = 0; i < tc.dimension; ++i) {
          ASSERT_FLOAT_EQ(expected, vector[i])
              << tc.name << " doc " << iter->doc_id() << " i " << i;
        }
      }
      total++;
      iter->next();
    }
    ASSERT_EQ(group_results->count(), total) << tc.name;
  }
};

}  // namespace

TEST_F(GroupByIndexerTest, Dense) {
  auto hnsw_linear_qp = std::make_shared<HnswQueryParams>(300);
  hnsw_linear_qp->set_is_linear(true);

  std::vector<GroupByCase> cases{
      {"dense_flat_graph", std::make_shared<FlatIndexParams>(MetricType::IP),
       std::make_shared<QueryParams>(IndexType::FLAT)},
      {"dense_hnsw_graph",
       std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       std::make_shared<HnswQueryParams>(300)},
      {"dense_hnsw_linear",
       std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       hnsw_linear_qp},
      {"dense_hnsw_bf_pks",
       std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       std::make_shared<HnswQueryParams>(300),
       /*is_sparse=*/false, /*dimension=*/kGbDimension,
       /*optional=*/false, /*with_bf_pks=*/true},
      {"dense_hnsw_fetch_vector",
       std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       std::make_shared<HnswQueryParams>(300),
       /*is_sparse=*/false, /*dimension=*/kGbDimension,
       /*optional=*/false, /*with_bf_pks=*/false, /*fetch_vector=*/true},
      {"dense_hnsw_fp16_fetch_vector",
       std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100,
                                         QuantizeType::FP16),
       std::make_shared<HnswQueryParams>(300),
       /*is_sparse=*/false, /*dimension=*/kGbDimension,
       /*optional=*/false, /*with_bf_pks=*/false, /*fetch_vector=*/true},
  };

  for (const auto &tc : cases) {
    RunOk(tc);
  }
}

TEST_F(GroupByIndexerTest, CombinedSortsGroupsBeforeTruncating) {
  // Build two blocks where the best group is in the later block; group_count
  // must be applied after cross-block group sorting, not merge order.
  const std::string block0_path = "test_groupby_combined_block0.index";
  const std::string block1_path = "test_groupby_combined_block1.index";
  zvec::test_util::RemoveTestFiles(block0_path);
  zvec::test_util::RemoveTestFiles(block1_path);

  auto index_params = std::make_shared<FlatIndexParams>(MetricType::IP);
  FieldSchema schema("test", DataType::VECTOR_FP32, kGbDimension, false,
                     index_params);

  auto block0 = std::make_shared<VectorColumnIndexer>(block0_path, schema);
  auto block1 = std::make_shared<VectorColumnIndexer>(block1_path, schema);
  ASSERT_TRUE(block0->Open(vector_column_params::ReadOptions{true, true}).ok());
  ASSERT_TRUE(block1->Open(vector_column_params::ReadOptions{true, true}).ok());

  auto insert_dense = [](const VectorColumnIndexer::Ptr &indexer,
                         uint32_t doc_id, float value) {
    std::vector<float> vec(kGbDimension, value);
    vector_column_params::DenseVector dense{vec.data()};
    ASSERT_TRUE(
        indexer->Insert(vector_column_params::VectorData{dense}, doc_id).ok());
  };
  insert_dense(block0, 0, 0.0f);
  insert_dense(block0, 1, 1.0f);
  insert_dense(block1, 0, 10.0f);
  insert_dense(block1, 1, 11.0f);

  std::vector<BlockMeta> blocks{
      BlockMeta(0, BlockType::VECTOR_INDEX, 0, 1, 2, {"test"}),
      BlockMeta(1, BlockType::VECTOR_INDEX, 2, 3, 2, {"test"}),
  };
  SegmentMeta segment_meta;
  CombinedVectorColumnIndexer combined({block0, block1}, {}, schema,
                                       segment_meta, blocks, MetricType::IP);

  std::vector<float> query(kGbDimension, 1.0f);
  vector_column_params::DenseVector dense_query{query.data()};
  vector_column_params::QueryParams query_params;
  query_params.topk = kGbSearchTopk;
  query_params.query_params = std::make_shared<QueryParams>(IndexType::FLAT);
  query_params.group_by = MakeSegmentGroupByParams(/*group_topk=*/1,
                                                   /*group_count=*/1);

  auto results = combined.Search(vector_column_params::VectorData{dense_query},
                                 query_params);
  ASSERT_TRUE(results.has_value());
  auto *group_results =
      dynamic_cast<GroupVectorIndexResults *>(results.value().get());
  ASSERT_TRUE(group_results);
  ASSERT_EQ(1u, group_results->groups().size());
  ASSERT_EQ("high", group_results->groups()[0].group_id());
  ASSERT_EQ(1u, group_results->groups()[0].docs().size());
  ASSERT_EQ(3u, group_results->groups()[0].docs()[0].key());
  ASSERT_FLOAT_EQ(44.0f, group_results->groups()[0].docs()[0].score());

  ASSERT_TRUE(block0->Close().ok());
  ASSERT_TRUE(block1->Close().ok());
  zvec::test_util::RemoveTestFiles(block0_path);
  zvec::test_util::RemoveTestFiles(block1_path);
}

TEST_F(GroupByIndexerTest, Sparse) {
  std::vector<GroupByCase> cases{
      {"sparse_flat_graph", std::make_shared<FlatIndexParams>(MetricType::IP),
       /*query_params=*/nullptr,
       /*is_sparse=*/true},
      {"sparse_hnsw_graph",
       std::make_shared<HnswIndexParams>(MetricType::IP, 10, 100),
       /*query_params=*/nullptr,
       /*is_sparse=*/true},
  };

  for (const auto &tc : cases) {
    RunOk(tc);
  }
}

TEST_F(GroupByIndexerTest, UnsupportedIndexTypes) {
  std::vector<GroupByCase> cases{
      {"unsupported_ivf", std::make_shared<IVFIndexParams>(MetricType::IP, 4),
       std::make_shared<IVFQueryParams>(4)},
      {"unsupported_diskann",
       std::make_shared<DiskAnnIndexParams>(MetricType::IP),
       std::make_shared<DiskAnnQueryParams>(),
       /*is_sparse=*/false, /*dimension=*/kGbDimension,
       /*optional=*/true},
  };

  for (const auto &tc : cases) {
    RunRejected(tc);
  }
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
